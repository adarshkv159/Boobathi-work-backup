import sys
import os
import time
import numpy as np
from collections import namedtuple

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QGroupBox, QRadioButton, QTextEdit, QCheckBox,
                             QDoubleSpinBox, QSlider)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

# Import cv2 AFTER PySide6
import cv2

# CRITICAL FIX: Remove cv2's Qt plugin paths after import
try:
    from cv2.version import ci_build, headless
    ci_and_not_headless = ci_build and not headless
except:
    ci_and_not_headless = False

if sys.platform.startswith("linux") and ci_and_not_headless:
    if "QT_QPA_PLATFORM_PLUGIN_PATH" in os.environ:
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH")
    if "QT_QPA_FONTDIR" in os.environ:
        os.environ.pop("QT_QPA_FONTDIR")

if "QT_PLUGIN_PATH" in os.environ:
    cv2_plugin_path = os.path.join(os.path.dirname(cv2.__file__), 'qt', 'plugins')
    if cv2_plugin_path in os.environ.get("QT_PLUGIN_PATH", ""):
        os.environ.pop("QT_PLUGIN_PATH")

import tflite_runtime.interpreter as tflite

# COCO Skeleton Definition
COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6)
]

FACE_KEYPOINTS = {0, 1, 2, 3, 4}

# Simple keypoint structure
Keypoint = namedtuple('Keypoint', ['x', 'y', 'score'])
Person = namedtuple('Person', ['points'])


class PoseEstimationThread(QThread):
    """Thread for pose estimation"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    stats_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.detection_model_path = "ssd_mobilenet_v1_quant.tflite"
        self.pose_model_path = "pose_resnet_50_256x192_int8.tflite"
        self.video_source = "0"
        self.delegate_path = "/usr/lib/libvx_delegate.so"
        self.use_npu = False
        self.detection_threshold = 0.5
        self.pose_threshold = 0.4
        self.show_bbox = True
        self.show_keypoints = True
        self.show_skeleton = True
        self.hide_face = True
        
        self.detection_interpreter = None
        self.pose_interpreter = None
        self.cap = None
        
        # Input/output normalization parameters
        self.pose_input_scale = None
        self.pose_input_zero_point = None
        self.pose_output_scale = None
        self.pose_output_zero_point = None
        
    def set_detection_model(self, path):
        self.detection_model_path = path
    
    def set_pose_model(self, path):
        self.pose_model_path = path
    
    def set_video_source(self, source):
        self.video_source = source
    
    def set_delegate(self, use_npu, delegate_path=""):
        self.use_npu = use_npu
        if delegate_path:
            self.delegate_path = delegate_path
    
    def set_detection_threshold(self, threshold):
        self.detection_threshold = threshold
    
    def set_pose_threshold(self, threshold):
        self.pose_threshold = threshold
    
    def set_display_options(self, bbox, keypoints, skeleton, hide_face):
        self.show_bbox = bbox
        self.show_keypoints = keypoints
        self.show_skeleton = skeleton
        self.hide_face = hide_face
    
    def keep_aspect(self, top_left, bottom_right, img):
        """Maintain aspect ratio for cropping"""
        x1, y1 = top_left
        x2, y2 = bottom_right
        h, w = img.shape[:2]
        
        crop_w = x2 - x1
        crop_h = y2 - y1
        
        # Target aspect ratio (256:192)
        target_ratio = 256 / 192
        current_ratio = crop_w / crop_h if crop_h > 0 else 1
        
        if current_ratio > target_ratio:
            # Too wide, adjust height
            new_h = int(crop_w / target_ratio)
            diff = new_h - crop_h
            y1 = max(0, y1 - diff // 2)
            y2 = min(h, y2 + diff // 2)
        else:
            # Too tall, adjust width
            new_w = int(crop_h * target_ratio)
            diff = new_w - crop_w
            x1 = max(0, x1 - diff // 2)
            x2 = min(w, x2 + diff // 2)
        
        return x1, y1, x2, y2
    
    def compute_pose(self, crop_img, offset_x, offset_y, scale_x, scale_y):
        """Run pose estimation on cropped person image"""
        # Resize to model input size (192x256)
        input_img = cv2.resize(crop_img, (192, 256))
        
        # Check if quantized model
        if self.pose_input_details[0]['dtype'] == np.int8:
            # Quantize input: convert to float32, normalize, then quantize
            input_img_float = input_img.astype(np.float32)
            # Normalize to [0, 1]
            input_img_float = input_img_float / 255.0
            # Quantize using scale and zero_point
            input_data = input_img_float / self.pose_input_scale + self.pose_input_zero_point
            input_data = np.clip(input_data, -128, 127).astype(np.int8)
        else:
            # Float32 model
            input_data = input_img.astype(np.float32) / 255.0
        
        input_data = np.expand_dims(input_data, axis=0)
        
        # Run inference
        self.pose_interpreter.set_tensor(self.pose_input_details[0]['index'], input_data)
        self.pose_interpreter.invoke()
        
        # Get output (heatmaps)
        output = self.pose_interpreter.get_tensor(self.pose_output_details[0]['index'])[0]
        
        # Dequantize output if needed
        if self.pose_output_details[0]['dtype'] == np.int8:
            output = (output.astype(np.float32) - self.pose_output_zero_point) * self.pose_output_scale
        
        # Extract keypoints from heatmaps
        keypoints = []
        num_keypoints = output.shape[2]
        
        for i in range(num_keypoints):
            heatmap = output[:, :, i]
            max_val = np.max(heatmap)
            
            if max_val > 0:
                y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                # Normalize to 0-1 range and apply offset/scale
                kp_x = (x / heatmap.shape[1]) * scale_x + offset_x
                kp_y = (y / heatmap.shape[0]) * scale_y + offset_y
                
                # Normalize confidence score
                if self.pose_output_details[0]['dtype'] == np.int8:
                    confidence = max_val
                else:
                    confidence = max_val
                
                keypoints.append(Keypoint(kp_x, kp_y, confidence))
            else:
                keypoints.append(Keypoint(0, 0, 0))
        
        return Person(keypoints)
    
    def draw_keypoints_and_skeleton(self, img, person):
        """Draw keypoints and skeleton on image"""
        points = []
        
        # Draw keypoints
        if self.show_keypoints:
            for i, point in enumerate(person.points):
                if point.score > self.pose_threshold:
                    x = int(img.shape[1] * point.x)
                    y = int(img.shape[0] * point.y)
                    points.append((x, y))
                    
                    # Skip face keypoints if enabled
                    if self.hide_face and i in FACE_KEYPOINTS:
                        points[-1] = None
                        continue
                    
                    cv2.circle(img, (x, y), 6, (0, 255, 0), -1)
                else:
                    points.append(None)
        else:
            for i, point in enumerate(person.points):
                if point.score > self.pose_threshold:
                    x = int(img.shape[1] * point.x)
                    y = int(img.shape[0] * point.y)
                    points.append((x, y))
                else:
                    points.append(None)
        
        # Draw skeleton
        if self.show_skeleton:
            for pair in COCO_SKELETON:
                kp1, kp2 = pair
                
                # Skip face connections if enabled
                if self.hide_face and (kp1 in FACE_KEYPOINTS or kp2 in FACE_KEYPOINTS):
                    continue
                
                if kp1 < len(points) and kp2 < len(points):
                    if points[kp1] is not None and points[kp2] is not None:
                        cv2.line(img, points[kp1], points[kp2], (0, 0, 255), 2)
        
        return len([p for p in points if p is not None])
    
    def initialize_models(self):
        """Initialize detection and pose estimation models"""
        try:
            # Initialize detection model
            self.log_signal.emit("\n=== Loading Detection Model ===")
            
            if self.use_npu and os.path.exists(self.delegate_path):
                try:
                    self.detection_interpreter = tflite.Interpreter(
                        model_path=self.detection_model_path,
                        experimental_delegates=[tflite.load_delegate(self.delegate_path)]
                    )
                    self.log_signal.emit("✓ Detection model loaded on NPU")
                except Exception as e:
                    self.log_signal.emit(f"NPU loading failed: {e}")
                    self.detection_interpreter = tflite.Interpreter(model_path=self.detection_model_path)
                    self.log_signal.emit("✓ Detection model loaded on CPU (fallback)")
            else:
                self.detection_interpreter = tflite.Interpreter(model_path=self.detection_model_path)
                self.log_signal.emit("✓ Detection model loaded on CPU")
            
            self.detection_interpreter.allocate_tensors()
            self.detect_input_details = self.detection_interpreter.get_input_details()
            self.detect_output_details = self.detection_interpreter.get_output_details()
            
            # Initialize pose model
            self.log_signal.emit("\n=== Loading Pose Estimation Model ===")
            
            if self.use_npu and os.path.exists(self.delegate_path):
                try:
                    self.pose_interpreter = tflite.Interpreter(
                        model_path=self.pose_model_path,
                        experimental_delegates=[tflite.load_delegate(self.delegate_path)]
                    )
                    self.log_signal.emit("✓ Pose model loaded on NPU")
                except Exception as e:
                    self.log_signal.emit(f"NPU loading failed: {e}")
                    self.pose_interpreter = tflite.Interpreter(model_path=self.pose_model_path)
                    self.log_signal.emit("✓ Pose model loaded on CPU (fallback)")
            else:
                self.pose_interpreter = tflite.Interpreter(model_path=self.pose_model_path)
                self.log_signal.emit("✓ Pose model loaded on CPU")
            
            self.pose_interpreter.allocate_tensors()
            self.pose_input_details = self.pose_interpreter.get_input_details()
            self.pose_output_details = self.pose_interpreter.get_output_details()
            
            # Get quantization parameters
            self.pose_input_scale = self.pose_input_details[0]['quantization'][0]
            self.pose_input_zero_point = self.pose_input_details[0]['quantization'][1]
            self.pose_output_scale = self.pose_output_details[0]['quantization'][0]
            self.pose_output_zero_point = self.pose_output_details[0]['quantization'][1]
            
            self.log_signal.emit(f"\nDetection input shape: {self.detect_input_details[0]['shape']}")
            self.log_signal.emit(f"Pose input shape: {self.pose_input_details[0]['shape']}")
            self.log_signal.emit(f"Pose input dtype: {self.pose_input_details[0]['dtype']}")
            self.log_signal.emit(f"Pose input scale: {self.pose_input_scale}")
            self.log_signal.emit(f"Pose input zero_point: {self.pose_input_zero_point}")
            
            return True
            
        except Exception as e:
            error_msg = f"Error initializing models: {e}"
            self.log_signal.emit(error_msg)
            return False
    
    def run(self):
        """Main video processing loop"""
        if not self.initialize_models():
            return
        
        # Initialize video capture
        if self.video_source.isdigit():
            self.cap = cv2.VideoCapture(int(self.video_source))
        else:
            self.cap = cv2.VideoCapture(self.video_source)
        
        if not self.cap.isOpened():
            error_msg = f"Error: Could not open video source {self.video_source}"
            self.log_signal.emit(error_msg)
            return
        
        self.log_signal.emit(f"\n✓ Video source initialized: {self.video_source}")
        self.log_signal.emit("Starting pose estimation...\n")
        
        # Get detection model dimensions
        detect_height = self.detect_input_details[0]['shape'][1]
        detect_width = self.detect_input_details[0]['shape'][2]
        
        # Performance tracking
        total_frames = 0
        total_time = 0
        self.running = True
        first_inference = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            
            total_frames += 1
            loop_start = time.time()
            
            frame_height, frame_width = frame.shape[:2]
            
            # Run person detection
            resized = cv2.resize(frame, (detect_width, detect_height))
            input_data = np.expand_dims(resized, axis=0).astype(np.uint8)
            
            self.detection_interpreter.set_tensor(self.detect_input_details[0]['index'], input_data)
            
            detect_start = time.time()
            self.detection_interpreter.invoke()
            detect_end = time.time()
            
            if first_inference:
                warmup_time = int((detect_end - detect_start) * 1000)
                self.log_signal.emit(f"First inference time: {warmup_time}ms (includes warmup)")
                first_inference = False
            
            # Get detection results
            boxes = self.detection_interpreter.get_tensor(self.detect_output_details[0]['index'])[0]
            labels = self.detection_interpreter.get_tensor(self.detect_output_details[1]['index'])[0]
            scores = self.detection_interpreter.get_tensor(self.detect_output_details[2]['index'])[0]
            num_detections = int(self.detection_interpreter.get_tensor(self.detect_output_details[3]['index'])[0])
            
            # Process person detections
            person_boxes = []
            for i in range(num_detections):
                if scores[i] > self.detection_threshold and int(labels[i]) == 0:  # Person class
                    ymin, xmin, ymax, xmax = boxes[i]
                    x0 = int(xmin * frame_width)
                    y0 = int(ymin * frame_height)
                    x1 = int(xmax * frame_width)
                    y1 = int(ymax * frame_height)
                    person_boxes.append([x0, y0, x1, y1])
            
            # Run pose estimation for each person
            total_keypoints = 0
            pose_start = time.time()
            
            for box in person_boxes:
                x1, y1, x2, y2 = self.keep_aspect((box[0], box[1]), (box[2], box[3]), frame)
                
                # Crop person region
                crop_img = frame[y1:y2, x1:x2]
                
                if crop_img.size == 0:
                    continue
                
                # Calculate offsets and scales
                offset_x = x1 / frame_width
                offset_y = y1 / frame_height
                scale_x = (x2 - x1) / frame_width
                scale_y = (y2 - y1) / frame_height
                
                # Run pose estimation
                person = self.compute_pose(crop_img, offset_x, offset_y, scale_x, scale_y)
                
                # Draw bounding box
                if self.show_bbox:
                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
                
                # Draw keypoints and skeleton
                kp_count = self.draw_keypoints_and_skeleton(frame, person)
                total_keypoints += kp_count
            
            pose_end = time.time()
            
            # Calculate performance metrics
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_frames / total_time) if total_time > 0 else 0
            detect_time = (detect_end - detect_start) * 1000
            pose_time = (pose_end - pose_start) * 1000
            
            # Status message
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {fps} | Detect: {detect_time:.1f}ms | Pose: {pose_time:.1f}ms | Mode: {mode}"
            
            cv2.putText(frame, msg, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Stats message
            stats = f"Persons: {len(person_boxes)}\n"
            stats += f"Total Keypoints: {total_keypoints}\n"
            stats += f"Avg Keypoints/Person: {total_keypoints/len(person_boxes):.1f}" if person_boxes else "N/A"
            
            # Convert to QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Emit signals
            self.change_pixmap_signal.emit(qt_image)
            self.fps_signal.emit(msg)
            self.stats_signal.emit(stats)
        
        # Cleanup
        if self.cap:
            self.cap.release()
        self.log_signal.emit("\n✓ Video processing stopped")
    
    def stop(self):
        """Stop the pose estimation thread"""
        self.running = False
        self.wait()


class PoseEstimationGUI(QMainWindow):
    """Main GUI window for pose estimation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pose Estimation - ResNet50 (i.MX8M Plus)")
        self.setGeometry(100, 100, 1500, 900)
        
        # Initialize thread
        self.pose_thread = PoseEstimationThread()
        self.pose_thread.change_pixmap_signal.connect(self.update_image)
        self.pose_thread.fps_signal.connect(self.update_fps)
        self.pose_thread.log_signal.connect(self.update_log)
        self.pose_thread.stats_signal.connect(self.update_stats)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top section
        top_layout = QHBoxLayout()
        
        control_panel = self.create_control_panel()
        top_layout.addWidget(control_panel, 1)
        
        video_panel = self.create_video_panel()
        top_layout.addWidget(video_panel, 2)
        
        stats_panel = self.create_stats_panel()
        top_layout.addWidget(stats_panel, 1)
        
        main_layout.addLayout(top_layout, 3)
        
        # Bottom section
        log_panel = self.create_log_panel()
        main_layout.addWidget(log_panel, 1)
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        det_label = QLabel("Detection Model:")
        self.detection_model_input = QLineEdit("ssd_mobilenet_v1_quant.tflite")
        model_layout.addWidget(det_label)
        model_layout.addWidget(self.detection_model_input)
        
        pose_label = QLabel("Pose Model:")
        self.pose_model_input = QLineEdit("pose_resnet_50_256x192_int8.tflite")
        model_layout.addWidget(pose_label)
        model_layout.addWidget(self.pose_model_input)
        
        video_label = QLabel("Video Source:")
        self.video_input = QLineEdit("0")
        model_layout.addWidget(video_label)
        model_layout.addWidget(self.video_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Detection parameters
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()
        
        det_thresh_label = QLabel("Detection Threshold: 0.50")
        self.det_thresh_slider = QSlider(Qt.Horizontal)
        self.det_thresh_slider.setRange(10, 95)
        self.det_thresh_slider.setValue(50)
        self.det_thresh_slider.valueChanged.connect(
            lambda v: det_thresh_label.setText(f"Detection Threshold: {v/100:.2f}")
        )
        param_layout.addWidget(det_thresh_label)
        param_layout.addWidget(self.det_thresh_slider)
        
        pose_thresh_label = QLabel("Pose Threshold: 0.40")
        self.pose_thresh_slider = QSlider(Qt.Horizontal)
        self.pose_thresh_slider.setRange(10, 95)
        self.pose_thresh_slider.setValue(40)
        self.pose_thresh_slider.valueChanged.connect(
            lambda v: pose_thresh_label.setText(f"Pose Threshold: {v/100:.2f}")
        )
        param_layout.addWidget(pose_thresh_label)
        param_layout.addWidget(self.pose_thresh_slider)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.bbox_checkbox = QCheckBox("Show Bounding Boxes")
        self.bbox_checkbox.setChecked(True)
        self.keypoints_checkbox = QCheckBox("Show Keypoints")
        self.keypoints_checkbox.setChecked(True)
        self.skeleton_checkbox = QCheckBox("Show Skeleton")
        self.skeleton_checkbox.setChecked(True)
        self.hide_face_checkbox = QCheckBox("Hide Face Keypoints")
        self.hide_face_checkbox.setChecked(True)
        
        display_layout.addWidget(self.bbox_checkbox)
        display_layout.addWidget(self.keypoints_checkbox)
        display_layout.addWidget(self.skeleton_checkbox)
        display_layout.addWidget(self.hide_face_checkbox)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Acceleration
        accel_group = QGroupBox("Acceleration")
        accel_layout = QVBoxLayout()
        
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setChecked(True)
        self.npu_radio = QRadioButton("NPU (VX Delegate)")
        
        accel_layout.addWidget(self.cpu_radio)
        accel_layout.addWidget(self.npu_radio)
        
        delegate_label = QLabel("Delegate Path:")
        self.delegate_input = QLineEdit("/usr/lib/libvx_delegate.so")
        accel_layout.addWidget(delegate_label)
        accel_layout.addWidget(self.delegate_input)
        
        accel_group.setLayout(accel_layout)
        layout.addWidget(accel_group)
        
        # Control buttons
        self.start_button = QPushButton("Start Pose Estimation")
        self.start_button.clicked.connect(self.start_pose_estimation)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_pose_estimation)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")
        
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        # FPS display
        self.fps_label = QLabel("FPS: -- | Inference: --ms")
        self.fps_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; padding: 8px; background-color: #333; color: #0f0; border-radius: 3px; }")
        layout.addWidget(self.fps_label)
        
        layout.addStretch()
        
        return panel
    
    def create_video_panel(self):
        """Create video display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        title = QLabel("Live Pose Estimation")
        title.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; border: 2px solid #555; }")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(False)
        
        layout.addWidget(self.video_label)
        
        return panel
    
    def create_stats_panel(self):
        """Create statistics panel"""
        stats_group = QGroupBox("Statistics")
        layout = QVBoxLayout()
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: #00ff00; font-family: monospace; font-size: 14px; }")
        
        layout.addWidget(self.stats_text)
        stats_group.setLayout(layout)
        
        return stats_group
    
    def create_log_panel(self):
        """Create log panel"""
        log_group = QGroupBox("System Log")
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: monospace; font-size: 11px; }")
        self.log_text.setMaximumHeight(180)
        
        layout.addWidget(self.log_text)
        log_group.setLayout(layout)
        
        return log_group
    
    @Slot(QImage)
    def update_image(self, qt_image):
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    @Slot(str)
    def update_fps(self, msg):
        self.fps_label.setText(msg)
    
    @Slot(str)
    def update_log(self, msg):
        self.log_text.append(msg)
    
    @Slot(str)
    def update_stats(self, stats):
        self.stats_text.setText(stats)
    
    def start_pose_estimation(self):
        self.log_text.clear()
        self.log_text.append("=== Starting Pose Estimation ===\n")
        
        # Get settings
        det_model = self.detection_model_input.text()
        pose_model = self.pose_model_input.text()
        video_source = self.video_input.text()
        use_npu = self.npu_radio.isChecked()
        delegate_path = self.delegate_input.text()
        det_threshold = self.det_thresh_slider.value() / 100.0
        pose_threshold = self.pose_thresh_slider.value() / 100.0
        show_bbox = self.bbox_checkbox.isChecked()
        show_keypoints = self.keypoints_checkbox.isChecked()
        show_skeleton = self.skeleton_checkbox.isChecked()
        hide_face = self.hide_face_checkbox.isChecked()
        
        # Validate
        if not os.path.exists(det_model):
            self.log_text.append(f"ERROR: Detection model not found: {det_model}")
            return
        
        if not os.path.exists(pose_model):
            self.log_text.append(f"ERROR: Pose model not found: {pose_model}")
            return
        
        # Configure thread
        self.pose_thread.set_detection_model(det_model)
        self.pose_thread.set_pose_model(pose_model)
        self.pose_thread.set_video_source(video_source)
        self.pose_thread.set_delegate(use_npu, delegate_path)
        self.pose_thread.set_detection_threshold(det_threshold)
        self.pose_thread.set_pose_threshold(pose_threshold)
        self.pose_thread.set_display_options(show_bbox, show_keypoints, show_skeleton, hide_face)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.disable_controls(True)
        
        # Start
        self.pose_thread.start()
    
    def stop_pose_estimation(self):
        self.pose_thread.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.disable_controls(False)
        
        self.video_label.clear()
        self.video_label.setText("Video Stopped")
    
    def disable_controls(self, disabled):
        self.cpu_radio.setEnabled(not disabled)
        self.npu_radio.setEnabled(not disabled)
        self.detection_model_input.setEnabled(not disabled)
        self.pose_model_input.setEnabled(not disabled)
        self.video_input.setEnabled(not disabled)
        self.delegate_input.setEnabled(not disabled)
        self.det_thresh_slider.setEnabled(not disabled)
        self.pose_thresh_slider.setEnabled(not disabled)
    
    def closeEvent(self, event):
        self.pose_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = PoseEstimationGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

