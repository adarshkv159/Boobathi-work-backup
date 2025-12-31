import sys
import os
import time
import numpy as np

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QLineEdit, QGroupBox, QRadioButton, QTextEdit)
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

# Also remove general Qt plugin path if set by cv2
if "QT_PLUGIN_PATH" in os.environ:
    cv2_plugin_path = os.path.join(os.path.dirname(cv2.__file__), 'qt', 'plugins')
    if cv2_plugin_path in os.environ.get("QT_PLUGIN_PATH", ""):
        os.environ.pop("QT_PLUGIN_PATH")

import tflite_runtime.interpreter as tflite
from labels import label2string


class VideoThread(QThread):
    """Thread for video capture and object detection"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.model_path = "ssd_mobilenet_v1_quant.tflite"
        self.camera_index = 0
        self.delegate_path = ""
        self.use_npu = False
        self.interpreter = None
        self.vid = None
        
    def set_model_path(self, path):
        """Set the model path"""
        self.model_path = path
        
    def set_camera(self, camera_input):
        """Set camera index or video file path"""
        if camera_input.isdigit():
            self.camera_index = int(camera_input)
        else:
            self.camera_index = camera_input
            
    def set_delegate(self, use_npu, delegate_path=""):
        """Set NPU delegate configuration"""
        self.use_npu = use_npu
        self.delegate_path = delegate_path
        
    def initialize_model(self):
        """Initialize TFLite interpreter with or without NPU delegate"""
        try:
            if self.use_npu and self.delegate_path:
                # Check if delegate file exists
                if not os.path.exists(self.delegate_path):
                    error_msg = f"ERROR: Delegate file not found at {self.delegate_path}"
                    print(error_msg)
                    self.log_signal.emit(error_msg)
                    return False
                
                # Delegate options for i.MX8M Plus NPU
                ext_delegate_options = {}
                
                self.log_signal.emit(f"Loading NPU delegate from: {self.delegate_path}")
                self.log_signal.emit(f"Delegate options: {ext_delegate_options}")
                
                try:
                    ext_delegate = [tflite.load_delegate(
                        self.delegate_path, 
                        ext_delegate_options
                    )]
                    
                    self.interpreter = tflite.Interpreter(
                        model_path=self.model_path, 
                        experimental_delegates=ext_delegate
                    )
                    self.log_signal.emit("✓ NPU delegate loaded successfully")
                    
                except Exception as e:
                    error_msg = f"ERROR loading delegate: {str(e)}"
                    print(error_msg)
                    self.log_signal.emit(error_msg)
                    self.log_signal.emit("Falling back to CPU inference")
                    self.interpreter = tflite.Interpreter(model_path=self.model_path)
                    self.use_npu = False
            else:
                self.log_signal.emit("Using CPU inference")
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            
            self.interpreter.allocate_tensors()
            
            # Verify which backend is being used
            self.verify_npu_usage()
            
            return True
        except Exception as e:
            error_msg = f"Error initializing model: {e}"
            print(error_msg)
            self.log_signal.emit(error_msg)
            return False
    
    def verify_npu_usage(self):
        """Verify if NPU is actually being used"""
        try:
            # Get tensor details to check if delegate is applied
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            self.log_signal.emit(f"\n=== Model Information ===")
            self.log_signal.emit(f"Input shape: {input_details[0]['shape']}")
            self.log_signal.emit(f"Input dtype: {input_details[0]['dtype']}")
            self.log_signal.emit(f"Output count: {len(output_details)}")
            
            if self.use_npu:
                self.log_signal.emit(f"\n✓ NPU mode ENABLED")
                self.log_signal.emit(f"NOTE: First inference will be slower (NPU warmup)")
            else:
                self.log_signal.emit(f"\n✓ CPU mode ENABLED")
                
        except Exception as e:
            self.log_signal.emit(f"Verification error: {e}")
            
    def run(self):
        """Main video processing loop"""
        if not self.initialize_model():
            return
            
        # Get model details
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        
        # Initialize video capture
        self.vid = cv2.VideoCapture(self.camera_index)
        if not self.vid.isOpened():
            error_msg = f"Error: Could not open video source {self.camera_index}"
            print(error_msg)
            self.log_signal.emit(error_msg)
            return
        
        self.log_signal.emit(f"\n✓ Video capture initialized: {self.camera_index}")
        self.log_signal.emit("Starting inference loop...\n")
            
        # Performance tracking
        total_fps = 0
        total_time = 0
        self.running = True
        first_inference = True
        
        while self.running:
            ret, frame = self.vid.read()
            if not ret or frame is None:
                break
                
            total_fps += 1
            loop_start = time.time()
            
            # Preprocess frame
            img = cv2.resize(frame, (width, height)).astype(np.uint8)
            input_data = np.expand_dims(img, axis=0)
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference
            invoke_start = time.time()
            self.interpreter.invoke()
            invoke_end = time.time()
            
            # Log first inference time (includes NPU warmup)
            if first_inference:
                warmup_time = int((invoke_end - invoke_start) * 1000)
                self.log_signal.emit(f"First inference time: {warmup_time}ms (includes warmup)")
                first_inference = False
            
            # Get results
            boxes = self.interpreter.get_tensor(output_details[0]['index'])[0]
            labels = self.interpreter.get_tensor(output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(output_details[2]['index'])[0]
            number = self.interpreter.get_tensor(output_details[3]['index'])[0]
            
            # Draw detections
            detection_count = 0
            for i in range(int(number)):
                if scores[i] > 0.5:  # Confidence threshold
                    detection_count += 1
                    box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
                    x0 = max(2, int(box[1] * frame.shape[1]))
                    y0 = max(2, int(box[0] * frame.shape[0]))
                    x1 = int(box[3] * frame.shape[1])
                    y1 = int(box[2] * frame.shape[0])
                    
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    label_text = f"{label2string[labels[i]]} {scores[i]:.2f}"
                    cv2.putText(frame, label_text, (x0, y0 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Calculate performance metrics
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_fps / total_time) if total_time > 0 else 0
            invoke_time = int((invoke_end - invoke_start) * 1000)
            
            # Status message with detection count
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {fps} | Inference: {invoke_time}ms | Mode: {mode} | Detections: {detection_count}"
            
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)
            
            # Convert frame to QImage for display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Emit signals
            self.change_pixmap_signal.emit(qt_image)
            self.fps_signal.emit(msg)
            
        # Cleanup
        if self.vid:
            self.vid.release()
        self.log_signal.emit("\n✓ Video capture stopped")
            
    def stop(self):
        """Stop the video thread"""
        self.running = False
        self.wait()


class ObjectDetectionGUI(QMainWindow):
    """Main GUI window for object detection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection - SSD MobileNet (i.MX8M Plus)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize video thread
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.fps_signal.connect(self.update_fps)
        self.video_thread.log_signal.connect(self.update_log)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top section - video and controls
        top_layout = QHBoxLayout()
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        top_layout.addWidget(control_panel, 1)
        
        # Right panel - Video display
        video_panel = self.create_video_panel()
        top_layout.addWidget(video_panel, 3)
        
        main_layout.addLayout(top_layout, 3)
        
        # Bottom section - Log display
        log_panel = self.create_log_panel()
        main_layout.addWidget(log_panel, 1)
        
    def create_control_panel(self):
        """Create the control panel with settings"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Model settings group
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        # Model path input
        model_path_label = QLabel("Model Path:")
        self.model_path_input = QLineEdit("ssd_mobilenet_v1_quant.tflite")
        model_layout.addWidget(model_path_label)
        model_layout.addWidget(self.model_path_input)
        
        # Camera input
        camera_label = QLabel("Camera/Video Source:")
        self.camera_input = QLineEdit("0")
        model_layout.addWidget(camera_label)
        model_layout.addWidget(self.camera_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Acceleration settings group
        accel_group = QGroupBox("Acceleration")
        accel_layout = QVBoxLayout()
        
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setChecked(True)
        self.npu_radio = QRadioButton("NPU (VX Delegate)")
        
        accel_layout.addWidget(self.cpu_radio)
        accel_layout.addWidget(self.npu_radio)
        
        # NPU delegate path
        delegate_label = QLabel("NPU Delegate Path:")
        self.delegate_input = QLineEdit("/usr/lib/libvx_delegate.so")
        accel_layout.addWidget(delegate_label)
        accel_layout.addWidget(self.delegate_input)
        
        # Check delegate button
        self.check_delegate_btn = QPushButton("Check Delegate")
        self.check_delegate_btn.clicked.connect(self.check_delegate)
        accel_layout.addWidget(self.check_delegate_btn)
        
        accel_group.setLayout(accel_layout)
        layout.addWidget(accel_group)
        
        # Control buttons
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; }")
        
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; font-size: 14px; }")
        
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        # FPS display
        self.fps_label = QLabel("FPS: -- | Inference: --ms")
        self.fps_label.setStyleSheet("QLabel { font-size: 12px; font-weight: bold; padding: 10px; background-color: #333; color: #0f0; }")
        layout.addWidget(self.fps_label)
        
        layout.addStretch()
        
        return panel
        
    def create_video_panel(self):
        """Create the video display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; }")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(False)
        
        layout.addWidget(self.video_label)
        
        return panel
    
    def create_log_panel(self):
        """Create the log display panel"""
        log_group = QGroupBox("System Log")
        layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: monospace; }")
        self.log_text.setMaximumHeight(200)
        
        layout.addWidget(self.log_text)
        log_group.setLayout(layout)
        
        return log_group
        
    @Slot(QImage)
    def update_image(self, qt_image):
        """Update the video display with new frame"""
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    @Slot(str)
    def update_fps(self, msg):
        """Update FPS and inference time display"""
        self.fps_label.setText(msg)
    
    @Slot(str)
    def update_log(self, msg):
        """Update log display"""
        self.log_text.append(msg)
        
    def check_delegate(self):
        """Check if delegate file exists and display info"""
        delegate_path = self.delegate_input.text()
        
        if not delegate_path:
            self.log_text.append("ERROR: No delegate path specified")
            return
            
        if os.path.exists(delegate_path):
            file_size = os.path.getsize(delegate_path) / (1024 * 1024)
            self.log_text.append(f"✓ Delegate found: {delegate_path}")
            self.log_text.append(f"  Size: {file_size:.2f} MB")
        else:
            self.log_text.append(f"✗ Delegate NOT found: {delegate_path}")
            self.log_text.append("\nCommon i.MX8M Plus delegate paths:")
            self.log_text.append("  /usr/lib/libvx_delegate.so")
            self.log_text.append("  /usr/lib64/libvx_delegate.so")
            self.log_text.append("\nCheck with: find /usr -name 'libvx_delegate.so'")
        
    def start_detection(self):
        """Start the object detection process"""
        # Clear log
        self.log_text.clear()
        self.log_text.append("=== Starting Object Detection ===\n")
        
        # Get settings from UI
        model_path = self.model_path_input.text()
        camera_input = self.camera_input.text()
        use_npu = self.npu_radio.isChecked()
        delegate_path = self.delegate_input.text()
        
        # Validate inputs
        if not os.path.exists(model_path):
            self.log_text.append(f"ERROR: Model file not found: {model_path}")
            return
        
        if use_npu and not os.path.exists(delegate_path):
            self.log_text.append(f"ERROR: Delegate file not found: {delegate_path}")
            self.log_text.append("Either fix the path or switch to CPU mode")
            return
        
        # Configure video thread
        self.video_thread.set_model_path(model_path)
        self.video_thread.set_camera(camera_input)
        self.video_thread.set_delegate(use_npu, delegate_path)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cpu_radio.setEnabled(False)
        self.npu_radio.setEnabled(False)
        self.model_path_input.setEnabled(False)
        self.camera_input.setEnabled(False)
        self.delegate_input.setEnabled(False)
        self.check_delegate_btn.setEnabled(False)
        
        # Start thread
        self.video_thread.start()
        
    def stop_detection(self):
        """Stop the object detection process"""
        self.video_thread.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cpu_radio.setEnabled(True)
        self.npu_radio.setEnabled(True)
        self.model_path_input.setEnabled(True)
        self.camera_input.setEnabled(True)
        self.delegate_input.setEnabled(True)
        self.check_delegate_btn.setEnabled(True)
        
        # Clear video display
        self.video_label.clear()
        self.video_label.setText("Video Stopped")
        
    def closeEvent(self, event):
        """Handle window close event"""
        self.video_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ObjectDetectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

