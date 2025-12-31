import sys
import os
import time
import numpy as np

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGroupBox, 
                             QTextEdit, QRadioButton, QLineEdit, QComboBox,
                             QCheckBox)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap

# Import cv2 AFTER PySide6
import cv2

# CRITICAL FIX: Remove cv2's Qt plugin paths
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

from ultrafastLaneDetector import UltrafastLaneDetector, ModelType


class LaneDetectionThread(QThread):
    """Thread for lane detection"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    lane_stats_signal = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.model_path = "models/model_full_integer_quant.tflite"
        self.video_source = "input.mp4"
        self.model_type = ModelType.TUSIMPLE
        self.use_npu = False
        self.model_dtype = 'int8'
        
        self.lane_detector = None
    
    def set_model_path(self, path):
        """Set model path"""
        self.model_path = path
    
    def set_video_source(self, source):
        """Set video source"""
        self.video_source = source
    
    def set_model_type(self, model_type_str):
        """Set model type"""
        if model_type_str == "TUSIMPLE":
            self.model_type = ModelType.TUSIMPLE
        elif model_type_str == "CULANE":
            self.model_type = ModelType.CULANE
        else:
            self.model_type = ModelType.TUSIMPLE
    
    def set_npu(self, use_npu):
        """Set NPU usage"""
        self.use_npu = use_npu
    
    def set_model_dtype(self, dtype):
        """Set model data type"""
        self.model_dtype = dtype
    
    def initialize_detector(self):
        """Initialize lane detector"""
        try:
            self.log_signal.emit(f"Loading lane detection model: {self.model_path}")
            self.log_signal.emit(f"Model type: {self.model_type}")
            self.log_signal.emit(f"Model dtype: {self.model_dtype}")
            self.log_signal.emit(f"NPU: {'Enabled' if self.use_npu else 'Disabled'}")
            
            self.lane_detector = UltrafastLaneDetector(
                self.model_path,
                self.model_type,
                use_npu=self.use_npu,
                model_dtype=self.model_dtype
            )
            
            mode = "NPU" if self.use_npu else "CPU"
            self.log_signal.emit(f"✓ Lane detector initialized on {mode}")
            
            return True
            
        except Exception as e:
            self.log_signal.emit(f"Error initializing detector: {e}")
            return False
    
    def get_lane_statistics(self):
        """Get lane detection statistics from detector"""
        lane_stats = {
            'lanes_detected': 0,
            'left_lane': False,
            'right_lane': False,
            'center_lane': False
        }
        
        try:
            # Check if lanes_points attribute exists
            if hasattr(self.lane_detector, 'lanes_points'):
                lanes_points = self.lane_detector.lanes_points
                
                # Handle numpy array properly
                if isinstance(lanes_points, np.ndarray):
                    # Count non-None lanes
                    num_lanes = 0
                    for i in range(lanes_points.shape[0]):
                        lane = lanes_points[i]
                        if lane is not None and len(lane) > 0:
                            num_lanes += 1
                    
                    lane_stats['lanes_detected'] = num_lanes
                    
                    # Determine which lanes are detected (assuming left to right order)
                    if num_lanes >= 1:
                        lane_stats['left_lane'] = True
                    if num_lanes >= 2:
                        lane_stats['center_lane'] = True
                    if num_lanes >= 3:
                        lane_stats['right_lane'] = True
                elif isinstance(lanes_points, list):
                    # Handle as list
                    num_lanes = len([lane for lane in lanes_points if lane is not None and len(lane) > 0])
                    lane_stats['lanes_detected'] = num_lanes
                    
                    if num_lanes >= 1:
                        lane_stats['left_lane'] = True
                    if num_lanes >= 2:
                        lane_stats['center_lane'] = True
                    if num_lanes >= 3:
                        lane_stats['right_lane'] = True
        except Exception as e:
            # If there's any error, just return default stats
            self.log_signal.emit(f"Warning: Could not get lane stats: {e}")
        
        return lane_stats
    
    def run(self):
        """Main processing loop"""
        if not self.initialize_detector():
            return
        
        # Initialize video capture
        if self.video_source.isdigit():
            cap = cv2.VideoCapture(int(self.video_source))
        else:
            cap = cv2.VideoCapture(self.video_source)
        
        if not cap.isOpened():
            self.log_signal.emit(f"Error: Cannot open video source {self.video_source}")
            return
        
        self.log_signal.emit(f"✓ Video source initialized: {self.video_source}")
        self.log_signal.emit("Starting lane detection...\n")
        
        # Performance tracking
        total_frames = 0
        total_time = 0
        self.running = True
        first_inference = True
        
        prev_time = time.time()
        
        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Loop video if it's a file
                if not self.video_source.isdigit():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
            
            total_frames += 1
            loop_start = time.time()
            
            # Detect lanes
            detect_start = time.time()
            output_img = self.lane_detector.detect_lanes(frame)
            detect_end = time.time()
            
            if first_inference:
                warmup_time = int((detect_end - detect_start) * 1000)
                self.log_signal.emit(f"First inference time: {warmup_time}ms (includes warmup)")
                first_inference = False
            
            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time
            
            # Calculate average FPS
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            avg_fps = int(total_frames / total_time) if total_time > 0 else 0
            detect_time = (detect_end - detect_start) * 1000
            
            # Get lane statistics
            lane_stats = self.get_lane_statistics()
            
            # Status message
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {int(fps)} | Avg: {avg_fps} | Detection: {detect_time:.2f}ms | Mode: {mode}"
            
            # Draw FPS on frame
            cv2.putText(output_img, f"FPS: {int(fps)}", (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Draw lane count
            lanes_text = f"Lanes: {lane_stats['lanes_detected']}"
            cv2.putText(output_img, lanes_text, (20, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # Convert frame to QImage
            rgb_image = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Emit signals
            self.change_pixmap_signal.emit(qt_image)
            self.fps_signal.emit(msg)
            self.lane_stats_signal.emit(lane_stats)
        
        # Cleanup
        cap.release()
        self.log_signal.emit("\n✓ Lane detection stopped")
    
    def stop(self):
        """Stop the detection thread"""
        self.running = False
        self.wait()


class LaneDetectionGUI(QMainWindow):
    """Main GUI for lane detection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Lane Detection - UltraFast (i.MX8M Plus)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize detection thread
        self.detection_thread = LaneDetectionThread()
        self.detection_thread.change_pixmap_signal.connect(self.update_image)
        self.detection_thread.fps_signal.connect(self.update_fps)
        self.detection_thread.log_signal.connect(self.update_log)
        self.detection_thread.lane_stats_signal.connect(self.update_lane_stats)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right panel - Video and Log
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)
    
    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        model_label = QLabel("Model Path:")
        self.model_input = QLineEdit("models/model_full_integer_quant.tflite")
        model_layout.addWidget(model_label)
        model_layout.addWidget(self.model_input)
        
        type_label = QLabel("Model Type:")
        self.model_type_combo = QComboBox()
        self.model_type_combo.addItems(["TUSIMPLE", "CULANE"])
        model_layout.addWidget(type_label)
        model_layout.addWidget(self.model_type_combo)
        
        dtype_label = QLabel("Model Data Type:")
        self.dtype_combo = QComboBox()
        self.dtype_combo.addItems(["int8", "float32"])
        model_layout.addWidget(dtype_label)
        model_layout.addWidget(self.dtype_combo)
        
        video_label = QLabel("Video Source:")
        self.video_input = QLineEdit("input.mp4")
        model_layout.addWidget(video_label)
        model_layout.addWidget(self.video_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Lane statistics
        stats_group = QGroupBox("Lane Detection Status")
        stats_layout = QVBoxLayout()
        
        self.lanes_count_label = QLabel("Lanes Detected: 0")
        self.lanes_count_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; padding: 10px; background-color: #333; color: #0f0; border-radius: 5px; }")
        self.lanes_count_label.setAlignment(Qt.AlignCenter)
        stats_layout.addWidget(self.lanes_count_label)
        
        self.left_lane_indicator = QLabel("◯ Left Lane")
        self.left_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #888; }")
        stats_layout.addWidget(self.left_lane_indicator)
        
        self.center_lane_indicator = QLabel("◯ Center Lane")
        self.center_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #888; }")
        stats_layout.addWidget(self.center_lane_indicator)
        
        self.right_lane_indicator = QLabel("◯ Right Lane")
        self.right_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #888; }")
        stats_layout.addWidget(self.right_lane_indicator)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Acceleration settings
        accel_group = QGroupBox("Acceleration")
        accel_layout = QVBoxLayout()
        
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setChecked(True)
        self.npu_radio = QRadioButton("NPU (VX Delegate)")
        
        accel_layout.addWidget(self.cpu_radio)
        accel_layout.addWidget(self.npu_radio)
        
        info_label = QLabel("Note: NPU requires VX delegate\nto be configured in the model")
        info_label.setStyleSheet("QLabel { font-size: 10px; color: #888; padding: 5px; }")
        accel_layout.addWidget(info_label)
        
        accel_group.setLayout(accel_layout)
        layout.addWidget(accel_group)
        
        # Control buttons
        self.start_button = QPushButton("Start Detection")
        self.start_button.clicked.connect(self.start_detection)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")
        
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")
        
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        
        # FPS display
        self.fps_label = QLabel("FPS: -- | Detection: --ms")
        self.fps_label.setStyleSheet("QLabel { font-size: 11px; font-weight: bold; padding: 8px; background-color: #333; color: #0f0; border-radius: 3px; }")
        layout.addWidget(self.fps_label)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create right panel with video and log"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Video display
        title = QLabel("Lane Detection Video")
        title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; }")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; border: 2px solid #555; }")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(False)
        
        layout.addWidget(self.video_label, 3)
        
        # Log display
        log_group = QGroupBox("System Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: monospace; font-size: 11px; }")
        self.log_text.setMaximumHeight(180)
        
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        
        layout.addWidget(log_group, 1)
        
        return panel
    
    @Slot(QImage)
    def update_image(self, qt_image):
        """Update video display"""
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    @Slot(str)
    def update_fps(self, msg):
        """Update FPS display"""
        self.fps_label.setText(msg)
    
    @Slot(str)
    def update_log(self, msg):
        """Update log display"""
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {msg}")
    
    @Slot(dict)
    def update_lane_stats(self, stats):
        """Update lane detection statistics"""
        num_lanes = stats.get('lanes_detected', 0)
        
        # Update lane count
        self.lanes_count_label.setText(f"Lanes Detected: {num_lanes}")
        if num_lanes > 0:
            self.lanes_count_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; padding: 10px; background-color: #0f0; color: #000; border-radius: 5px; }")
        else:
            self.lanes_count_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; padding: 10px; background-color: #333; color: #888; border-radius: 5px; }")
        
        # Update individual lane indicators
        if stats.get('left_lane', False):
            self.left_lane_indicator.setText("● Left Lane")
            self.left_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #0f0; font-weight: bold; }")
        else:
            self.left_lane_indicator.setText("◯ Left Lane")
            self.left_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #888; }")
        
        if stats.get('center_lane', False):
            self.center_lane_indicator.setText("● Center Lane")
            self.center_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #0f0; font-weight: bold; }")
        else:
            self.center_lane_indicator.setText("◯ Center Lane")
            self.center_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #888; }")
        
        if stats.get('right_lane', False):
            self.right_lane_indicator.setText("● Right Lane")
            self.right_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #0f0; font-weight: bold; }")
        else:
            self.right_lane_indicator.setText("◯ Right Lane")
            self.right_lane_indicator.setStyleSheet("QLabel { font-size: 12px; padding: 5px; color: #888; }")
    
    def start_detection(self):
        """Start lane detection"""
        self.log_text.clear()
        self.update_log("=== Starting Lane Detection ===")
        
        # Get settings
        model_path = self.model_input.text()
        model_type = self.model_type_combo.currentText()
        model_dtype = self.dtype_combo.currentText()
        video_source = self.video_input.text()
        use_npu = self.npu_radio.isChecked()
        
        # Validate model file
        if not os.path.exists(model_path):
            self.update_log(f"ERROR: Model file not found: {model_path}")
            return
        
        # Configure detection thread
        self.detection_thread.set_model_path(model_path)
        self.detection_thread.set_model_type(model_type)
        self.detection_thread.set_model_dtype(model_dtype)
        self.detection_thread.set_video_source(video_source)
        self.detection_thread.set_npu(use_npu)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cpu_radio.setEnabled(False)
        self.npu_radio.setEnabled(False)
        self.model_input.setEnabled(False)
        self.model_type_combo.setEnabled(False)
        self.dtype_combo.setEnabled(False)
        self.video_input.setEnabled(False)
        
        # Start thread
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop lane detection"""
        self.detection_thread.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cpu_radio.setEnabled(True)
        self.npu_radio.setEnabled(True)
        self.model_input.setEnabled(True)
        self.model_type_combo.setEnabled(True)
        self.dtype_combo.setEnabled(True)
        self.video_input.setEnabled(True)
        
        self.video_label.clear()
        self.video_label.setText("Detection Stopped")
        self.update_log("Lane detection stopped")
    
    def closeEvent(self, event):
        """Handle window close"""
        self.detection_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = LaneDetectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

