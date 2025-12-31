import sys
import os
import time
import numpy as np

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGroupBox, 
                             QTextEdit, QRadioButton, QLineEdit, QCheckBox,
                             QSlider)
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

from hand_tracker import HandTracker

# Model paths
PALM_MODEL_PATH = "palm_detection_builtin_256_integer_quant.tflite"
LANDMARK_MODEL_PATH = "hand_landmark_3d_256_integer_quant.tflite"
ANCHORS_PATH = "anchors.csv"


class HandGestureThread(QThread):
    """Thread for hand gesture detection and tracking"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    hands_detected_signal = Signal(int)  # Number of hands detected
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.video_source = "0"
        self.delegate_path = ""
        self.use_npu = False
        self.show_landmarks = True
        self.show_connections = True
        self.box_shift = 0.2
        self.box_enlarge = 1.3
        
        self.detector = None
        
        # Hand landmark connections
        self.connections = [
            (5, 6), (6, 7), (7, 8),
            (9, 10), (10, 11), (11, 12),
            (13, 14), (14, 15), (15, 16),
            (0, 5), (5, 9), (9, 13), (13, 17), (0, 9), (0, 13),
            (0, 17), (17, 18), (18, 19), (19, 20), 
            (0, 1), (1, 2), (2, 3), (3, 4)
        ]
    
    def set_video_source(self, source):
        """Set video source"""
        self.video_source = source
    
    def set_delegate(self, use_npu, delegate_path):
        """Set NPU delegate configuration"""
        self.use_npu = use_npu
        self.delegate_path = delegate_path
    
    def set_display_options(self, landmarks, connections):
        """Set display options"""
        self.show_landmarks = landmarks
        self.show_connections = connections
    
    def set_detection_params(self, box_shift, box_enlarge):
        """Set detection parameters"""
        self.box_shift = box_shift
        self.box_enlarge = box_enlarge
    
    def draw_landmarks(self, points, frame):
        """Draw hand landmarks and connections"""
        if points is None:
            return
        
        # Draw connections first (underneath)
        if self.show_connections:
            for connection in self.connections:
                x0, y0 = points[connection[0]]
                x1, y1 = points[connection[1]]
                cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), (255, 0, 0), 2)
        
        # Draw landmarks on top
        if self.show_landmarks:
            for i, point in enumerate(points):
                x, y = point
                # Different colors for different parts of hand
                if i == 0:  # Wrist
                    color = (255, 255, 0)  # Yellow
                elif i in [4, 8, 12, 16, 20]:  # Fingertips
                    color = (0, 0, 255)  # Red
                else:
                    color = (0, 255, 0)  # Green
                
                cv2.circle(frame, (int(x), int(y)), 4, color, -1)
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)
    
    def initialize_detector(self):
        """Initialize hand detector"""
        try:
            delegate = self.delegate_path if self.use_npu else ""
            
            self.log_signal.emit("Initializing hand gesture detector...")
            
            self.detector = HandTracker(
                PALM_MODEL_PATH, 
                LANDMARK_MODEL_PATH, 
                ANCHORS_PATH, 
                delegate,
                box_shift=self.box_shift,
                box_enlarge=self.box_enlarge
            )
            
            mode = "NPU" if self.use_npu and delegate else "CPU"
            self.log_signal.emit(f"✓ Hand detector initialized on {mode}")
            
            return True
            
        except Exception as e:
            self.log_signal.emit(f"Error initializing detector: {e}")
            return False
    
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
        self.log_signal.emit("Starting hand gesture detection...\n")
        
        # Performance tracking
        total_frames = 0
        total_time = 0
        self.running = True
        first_inference = True
        
        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            total_frames += 1
            loop_start = time.time()
            
            # Convert to RGB for detector
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run hand detection
            detect_start = time.time()
            points, _ = self.detector(image_rgb)
            detect_end = time.time()
            
            if first_inference:
                warmup_time = int((detect_end - detect_start) * 1000)
                self.log_signal.emit(f"First inference time: {warmup_time}ms (includes warmup)")
                first_inference = False
            
            # Draw landmarks
            num_hands = 1 if points is not None else 0
            self.draw_landmarks(points, frame)
            
            # Emit hands detected signal
            self.hands_detected_signal.emit(num_hands)
            
            # Calculate performance metrics
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_frames / total_time) if total_time > 0 else 0
            detect_time = (detect_end - detect_start) * 1000
            
            # Status message
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {fps} | Detection: {detect_time:.2f}ms | Mode: {mode} | Hands: {num_hands}"
            
            # Draw info on frame
            cv2.putText(frame, msg, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Convert frame to QImage
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Emit signals
            self.change_pixmap_signal.emit(qt_image)
            self.fps_signal.emit(msg)
        
        # Cleanup
        cap.release()
        self.log_signal.emit("\n✓ Video processing stopped")
    
    def stop(self):
        """Stop the detection thread"""
        self.running = False
        self.wait()


class HandGestureGUI(QMainWindow):
    """Main GUI for hand gesture detection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Gesture Detection - MediaPipe (i.MX8M Plus)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize detection thread
        self.detection_thread = HandGestureThread()
        self.detection_thread.change_pixmap_signal.connect(self.update_image)
        self.detection_thread.fps_signal.connect(self.update_fps)
        self.detection_thread.log_signal.connect(self.update_log)
        self.detection_thread.hands_detected_signal.connect(self.update_hands_count)
        
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
        
        video_label = QLabel("Video Source:")
        self.video_input = QLineEdit("0")
        model_layout.addWidget(video_label)
        model_layout.addWidget(self.video_input)
        
        model_info = QLabel("Models:\n• Palm Detection\n• Hand Landmarks 3D")
        model_info.setStyleSheet("QLabel { font-size: 10px; color: #888; padding: 5px; }")
        model_layout.addWidget(model_info)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Detection parameters
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()
        
        shift_label = QLabel("Box Shift: 0.20")
        self.shift_slider = QSlider(Qt.Horizontal)
        self.shift_slider.setRange(0, 50)
        self.shift_slider.setValue(20)
        self.shift_slider.valueChanged.connect(
            lambda v: shift_label.setText(f"Box Shift: {v/100:.2f}")
        )
        param_layout.addWidget(shift_label)
        param_layout.addWidget(self.shift_slider)
        
        enlarge_label = QLabel("Box Enlarge: 1.30")
        self.enlarge_slider = QSlider(Qt.Horizontal)
        self.enlarge_slider.setRange(100, 200)
        self.enlarge_slider.setValue(130)
        self.enlarge_slider.valueChanged.connect(
            lambda v: enlarge_label.setText(f"Box Enlarge: {v/100:.2f}")
        )
        param_layout.addWidget(enlarge_label)
        param_layout.addWidget(self.enlarge_slider)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Display options
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()
        
        self.landmarks_checkbox = QCheckBox("Show Landmarks")
        self.landmarks_checkbox.setChecked(True)
        self.connections_checkbox = QCheckBox("Show Connections")
        self.connections_checkbox.setChecked(True)
        
        display_layout.addWidget(self.landmarks_checkbox)
        display_layout.addWidget(self.connections_checkbox)
        
        legend_label = QLabel("Legend:\n🟡 Wrist\n🔴 Fingertips\n🟢 Joints")
        legend_label.setStyleSheet("QLabel { font-size: 10px; padding: 5px; background-color: #2b2b2b; border-radius: 3px; }")
        display_layout.addWidget(legend_label)
        
        display_group.setLayout(display_layout)
        layout.addWidget(display_group)
        
        # Acceleration settings
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
        
        # Hands counter
        hands_group = QGroupBox("Detection Status")
        hands_layout = QVBoxLayout()
        
        self.hands_label = QLabel("Hands Detected: 0")
        self.hands_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; padding: 10px; background-color: #333; color: #0f0; border-radius: 5px; }")
        self.hands_label.setAlignment(Qt.AlignCenter)
        hands_layout.addWidget(self.hands_label)
        
        hands_group.setLayout(hands_layout)
        layout.addWidget(hands_group)
        
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
        title = QLabel("Live Hand Tracking")
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
    
    @Slot(int)
    def update_hands_count(self, count):
        """Update hands detected counter"""
        self.hands_label.setText(f"Hands Detected: {count}")
        if count > 0:
            self.hands_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; padding: 10px; background-color: #0f0; color: #000; border-radius: 5px; }")
        else:
            self.hands_label.setStyleSheet("QLabel { font-size: 18px; font-weight: bold; padding: 10px; background-color: #333; color: #888; border-radius: 5px; }")
    
    def start_detection(self):
        """Start hand gesture detection"""
        self.log_text.clear()
        self.update_log("=== Starting Hand Gesture Detection ===")
        
        # Get settings
        video_source = self.video_input.text()
        use_npu = self.npu_radio.isChecked()
        delegate_path = self.delegate_input.text()
        box_shift = self.shift_slider.value() / 100.0
        box_enlarge = self.enlarge_slider.value() / 100.0
        show_landmarks = self.landmarks_checkbox.isChecked()
        show_connections = self.connections_checkbox.isChecked()
        
        # Validate NPU settings
        if use_npu and not os.path.exists(delegate_path):
            self.update_log(f"NPU delegate not found: {delegate_path}")
            self.update_log("Switching to CPU mode")
            use_npu = False
        
        # Configure detection thread
        self.detection_thread.set_video_source(video_source)
        self.detection_thread.set_delegate(use_npu, delegate_path)
        self.detection_thread.set_detection_params(box_shift, box_enlarge)
        self.detection_thread.set_display_options(show_landmarks, show_connections)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cpu_radio.setEnabled(False)
        self.npu_radio.setEnabled(False)
        self.video_input.setEnabled(False)
        self.delegate_input.setEnabled(False)
        self.shift_slider.setEnabled(False)
        self.enlarge_slider.setEnabled(False)
        
        # Start thread
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop hand gesture detection"""
        self.detection_thread.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cpu_radio.setEnabled(True)
        self.npu_radio.setEnabled(True)
        self.video_input.setEnabled(True)
        self.delegate_input.setEnabled(True)
        self.shift_slider.setEnabled(True)
        self.enlarge_slider.setEnabled(True)
        
        self.video_label.clear()
        self.video_label.setText("Video Stopped")
        self.update_log("Detection stopped")
    
    def closeEvent(self, event):
        """Handle window close"""
        self.detection_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = HandGestureGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

