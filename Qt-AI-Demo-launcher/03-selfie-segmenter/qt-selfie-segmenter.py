import sys
import os
import time
import numpy as np

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QGroupBox, QRadioButton, QTextEdit, QDoubleSpinBox,
                             QSlider, QColorDialog)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QImage, QPixmap, QColor

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


def normalize_input(input_data, input_shape):
    """Fit the image size for model and change colorspace"""
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    resized_data = cv2.resize(input_data, input_shape)
    normalized_data = np.ascontiguousarray(resized_data / 255.0)
    normalized_data = normalized_data.astype("float32")
    normalized_data = normalized_data[None, ...]
    return normalized_data


class SegmentationThread(QThread):
    """Thread for video capture and selfie segmentation"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.model_path = "selfie_segmenter_int8.tflite"
        self.camera_index = 0
        self.delegate_path = ""
        self.use_npu = False
        self.threshold = 0.1
        self.interpreter = None
        self.vid = None
        self.background_mode = "white"  # white, black, blur, green, custom
        self.background_color = (0, 255, 0)  # Default green
        
    def set_model_path(self, path):
        """Set the model path"""
        self.model_path = path
        
    def set_camera(self, camera_input):
        """Set camera index or video file path"""
        if str(camera_input).isdigit():
            self.camera_index = int(camera_input)
        else:
            self.camera_index = camera_input
    
    def set_threshold(self, threshold):
        """Set segmentation threshold"""
        self.threshold = threshold
    
    def set_background_mode(self, mode):
        """Set background replacement mode"""
        self.background_mode = mode
    
    def set_background_color(self, color):
        """Set custom background color (BGR tuple)"""
        self.background_color = color
            
    def set_delegate(self, use_npu, delegate_path=""):
        """Set NPU delegate configuration"""
        self.use_npu = use_npu
        self.delegate_path = delegate_path
        
    def initialize_model(self):
        """Initialize TFLite interpreter with or without NPU delegate"""
        try:
            # Load model with or without delegate
            if self.use_npu and self.delegate_path:
                # Check if delegate file exists
                if not os.path.exists(self.delegate_path):
                    error_msg = f"ERROR: Delegate file not found at {self.delegate_path}"
                    print(error_msg)
                    self.log_signal.emit(error_msg)
                    return False
                
                self.log_signal.emit(f"Loading NPU delegate from: {self.delegate_path}")
                
                try:
                    ext_delegate = [tflite.load_delegate(self.delegate_path)]
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
            
            # Get model details
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            
            self.input_index = input_details[0]['index']
            self.input_shape = input_details[0]['shape']
            self.output_index = output_details[0]['index']
            
            # Log model info
            self.log_signal.emit(f"\n=== Model Information ===")
            self.log_signal.emit(f"Input shape: {self.input_shape}")
            self.log_signal.emit(f"Input dtype: {input_details[0]['dtype']}")
            self.log_signal.emit(f"Output shape: {output_details[0]['shape']}")
            
            if self.use_npu:
                self.log_signal.emit(f"\n✓ NPU mode ENABLED")
                self.log_signal.emit(f"NOTE: First inference will be slower (NPU warmup)")
            else:
                self.log_signal.emit(f"\n✓ CPU mode ENABLED")
            
            return True
            
        except Exception as e:
            error_msg = f"Error initializing model: {e}"
            print(error_msg)
            self.log_signal.emit(error_msg)
            return False
    
    def apply_background(self, frame, mask):
        """Apply background replacement based on mode"""
        condition = np.stack((mask,) * 3, axis=-1) > self.threshold
        
        if self.background_mode == "white":
            background = np.full(shape=frame.shape, fill_value=255, dtype=np.uint8)
        elif self.background_mode == "black":
            background = np.full(shape=frame.shape, fill_value=0, dtype=np.uint8)
        elif self.background_mode == "blur":
            background = cv2.GaussianBlur(frame, (51, 51), 0)
        elif self.background_mode == "green":
            background = np.full(shape=frame.shape, fill_value=[0, 255, 0], dtype=np.uint8)
        elif self.background_mode == "custom":
            background = np.full(shape=frame.shape, fill_value=self.background_color, dtype=np.uint8)
        else:
            background = frame
        
        # Blend foreground and background
        segmentation = np.where(condition, frame, background)
        return segmentation
            
    def run(self):
        """Main video processing loop"""
        if not self.initialize_model():
            return
            
        # Initialize video capture
        self.vid = cv2.VideoCapture(self.camera_index)
        if not self.vid.isOpened():
            error_msg = f"Error: Could not open video source {self.camera_index}"
            print(error_msg)
            self.log_signal.emit(error_msg)
            return
        
        self.log_signal.emit(f"\n✓ Video capture initialized: {self.camera_index}")
        self.log_signal.emit("Starting segmentation loop...\n")
            
        # Performance tracking
        total_frames = 0
        total_time = 0
        self.running = True
        first_inference = True
        
        while self.running:
            ret, frame = self.vid.read()
            if not ret or frame is None:
                break
                
            total_frames += 1
            loop_start = time.time()
            
            # Preprocess frame
            input_frame = normalize_input(frame, (self.input_shape[2], self.input_shape[1]))
            self.interpreter.set_tensor(self.input_index, input_frame)
            
            # Run inference
            invoke_start = time.time()
            self.interpreter.invoke()
            invoke_end = time.time()
            
            # Log first inference time (includes NPU warmup)
            if first_inference:
                warmup_time = int((invoke_end - invoke_start) * 1000)
                self.log_signal.emit(f"First inference time: {warmup_time}ms (includes warmup)")
                first_inference = False
            
            # Get mask and resize to frame size
            mask = self.interpreter.get_tensor(self.output_index)[0]
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), cv2.INTER_CUBIC)
            
            # Apply background effect
            segmentation = self.apply_background(frame, mask)
            
            # Calculate performance metrics
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_frames / total_time) if total_time > 0 else 0
            invoke_time = (invoke_end - invoke_start) * 1000
            
            # Status message
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {fps} | Inference: {invoke_time:.2f}ms | Mode: {mode} | Background: {self.background_mode}"
            
            # Draw info on frame
            cv2.putText(segmentation, msg, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Convert frame to QImage for display
            rgb_image = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)
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
        """Stop the segmentation thread"""
        self.running = False
        self.wait()


class SelfieSegmentationGUI(QMainWindow):
    """Main GUI window for selfie segmentation"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Selfie Segmentation (i.MX8M Plus)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize segmentation thread
        self.segmentation_thread = SegmentationThread()
        self.segmentation_thread.change_pixmap_signal.connect(self.update_image)
        self.segmentation_thread.fps_signal.connect(self.update_fps)
        self.segmentation_thread.log_signal.connect(self.update_log)
        
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
        self.model_path_input = QLineEdit("selfie_segmenter_int8.tflite")
        model_layout.addWidget(model_path_label)
        model_layout.addWidget(self.model_path_input)
        
        # Camera input
        camera_label = QLabel("Camera/Video Source:")
        self.camera_input = QLineEdit("0")
        model_layout.addWidget(camera_label)
        model_layout.addWidget(self.camera_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Segmentation parameters group
        param_group = QGroupBox("Segmentation Parameters")
        param_layout = QVBoxLayout()
        
        # Threshold slider
        threshold_label = QLabel("Mask Threshold: 0.10")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(10)
        self.threshold_slider.valueChanged.connect(
            lambda v: threshold_label.setText(f"Mask Threshold: {v/100:.2f}")
        )
        param_layout.addWidget(threshold_label)
        param_layout.addWidget(self.threshold_slider)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Background settings group
        bg_group = QGroupBox("Background Settings")
        bg_layout = QVBoxLayout()
        
        self.bg_white_radio = QRadioButton("White Background")
        self.bg_white_radio.setChecked(True)
        self.bg_black_radio = QRadioButton("Black Background")
        self.bg_blur_radio = QRadioButton("Blur Background")
        self.bg_green_radio = QRadioButton("Green Screen")
        self.bg_custom_radio = QRadioButton("Custom Color")
        
        bg_layout.addWidget(self.bg_white_radio)
        bg_layout.addWidget(self.bg_black_radio)
        bg_layout.addWidget(self.bg_blur_radio)
        bg_layout.addWidget(self.bg_green_radio)
        bg_layout.addWidget(self.bg_custom_radio)
        
        # Custom color picker
        self.color_picker_btn = QPushButton("Pick Custom Color")
        self.color_picker_btn.clicked.connect(self.pick_color)
        self.custom_color = QColor(0, 255, 0)
        self.color_preview = QLabel()
        self.color_preview.setFixedHeight(30)
        self.update_color_preview()
        
        bg_layout.addWidget(self.color_picker_btn)
        bg_layout.addWidget(self.color_preview)
        
        bg_group.setLayout(bg_layout)
        layout.addWidget(bg_group)
        
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
        self.start_button = QPushButton("Start Segmentation")
        self.start_button.clicked.connect(self.start_segmentation)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")
        
        self.stop_button = QPushButton("Stop Segmentation")
        self.stop_button.clicked.connect(self.stop_segmentation)
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
        """Create the video display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; border: 2px solid #555; }")
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
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: monospace; font-size: 11px; }")
        self.log_text.setMaximumHeight(180)
        
        layout.addWidget(self.log_text)
        log_group.setLayout(layout)
        
        return log_group
    
    def pick_color(self):
        """Open color picker dialog"""
        color = QColorDialog.getColor(self.custom_color, self, "Select Background Color")
        if color.isValid():
            self.custom_color = color
            self.update_color_preview()
    
    def update_color_preview(self):
        """Update the color preview label"""
        self.color_preview.setStyleSheet(f"QLabel {{ background-color: {self.custom_color.name()}; border: 1px solid #666; }}")
        
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
    
    def get_background_mode(self):
        """Get selected background mode"""
        if self.bg_white_radio.isChecked():
            return "white"
        elif self.bg_black_radio.isChecked():
            return "black"
        elif self.bg_blur_radio.isChecked():
            return "blur"
        elif self.bg_green_radio.isChecked():
            return "green"
        elif self.bg_custom_radio.isChecked():
            return "custom"
        return "white"
        
    def start_segmentation(self):
        """Start the selfie segmentation process"""
        # Clear displays
        self.log_text.clear()
        self.log_text.append("=== Starting Selfie Segmentation ===\n")
        
        # Get settings from UI
        model_path = self.model_path_input.text()
        camera_input = self.camera_input.text()
        use_npu = self.npu_radio.isChecked()
        delegate_path = self.delegate_input.text()
        threshold = self.threshold_slider.value() / 100.0
        background_mode = self.get_background_mode()
        
        # Get custom color (BGR format for OpenCV)
        custom_color_bgr = (
            self.custom_color.blue(),
            self.custom_color.green(),
            self.custom_color.red()
        )
        
        # Validate inputs
        if not os.path.exists(model_path):
            self.log_text.append(f"ERROR: Model file not found: {model_path}")
            return
        
        if use_npu and not os.path.exists(delegate_path):
            self.log_text.append(f"ERROR: Delegate file not found: {delegate_path}")
            self.log_text.append("Either fix the path or switch to CPU mode")
            return
        
        # Configure segmentation thread
        self.segmentation_thread.set_model_path(model_path)
        self.segmentation_thread.set_camera(camera_input)
        self.segmentation_thread.set_threshold(threshold)
        self.segmentation_thread.set_background_mode(background_mode)
        self.segmentation_thread.set_background_color(custom_color_bgr)
        self.segmentation_thread.set_delegate(use_npu, delegate_path)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cpu_radio.setEnabled(False)
        self.npu_radio.setEnabled(False)
        self.model_path_input.setEnabled(False)
        self.camera_input.setEnabled(False)
        self.threshold_slider.setEnabled(False)
        self.bg_white_radio.setEnabled(False)
        self.bg_black_radio.setEnabled(False)
        self.bg_blur_radio.setEnabled(False)
        self.bg_green_radio.setEnabled(False)
        self.bg_custom_radio.setEnabled(False)
        self.color_picker_btn.setEnabled(False)
        self.delegate_input.setEnabled(False)
        self.check_delegate_btn.setEnabled(False)
        
        # Start thread
        self.segmentation_thread.start()
        
    def stop_segmentation(self):
        """Stop the selfie segmentation process"""
        self.segmentation_thread.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cpu_radio.setEnabled(True)
        self.npu_radio.setEnabled(True)
        self.model_path_input.setEnabled(True)
        self.camera_input.setEnabled(True)
        self.threshold_slider.setEnabled(True)
        self.bg_white_radio.setEnabled(True)
        self.bg_black_radio.setEnabled(True)
        self.bg_blur_radio.setEnabled(True)
        self.bg_green_radio.setEnabled(True)
        self.bg_custom_radio.setEnabled(True)
        self.color_picker_btn.setEnabled(True)
        self.delegate_input.setEnabled(True)
        self.check_delegate_btn.setEnabled(True)
        
        # Clear video display
        self.video_label.clear()
        self.video_label.setText("Video Stopped")
        
    def closeEvent(self, event):
        """Handle window close event"""
        self.segmentation_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = SelfieSegmentationGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

