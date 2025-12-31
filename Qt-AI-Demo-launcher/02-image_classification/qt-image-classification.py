import sys
import os
import time
import numpy as np

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QGroupBox, QRadioButton, QTextEdit, QDoubleSpinBox)
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


def load_labels(filename):
    """Load labels from file"""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


class ClassificationThread(QThread):
    """Thread for video capture and image classification"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    results_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.model_path = "mobilenet_v1_1.0_224_quant.tflite"
        self.label_path = "labels.txt"
        self.camera_index = 0
        self.delegate_path = ""
        self.use_npu = False
        self.threshold = 0.6
        self.input_mean = 127.5
        self.input_std = 127.5
        self.interpreter = None
        self.vid = None
        self.labels = []
        
    def set_model_path(self, path):
        """Set the model path"""
        self.model_path = path
    
    def set_label_path(self, path):
        """Set the label file path"""
        self.label_path = path
        
    def set_camera(self, camera_input):
        """Set camera index or video file path"""
        if str(camera_input).isdigit():
            self.camera_index = int(camera_input)
        else:
            self.camera_index = camera_input
    
    def set_threshold(self, threshold):
        """Set confidence threshold"""
        self.threshold = threshold
    
    def set_normalization(self, mean, std):
        """Set input normalization parameters"""
        self.input_mean = mean
        self.input_std = std
            
    def set_delegate(self, use_npu, delegate_path=""):
        """Set NPU delegate configuration"""
        self.use_npu = use_npu
        self.delegate_path = delegate_path
        
    def initialize_model(self):
        """Initialize TFLite interpreter with or without NPU delegate"""
        try:
            # Load labels
            if not os.path.exists(self.label_path):
                error_msg = f"ERROR: Label file not found: {self.label_path}"
                print(error_msg)
                self.log_signal.emit(error_msg)
                return False
            
            self.labels = load_labels(self.label_path)
            self.log_signal.emit(f"✓ Loaded {len(self.labels)} labels")
            
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
            
            self.input_details = input_details
            self.output_details = output_details
            self.floating_model = input_details[0]['dtype'] == np.float32
            
            # Log model info
            self.log_signal.emit(f"\n=== Model Information ===")
            self.log_signal.emit(f"Input shape: {input_details[0]['shape']}")
            self.log_signal.emit(f"Input dtype: {input_details[0]['dtype']}")
            self.log_signal.emit(f"Floating model: {self.floating_model}")
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
            
    def run(self):
        """Main video processing loop"""
        if not self.initialize_model():
            return
            
        # Get model dimensions
        height = self.input_details[0]['shape'][1]
        width = self.input_details[0]['shape'][2]
        
        # Initialize video capture
        self.vid = cv2.VideoCapture(self.camera_index)
        if not self.vid.isOpened():
            error_msg = f"Error: Could not open video source {self.camera_index}"
            print(error_msg)
            self.log_signal.emit(error_msg)
            return
        
        self.log_signal.emit(f"\n✓ Video capture initialized: {self.camera_index}")
        self.log_signal.emit("Starting classification loop...\n")
            
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
            input_data = cv2.resize(frame, (width, height))
            input_data = np.expand_dims(input_data, axis=0)
            
            if self.floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std
            
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
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
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            results = np.squeeze(output_data)
            
            # Get top 5 results
            top_k = results.argsort()[-5:][::-1]
            
            # Display results on frame
            y_pos = 30
            results_text = ""
            detection_count = 0
            
            for i in top_k:
                if self.floating_model:
                    score = float(results[i])
                else:
                    score = float(results[i] / 255.0)
                
                if score < self.threshold:
                    continue
                
                detection_count += 1
                label = self.labels[i] if i < len(self.labels) else f"Class_{i}"
                
                # Draw on frame
                text = f'{label}: {score:.2f}'
                cv2.putText(frame, text, (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                # Collect for results panel
                results_text += f'{score:.4f}: {label}\n'
                y_pos += 30
            
            # Display inference time
            invoke_time = (invoke_end - invoke_start) * 1000
            cv2.putText(frame, f'Inference: {invoke_time:.2f} ms', 
                       (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Calculate FPS
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_frames / total_time) if total_time > 0 else 0
            
            # Status message
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {fps} | Inference: {invoke_time:.2f}ms | Mode: {mode} | Detections: {detection_count}"
            
            cv2.putText(frame, msg, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Convert frame to QImage for display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Emit signals
            self.change_pixmap_signal.emit(qt_image)
            self.fps_signal.emit(msg)
            self.results_signal.emit(results_text if results_text else "No detections above threshold")
            
        # Cleanup
        if self.vid:
            self.vid.release()
        self.log_signal.emit("\n✓ Video capture stopped")
            
    def stop(self):
        """Stop the classification thread"""
        self.running = False
        self.wait()


class ImageClassificationGUI(QMainWindow):
    """Main GUI window for image classification"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classification - MobileNet (i.MX8M Plus)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize classification thread
        self.classification_thread = ClassificationThread()
        self.classification_thread.change_pixmap_signal.connect(self.update_image)
        self.classification_thread.fps_signal.connect(self.update_fps)
        self.classification_thread.log_signal.connect(self.update_log)
        self.classification_thread.results_signal.connect(self.update_results)
        
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
        
        # Middle panel - Video display
        video_panel = self.create_video_panel()
        top_layout.addWidget(video_panel, 2)
        
        # Right panel - Results
        results_panel = self.create_results_panel()
        top_layout.addWidget(results_panel, 1)
        
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
        self.model_path_input = QLineEdit("mobilenet_v1_1.0_224_quant.tflite")
        model_layout.addWidget(model_path_label)
        model_layout.addWidget(self.model_path_input)
        
        # Label file path
        label_path_label = QLabel("Labels Path:")
        self.label_path_input = QLineEdit("labels.txt")
        model_layout.addWidget(label_path_label)
        model_layout.addWidget(self.label_path_input)
        
        # Camera input
        camera_label = QLabel("Camera/Video Source:")
        self.camera_input = QLineEdit("0")
        model_layout.addWidget(camera_label)
        model_layout.addWidget(self.camera_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Parameters group
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout()
        
        # Threshold
        threshold_label = QLabel("Confidence Threshold:")
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setRange(0.0, 1.0)
        self.threshold_input.setSingleStep(0.05)
        self.threshold_input.setValue(0.6)
        param_layout.addWidget(threshold_label)
        param_layout.addWidget(self.threshold_input)
        
        # Input mean
        mean_label = QLabel("Input Mean:")
        self.mean_input = QDoubleSpinBox()
        self.mean_input.setRange(0.0, 255.0)
        self.mean_input.setValue(127.5)
        param_layout.addWidget(mean_label)
        param_layout.addWidget(self.mean_input)
        
        # Input std
        std_label = QLabel("Input Std Dev:")
        self.std_input = QDoubleSpinBox()
        self.std_input.setRange(0.1, 255.0)
        self.std_input.setValue(127.5)
        param_layout.addWidget(std_label)
        param_layout.addWidget(self.std_input)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
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
        self.start_button = QPushButton("Start Classification")
        self.start_button.clicked.connect(self.start_classification)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")
        
        self.stop_button = QPushButton("Stop Classification")
        self.stop_button.clicked.connect(self.stop_classification)
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
    
    def create_results_panel(self):
        """Create the results display panel"""
        results_group = QGroupBox("Top 5 Predictions")
        layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: #00ff00; font-family: monospace; font-size: 12px; }")
        self.results_text.setPlaceholderText("Results will appear here...")
        
        layout.addWidget(self.results_text)
        results_group.setLayout(layout)
        
        return results_group
    
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
    
    @Slot(str)
    def update_results(self, results):
        """Update results display"""
        self.results_text.setText(results)
        
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
        
    def start_classification(self):
        """Start the image classification process"""
        # Clear displays
        self.log_text.clear()
        self.results_text.clear()
        self.log_text.append("=== Starting Image Classification ===\n")
        
        # Get settings from UI
        model_path = self.model_path_input.text()
        label_path = self.label_path_input.text()
        camera_input = self.camera_input.text()
        use_npu = self.npu_radio.isChecked()
        delegate_path = self.delegate_input.text()
        threshold = self.threshold_input.value()
        input_mean = self.mean_input.value()
        input_std = self.std_input.value()
        
        # Validate inputs
        if not os.path.exists(model_path):
            self.log_text.append(f"ERROR: Model file not found: {model_path}")
            return
        
        if not os.path.exists(label_path):
            self.log_text.append(f"ERROR: Label file not found: {label_path}")
            return
        
        if use_npu and not os.path.exists(delegate_path):
            self.log_text.append(f"ERROR: Delegate file not found: {delegate_path}")
            self.log_text.append("Either fix the path or switch to CPU mode")
            return
        
        # Configure classification thread
        self.classification_thread.set_model_path(model_path)
        self.classification_thread.set_label_path(label_path)
        self.classification_thread.set_camera(camera_input)
        self.classification_thread.set_threshold(threshold)
        self.classification_thread.set_normalization(input_mean, input_std)
        self.classification_thread.set_delegate(use_npu, delegate_path)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cpu_radio.setEnabled(False)
        self.npu_radio.setEnabled(False)
        self.model_path_input.setEnabled(False)
        self.label_path_input.setEnabled(False)
        self.camera_input.setEnabled(False)
        self.threshold_input.setEnabled(False)
        self.mean_input.setEnabled(False)
        self.std_input.setEnabled(False)
        self.delegate_input.setEnabled(False)
        self.check_delegate_btn.setEnabled(False)
        
        # Start thread
        self.classification_thread.start()
        
    def stop_classification(self):
        """Stop the image classification process"""
        self.classification_thread.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cpu_radio.setEnabled(True)
        self.npu_radio.setEnabled(True)
        self.model_path_input.setEnabled(True)
        self.label_path_input.setEnabled(True)
        self.camera_input.setEnabled(True)
        self.threshold_input.setEnabled(True)
        self.mean_input.setEnabled(True)
        self.std_input.setEnabled(True)
        self.delegate_input.setEnabled(True)
        self.check_delegate_btn.setEnabled(True)
        
        # Clear video display
        self.video_label.clear()
        self.video_label.setText("Video Stopped")
        
    def closeEvent(self, event):
        """Handle window close event"""
        self.classification_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = ImageClassificationGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

