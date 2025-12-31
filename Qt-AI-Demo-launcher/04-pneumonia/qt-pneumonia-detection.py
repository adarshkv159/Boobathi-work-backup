import sys
import os
import time
import numpy as np

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QGroupBox, QRadioButton, QTextEdit)
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


class PneumoniaDetectionThread(QThread):
    """Thread for video capture and pneumonia detection"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    results_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.model_path = "trained.tflite"
        self.camera_index = 0
        self.delegate_path = ""
        self.use_npu = False
        self.interpreter = None
        self.vid = None
        self.labels = ["bacteria", "normal", "virus"]
        
    def set_model_path(self, path):
        """Set the model path"""
        self.model_path = path
        
    def set_camera(self, camera_input):
        """Set camera index or video file path"""
        if str(camera_input).isdigit():
            self.camera_index = int(camera_input)
        else:
            self.camera_index = camera_input
            
    def set_delegate(self, use_npu, delegate_path=""):
        """Set NPU delegate configuration"""
        self.use_npu = use_npu
        self.delegate_path = delegate_path
    
    def preprocess_frame(self, frame, input_shape):
        """Preprocess the frame for model input"""
        frame = cv2.resize(frame, (input_shape[2], input_shape[1]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.expand_dims(frame, axis=0)
        frame = frame.astype(self.input_details[0]['dtype'])
        return frame
    
    def dequantize_output(self, quantized_values, scale, zero_point):
        """Dequantize the output values"""
        return [(val - zero_point) * scale for val in quantized_values]
    
    def interpret_prediction(self, output_data, labels, scale, zero_point):
        """Interpret the prediction from model output"""
        dequantized_values = self.dequantize_output(output_data[0], scale, zero_point)
        max_index = np.argmax(dequantized_values)
        predicted_class = labels[max_index]
        confidence = dequantized_values[max_index]
        return predicted_class, confidence, dequantized_values
        
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
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Get quantization parameters
            self.scale = self.output_details[0]['quantization'][0]
            self.zero_point = self.output_details[0]['quantization'][1]
            
            # Log model info
            self.log_signal.emit(f"\n=== Model Information ===")
            self.log_signal.emit(f"Input shape: {self.input_details[0]['shape']}")
            self.log_signal.emit(f"Input dtype: {self.input_details[0]['dtype']}")
            self.log_signal.emit(f"Output shape: {self.output_details[0]['shape']}")
            self.log_signal.emit(f"Quantization scale: {self.scale}")
            self.log_signal.emit(f"Quantization zero_point: {self.zero_point}")
            self.log_signal.emit(f"Classes: {', '.join(self.labels)}")
            
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
            
        # Initialize video capture
        self.vid = cv2.VideoCapture(self.camera_index)
        if not self.vid.isOpened():
            error_msg = f"Error: Could not open video source {self.camera_index}"
            print(error_msg)
            self.log_signal.emit(error_msg)
            return
        
        self.log_signal.emit(f"\n✓ Video capture initialized: {self.camera_index}")
        self.log_signal.emit("Starting pneumonia detection loop...\n")
            
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
            input_data = self.preprocess_frame(frame, self.input_details[0]['shape'])
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
            
            # Get prediction
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            predicted_class, confidence, dequantized_values = self.interpret_prediction(
                output_data, self.labels, self.scale, self.zero_point
            )
            
            # Calculate performance metrics
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_frames / total_time) if total_time > 0 else 0
            invoke_time = (invoke_end - invoke_start) * 1000
            
            # Status message
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {fps} | Inference: {invoke_time:.2f}ms | Mode: {mode}"
            
            # Draw prediction on frame
            result_text = f"Prediction: {predicted_class} ({confidence:.2f})"
            
            # Color coding based on prediction
            if predicted_class == "normal":
                color = (0, 255, 0)  # Green for normal
            elif predicted_class == "bacteria":
                color = (0, 165, 255)  # Orange for bacteria
            else:  # virus
                color = (0, 0, 255)  # Red for virus
            
            cv2.putText(frame, result_text, (10, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            # Draw status
            cv2.putText(frame, msg, (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Format detailed results
            results_text = f"=== Prediction Results ===\n\n"
            results_text += f"Primary: {predicted_class.upper()}\n"
            results_text += f"Confidence: {confidence:.4f}\n\n"
            results_text += "All Class Scores:\n"
            for i, label in enumerate(self.labels):
                results_text += f"  {label}: {dequantized_values[i]:.4f}\n"
            
            # Convert frame to QImage for display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            
            # Emit signals
            self.change_pixmap_signal.emit(qt_image)
            self.fps_signal.emit(msg)
            self.results_signal.emit(results_text)
            
        # Cleanup
        if self.vid:
            self.vid.release()
        self.log_signal.emit("\n✓ Video capture stopped")
            
    def stop(self):
        """Stop the detection thread"""
        self.running = False
        self.wait()


class PneumoniaDetectionGUI(QMainWindow):
    """Main GUI window for pneumonia detection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pneumonia Detection - AI Diagnosis (i.MX8M Plus)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize detection thread
        self.detection_thread = PneumoniaDetectionThread()
        self.detection_thread.change_pixmap_signal.connect(self.update_image)
        self.detection_thread.fps_signal.connect(self.update_fps)
        self.detection_thread.log_signal.connect(self.update_log)
        self.detection_thread.results_signal.connect(self.update_results)
        
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
        self.model_path_input = QLineEdit("trained.tflite")
        model_layout.addWidget(model_path_label)
        model_layout.addWidget(self.model_path_input)
        
        # Camera input
        camera_label = QLabel("Camera/Video Source:")
        self.camera_input = QLineEdit("0")
        model_layout.addWidget(camera_label)
        model_layout.addWidget(self.camera_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Info label
        info_group = QGroupBox("Detection Classes")
        info_layout = QVBoxLayout()
        
        bacteria_label = QLabel("🟠 Bacteria: Bacterial pneumonia")
        bacteria_label.setStyleSheet("QLabel { color: #FFA500; font-size: 12px; }")
        normal_label = QLabel("🟢 Normal: Healthy lungs")
        normal_label.setStyleSheet("QLabel { color: #00FF00; font-size: 12px; }")
        virus_label = QLabel("🔴 Virus: Viral pneumonia")
        virus_label.setStyleSheet("QLabel { color: #FF0000; font-size: 12px; }")
        
        info_layout.addWidget(bacteria_label)
        info_layout.addWidget(normal_label)
        info_layout.addWidget(virus_label)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
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
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")
        
        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.clicked.connect(self.stop_detection)
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
        
        # Title
        title = QLabel("Live Camera Feed")
        title.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; }")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
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
        results_group = QGroupBox("Detection Results")
        layout = QVBoxLayout()
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("QTextEdit { background-color: #2b2b2b; color: #00ff00; font-family: monospace; font-size: 12px; }")
        self.results_text.setPlaceholderText("Results will appear here...")
        
        # Disclaimer
        disclaimer = QLabel("⚠ For research purposes only.\nNot for clinical diagnosis.")
        disclaimer.setStyleSheet("QLabel { color: #FFA500; font-size: 10px; font-style: italic; padding: 5px; }")
        disclaimer.setWordWrap(True)
        
        layout.addWidget(self.results_text)
        layout.addWidget(disclaimer)
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
        
    def start_detection(self):
        """Start the pneumonia detection process"""
        # Clear displays
        self.log_text.clear()
        self.results_text.clear()
        self.log_text.append("=== Starting Pneumonia Detection ===\n")
        
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
        
        # Configure detection thread
        self.detection_thread.set_model_path(model_path)
        self.detection_thread.set_camera(camera_input)
        self.detection_thread.set_delegate(use_npu, delegate_path)
        
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
        self.detection_thread.start()
        
    def stop_detection(self):
        """Stop the pneumonia detection process"""
        self.detection_thread.stop()
        
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
        self.detection_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = PneumoniaDetectionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

