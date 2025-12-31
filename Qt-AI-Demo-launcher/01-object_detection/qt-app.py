# Copyright 2020-2022 NXP
# SPDX-License-Identifier: Apache-2.0

import sys
import os
import time
import numpy as np

# Import PyQt5 FIRST
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QComboBox, 
                             QLineEdit, QGroupBox, QRadioButton)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

# Import cv2 AFTER PyQt5
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
    change_pixmap_signal = pyqtSignal(QImage)
    fps_signal = pyqtSignal(str)
    
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
                ext_delegate = [tflite.load_delegate(self.delegate_path)]
                self.interpreter = tflite.Interpreter(
                    model_path=self.model_path, 
                    experimental_delegates=ext_delegate
                )
            else:
                self.interpreter = tflite.Interpreter(model_path=self.model_path)
            
            self.interpreter.allocate_tensors()
            return True
        except Exception as e:
            print(f"Error initializing model: {e}")
            return False
            
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
            print(f"Error: Could not open video source {self.camera_index}")
            return
            
        # Performance tracking
        total_fps = 0
        total_time = 0
        self.running = True
        
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
            
            # Get results
            boxes = self.interpreter.get_tensor(output_details[0]['index'])[0]
            labels = self.interpreter.get_tensor(output_details[1]['index'])[0]
            scores = self.interpreter.get_tensor(output_details[2]['index'])[0]
            number = self.interpreter.get_tensor(output_details[3]['index'])[0]
            
            # Draw detections
            for i in range(int(number)):
                if scores[i] > 0.5:  # Confidence threshold
                    box = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]]
                    x0 = max(2, int(box[1] * frame.shape[1]))
                    y0 = max(2, int(box[0] * frame.shape[0]))
                    x1 = int(box[3] * frame.shape[1])
                    y1 = int(box[2] * frame.shape[0])
                    
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    cv2.putText(frame, label2string[labels[i]], (x0, y0 + 13),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # Calculate performance metrics
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_fps / total_time) if total_time > 0 else 0
            invoke_time = int((invoke_end - invoke_start) * 1000)
            
            msg = f"FPS: {fps} | Inference: {invoke_time}ms | Mode: {'NPU' if self.use_npu else 'CPU'}"
            cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 0), 2)
            
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
            
    def stop(self):
        """Stop the video thread"""
        self.running = False
        self.wait()


class ObjectDetectionGUI(QMainWindow):
    """Main GUI window for object detection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Object Detection - SSD MobileNet")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize video thread
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.fps_signal.connect(self.update_fps)
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        main_layout.addWidget(control_panel, 1)
        
        # Right panel - Video display
        video_panel = self.create_video_panel()
        main_layout.addWidget(video_panel, 3)
        
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
        self.camera_input = QLineEdit("1")
        model_layout.addWidget(camera_label)
        model_layout.addWidget(self.camera_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Acceleration settings group
        accel_group = QGroupBox("Acceleration")
        accel_layout = QVBoxLayout()
        
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setChecked(True)
        self.npu_radio = QRadioButton("NPU")
        
        accel_layout.addWidget(self.cpu_radio)
        accel_layout.addWidget(self.npu_radio)
        
        # NPU delegate path
        delegate_label = QLabel("NPU Delegate Path:")
        self.delegate_input = QLineEdit("")
        self.delegate_input.setPlaceholderText("/usr/lib/libvx_delegate.so")
        accel_layout.addWidget(delegate_label)
        accel_layout.addWidget(self.delegate_input)
        
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
        self.fps_label.setStyleSheet("QLabel { font-size: 12px; font-weight: bold; padding: 10px; }")
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
        
    @pyqtSlot(QImage)
    def update_image(self, qt_image):
        """Update the video display with new frame"""
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
    @pyqtSlot(str)
    def update_fps(self, msg):
        """Update FPS and inference time display"""
        self.fps_label.setText(msg)
        
    def start_detection(self):
        """Start the object detection process"""
        # Get settings from UI
        model_path = self.model_path_input.text()
        camera_input = self.camera_input.text()
        use_npu = self.npu_radio.isChecked()
        delegate_path = self.delegate_input.text()
        
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
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()

