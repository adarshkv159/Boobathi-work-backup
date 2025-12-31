import sys
import os
import time
import numpy as np
import re
import threading

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QGroupBox, QRadioButton, QTextEdit, QCheckBox,
                             QListWidget, QDoubleSpinBox)
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


class LicensePlateThread(QThread):
    """Thread for license plate detection and OCR"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    plate_detected_signal = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.detection_model_path = "quant_model_NPU_3k.tflite"
        self.ocr_model_path = "license_plate_character_recognition.tflite"
        self.label_path = "labelmap.txt"
        self.video_path = "demo.webm"
        self.delegate_path = "libvx_delegate.so"
        self.use_npu = False
        self.enable_audio = True
        self.confidence_threshold = 0.5
        self.stability_time = 0.5
        
        self.detection_interpreter = None
        self.ocr_interpreter = None
        self.cap = None
        self.labels = []
        
        self.last_text = ""
        self.spoken_plates = set()
        self.last_stable_text = ""
        self.last_stable_time = time.time()
        self.tts_lock = threading.Lock()
        
        # Character map for OCR
        self.char_map = {i: c for i, c in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")}
        
    def set_detection_model(self, path):
        self.detection_model_path = path
    
    def set_ocr_model(self, path):
        self.ocr_model_path = path
    
    def set_label_path(self, path):
        self.label_path = path
    
    def set_video_source(self, source):
        self.video_path = source
    
    def set_delegate(self, use_npu, delegate_path=""):
        self.use_npu = use_npu
        if delegate_path:
            self.delegate_path = delegate_path
    
    def set_audio_enabled(self, enabled):
        self.enable_audio = enabled
    
    def set_confidence_threshold(self, threshold):
        self.confidence_threshold = threshold
    
    def set_stability_time(self, time_sec):
        self.stability_time = time_sec
    
    def predict_character(self, img):
        """Predict single character using OCR model"""
        img = img.astype(np.float32) / 255.0
        input_tensor = np.expand_dims(img, axis=0)
        self.ocr_interpreter.set_tensor(self.ocr_input[0]['index'], input_tensor)
        self.ocr_interpreter.invoke()
        output = self.ocr_interpreter.get_tensor(self.ocr_output[0]['index'])
        return np.argmax(output)
    
    def recognize_plate(self, chars):
        """Recognize full plate from character images"""
        text = ''
        for ch in chars:
            ch_rgb = cv2.cvtColor(ch, cv2.COLOR_GRAY2RGB)
            resized = cv2.resize(ch_rgb, (28, 28))
            label = self.predict_character(resized)
            text += self.char_map[label]
        return text
    
    def segment_characters(self, plate_img):
        """Segment individual characters from plate image"""
        plate_img = cv2.resize(plate_img, (333, 75))
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = 255 - thresh
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        char_regions = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if 15 < h < 70 and 5 < w < 60:
                char = thresh[y:y+h, x:x+w]
                padded = np.full((28, 28), 0, dtype=np.uint8)
                char_resized = cv2.resize(char, (20, 20))
                padded[4:24, 4:24] = char_resized
                char_regions.append((x, padded))
        
        char_regions = sorted(char_regions, key=lambda tup: tup[0])
        return [img for _, img in char_regions]
    
    def speak_plate(self, text):
        """Text-to-speech for detected plate"""
        def speak():
            with self.tts_lock:
                audio_path = "/tmp/plate.wav"
                os.system(f'espeak -w {audio_path} "{text}"')
                os.system(f'aplay -D plughw:3,0 {audio_path} > /dev/null 2>&1')
        
        if self.enable_audio:
            threading.Thread(target=speak, daemon=True).start()
    
    def initialize_models(self):
        """Initialize both detection and OCR models"""
        try:
            # Load labels
            if not os.path.exists(self.label_path):
                error_msg = f"ERROR: Label file not found: {self.label_path}"
                self.log_signal.emit(error_msg)
                return False
            
            with open(self.label_path, 'r') as f:
                self.labels = [line.strip() for line in f.readlines()]
            
            self.log_signal.emit(f"✓ Loaded {len(self.labels)} labels")
            
            # Initialize OCR model
            self.log_signal.emit("\n=== Loading OCR Model ===")
            try:
                if self.use_npu and os.path.exists(self.delegate_path):
                    self.ocr_interpreter = tflite.Interpreter(
                        model_path=self.ocr_model_path,
                        experimental_delegates=[tflite.load_delegate(self.delegate_path)]
                    )
                    self.log_signal.emit("✓ OCR model loaded on NPU")
                else:
                    self.ocr_interpreter = tflite.Interpreter(model_path=self.ocr_model_path)
                    self.log_signal.emit("✓ OCR model loaded on CPU")
                
                self.ocr_interpreter.allocate_tensors()
                self.ocr_input = self.ocr_interpreter.get_input_details()
                self.ocr_output = self.ocr_interpreter.get_output_details()
                
            except Exception as e:
                error_msg = f"ERROR loading OCR model: {e}"
                self.log_signal.emit(error_msg)
                return False
            
            # Initialize Detection model
            self.log_signal.emit("\n=== Loading Detection Model ===")
            try:
                if self.use_npu and os.path.exists(self.delegate_path):
                    self.detection_interpreter = tflite.Interpreter(
                        model_path=self.detection_model_path,
                        experimental_delegates=[tflite.load_delegate(self.delegate_path)]
                    )
                    self.log_signal.emit("✓ Detection model loaded on NPU")
                else:
                    self.detection_interpreter = tflite.Interpreter(model_path=self.detection_model_path)
                    self.log_signal.emit("✓ Detection model loaded on CPU")
                
                self.detection_interpreter.allocate_tensors()
                self.input_details = self.detection_interpreter.get_input_details()
                self.output_details = self.detection_interpreter.get_output_details()
                
                self.height = self.input_details[0]['shape'][1]
                self.width = self.input_details[0]['shape'][2]
                self.float_input = (self.input_details[0]['dtype'] == np.float32)
                
                self.log_signal.emit(f"Input shape: {self.input_details[0]['shape']}")
                self.log_signal.emit(f"Float input: {self.float_input}")
                
            except Exception as e:
                error_msg = f"ERROR loading detection model: {e}"
                self.log_signal.emit(error_msg)
                return False
            
            if self.use_npu:
                self.log_signal.emit(f"\n✓ NPU mode ENABLED")
            else:
                self.log_signal.emit(f"\n✓ CPU mode ENABLED")
            
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
        if self.video_path.isdigit():
            self.cap = cv2.VideoCapture(int(self.video_path))
        else:
            self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            error_msg = f"Error: Could not open video source {self.video_path}"
            self.log_signal.emit(error_msg)
            return
        
        self.log_signal.emit(f"\n✓ Video source initialized: {self.video_path}")
        self.log_signal.emit("Starting license plate detection...\n")
        
        # Performance tracking
        total_frames = 0
        total_time = 0
        self.running = True
        first_inference = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Loop video
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            total_frames += 1
            loop_start = time.time()
            
            # Preprocess for detection
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imH, imW, _ = frame.shape
            image_resized = cv2.resize(image_rgb, (self.width, self.height))
            input_data = np.expand_dims(image_resized, axis=0)
            
            if self.float_input:
                input_data = (np.float32(input_data) - 127.5) / 127.5
            else:
                input_data = input_data.astype(np.uint8)
            
            # Run detection
            self.detection_interpreter.set_tensor(self.input_details[0]['index'], input_data)
            
            invoke_start = time.time()
            self.detection_interpreter.invoke()
            invoke_end = time.time()
            
            if first_inference:
                warmup_time = int((invoke_end - invoke_start) * 1000)
                self.log_signal.emit(f"First inference time: {warmup_time}ms (includes warmup)")
                first_inference = False
            
            # Get detection results
            boxes = self.detection_interpreter.get_tensor(self.output_details[1]['index'])[0]
            classes = self.detection_interpreter.get_tensor(self.output_details[3]['index'])[0]
            scores = self.detection_interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            detection_count = 0
            
            # Process detections
            for i in range(len(scores)):
                if scores[i] > self.confidence_threshold:
                    detection_count += 1
                    ymin = int(max(1, boxes[i][0] * imH))
                    xmin = int(max(1, boxes[i][1] * imW))
                    ymax = int(min(imH, boxes[i][2] * imH))
                    xmax = int(min(imW, boxes[i][3] * imW))
                    
                    # Crop plate region
                    plate_crop = frame[ymin:ymax, xmin:xmax]
                    
                    # OCR recognition
                    chars = self.segment_characters(plate_crop)
                    plate_text = self.recognize_plate(chars) if chars else ""
                    
                    # Validate plate format
                    if re.fullmatch(r'[A-Z]{1}\d{3}[A-Z]{2}', plate_text) or \
                       re.fullmatch(r'\d{2}[A-Z]{2}\d{2}', plate_text) or \
                       re.fullmatch(r'[A-Z]{2}\d{2}[A-Z]{2}\d{4}', plate_text):
                        
                        current_time = time.time()
                        
                        # Stability check
                        if plate_text == self.last_stable_text:
                            if (current_time - self.last_stable_time) >= self.stability_time and \
                               plate_text not in self.spoken_plates:
                                self.log_signal.emit(f"✓ Detected plate: {plate_text}")
                                self.spoken_plates.add(plate_text)
                                self.plate_detected_signal.emit(plate_text)
                                self.speak_plate(plate_text)
                        else:
                            self.last_stable_text = plate_text
                            self.last_stable_time = current_time
                        
                        self.last_text = plate_text
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    
                    # Draw label
                    label = f"{self.labels[int(classes[i])]}: {int(scores[i]*100)}% {plate_text}"
                    cv2.putText(frame, label, (xmin, ymin - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw last detected plate in corner
            if self.last_text:
                text_size, _ = cv2.getTextSize(self.last_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
                text_w, text_h = text_size
                cv2.rectangle(frame, (imW - text_w - 20, 10), 
                            (imW - 10, 10 + text_h + 10), (0, 0, 0), -1)
                cv2.putText(frame, self.last_text, (imW - text_w - 15, 10 + text_h),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            
            # Calculate performance metrics
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_frames / total_time) if total_time > 0 else 0
            invoke_time = (invoke_end - invoke_start) * 1000
            
            # Status message
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {fps} | Inference: {invoke_time:.2f}ms | Mode: {mode} | Plates: {len(self.spoken_plates)}"
            
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
        if self.cap:
            self.cap.release()
        self.log_signal.emit("\n✓ Video processing stopped")
    
    def stop(self):
        """Stop the detection thread"""
        self.running = False
        self.wait()


class LicensePlateGUI(QMainWindow):
    """Main GUI window for license plate detection"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("License Plate Detection & OCR (i.MX8M Plus)")
        self.setGeometry(100, 100, 1500, 900)
        
        # Initialize detection thread
        self.detection_thread = LicensePlateThread()
        self.detection_thread.change_pixmap_signal.connect(self.update_image)
        self.detection_thread.fps_signal.connect(self.update_fps)
        self.detection_thread.log_signal.connect(self.update_log)
        self.detection_thread.plate_detected_signal.connect(self.add_detected_plate)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Top section
        top_layout = QHBoxLayout()
        
        # Left panel - Controls
        control_panel = self.create_control_panel()
        top_layout.addWidget(control_panel, 1)
        
        # Middle panel - Video display
        video_panel = self.create_video_panel()
        top_layout.addWidget(video_panel, 2)
        
        # Right panel - Detected plates
        plates_panel = self.create_plates_panel()
        top_layout.addWidget(plates_panel, 1)
        
        main_layout.addLayout(top_layout, 3)
        
        # Bottom section - Log
        log_panel = self.create_log_panel()
        main_layout.addWidget(log_panel, 1)
    
    def create_control_panel(self):
        """Create the control panel with settings"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Model settings
        model_group = QGroupBox("Model Settings")
        model_layout = QVBoxLayout()
        
        det_label = QLabel("Detection Model:")
        self.detection_model_input = QLineEdit("quant_model_NPU_3k.tflite")
        model_layout.addWidget(det_label)
        model_layout.addWidget(self.detection_model_input)
        
        ocr_label = QLabel("OCR Model:")
        self.ocr_model_input = QLineEdit("license_plate_character_recognition.tflite")
        model_layout.addWidget(ocr_label)
        model_layout.addWidget(self.ocr_model_input)
        
        label_label = QLabel("Labels File:")
        self.label_input = QLineEdit("labelmap.txt")
        model_layout.addWidget(label_label)
        model_layout.addWidget(self.label_input)
        
        video_label = QLabel("Video Source:")
        self.video_input = QLineEdit("demo.webm")
        model_layout.addWidget(video_label)
        model_layout.addWidget(self.video_input)
        
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Detection parameters
        param_group = QGroupBox("Detection Parameters")
        param_layout = QVBoxLayout()
        
        conf_label = QLabel("Confidence Threshold:")
        self.conf_spinbox = QDoubleSpinBox()
        self.conf_spinbox.setRange(0.1, 1.0)
        self.conf_spinbox.setSingleStep(0.05)
        self.conf_spinbox.setValue(0.5)
        param_layout.addWidget(conf_label)
        param_layout.addWidget(self.conf_spinbox)
        
        stab_label = QLabel("Stability Time (sec):")
        self.stab_spinbox = QDoubleSpinBox()
        self.stab_spinbox.setRange(0.1, 2.0)
        self.stab_spinbox.setSingleStep(0.1)
        self.stab_spinbox.setValue(0.5)
        param_layout.addWidget(stab_label)
        param_layout.addWidget(self.stab_spinbox)
        
        self.audio_checkbox = QCheckBox("Enable Audio Feedback")
        self.audio_checkbox.setChecked(True)
        param_layout.addWidget(self.audio_checkbox)
        
        param_group.setLayout(param_layout)
        layout.addWidget(param_group)
        
        # Acceleration
        accel_group = QGroupBox("Acceleration")
        accel_layout = QVBoxLayout()
        
        self.cpu_radio = QRadioButton("CPU")
        self.cpu_radio.setChecked(True)
        self.npu_radio = QRadioButton("NPU (VX Delegate)")
        
        accel_layout.addWidget(self.cpu_radio)
        accel_layout.addWidget(self.npu_radio)
        
        delegate_label = QLabel("Delegate Path:")
        self.delegate_input = QLineEdit("libvx_delegate.so")
        accel_layout.addWidget(delegate_label)
        accel_layout.addWidget(self.delegate_input)
        
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
        
        title = QLabel("Live Detection Feed")
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
    
    def create_plates_panel(self):
        """Create the detected plates panel"""
        plates_group = QGroupBox("Detected License Plates")
        layout = QVBoxLayout()
        
        self.plates_list = QListWidget()
        self.plates_list.setStyleSheet("QListWidget { background-color: #2b2b2b; color: #00ff00; font-family: monospace; font-size: 13px; }")
        
        clear_btn = QPushButton("Clear List")
        clear_btn.clicked.connect(self.plates_list.clear)
        clear_btn.setStyleSheet("QPushButton { padding: 5px; }")
        
        layout.addWidget(self.plates_list)
        layout.addWidget(clear_btn)
        plates_group.setLayout(layout)
        
        return plates_group
    
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
        """Update the video display"""
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
        self.log_text.append(msg)
    
    @Slot(str)
    def add_detected_plate(self, plate):
        """Add detected plate to list"""
        timestamp = time.strftime("%H:%M:%S")
        self.plates_list.addItem(f"[{timestamp}] {plate}")
    
    def start_detection(self):
        """Start license plate detection"""
        self.log_text.clear()
        self.log_text.append("=== Starting License Plate Detection ===\n")
        
        # Get settings
        det_model = self.detection_model_input.text()
        ocr_model = self.ocr_model_input.text()
        label_path = self.label_input.text()
        video_source = self.video_input.text()
        use_npu = self.npu_radio.isChecked()
        delegate_path = self.delegate_input.text()
        conf_threshold = self.conf_spinbox.value()
        stab_time = self.stab_spinbox.value()
        audio_enabled = self.audio_checkbox.isChecked()
        
        # Validate
        if not os.path.exists(det_model):
            self.log_text.append(f"ERROR: Detection model not found: {det_model}")
            return
        
        if not os.path.exists(ocr_model):
            self.log_text.append(f"ERROR: OCR model not found: {ocr_model}")
            return
        
        # Configure thread
        self.detection_thread.set_detection_model(det_model)
        self.detection_thread.set_ocr_model(ocr_model)
        self.detection_thread.set_label_path(label_path)
        self.detection_thread.set_video_source(video_source)
        self.detection_thread.set_delegate(use_npu, delegate_path)
        self.detection_thread.set_confidence_threshold(conf_threshold)
        self.detection_thread.set_stability_time(stab_time)
        self.detection_thread.set_audio_enabled(audio_enabled)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cpu_radio.setEnabled(False)
        self.npu_radio.setEnabled(False)
        self.detection_model_input.setEnabled(False)
        self.ocr_model_input.setEnabled(False)
        self.label_input.setEnabled(False)
        self.video_input.setEnabled(False)
        self.delegate_input.setEnabled(False)
        self.conf_spinbox.setEnabled(False)
        self.stab_spinbox.setEnabled(False)
        
        # Start
        self.detection_thread.start()
    
    def stop_detection(self):
        """Stop detection"""
        self.detection_thread.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cpu_radio.setEnabled(True)
        self.npu_radio.setEnabled(True)
        self.detection_model_input.setEnabled(True)
        self.ocr_model_input.setEnabled(True)
        self.label_input.setEnabled(True)
        self.video_input.setEnabled(True)
        self.delegate_input.setEnabled(True)
        self.conf_spinbox.setEnabled(True)
        self.stab_spinbox.setEnabled(True)
        
        self.video_label.clear()
        self.video_label.setText("Video Stopped")
    
    def closeEvent(self, event):
        """Handle window close"""
        self.detection_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = LicensePlateGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

