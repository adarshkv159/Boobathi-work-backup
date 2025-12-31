import sys
import os
import time
import numpy as np
import threading
import subprocess
import tempfile
import uuid
import string

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QGroupBox, 
                             QTextEdit, QListWidget, QProgressBar, QTabWidget)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont

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

import whisper
import pyttsx3
from face_detection import YoloFace
from face_recognition import Facenet
from face_database import FaceDatabase

# Configuration Constants
WHISPER_MODEL = "tiny"
AUDIO_DEVICE = "default"
TTS_DEVICE = "default"
COMMAND_DURATION = 3
NAME_DURATION = 4
CONFIRM_DURATION = 3


class FaceRecognitionThread(QThread):
    """Thread for face recognition and voice command processing"""
    change_pixmap_signal = Signal(QImage)
    status_signal = Signal(str, str)  # (status, color)
    message_signal = Signal(str, str)  # (message, type)
    face_detected_signal = Signal(str, float)  # (name, confidence)
    recording_progress_signal = Signal(float)  # progress 0-1
    database_updated_signal = Signal()
    
    def __init__(self, delegate_path=""):
        super().__init__()
        self.running = False
        self.delegate_path = delegate_path
        self.camera_index = 0
        
        # State management
        self.current_operation = "idle"
        self.is_recording = False
        self.is_processing = False
        self.add_state = None
        self.add_embeddings = None
        self.add_name = None
        self.remove_names = None
        self.last_spoken_name = None
        self.voice_command_active = False
        
        # Models
        self.detector = None
        self.recognizer = None
        self.database = None
        self.whisper_model = None
        self.tts_engine = None
        self.tts_lock = threading.Lock()
        
    def initialize_models(self):
        """Initialize all AI models"""
        try:
            self.message_signal.emit("Loading Whisper model...", "info")
            self.whisper_model = whisper.load_model(WHISPER_MODEL)
            
            self.message_signal.emit("Loading face detection model...", "info")
            self.detector = YoloFace("yoloface_int8.tflite", self.delegate_path)
            
            self.message_signal.emit("Loading face recognition model...", "info")
            self.recognizer = Facenet("facenet_512_int_quantized.tflite", self.delegate_path)
            
            self.message_signal.emit("Loading face database...", "info")
            self.database = FaceDatabase()
            
            # Initialize TTS
            try:
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('volume', 1.0)
                self.tts_engine.setProperty('rate', 150)
            except Exception as e:
                self.message_signal.emit(f"TTS init warning: {e}", "warning")
            
            self.message_signal.emit("All models loaded successfully!", "success")
            return True
            
        except Exception as e:
            self.message_signal.emit(f"Model initialization error: {e}", "error")
            return False
    
    def speak(self, text):
        """Text-to-speech output"""
        def tts_worker():
            with self.tts_lock:
                if not self.tts_engine:
                    return
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    tmp_file = tmpfile.name
                try:
                    self.tts_engine.save_to_file(text, tmp_file)
                    self.tts_engine.runAndWait()
                    subprocess.run(f"aplay -D {TTS_DEVICE} {tmp_file}", shell=True,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"TTS error: {e}")
                finally:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)
        
        threading.Thread(target=tts_worker, daemon=True).start()
        self.message_signal.emit(f"🔊 {text}", "tts")
    
    def record_audio(self, filename, duration=4):
        """Record audio with progress updates"""
        try:
            self.is_recording = True
            self.recording_progress_signal.emit(0.0)
            
            cmd = ['arecord', '-D', AUDIO_DEVICE, '-f', 'S16_LE', '-r', '16000',
                   '-c', '1', '-d', str(duration), filename]
            
            process = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            
            start_time = time.time()
            while process.poll() is None:
                elapsed = time.time() - start_time
                progress = min(1.0, elapsed / duration)
                self.recording_progress_signal.emit(progress)
                time.sleep(0.1)
            
            self.recording_progress_signal.emit(1.0)
            
            if process.returncode == 0 and os.path.exists(filename):
                return True
            return False
            
        except Exception as e:
            self.message_signal.emit(f"Recording error: {e}", "error")
            return False
        finally:
            self.is_recording = False
            self.recording_progress_signal.emit(0.0)
    
    def whisper_transcribe(self, audio_file, prompt=""):
        """Transcribe audio with Whisper"""
        try:
            self.is_processing = True
            self.status_signal.emit("Processing speech...", "#FF00FF")
            
            audio_input = whisper.load_audio(audio_file)
            audio_input = whisper.pad_or_trim(audio_input)
            mel = whisper.log_mel_spectrogram(audio_input).to(self.whisper_model.device)
            
            options = whisper.DecodingOptions(language="en", fp16=False, 
                                             prompt=prompt, temperature=0.5)
            result = whisper.decode(self.whisper_model, mel, options)
            
            text = result.text.strip().lower()
            self.message_signal.emit(f"Recognized: '{text}'", "info")
            return text
            
        except Exception as e:
            self.message_signal.emit(f"Transcription error: {e}", "error")
            return ""
        finally:
            self.is_processing = False
            if os.path.exists(audio_file):
                os.remove(audio_file)
    
    def recognize_command(self):
        """Recognize voice command"""
        tmp_file = f"cmd_{uuid.uuid4()}.wav"
        self.message_signal.emit("🎤 Listening for command...", "recording")
        
        if not self.record_audio(tmp_file, COMMAND_DURATION):
            return ""
        
        text = self.whisper_transcribe(tmp_file, "Commands: add, remove, delete, quit, cancel")
        
        if "add" in text or "new" in text:
            return "add"
        elif "remove" in text or "delete" in text:
            return "remove"
        elif "quit" in text or "cancel" in text:
            return "quit"
        
        return ""
    
    def recognize_name(self):
        """Recognize person name"""
        tmp_file = f"name_{uuid.uuid4()}.wav"
        self.message_signal.emit("🎤 Listening for name...", "recording")
        
        if not self.record_audio(tmp_file, NAME_DURATION):
            return None
        
        text = self.whisper_transcribe(tmp_file, "Example names: John, Sarah, David, Emma")
        
        if not text:
            return None
        
        # Clean and validate name
        translator = str.maketrans('', '', string.punctuation)
        cleaned = text.translate(translator).strip()
        words = [w.strip().capitalize() for w in cleaned.split() if w.strip()]
        
        if not words or len(words) > 3:
            return None
        
        name = ' '.join(words)
        if len(name) < 2 or len(name) > 30:
            return None
        
        return name
    
    def recognize_confirmation(self):
        """Recognize yes/no confirmation"""
        tmp_file = f"confirm_{uuid.uuid4()}.wav"
        self.message_signal.emit("🎤 Listening for confirmation...", "recording")
        
        if not self.record_audio(tmp_file, CONFIRM_DURATION):
            return None
        
        text = self.whisper_transcribe(tmp_file, "Responses: yes, no")
        
        if "yes" in text:
            return "yes"
        elif "no" in text:
            return "no"
        
        return None
    
    def process_add_face(self, frame):
        """Handle adding new face"""
        if self.add_state == "capture_face":
            boxes = self.detector.detect(frame)
            
            if boxes is None or len(boxes) == 0:
                return
            
            box = boxes[0]
            box[[0, 2]] *= frame.shape[1]
            box[[1, 3]] *= frame.shape[0]
            x1, y1, x2, y2 = box.astype(np.int32)
            
            # Add margin
            x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
            x2, y2 = min(x2 + 10, frame.shape[1]), min(y2 + 10, frame.shape[0])
            
            face = frame[y1:y2, x1:x2]
            self.add_embeddings = self.recognizer.get_embeddings(face)
            
            if self.add_embeddings is not None:
                self.add_state = "say_name"
                self.speak("Face captured. Please say the name")
                self.message_signal.emit("✓ Face captured successfully", "success")
            else:
                self.speak("Face capture failed. Please try again")
                self.add_state = None
                
        elif self.add_state == "say_name":
            name = self.recognize_name()
            if not name:
                self.speak("Name not recognized. Please try again")
                return
            
            self.add_name = name
            self.add_state = "confirm_name"
            self.speak(f"Did you say {name}? Please say yes or no")
            
        elif self.add_state == "confirm_name":
            confirmation = self.recognize_confirmation()
            
            if confirmation == "yes":
                self.database.add_name(self.add_name, self.add_embeddings)
                self.speak(f"{self.add_name} added successfully")
                self.message_signal.emit(f"✓ Added {self.add_name} to database", "success")
                self.database_updated_signal.emit()
                self.add_state = None
                self.current_operation = "idle"
            elif confirmation == "no":
                self.add_state = "say_name"
                self.speak("Please say the name again")
    
    def handle_voice_command(self):
        """Handle voice command processing"""
        command = self.recognize_command()
        
        if command == "add":
            self.add_state = "capture_face"
            self.current_operation = "adding"
            self.speak("Please show your face to the camera")
            self.message_signal.emit("Starting add person flow...", "info")
            
        elif command == "remove":
            names = self.database.get_names()
            if not names:
                self.speak("No names in database")
                return
            
            self.remove_names = names
            self.current_operation = "removing"
            self.message_signal.emit(f"Found {len(names)} people in database", "info")
            
        elif command == "quit":
            self.add_state = None
            self.current_operation = "idle"
            self.speak("Operation cancelled")
            self.message_signal.emit("Operation cancelled", "info")
    
    def run(self):
        """Main processing loop"""
        if not self.initialize_models():
            return
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.message_signal.emit("Cannot open camera", "error")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.running = True
        self.status_signal.emit("Ready", "#00FF00")
        
        frame_count = 0
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process add face if in progress
            if self.add_state:
                self.process_add_face(frame)
            
            # Face detection and recognition (every 5 frames)
            if frame_count % 5 == 0:
                boxes = self.detector.detect(frame)
                
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    box[[0, 2]] *= frame.shape[1]
                    box[[1, 3]] *= frame.shape[0]
                    x1, y1, x2, y2 = box.astype(np.int32)
                    
                    x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
                    x2, y2 = min(x2 + 10, frame.shape[1]), min(y2 + 10, frame.shape[0])
                    
                    face = frame[y1:y2, x1:x2]
                    embeddings = self.recognizer.get_embeddings(face)
                    name, confidence = self.database.find_name(embeddings)
                    
                    label = f"{name} ({int(confidence*100)}%)" if name else "Unknown"
                    
                    # Draw bounding box and label
                    color = (0, 255, 0) if name else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Emit face detection signal
                    if name:
                        self.face_detected_signal.emit(name, confidence)
                        
                        # Greet if new person
                        if name != self.last_spoken_name:
                            self.speak(f"Hello {name}")
                            self.last_spoken_name = name
            
            # Add status overlay
            if self.current_operation != "idle":
                cv2.putText(frame, f"Operation: {self.current_operation}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Convert and emit frame
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.change_pixmap_signal.emit(qt_image)
            
            # Handle voice command if triggered
            if self.voice_command_active:
                self.voice_command_active = False
                self.handle_voice_command()
        
        cap.release()
    
    def trigger_voice_command(self):
        """Trigger voice command processing"""
        if not self.is_recording and not self.is_processing:
            self.voice_command_active = True
    
    def stop(self):
        """Stop the thread"""
        self.running = False
        self.wait()


class FaceRecognitionGUI(QMainWindow):
    """Main GUI for Face Recognition with Whisper ASR"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition with Whisper ASR (i.MX8M Plus)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize thread
        self.recognition_thread = FaceRecognitionThread()
        self.recognition_thread.change_pixmap_signal.connect(self.update_image)
        self.recognition_thread.status_signal.connect(self.update_status)
        self.recognition_thread.message_signal.connect(self.add_message)
        self.recognition_thread.face_detected_signal.connect(self.on_face_detected)
        self.recognition_thread.recording_progress_signal.connect(self.update_recording_progress)
        self.recognition_thread.database_updated_signal.connect(self.update_database_list)
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel - Video
        video_panel = self.create_video_panel()
        main_layout.addWidget(video_panel, 2)
        
        # Right panel - Controls and Info
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 1)
    
    def create_video_panel(self):
        """Create video display panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        title = QLabel("Live Camera Feed")
        title.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; }")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("QLabel { background-color: black; border: 2px solid #555; }")
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setScaledContents(False)
        
        layout.addWidget(self.video_label)
        
        return panel
    
    def create_right_panel(self):
        """Create right control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Status display
        status_group = QGroupBox("System Status")
        status_layout = QVBoxLayout()
        
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("QLabel { font-size: 14px; font-weight: bold; padding: 10px; background-color: #333; color: #FFF; border-radius: 5px; }")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        # Recording progress
        self.recording_progress = QProgressBar()
        self.recording_progress.setVisible(False)
        self.recording_progress.setStyleSheet("QProgressBar { border: 2px solid grey; border-radius: 5px; text-align: center; } QProgressBar::chunk { background-color: #00FFFF; }")
        status_layout.addWidget(self.recording_progress)
        
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)
        
        # Control buttons
        control_group = QGroupBox("Voice Controls")
        control_layout = QVBoxLayout()
        
        self.voice_cmd_button = QPushButton("🎤 Press to Speak Command")
        self.voice_cmd_button.clicked.connect(self.on_voice_command)
        self.voice_cmd_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 15px; font-size: 14px; font-weight: bold; }")
        control_layout.addWidget(self.voice_cmd_button)
        
        help_label = QLabel("Voice Commands:\n• Add - Add new person\n• Remove - Remove person\n• Quit/Cancel - Cancel operation")
        help_label.setStyleSheet("QLabel { font-size: 11px; padding: 10px; background-color: #2b2b2b; color: #AAA; border-radius: 3px; }")
        control_layout.addWidget(help_label)
        
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)
        
        # Tabs for messages and database
        tabs = QTabWidget()
        
        # Messages tab
        messages_widget = QWidget()
        messages_layout = QVBoxLayout()
        messages_widget.setLayout(messages_layout)
        
        self.message_log = QTextEdit()
        self.message_log.setReadOnly(True)
        self.message_log.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #00ff00; font-family: monospace; font-size: 11px; }")
        messages_layout.addWidget(self.message_log)
        
        tabs.addTab(messages_widget, "📝 Messages")
        
        # Database tab
        database_widget = QWidget()
        database_layout = QVBoxLayout()
        database_widget.setLayout(database_layout)
        
        self.database_list = QListWidget()
        self.database_list.setStyleSheet("QListWidget { background-color: #2b2b2b; color: #00ff00; font-family: monospace; font-size: 12px; }")
        database_layout.addWidget(self.database_list)
        
        tabs.addTab(database_widget, "👤 Database")
        
        layout.addWidget(tabs, 1)
        
        # System control buttons
        sys_control_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start System")
        self.start_button.clicked.connect(self.start_system)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 13px; font-weight: bold; }")
        sys_control_layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop System")
        self.stop_button.clicked.connect(self.stop_system)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; font-size: 13px; font-weight: bold; }")
        sys_control_layout.addWidget(self.stop_button)
        
        layout.addLayout(sys_control_layout)
        
        return panel
    
    @Slot(QImage)
    def update_image(self, qt_image):
        scaled_pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
    
    @Slot(str, str)
    def update_status(self, status, color):
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"QLabel {{ font-size: 14px; font-weight: bold; padding: 10px; background-color: {color}; color: #FFF; border-radius: 5px; }}")
    
    @Slot(str, str)
    def add_message(self, message, msg_type):
        timestamp = time.strftime("%H:%M:%S")
        
        if msg_type == "error":
            color = "#FF0000"
            prefix = "❌"
        elif msg_type == "success":
            color = "#00FF00"
            prefix = "✓"
        elif msg_type == "warning":
            color = "#FFA500"
            prefix = "⚠"
        elif msg_type == "recording":
            color = "#00FFFF"
            prefix = "🎤"
        elif msg_type == "tts":
            color = "#FF00FF"
            prefix = "🔊"
        else:
            color = "#FFFFFF"
            prefix = "ℹ"
        
        self.message_log.append(f'<span style="color: {color};">[{timestamp}] {prefix} {message}</span>')
    
    @Slot(str, float)
    def on_face_detected(self, name, confidence):
        # Could add face detection history here
        pass
    
    @Slot(float)
    def update_recording_progress(self, progress):
        if progress > 0:
            self.recording_progress.setVisible(True)
            self.recording_progress.setValue(int(progress * 100))
        else:
            self.recording_progress.setVisible(False)
    
    @Slot()
    def update_database_list(self):
        self.database_list.clear()
        if self.recognition_thread.database:
            names = self.recognition_thread.database.get_names()
            for i, name in enumerate(names, 1):
                self.database_list.addItem(f"{i}. {name}")
    
    def on_voice_command(self):
        """Trigger voice command"""
        if not self.recognition_thread.is_recording and not self.recognition_thread.is_processing:
            self.recognition_thread.trigger_voice_command()
    
    def start_system(self):
        self.add_message("Starting face recognition system...", "info")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.voice_cmd_button.setEnabled(True)
        self.recognition_thread.start()
        self.update_database_list()
    
    def stop_system(self):
        self.recognition_thread.stop()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.voice_cmd_button.setEnabled(False)
        self.video_label.clear()
        self.video_label.setText("System Stopped")
        self.add_message("System stopped", "info")
    
    def closeEvent(self, event):
        self.recognition_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

