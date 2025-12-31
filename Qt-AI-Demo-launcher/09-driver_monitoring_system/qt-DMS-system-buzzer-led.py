import sys
import os
import time
import pathlib
import numpy as np
import threading

# Import PySide6 FIRST
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                QHBoxLayout, QPushButton, QLabel, QGroupBox,
                                QTextEdit, QRadioButton, QLineEdit, QProgressBar,
                                QCheckBox, QSpinBox, QFormLayout, QScrollArea)
from PySide6.QtCore import Qt, QThread, Signal, Slot, QTimer
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

# Import GPIO for buzzer control
try:
    import gpiod
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("Warning: gpiod not available. Buzzer features disabled.")

from face_detection import FaceDetector
from eye_landmark import EyeMesher
from face_landmark import FaceMesher
from utils import *

# Model paths
MODEL_PATH = pathlib.Path(".")
DETECT_MODEL = "face_detection_front_128_full_integer_quant.tflite"
LANDMARK_MODEL = "face_landmark_192_full_integer_quant.tflite"
EYE_MODEL = "iris_landmark_quant.tflite"

# LED paths
LED_RED_PATH = "/sys/class/leds/led-1/brightness"
LED_GREEN_PATH = "/sys/class/leds/led-2/brightness"
LED_BLUE_PATH = "/sys/class/leds/led-3/brightness"

# Audio path for TTS
AUDIO_PATH = "/tmp/alert.wav"


class DriverMonitorThread(QThread):
    """Thread for driver monitoring system with buzzer and LED control"""
    change_pixmap_signal = Signal(QImage)
    fps_signal = Signal(str)
    log_signal = Signal(str)
    status_signal = Signal(dict)  # Dictionary with all status info
    alert_signal = Signal(str, str)  # (alert_type, message)
    hardware_status_signal = Signal(dict)  # LED and buzzer status

    def __init__(self):
        super().__init__()
        self.running = False
        self.video_source = "/dev/video0"
        self.delegate_path = ""
        self.use_npu = False

        # Models
        self.face_detector = None
        self.face_mesher = None
        self.eye_mesher = None

        # Alert counters
        self.closed_eye_frames = 0
        self.yawn_frames = 0
        self.distracted_frames = 0

        # Thresholds
        self.EYE_CLOSED_THRESHOLD = 0.2
        self.MOUTH_OPEN_THRESHOLD = 0.3
        self.CLOSED_EYE_ALERT_FRAMES = 30  # ~1 second at 30fps
        self.YAWN_ALERT_FRAMES = 45  # ~1.5 seconds
        self.DISTRACTION_ALERT_FRAMES = 60  # ~2 seconds

        # Buzzer and LED control
        self.buzzer_enabled = True
        self.led_enabled = True
        self.tts_enabled = False
        self.buzzer_chip = None
        self.buzzer_line = None
        self.buzzer_active = False
        self.current_led_state = None

        # TTS timing
        self.eye_closed_start_time = None
        self.eye_alert_triggered = False

    def set_video_source(self, source):
        """Set video source"""
        self.video_source = source

    def set_delegate(self, use_npu, delegate_path):
        """Set NPU delegate configuration"""
        self.use_npu = use_npu
        self.delegate_path = delegate_path

    def set_hardware_options(self, buzzer_enabled, led_enabled, tts_enabled):
        """Set buzzer, LED, and TTS options"""
        self.buzzer_enabled = buzzer_enabled
        self.led_enabled = led_enabled
        self.tts_enabled = tts_enabled

    # ===== GPIO BUZZER CONTROL =====
    def init_buzzer(self):
        """Initialize GPIO buzzer control"""
        if not GPIO_AVAILABLE or not self.buzzer_enabled:
            self.log_signal.emit("Buzzer disabled or gpiod not available")
            return False

        try:
            # Open GPIO2 bank (gpiochip1)
            self.buzzer_chip = gpiod.Chip("gpiochip1")
            # Get line 1 (GPIO2_IO01)
            self.buzzer_line = self.buzzer_chip.get_line(1)
            # Request line as output, initial value 0
            self.buzzer_line.request(consumer="drowsiness_buzzer", 
                                    type=gpiod.LINE_REQ_DIR_OUT, 
                                    default_vals=[0])
            self.log_signal.emit("✓ Buzzer initialized successfully")
            return True
        except Exception as e:
            self.log_signal.emit(f"Error initializing buzzer: {e}")
            return False

    def cleanup_buzzer(self):
        """Clean up buzzer resources"""
        try:
            if self.buzzer_line:
                self.buzzer_line.set_value(0)  # Turn off buzzer
                self.buzzer_line.release()
            if self.buzzer_chip:
                self.buzzer_chip.close()
            self.log_signal.emit("Buzzer cleanup completed")
        except Exception as e:
            self.log_signal.emit(f"Error during buzzer cleanup: {e}")

    def control_buzzer(self, should_activate):
        """Control buzzer based on drowsiness detection"""
        if not self.buzzer_enabled or not self.buzzer_line:
            return

        try:
            if should_activate and not self.buzzer_active:
                # Turn on buzzer
                self.buzzer_line.set_value(1)
                self.buzzer_active = True
                self.log_signal.emit("🔊 Buzzer activated - Drowsiness detected!")
            elif not should_activate and self.buzzer_active:
                # Turn off buzzer
                self.buzzer_line.set_value(0)
                self.buzzer_active = False
                self.log_signal.emit("🔇 Buzzer deactivated")
        except Exception as e:
            self.log_signal.emit(f"Error controlling buzzer: {e}")

    # ===== LED CONTROL =====
    def set_led(self, led_path, state):
        """Set LED state (255 for on, 0 for off)"""
        if not self.led_enabled:
            return
        try:
            with open(led_path, 'w') as f:
                f.write(str(state))
        except Exception as e:
            self.log_signal.emit(f"Error controlling LED {led_path}: {e}")

    def turn_off_all_leds(self):
        """Turn off all LEDs"""
        if not self.led_enabled:
            return
        self.set_led(LED_RED_PATH, 0)
        self.set_led(LED_GREEN_PATH, 0)
        self.set_led(LED_BLUE_PATH, 0)

    def control_leds_by_face_direction(self, face_direction):
        """Control LEDs based on face direction"""
        if not self.led_enabled:
            return

        # Only change LEDs if the state is different
        if self.current_led_state != face_direction:
            # Turn off all LEDs first
            self.turn_off_all_leds()

            # Turn on appropriate LED based on face direction
            if face_direction == "Left":
                self.set_led(LED_GREEN_PATH, 255)  # Green LED for left
            elif face_direction == "Right":
                self.set_led(LED_BLUE_PATH, 255)  # Blue LED for right
            elif face_direction == "Up" or face_direction == "Down":
                self.set_led(LED_RED_PATH, 255)  # Red LED for up/down
            # Forward direction keeps all LEDs off

            self.current_led_state = face_direction

    # ===== TTS CONTROL =====
    def prepare_audio(self):
        """Prepare TTS audio file"""
        if not self.tts_enabled:
            return
        if not os.path.exists(AUDIO_PATH):
            os.system(f'espeak -w {AUDIO_PATH} "Wake up!"')

    def speak_wake_up(self):
        """Play TTS alert"""
        if not self.tts_enabled:
            return
        def run():
            os.system(f'aplay -D plughw:0,0 {AUDIO_PATH} > /dev/null 2>&1')
        threading.Thread(target=run).start()

    def draw_face_box(self, image, bboxes, landmarks, scores):
        """Draw face detection boxes"""
        for bbox, landmark, score in zip(bboxes.astype(int), landmarks.astype(int), scores):
            cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(255, 0, 0), thickness=2)
            score_label = f"{score:.2f}"
            (label_width, label_height), baseline = cv2.getTextSize(
                score_label, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=2
            )
            label_btmleft = bbox[:2].copy() + 10
            label_btmleft[0] += label_width
            label_btmleft[1] += label_height
            cv2.rectangle(image, tuple(bbox[:2]), tuple(label_btmleft),
                         color=(255, 0, 0), thickness=cv2.FILLED)
            cv2.putText(image, score_label, (bbox[0] + 5, label_btmleft[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
        return image

    def initialize_models(self):
        """Initialize all detection models"""
        try:
            delegate = self.delegate_path if self.use_npu else ""
            self.log_signal.emit("Initializing face detector...")

            # Get initial frame to determine dimensions
            if self.video_source.isdigit():
                cap_input = int(self.video_source)
            else:
                cap_input = self.video_source

            temp_cap = cv2.VideoCapture(cap_input)
            ret, image = temp_cap.read()
            temp_cap.release()

            if not ret:
                self.log_signal.emit(f"Error: Cannot read from {self.video_source}")
                return False, None, None

            h, w, _ = image.shape
            target_dim = max(w, h)

            # Initialize models
            self.face_detector = FaceDetector(
                model_path=str(MODEL_PATH / DETECT_MODEL),
                delegate_path=delegate,
                img_size=(target_dim, target_dim)
            )
            self.log_signal.emit("✓ Face detector loaded")

            self.face_mesher = FaceMesher(
                model_path=str(MODEL_PATH / LANDMARK_MODEL),
                delegate_path=delegate
            )
            self.log_signal.emit("✓ Face mesher loaded")

            self.eye_mesher = EyeMesher(
                model_path=str(MODEL_PATH / EYE_MODEL),
                delegate_path=delegate
            )
            self.log_signal.emit("✓ Eye mesher loaded")

            mode = "NPU" if self.use_npu else "CPU"
            self.log_signal.emit(f"All models loaded on {mode}!")

            # Initialize buzzer if enabled
            if self.buzzer_enabled:
                self.init_buzzer()

            # Initialize LEDs
            if self.led_enabled:
                self.turn_off_all_leds()
                self.log_signal.emit("✓ LEDs initialized")

            # Prepare TTS audio
            if self.tts_enabled:
                self.prepare_audio()
                self.log_signal.emit("✓ TTS audio prepared")

            return True, target_dim, (h, w)

        except Exception as e:
            self.log_signal.emit(f"Error initializing models: {e}")
            return False, None, None

    def process_frame(self, image, target_dim, padded_size):
        """Process single frame for driver monitoring"""
        # Pad image
        padded = cv2.copyMakeBorder(
            image.copy(),
            *padded_size,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
        padded = cv2.flip(padded, 3)

        # Face detection
        bboxes_decoded, landmarks, scores = self.face_detector.inference(padded)

        if len(bboxes_decoded) == 0:
            self.distracted_frames += 1
            # Turn off buzzer when no face
            self.control_buzzer(False)
            return padded, {
                'face_detected': False,
                'yawning': False,
                'eyes_closed': False,
                'face_direction': 'No Face',
                'alert_level': 'warning' if self.distracted_frames > self.DISTRACTION_ALERT_FRAMES else 'ok'
            }

        # Reset distraction counter
        self.distracted_frames = 0

        mesh_landmarks_inverse = []
        r_vecs, t_vecs = [], []

        for i, (bbox, landmark) in enumerate(zip(bboxes_decoded, landmarks)):
            # Landmark detection
            aligned_face, M, angel = self.face_detector.align(padded, landmark)
            mesh_landmark, mesh_scores = self.face_mesher.inference(aligned_face)
            mesh_landmark_inverse = self.face_detector.inverse(mesh_landmark, M)
            mesh_landmarks_inverse.append(mesh_landmark_inverse)

            # Pose detection
            r_vec, t_vec = self.face_detector.decode_pose(landmark)
            r_vecs.append(r_vec)
            t_vecs.append(t_vec)

        # Draw face boxes
        image_show = padded.copy()
        self.draw_face_box(image_show, bboxes_decoded, landmarks, scores)

        # Analyze first detected face
        mesh_landmark = mesh_landmarks_inverse[0]
        r_vec, t_vec = r_vecs[0], t_vecs[0]

        # Get mouth ratio
        mouth_ratio = get_mouth_ratio(mesh_landmark, image_show)

        # Get eye boxes and analyze
        left_box, right_box = get_eye_boxes(mesh_landmark, padded.shape)
        left_eye_img = padded[left_box[0][1]:left_box[1][1], left_box[0][0]:left_box[1][0]]
        right_eye_img = padded[right_box[0][1]:right_box[1][1], right_box[0][0]:right_box[1][0]]

        left_eye_landmarks, left_iris_landmarks = self.eye_mesher.inference(left_eye_img)
        right_eye_landmarks, right_iris_landmarks = self.eye_mesher.inference(right_eye_img)

        left_eye_ratio = get_eye_ratio(left_eye_landmarks, image_show, left_box[0])
        right_eye_ratio = get_eye_ratio(right_eye_landmarks, image_show, right_box[0])

        # Get face angles
        pitch, roll, yaw = get_face_angle(r_vec, t_vec)
        iris_ratio = get_iris_ratio(left_eye_landmarks, right_eye_landmarks)

        # Determine states
        yawning = mouth_ratio > self.MOUTH_OPEN_THRESHOLD
        eyes_closed = left_eye_ratio < self.EYE_CLOSED_THRESHOLD and right_eye_ratio < self.EYE_CLOSED_THRESHOLD

        # Determine face direction
        if yaw > 15 and iris_ratio > 1.15:
            face_direction = "Left"
            distracted = True
        elif yaw < -15 and iris_ratio < 0.85:
            face_direction = "Right"
            distracted = True
        elif pitch > 30:
            face_direction = "Up"
            distracted = True
        elif pitch < -13:
            face_direction = "Down"
            distracted = True
        else:
            face_direction = "Forward"
            distracted = False

        # Update counters
        if yawning:
            self.yawn_frames += 1
        else:
            self.yawn_frames = 0

        if eyes_closed:
            self.closed_eye_frames += 1
            # TTS alert for prolonged eye closure
            if self.tts_enabled:
                if self.eye_closed_start_time is None:
                    self.eye_closed_start_time = time.time()
                elif (time.time() - self.eye_closed_start_time > 0.5) and not self.eye_alert_triggered:
                    self.speak_wake_up()
                    self.eye_alert_triggered = True
        else:
            self.closed_eye_frames = 0
            self.eye_closed_start_time = None
            self.eye_alert_triggered = False

        if distracted:
            self.distracted_frames += 1
        else:
            self.distracted_frames = 0

        # Control LEDs based on face direction
        self.control_leds_by_face_direction(face_direction)

        # Control buzzer based on drowsiness (yawning or eyes closed)
        if yawning:
            if self.yawn_start_time is None:
                self.yawn_start_time = time.time()
            else:
                if (time.time() - self.yawn_start_time >= 2.0) and not self.yawn_alert_triggered:
                    self.yawn_alert_triggered = True
                    self.control_buzzer(True) 
        else:
        
            self.yawn_start_time = None
            self.yawn_alert_triggered = False
            self.control_buzzer(False)
            

        # Draw status on image
        text_x = padded_size[2] + 70
        text_y_base = padded_size[0] + 70

        # Yawning status
        yawn_color = (255, 0, 0) if yawning else (0, 255, 0)
        yawn_text = "Yawning: Detected" if yawning else "Yawning: No"
        cv2.putText(image_show, yawn_text, (text_x, text_y_base),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, yawn_color, 2)

        # Eye status
        eye_color = (255, 0, 0) if eyes_closed else (0, 255, 0)
        eye_text = "Eye: Closed" if eyes_closed else "Eye: Open"
        cv2.putText(image_show, eye_text, (text_x, text_y_base + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, eye_color, 2)

        # Face direction
        face_color = (255, 0, 0) if distracted else (0, 255, 0)
        face_text = f"Face: {face_direction}"
        cv2.putText(image_show, face_text, (text_x, text_y_base + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, face_color, 2)

        # Buzzer status
        buzzer_status = "BUZZER: ON" if self.buzzer_active else "BUZZER: OFF"
        buzzer_color = (255, 0, 0) if self.buzzer_active else (0, 255, 0)
        cv2.putText(image_show, buzzer_status, (text_x, text_y_base + 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, buzzer_color, 2)

        # Determine alert level
        alert_level = 'ok'
        if self.closed_eye_frames > self.CLOSED_EYE_ALERT_FRAMES:
            alert_level = 'critical'
            self.alert_signal.emit('drowsy', 'Driver appears drowsy - eyes closed!')
        elif self.yawn_frames > self.YAWN_ALERT_FRAMES:
            alert_level = 'warning'
            self.alert_signal.emit('fatigue', 'Driver showing signs of fatigue')
        elif self.distracted_frames > self.DISTRACTION_ALERT_FRAMES:
            alert_level = 'warning'
            self.alert_signal.emit('distracted', f'Driver distracted - looking {face_direction}')

        # Emit hardware status
        self.hardware_status_signal.emit({
            'buzzer_active': self.buzzer_active,
            'led_state': self.current_led_state or 'Off',
            'face_direction': face_direction
        })

        # Remove padding
        image_show = image_show[padded_size[0]:target_dim - padded_size[1],
                                padded_size[2]:target_dim - padded_size[3]]

        return image_show, {
            'face_detected': True,
            'yawning': yawning,
            'eyes_closed': eyes_closed,
            'face_direction': face_direction,
            'mouth_ratio': mouth_ratio,
            'left_eye_ratio': left_eye_ratio,
            'right_eye_ratio': right_eye_ratio,
            'pitch': pitch,
            'yaw': yaw,
            'roll': roll,
            'alert_level': alert_level
        }

    def run(self):
        """Main processing loop"""
        success, target_dim, (h, w) = self.initialize_models()
        if not success:
            return

        # Calculate padding
        padded_size = [
            (target_dim - h) // 2, (target_dim - h + 1) // 2,
            (target_dim - w) // 2, (target_dim - w + 1) // 2
        ]

        # Initialize video capture
        if self.video_source.isdigit():
            cap = cv2.VideoCapture(int(self.video_source))
        else:
            cap = cv2.VideoCapture(self.video_source)

        if not cap.isOpened():
            self.log_signal.emit(f"Error: Cannot open video source {self.video_source}")
            return

        self.log_signal.emit(f"✓ Video source initialized: {self.video_source}")
        self.log_signal.emit("Starting driver monitoring...\n")

        # Performance tracking
        total_frames = 0
        total_time = 0
        self.running = True
        first_inference = True

        while self.running:
            ret, image = cap.read()
            if not ret or image is None:
                break

            total_frames += 1
            loop_start = time.time()

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process frame
            process_start = time.time()
            image_show, status = self.process_frame(image_rgb, target_dim, padded_size)
            process_end = time.time()

            if first_inference:
                warmup_time = int((process_end - process_start) * 1000)
                self.log_signal.emit(f"First inference time: {warmup_time}ms (includes warmup)")
                first_inference = False

            # Calculate performance metrics
            loop_end = time.time()
            total_time += (loop_end - loop_start)
            fps = int(total_frames / total_time) if total_time > 0 else 0
            process_time = (process_end - process_start) * 1000

            # Status message
            mode = 'NPU' if self.use_npu else 'CPU'
            msg = f"FPS: {fps} | Processing: {process_time:.2f}ms | Mode: {mode}"

            # Convert to BGR for display
            result = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)

            # Convert frame to QImage
            rgb_image = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            h_img, w_img, ch = rgb_image.shape
            bytes_per_line = ch * w_img
            qt_image = QImage(rgb_image.data, w_img, h_img, bytes_per_line, QImage.Format_RGB888)

            # Emit signals
            self.change_pixmap_signal.emit(qt_image)
            self.fps_signal.emit(msg)
            self.status_signal.emit(status)

        # Cleanup
        cap.release()
        self.cleanup_buzzer()
        self.turn_off_all_leds()
        self.log_signal.emit("\n✓ Driver monitoring stopped")

    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        self.cleanup_buzzer()
        self.turn_off_all_leds()
        self.wait()


class DriverMonitorGUI(QMainWindow):
    """Main GUI for driver monitoring system with buzzer and LED controls"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Driver Monitoring System with Buzzer & LED Control")
        self.setGeometry(100, 100, 1600, 900)
        
        self.yawn_start_time = None
        self.yawn_alert_triggered = False
        
        
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.setWindowFlag(Qt.WindowTitleHint, True)
        self.setWindowFlag(Qt.WindowSystemMenuHint, True)

        # Initialize monitoring thread
        self.monitor_thread = DriverMonitorThread()
        self.monitor_thread.change_pixmap_signal.connect(self.update_image)
        self.monitor_thread.fps_signal.connect(self.update_fps)
        self.monitor_thread.log_signal.connect(self.update_log)
        self.monitor_thread.status_signal.connect(self.update_status)
        self.monitor_thread.alert_signal.connect(self.handle_alert)
        self.monitor_thread.hardware_status_signal.connect(self.update_hardware_status)

        self.init_ui()

    def init_ui(self):
        """Initialize UI"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left panel - Controls
        control_panel = self.create_control_panel()
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(control_panel)
        main_layout.addWidget(scroll_area, 1)
        
        

        # Right panel - Video and Log
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, 2)

    def create_control_panel(self):
        """Create control panel"""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)

        # Model settings
        model_group = QGroupBox("System Settings")
        model_layout = QVBoxLayout()

        video_label = QLabel("Video Source:")
        self.video_input = QLineEdit("/dev/video0")
        model_layout.addWidget(video_label)
        model_layout.addWidget(self.video_input)

        model_info = QLabel("Models:\n• Face Detection\n• Face Landmarks\n• Eye/Iris Tracking")
        model_info.setStyleSheet("QLabel { font-size: 10px; color: #888; padding: 5px; }")
        model_layout.addWidget(model_info)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Hardware Control Options
        hardware_group = QGroupBox("Hardware Control")
        hardware_layout = QVBoxLayout()

        self.buzzer_checkbox = QCheckBox("Enable Buzzer Alert")
        self.buzzer_checkbox.setChecked(True)
        self.buzzer_checkbox.setStyleSheet("QCheckBox { font-size: 12px; padding: 5px; }")
        hardware_layout.addWidget(self.buzzer_checkbox)

        self.led_checkbox = QCheckBox("Enable LED Indicators")
        self.led_checkbox.setChecked(True)
        self.led_checkbox.setStyleSheet("QCheckBox { font-size: 12px; padding: 5px; }")
        hardware_layout.addWidget(self.led_checkbox)

        self.tts_checkbox = QCheckBox("Enable TTS Alert (espeak)")
        self.tts_checkbox.setChecked(False)
        self.tts_checkbox.setStyleSheet("QCheckBox { font-size: 12px; padding: 5px; }")
        hardware_layout.addWidget(self.tts_checkbox)

        hardware_info = QLabel("LED Mapping:\n• Green: Looking Left\n• Blue: Looking Right\n• Red: Looking Up/Down")
        hardware_info.setStyleSheet("QLabel { font-size: 9px; color: #888; padding: 5px; }")
        hardware_layout.addWidget(hardware_info)

        hardware_group.setLayout(hardware_layout)
        layout.addWidget(hardware_group)

        # Hardware Status Indicators
        hw_status_group = QGroupBox("Hardware Status")
        hw_status_layout = QVBoxLayout()

        self.buzzer_status_label = QLabel("Buzzer: OFF")
        self.buzzer_status_label.setStyleSheet("QLabel { font-size: 11px; padding: 5px; background-color: #4CAF50; border-radius: 3px; }")
        hw_status_layout.addWidget(self.buzzer_status_label)

        self.led_status_label = QLabel("LED: Off")
        self.led_status_label.setStyleSheet("QLabel { font-size: 11px; padding: 5px; background-color: #666; border-radius: 3px; }")
        hw_status_layout.addWidget(self.led_status_label)

        hw_status_group.setLayout(hw_status_layout)
        layout.addWidget(hw_status_group)

        # Driver status indicators
        status_group = QGroupBox("Driver Status")
        status_layout = QVBoxLayout()

        self.face_status_label = QLabel("Face: Not Detected")
        self.face_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #666; border-radius: 3px; }")
        status_layout.addWidget(self.face_status_label)

        self.eye_status_label = QLabel("Eyes: Unknown")
        self.eye_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #666; border-radius: 3px; }")
        status_layout.addWidget(self.eye_status_label)

        self.yawn_status_label = QLabel("Yawning: No")
        self.yawn_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #666; border-radius: 3px; }")
        status_layout.addWidget(self.yawn_status_label)

        self.direction_status_label = QLabel("Direction: Unknown")
        self.direction_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #666; border-radius: 3px; }")
        status_layout.addWidget(self.direction_status_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # Alert level indicator
        alert_group = QGroupBox("Alert Level")
        alert_layout = QVBoxLayout()

        self.alert_label = QLabel("SYSTEM READY")
        self.alert_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 15px; background-color: #4CAF50; color: white; border-radius: 5px; }")
        self.alert_label.setAlignment(Qt.AlignCenter)
        alert_layout.addWidget(self.alert_label)

        alert_group.setLayout(alert_layout)
        layout.addWidget(alert_group)

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

        # Control buttons
        self.start_button = QPushButton("Start Monitoring")
        self.start_button.clicked.connect(self.start_monitoring)
        self.start_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")

        self.stop_button = QPushButton("Stop Monitoring")
        self.stop_button.clicked.connect(self.stop_monitoring)
        self.stop_button.setEnabled(False)
        self.stop_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; padding: 10px; font-size: 14px; font-weight: bold; }")

        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)

        # FPS display
        self.fps_label = QLabel("FPS: -- | Processing: --ms")
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
        title = QLabel("Driver Monitoring Camera")
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
    def update_status(self, status):
        """Update status indicators"""
        # Face detection status
        if status['face_detected']:
            self.face_status_label.setText(f"Face: Detected")
            self.face_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #4CAF50; color: white; border-radius: 3px; }")
        else:
            self.face_status_label.setText("Face: Not Detected")
            self.face_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #f44336; color: white; border-radius: 3px; }")

        # Eye status
        if status.get('eyes_closed', False):
            self.eye_status_label.setText("Eyes: CLOSED ⚠")
            self.eye_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #f44336; color: white; border-radius: 3px; font-weight: bold; }")
        else:
            self.eye_status_label.setText("Eyes: Open")
            self.eye_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #4CAF50; color: white; border-radius: 3px; }")

        # Yawn status
        if status.get('yawning', False):
            self.yawn_status_label.setText("Yawning: DETECTED ⚠")
            self.yawn_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #FFA500; color: white; border-radius: 3px; font-weight: bold; }")
        else:
            self.yawn_status_label.setText("Yawning: No")
            self.yawn_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #4CAF50; color: white; border-radius: 3px; }")

        # Direction status
        direction = status.get('face_direction', 'Unknown')
        if direction == 'Forward':
            self.direction_status_label.setText(f"Direction: {direction}")
            self.direction_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #4CAF50; color: white; border-radius: 3px; }")
        else:
            self.direction_status_label.setText(f"Direction: {direction} ⚠")
            self.direction_status_label.setStyleSheet("QLabel { font-size: 12px; padding: 8px; background-color: #FFA500; color: white; border-radius: 3px; font-weight: bold; }")

        # Alert level
        alert_level = status.get('alert_level', 'ok')
        if alert_level == 'critical':
            self.alert_label.setText("⚠ CRITICAL ALERT ⚠")
            self.alert_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 15px; background-color: #f44336; color: white; border-radius: 5px; }")
        elif alert_level == 'warning':
            self.alert_label.setText("⚠ WARNING ⚠")
            self.alert_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 15px; background-color: #FFA500; color: white; border-radius: 5px; }")
        else:
            self.alert_label.setText("MONITORING ACTIVE")
            self.alert_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 15px; background-color: #4CAF50; color: white; border-radius: 5px; }")

    @Slot(dict)
    def update_hardware_status(self, hw_status):
        """Update hardware status indicators"""
        # Buzzer status
        if hw_status.get('buzzer_active', False):
            self.buzzer_status_label.setText("🔊 Buzzer: ACTIVE")
            self.buzzer_status_label.setStyleSheet("QLabel { font-size: 11px; padding: 5px; background-color: #f44336; color: white; font-weight: bold; border-radius: 3px; }")
        else:
            self.buzzer_status_label.setText("🔇 Buzzer: OFF")
            self.buzzer_status_label.setStyleSheet("QLabel { font-size: 11px; padding: 5px; background-color: #4CAF50; color: white; border-radius: 3px; }")

        # LED status
        led_state = hw_status.get('led_state', 'Off')
        face_dir = hw_status.get('face_direction', 'Unknown')
        
        if led_state == 'Left':
            self.led_status_label.setText("💡 LED: Green (Left)")
            self.led_status_label.setStyleSheet("QLabel { font-size: 11px; padding: 5px; background-color: #4CAF50; color: white; border-radius: 3px; }")
        elif led_state == 'Right':
            self.led_status_label.setText("💡 LED: Blue (Right)")
            self.led_status_label.setStyleSheet("QLabel { font-size: 11px; padding: 5px; background-color: #2196F3; color: white; border-radius: 3px; }")
        elif led_state == 'Up' or led_state == 'Down':
            self.led_status_label.setText(f"💡 LED: Red ({led_state})")
            self.led_status_label.setStyleSheet("QLabel { font-size: 11px; padding: 5px; background-color: #f44336; color: white; border-radius: 3px; }")
        else:
            self.led_status_label.setText("💡 LED: Off (Forward)")
            self.led_status_label.setStyleSheet("QLabel { font-size: 11px; padding: 5px; background-color: #666; color: white; border-radius: 3px; }")

    @Slot(str, str)
    def handle_alert(self, alert_type, message):
        """Handle alert notifications"""
        self.update_log(f"ALERT - {alert_type}: {message}")

    def start_monitoring(self):
        """Start driver monitoring"""
        self.log_text.clear()
        self.update_log("=== Starting Driver Monitoring System ===")

        # Get settings
        video_source = self.video_input.text()
        use_npu = self.npu_radio.isChecked()
        delegate_path = self.delegate_input.text()

        # Hardware options
        buzzer_enabled = self.buzzer_checkbox.isChecked()
        led_enabled = self.led_checkbox.isChecked()
        tts_enabled = self.tts_checkbox.isChecked()

        # Validate NPU settings
        if use_npu and not os.path.exists(delegate_path):
            self.update_log(f"NPU delegate not found: {delegate_path}")
            self.update_log("Switching to CPU mode")
            use_npu = False

        # Configure monitoring thread
        self.monitor_thread.set_video_source(video_source)
        self.monitor_thread.set_delegate(use_npu, delegate_path)
        self.monitor_thread.set_hardware_options(buzzer_enabled, led_enabled, tts_enabled)

        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.cpu_radio.setEnabled(False)
        self.npu_radio.setEnabled(False)
        self.video_input.setEnabled(False)
        self.delegate_input.setEnabled(False)
        self.buzzer_checkbox.setEnabled(False)
        self.led_checkbox.setEnabled(False)
        self.tts_checkbox.setEnabled(False)

        # Start thread
        self.monitor_thread.start()

    def stop_monitoring(self):
        """Stop driver monitoring"""
        self.monitor_thread.stop()

        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.cpu_radio.setEnabled(True)
        self.npu_radio.setEnabled(True)
        self.video_input.setEnabled(True)
        self.delegate_input.setEnabled(True)
        self.buzzer_checkbox.setEnabled(True)
        self.led_checkbox.setEnabled(True)
        self.tts_checkbox.setEnabled(True)

        self.video_label.clear()
        self.video_label.setText("Monitoring Stopped")
        self.update_log("Monitoring stopped")

        # Reset status indicators
        self.alert_label.setText("SYSTEM STOPPED")
        self.alert_label.setStyleSheet("QLabel { font-size: 16px; font-weight: bold; padding: 15px; background-color: #666; color: white; border-radius: 5px; }")

    def closeEvent(self, event):
        """Handle window close"""
        self.monitor_thread.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = DriverMonitorGUI()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()

