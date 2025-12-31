import cv2
import time
import numpy as np
import argparse
import pyttsx3
import os
import subprocess
import threading
import whisper
import tempfile
import uuid
import string
from face_detection import YoloFace
from face_recognition import Facenet
from face_database import FaceDatabase

# Configuration
WHISPER_MODEL = "tiny"
NAME_PROMPT = "Adarsh, Aarav, Nayan, Riya, Lakshmi, Arjun, Priya, Deepa, Neha, Rohan, Anjali, Vijay, Vinay, Vikram, Sanjay, Suraj"
COMMAND_PROMPT = "new, add, remove, delete, quit"
CONFIRM_PROMPT = "yes, no"
AUDIO_DEVICE = "plughw:3,0"  # Default audio device
TTS_DEVICE = "plughw:4,0"

COMMAND_DURATION = 3
NAME_DURATION = 4
CONFIRM_DURATION = 3

TTS_DEBOUNCE = 3.0
COMMAND_DEBOUNCE = 2.0
NO_FACE_TIMEOUT = 10.0
FRAME_SKIP = 5

SIDE_PANEL_WIDTH = 300
SIDE_PANEL_HEIGHT = 480

TEXT_TIMEOUT = 5.0
RECORDING_TIMEOUT = 6.0
MIN_MESSAGE_DISPLAY = 1.0  # Minimum time to show messages (seconds)

class AppState:
    def __init__(self):
        self.last_spoken_name = None
        self.current_command = None
        self.running = True
        self.processing_command = False
        self.last_command_time = 0
        self.no_face_time = None
        self.frame_count = 0
        self.command_thread = None
        self.add_state = None
        self.add_embeddings = None
        self.add_name = None
        self.face_capture_start = None
        # New attributes for improved face capture
        self.face_samples = []
        self.face_quality_scores = []
        # Existing attributes
        self.remove_names = None
        self.recording_message = None
        self.recording_message_time = 0
        self.is_recording = False
        self.waiting_for_number = False
        self.last_audio_check = 0

    def process_command(self):
        """Process the current voice command"""
        if self.command_thread and self.command_thread.is_alive():
            print("Command processing skipped (thread busy)")
            return

        def run_command():
            current_time = time.time()
            if self.processing_command or (current_time - self.last_command_time < COMMAND_DEBOUNCE):
                print("Command processing skipped (debounce or busy)")
                return

            self.processing_command = True
            self.last_command_time = current_time

            try:
                print(f"Processing command: {self.current_command}")
                if self.current_command == "add":
                    self.add_state = "capture_face"
                    self.face_capture_start = None
                    self.face_samples = []
                    self.face_quality_scores = []
                    self.remove_names = None
                    tts.say("Please look at the camera for 5 seconds")

                elif self.current_command == "remove":
                    self.remove_names = database.get_names()
                    self.waiting_for_number = True
                    handle_remove_command()

                elif self.current_command == "quit":
                    self.running = False

            finally:
                self.current_command = None
                self.processing_command = False

        self.command_thread = threading.Thread(target=run_command, daemon=True)
        self.command_thread.start()

    def process_add_state(self, frame):
        """Handle the face addition process with improved face capture"""
        if self.add_state == "capture_face":
            if self.face_capture_start is None:
                if tts.tts_thread and tts.tts_thread.is_alive():
                    return
                self.face_capture_start = time.time()
                # Initialize face sample storage
                self.face_samples = []
                self.face_quality_scores = []

            capture_elapsed = time.time() - self.face_capture_start
            
            # Capture phase - collect multiple samples over 5 seconds
            if capture_elapsed < 5.0:
                boxes = detector.detect(frame)
                if boxes is not None and len(boxes) > 0:
                    box = boxes[0]
                    box[[0, 2]] *= frame.shape[1]
                    box[[1, 3]] *= frame.shape[0]
                    x1, y1, x2, y2 = box.astype(np.int32)
                    
                    # Add padding around face
                    x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
                    x2, y2 = min(x2 + 10, frame.shape[1]), min(y2 + 10, frame.shape[0])
                    
                    face = frame[y1:y2, x1:x2]
                    
                    # Calculate face quality score
                    quality_score = calculate_face_quality(face, box)
                    
                    # Only keep samples with reasonable quality
                    if quality_score > 0.3:  # Minimum quality threshold
                        embeddings = recognizer.get_embeddings(face)
                        self.face_samples.append({
                            'face': face.copy(),
                            'embeddings': embeddings,
                            'quality': quality_score,
                            'bbox': box
                        })
                        self.face_quality_scores.append(quality_score)
                    
                    # Visual feedback - show capture progress
                    progress = int((capture_elapsed / 5.0) * 100)
                    cv2.putText(frame, f"Capturing face: {progress}% ({len(self.face_samples)} samples)", 
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw face rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Quality: {quality_score:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    # No face detected
                    progress = int((capture_elapsed / 5.0) * 100)
                    cv2.putText(frame, f"No face detected: {progress}%", 
                               (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                return

            # Selection phase - choose the best sample
            if not self.face_samples:
                tts.say("No good face samples captured. Please try again.")
                self.add_state = None
                self.face_capture_start = None
                self.face_samples = []
                self.face_quality_scores = []
                return

            # Select the best sample based on quality score
            best_sample = max(self.face_samples, key=lambda x: x['quality'])
            self.add_embeddings = best_sample['embeddings']
            
            print(f"Selected best face sample with quality: {best_sample['quality']:.3f} from {len(self.face_samples)} samples")
            tts.say(f"Face captured successfully with {len(self.face_samples)} samples . Now Please say the name")
            
            # Clean up temporary storage
            self.face_samples = []
            self.face_quality_scores = []
            
            self.add_state = "say_name"
            #tts.say("Please say the name")

        elif self.add_state == "say_name":
            if tts.tts_thread and tts.tts_thread.is_alive():
                return
            name = recognize_name()
            if not name:
                tts.say("Name not recognized")
                self.add_state = "say_name"
                tts.say("Please say the name")
                return

            self.add_state = "confirm_name"
            tts.say(f"Did you say {name}? Please say yes or no")
            self.add_name = name

        elif self.add_state == "confirm_name":
            if tts.tts_thread and tts.tts_thread.is_alive():
                return
            confirmation = recognize_confirmation()
            if confirmation == "yes":
                database.add_name(self.add_name, self.add_embeddings)
                tts.say(f"{self.add_name} added successfully")
                self.add_state = None
                self.add_embeddings = None
                self.add_name = None
                self.face_capture_start = None
            elif confirmation == "no":
                self.add_state = "say_name"
                tts.say("Please say the name")
            else:
                tts.say("Please say yes or no")

def calculate_face_quality(face, bbox):
    """Calculate face quality score based on multiple factors"""
    if face.size == 0:
        return 0.0
    
    try:
        # Convert to grayscale for analysis
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if len(face.shape) == 3 else face
        
        # Factor 1: Face size (larger is generally better for recognition)
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        size_score = min(face_area / 10000.0, 1.0)  # Normalize to 0-1
        
        # Factor 2: Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray_face, cv2.CV_64F).var()
        sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize to 0-1
        
        # Factor 3: Brightness (avoid too dark or too bright faces)
        mean_brightness = np.mean(gray_face)
        brightness_score = 1.0 - abs(mean_brightness - 128) / 128.0
        
        # Factor 4: Contrast (standard deviation of pixel values)
        contrast_score = min(np.std(gray_face) / 64.0, 1.0)
        
        # Weighted combination of factors
        quality_score = (
            0.3 * size_score +
            0.4 * sharpness_score +
            0.2 * brightness_score +
            0.1 * contrast_score
        )
        
        return max(0.0, min(1.0, quality_score))
        
    except Exception as e:
        print(f"Error calculating face quality: {e}")
        return 0.0

def check_audio_device():
    """Verify audio device is available and working"""
    try:
        test_file = f"test_{uuid.uuid4()}.wav"
        result = subprocess.run(
            ["arecord", "-D", AUDIO_DEVICE, "-d", "1", test_file],
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            timeout=2
        )
        if os.path.exists(test_file):
            os.remove(test_file)
            return True
        print(f"Audio check failed: {result.stderr.decode()}")
    except Exception as e:
        print(f"Audio device error: {e}")
    return False

# Initialize critical components
print("Initializing system...")
parser = argparse.ArgumentParser(description="Face recognition with voice command support")
parser.add_argument('-i', '--input', default='0', help='input device index (e.g., 0 for /dev/video0)')
parser.add_argument('-d', '--delegate', default='', help='delegate path')
args = parser.parse_args()

if not check_audio_device():
    print(f"ERROR: Cannot access audio device {AUDIO_DEVICE}. Try:")
    print("1. Run 'arecord -l' to list available devices")
    print("2. Update AUDIO_DEVICE in configuration")
    exit(1)

print("Loading Whisper model...")
whisper_model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded.")

vid = cv2.VideoCapture(int(args.input) if args.input.isdigit() else 0)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

detector = YoloFace("yoloface_int8.tflite", args.delegate)
recognizer = Facenet("facenet_512_int_quantized.tflite", args.delegate)
database = FaceDatabase()

tts_lock = threading.Lock()
app_state = AppState()

class TTSEngine:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('volume', 1.0)
            self.engine.setProperty('rate', 150)
        except Exception as e:
            print(f"TTS init error: {e}")
            self.engine = None
        self.tts_thread = None
        self.last_tts_time = 0
        self.current_text = ""
        self.text_start_time = 0

    def say(self, text):
        current_time = time.time()
        if self.tts_thread and self.tts_thread.is_alive():
            print(f"TTS skipped: {text} (busy)")
            return
        if current_time - self.last_tts_time < TTS_DEBOUNCE:
            print(f"TTS skipped: {text} (debounce)")
            return

        self.current_text = text
        self.text_start_time = current_time

        def run_tts():
            with tts_lock:
                if not self.engine:
                    return
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                    tmp_file = tmpfile.name
                try:
                    self.engine.save_to_file(text, tmp_file)
                    self.engine.runAndWait()
                    subprocess.run(f"aplay -D {TTS_DEVICE} {tmp_file}", shell=True,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"TTS error: {e}")
                finally:
                    if os.path.exists(tmp_file):
                        os.remove(tmp_file)

        self.tts_thread = threading.Thread(target=run_tts, daemon=True)
        self.tts_thread.start()
        self.last_tts_time = current_time
        print(f"TTS: {text}")

tts = TTSEngine()

def record_audio(filename, duration=4):
    """Record audio with comprehensive error handling"""
    try:
        # Set initial state
        app_state.recording_message = "Recording started..."
        app_state.recording_message_time = time.time()
        app_state.is_recording = True
        print("[record_audio] Recording started...")

        # Build arecord command
        cmd = [
            'arecord',
            '-D', AUDIO_DEVICE,
            '-f', 'S16_LE',
            '-r', '16000',
            '-c', '1',
            '-d', str(duration),
            filename
        ]

        with subprocess.Popen(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        ) as proc:
            try:
                return_code = proc.wait(timeout=duration + 2)
                if return_code != 0:
                    error_output = proc.stderr.read() if proc.stderr else "Unknown error"
                    raise RuntimeError(f"arecord failed (code {return_code}): {error_output}")
                if not os.path.exists(filename):
                    raise RuntimeError("No recording file created")
                if os.path.getsize(filename) < 1024:
                    raise RuntimeError("Recording file too small (silent recording?)")
                print("[record_audio] Recording completed successfully")
                app_state.recording_message = "Recording completed"
                app_state.recording_message_time = time.time()
                return True
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                raise RuntimeError("Recording timed out")
    except Exception as e:
        error_msg = str(e)
        print(f"[record_audio] Error: {error_msg}")
        app_state.recording_message = f"Error: {error_msg}"
        app_state.recording_message_time = time.time()
        return False
    finally:
        app_state.is_recording = False
        # Clean up failed temp files
        if 'filename' in locals() and os.path.exists(filename) and not app_state.recording_message.startswith("Recording completed"):
            try:
                os.remove(filename)
            except:
                pass

def whisper_transcribe(audio_file, prompt_context=""):
    try:
        audio_input = whisper.load_audio(audio_file)
        audio_input = whisper.pad_or_trim(audio_input)
        mel = whisper.log_mel_spectrogram(audio_input).to(whisper_model.device)
        options = whisper.DecodingOptions(language="en", fp16=False, prompt=prompt_context, temperature=0.5)
        result = whisper.decode(whisper_model, mel, options)
        text = result.text.strip().lower()
        print(f"Transcribed text: {text}")
        return text
    except Exception as e:
        print(f"Whisper error: {e}")
        return ""
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

def recognize_command():
    tmp_file = f"cmd_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, COMMAND_DURATION):
        print("Failed to record command audio")
        return ""
    cmd_text = whisper_transcribe(tmp_file, f"Commands: {COMMAND_PROMPT}")
    if "new" in cmd_text or "add" in cmd_text:
        return "add"
    elif "remove" in cmd_text or "delete" in cmd_text:
        return "remove"
    elif "quit" in cmd_text or "exit" in cmd_text:
        return "quit"
    print(f"Command not recognized: {cmd_text}")
    return ""

def recognize_name():
    tmp_file = f"name_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, NAME_DURATION):
        print("Failed to record name audio")
        return None

    name_text = whisper_transcribe(tmp_file, f"Example names: {NAME_PROMPT}")
    if not name_text:
        print("No name transcribed")
        return None

    # Improved name cleaning and validation
    translator = str.maketrans('', '', string.punctuation)
    cleaned_text = name_text.translate(translator).strip()

    # Split into words and clean each one
    words = [word.strip().capitalize() for word in cleaned_text.split() if word.strip()]

    # More flexible validation (1–3 parts)
    if not words or len(words) > 3:
        print(f"Invalid name format: {words}")
        return None

    invalid_substrings = ['example', 'name', 'names', 'call', 'my']
    if any(sub in name_text.lower() for sub in invalid_substrings):
        print(f"Contains invalid substring: {name_text}")
        return None

    allowed_suffixes = ['jr', 'sr', 'ii', 'iii', 'iv']
    filtered_words = []
    for word in words:
        if word.lower() in allowed_suffixes and word == words[-1]:
            continue
        filtered_words.append(word)

    if not filtered_words:
        print("No valid name parts after filtering")
        return None

    name = ' '.join(filtered_words)

    if len(name) < 2 or len(name) > 30:
        print(f"Name length invalid: {name}")
        return None

    print(f"Recognized name: {name}")
    return name

def recognize_confirmation():
    tmp_file = f"confirm_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, CONFIRM_DURATION):
        print("Failed to record confirmation audio")
        return None
    confirm_text = whisper_transcribe(tmp_file, f"Responses: {CONFIRM_PROMPT}")
    if "yes" in confirm_text:
        return "yes"
    elif "no" in confirm_text:
        return "no"
    print(f"Confirmation not recognized: {confirm_text}")
    return None

def recognize_number():
    tmp_file = f"number_{uuid.uuid4()}.wav"
    if not record_audio(tmp_file, CONFIRM_DURATION):
        print("Failed to record number audio")
        return None
    number_text = whisper_transcribe(tmp_file, "Numbers: one, two, three, four, five, six, seven, eight, nine, ten")
    number_map = {
        'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
    }
    number = number_map.get(number_text.lower(), None)
    print(f"Recognized number: {number}")
    return number

def process_faces(frame):
    app_state.frame_count += 1
    if app_state.frame_count % FRAME_SKIP != 0:
        return None

    boxes = detector.detect(frame)
    current_time = time.time()

    if boxes is None or len(boxes) == 0:
        if app_state.last_spoken_name is not None:
            app_state.last_spoken_name = None
            app_state.no_face_time = current_time
            print("No face detected: Cleared last spoken name")
        elif app_state.no_face_time and (current_time - app_state.no_face_time) > NO_FACE_TIMEOUT:
            app_state.no_face_time = None
            print("No face timeout: Ready for new greeting")
        return None

    app_state.no_face_time = None
    box = boxes[0]
    box[[0, 2]] *= frame.shape[1]
    box[[1, 3]] *= frame.shape[0]
    x1, y1, x2, y2 = box.astype(np.int32)
    x1, y1 = max(x1 - 10, 0), max(y1 - 10, 0)
    x2, y2 = min(x2 + 10, frame.shape[1]), min(y2 + 10, frame.shape[0])
    face = frame[y1:y2, x1:x2]
    embeddings = recognizer.get_embeddings(face)
    name, confidence = database.find_name(embeddings)
    label = f"{name} ({int(confidence*100)}%)" if name else "unkown"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if name and name.lower()  != "unknown" and name != app_state.last_spoken_name:
        print(f"TTS say hello to {name}")
        tts.say(f"Hello {name}")
        app_state.last_spoken_name = name
        print(f"Set last spoken name to {name}")
    return embeddings

def handle_remove_command():
    names = database.get_names()
    if not names:
        tts.say("No names stored")
        app_state.remove_names = None
        app_state.waiting_for_number = False
        return
    app_state.remove_names = names
    print("Known names:")
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")
    tts.say("Please select the number to remove person from database")
    

def render_side_panel():
    """Create the side panel image (to be composited in one window)."""
    side_panel = np.zeros((SIDE_PANEL_HEIGHT, SIDE_PANEL_WIDTH, 3), dtype=np.uint8)

    y_offset = 20
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (255, 255, 255)
    thickness = 1

    # System status header
    status = "Recording..." if app_state.is_recording else "Ready"
    cv2.putText(side_panel, f"Status: {status}", (10, y_offset), font, font_scale,
                (0, 255, 0) if app_state.is_recording else (255, 255, 255), thickness)
    y_offset += 30

    # Face capture progress
    if app_state.add_state == "capture_face" and app_state.face_capture_start:
        elapsed = time.time() - app_state.face_capture_start
        progress = min(int((elapsed / 5.0) * 100), 100)
        cv2.putText(side_panel, f"Capturing: {progress}%", (10, y_offset), font, font_scale, (0, 255, 255), thickness)
        y_offset += 25
        cv2.putText(side_panel, f"Samples: {len(app_state.face_samples)}", (10, y_offset), font, font_scale, (0, 255, 255), thickness)
        y_offset += 25

    # Recording messages
    if app_state.recording_message:
        message_time = time.time() - app_state.recording_message_time
        if message_time < RECORDING_TIMEOUT:
            cv2.putText(side_panel, "Audio:", (10, y_offset), font, font_scale, (200, 200, 255), thickness)
            y_offset += 25
            lines = app_state.recording_message.split('\n')
            for line in lines:
                cv2.putText(side_panel, line, (20, y_offset), font, font_scale, color, thickness)
                y_offset += 25

    # TTS responses
    if tts.current_text and (time.time() - tts.text_start_time < TEXT_TIMEOUT):
        y_offset += 15
        cv2.putText(side_panel, "System Response:", (10, y_offset), font, font_scale, (0, 255, 0), thickness)
        y_offset += 25
        words = tts.current_text.split()
        line = ""
        for word in words:
            test_line = f"{line} {word}".strip()
            if cv2.getTextSize(test_line, font, font_scale, thickness)[0][0] < SIDE_PANEL_WIDTH - 20:
                line = test_line
            else:
                cv2.putText(side_panel, line, (20, y_offset), font, font_scale, color, thickness)
                y_offset += 25
                line = word
        if line:
            cv2.putText(side_panel, line, (20, y_offset), font, font_scale, color, thickness)
            y_offset += 25

    # Names list for removal
    if app_state.remove_names:
        y_offset += 15
        cv2.putText(side_panel, "Select name to remove:", (10, y_offset), font, font_scale, (0, 255, 255), thickness)
        y_offset += 25
        for i, name in enumerate(app_state.remove_names, 1):
            cv2.putText(side_panel, f"{i}. {name}", (30, y_offset), font, font_scale*0.9, color, thickness)
            y_offset += 22

    return side_panel

def compose_frame_with_panel(frame):
    """Compose a single canvas that contains the webcam frame and the side panel."""
    # Ensure frame height matches panel height for cleaner composition (optional: we can pad instead)
    h, w = frame.shape[:2]
    panel = render_side_panel()

    # Make the canvas height the max of frame and panel heights
    canvas_h = max(h, panel.shape[0])
    canvas_w = w + SIDE_PANEL_WIDTH

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Place the frame on the left
    canvas[0:h, 0:w] = frame

    # Place the panel on the right, top-aligned
    canvas[0:panel.shape[0], w:w+SIDE_PANEL_WIDTH] = panel

    return canvas

def main():
    last_key_time = 0
    cv2.namedWindow('Face Recognition')
    # Initial system message
    app_state.recording_message = "System initialized - Enhanced face capture enabled"
    app_state.recording_message_time = time.time()

    while app_state.running:
        ret, frame = vid.read()
        if not ret:
            print("Camera read error")
            break

        current_time = time.time()

        # Process faces and commands
        process_faces(frame)
        if app_state.add_state:
            app_state.process_add_state(frame)

        # Display recording indicator over the frame
        if app_state.is_recording:
            cv2.putText(frame, "Processing audio...", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # UI controls
        cv2.putText(frame, "Press 'v' to speak command, 'q' to quit",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        # Compose single window content
        composed = compose_frame_with_panel(frame)

        # Show single window
        cv2.imshow('Face Recognition', composed)

        # Key handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('v') and not app_state.processing_command and (current_time - last_key_time > COMMAND_DEBOUNCE):
            last_key_time = current_time
            print("Key 'v' pressed: starting command recognition")
            app_state.current_command = recognize_command()
            if app_state.current_command:
                app_state.process_command()
        elif app_state.waiting_for_number and ord('0') <= key <= ord('9'):
            number = int(chr(key))
            names = app_state.remove_names
            if names and 1 <= number <= len(names):
                name = names[number - 1]
                if database.del_name(name):
                    tts.say(f"Removed {name}")
                else:
                    tts.say(f"{name} not found")
                app_state.remove_names = None
                app_state.waiting_for_number = False
            else:
                tts.say("Invalid number")
                app_state.remove_names = None
                app_state.waiting_for_number = False
        elif key == ord('q'):
            print("Key 'q' pressed: exiting")
            app_state.running = False

    vid.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

