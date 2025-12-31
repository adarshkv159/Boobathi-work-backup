import eventlet
eventlet.monkey_patch()  # Must be first for async Socket.IO

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import time
import threading
import subprocess
import os
import cv2
import numpy as np
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
import uuid
import json

# Lazy imports for heavy models
YoloFace = None
Facenet = None
FaceDatabase = None

# --- CONFIGURATIONS ---
DEFAULT_TTS_DEVICE = "default"
PADDING = 10
DEFAULT_VIDEO_DEVICE_INDEX = 0

# Optimized video settings for ARM
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
VIDEO_FPS = 5
FRAME_SKIP = 0.2  # 200ms between frames

# Face capture settings
FACE_CAPTURE_SAMPLES = 4
CAPTURE_DELAY = 1.0  # Delay between captures in seconds

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Thread pool for heavy operations
executor = ThreadPoolExecutor(max_workers=2)

# Global state
app_state = {
    'operation': None,
    'add_name': None,
    'remove_name': None,
    'last_embedding': None,
    'running': False,
    'status_message': 'System Ready',
    'status_type': 'info',
    'camera_status': 'Disconnected',
    'total_faces': 0,
    'faces_detected': 0,
    'tts_device': DEFAULT_TTS_DEVICE,
    'video_device_index': DEFAULT_VIDEO_DEVICE_INDEX,
    'models_loaded': False,
    'capturing_samples': False,
    'captured_samples': [],
    'pending_add_name': None,
}

# Lazy-loaded global variables
detector = None
recognizer = None
face_db = None
camera_capture = None
camera_thread = None

# --- LAZY MODEL LOADING ---
def load_models():
    """Load heavy models in background thread"""
    global detector, recognizer, face_db, YoloFace, Facenet, FaceDatabase
    
    if app_state['models_loaded']:
        return
    
    try:
        logger.info("Loading models...")
        update_status("Loading AI models...", "warning")
        
        # Import modules only when needed
        if not YoloFace:
            from face_detection import YoloFace
        if not Facenet:
            from face_recognition import Facenet
        if not FaceDatabase:
            from face_database import FaceDatabase
        
        # Load models
        detector = YoloFace("yoloface_int8.tflite", "")
        recognizer = Facenet("facenet_512_int_quantized.tflite", "")
        face_db = FaceDatabase()
        
        app_state['models_loaded'] = True
        app_state['total_faces'] = len(get_all_names()) if face_db else 0
        
        logger.info("Models loaded successfully")
        update_status("AI models loaded", "success")
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        update_status(f"Model loading failed: {str(e)}", "error")

# --- DEVICE ENUMERATION HELPERS ---
def get_audio_output_devices():
    devices = []
    try:
        aplay_output = subprocess.check_output(['aplay', '-l'], stderr=subprocess.DEVNULL, timeout=5).decode()
        for line in aplay_output.splitlines():
            if 'card' in line and 'device' in line:
                import re
                m = re.search(r'card (\d+): ([^\[]+)\[([^\]]+)\], device (\d+): ([^\[]+)\[([^\]]+)\]', line)
                if m:
                    card_num = m.group(1)
                    card_desc = m.group(3).strip()
                    device_num = m.group(4)
                    device_desc = m.group(6).strip()
                    hw_id = f"plughw:{card_num},{device_num}"
                    disp = f"{card_desc} - {device_desc} ({hw_id})"
                    devices.append({"display": disp, "id": hw_id})
    except Exception as e:
        logger.warning(f"Audio output enumeration failed: {e}")
    
    if not devices:
        devices = [{"display": "Default", "id": DEFAULT_TTS_DEVICE}]
    return devices

def get_video_devices(max_devices=5):
    available = []
    for idx in range(max_devices):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            available.append({"display": f"Camera {idx}", "id": idx})
            cap.release()
    if not available:
        available = [{"display": "Default (0)", "id": 0}]
    return available

# --- STATUS MANAGEMENT ---
def update_status(message, status_type="info"):
    app_state['status_message'] = message
    app_state['status_type'] = status_type
    app_state['status_time'] = time.time()
    socketio.emit('status_update', {
        'message': message,
        'type': status_type
    })
    logger.info(f"Status: {message} ({status_type})")

# --- LIGHTWEIGHT TTS ENGINE ---
class LightweightTTS:
    """Lightweight TTS using espeak-ng with message queue"""
    
    def __init__(self, tts_device=None):
        self.tts_device = tts_device or app_state.get("tts_device", DEFAULT_TTS_DEVICE)
        self.use_espeak = self._check_espeak()
        self.message_queue = []
        self.queue_lock = threading.Lock()
        self.is_speaking = False
        self.tts_thread = None
    
    def _check_espeak(self):
        """Check if espeak-ng is available"""
        try:
            subprocess.run(['espeak-ng', '--version'], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            return True
        except:
            logger.warning("espeak-ng not found, TTS disabled")
            return False
    
    def _process_queue(self):
        """Process queued TTS messages"""
        while True:
            text = None
            with self.queue_lock:
                if self.message_queue:
                    text = self.message_queue.pop(0)
                    self.is_speaking = True
                else:
                    self.is_speaking = False
                    break
            
            if text:
                try:
                    if self.tts_device != DEFAULT_TTS_DEVICE:
                        tts_proc = subprocess.Popen(['espeak-ng', text, '--stdout'],
                                                  stdout=subprocess.PIPE,
                                                  stderr=subprocess.DEVNULL)
                        subprocess.run(['aplay', '-D', self.tts_device],
                                     stdin=tts_proc.stdout,
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     timeout=10)
                    else:
                        subprocess.run(['espeak-ng', text],
                                     stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL,
                                     timeout=10)
                    time.sleep(0.3)  # Small delay between messages
                except Exception as e:
                    logger.warning(f"TTS failed: {e}")
    
    def say(self, text):
        """Queue text to be spoken"""
        if not text or not self.use_espeak:
            return
            
        with self.queue_lock:
            self.message_queue.append(text)
            if not self.is_speaking:
                self.tts_thread = threading.Thread(target=self._process_queue, daemon=True)
                self.tts_thread.start()

tts = LightweightTTS()

# --- DATABASE HELPERS ---
def add_name_to_db(name, embedding):
    if face_db:
        face_db.add_name(name, embedding)
        app_state['total_faces'] = len(get_all_names())

def delete_name_from_db(name):
    if face_db:
        face_db.del_name(name)
        app_state['total_faces'] = len(get_all_names())

def get_all_names():
    if face_db:
        return face_db.get_names()
    return []

def find_name_from_embedding(embedding):
    if face_db:
        return face_db.find_name(embedding)
    return "Unknown"

# --- FACE CAPTURE FUNCTIONS ---
def capture_face_samples():
    """Capture multiple face samples for selection"""
    if not app_state['running'] or not detector or not recognizer:
        return
    
    app_state['capturing_samples'] = True
    app_state['captured_samples'] = []
    
    # TTS notification to look at camera
    tts.say("See the camera properly")
    update_status("Look at the camera properly", "warning")
    
    # Wait a moment for user to position
    time.sleep(2)
    
    samples_captured = 0
    while samples_captured < FACE_CAPTURE_SAMPLES and app_state['running']:
        if not camera_capture or not camera_capture.isOpened():
            break
            
        ret, frame = camera_capture.read()
        if not ret:
            continue
            
        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        
        try:
            boxes = detector.detect(frame)
            if len(boxes) > 0:
                # Use the first detected face
                box = boxes[0]
                box[[0, 2]] *= frame.shape[1]
                box[[1, 3]] *= frame.shape[0]
                x1, y1, x2, y2 = box.astype(np.int32)
                x1, y1 = max(x1 - PADDING, 0), max(y1 - PADDING, 0)
                x2, y2 = min(x2 + PADDING, frame.shape[1]), min(y2 + PADDING, frame.shape[0])
                
                face = frame[y1:y2, x1:x2]
                if face.size > 0:
                    # Get embedding for this face
                    embeddings = recognizer.get_embeddings(face)
                    if embeddings is not None:
                        # Encode face image to base64
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                        _, buffer = cv2.imencode('.jpg', face, encode_param)
                        face_data = base64.b64encode(buffer).decode('utf-8')
                        
                        sample_id = str(uuid.uuid4())
                        sample_data = {
                            'id': sample_id,
                            'image': face_data,
                            'embedding': embeddings.tolist(),
                            'captured_at': time.time()
                        }
                        
                        app_state['captured_samples'].append(sample_data)
                        samples_captured += 1
                        
                        update_status(f"Captured sample {samples_captured}/{FACE_CAPTURE_SAMPLES}", "info")
                        
                        # Emit the captured sample to frontend
                        socketio.emit('sample_captured', {
                            'sample': sample_data,
                            'count': samples_captured,
                            'total': FACE_CAPTURE_SAMPLES
                        })
                        
                        # Wait before next capture
                        time.sleep(CAPTURE_DELAY)
            
        except Exception as e:
            logger.error(f"Error during sample capture: {e}")
            continue
    
    app_state['capturing_samples'] = False
    
    if len(app_state['captured_samples']) > 0:
        tts.say("Select one picture from sample")
        update_status("Select the best picture from captured samples", "success")
        socketio.emit('samples_ready', {
            'samples': app_state['captured_samples'],
            'total_captured': len(app_state['captured_samples'])
        })
    else:
        update_status("Failed to capture face samples", "error")
        tts.say("Face capture failed")

# --- OPTIMIZED CAMERA PROCESSING ---
def camera_process():
    global camera_capture
    video_index = app_state.get("video_device_index", 0)
    
    try:
        camera_capture = cv2.VideoCapture(video_index)
        #camera_capture = cv2.VideoCapture("http://192.168.11.117:8081/video")
        camera_capture.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        camera_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        camera_capture.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
        camera_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if not camera_capture.isOpened():
            app_state['camera_status'] = 'Error'
            update_status("Camera access failed", "error")
            tts.say("Camera access failed")
            return
        
        app_state['camera_status'] = 'Connected'
        tts.say("Camera started")
        update_status("Camera connected and running", "success")
        frame_count = 0
        
        while app_state['running']:
            ret, frame = camera_capture.read()
            if not ret:
                continue
            
            # Skip frames to reduce CPU load (except during sample capture)
            if not app_state['capturing_samples']:
                frame_count += 1
                if frame_count % 2 != 0:
                    continue
                
            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            
            current_embedding = None
            detected_count = 0
            
            # Only process faces if models are loaded and not capturing samples
            if detector and recognizer and not app_state['capturing_samples']:
                try:
                    boxes = detector.detect(frame)
                    detected_count = len(boxes)
                    
                    for box in boxes:
                        box[[0, 2]] *= frame.shape[1]
                        box[[1, 3]] *= frame.shape[0]
                        x1, y1, x2, y2 = box.astype(np.int32)
                        x1, y1 = max(x1 - PADDING, 0), max(y1 - PADDING, 0)
                        x2, y2 = min(x2 + PADDING, frame.shape[1]), min(y2 + PADDING, frame.shape[0])
                        
                        face = frame[y1:y2, x1:x2]
                        if face.size == 0:
                            continue
                            
                        embeddings = recognizer.get_embeddings(face)
                        name = "Unknown"
                        if embeddings is not None:
                            current_embedding = embeddings
                            try:
                                name = find_name_from_embedding(embeddings)
                            except Exception:
                                name = "Error"
                        
                        color = (0, 255, 0) if name != "Unknown" else (0, 165, 255)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, name, (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                except Exception as e:
                    logger.error(f"Face processing error: {e}")
            
            app_state['last_embedding'] = current_embedding
            app_state['faces_detected'] = detected_count
            
            # Optimized frame encoding
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_data = base64.b64encode(buffer).decode('utf-8')
            
            socketio.emit('video_frame', {
                'frame': frame_data,
                'faces_detected': detected_count,
                'capturing_samples': app_state['capturing_samples']
            })
            
            time.sleep(FRAME_SKIP)
            
    except Exception as e:
        logger.error(f"Camera process error: {e}")
        app_state['camera_status'] = 'Error'
        update_status(f"Camera error: {str(e)}", "error")
        tts.say("Camera error occurred")
    finally:
        if camera_capture:
            camera_capture.release()

# --- ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/devices')
def get_devices():
    return jsonify({
        'video_devices': get_video_devices(),
        'audio_output_devices': get_audio_output_devices()
    })

@app.route('/api/names')
def get_names():
    names = get_all_names()
    return jsonify({
        'names': names,
        'total_faces': len(names)
    })

@app.route('/api/status')
def get_status():
    return jsonify({
        'status_message': app_state['status_message'],
        'status_type': app_state['status_type'],
        'camera_status': app_state['camera_status'],
        'faces_detected': app_state['faces_detected'],
        'running': app_state['running'],
        'models_loaded': app_state['models_loaded'],
        'capturing_samples': app_state['capturing_samples']
    })

# --- SOCKET EVENTS ---
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected: {request.sid}")
    if not app_state['models_loaded']:
        executor.submit(load_models)

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('start_camera')
def handle_start_camera():
    global camera_thread
    if not app_state['running']:
        if not app_state['models_loaded']:
            update_status("Loading models first...", "warning")
            load_models()
        
        app_state['running'] = True
        app_state['camera_status'] = 'Starting'
        update_status("Camera starting...", "info")
        camera_thread = threading.Thread(target=camera_process, daemon=True)
        camera_thread.start()

@socketio.on('stop_camera')
def handle_stop_camera():
    app_state['running'] = False
    app_state['camera_status'] = 'Disconnected'
    app_state['capturing_samples'] = False
    app_state['captured_samples'] = []
    tts.say("Camera stopped")
    update_status("Camera stopped", "warning")

@socketio.on('manual_add_face')
def handle_manual_add_face(data):
    """Start face capture process with name input"""
    name = data.get('name', '').strip()
    if not name:
        update_status("Please enter a name", "error")
        return
    
    if not app_state['models_loaded']:
        update_status("Models not loaded", "error")
        return
    
    if not app_state['running']:
        update_status("Camera is not running", "error")
        return
    
    # Check if name already exists
    existing_names = get_all_names()
    if name in existing_names:
        update_status(f"Name '{name}' already exists", "error")
        return
    
    app_state['pending_add_name'] = name

    update_status(f"Starting capture for {name}", "info")
    tts.say(f"Starting face capture for {name}")
    
    # Start face capture process in background
    capture_thread = threading.Thread(target=capture_face_samples, daemon=True)
    capture_thread.start()

@socketio.on('select_face_sample')
def handle_select_face_sample(data):
    """Handle selection of face sample for adding to database"""
    sample_id = data.get('sample_id')
    name = app_state.get('pending_add_name')
    
    if not sample_id or not name:
        update_status("Invalid sample selection", "error")
        return
    
    # Find the selected sample
    selected_sample = None
    for sample in app_state['captured_samples']:
        if sample['id'] == sample_id:
            selected_sample = sample
            break
    
    if not selected_sample:
        update_status("Selected sample not found", "error")
        return
    
    # Convert embedding back to numpy array
    embedding = np.array(selected_sample['embedding'])
    
    # Add to database
    add_name_to_db(name, embedding)
    update_status(f"Added face: {name}", "success")
    tts.say(f"Added {name}")
    
    # Clear captured samples and pending name
    app_state['captured_samples'] = []
    app_state['pending_add_name'] = None
    
    socketio.emit('face_added', {'name': name})
    socketio.emit('samples_cleared')

@socketio.on('cancel_face_capture')
def handle_cancel_face_capture():
    """Cancel face capture process"""
    app_state['capturing_samples'] = False
    app_state['captured_samples'] = []
    app_state['pending_add_name'] = None
    update_status("Face capture cancelled", "warning")
    tts.say("Face capture cancelled")
    socketio.emit('samples_cleared')

@socketio.on('manual_delete_face')
def handle_manual_delete_face(data):
    """Manual delete face by name"""
    name = data.get('name', '').strip()
    if not name:
        update_status("Please enter a name to delete", "error")
        return
    
    existing_names = get_all_names()
    if name not in existing_names:
        update_status(f"Name '{name}' not found", "error")
        return
    
    delete_name_from_db(name)
    update_status(f"Deleted face: {name}", "success")
    tts.say(f"Deleted {name}")
    socketio.emit('face_deleted', {'name': name})

@socketio.on('quick_delete')
def handle_quick_delete(data):
    """Quick delete from name list"""
    name = data.get('name')
    if name:
        delete_name_from_db(name)
        update_status(f"Deleted {name}", "success")
        tts.say(f"Deleted {name}")
        socketio.emit('face_deleted', {'name': name})

@socketio.on('update_device')
def handle_update_device(data):
    device_type = data.get('type')
    device_id = data.get('id')
    
    if device_type == 'video':
        app_state['video_device_index'] = device_id
    elif device_type == 'audio_output':
        app_state['tts_device'] = device_id
        global tts
        tts = LightweightTTS(device_id)

@socketio.on('reset_system')
def handle_reset_system():
    """Reset system state"""
    app_state.update({
        'operation': None,
        'add_name': None,
        'remove_name': None,
        'last_embedding': None,
        'running': False,
        'status_message': 'System Ready',
        'status_type': 'info',
        'camera_status': 'Disconnected',
        'faces_detected': 0,
        'capturing_samples': False,
        'captured_samples': [],
        'pending_add_name': None,
    })
    update_status("System reset", "info")
    socketio.emit('system_reset')

@socketio.on('load_models')
def handle_load_models():
    """Manual model loading trigger"""
    if not app_state['models_loaded']:
        executor.submit(load_models)
    else:
        update_status("Models already loaded", "info")

@socketio.on('get_face_list')
def handle_get_face_list():
    """Get current list of faces"""
    names = get_all_names()
    socketio.emit('face_list_update', {
        'names': names,
        'total_faces': len(names)
    })

if __name__ == '__main__':
    logger.info("Starting Enhanced Face Recognition Server for i.MX8M Plus")
    logger.info("New features: 4-sample face capture with selection")
    
    # Load models in background
    executor.submit(load_models)
    
    # Run with eventlet for better performance
    socketio.run(
        app, 
        host='0.0.0.0', 
        port=5000, 
        debug=False,
        use_reloader=False,
        log_output=False
    )

