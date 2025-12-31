#
# Copyright 2020-2023 NXP
#
# SPDX-License-Identifier: Apache-2.0
#

import pathlib
import sys
import time
import argparse
import threading
import os
import cv2
import subprocess
import gpiod

from face_detection import *
from eye_landmark import EyeMesher
from face_landmark import FaceMesher
from utils import *

# === Global Variables ===
MODEL_PATH = pathlib.Path(".")
DETECT_MODEL = "face_detection_front_128_full_integer_quant.tflite"
LANDMARK_MODEL = "face_landmark_192_full_integer_quant.tflite"
EYE_MODEL = "iris_landmark_quant.tflite"

eye_closed_start_time = None
eye_alert_triggered = False
AUDIO_PATH = "/tmp/alert.wav"

# === LED Control Variables ===
LED_RED_PATH = "/sys/class/leds/led-1/brightness"
LED_GREEN_PATH = "/sys/class/leds/led-2/brightness"
LED_BLUE_PATH = "/sys/class/leds/led-3/brightness"

current_led_state = None  # Track current LED state to avoid redundant commands

# === Buzzer Control Variables ===
buzzer_chip = None
buzzer_line = None
buzzer_active = False

# === Initialize Buzzer ===
def init_buzzer():
    """Initialize GPIO buzzer control"""
    global buzzer_chip, buzzer_line
    try:
        # Open GPIO2 bank (gpiochip1)
        buzzer_chip = gpiod.Chip("gpiochip1")
        # Get line 1 (GPIO2_IO01)
        buzzer_line = buzzer_chip.get_line(1)
        # Request line as output, initial value 0
        buzzer_line.request(consumer="drowsiness_buzzer", type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
        print("Buzzer initialized successfully")
        return True
    except Exception as e:
        print(f"Error initializing buzzer: {e}")
        return False

def cleanup_buzzer():
    """Clean up buzzer resources"""
    global buzzer_line, buzzer_chip
    try:
        if buzzer_line:
            buzzer_line.set_value(0)  # Turn off buzzer
            buzzer_line.release()
        if buzzer_chip:
            buzzer_chip.close()
        print("Buzzer cleanup completed")
    except Exception as e:
        print(f"Error during buzzer cleanup: {e}")

def control_buzzer(should_activate):
    """Control buzzer based on drowsiness detection"""
    global buzzer_active, buzzer_line
    
    if not buzzer_line:
        return
    
    try:
        if should_activate and not buzzer_active:
            # Turn on buzzer
            buzzer_line.set_value(1)
            buzzer_active = True
            print("Buzzer activated - Drowsiness detected!")
        elif not should_activate and buzzer_active:
            # Turn off buzzer
            buzzer_line.set_value(0) 
            buzzer_active = False
            print("Buzzer deactivated - Alert state cleared")
    except Exception as e:
        print(f"Error controlling buzzer: {e}")

# === LED Control Functions ===
def set_led(led_path, state):
    """Set LED state (255 for on, 0 for off)"""
    try:
        with open(led_path, 'w') as f:
            f.write(str(state))
    except Exception as e:
        print(f"Error controlling LED {led_path}: {e}")

def turn_off_all_leds():
    """Turn off all LEDs"""
    set_led(LED_RED_PATH, 0)
    set_led(LED_GREEN_PATH, 0)
    set_led(LED_BLUE_PATH, 0)

def control_leds_by_face_direction(face_direction):
    """Control LEDs based on face direction"""
    global current_led_state
    
    # Only change LEDs if the state is different
    if current_led_state != face_direction:
        # Turn off all LEDs first
        turn_off_all_leds()
        
        # Turn on appropriate LED based on face direction
        if face_direction == "Left":
            set_led(LED_GREEN_PATH, 255)  # Green LED for left
        elif face_direction == "Right":
            set_led(LED_BLUE_PATH, 255)   # Blue LED for right
        elif face_direction == "UP" or face_direction == "Down":
            set_led(LED_RED_PATH, 255)    # Red LED for up/down
        # Forward direction keeps all LEDs off
        
        current_led_state = face_direction

# === TTS Function ===
def prepare_audio():
    if not os.path.exists(AUDIO_PATH):
        os.system(f'espeak -w {AUDIO_PATH} "Wake up!"')

def speak_wake_up():
    def run():
        os.system(f'aplay -D plughw:0,0 {AUDIO_PATH} > /dev/null 2>&1')
    threading.Thread(target=run).start()

prepare_audio()

# === Camera Input Setup ===
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i',
    '--input',
    default='/dev/video0',
    help='input to be classified')
parser.add_argument(
    '-d',
    '--delegate',
    default='',
    help='delegate path')
args = parser.parse_args()

if args.input.isdigit():
    cap_input = int(args.input)
else:
    cap_input = args.input
cap = cv2.VideoCapture(cap_input)
ret, image = cap.read()
if not ret:
    print("Can't read frame from source file ", args.input)
    sys.exit(-1)

h, w, _ = image.shape
target_dim = max(w, h)

# === Model Initialization ===
face_detector = FaceDetector(model_path = str(MODEL_PATH / DETECT_MODEL),
                             delegate_path = args.delegate,
                             img_size = (target_dim, target_dim))
face_mesher = FaceMesher(model_path=str((MODEL_PATH / LANDMARK_MODEL)), delegate_path = args.delegate)
eye_mesher = EyeMesher(model_path=str((MODEL_PATH / EYE_MODEL)), delegate_path = args.delegate)

# Initialize LEDs and Buzzer
turn_off_all_leds()
buzzer_initialized = init_buzzer()

def draw_face_box(image, bboxes, landmarks, scores):
    for bbox, landmark, score in zip(bboxes.astype(int), landmarks.astype(int), scores):
        image = cv2.rectangle(image, tuple(bbox[:2]), tuple(bbox[2:]), color=(255, 0, 0), thickness=2)
        landmark = landmark.reshape(-1, 2)
        score_label = "{:.2f}".format(score)
        (label_width, label_height), baseline = cv2.getTextSize(score_label,
                                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                                fontScale=1.0,
                                                                thickness=2)
        label_btmleft = bbox[:2].copy() + 10
        label_btmleft[0] += label_width
        label_btmleft[1] += label_height
        cv2.rectangle(image, tuple(bbox[:2]), tuple(label_btmleft), color=(255, 0, 0), thickness=cv2.FILLED)
        cv2.putText(image, score_label, (bbox[0] + 5, label_btmleft[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 255, 255), thickness=2)
    return image

# === Main Processing Function ===
def main(image):
    global eye_closed_start_time, eye_alert_triggered

    padded_size = [(target_dim - h) // 2, (target_dim - h + 1) // 2,
                   (target_dim - w) // 2, (target_dim - w + 1) // 2]
    padded = cv2.copyMakeBorder(image.copy(),
                                *padded_size,
                                cv2.BORDER_CONSTANT,
                                value=[0, 0, 0])
    padded = cv2.flip(padded, 3)

    bboxes_decoded, landmarks, scores = face_detector.inference(padded)

    mesh_landmarks_inverse = []
    r_vecs, t_vecs = [], []

    for i, (bbox, landmark) in enumerate(zip(bboxes_decoded, landmarks)):
        aligned_face, M, angel = face_detector.align(padded, landmark)
        mesh_landmark, mesh_scores = face_mesher.inference(aligned_face)
        mesh_landmark_inverse = face_detector.inverse(mesh_landmark, M)
        mesh_landmarks_inverse.append(mesh_landmark_inverse)
        r_vec, t_vec = face_detector.decode_pose(landmark)
        r_vecs.append(r_vec)
        t_vecs.append(t_vec)

    image_show = padded.copy()
    draw_face_box(image_show, bboxes_decoded, landmarks, scores)

    # Initialize drowsiness detection flags
    is_yawning = False
    eyes_closed = False

    for i, (mesh_landmark, r_vec, t_vec) in enumerate(zip(mesh_landmarks_inverse, r_vecs, t_vecs)):
        mouth_ratio = get_mouth_ratio(mesh_landmark, image_show)
        left_box, right_box = get_eye_boxes(mesh_landmark, padded.shape)

        left_eye_img = padded[left_box[0][1]:left_box[1][1], left_box[0][0]:left_box[1][0]]
        right_eye_img = padded[right_box[0][1]:right_box[1][1], right_box[0][0]:right_box[1][0]]
        left_eye_landmarks, left_iris_landmarks = eye_mesher.inference(left_eye_img)
        right_eye_landmarks, right_iris_landmarks = eye_mesher.inference(right_eye_img)

        left_eye_ratio = get_eye_ratio(left_eye_landmarks, image_show, left_box[0])
        right_eye_ratio = get_eye_ratio(right_eye_landmarks, image_show, right_box[0])

        pitch, roll, yaw = get_face_angle(r_vec, t_vec)
        iris_ratio = get_iris_ratio(left_eye_landmarks, right_eye_landmarks)

        # === Yawning Detection with Buzzer Control ===
        if mouth_ratio > 0.3:
            is_yawning = True
            cv2.putText(image_show, "Yawning: Detected (BUZZER ON)", (padded_size[2] + 70, padded_size[0] + 70),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 0), thickness=2)
        else:
            cv2.putText(image_show, "Yawning: No", (padded_size[2] + 70, padded_size[0] + 70),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2)

        # === Eye Close Detection with Buzzer Control ===
        if left_eye_ratio < 0.2 and right_eye_ratio < 0.2:
            eyes_closed = True
            cv2.putText(image_show, "Eye: Closed (BUZZER ON)", (padded_size[2] + 70, padded_size[0] + 100),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(255, 0, 0), thickness=2)
            
            # Keep the original TTS alert logic
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            elif (time.time() - eye_closed_start_time > 0.5) and not eye_alert_triggered:
                speak_wake_up()
                eye_alert_triggered = True
        else:
            cv2.putText(image_show, "Eye: Open", (padded_size[2] + 70, padded_size[0] + 100),
                  fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=(0, 255, 0), thickness=2)
            eye_closed_start_time = None
            eye_alert_triggered = False

        # === Control Buzzer Based on Drowsiness Detection ===
        drowsiness_detected = is_yawning or eyes_closed
        if buzzer_initialized:
            control_buzzer(drowsiness_detected)

        # === Face Orientation Detection with LED Control ===
        face_direction = None
        
        if yaw > 15 and iris_ratio > 1.15:
            face_direction = "Left"
            cv2.putText(image_show, "Face: Left (Green LED)",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 255, 0], thickness=2)
        elif yaw < -15 and iris_ratio < 0.85:
            face_direction = "Right"
            cv2.putText(image_show, "Face: Right (Blue LED)",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 0, 255], thickness=2)
        elif pitch > 30:
            face_direction = "UP"
            cv2.putText(image_show, "Face: UP (Red LED)",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 0, 0], thickness=2)
        elif pitch < -13:
            face_direction = "Down"
            cv2.putText(image_show, "Face: Down (Red LED)",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[255, 0, 0], thickness=2)
        else:
            face_direction = "Forward"
            cv2.putText(image_show, "Face: Forward (LEDs OFF)",(padded_size[2] + 70, padded_size[0] + 130),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=[0, 255, 0], thickness=2)

        # Control LEDs based on face direction
        control_leds_by_face_direction(face_direction)

        # === Display Buzzer Status ===
        buzzer_status = "BUZZER: ON" if buzzer_active else "BUZZER: OFF"
        buzzer_color = (255, 0, 0) if buzzer_active else (0, 255, 0)
        cv2.putText(image_show, buzzer_status, (padded_size[2] + 70, padded_size[0] + 160),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, color=buzzer_color, thickness=2)

    image_show = image_show[padded_size[0]:target_dim - padded_size[1], padded_size[2]:target_dim - padded_size[3]]
    return image_show

# === Frame Loop ===
try:
    print("Starting drowsiness detection with buzzer alert...")
    print("Press 'q' to quit")
    
    while ret:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_show = main(image)
        result = cv2.cvtColor(image_show, cv2.COLOR_RGB2BGR)
        cv2.imshow('Drowsiness Detection with Buzzer', result)

        ret, image = cap.read()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nExiting...")

finally:
    # Cleanup: Turn off all LEDs and buzzer when exiting
    turn_off_all_leds()
    cleanup_buzzer()
    time.sleep(2)
    cap.release()
    cv2.destroyAllWindows()
    print("All resources cleaned up successfully.")

