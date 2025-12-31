import cv2
import numpy as np
import time
import argparse
import threading
from hand_tracker import HandTracker
import gpiod

# Hand Tracking Model Paths
PALM_MODEL_PATH = "palm_detection_builtin_256_integer_quant.tflite"
LANDMARK_MODEL_PATH = "hand_landmark_3d_256_integer_quant.tflite"
ANCHORS_PATH = "anchors.csv"

class EnhancedGestureMotorController:
    def __init__(self, delegate_path=''):
        # Initialize Hand Tracker
        self.detector = HandTracker(
            PALM_MODEL_PATH,
            LANDMARK_MODEL_PATH,
            ANCHORS_PATH,
            delegate_path,
            box_shift=0.2,
            box_enlarge=1.3
        )

        # Initialize GPIO for Stepper Motor
        self.setup_motor()

        # Motor control state
        self.motor_running = False
        self.motor_direction = "clockwise"  # "clockwise" or "anticlockwise"
        self.motor_speed = 0.002  # Fixed delay between steps
        self.motor_thread = None

        # Gesture recognition variables
        self.last_gesture = "unknown"
        self.gesture_stability_count = 0
        self.gesture_threshold = 3  # Frames to confirm gesture

    def setup_motor(self):
        """Initialize stepper motor GPIO pins"""
        try:
            # GPIO chip and line offsets for motor IN1-IN4
            self.chip = gpiod.Chip('gpiochip1')  # GPIO2 bank
            self.pins = [0, 1, 2, 3]  # Offsets for IN3, IN1, IN2, IN4

            # Motor step sequences
            # Clockwise sequence (full-step)
            self.step_seq_cw = [
                [1,0,1,0],
                [0,1,1,0],
                [0,1,0,1],
                [1,0,0,1]
            ]

            # Anti-clockwise sequence (reverse order)
            self.step_seq_ccw = [
                [1,0,0,1],
                [0,1,0,1],
                [0,1,1,0],
                [1,0,1,0]
            ]

            # Initialize GPIO lines
            self.lines = []
            for pin in self.pins:
                line = self.chip.get_line(pin)
                line.request(consumer='stepper', type=gpiod.LINE_REQ_DIR_OUT, default_vals=[0])
                self.lines.append(line)

            print("Stepper motor GPIO initialized successfully")

        except Exception as e:
            print(f"Error initializing motor GPIO: {e}")
            self.lines = None

    def detect_gesture(self, landmarks):
        """Detect open palm, closed fist, and MRP-only (middle+ring+pinky extended)"""
        if landmarks is None or len(landmarks) != 21:
            return "no_hand"

        # Key landmarks
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]

        index_tip = landmarks[8]
        index_pip = landmarks[6]

        middle_tip = landmarks[12]
        middle_pip = landmarks[10]

        ring_tip = landmarks[16]
        ring_pip = landmarks[14]

        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]

        # Helper: extended if tip above joint with margin
        def is_extended(tip_y, joint_y, margin=20):
            return tip_y < (joint_y - margin)

        # Determine extension
        thumb_extended = abs(thumb_tip[0] - thumb_ip[0]) > 30  # lateral spread
        index_extended = is_extended(index_tip[1], index_pip[1])
        middle_extended = is_extended(middle_tip[1], middle_pip[1])
        ring_extended = is_extended(ring_tip[1], ring_pip[1])
        pinky_extended = is_extended(pinky_tip[1], pinky_pip[1])

        extended_count = sum([thumb_extended, index_extended, middle_extended, ring_extended, pinky_extended])

        # New gesture: only middle, ring, and pinky extended (index folded). Thumb ignored.
        mrp_only = (not index_extended) and middle_extended and ring_extended and pinky_extended

        if mrp_only:
            return "mrp_only"
        if extended_count >= 4:
            return "open_palm"
        elif extended_count <= 1:
            return "closed_fist"
        else:
            return "unknown"

    def step_motor(self, steps, delay_time, direction="clockwise"):
        """Move motor specified number of steps in given direction"""
        if not self.lines:
            return

        step_seq = self.step_seq_cw if direction == "clockwise" else self.step_seq_ccw
        seq_len = len(step_seq)

        for i in range(steps):
            if not self.motor_running:
                break
            step = step_seq[i % seq_len]
            for line, val in zip(self.lines, step):
                line.set_value(val)
            time.sleep(delay_time)

    def motor_control_thread(self):
        """Continuous motor control in separate thread"""
        while self.motor_running:
            self.step_motor(100, self.motor_speed, self.motor_direction)

    def start_motor(self, direction="clockwise"):
        """Start motor in continuous mode with specified direction"""
        if not self.motor_running and self.lines:
            self.motor_running = True
            self.motor_direction = direction
            self.motor_thread = threading.Thread(target=self.motor_control_thread)
            self.motor_thread.daemon = True
            self.motor_thread.start()

            direction_text = "Clockwise" if direction == "clockwise" else "Anti-Clockwise"
            print(f"Motor started {direction_text}")

    def stop_motor(self):
        """Stop motor and turn off all coils"""
        if self.motor_running:
            self.motor_running = False
            if self.motor_thread and self.motor_thread.is_alive():
                self.motor_thread.join(timeout=1.0)

            if self.lines:
                for line in self.lines:
                    line.set_value(0)
            print("Motor stopped")

    def process_gesture(self, gesture):
        """Map gestures: open palm -> CW, MRP-only -> CCW, closed fist -> stop"""
        if gesture != self.last_gesture:
            self.gesture_stability_count = 0
        else:
            self.gesture_stability_count += 1

        if self.gesture_stability_count >= self.gesture_threshold:
            if gesture == "open_palm" and (not self.motor_running or self.motor_direction != "clockwise"):
                if self.motor_running:
                    self.stop_motor()
                    time.sleep(0.1)
                self.start_motor("clockwise")
            elif gesture == "mrp_only" and (not self.motor_running or self.motor_direction != "anticlockwise"):
                if self.motor_running:
                    self.stop_motor()
                    time.sleep(0.1)
                self.start_motor("anticlockwise")
            elif gesture == "closed_fist" and self.motor_running:
                self.stop_motor()

        self.last_gesture = gesture

    def draw_landmarks_and_info(self, points, frame, gesture):
        """Draw hand landmarks and gesture information"""
        connections = [
            (5, 6), (6, 7), (7, 8),      # Index finger
            (9, 10), (10, 11), (11, 12), # Middle finger
            (13, 14), (14, 15), (15, 16),# Ring finger
            (17, 18), (18, 19), (19, 20),# Pinky finger
            (0, 5), (5, 9), (9, 13), (13, 17), # Palm
            (0, 17), (0, 1), (1, 2), (2, 3), (3, 4)  # Thumb
        ]

        if points is not None:
            for i, point in enumerate(points):
                x, y = int(point[0]), int(point[1])
                # Highlight middle, ring, pinky tips when MRP gesture detected
                if gesture == "mrp_only" and i in (12, 16, 20):
                    color = (0, 255, 255)
                elif i == 0:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.circle(frame, (x, y), 4, color, 2)

            for connection in connections:
                x0, y0 = int(points[connection[0]][0]), int(points[connection[0]][1])
                x1, y1 = int(points[connection[1]][0]), int(points[connection[1]][1])
                if gesture == "mrp_only" and connection in [(9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]:
                    line_color = (0, 255, 255)
                else:
                    line_color = (255, 0, 255)
                cv2.line(frame, (x0, y0), (x1, y1), line_color, 2)

        info_y = 30
        if gesture == "open_palm":
            gesture_color = (0, 255, 0)
            gesture_text = "Open Palm"
        elif gesture == "closed_fist":
            gesture_color = (0, 0, 255)
            gesture_text = "Closed Fist"
        elif gesture == "mrp_only":
            gesture_color = (0, 255, 255)
            gesture_text = "Middle+Ring+Pinky"
        else:
            gesture_color = (255, 255, 255)
            gesture_text = "Unknown"

        cv2.putText(frame, f"Gesture: {gesture_text}", (10, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)

        if self.motor_running:
            motor_status = f"ON ({self.motor_direction.upper()})"
            motor_color = (0, 255, 0)
        else:
            motor_status = "OFF"
            motor_color = (0, 0, 255)

        cv2.putText(frame, f"Motor: {motor_status}", (10, info_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, motor_color, 2)

        guide_y = frame.shape[0] - 90
        cv2.putText(frame, "Controls:", (10, guide_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Open Palm = Motor Clockwise", (10, guide_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, "MRP (Middle+Ring+Pinky) = Motor Anti-Clockwise", (10, guide_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(frame, "Closed Fist = Motor Stop", (10, guide_y + 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    def cleanup(self):
        """Cleanup resources"""
        self.stop_motor()
        if self.lines:
            for line in self.lines:
                line.release()
        if hasattr(self, 'chip'):
            self.chip.close()
        print("Cleanup completed")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', default='/dev/video0',
                       help='input camera device')
    parser.add_argument('-d', '--delegate', default='',
                       help='TensorFlow Lite delegate path')
    args = parser.parse_args()

    # Initialize camera
    if args.input.isdigit():
        cap_input = int(args.input)
    else:
        cap_input = args.input

    capture = cv2.VideoCapture(cap_input)
    ret, frame = capture.read()

    if frame is None:
        print(f"Can't read frame from source: {args.input}")
        return

    controller = EnhancedGestureMotorController(args.delegate)

    print("Enhanced Hand Gesture Motor Control Started")
    print("Open Palm = Motor Clockwise | MRP (Middle+Ring+Pinky) = Motor Anti-Clockwise | Closed Fist = Motor Stop")
    print("Press 'q' to quit")

    try:
        while ret:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            points, _ = controller.detector(image)

            gesture = controller.detect_gesture(points)
            controller.process_gesture(gesture)

            controller.draw_landmarks_and_info(points, frame, gesture)
            cv2.imshow("Enhanced Hand Gesture Motor Control", frame)

            ret, frame = capture.read()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        controller.cleanup()
        capture.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()

