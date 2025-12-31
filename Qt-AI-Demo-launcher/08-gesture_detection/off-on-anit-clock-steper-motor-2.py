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

        # Display settings
        self.display_scale = 1.5  # Scale factor for display size
        self.window_name = "Enhanced Hand Gesture Motor Control"
        
        # Store original frame dimensions for coordinate mapping
        self.original_frame_size = None
        self.display_frame_size = None

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

    def transform_landmarks_to_display(self, landmarks, original_shape, display_shape):
        """Transform landmark coordinates from original frame to display frame"""
        if landmarks is None:
            return None
            
        original_h, original_w = original_shape[:2]
        display_h, display_w = display_shape[:2]
        
        # Calculate scaling factors
        scale_x = display_w / original_w
        scale_y = display_h / original_h
        
        # Transform each landmark
        transformed_landmarks = []
        for landmark in landmarks:
            # Ensure landmark coordinates are within original frame bounds
            x = max(0, min(landmark[0], original_w - 1))
            y = max(0, min(landmark[1], original_h - 1))
            
            # Scale to display coordinates
            display_x = int(x * scale_x)
            display_y = int(y * scale_y)
            
            transformed_landmarks.append([display_x, display_y])
        
        return transformed_landmarks

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

        # Helper: extended if tip above joint with margin (adjusted for better accuracy)
        def is_extended(tip_y, joint_y, margin=15):
            return tip_y < (joint_y - margin)

        # Determine extension with improved threshold
        thumb_extended = abs(thumb_tip[0] - thumb_ip[0]) > 25  # lateral spread
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

    def draw_landmarks_and_info(self, points, frame, gesture, original_frame_shape):
        """Draw hand landmarks and gesture information with proper coordinate transformation"""
        # MediaPipe hand landmark connections (correct anatomical structure)
        connections = [
            # Thumb
            (0, 1), (1, 2), (2, 3), (3, 4),
            # Index finger  
            (0, 5), (5, 6), (6, 7), (7, 8),
            # Middle finger
            (0, 9), (9, 10), (10, 11), (11, 12),
            # Ring finger
            (0, 13), (13, 14), (14, 15), (15, 16),
            # Pinky
            (0, 17), (17, 18), (18, 19), (19, 20),
            # Palm connections
            (5, 9), (9, 13), (13, 17)
        ]

        # Transform landmarks to display coordinates
        display_points = self.transform_landmarks_to_display(points, original_frame_shape, frame.shape)
        
        # Scale factors for larger display
        circle_radius = max(2, int(4 * self.display_scale))
        line_thickness = max(1, int(2 * self.display_scale))
        text_scale = 0.8 * self.display_scale
        text_thickness = max(1, int(2 * self.display_scale))

        if display_points is not None and len(display_points) == 21:
            # Draw connections first (lines)
            for connection in connections:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(display_points) and pt2_idx < len(display_points):
                    x0, y0 = display_points[pt1_idx]
                    x1, y1 = display_points[pt2_idx]
                    
                    # Ensure coordinates are within frame bounds
                    x0 = max(0, min(x0, frame.shape[1] - 1))
                    y0 = max(0, min(y0, frame.shape[0] - 1))
                    x1 = max(0, min(x1, frame.shape[1] - 1))
                    y1 = max(0, min(y1, frame.shape[0] - 1))
                    
                    # Color coding for MRP gesture
                    if gesture == "mrp_only" and connection in [(9,10),(10,11),(11,12),(13,14),(14,15),(15,16),(17,18),(18,19),(19,20)]:
                        line_color = (0, 255, 255)  # Yellow for highlighted fingers
                    else:
                        line_color = (255, 0, 255)  # Magenta for other connections
                    
                    cv2.line(frame, (x0, y0), (x1, y1), line_color, line_thickness)

            # Draw landmark points on top of lines
            for i, point in enumerate(display_points):
                x, y = point
                
                # Ensure coordinates are within frame bounds
                x = max(circle_radius, min(x, frame.shape[1] - circle_radius))
                y = max(circle_radius, min(y, frame.shape[0] - circle_radius))
                
                # Color coding for different landmarks
                if gesture == "mrp_only" and i in (12, 16, 20):  # Fingertips for MRP
                    color = (0, 255, 255)  # Yellow
                elif i == 0:  # Wrist
                    color = (0, 255, 0)  # Green
                elif i in (4, 8, 12, 16, 20):  # All fingertips
                    color = (255, 0, 0)  # Blue
                else:  # Joint points
                    color = (0, 165, 255)  # Orange
                
                cv2.circle(frame, (x, y), circle_radius, color, -1)
                
                # Draw landmark numbers for debugging (optional)
                if self.display_scale > 1.2:  # Only show numbers when zoomed in
                    cv2.putText(frame, str(i), (x + 5, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3 * self.display_scale, 
                               (255, 255, 255), 1)

        # Display gesture and motor information
        self.draw_status_info(frame, gesture)

    def draw_status_info(self, frame, gesture):
        """Draw status information on frame"""
        # Scaled text positioning
        info_y = int(50 * self.display_scale)
        line_spacing = int(60 * self.display_scale)
        text_scale = 1.0 * self.display_scale
        text_thickness = max(1, int(2 * self.display_scale))
        
        if gesture == "open_palm":
            gesture_color = (0, 255, 0)
            gesture_text = "Open Palm"
        elif gesture == "closed_fist":
            gesture_color = (0, 0, 255)
            gesture_text = "Closed Fist"
        elif gesture == "mrp_only":
            gesture_color = (0, 255, 255)
            gesture_text = "Middle+Ring+Pinky"
        elif gesture == "no_hand":
            gesture_color = (128, 128, 128)
            gesture_text = "No Hand Detected"
        else:
            gesture_color = (255, 255, 255)
            gesture_text = "Unknown"

        cv2.putText(frame, f"Gesture: {gesture_text}", (20, info_y),
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, gesture_color, text_thickness)

        if self.motor_running:
            motor_status = f"ON ({self.motor_direction.upper()})"
            motor_color = (0, 255, 0)
        else:
            motor_status = "OFF"
            motor_color = (0, 0, 255)

        cv2.putText(frame, f"Motor: {motor_status}", (20, info_y + line_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, text_scale, motor_color, text_thickness)

        # Control guide at bottom
        guide_y = frame.shape[0] - int(140 * self.display_scale)
        guide_spacing = int(35 * self.display_scale)
        guide_text_scale = 0.6 * self.display_scale
        guide_thickness = max(1, int(2 * self.display_scale))
        
        cv2.putText(frame, "Controls:", (20, guide_y),
                   cv2.FONT_HERSHEY_SIMPLEX, guide_text_scale, (255, 255, 255), guide_thickness)
        cv2.putText(frame, "Open Palm = Motor Clockwise", (20, guide_y + guide_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, guide_text_scale, (0, 255, 0), guide_thickness)
        cv2.putText(frame, "MRP (Middle+Ring+Pinky) = Motor Anti-Clockwise", (20, guide_y + 2*guide_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, guide_text_scale, (0, 255, 255), guide_thickness)
        cv2.putText(frame, "Closed Fist = Motor Stop", (20, guide_y + 3*guide_spacing),
                   cv2.FONT_HERSHEY_SIMPLEX, guide_text_scale, (0, 0, 255), guide_thickness)

    def resize_frame(self, frame):
        """Resize frame based on display scale"""
        if self.display_scale != 1.0:
            height, width = frame.shape[:2]
            new_width = int(width * self.display_scale)
            new_height = int(height * self.display_scale)
            frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        return frame

    def setup_window(self, fullscreen=False):
        """Setup display window with resizable or fullscreen options"""
        if fullscreen:
            cv2.namedWindow(self.window_name, cv2.WND_PROP_FULLSCREEN)
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            # Set initial window size
            cv2.resizeWindow(self.window_name, 1200, 900)

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
    parser.add_argument('-s', '--scale', type=float, default=1.5,
                       help='display scale factor (default: 1.5)')
    parser.add_argument('-f', '--fullscreen', action='store_true',
                       help='start in fullscreen mode')
    parser.add_argument('--window-size', nargs=2, type=int, metavar=('WIDTH', 'HEIGHT'),
                       help='set custom window size (e.g., --window-size 1200 900)')
    args = parser.parse_args()

    # Initialize camera
    if args.input.isdigit():
        cap_input = int(args.input)
    else:
        cap_input = args.input

    capture = cv2.VideoCapture(cap_input)
    
    # Set camera properties for better hand detection
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    
    ret, frame = capture.read()

    if frame is None:
        print(f"Can't read frame from source: {args.input}")
        return

    controller = EnhancedGestureMotorController(args.delegate)
    controller.display_scale = args.scale

    # Store original frame size for coordinate transformation
    controller.original_frame_size = frame.shape

    # Setup window
    controller.setup_window(args.fullscreen)
    
    # Set custom window size if specified
    if args.window_size:
        cv2.resizeWindow(controller.window_name, args.window_size[0], args.window_size[1])

    print("Enhanced Hand Gesture Motor Control Started")
    print("Hand skeleton should now properly align with your hand")
    print("Open Palm = Motor Clockwise | MRP (Middle+Ring+Pinky) = Motor Anti-Clockwise | Closed Fist = Motor Stop")
    print("Controls: 'q' to quit, 'f' to toggle fullscreen, '+'/'-' to scale up/down")

    fullscreen_mode = args.fullscreen

    try:
        while ret:
            # Convert to RGB for hand detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            points, _ = controller.detector(image)

            # Detect gesture using original coordinates
            gesture = controller.detect_gesture(points)
            controller.process_gesture(gesture)

            # Create display frame
            display_frame = controller.resize_frame(frame.copy())
            
            # Draw landmarks with proper coordinate transformation
            controller.draw_landmarks_and_info(points, display_frame, gesture, frame.shape)
            
            cv2.imshow(controller.window_name, display_frame)

            ret, frame = capture.read()

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen_mode = not fullscreen_mode
                cv2.destroyWindow(controller.window_name)
                controller.setup_window(fullscreen_mode)
            elif key == ord('+') or key == ord('='):  # Scale up
                controller.display_scale = min(controller.display_scale + 0.1, 3.0)
                print(f"Display scale: {controller.display_scale:.1f}")
            elif key == ord('-') or key == ord('_'):  # Scale down
                controller.display_scale = max(controller.display_scale - 0.1, 0.5)
                print(f"Display scale: {controller.display_scale:.1f}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        controller.cleanup()
        capture.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()

