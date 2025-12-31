#!/usr/bin/env python3
# Copyright 2020-2023 NXP
# SPDX-License-Identifier: Apache-2.0

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

class GestureMotorController:
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
        self.motor_direction = 1  # 1 for forward, -1 for backward
        self.motor_speed = 0.002  # Default delay between steps
        self.motor_thread = None
        
        # Gesture recognition variables
        self.last_gesture = "unknown"
        self.gesture_stability_count = 0
        self.gesture_threshold = 5  # Frames to confirm gesture
        
    def setup_motor(self):
        """Initialize stepper motor GPIO pins"""
        try:
            # GPIO chip and line offsets for motor IN1-IN4
            self.chip = gpiod.Chip('gpiochip1')  # GPIO2 bank
            self.pins = [0, 1, 2, 3]  # Offsets for IN3, IN1, IN2, IN4
            
            # Motor step sequence (full-step)
            self.step_seq = [
                [1,0,1,0],
                [0,1,1,0], 
                [0,1,0,1],
                [1,0,0,1]
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
            
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_gesture(self, landmarks):
        """Detect hand gestures based on landmark positions"""
        if landmarks is None or len(landmarks) != 21:
            return "no_hand"
            
        # Key landmarks for gesture detection
        wrist = landmarks[0]           # Wrist
        thumb_tip = landmarks[4]       # Thumb tip
        index_tip = landmarks[8]       # Index finger tip
        middle_tip = landmarks[12]     # Middle finger tip
        ring_tip = landmarks[16]       # Ring finger tip
        pinky_tip = landmarks[20]      # Pinky finger tip
        
        # MCP joints (base of fingers)
        index_mcp = landmarks[5]       # Index MCP
        middle_mcp = landmarks[9]      # Middle MCP
        ring_mcp = landmarks[13]       # Ring MCP
        pinky_mcp = landmarks[17]      # Pinky MCP
        
        # Calculate if fingers are extended
        finger_extended = []
        
        # Thumb (different logic due to orientation)
        thumb_extended = thumb_tip[0] > landmarks[3][0]  # Thumb IP joint
        finger_extended.append(thumb_extended)
        
        # Other fingers (tip above MCP joint)
        for tip, mcp in [(index_tip, index_mcp), (middle_tip, middle_mcp), 
                        (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)]:
            finger_extended.append(tip[1] < mcp[1])  # Y coordinate decreases upward
            
        extended_count = sum(finger_extended)
        
        # Gesture Recognition Logic
        if extended_count == 5:
            return "open_palm"  # All fingers extended
        elif extended_count == 0:
            return "closed_fist"  # All fingers closed
        elif extended_count == 1 and finger_extended[1]:  # Only index finger
            return "point_forward"
        elif extended_count == 2 and finger_extended[1] and finger_extended[2]:  # Peace sign
            return "point_backward"
        elif extended_count == 1 and finger_extended[0]:  # Only thumb
            return "speed_up"
        elif extended_count == 1 and finger_extended[4]:  # Only pinky
            return "speed_down"
        else:
            return "unknown"
    
    def step_motor(self, steps, delay_time, direction=1):
        """Move motor specified number of steps"""
        if not self.lines:
            return
            
        seq_len = len(self.step_seq)
        for i in range(steps):
            if not self.motor_running:
                break
                
            if direction > 0:
                step = self.step_seq[i % seq_len]
            else:
                step = self.step_seq[-(i % seq_len)]
                
            for line, val in zip(self.lines, step):
                line.set_value(val)
            time.sleep(delay_time)
    
    def motor_control_thread(self):
        """Continuous motor control in separate thread"""
        while self.motor_running:
            self.step_motor(100, self.motor_speed, self.motor_direction)
    
    def start_motor(self):
        """Start motor in continuous mode"""
        if not self.motor_running and self.lines:
            self.motor_running = True
            self.motor_thread = threading.Thread(target=self.motor_control_thread)
            self.motor_thread.daemon = True
            self.motor_thread.start()
            print(f"Motor started - Direction: {'Forward' if self.motor_direction > 0 else 'Backward'}, Speed: {1/self.motor_speed:.0f} steps/sec")
    
    def stop_motor(self):
        """Stop motor and turn off all coils"""
        self.motor_running = False
        if self.motor_thread and self.motor_thread.is_alive():
            self.motor_thread.join(timeout=1.0)
        
        # Turn off all motor coils
        if self.lines:
            for line in self.lines:
                line.set_value(0)
        print("Motor stopped")
    
    def process_gesture(self, gesture):
        """Process detected gesture and control motor accordingly"""
        if gesture != self.last_gesture:
            self.gesture_stability_count = 0
        else:
            self.gesture_stability_count += 1
            
        # Only act on stable gestures
        if self.gesture_stability_count >= self.gesture_threshold:
            if gesture == "open_palm":
                self.motor_direction = 1
                self.start_motor()
            elif gesture == "point_forward":
                self.motor_direction = 1
                self.motor_speed = 0.001  # Faster
                self.start_motor()
            elif gesture == "point_backward":
                self.motor_direction = -1
                self.start_motor()
            elif gesture == "speed_up":
                self.motor_speed = max(0.0005, self.motor_speed * 0.8)
                print(f"Speed increased: {1/self.motor_speed:.0f} steps/sec")
            elif gesture == "speed_down":
                self.motor_speed = min(0.01, self.motor_speed * 1.2)
                print(f"Speed decreased: {1/self.motor_speed:.0f} steps/sec")
            elif gesture in ["closed_fist", "no_hand"]:
                self.stop_motor()
        
        self.last_gesture = gesture
    
    def draw_landmarks_and_info(self, points, frame, gesture):
        """Draw hand landmarks and gesture information on frame"""
        # Hand landmark connections
        connections = [
            (5, 6), (6, 7), (7, 8),      # Index finger
            (9, 10), (10, 11), (11, 12), # Middle finger
            (13, 14), (14, 15), (15, 16), # Ring finger
            (17, 18), (18, 19), (19, 20), # Pinky finger
            (0, 5), (5, 9), (9, 13), (13, 17), # Palm
            (0, 17), (0, 1), (1, 2), (2, 3), (3, 4)  # Thumb
        ]
        
        if points is not None:
            # Draw landmarks
            for i, point in enumerate(points):
                x, y = int(point[0]), int(point[1])
                color = (0, 255, 0) if i == 0 else (255, 0, 0)  # Wrist in green
                cv2.circle(frame, (x, y), 4, color, 2)
                cv2.putText(frame, str(i), (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            # Draw connections
            for connection in connections:
                x0, y0 = int(points[connection[0]][0]), int(points[connection[0]][1])
                x1, y1 = int(points[connection[1]][0]), int(points[connection[1]][1])
                cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
        
        # Draw gesture and motor status information
        info_y = 30
        cv2.putText(frame, f"Gesture: {gesture}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        motor_status = "Running" if self.motor_running else "Stopped"
        direction = "Forward" if self.motor_direction > 0 else "Backward"
        cv2.putText(frame, f"Motor: {motor_status} ({direction})", (10, info_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        speed_info = f"Speed: {1/self.motor_speed:.0f} steps/sec"
        cv2.putText(frame, speed_info, (10, info_y + 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw gesture guide
        guide_y = frame.shape[0] - 150
        cv2.putText(frame, "Gesture Guide:", (10, guide_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Open Palm - Forward", (10, guide_y + 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Point (1 finger) - Fast Forward", (10, guide_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Peace (2 fingers) - Backward", (10, guide_y + 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Thumb Up - Speed Up", (10, guide_y + 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Pinky - Speed Down", (10, guide_y + 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Closed Fist - Stop", (10, guide_y + 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
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

    # Initialize gesture motor controller
    controller = GestureMotorController(args.delegate)
    
    print("Hand Gesture Motor Control Started")
    print("Press 'q' to quit")
    
    try:
        while ret:
            # Convert BGR to RGB for hand tracking
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hand landmarks
            points, _ = controller.detector(image)
            
            # Detect gesture
            gesture = controller.detect_gesture(points)
            
            # Process gesture for motor control
            controller.process_gesture(gesture)
            
            # Draw landmarks and information
            controller.draw_landmarks_and_info(points, frame, gesture)
            
            # Display frame
            cv2.imshow("Hand Gesture Motor Control", frame)
            
            # Read next frame
            ret, frame = capture.read()
            
            # Check for exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Cleanup
        controller.cleanup()
        capture.release()
        cv2.destroyAllWindows()
        print("Application closed")

if __name__ == "__main__":
    main()

