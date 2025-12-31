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

class SimpleGestureMotorController:
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
    
    def detect_gesture(self, landmarks):
        """Detect only open palm or closed fist gestures"""
        if landmarks is None or len(landmarks) != 21:
            return "no_hand"
            
        # Key landmarks for simple gesture detection
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
        
        # Calculate if fingers are extended (simple check)
        fingers_extended = 0
        
        # Thumb (check if tip is far from palm)
        if thumb_tip[0] > landmarks[3][0]:  # Thumb IP joint
            fingers_extended += 1
        
        # Other fingers (tip above MCP joint)
        for tip, mcp in [(index_tip, index_mcp), (middle_tip, middle_mcp), 
                        (ring_tip, ring_mcp), (pinky_tip, pinky_mcp)]:
            if tip[1] < mcp[1]:  # Y coordinate decreases upward
                fingers_extended += 1
        
        # Simple gesture classification
        if fingers_extended >= 4:  # Most fingers extended
            return "open_palm"
        elif fingers_extended <= 1:  # Most fingers closed
            return "closed_fist"
        else:
            return "unknown"
    
    def step_motor(self, steps, delay_time):
        """Move motor specified number of steps"""
        if not self.lines:
            return
            
        seq_len = len(self.step_seq)
        for i in range(steps):
            if not self.motor_running:
                break
                
            step = self.step_seq[i % seq_len]
            for line, val in zip(self.lines, step):
                line.set_value(val)
            time.sleep(delay_time)
    
    def motor_control_thread(self):
        """Continuous motor control in separate thread"""
        while self.motor_running:
            self.step_motor(100, self.motor_speed)
    
    def start_motor(self):
        """Start motor in continuous mode"""
        if not self.motor_running and self.lines:
            self.motor_running = True
            self.motor_thread = threading.Thread(target=self.motor_control_thread)
            self.motor_thread.daemon = True
            self.motor_thread.start()
            print("Motor started (Open Palm detected)")
    
    def stop_motor(self):
        """Stop motor and turn off all coils"""
        if self.motor_running:
            self.motor_running = False
            if self.motor_thread and self.motor_thread.is_alive():
                self.motor_thread.join(timeout=1.0)
            
            # Turn off all motor coils
            if self.lines:
                for line in self.lines:
                    line.set_value(0)
            print("Motor stopped (Closed Fist detected)")
    
    def process_gesture(self, gesture):
        """Process detected gesture - only open palm and closed fist"""
        if gesture != self.last_gesture:
            self.gesture_stability_count = 0
        else:
            self.gesture_stability_count += 1
            
        # Only act on stable gestures
        if self.gesture_stability_count >= self.gesture_threshold:
            if gesture == "open_palm" and not self.motor_running:
                self.start_motor()
            elif gesture == "closed_fist" and self.motor_running:
                self.stop_motor()
        
        self.last_gesture = gesture
    
    def draw_landmarks_and_info(self, points, frame, gesture):
        """Draw hand landmarks and simple gesture information"""
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
            
            # Draw connections
            for connection in connections:
                x0, y0 = int(points[connection[0]][0]), int(points[connection[0]][1])
                x1, y1 = int(points[connection[1]][0]), int(points[connection[1]][1])
                cv2.line(frame, (x0, y0), (x1, y1), (255, 0, 255), 2)
        
        # Draw gesture and motor status - simple display
        info_y = 30
        gesture_color = (0, 255, 0) if gesture == "open_palm" else (0, 0, 255) if gesture == "closed_fist" else (255, 255, 255)
        cv2.putText(frame, f"Gesture: {gesture.replace('_', ' ').title()}", (10, info_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, gesture_color, 2)
        
        motor_status = "ON" if self.motor_running else "OFF"
        motor_color = (0, 255, 0) if self.motor_running else (0, 0, 255)
        cv2.putText(frame, f"Motor: {motor_status}", (10, info_y + 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, motor_color, 2)
        
        # Simple control guide
        cv2.putText(frame, "Open Palm = Motor ON", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, "Closed Fist = Motor OFF", (10, frame.shape[0] - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
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

    # Initialize simple gesture motor controller
    controller = SimpleGestureMotorController(args.delegate)
    
    print("Simple Hand Gesture Motor Control Started")
    print("Open Palm = Motor ON | Closed Fist = Motor OFF")
    print("Press 'q' to quit")
    
    try:
        while ret:
            # Convert BGR to RGB for hand tracking
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect hand landmarks
            points, _ = controller.detector(image)
            
            # Detect simple gesture (only open palm or closed fist)
            gesture = controller.detect_gesture(points)
            
            # Process gesture for motor control
            controller.process_gesture(gesture)
            
            # Draw landmarks and information
            controller.draw_landmarks_and_info(points, frame, gesture)
            
            # Display frame
            cv2.imshow("Simple Hand Gesture Motor Control", frame)
            
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

