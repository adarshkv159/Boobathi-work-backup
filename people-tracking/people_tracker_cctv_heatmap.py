#!/usr/bin/env python3
"""
TFLite People Tracking + HeatMap (RTSP IP Camera)
LOW LATENCY VERSION (Dahua Tested)
"""

import os
import argparse
import time
import math
import numpy as np
import cv2

# RTSP LOW-LATENCY SETTINGS

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|"
    "fflags;nobuffer|"
    "flags;low_delay|"
    "probesize;32|"
    "analyzeduration;0|"
    "buffer_size;102400"
)

RTSP_URL = (
    "rtsp://admin:admin123@192.168.2.34:554/"
    "cam/realmonitor?channel=1&subtype=1&unicast=true&proto=Onvif"
)

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    from tensorflow.lite import Interpreter as tflite


from norfair import Detection, Tracker
from norfair.tracker import TrackedObject


class HeatMap:
    def __init__(self, shape, radius=20):
        self.radius = radius
        self.map_shape = shape
        self.map = np.zeros(self.map_shape, dtype=np.float32)

        # gradient circle
        size = radius * 2
        circle = np.zeros((size, size), np.float32)
        circle = cv2.circle(circle, (radius - 1, radius - 1), radius, 1, -1)

        grad = np.zeros_like(circle)
        for y in range(size):
            for x in range(size):
                grad[y, x] = math.sqrt((radius - x) ** 2 + (radius - y) ** 2)

        grad = grad.max() - grad
        circle = circle * grad
        self.circle = cv2.normalize(circle, None, 0, 1, cv2.NORM_MINMAX)

    def update(self, objects):
        for obj in objects:
            if isinstance(obj, TrackedObject):
                pt = np.mean(obj.estimate, axis=0).astype(int)
            else:
                continue

            cx, cy = int(pt[0]), int(pt[1])
            x1 = cx - self.radius
            y1 = cy - self.radius
            x2 = cx + self.radius
            y2 = cy + self.radius

            if x2 < 0 or y2 < 0 or x1 >= self.map_shape[1] or y1 >= self.map_shape[0]:
                continue

            cx1 = max(0, -x1)
            cy1 = max(0, -y1)
            cx2 = self.circle.shape[1] - max(0, x2 - self.map_shape[1])
            cy2 = self.circle.shape[0] - max(0, y2 - self.map_shape[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(self.map_shape[1], x2)
            y2 = min(self.map_shape[0], y2)

            self.map[y1:y2, x1:x2] += self.circle[cy1:cy2, cx1:cx2]

    def draw(self, frame):
        heat = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX)
        heat = heat.astype(np.uint8)
        heat = cv2.equalizeHist(heat)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.5, heat_color, 0.5, 0)


def preprocess(frame, input_shape, dtype):
    h, w = input_shape[1], input_shape[2]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if dtype == np.uint8:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)

def parse_outputs(outputs):
    boxes = classes = scores = None
    for v in outputs.values():
        a = np.asarray(v)
        if a.ndim == 3 and a.shape[-1] == 4:
            boxes = a
        elif a.ndim == 2:
            if np.max(a) <= 1.0:
                scores = a
            else:
                classes = a
    return boxes, classes, scores

def extract_persons(boxes, classes, scores, frame_shape, th, person_id):
    h, w = frame_shape
    dets = []

    if boxes is None:
        return dets

    boxes = boxes.squeeze()
    classes = classes.squeeze()
    scores = scores.squeeze()

    for i in range(len(scores)):
        if scores[i] < th or int(classes[i]) != person_id:
            continue

        y1, x1, y2, x2 = boxes[i]
        dets.append([
            int(x1 * w), int(y1 * h),
            int(x2 * w), int(y2 * h),
            float(scores[i])
        ])
    return dets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--person_id", type=int, default=0)
    parser.add_argument("--show_fps", action="store_true")
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise RuntimeError("❌ RTSP open failed")

    tracker = Tracker(distance_function="iou", distance_threshold=0.7)

    ret, frame = cap.read()
    heatmap = HeatMap(frame.shape[:2], radius=25)

    prev = time.time()
    fps = 0.0

    while True:
        for _ in range(3):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        inp = preprocess(frame, input_details[0]["shape"],
                         np.dtype(input_details[0]["dtype"]))

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()

        outputs = {od["name"]: interpreter.get_tensor(od["index"])
                   for od in output_details}

        boxes, classes, scores = parse_outputs(outputs)
        dets = extract_persons(boxes, classes, scores,
                               frame.shape[:2],
                               args.threshold,
                               args.person_id)

        nf_dets = []
        for x1, y1, x2, y2, sc in dets:
            pts = np.array([[x1, y1], [x2, y2]], np.float32)
            scs = np.array([sc, sc], np.float32)
            nf_dets.append(Detection(points=pts, scores=scs))

        tracks = tracker.update(nf_dets)

        #  HEATMAP UPDATE
        heatmap.update(tracks)
        out = heatmap.draw(frame)

        for t in tracks:
            cx, cy = map(int, t.estimate[0])
            cv2.circle(out, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(out, f"ID {t.id}", (cx + 6, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if args.show_fps:
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(out, f"FPS {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("People Tracking + HeatMap", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

