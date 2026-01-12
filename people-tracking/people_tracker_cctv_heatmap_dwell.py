#!/usr/bin/env python3
"""
People Tracking + HeatMap + Correct ID + Proper Dwell Time
"""

import os
import argparse
import time
import math
import numpy as np
import cv2


# RTSP LOW LATENCY

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
    "cam/realmonitor?channel=1&subtype=1"
)


try:
    import tflite_runtime.interpreter as tflite
except Exception:
    from tensorflow.lite import Interpreter as tflite


from norfair import Detection, Tracker
from norfair.tracker import TrackedObject


class HeatMap:
    def __init__(self, shape, radius=25):
        self.radius = radius
        self.map = np.zeros(shape[:2], np.float32)

        size = radius * 2
        circle = np.zeros((size, size), np.float32)
        circle = cv2.circle(circle, (radius - 1, radius - 1), radius, 1, -1)

        grad = np.zeros_like(circle)
        for y in range(size):
            for x in range(size):
                grad[y, x] = math.hypot(radius - x, radius - y)

        grad = grad.max() - grad
        self.circle = cv2.normalize(circle * grad, None, 0, 1, cv2.NORM_MINMAX)

    def update(self, tracks):
        for t in tracks:
            pts = np.array(t.estimate)
            cx = int((pts[0][0] + pts[1][0]) / 2)
            cy = int((pts[0][1] + pts[1][1]) / 2)

            x1, y1 = cx - self.radius, cy - self.radius
            x2, y2 = cx + self.radius, cy + self.radius

            if x2 < 0 or y2 < 0 or x1 >= self.map.shape[1] or y1 >= self.map.shape[0]:
                continue

            cx1, cy1 = max(0, -x1), max(0, -y1)
            cx2 = self.circle.shape[1] - max(0, x2 - self.map.shape[1])
            cy2 = self.circle.shape[0] - max(0, y2 - self.map.shape[0])

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(self.map.shape[1], x2), min(self.map.shape[0], y2)

            self.map[y1:y2, x1:x2] += self.circle[cy1:cy2, cx1:cx2]

    def draw(self, frame):
        heat = cv2.normalize(self.map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat = cv2.equalizeHist(heat)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return cv2.addWeighted(frame, 0.5, heat, 0.5, 0)


def preprocess(frame, input_shape, dtype):
    h, w = input_shape[1], input_shape[2]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if dtype == np.uint8:
        img = img.astype(np.uint8)
    else:
        img = img.astype(np.float32) / 255.0

    return np.expand_dims(img, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.55)
    args = parser.parse_args()

    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError("RTSP open failed")

    tracker = Tracker(distance_function="iou", distance_threshold=0.7)

    ret, frame = cap.read()
    heatmap = HeatMap(frame.shape)

    # ---------------- DWELL DATA ----------------
    dwell_radius = 30
    dwell = {}  # id -> {pos, last_time, total}

    last_print = time.time()

    while True:
        for _ in range(3):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        img = preprocess(frame, inp[0]["shape"], np.dtype(inp[0]["dtype"]))
        interpreter.set_tensor(inp[0]["index"], img)
        interpreter.invoke()

        boxes = interpreter.get_tensor(out[0]["index"]).squeeze()
        classes = interpreter.get_tensor(out[1]["index"]).squeeze()
        scores = interpreter.get_tensor(out[2]["index"]).squeeze()

        detections = []
        h, w = frame.shape[:2]
        for i in range(len(scores)):
            if scores[i] < args.threshold or int(classes[i]) != 0:
                continue
            y1, x1, y2, x2 = boxes[i]
            pts = np.array([[x1 * w, y1 * h], [x2 * w, y2 * h]], np.float32)
            sc = np.array([scores[i], scores[i]], np.float32)
            detections.append(Detection(points=pts, scores=sc))

        tracks = tracker.update(detections)

        # ---------------- HEATMAP ----------------
        heatmap.update(tracks)
        out_frame = heatmap.draw(frame)

        now = time.time()

        # ---------------- DWELL TIME ----------------
        for t in tracks:
            pts = np.array(t.estimate)
            cx = int((pts[0][0] + pts[1][0]) / 2)
            cy = int((pts[0][1] + pts[1][1]) / 2)

            tid = t.id
            if tid not in dwell:
                dwell[tid] = {"pos": (cx, cy), "last": now, "total": 0.0}
            else:
                px, py = dwell[tid]["pos"]
                dist = math.hypot(cx - px, cy - py)

                if dist <= dwell_radius:
                    dwell[tid]["total"] += now - dwell[tid]["last"]
                else:
                    dwell[tid]["pos"] = (cx, cy)

                dwell[tid]["last"] = now

            cv2.putText(out_frame, f"ID {tid}",
                        (cx + 6, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2)
            cv2.circle(out_frame, (cx, cy), 5, (0, 0, 255), -1)

        # -------- TERMINAL PRINT (EVERY 1s) --------
        if now - last_print > 1.0:
            print("\n--- DWELL TIME (seconds) ---")
            for tid, v in dwell.items():
                print(f"ID {tid}: {v['total']:.2f}")
            last_print = now

        cv2.imshow("People Tracking + Heatmap", out_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

