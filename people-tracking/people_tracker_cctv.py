#!/usr/bin/env python3
"""
TFLite People Tracking with RTSP IP Camera
LOW LATENCY VERSION (Dahua Tested)
"""

import os
import argparse
import time
import numpy as np
import cv2


# RTSP LOW-LATENCY SETTINGS (CRITICAL)

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
        x1 = int(x1 * w)
        x2 = int(x2 * w)
        y1 = int(y1 * h)
        y2 = int(y2 * h)

        dets.append({
            "bbox": [x1, y1, x2, y2],
            "score": float(scores[i])
        })

    return dets


def draw(frame, dets, tracks):
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"P {d['score']:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)

    for t in tracks:
        cx, cy = map(int, t.estimate[0])
        cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(frame, f"ID {t.id}",
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2)

    return frame



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--person_id", type=int, default=0)
    parser.add_argument("--show_fps", action="store_true")
    args = parser.parse_args()

    # -------- TFLite --------
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # -------- RTSP --------
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FPS, 25)

    if not cap.isOpened():
        raise RuntimeError("❌ Failed to open RTSP stream")

    # -------- Tracker --------
    tracker = Tracker(
        distance_function="iou",
        distance_threshold=0.7
    )

    prev = time.time()
    fps = 0.0
    frame_id = 0


    while True:
        # DROP OLD FRAMES (MOST IMPORTANT)
        for _ in range(3):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        frame_id += 1

        # Skip alternate frames (reduce latency)
        if frame_id % 2 != 0:
            continue

        inp = preprocess(
            frame,
            input_details[0]["shape"],
            np.dtype(input_details[0]["dtype"])
        )

        interpreter.set_tensor(input_details[0]["index"], inp)
        interpreter.invoke()

        outputs = {
            od["name"]: interpreter.get_tensor(od["index"])
            for od in output_details
        }

        boxes, classes, scores = parse_outputs(outputs)

        dets = extract_persons(
            boxes, classes, scores,
            frame.shape[:2],
            args.threshold,
            args.person_id
        )

        nf_dets = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            sc = np.array([d["score"], d["score"]], dtype=np.float32)
            nf_dets.append(Detection(points=pts, scores=sc))

        tracks = tracker.update(nf_dets)

        out = draw(frame.copy(), dets, tracks)

        if args.show_fps:
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1 / max(now - prev, 1e-6))
            prev = now
            cv2.putText(out, f"FPS {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2)

        cv2.imshow("People Tracker (RTSP Low Latency)", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

