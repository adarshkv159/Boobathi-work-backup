#!/usr/bin/env python3
"""
People Detection + Tracking + Re-Identification (TFLite + Norfair)

- Detect persons using TFLite SSD model
- Track them using Norfair
- Re-identify persons using embedding model
- Maintain persistent person IDs
"""

import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from norfair import Detection, Tracker

# ==========================
# CONFIG
# ==========================
DET_SCORE_TH = 0.5
REID_SIM_TH = 0.7
REID_INPUT_SIZE = (256, 128)  # H, W
REID_INTERVAL = 5             # run re-id every N frames

# ==========================
# REID DATABASE
# ==========================
reid_db = {}         # person_id -> embedding
next_person_id = 0
trackid_to_pid = {} # tracker_id -> person_id


# ==========================
# UTILS
# ==========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def assign_person_id(embedding):
    global next_person_id

    best_pid = None
    best_score = -1

    for pid, ref_emb in reid_db.items():
        score = cosine_similarity(embedding, ref_emb)
        if score > best_score:
            best_score = score
            best_pid = pid

    if best_score >= REID_SIM_TH:
        return best_pid, best_score

    pid = next_person_id
    reid_db[pid] = embedding
    next_person_id += 1
    return pid, 1.0


def crop_person(frame, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    return frame[y1:y2, x1:x2]


# ==========================
# TFLITE HELPERS
# ==========================
def load_tflite_model(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def tflite_infer(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], image)
    interpreter.invoke()

    outputs = {}
    for od in output_details:
        outputs[od["name"]] = interpreter.get_tensor(od["index"])
    return outputs


# ==========================
# PERSON DETECTION
# ==========================
def preprocess_det(frame, input_shape, dtype):
    h, w = input_shape[1], input_shape[2]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dtype == np.float32:
        img = img.astype(np.float32) / 255.0
    else:
        img = img.astype(np.uint8)
    return np.expand_dims(img, axis=0)


def parse_detections(outputs, frame_shape):
    h, w = frame_shape[:2]

    boxes = np.squeeze(list(outputs.values())[0])
    classes = np.squeeze(list(outputs.values())[1])
    scores = np.squeeze(list(outputs.values())[2])

    dets = []
    for i in range(len(scores)):
        if scores[i] < DET_SCORE_TH:
            continue
        if int(classes[i]) != 0:  # person class
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        dets.append({
            "bbox": [x1, y1, x2, y2],
            "score": scores[i]
        })
    return dets


# ==========================
# REID INFERENCE
# ==========================
def extract_embedding(interpreter, person_img):
    img = cv2.resize(person_img, (REID_INPUT_SIZE[1], REID_INPUT_SIZE[0]))
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    emb = interpreter.get_tensor(output_details[0]["index"])[0]
    return emb / np.linalg.norm(emb)


# ==========================
# MAIN
# ==========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model", required=True, help="TFLite detection model")
    parser.add_argument("--reid_model", required=True, help="TFLite re-id model")
    parser.add_argument("--source", default=0)
    args = parser.parse_args()

    det_interpreter = load_tflite_model(args.det_model)
    reid_interpreter = load_tflite_model(args.reid_model)

    det_input = det_interpreter.get_input_details()[0]
    det_shape = det_input["shape"]
    det_dtype = np.dtype(det_input["dtype"])

    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else args.source)
    if not cap.isOpened():
        print("❌ Cannot open source")
        return

    tracker = Tracker(distance_function="iou", distance_threshold=0.7)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # Detection
        inp = preprocess_det(frame, det_shape, det_dtype)
        outputs = tflite_infer(det_interpreter, inp)
        dets = parse_detections(outputs, frame.shape)

        # Norfair detections
        nf_dets = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            scores = np.array([d["score"], d["score"]], dtype=np.float32)
            nf_dets.append(Detection(points=points, scores=scores))

        tracked = tracker.update(detections=nf_dets)

        # Visualization + ReID
        for tobj, det in zip(tracked, dets):
            x1, y1, x2, y2 = det["bbox"]
            track_id = tobj.id

            if track_id not in trackid_to_pid or frame_id % REID_INTERVAL == 0:
                person_img = crop_person(frame, det["bbox"])
                if person_img.size > 0:
                    emb = extract_embedding(reid_interpreter, person_img)
                    pid, sim = assign_person_id(emb)
                    trackid_to_pid[track_id] = pid

            pid = trackid_to_pid.get(track_id, -1)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"PID:{pid}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        cv2.imshow("People Tracking + ReID", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

