#!/usr/bin/env python3
"""
People Detection + Tracking + Re-Identification
WITH DATASET CREATION (MAX 10 IMAGES PER PERSON)
"""

import os
import cv2
import argparse
import numpy as np
import tensorflow as tf
from norfair import Detection, Tracker


DET_SCORE_TH = 0.5
REID_SIM_TH = 0.7
REID_INPUT_SIZE = (256, 128)   # H, W
REID_INTERVAL = 5              # run reid every N frames
SAVE_INTERVAL = 5              # save image every N embeddings
MAX_IMAGES_PER_PERSON = 10
MIN_BBOX_AREA = 120 * 250
DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)


reid_db = {}            # pid -> list of embeddings
trackid_to_pid = {}    # tracker_id -> pid
next_person_id = 0



def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def crop_person(frame, bbox):
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, x1), max(0, y1)
    return frame[y1:y2, x1:x2]


def person_dataset_full(pid):
    person_dir = os.path.join(DATASET_DIR, f"person_{pid}")
    if not os.path.exists(person_dir):
        return False
    images = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
    return len(images) >= MAX_IMAGES_PER_PERSON


def save_person_image(pid, img):
    person_dir = os.path.join(DATASET_DIR, f"person_{pid}")
    os.makedirs(person_dir, exist_ok=True)

    images = sorted([
        f for f in os.listdir(person_dir)
        if f.endswith(".jpg")
    ])

    if len(images) >= MAX_IMAGES_PER_PERSON:
        return False

    img_path = os.path.join(
        person_dir,
        f"img_{len(images)+1:04d}.jpg"
    )
    cv2.imwrite(img_path, img)
    return True


def save_embeddings(pid):
    person_dir = os.path.join(DATASET_DIR, f"person_{pid}")
    emb_path = os.path.join(person_dir, "embeddings.npy")
    np.save(emb_path, np.array(reid_db[pid]))


def match_reid(embedding):
    best_pid = None
    best_score = -1

    for pid, emb_list in reid_db.items():
        scores = [np.dot(embedding, e) for e in emb_list]
        score = max(scores)

        if score > best_score:
            best_score = score
            best_pid = pid

    if best_score >= REID_SIM_TH:
        return best_pid, best_score

    return None, best_score


def assign_person_id(embedding, person_img):
    global next_person_id

    pid, score = match_reid(embedding)

    if pid is not None:
        if not person_dataset_full(pid):
            reid_db[pid].append(embedding)
            if len(reid_db[pid]) % SAVE_INTERVAL == 0:
                if save_person_image(pid, person_img):
                    save_embeddings(pid)
        return pid

    # New person
    pid = next_person_id
    next_person_id += 1

    reid_db[pid] = [embedding]
    save_person_image(pid, person_img)
    save_embeddings(pid)

    return pid



def load_tflite(path):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter


def tflite_infer(interpreter, input_tensor):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    outputs = {}
    for od in output_details:
        outputs[od["index"]] = interpreter.get_tensor(od["index"])
    return outputs


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
        if int(classes[i]) != 0:
            continue

        ymin, xmin, ymax, xmax = boxes[i]
        x1, y1 = int(xmin * w), int(ymin * h)
        x2, y2 = int(xmax * w), int(ymax * h)

        area = (x2 - x1) * (y2 - y1)
        if area < MIN_BBOX_AREA:
            continue

        dets.append({"bbox": [x1, y1, x2, y2], "score": scores[i]})
    return dets



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
    parser.add_argument("--det_model", required=True)
    parser.add_argument("--reid_model", required=True)
    parser.add_argument("--source", default=0)
    args = parser.parse_args()

    det_interp = load_tflite(args.det_model)
    reid_interp = load_tflite(args.reid_model)

    det_input = det_interp.get_input_details()[0]
    det_shape = det_input["shape"]
    det_dtype = np.dtype(det_input["dtype"])

    cap = cv2.VideoCapture(
        int(args.source) if str(args.source).isdigit() else args.source
    )

    tracker = Tracker(distance_function="iou", distance_threshold=0.7)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        inp = preprocess_det(frame, det_shape, det_dtype)
        outputs = tflite_infer(det_interp, inp)
        dets = parse_detections(outputs, frame.shape)

        nf_dets = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            pts = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            scores = np.array([d["score"], d["score"]], dtype=np.float32)
            nf_dets.append(Detection(points=pts, scores=scores))

        tracked = tracker.update(nf_dets)

        for tobj, det in zip(tracked, dets):
            x1, y1, x2, y2 = det["bbox"]
            tid = tobj.id

            if tid not in trackid_to_pid or frame_id % REID_INTERVAL == 0:
                person_img = crop_person(frame, det["bbox"])
                emb = extract_embedding(reid_interp, person_img)
                pid = assign_person_id(emb, person_img)
                trackid_to_pid[tid] = pid

            pid = trackid_to_pid[tid]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame,
                f"PID:{pid}",
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        cv2.imshow("People Tracking + ReID (Max 10 Images)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

