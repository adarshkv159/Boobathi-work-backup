#!/usr/bin/env python3
"""
CCTV People Tracking + Re-Identification (FIXED)
NO LAG VERSION – same design philosophy as heatmap app
"""

import os
import cv2
import time
import argparse
import numpy as np
import tensorflow as tf
from norfair import Detection, Tracker


# RTSP LOW LATENCY SETTINGS

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


DET_SCORE_TH = 0.5
REID_SIM_TH = 0.7
REID_INPUT_SIZE = (256, 128)
REID_FPS = 1.0                
MAX_IMAGES_PER_PERSON = 10
MIN_BBOX_AREA = 120 * 250
DATASET_DIR = "dataset"

os.makedirs(DATASET_DIR, exist_ok=True)


reid_db = {}            # pid -> embeddings
trackid_to_pid = {}    # tracker_id -> pid
next_person_id = 0
last_reid_time = 0.0



def crop_person(frame, bbox):
    x1, y1, x2, y2 = bbox
    return frame[max(0,y1):y2, max(0,x1):x2]


def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def match_reid(emb):
    best_pid, best_score = None, -1
    for pid, embs in reid_db.items():
        score = max(cosine(emb, e) for e in embs)
        if score > best_score:
            best_score, best_pid = score, pid
    return best_pid if best_score >= REID_SIM_TH else None


def assign_pid(emb, person_img):
    global next_person_id
    pid = match_reid(emb)

    if pid is not None:
        return pid

    pid = next_person_id
    next_person_id += 1
    reid_db[pid] = [emb]

    pdir = os.path.join(DATASET_DIR, f"person_{pid}")
    os.makedirs(pdir, exist_ok=True)
    cv2.imwrite(os.path.join(pdir, "img_0001.jpg"), person_img)

    return pid


def load_tflite(path):
    i = tf.lite.Interpreter(model_path=path)
    i.allocate_tensors()
    return i


def extract_embedding(i, img):
    img = cv2.resize(img, (REID_INPUT_SIZE[1], REID_INPUT_SIZE[0]))
    img = np.expand_dims(img.astype(np.float32), 0)
    i.set_tensor(i.get_input_details()[0]["index"], img)
    i.invoke()
    e = i.get_tensor(i.get_output_details()[0]["index"])[0]
    return e / np.linalg.norm(e)


def preprocess_det(frame, shape, dtype):
    h, w = shape[1], shape[2]
    img = cv2.resize(frame, (w, h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)/255.0 if dtype==np.float32 else img.astype(np.uint8)
    return np.expand_dims(img, 0)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--det_model", required=True)
    parser.add_argument("--reid_model", required=True)
    args = parser.parse_args()

    det_i = load_tflite(args.det_model)
    reid_i = load_tflite(args.reid_model)

    det_in = det_i.get_input_details()[0]
    det_shape = det_in["shape"]
    det_dtype = np.dtype(det_in["dtype"])

    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("❌ RTSP open failed")
        return

    tracker = Tracker(distance_function="iou", distance_threshold=0.7)

    global last_reid_time

    while True:

        for _ in range(3):
            cap.grab()
        ret, frame = cap.retrieve()
        if not ret:
            continue

        # -------- Detection --------
        inp = preprocess_det(frame, det_shape, det_dtype)
        det_i.set_tensor(det_in["index"], inp)
        det_i.invoke()

        boxes = det_i.get_tensor(det_i.get_output_details()[0]["index"]).squeeze()
        classes = det_i.get_tensor(det_i.get_output_details()[1]["index"]).squeeze()
        scores = det_i.get_tensor(det_i.get_output_details()[2]["index"]).squeeze()

        detections = []
        h, w = frame.shape[:2]

        for i in range(len(scores)):
            if scores[i] < DET_SCORE_TH or int(classes[i]) != 0:
                continue
            y1, x1, y2, x2 = boxes[i]
            x1, y1, x2, y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            if (x2-x1)*(y2-y1) < MIN_BBOX_AREA:
                continue
            pts = np.array([[x1,y1],[x2,y2]], np.float32)
            sc = np.array([scores[i],scores[i]], np.float32)
            detections.append(Detection(points=pts, scores=sc))

        tracks = tracker.update(detections)

        now = time.time()
        do_reid = (now - last_reid_time) >= (1.0 / REID_FPS)

        # -------- Tracking + (SLOW) ReID --------
        for t in tracks:
            pts = np.array(t.estimate)
            x1,y1 = int(pts[0][0]), int(pts[0][1])
            x2,y2 = int(pts[1][0]), int(pts[1][1])
            tid = t.id

            if do_reid and tid not in trackid_to_pid:
                person_img = crop_person(frame, (x1,y1,x2,y2))
                if person_img.size > 0:
                    emb = extract_embedding(reid_i, person_img)
                    trackid_to_pid[tid] = assign_pid(emb, person_img)
                    last_reid_time = now

            pid = trackid_to_pid.get(tid, -1)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.putText(frame,f"PID:{pid}",(x1,y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

        cv2.imshow("CCTV People Tracking + ReID (FIXED)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

