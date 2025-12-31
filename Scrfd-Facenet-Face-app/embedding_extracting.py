import os
import cv2
import numpy as np
from face_database import FaceDatabase
from face_recognition import Facenet
from face_detection_scrfd import SCRFD

DATASET_DIR = "dataset"
MODEL_PATH = "facenet_512_int_quantized.tflite"

face_db = FaceDatabase(db_file="database.npy")
facenet = Facenet(model_path=MODEL_PATH, delegate_path=None)
detector = SCRFD("model_float32.tflite")

REFERENCE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(img, kps):
    kps = np.array(kps, dtype=np.float32)
    M = cv2.estimateAffinePartial2D(kps, REFERENCE, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(img, M, (112, 112))
    return aligned

def extract_embedding(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None

    # detect face + keypoints
    bboxes, kpss = detector.detect(img, thresh=0.5)
    if len(bboxes) == 0:
        return None

    kp = kpss[0]
    x1, y1, x2, y2, _ = bboxes[0].astype(int)

    crop = img[y1:y2, x1:x2]
    kp_crop = kp - np.array([x1, y1])

    aligned = align_face(crop, kp_crop)
    emb = facenet.get_embeddings(aligned)  # already normalized

    return emb

print("Rebuilding database...")

for person_name in os.listdir(DATASET_DIR):
    person_folder = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_folder):
        continue

    for file in os.listdir(person_folder):
        image_path = os.path.join(person_folder, file)

        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        emb = extract_embedding(image_path)
        if emb is None:
            print("Face not found in", image_path)
            continue

        face_db.add_name(person_name, emb.tolist())

print("Database rebuilt successfully!")

