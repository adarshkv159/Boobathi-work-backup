from face_recognition import Facenet
from face_database import FaceDatabase
from face_detection_scrfd import SCRFD
import cv2
import numpy as np

detector = SCRFD("model_float32.tflite")
facenet  = Facenet("facenet_512_int_quantized.tflite", delegate_path=None)
face_db  = FaceDatabase()

cap = cv2.VideoCapture(0)

REFERENCE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)

def align_face(img, kps):
    kps = np.array(kps, dtype=np.float32)
    transform = cv2.estimateAffinePartial2D(kps, REFERENCE, method=cv2.LMEDS)[0]
    aligned = cv2.warpAffine(img, transform, (112, 112))
    return aligned

while True:
    ret, frame = cap.read()
    if not ret:
        break

    bboxes, kpss = detector.detect(frame, thresh=0.5)

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, score = bbox.astype(int)
        
        # facenet takes 112x112 pixel to recognize but we implementing longrage detection only give less pixel input .
        """min_size = 90
        face_h = y2 - y1
        face_w = x2 - x1
        
        if face_h < min_size or face_w < min_size:
            cv2.putText(frame, "Too far", (x1, y1 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            continue """
        
        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))
        
        

        # Crop
        face_img = frame[y1:y2, x1:x2]

        kp = kpss[i]
        kp_crop = kp - np.array([x1, y1])
        aligned_face = align_face(face_img, kp_crop)
        emb = facenet.get_embeddings(aligned_face)
        
        
        face_h = y2 - y1
        face_w = x2 - x1
        face_size = max(face_h, face_w)
        
        if face_size < 120:
            threshold = 1.0
        else:
            threshold = 0.9
            
        face_db.threshold = threshold
        
        # Recognize
        name, conf = face_db.find_name(emb)

        # Draw
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{name} {conf:.2f}", 
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

