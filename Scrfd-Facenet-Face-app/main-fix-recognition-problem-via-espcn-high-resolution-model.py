from face_recognition import Facenet
from face_database import FaceDatabase
from face_detection_scrfd import SCRFD

import cv2
import numpy as np
import os

# ======================
# ESPCN CONFIG
# ======================
USE_TFLITE = True          # True: tflite_runtime / TF Lite, False: ailia_tflite
USE_FLOAT_MODEL = False     # True: espcn.tflite, False: espcn_quant.tflite

if USE_TFLITE:
    import tflite_runtime.interpreter as tflite
else:
    import ailia_tflite

if USE_FLOAT_MODEL:
    ESPCN_MODEL_NAME = "espcn.tflite"
else:
    ESPCN_MODEL_NAME = "espcn_quant.tflite"

ESPCN_MODEL_PATH = os.path.join(os.path.dirname(__file__), ESPCN_MODEL_NAME)

# ======================
# ESPCN UTILS
# ======================
def espcn_get_input_tensor(tensor, input_details):
    details = input_details[0]
    dtype = details["dtype"]
    if dtype == np.uint8 or dtype == np.int8:
        quant_params = details["quantization_parameters"]
        scale = quant_params["scales"]
        zero = quant_params["zero_points"]
        tensor = tensor / scale + zero
        tensor = tensor.clip(-128, 127)
        return tensor.astype(dtype)
    else:
        return tensor

def espcn_get_real_tensor(interpreter, output_details):
    details = output_details[0]
    if details["dtype"] == np.uint8 or details["dtype"] == np.int8:
        quant_params = details["quantization_parameters"]
        scale = quant_params["scales"]
        zero = quant_params["zero_points"]
        int_tensor = interpreter.get_tensor(details["index"])
        real_tensor = (int_tensor - zero).astype(np.float32) * scale
    else:
        real_tensor = interpreter.get_tensor(details["index"])
    return real_tensor

def create_espcn_interpreter():
    if USE_TFLITE:
        interpreter = tflite.Interpreter(model_path=ESPCN_MODEL_PATH)
    else:
        interpreter = ailia_tflite.Interpreter(model_path=ESPCN_MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

def run_espcn(interpreter, bgr_img):
    # Convert to YCrCb
    ycrcb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = ycrcb[:, :, 0], ycrcb[:, :, 1], ycrcb[:, :, 2]

    # Prepare Y
    y = np.expand_dims(y, axis=2)              # H, W, 1
    y = y.astype("float32") / 255.0
    input_data = np.expand_dims(y, axis=0)     # 1, H, W, 1

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Resize input tensor if needed
    in_shape = input_details[0]["shape"]
    if (in_shape[1] != input_data.shape[1]) or (in_shape[2] != input_data.shape[2]):
        interpreter.resize_tensor_input(input_details[0]["index"], input_data.shape)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

    inputs = espcn_get_input_tensor(input_data, input_details)

    interpreter.set_tensor(input_details[0]["index"], inputs)
    interpreter.invoke()

    out_img_y = espcn_get_real_tensor(interpreter, output_details)
    out_img_y = out_img_y[0, :, :, 0]

    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = out_img_y.astype(np.uint8)

    # Resize Cr/Cb to SR size
    out_img_cr = cv2.resize(cr, (out_img_y.shape[1], out_img_y.shape[0]),
                            cv2.INTER_CUBIC).astype(np.uint8)
    out_img_cb = cv2.resize(cb, (out_img_y.shape[1], out_img_y.shape[0]),
                            cv2.INTER_CUBIC).astype(np.uint8)

    out_img = np.zeros((out_img_y.shape[0], out_img_y.shape[1], 3),
                       dtype=np.uint8)
    out_img[:, :, 0] = out_img_y
    out_img[:, :, 1] = out_img_cr
    out_img[:, :, 2] = out_img_cb

    out_bgr = cv2.cvtColor(out_img, cv2.COLOR_YCrCb2BGR)
    return out_bgr

# ======================
# FACE RECOGNITION PART
# ======================
detector = SCRFD("model_float32.tflite")
facenet = Facenet("facenet_512_int_quantized.tflite", delegate_path=None)
face_db = FaceDatabase()

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

# Create ESPCN interpreter once
espcn_interpreter = create_espcn_interpreter()

# Thresholds
SR_MIN_FACE = 60   # below this we use SR
ABS_MIN_FACE = 32  # below this we skip recognition as too small

while True:
    ret, frame = cap.read()
    if not ret:
        break

    bboxes, kpss = detector.detect(frame, thresh=0.5)

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2, score = bbox.astype(int)

        h, w = frame.shape[:2]
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        face_h, face_w = face_img.shape[:2]

        # Skip extremely small faces (unreliable)
        if face_h < ABS_MIN_FACE or face_w < ABS_MIN_FACE:
            cv2.putText(frame, "Too far", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            continue

        # Save original size for scaling keypoints
        orig_h, orig_w = face_h, face_w

        # Apply ESPCN only for smaller faces
        if face_h < SR_MIN_FACE or face_w < SR_MIN_FACE:
            face_img = run_espcn(espcn_interpreter, face_img)
            # After SR, recompute size
            face_h, face_w = face_img.shape[:2]

        kp = kpss[i]
        kp_crop = kp - np.array([x1, y1])

        # Scale keypoints if SR changed size
        scale_x = face_w / float(orig_w)
        scale_y = face_h / float(orig_h)
        kp_crop[:, 0] *= scale_x
        kp_crop[:, 1] *= scale_y

        aligned_face = align_face(face_img, kp_crop)
        emb = facenet.get_embeddings(aligned_face)

        name, conf = face_db.find_name(emb)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name} {conf:.2f}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # For debugging: show face size
        cv2.putText(
            frame,
            f"{orig_w}x{orig_h}",
            (x1, y2 + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

