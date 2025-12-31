import os
import sys
import glob
import time
import traceback
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2 as cv
import tflite_runtime.interpreter as tflite
from skimage.transform import SimilarityTransform
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize
from PySide6.QtGui import QImage, QPixmap, QAction
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QLineEdit,
    QSpinBox, QDoubleSpinBox, QFileDialog, QTabWidget,
    QGroupBox, QFormLayout, QHBoxLayout, QVBoxLayout, QTextEdit,
    QMessageBox
)


DB_DEFAULT = "database.npy"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------------------
# Alignment (same as original)
# ---------------------------
_REFERENCE = np.array([[
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
]], dtype=np.float32)


def estimate_norm(kpt, image_size=112):
    assert kpt.shape == (5, 2)
    transform = SimilarityTransform()
    src = _REFERENCE if image_size == 112 else (float(image_size) / 112.0 * _REFERENCE)
    min_error = float("inf")
    min_matrix = None
    kpt_transform = np.insert(kpt, 2, values=np.ones(5), axis=1)
    for i in np.arange(src.shape[0]):
        transform.estimate(kpt, src[i])
        matrix = transform.params[0:2, :]
        results = np.dot(matrix, kpt_transform.T).T
        error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
        if error < min_error:
            min_error = error
            min_matrix = matrix
    return min_matrix


def norm_crop_image(image, landmark, image_size=112):
    matrix = estimate_norm(landmark, image_size)
    return cv.warpAffine(image, matrix, (image_size, image_size), borderValue=0.0)


# ---------------------------
# Utils
# ---------------------------
def l2_normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.sqrt(np.sum(v * v)))
    return v / (n + eps)


def try_load_delegate(delegate_path: str):
    if not delegate_path:
        return None
    if not os.path.exists(delegate_path):
        print(f"[WARN] Delegate not found: {delegate_path} (CPU fallback)")
        return None
    try:
        return tflite.load_delegate(delegate_path)
    except Exception as e:
        print(f"[WARN] Failed to load delegate {delegate_path}: {e} (CPU fallback)")
        return None


def make_interpreter(model_path: str, num_threads: int, delegate):
    if delegate is not None:
        itp = tflite.Interpreter(
            model_path=model_path,
            num_threads=num_threads,
            experimental_delegates=[delegate],
        )
    else:
        itp = tflite.Interpreter(model_path=model_path, num_threads=num_threads)
    itp.allocate_tensors()
    return itp


# ---------------------------
# DBFace BBox class
# ---------------------------
class BBox:
    def __init__(self, label, xyrb, score=0, landmark=None, rotate=False):
        self.label = label
        self.score = score
        self.landmark = landmark
        self.x, self.y, self.r, self.b = xyrb
        self.rotate = rotate

        minx = min(self.x, self.r)
        maxx = max(self.x, self.r)
        miny = min(self.y, self.b)
        maxy = max(self.y, self.b)
        self.x, self.y, self.r, self.b = minx, miny, maxx, maxy

    def __repr__(self):
        landmark_formated = ",".join([str(item[:2]) for item in self.landmark]) if self.landmark else "empty"
        return f"(BBox[{self.label}]: x={self.x:.2f}, y={self.y:.2f}, r={self.r:.2f}, " + \
            f"b={self.b:.2f}, width={self.width:.2f}, height={self.height:.2f}, landmark={landmark_formated})"

    @property
    def width(self):
        return self.r - self.x + 1

    @property
    def height(self):
        return self.b - self.y + 1

    @property
    def area(self):
        return self.width * self.height

    @property
    def haslandmark(self):
        return self.landmark is not None

    @property
    def xxxxxyyyyy(self):
        x = [p[0] for p in self.landmark]
        y = [p[1] for p in self.landmark]
        return x, y

    @property
    def box(self):
        return [self.x, self.y, self.r, self.b]

    @property
    def xywh(self):
        return [self.x, self.y, self.width, self.height]

    def iou(self, other):
        return computeIOU(self.box, other.box)


def computeIOU(rec1, rec2):
    cx1, cy1, cx2, cy2 = rec1
    gx1, gy1, gx2, gy2 = rec2
    S_rec1 = (cx2 - cx1 + 1) * (cy2 - cy1 + 1)
    S_rec2 = (gx2 - gx1 + 1) * (gy2 - gy1 + 1)
    x1 = max(cx1, gx1)
    y1 = max(cy1, gy1)
    x2 = min(cx2, gx2)
    y2 = min(cy2, gy2)

    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    area = w * h
    iou = area / (S_rec1 + S_rec2 - area)
    return iou


def exp(v):
    if isinstance(v, tuple) or isinstance(v, list):
        return [exp(item) for item in v]
    elif isinstance(v, np.ndarray):
        return np.exp(v)
    else:
        return np.exp(v)


# ---------------------------
# DBFace detector
# ---------------------------
def nms(objs, iou=0.5):
    if objs is None or len(objs) <= 1:
        return objs

    objs = sorted(objs, key=lambda obj: obj.score, reverse=True)
    keep = []
    flags = [0] * len(objs)
    for index, obj in enumerate(objs):
        if flags[index] != 0:
            continue

        keep.append(obj)
        for j in range(index + 1, len(objs)):
            if flags[j] == 0 and obj.iou(objs[j]) > iou:
                flags[j] = 1
    return keep


class DBFace:
    def __init__(self, model_file, nms_thresh=0.3, use_npu=False,
                 delegate_path="/usr/lib/libvx_delegate.so", num_threads=4):
        self.model_file = model_file
        self.nms_thresh = nms_thresh
        self.use_npu = use_npu
        self.delegate_path = delegate_path
        self.num_threads = num_threads
        self.interpreter = None
        self._create_interpreter()
        self._init_vars()

    def _create_interpreter(self):
        delegates = []
        if self.use_npu:
            try:
                if os.path.exists(self.delegate_path):
                    print(f"[DBFace] Loading delegate: {self.delegate_path}")
                    ext_delegate_options = {
                        'cache_file_path': './vx_cache',
                        'allowed_builtin_code': '0,1,2,3,4,5,6,7,8,9,10',
                        'allowed_cache_mode': 'TRUE',
                        'device': '0'
                    }
                    delegates.append(tflite.load_delegate(self.delegate_path, ext_delegate_options))
                    print("[DBFace] Delegate loaded")
                else:
                    print(f"[DBFace] Delegate not found: {self.delegate_path} (CPU fallback)")
            except Exception as e:
                print(f"[DBFace] Delegate load failed: {e} (CPU fallback)")

        if delegates:
            self.interpreter = tflite.Interpreter(
                model_path=self.model_file,
                experimental_delegates=delegates
            )
        else:
            self.interpreter = tflite.Interpreter(
                model_path=self.model_file,
                num_threads=self.num_threads
            )
        self.interpreter.allocate_tensors()

    def _init_vars(self):
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        input_shape = self.input_details[0]['shape']  # [1, H, W, 3]
        self.input_size = (int(input_shape[2]), int(input_shape[1]))  # (W, H)

        print(f"[DBFace] Input size (W,H): {self.input_size}")
        print(f"[DBFace] Input shape: {input_shape}")
        print(f"[DBFace] Input dtype: {self.input_details[0]['dtype']}")

        # Check if model is quantized
        self.is_quantized = self.input_details[0]['dtype'] == np.uint8

        if self.is_quantized:
            print("[DBFace] Model is quantized (uint8)")
        else:
            print("[DBFace] Model is float32")

    def _detect_forward(self, image, threshold=0.4):
        # image shape: [1, 480, 640, 3]

        if self.is_quantized:
            # Quantized model: uint8 input
            input_data = image.astype(np.uint8)
        else:
            # Float32 model: normalize to [0, 1]
            input_data = image.astype(np.float32) / 255.0

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        if self.is_quantized:
            # Get quantized outputs and dequantize
            lm_quant = self.interpreter.get_tensor(self.output_details[0]['index'])
            hm_quant = self.interpreter.get_tensor(self.output_details[1]['index'])
            box_quant = self.interpreter.get_tensor(self.output_details[2]['index'])

            # Dequantize
            lm_scale, lm_zero_point = self.output_details[0]['quantization']
            lm = (lm_quant.astype(np.float32) - lm_zero_point) * lm_scale

            hm_scale, hm_zero_point = self.output_details[1]['quantization']
            hm = (hm_quant.astype(np.float32) - hm_zero_point) * hm_scale

            box_scale, box_zero_point = self.output_details[2]['quantization']
            box = (box_quant.astype(np.float32) - box_zero_point) * box_scale

            # Transpose: (1,h,w,c) -> (1,c,h,w)
            lm = lm.transpose((0, 3, 1, 2))
            hm = hm.transpose((0, 3, 1, 2))
            box = box.transpose((0, 3, 1, 2))
        else:
            # Float32 model: outputs are already in correct format
            lm = self.interpreter.get_tensor(self.output_details[0]['index'])
            hm = self.interpreter.get_tensor(self.output_details[1]['index'])
            box = self.interpreter.get_tensor(self.output_details[2]['index'])

        # Apply sigmoid to heatmap
        hm = 1.0 / (1.0 + np.exp(-hm))

        # Simple max pooling using scipy or numpy
        from scipy.ndimage import maximum_filter
        hm_pool = maximum_filter(hm, size=(1, 1, 3, 3), mode='constant')

        # Find peaks
        hm_flat = ((hm == hm_pool).astype(np.float32) * hm).reshape(1, -1)

        # Get top 1000 scores
        scores_indices = np.argsort(hm_flat[0])[::-1][:1000]
        scores = hm_flat[0][scores_indices]

        hm_height, hm_width = hm.shape[2:]

        ys = (scores_indices // hm_width).astype(int)
        xs = (scores_indices % hm_width).astype(int)

        box = box[0]  # Remove batch dimension
        lm = lm[0]    # Remove batch dimension

        stride = 4
        objs = []

        for cx, cy, score in zip(xs, ys, scores):
            if score < threshold:
                break

            x, y, r, b = box[:, cy, cx]
            xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride

            x5y5 = lm[:, cy, cx]
            x5y5 = (exp(x5y5 * 4) + np.array([cx]*5 + [cy]*5)) * stride
            box_landmark = list(zip(x5y5[:5], x5y5[5:]))

            objs.append(BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))

        return nms(objs, iou=self.nms_thresh)

    def detect(self, img, thresh=0.5, input_size=None, max_num=0):
        """
        Detect faces in image.

        Args:
            img: BGR image (numpy array)
            thresh: detection threshold
            input_size: tuple (W, H) for model input size
            max_num: maximum number of faces to return (0 = all)

        Returns:
            det: numpy array of shape (N, 5) with [x1, y1, x2, y2, score]
            kpss: numpy array of shape (N, 5, 2) with 5 landmarks per face
        """
        input_size = self.input_size if input_size is None else input_size  # (W, H)

        # Resize image to input size
        h, w = img.shape[:2]
        im_ratio = float(h) / w
        model_ratio = float(input_size[1]) / input_size[0]

        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / h

        resized_img = cv.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        # Convert BGR to RGB
        det_img = cv.cvtColor(det_img, cv.COLOR_BGR2RGB)
        det_img = det_img[np.newaxis, :, :, :]  # Add batch dimension

        # Detect
        objs = self._detect_forward(det_img, threshold=thresh)

        if len(objs) == 0:
            return np.zeros((0, 5), dtype=np.float32), np.zeros((0, 5, 2), dtype=np.float32)

        # Convert to SCRFD-like format
        det = []
        kpss = []

        for obj in objs:
            # Scale back to original image size
            x1, y1, x2, y2 = obj.box
            x1, y1, x2, y2 = x1 / det_scale, y1 / det_scale, x2 / det_scale, y2 / det_scale
            det.append([x1, y1, x2, y2, obj.score])

            # Scale landmarks
            landmarks = np.array(obj.landmark, dtype=np.float32)
            landmarks = landmarks / det_scale
            kpss.append(landmarks)

        det = np.array(det, dtype=np.float32)
        kpss = np.array(kpss, dtype=np.float32)

        # Apply max_num if specified
        if max_num > 0 and det.shape[0] > max_num:
            # Sort by score and keep top max_num
            order = det[:, 4].argsort()[::-1][:max_num]
            det = det[order]
            kpss = kpss[order]

        return det, kpss


# ---------------------------
# Recognition int8
# ---------------------------
class FaceRecognitionTFLiteInt8:
    def __init__(self, tflite_path: str, num_threads: int = 2, delegate=None, rgb: bool = True):
        self.rgb = rgb
        self.itp = make_interpreter(tflite_path, num_threads, delegate)
        self.inp = self.itp.get_input_details()[0]
        self.out = self.itp.get_output_details()[0]

        assert self.inp["dtype"] == np.int8
        assert self.out["dtype"] == np.int8

        self.in_scale, self.in_zp = self.inp["quantization"]
        self.out_scale, self.out_zp = self.out["quantization"]

        shp = self.inp["shape"]
        self.in_h = int(shp[1])
        self.in_w = int(shp[2])

    def __call__(self, face_bgr_112: np.ndarray) -> np.ndarray:
        img = face_bgr_112
        if img.shape[0] != self.in_h or img.shape[1] != self.in_w:
            img = cv.resize(img, (self.in_w, self.in_h))

        if self.rgb:
            img = img[:, :, ::-1]  # BGR->RGB

        x = img.astype(np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        x = np.expand_dims(x, 0)

        x_q = np.round(x / self.in_scale + self.in_zp).astype(np.int32)
        x_q = np.clip(x_q, -128, 127).astype(np.int8)

        self.itp.set_tensor(self.inp["index"], x_q)
        self.itp.invoke()

        y_q = self.itp.get_tensor(self.out["index"])
        y = (y_q.astype(np.float32) - self.out_zp) * self.out_scale

        return y[0].astype(np.float32)


# ---------------------------
# DB helpers
# ---------------------------
def db_empty():
    return {
        "person_names": [],
        "labels": np.zeros((0,), dtype=np.int32),
        "embeddings": np.zeros((0, 512), dtype=np.float32),
        "paths": np.array([], dtype=object),
    }


def db_save(path: str, db: dict):
    np.save(path, db)


def db_load(path: str):
    if not os.path.exists(path):
        return db_empty()
    db = np.load(path, allow_pickle=True).item()
    for k in ["person_names", "labels", "embeddings", "paths"]:
        if k not in db:
            raise ValueError(f"DB missing key: {k}")
    return db


def db_add_embedding(db: dict, person_name: str, emb: np.ndarray, src_path: str):
    emb = l2_normalize(emb.astype(np.float32).reshape(-1))
    if person_name not in db["person_names"]:
        db["person_names"].append(person_name)

    person_id = db["person_names"].index(person_name)
    db["labels"] = np.concatenate([db["labels"], np.array([person_id], dtype=np.int32)], axis=0)
    db["embeddings"] = np.vstack([db["embeddings"], emb[None, :]]).astype(np.float32)
    db["paths"] = np.concatenate([db["paths"], np.array([src_path], dtype=object)], axis=0)


def match_identity(db: dict, emb: np.ndarray, threshold: float):
    if db["embeddings"].shape[0] == 0:
        return "Unknown", None, None

    scores = db["embeddings"] @ emb
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    person_id = int(db["labels"][best_idx])

    name = db["person_names"][person_id] if best_score >= threshold else "Unknown"
    return name, best_score, best_idx


# ---------------------------
# Dataset helpers
# ---------------------------
def iter_dataset_images(dataset_dir: str):
    if not os.path.exists(dataset_dir):
        return

    for person in sorted(os.listdir(dataset_dir)):
        pdir = os.path.join(dataset_dir, person)
        if not os.path.isdir(pdir):
            continue

        for ext in IMG_EXTS:
            for path in sorted(glob.glob(os.path.join(pdir, f"*{ext}"))):
                yield person, path


def compute_embedding_from_bgr(img_bgr, detector: DBFace, recognizer: FaceRecognitionTFLiteInt8, det_thresh: float):
    det, kpss = detector.detect(img_bgr, thresh=det_thresh, input_size=detector.input_size, max_num=1)

    if kpss is None or len(kpss) == 0:
        return None

    kpt = kpss[0].astype(np.float32)
    face = norm_crop_image(img_bgr, kpt, 112)
    emb = l2_normalize(recognizer(face).astype(np.float32))

    return emb


# ---------------------------
# Qt helpers
# ---------------------------
def cv_bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv.cvtColor(bgr, cv.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


# ---------------------------
# Worker threads
# ---------------------------
@dataclass
class AppConfig:
    threads: int = 4
    dbface_model: str = "model_full_integer_quant.tflite"
    rec_model: str = "recognition_full_integer_quant.tflite"
    dbface_force_cpu: bool = False  # DBFace can use NPU
    rec_use_npu: bool = True
    dbface_delegate: str = "/usr/lib/libvx_delegate.so"
    rec_delegate: str = "/usr/lib/libvx_delegate.so"
    det_thresh: float = 0.62
    nms_thresh: float = 0.3


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    status = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False
        self.mode = "preview"  # preview | capture | recognize
        self.cam_index = 0
        self.detector: Optional[DBFace] = None
        self.recognizer: Optional[FaceRecognitionTFLiteInt8] = None
        self.db_path: str = DB_DEFAULT
        self.threshold: float = 0.60
        self.det_thresh: float = 0.62
        self.max_faces: int = 0
        self.dataset_dir: str = "dataset"
        self.person_name: str = "person"
        self._last_frame_bgr: Optional[np.ndarray] = None

    def configure_models(self, detector: DBFace, recognizer: Optional[FaceRecognitionTFLiteInt8]):
        self.detector = detector
        self.recognizer = recognizer

    def set_mode_preview(self, cam_index: int):
        self.mode = "preview"
        self.cam_index = cam_index

    def set_mode_capture(self, cam_index: int, dataset_dir: str, person_name: str, det_thresh: float):
        self.mode = "capture"
        self.cam_index = cam_index
        self.dataset_dir = dataset_dir
        self.person_name = person_name
        self.det_thresh = det_thresh

    def set_mode_recognize(self, cam_index: int, db_path: str, threshold: float, det_thresh: float, max_faces: int):
        self.mode = "recognize"
        self.cam_index = cam_index
        self.db_path = db_path
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.max_faces = max_faces

    def last_frame(self) -> Optional[np.ndarray]:
        return None if self._last_frame_bgr is None else self._last_frame_bgr.copy()

    def stop(self):
        self._running = False

    def save_current_frame_to_dataset(self) -> Tuple[bool, str]:
        frame = self.last_frame()
        if frame is None:
            return False, "No frame available"

        out_dir = os.path.join(self.dataset_dir, self.person_name)
        os.makedirs(out_dir, exist_ok=True)

        fn = os.path.join(out_dir, f"{int(time.time())}.jpg")
        ok = cv.imwrite(fn, frame)

        return (ok, fn if ok else "Failed to write image")

    def run(self):
        try:
            if self.detector is None:
                raise RuntimeError("Detector not configured")

            cap = cv.VideoCapture(self.cam_index)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.cam_index}")

            self._running = True
            self.status.emit(f"Camera started (index={self.cam_index}, mode={self.mode})")

            db = None
            if self.mode == "recognize":
                if self.recognizer is None:
                    raise RuntimeError("Recognizer not configured for recognize mode")

                db = db_load(self.db_path)
                if db["embeddings"].shape[0] == 0:
                    raise RuntimeError(f"DB empty: {self.db_path} (build DB first)")

                self.status.emit(f"DB loaded: persons={len(db['person_names'])}, templates={db['embeddings'].shape[0]}")

            while self._running:
                ok, frame = cap.read()
                if not ok:
                    self.status.emit("Camera read failed")
                    break

                self._last_frame_bgr = frame
                view = frame.copy()

                if self.mode in ("capture", "recognize"):
                    det, kpss = self.detector.detect(
                        view, thresh=self.det_thresh, input_size=self.detector.input_size,
                        max_num=(self.max_faces if self.mode == "recognize" else 0)
                    )

                    if kpss is not None and len(kpss) > 0:
                        for i in range(len(det)):
                            x1, y1, x2, y2 = det[i][:4].astype(int)
                            det_score = float(det[i][4])

                            cv.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            if self.mode == "recognize":
                                kpt = kpss[i].astype(np.float32)
                                face = norm_crop_image(frame, kpt, 112)
                                emb = l2_normalize(self.recognizer(face).astype(np.float32))

                                name, sim, _ = match_identity(db, emb, self.threshold)

                                ty = y1 - 10
                                if ty < 20:
                                    ty = y1 + 20

                                txt1 = f"{name} sim={sim:.2f}" if sim is not None else name
                                txt2 = f"det={det_score:.2f}"

                                cv.putText(view, txt1, (x1, ty), cv.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
                                cv.putText(view, txt2, (x1, ty + 22), cv.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2)
                            else:
                                cv.putText(view, f"Faces: {len(det)}", (20, 35),
                                          cv.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)

                qimg = cv_bgr_to_qimage(view)
                self.frame_ready.emit(qimg)

                cv.waitKey(1)

            cap.release()
            self.status.emit("Camera stopped")

        except Exception as e:
            self.error.emit(f"{e}\n{traceback.format_exc()}")


class BuildDbWorker(QThread):
    log = Signal(str)
    done = Signal(bool, str)

    def __init__(self, dataset_dir: str, db_path: str, detector: DBFace,
                 recognizer: FaceRecognitionTFLiteInt8, det_thresh: float):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.db_path = db_path
        self.detector = detector
        self.recognizer = recognizer
        self.det_thresh = det_thresh

    def run(self):
        try:
            db = db_empty()
            total = 0
            kept = 0

            for person, path in iter_dataset_images(self.dataset_dir):
                total += 1
                img = cv.imread(path)

                if img is None:
                    self.log.emit(f"[SKIP] cannot read: {path}")
                    continue

                emb = compute_embedding_from_bgr(img, self.detector, self.recognizer, self.det_thresh)

                if emb is None:
                    self.log.emit(f"[SKIP] no face: {path}")
                    continue

                db_add_embedding(db, person, emb, path)
                kept += 1
                self.log.emit(f"[OK] {person}: {os.path.basename(path)}")

            db_save(self.db_path, db)

            msg = f"Saved DB: {self.db_path} | persons={len(db['person_names'])} templates={db['embeddings'].shape[0]} (kept {kept}/{total})"
            self.done.emit(True, msg)

        except Exception as e:
            self.done.emit(False, f"{e}\n{traceback.format_exc()}")


# ---------------------------
# Main Window
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face App - DBFace Detection")
        self.resize(1200, 800)

        self.detector: Optional[DBFace] = None
        self.recognizer: Optional[FaceRecognitionTFLiteInt8] = None
        self.cfg = AppConfig()

        self.camera_worker = CameraWorker()
        self.camera_worker.frame_ready.connect(self.on_frame)
        self.camera_worker.status.connect(self.append_log)
        self.camera_worker.error.connect(self.on_worker_error)

        self.build_worker: Optional[BuildDbWorker] = None

        self._build_ui()
        self._create_menu()

    def _create_menu(self):
        act_quit = QAction("Quit", self)
        act_quit.triggered.connect(self.close)
        self.menuBar().addMenu("").addAction(act_quit)

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        self.tabs = QTabWidget()

        # Right: log
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        main_layout = QHBoxLayout(root)
        main_layout.addWidget(self.tabs, 5)
        main_layout.addWidget(self.log_view, 1)

        # ---------------- Models tab ----------------
        tab_models = QWidget()
        self.tabs.addTab(tab_models, "Models")

        self.ed_dbface_model = QLineEdit(self.cfg.dbface_model)
        self.ed_rec_model = QLineEdit(self.cfg.rec_model)

        # DBFace NPU toggle
        self.btn_dbface_npu_toggle = QPushButton("DBFace: CPU")
        self.btn_dbface_npu_toggle.setCheckable(True)
        self.btn_dbface_npu_toggle.setChecked(not self.cfg.dbface_force_cpu)
        self._sync_dbface_toggle_text()
        self.btn_dbface_npu_toggle.toggled.connect(self.on_dbface_toggle_changed)

        self.ed_dbface_delegate = QLineEdit(self.cfg.dbface_delegate)

        # Recognition NPU toggle
        self.btn_rec_npu_toggle = QPushButton("Recognition: NPU ON")
        self.btn_rec_npu_toggle.setCheckable(True)
        self.btn_rec_npu_toggle.setChecked(self.cfg.rec_use_npu)
        self._sync_rec_toggle_text()
        self.btn_rec_npu_toggle.toggled.connect(self.on_rec_toggle_changed)

        self.ed_rec_delegate = QLineEdit(self.cfg.rec_delegate)

        self.sp_threads = QSpinBox()
        self.sp_threads.setRange(1, 32)
        self.sp_threads.setValue(self.cfg.threads)

        self.sp_det_thresh = QDoubleSpinBox()
        self.sp_det_thresh.setRange(0.01, 0.99)
        self.sp_det_thresh.setSingleStep(0.01)
        self.sp_det_thresh.setValue(self.cfg.det_thresh)

        self.sp_nms_thresh = QDoubleSpinBox()
        self.sp_nms_thresh.setRange(0.01, 0.99)
        self.sp_nms_thresh.setSingleStep(0.01)
        self.sp_nms_thresh.setValue(self.cfg.nms_thresh)

        btn_load_models = QPushButton("Load / Reload Models")
        btn_load_models.clicked.connect(self.load_models)

        form = QFormLayout(tab_models)
        form.addRow("DBFace model (.tflite):", self._file_row(self.ed_dbface_model))
        form.addRow("DBFace delegate path:", self.ed_dbface_delegate)
        form.addRow("", self.btn_dbface_npu_toggle)
        form.addRow("Recognition model (.tflite):", self._file_row(self.ed_rec_model))
        form.addRow("Recognition delegate path:", self.ed_rec_delegate)
        form.addRow("", self.btn_rec_npu_toggle)
        form.addRow("Threads:", self.sp_threads)
        form.addRow("Detection threshold:", self.sp_det_thresh)
        form.addRow("NMS threshold:", self.sp_nms_thresh)
        form.addRow("", btn_load_models)

        # ---------------- Capture tab ----------------
        tab_capture = QWidget()
        self.tabs.addTab(tab_capture, "Capture")

        self.cap_video = QLabel("Video")
        self.cap_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.cap_video.setMinimumSize(QSize(320, 240))
        self.cap_video.setStyleSheet("background: #111; color: #aaa;")

        self.sp_cam_capture = QSpinBox()
        self.sp_cam_capture.setRange(0, 16)
        self.sp_cam_capture.setValue(0)

        self.ed_dataset_dir = QLineEdit("dataset")
        btn_browse_dataset = QPushButton("Browse")
        btn_browse_dataset.clicked.connect(self.browse_dataset_dir)

        self.ed_person_name = QLineEdit("person1")

        btn_start_capture = QPushButton("Start Capture Preview")
        btn_start_capture.clicked.connect(self.start_capture_mode)

        btn_save_frame = QPushButton("Save Frame to Dataset")
        btn_save_frame.clicked.connect(self.save_capture_frame)

        btn_delete_person = QPushButton("Delete Person from Dataset")
        btn_delete_person.clicked.connect(self.delete_person_from_dataset)

        btn_upload_images = QPushButton("Upload Images to Dataset")
        btn_upload_images.clicked.connect(self.upload_images_to_dataset)

        btn_stop_capture = QPushButton("Stop")
        btn_stop_capture.clicked.connect(self.stop_camera)

        left = QVBoxLayout()
        left.addWidget(self.cap_video, 1)

        g = QGroupBox("Capture Controls")
        f = QFormLayout(g)

        row_ds = QHBoxLayout()
        row_ds.addWidget(self.ed_dataset_dir, 1)
        row_ds.addWidget(btn_browse_dataset)

        f.addRow("Dataset folder:", row_ds)
        f.addRow("Person name:", self.ed_person_name)
        f.addRow("Camera index:", self.sp_cam_capture)
        f.addRow("", btn_start_capture)
        f.addRow("", btn_save_frame)
        f.addRow("", btn_upload_images)
        f.addRow("", btn_delete_person)
        f.addRow("", btn_stop_capture)

        left.addWidget(g, 0)
        tab_capture.setLayout(left)

        # ---------------- Build DB tab ----------------
        tab_build = QWidget()
        self.tabs.addTab(tab_build, "Build DB")

        self.ed_build_dataset = QLineEdit("dataset")
        btn_browse_build_dataset = QPushButton("Browse")
        btn_browse_build_dataset.clicked.connect(self.browse_build_dataset_dir)

        self.ed_db_path = QLineEdit(DB_DEFAULT)
        btn_browse_db = QPushButton("Browse")
        btn_browse_db.clicked.connect(self.browse_db_path)

        btn_build = QPushButton("Build / Rebuild DB (Extract embeddings)")
        btn_build.clicked.connect(self.build_db)

        lay_build = QVBoxLayout(tab_build)

        g2 = QGroupBox("Build DB Settings")
        f2 = QFormLayout(g2)

        row_bds = QHBoxLayout()
        row_bds.addWidget(self.ed_build_dataset, 1)
        row_bds.addWidget(btn_browse_build_dataset)

        f2.addRow("Dataset folder:", row_bds)

        row_db = QHBoxLayout()
        row_db.addWidget(self.ed_db_path, 1)
        row_db.addWidget(btn_browse_db)

        f2.addRow("DB file (.npy):", row_db)
        f2.addRow("", btn_build)

        lay_build.addWidget(g2)
        lay_build.addStretch(1)

        # ---------------- Recognize tab ----------------
        tab_rec = QWidget()
        self.tabs.addTab(tab_rec, "Recognize")

        self.rec_video = QLabel("Video")
        self.rec_video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rec_video.setMinimumSize(QSize(320, 240))
        self.rec_video.setStyleSheet("background: #111; color: #aaa;")

        self.sp_cam_rec = QSpinBox()
        self.sp_cam_rec.setRange(0, 16)
        self.sp_cam_rec.setValue(0)

        self.ed_rec_db = QLineEdit(DB_DEFAULT)
        btn_browse_rec_db = QPushButton("Browse")
        btn_browse_rec_db.clicked.connect(self.browse_rec_db)

        self.sp_threshold = QDoubleSpinBox()
        self.sp_threshold.setRange(0.0, 1.0)
        self.sp_threshold.setSingleStep(0.01)
        self.sp_threshold.setValue(0.60)

        self.sp_max_faces = QSpinBox()
        self.sp_max_faces.setRange(0, 50)
        self.sp_max_faces.setValue(0)

        btn_start_rec = QPushButton("Start Recognition")
        btn_start_rec.clicked.connect(self.start_recognize_mode)

        btn_stop_rec = QPushButton("Stop")
        btn_stop_rec.clicked.connect(self.stop_camera)

        lay_rec = QVBoxLayout(tab_rec)
        lay_rec.addWidget(self.rec_video, 1)

        g3 = QGroupBox("Recognition Controls")
        f3 = QFormLayout(g3)

        row_rdb = QHBoxLayout()
        row_rdb.addWidget(self.ed_rec_db, 1)
        row_rdb.addWidget(btn_browse_rec_db)

        f3.addRow("DB file (.npy):", row_rdb)
        f3.addRow("Camera index:", self.sp_cam_rec)
        f3.addRow("Match threshold:", self.sp_threshold)
        f3.addRow("Max faces (0=all):", self.sp_max_faces)
        f3.addRow("", btn_start_rec)
        f3.addRow("", btn_stop_rec)

        lay_rec.addWidget(g3, 0)

        self.tabs.setCurrentIndex(0)

    def _file_row(self, line_edit: QLineEdit) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        btn = QPushButton("Browse")
        lay.addWidget(line_edit, 1)
        lay.addWidget(btn, 0)

        def pick():
            fn, _ = QFileDialog.getOpenFileName(self, "Select file", line_edit.text(), "All Files (*)")
            if fn:
                line_edit.setText(fn)

        btn.clicked.connect(pick)
        return w

    @Slot(str)
    def append_log(self, msg: str):
        self.log_view.append(msg)

    def _require_models_loaded(self) -> bool:
        if self.detector is None:
            QMessageBox.warning(self, "Models not loaded", "Load models first (Models tab).")
            return False
        return True

    def _sync_dbface_toggle_text(self):
        use_npu = not self.cfg.dbface_force_cpu
        self.btn_dbface_npu_toggle.setText("DBFace: NPU ON" if use_npu else "DBFace: CPU")

    def _sync_rec_toggle_text(self):
        self.btn_rec_npu_toggle.setText("Recognition: NPU ON" if self.cfg.rec_use_npu else "Recognition: NPU OFF")

    @Slot(bool)
    def on_dbface_toggle_changed(self, checked: bool):
        self.cfg.dbface_force_cpu = not bool(checked)
        self._sync_dbface_toggle_text()
        self.append_log("DBFace backend toggled. Click 'Load / Reload Models' to apply.")

    @Slot(bool)
    def on_rec_toggle_changed(self, checked: bool):
        self.cfg.rec_use_npu = bool(checked)
        self._sync_rec_toggle_text()
        self.append_log("Recognition backend toggled. Click 'Load / Reload Models' to apply.")

    @Slot()
    def load_models(self):
        try:
            # Stop camera before reloading interpreters
            if self.camera_worker.isRunning():
                self.stop_camera()

            self.cfg.dbface_model = self.ed_dbface_model.text().strip()
            self.cfg.rec_model = self.ed_rec_model.text().strip()
            self.cfg.dbface_delegate = self.ed_dbface_delegate.text().strip()
            self.cfg.rec_delegate = self.ed_rec_delegate.text().strip()
            self.cfg.threads = int(self.sp_threads.value())
            self.cfg.det_thresh = float(self.sp_det_thresh.value())
            self.cfg.nms_thresh = float(self.sp_nms_thresh.value())

            if not os.path.exists(self.cfg.dbface_model):
                raise FileNotFoundError(f"DBFace model not found: {self.cfg.dbface_model}")

            if not os.path.exists(self.cfg.rec_model):
                raise FileNotFoundError(f"Recognition model not found: {self.cfg.rec_model}")

            # DBFace: CPU or NPU based on toggle
            self.append_log("Loading DBFace detector...")
            self.detector = DBFace(
                model_file=self.cfg.dbface_model,
                nms_thresh=self.cfg.nms_thresh,
                use_npu=not self.cfg.dbface_force_cpu,
                delegate_path=self.cfg.dbface_delegate,
                num_threads=self.cfg.threads,
            )

            # Recognition: CPU or NPU based on toggle
            self.append_log("Loading recognition model...")
            rec_delegate = None
            if self.cfg.rec_use_npu:
                rec_delegate = try_load_delegate(self.cfg.rec_delegate)

            self.recognizer = FaceRecognitionTFLiteInt8(
                self.cfg.rec_model,
                num_threads=self.cfg.threads,
                delegate=rec_delegate,
                rgb=True,
            )

            self.camera_worker.configure_models(self.detector, self.recognizer)

            dbface_backend = "NPU (delegate)" if not self.cfg.dbface_force_cpu else "CPU"
            self.append_log(f"DBFace backend: {dbface_backend}")
            self.append_log(f"DBFace input : {self.detector.input_size} (W,H)")
            self.append_log(f"REC backend : {'NPU (delegate)' if rec_delegate else 'CPU'}")

            QMessageBox.information(self, "OK", "Models loaded successfully.")

        except Exception as e:
            QMessageBox.critical(self, "Model load failed", f"{e}\n\n{traceback.format_exc()}")

    @Slot()
    def browse_dataset_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select dataset folder", self.ed_dataset_dir.text())
        if d:
            self.ed_dataset_dir.setText(d)

    @Slot()
    def browse_build_dataset_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select dataset folder", self.ed_build_dataset.text())
        if d:
            self.ed_build_dataset.setText(d)

    @Slot()
    def browse_db_path(self):
        fn, _ = QFileDialog.getSaveFileName(self, "Select DB file", self.ed_db_path.text(), "NumPy DB (*.npy)")
        if fn:
            if not fn.endswith(".npy"):
                fn += ".npy"
            self.ed_db_path.setText(fn)

    @Slot()
    def browse_rec_db(self):
        fn, _ = QFileDialog.getOpenFileName(self, "Select DB file", self.ed_rec_db.text(), "NumPy DB (*.npy)")
        if fn:
            self.ed_rec_db.setText(fn)

    def _start_camera_worker(self):
        if self.camera_worker.isRunning():
            self.append_log("Stopping previous camera session...")
            self.camera_worker.stop()
            self.camera_worker.wait(1500)

        self.camera_worker.start()

    @Slot()
    def start_capture_mode(self):
        if not self._require_models_loaded():
            return

        dataset_dir = self.ed_dataset_dir.text().strip()
        person = self.ed_person_name.text().strip()

        if not person:
            QMessageBox.warning(self, "Missing name", "Enter person name.")
            return

        cam = int(self.sp_cam_capture.value())
        det_thresh = float(self.sp_det_thresh.value())

        self.camera_worker.set_mode_capture(cam, dataset_dir, person, det_thresh)
        self._start_camera_worker()

    @Slot()
    def save_capture_frame(self):
        if not self.camera_worker.isRunning():
            QMessageBox.warning(self, "Not running", "Start capture preview first.")
            return

        ok, msg = self.camera_worker.save_current_frame_to_dataset()

        if ok:
            self.append_log(f"Saved: {msg}")
        else:
            QMessageBox.warning(self, "Save failed", msg)

    @Slot()
    def build_db(self):
        if not self._require_models_loaded():
            return

        if self.recognizer is None:
            QMessageBox.warning(self, "Recognizer missing", "Load recognition model first.")
            return

        if self.build_worker is not None and self.build_worker.isRunning():
            QMessageBox.information(self, "Busy", "Build DB already running.")
            return

        dataset_dir = self.ed_build_dataset.text().strip()
        db_path = self.ed_db_path.text().strip()
        det_thresh = float(self.sp_det_thresh.value())

        if not os.path.isdir(dataset_dir):
            QMessageBox.warning(self, "Invalid dataset", f"Dataset folder not found: {dataset_dir}")
            return

        self.append_log(f"Building DB from: {dataset_dir}")

        self.build_worker = BuildDbWorker(dataset_dir, db_path, self.detector, self.recognizer, det_thresh)
        self.build_worker.log.connect(self.append_log)
        self.build_worker.done.connect(self.on_build_done)
        self.build_worker.start()

    @Slot(bool, str)
    def on_build_done(self, ok: bool, msg: str):
        if ok:
            self.append_log(msg)
            QMessageBox.information(self, "DB build done", msg)
            self.ed_rec_db.setText(self.ed_db_path.text().strip())
        else:
            QMessageBox.critical(self, "DB build failed", msg)

    @Slot()
    def start_recognize_mode(self):
        if not self._require_models_loaded():
            return

        if self.recognizer is None:
            QMessageBox.warning(self, "Recognizer missing", "Load recognition model first.")
            return

        cam = int(self.sp_cam_rec.value())
        db_path = self.ed_rec_db.text().strip()
        threshold = float(self.sp_threshold.value())
        det_thresh = float(self.sp_det_thresh.value())
        max_faces = int(self.sp_max_faces.value())

        if not os.path.exists(db_path):
            QMessageBox.warning(self, "Missing DB", f"DB not found: {db_path}")
            return

        self.camera_worker.set_mode_recognize(cam, db_path, threshold, det_thresh, max_faces)
        self._start_camera_worker()

    @Slot()
    def stop_camera(self):
        if self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker.wait(1500)

    @Slot(QImage)
    def on_frame(self, img: QImage):
        idx = self.tabs.currentIndex()
        target = self.cap_video if self.tabs.tabText(idx) == "Capture" else self.rec_video if self.tabs.tabText(idx) == "Recognize" else None

        if target is None:
            target = self.cap_video

        pix = QPixmap.fromImage(img)
        target.setPixmap(
            pix.scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    @Slot(str)
    def on_worker_error(self, msg: str):
        self.append_log("ERROR:\n" + msg)
        QMessageBox.critical(self, "Runtime error", msg)

    def closeEvent(self, event):
        try:
            self.stop_camera()
            if self.build_worker is not None and self.build_worker.isRunning():
                self.build_worker.wait(1500)
        finally:
            super().closeEvent(event)

    @Slot()
    def upload_images_to_dataset(self):
        dataset_dir = self.ed_dataset_dir.text().strip()
        person = self.ed_person_name.text().strip()

        if not dataset_dir:
            QMessageBox.warning(self, "Invalid dataset", "Dataset folder is empty.")
            return

        if not person:
            QMessageBox.warning(self, "Missing name", "Enter person name.")
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select images to add",
            "",
            "Images (*.jpg *.jpeg *.png *.bmp)"
        )

        if not files:
            return

        person_dir = os.path.join(dataset_dir, person)
        os.makedirs(person_dir, exist_ok=True)

        copied = 0
        skipped = 0

        for src in files:
            try:
                ext = os.path.splitext(src)[1].lower()
                if ext not in IMG_EXTS:
                    skipped += 1
                    continue

                # Avoid overwrite: prefix with timestamp
                dst = os.path.join(
                    person_dir,
                    f"{int(time.time() * 1000)}_{os.path.basename(src)}"
                )

                shutil.copy2(src, dst)
                copied += 1
                self.append_log(f"[UPLOAD] {person}: {os.path.basename(dst)}")

            except Exception as e:
                skipped += 1
                self.append_log(f"[SKIP] {src} ({e})")

        QMessageBox.information(
            self,
            "Upload complete",
            f"Uploaded: {copied}\nSkipped: {skipped}\n\n"
            f"Location:\n{person_dir}\n\n"
            f"Now rebuild DB to include these images."
        )

    @Slot()
    def delete_person_from_dataset(self):
        dataset_dir = self.ed_dataset_dir.text().strip()
        person = self.ed_person_name.text().strip()

        if not dataset_dir:
            QMessageBox.warning(self, "Invalid dataset", "Dataset folder is empty.")
            return

        if not person:
            QMessageBox.warning(self, "Missing name", "Enter person name to delete.")
            return

        person_dir = os.path.join(dataset_dir, person)

        if not os.path.isdir(person_dir):
            QMessageBox.information(self, "Not found", f"Person folder not found:\n{person_dir}")
            return

        if self.camera_worker.isRunning():
            self.stop_camera()

        msg = (
            f"Delete this person from dataset?\n\n"
            f"Person: {person}\n"
            f"Folder: {person_dir}\n\n"
            f"This will permanently delete all images for this person."
        )

        ret = QMessageBox.question(
            self,
            "Confirm delete",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if ret != QMessageBox.StandardButton.Yes:
            return

        try:
            shutil.rmtree(person_dir)
            self.append_log(f"Deleted person dataset folder: {person_dir}")

            QMessageBox.information(
                self,
                "Deleted",
                f"Deleted:\n{person_dir}\n\nNow rebuild DB to remove embeddings from the DB file."
            )

        except Exception as e:
            QMessageBox.critical(self, "Delete failed", f"{e}\n\n{traceback.format_exc()}")


APP_QSS = """
/* ---------------- GLOBAL ---------------- */
QWidget {
    background-color: #121212;
    color: #E0E0E0;
    font-family: "Segoe UI", "Ubuntu", "Arial";
    font-size: 12px;
}

/* ---------------- MAIN WINDOW ---------------- */
QMainWindow {
    background-color: #121212;
}

/* ---------------- TABS ---------------- */
QTabWidget::pane {
    border: 1px solid #2A2A2A;
    background: #161616;
}

QTabBar::tab {
    background: #1E1E1E;
    padding: 10px 18px;
    margin: 2px;
    border-radius: 6px;
    color: #B0B0B0;
}

QTabBar::tab:selected {
    background: #2979FF;
    color: #FFFFFF;
}

QTabBar::tab:hover {
    background: #333333;
}

/* ---------------- GROUP BOX ---------------- */
QGroupBox {
    border: 1px solid #2F2F2F;
    border-radius: 8px;
    margin-top: 12px;
    padding: 10px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: #90CAF9;
}

/* ---------------- BUTTONS ---------------- */
QPushButton {
    background-color: #2A2A2A;
    border: 1px solid #3A3A3A;
    padding: 8px 14px;
    border-radius: 6px;
}

QPushButton:hover {
    background-color: #3A3A3A;
}

QPushButton:pressed {
    background-color: #2979FF;
    border-color: #2979FF;
}

/* Toggle button (NPU ON/OFF) */
QPushButton:checked {
    background-color: #2E7D32;
    border-color: #2E7D32;
    color: #FFFFFF;
}

/* ---------------- INPUTS ---------------- */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #1C1C1C;
    border: 1px solid #3A3A3A;
    padding: 6px;
    border-radius: 5px;
}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: #2979FF;
}

/* ---------------- TEXT EDIT (LOG) ---------------- */
QTextEdit {
    background-color: #0D0D0D;
    border: 1px solid #2F2F2F;
    font-family: Consolas, monospace;
    font-size: 11px;
}

/* ---------------- LABEL (VIDEO) ---------------- */
QLabel {
    background-color: #0E0E0E;
    border: 1px solid #2A2A2A;
}

/* ---------------- SCROLL BAR ---------------- */
QScrollBar:vertical {
    background: #121212;
    width: 10px;
}

QScrollBar::handle:vertical {
    background: #3A3A3A;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #555555;
}
"""


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_QSS)

    w = MainWindow()
    w.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
