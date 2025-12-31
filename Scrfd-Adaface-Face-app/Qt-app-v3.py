import os
import sys
import glob
import time
import traceback
import shutil
from dataclasses import dataclass
from typing import Optional, Tuple, List
from collections import defaultdict

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
    QMessageBox, QCheckBox
)

DB_DEFAULT = "database.npy"
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# ---------------------------
# Alignment
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
# Image Quality Functions (NEW)
# ---------------------------
def is_blurry(img_bgr: np.ndarray, thresh: float = 80.0) -> Tuple[bool, float]:
    """
    Detect blur using Laplacian variance.
    Returns (is_blurry, blur_score)
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    score = cv.Laplacian(gray, cv.CV_64F).var()
    return score < thresh, score


def compute_sharpness(img_bgr: np.ndarray) -> float:
    """
    Compute sharpness score (higher = sharper).
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    return cv.Laplacian(gray, cv.CV_64F).var()


def is_face_size_valid(bbox, min_size: int = 80) -> bool:
    """
    Check if face bounding box meets minimum size requirement.
    """
    x1, y1, x2, y2 = bbox[:4]
    w = x2 - x1
    h = y2 - y1
    return w >= min_size and h >= min_size


# ---------------------------
# SCRFD detector
# ---------------------------
def distance2bbox(points, distance, max_shape=None):
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = np.clip(x1, 0, max_shape[1])
        y1 = np.clip(y1, 0, max_shape[0])
        x2 = np.clip(x2, 0, max_shape[1])
        y2 = np.clip(y2, 0, max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = np.clip(px, 0, max_shape[1])
            py = np.clip(py, 0, max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


class SCRFD:
    def __init__(self, model_file, nms_thresh=0.4, use_npu=False,
                 delegate_path="/usr/lib/libvx_delegate.so", num_threads=4):
        self.model_file = model_file
        self.center_cache = {}
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
                    print(f"[SCRFD] Loading delegate: {self.delegate_path}")
                    delegates.append(tflite.load_delegate(self.delegate_path))
                    print("[SCRFD] Delegate loaded")
                else:
                    print(f"[SCRFD] Delegate not found: {self.delegate_path} (CPU fallback)")
            except Exception as e:
                print(f"[SCRFD] Delegate load failed: {e} (CPU fallback)")

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
        input_cfg = self.interpreter.get_input_details()[0]
        input_shape = input_cfg['shape']  # [1, H, W, 3]
        # FIXED: Match standalone script
        self.input_size = tuple(input_shape[1:3][::-1])  # (W, H)

        outputs = self.interpreter.get_output_details()
        self.batched = len(outputs[0]['shape']) == 3

        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        self.output_map = {
            "stride_32": {"bbox_idx": 7, "score_idx": 1, "kps_idx": 2},
            "stride_16": {"bbox_idx": 0, "score_idx": 6, "kps_idx": 5},
            "stride_8":  {"bbox_idx": 3, "score_idx": 4, "kps_idx": 8},
        }
        print(f"[SCRFD] Input size (W,H): {self.input_size}")

    def forward(self, img, thresh):
        scores_list, bboxes_list, kpss_list = [], [], []
        input_size = tuple(img.shape[0:2][::-1])  # (W, H)

        blob = cv.dnn.blobFromImage(
            img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True
        )
        blob = blob.transpose(0, 2, 3, 1)  # NCHW->NHWC

        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]["index"], blob.astype(np.float32))
        self.interpreter.invoke()

        output_details = self.interpreter.get_output_details()
        input_height, input_width = blob.shape[1], blob.shape[2]

        for stride in self._feat_stride_fpn:
            mapping = self.output_map[f"stride_{stride}"]
            scores = self.interpreter.get_tensor(output_details[mapping["score_idx"]]["index"])
            bbox_preds = self.interpreter.get_tensor(output_details[mapping["bbox_idx"]]["index"])
            kps_preds = self.interpreter.get_tensor(output_details[mapping["kps_idx"]]["index"])

            if self.batched:
                scores, bbox_preds, kps_preds = scores[0], bbox_preds[0], kps_preds[0]

            bbox_preds = bbox_preds * stride
            kps_preds = kps_preds * stride

            if len(scores.shape) == 2 and scores.shape[1] == 1:
                scores = scores.flatten()

            height, width = input_height // stride, input_width // stride
            key = (height, width, stride)

            if key in self.center_cache:
                anchor_centers = self.center_cache[key]
            else:
                anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
                anchor_centers = (anchor_centers * stride).reshape((-1, 2))
                if self._num_anchors > 1:
                    anchor_centers = np.stack([anchor_centers] * self._num_anchors, axis=1).reshape((-1, 2))
                if len(self.center_cache) < 100:
                    self.center_cache[key] = anchor_centers

            pos_inds = np.where(scores >= thresh)[0]
            if pos_inds.size == 0:
                continue

            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])

            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])

        return scores_list, bboxes_list, kpss_list

    def nms(self, dets):
        x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def detect(self, img, thresh=0.5, input_size=None, max_num=0):
        input_size = self.input_size if input_size is None else input_size  # (W,H)

        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]

        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)

        det_scale = float(new_height) / img.shape[0]
        resized_img = cv.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)

        valid_scores = [s for s in scores_list if len(s) > 0]
        valid_bboxes = [b for b in bboxes_list if len(b) > 0]
        valid_kpss = [k for k in kpss_list if len(k) > 0]

        if len(valid_scores) == 0:
            return np.zeros((0, 5), dtype=np.float32), None

        scores = np.concatenate(valid_scores)
        bboxes = np.concatenate(valid_bboxes) / det_scale
        
        # FIXED: Ensure scores are 2D
        if len(scores.shape) == 1:
            scores = scores.reshape(-1, 1)
        scores = scores.astype(np.float32)

        kpss = np.concatenate(valid_kpss) / det_scale if len(valid_kpss) > 0 else None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32)
        order = pre_det[:, 4].argsort()[::-1]
        pre_det = pre_det[order, :]

        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        if kpss is not None:
            kpss = kpss[order][keep]

        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([
                (det[:, 0] + det[:, 2]) / 2 - img_center[1],
                (det[:, 1] + det[:, 3]) / 2 - img_center[0]
            ])
            values = area - np.sum(np.power(offsets, 2.0), 0) * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex]
            if kpss is not None:
                kpss = kpss[bindex]

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
            img = img[:, :, ::-1]

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
# DB helpers (ENHANCED)
# ---------------------------
def db_empty():
    return {
        "person_names": [],
        "templates": [],  # List of template groups per person (NEW)
        "embeddings": np.zeros((0, 512), dtype=np.float32),  # For backward compatibility
        "labels": np.zeros((0,), dtype=np.int32),
        "paths": np.array([], dtype=object),
        "use_templates": True,  # Flag to indicate template-based DB (NEW)
    }


def db_save(path: str, db: dict):
    np.save(path, db)


def db_load(path: str):
    if not os.path.exists(path):
        return db_empty()
    db = np.load(path, allow_pickle=True).item()
    
    # Backward compatibility
    if "use_templates" not in db:
        db["use_templates"] = False
    if "templates" not in db:
        db["templates"] = []
    
    return db


def db_add_embedding(db: dict, person_name: str, emb: np.ndarray, src_path: str):
    """Add single embedding (backward compatible)"""
    emb = l2_normalize(emb.astype(np.float32).reshape(-1))
    if person_name not in db["person_names"]:
        db["person_names"].append(person_name)
    person_id = db["person_names"].index(person_name)
    db["labels"] = np.concatenate([db["labels"], np.array([person_id], dtype=np.int32)], axis=0)
    db["embeddings"] = np.vstack([db["embeddings"], emb[None, :]]).astype(np.float32)
    db["paths"] = np.concatenate([db["paths"], np.array([src_path], dtype=object)], axis=0)


def db_build_with_templates(person_embeddings: dict, use_mean: bool = True, max_templates: int = 5):
    """
    Build database using template strategy (NEW - PRODUCTION QUALITY)
    
    Args:
        person_embeddings: dict mapping person_name -> list of (embedding, path) tuples
        use_mean: If True, store one mean embedding per person. If False, store multiple templates
        max_templates: Maximum number of templates to store per person (if not using mean)
    
    Returns:
        Database dict with optimized embeddings
    """
    db = db_empty()
    db["use_templates"] = True
    
    for person_name, emb_list in person_embeddings.items():
        if not emb_list:
            continue
            
        # Extract embeddings and paths
        embeddings = np.array([e[0] for e in emb_list])
        paths = [e[1] for e in emb_list]
        
        if use_mean:
            # BEST PRACTICE: Store mean embedding
            mean_emb = np.mean(embeddings, axis=0)
            mean_emb = l2_normalize(mean_emb)
            
            person_id = len(db["person_names"])
            db["person_names"].append(person_name)
            db["templates"].append([mean_emb])  # Single template
            db["embeddings"] = np.vstack([db["embeddings"], mean_emb[None, :]])
            db["labels"] = np.concatenate([db["labels"], [person_id]])
            db["paths"] = np.concatenate([db["paths"], [f"mean_of_{len(embeddings)}_images"]])
        else:
            # ADVANCED: Store multiple representative templates
            # Select diverse templates based on clustering or simply best N
            if len(embeddings) <= max_templates:
                selected_embs = embeddings
                selected_paths = paths
            else:
                # Simple approach: select by quality distribution
                # You could implement k-means clustering here for better diversity
                indices = np.linspace(0, len(embeddings) - 1, max_templates, dtype=int)
                selected_embs = embeddings[indices]
                selected_paths = [paths[i] for i in indices]
            
            person_id = len(db["person_names"])
            db["person_names"].append(person_name)
            db["templates"].append(selected_embs.tolist())
            
            # Add all selected templates
            for emb, path in zip(selected_embs, selected_paths):
                db["embeddings"] = np.vstack([db["embeddings"], emb[None, :]])
                db["labels"] = np.concatenate([db["labels"], [person_id]])
                db["paths"] = np.concatenate([db["paths"], [path]])
    
    return db


def match_identity(db: dict, emb: np.ndarray, threshold: float):
    """
    Match with template support (ENHANCED)
    """
    if db["embeddings"].shape[0] == 0:
        return "Unknown", None, None
    
    # Clip scores for numerical stability
    scores = np.clip(db["embeddings"] @ emb, -1.0, 1.0)
    
    if db.get("use_templates", False) and db.get("templates"):
        # Template-based matching: use max score per person
        person_max_scores = []
        for person_id in range(len(db["person_names"])):
            person_mask = db["labels"] == person_id
            if np.any(person_mask):
                max_score = np.max(scores[person_mask])
                person_max_scores.append((person_id, max_score))
        
        if person_max_scores:
            person_id, best_score = max(person_max_scores, key=lambda x: x[1])
            best_idx = np.where((db["labels"] == person_id) & (scores == best_score))[0][0]
            name = db["person_names"][person_id] if best_score >= threshold else "Unknown"
            return name, float(best_score), int(best_idx)
    
    # Fallback: single embedding matching
    best_idx = int(np.argmax(scores))
    best_score = float(scores[best_idx])
    person_id = int(db["labels"][best_idx])
    name = db["person_names"][person_id] if best_score >= threshold else "Unknown"
    return name, best_score, best_idx


# ---------------------------
# Dataset helpers (ENHANCED)
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


def compute_embedding_from_bgr(img_bgr, detector: SCRFD, recognizer: FaceRecognitionTFLiteInt8, 
                               det_thresh: float, min_face_size: int = 80, 
                               blur_thresh: float = 80.0, check_quality: bool = True):
    """
    Compute embedding with quality checks (ENHANCED)
    
    Returns:
        tuple: (embedding, quality_info) or (None, reason_str)
    """
    det, kpss = detector.detect(img_bgr, thresh=det_thresh, input_size=detector.input_size, max_num=1)
    
    if kpss is None or len(kpss) == 0:
        return (None, "no_face_detected")
    
    bbox = det[0]
    det_score = float(bbox[4])
    
    # Quality checks
    if check_quality:
        # Check face size
        if not is_face_size_valid(bbox, min_face_size):
            return (None, f"face_too_small (min={min_face_size}px)")
        
        # Check blur
        kpt = kpss[0].astype(np.float32)
        face = norm_crop_image(img_bgr, kpt, 112)
        
        blurry, blur_score = is_blurry(face, blur_thresh)
        if blurry:
            return (None, f"blurry (score={blur_score:.1f}, thresh={blur_thresh})")
    else:
        kpt = kpss[0].astype(np.float32)
        face = norm_crop_image(img_bgr, kpt, 112)
        blur_score = compute_sharpness(face)
    
    # Compute embedding
    emb = l2_normalize(recognizer(face).astype(np.float32))
    
    quality_info = {
        "det_score": det_score,
        "blur_score": blur_score,
        "face_size": (bbox[2] - bbox[0], bbox[3] - bbox[1])
    }
    
    return (emb, quality_info)


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
    scrfd_model: str = "model_float32.tflite"
    rec_model: str = "recognition_full_integer_quant.tflite"

    scrfd_force_cpu: bool = True
    rec_use_npu: bool = True
    rec_delegate: str = "/usr/lib/libvx_delegate.so"

    det_thresh: float = 0.5
    
    # Quality settings (NEW)
    enable_quality_filter: bool = True
    min_face_size: int = 80
    blur_threshold: float = 80.0
    
    # Template settings (NEW)
    use_template_averaging: bool = True
    max_templates_per_person: int = 5


class CameraWorker(QThread):
    frame_ready = Signal(QImage)
    status = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self._running = False

        self.mode = "preview"
        self.cam_index = 0

        self.detector: Optional[SCRFD] = None
        self.recognizer: Optional[FaceRecognitionTFLiteInt8] = None

        self.db_path: str = DB_DEFAULT
        self.threshold: float = 0.60
        self.det_thresh: float = 0.5
        self.max_faces: int = 0

        self.dataset_dir: str = "dataset"
        self.person_name: str = "person"

        self._last_frame_bgr: Optional[np.ndarray] = None
        
        # Quality settings (NEW)
        self.min_face_size: int = 80
        self.blur_threshold: float = 80.0

    def configure_models(self, detector: SCRFD, recognizer: Optional[FaceRecognitionTFLiteInt8]):
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
        fn = os.path.join(out_dir, f"{int(time.time() * 1000)}.jpg")
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
                template_info = " (template-based)" if db.get("use_templates", False) else ""
                self.status.emit(f"DB loaded: persons={len(db['person_names'])}, "
                               f"embeddings={db['embeddings'].shape[0]}{template_info}")

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
                            
                            # Check face size and quality
                            face_valid = is_face_size_valid(det[i], self.min_face_size)
                            color = (0, 255, 0) if face_valid else (0, 165, 255)  # Green or Orange
                            
                            cv.rectangle(view, (x1, y1), (x2, y2), color, 2)

                            if self.mode == "recognize":
                                kpt = kpss[i].astype(np.float32)
                                face = norm_crop_image(frame, kpt, 112)
                                
                                # Check blur
                                blurry, blur_score = is_blurry(face, self.blur_threshold)
                                
                                emb = l2_normalize(self.recognizer(face).astype(np.float32))
                                name, sim, _ = match_identity(db, emb, self.threshold)

                                ty = y1 - 10
                                if ty < 20:
                                    ty = y1 + 20

                                txt1 = f"{name} sim={sim:.2f}" if sim is not None else name
                                txt2 = f"det={det_score:.2f} blur={blur_score:.1f}"
                                quality_txt = "[BLUR]" if blurry else "[OK]"
                                
                                cv.putText(view, txt1, (x1, ty), cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                cv.putText(view, txt2, (x1, ty + 22), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                cv.putText(view, quality_txt, (x1, ty + 40), cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                                         (0, 255, 0) if not blurry else (0, 0, 255), 2)
                            else:
                                # Capture mode - show quality info
                                kpt = kpss[i].astype(np.float32)
                                face = norm_crop_image(frame, kpt, 112)
                                blurry, blur_score = is_blurry(face, self.blur_threshold)
                                
                                cv.putText(view, f"Face {i+1}", (x1, y1 - 10),
                                         cv.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                cv.putText(view, f"blur={blur_score:.1f}", (x1, y2 + 20),
                                         cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

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

    def __init__(self, dataset_dir: str, db_path: str, detector: SCRFD,
                 recognizer: FaceRecognitionTFLiteInt8, det_thresh: float,
                 config: AppConfig):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.db_path = db_path
        self.detector = detector
        self.recognizer = recognizer
        self.det_thresh = det_thresh
        self.config = config

    def run(self):
        try:
            # Collect embeddings per person (NEW APPROACH)
            person_embeddings = defaultdict(list)
            total = 0
            kept = 0
            rejected = defaultdict(int)

            for person, path in iter_dataset_images(self.dataset_dir):
                total += 1
                img = cv.imread(path)
                if img is None:
                    self.log.emit(f"[SKIP] cannot read: {path}")
                    rejected["read_error"] += 1
                    continue
                
                result = compute_embedding_from_bgr(
                    img, self.detector, self.recognizer, self.det_thresh,
                    min_face_size=self.config.min_face_size,
                    blur_thresh=self.config.blur_threshold,
                    check_quality=self.config.enable_quality_filter
                )
                
                if result[0] is None:
                    reason = result[1]
                    self.log.emit(f"[SKIP] {reason}: {os.path.basename(path)}")
                    rejected[reason] += 1
                    continue
                
                emb, quality_info = result
                person_embeddings[person].append((emb, path))
                kept += 1
                self.log.emit(f"[OK] {person}: {os.path.basename(path)} "
                            f"(blur={quality_info['blur_score']:.1f}, "
                            f"det={quality_info['det_score']:.2f})")

            # Build database with templates (NEW)
            if self.config.use_template_averaging:
                self.log.emit("\n[INFO] Building DB with mean template per person...")
                db = db_build_with_templates(person_embeddings, use_mean=True)
            else:
                self.log.emit(f"\n[INFO] Building DB with up to {self.config.max_templates_per_person} templates per person...")
                db = db_build_with_templates(person_embeddings, use_mean=False, 
                                            max_templates=self.config.max_templates_per_person)

            db_save(self.db_path, db)
            
            # Build summary
            rejection_summary = "\n".join([f"  {reason}: {count}" for reason, count in rejected.items()])
            msg = (f"Saved DB: {self.db_path}\n"
                   f"Persons: {len(db['person_names'])}\n"
                   f"Templates: {db['embeddings'].shape[0]}\n"
                   f"Images processed: {total}\n"
                   f"Images kept: {kept}\n"
                   f"Images rejected: {total - kept}\n"
                   f"Rejection reasons:\n{rejection_summary}\n"
                   f"Strategy: {'Mean embedding' if self.config.use_template_averaging else f'{self.config.max_templates_per_person} templates'} per person")
            
            self.done.emit(True, msg)
        except Exception as e:
            self.done.emit(False, f"{e}\n{traceback.format_exc()}")


# ---------------------------
# Main Window (ENHANCED UI)
# ---------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition")
        self.resize(1400, 900)

        self.detector: Optional[SCRFD] = None
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
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)

        main_layout = QHBoxLayout(root)
        main_layout.addWidget(self.tabs, 5)
        main_layout.addWidget(self.log_view, 1)

        # ---------------- Models tab ----------------
        tab_models = QWidget()
        self.tabs.addTab(tab_models, "Models")

        self.ed_scrfd_model = QLineEdit(self.cfg.scrfd_model)
        self.ed_rec_model = QLineEdit(self.cfg.rec_model)

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

        btn_load_models = QPushButton("Load / Reload Models")
        btn_load_models.clicked.connect(self.load_models)

        form = QFormLayout(tab_models)
        form.addRow("SCRFD model (.tflite):", self._file_row(self.ed_scrfd_model))
        form.addRow("Recognition model (.tflite):", self._file_row(self.ed_rec_model))
        form.addRow("Recognition delegate path:", self.ed_rec_delegate)
        form.addRow("", self.btn_rec_npu_toggle)
        form.addRow("Threads:", self.sp_threads)
        form.addRow("Detection threshold:", self.sp_det_thresh)
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

        # ---------------- Build DB tab (ENHANCED) ----------------
        tab_build = QWidget()
        self.tabs.addTab(tab_build, "Build DB")

        self.ed_build_dataset = QLineEdit("dataset")
        btn_browse_build_dataset = QPushButton("Browse")
        btn_browse_build_dataset.clicked.connect(self.browse_build_dataset_dir)

        self.ed_db_path = QLineEdit(DB_DEFAULT)
        btn_browse_db = QPushButton("Browse")
        btn_browse_db.clicked.connect(self.browse_db_path)
        
        # Quality settings (NEW)
        self.chk_enable_quality = QCheckBox("Enable quality filtering")
        self.chk_enable_quality.setChecked(self.cfg.enable_quality_filter)
        
        self.sp_min_face_size = QSpinBox()
        self.sp_min_face_size.setRange(40, 200)
        self.sp_min_face_size.setValue(self.cfg.min_face_size)
        self.sp_min_face_size.setSuffix(" px")
        
        self.sp_blur_thresh = QDoubleSpinBox()
        self.sp_blur_thresh.setRange(20.0, 200.0)
        self.sp_blur_thresh.setValue(self.cfg.blur_threshold)
        
        # Template settings (NEW)
        self.chk_use_mean = QCheckBox("Use mean embedding (recommended)")
        self.chk_use_mean.setChecked(self.cfg.use_template_averaging)
        self.chk_use_mean.setToolTip("Store one averaged embedding per person for best stability")
        
        self.sp_max_templates = QSpinBox()
        self.sp_max_templates.setRange(1, 10)
        self.sp_max_templates.setValue(self.cfg.max_templates_per_person)
        self.sp_max_templates.setEnabled(not self.cfg.use_template_averaging)
        self.chk_use_mean.toggled.connect(lambda checked: self.sp_max_templates.setEnabled(not checked))

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
        
        # Quality section
        f2.addRow("", QLabel("<b>Quality Filtering</b>"))
        f2.addRow("", self.chk_enable_quality)
        f2.addRow("Min face size:", self.sp_min_face_size)
        f2.addRow("Blur threshold:", self.sp_blur_thresh)
        
        # Template section
        f2.addRow("", QLabel("<b>Template Strategy</b>"))
        f2.addRow("", self.chk_use_mean)
        f2.addRow("Max templates (if not mean):", self.sp_max_templates)
        
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

    def _sync_rec_toggle_text(self):
        self.btn_rec_npu_toggle.setText("Recognition: NPU ON" if self.cfg.rec_use_npu else "Recognition: NPU OFF")

    @Slot(bool)
    def on_rec_toggle_changed(self, checked: bool):
        self.cfg.rec_use_npu = bool(checked)
        self._sync_rec_toggle_text()
        self.append_log("Recognition backend toggled. Click 'Load / Reload Models' to apply.")

    @Slot()
    def load_models(self):
        try:
            if self.camera_worker.isRunning():
                self.stop_camera()

            self.cfg.scrfd_model = self.ed_scrfd_model.text().strip()
            self.cfg.rec_model = self.ed_rec_model.text().strip()
            self.cfg.rec_delegate = self.ed_rec_delegate.text().strip()
            self.cfg.threads = int(self.sp_threads.value())
            self.cfg.det_thresh = float(self.sp_det_thresh.value())

            if not os.path.exists(self.cfg.scrfd_model):
                raise FileNotFoundError(f"SCRFD model not found: {self.cfg.scrfd_model}")
            if not os.path.exists(self.cfg.rec_model):
                raise FileNotFoundError(f"Recognition model not found: {self.cfg.rec_model}")

            self.append_log("Loading SCRFD (CPU only)...")
            self.detector = SCRFD(
                model_file=self.cfg.scrfd_model,
                nms_thresh=0.4,
                use_npu=False,
                delegate_path="",
                num_threads=self.cfg.threads,
            )

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

            self.append_log("SCRFD backend: CPU (forced)")
            self.append_log(f"SCRFD input  : {self.detector.input_size} (W,H)")
            self.append_log(f"REC backend  : {'NPU (delegate)' if rec_delegate else 'CPU'}")
            QMessageBox.information(self, "OK", "Models loaded successfully!")
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

        # Update config from UI
        self.cfg.enable_quality_filter = self.chk_enable_quality.isChecked()
        self.cfg.min_face_size = int(self.sp_min_face_size.value())
        self.cfg.blur_threshold = float(self.sp_blur_thresh.value())
        self.cfg.use_template_averaging = self.chk_use_mean.isChecked()
        self.cfg.max_templates_per_person = int(self.sp_max_templates.value())

        self.append_log(f"Building DB from: {dataset_dir}")
        self.append_log(f"Quality filter: {'ON' if self.cfg.enable_quality_filter else 'OFF'}")
        self.append_log(f"Template strategy: {'Mean embedding' if self.cfg.use_template_averaging else f'{self.cfg.max_templates_per_person} templates'}")
        
        self.build_worker = BuildDbWorker(dataset_dir, db_path, self.detector, self.recognizer, 
                                        det_thresh, self.cfg)
        self.build_worker.log.connect(self.append_log)
        self.build_worker.done.connect(self.on_build_done)
        self.build_worker.start()

    @Slot(bool, str)
    def on_build_done(self, ok: bool, msg: str):
        if ok:
            self.append_log("\n" + msg)
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
        
        # Pass quality settings to worker
        self.camera_worker.min_face_size = self.cfg.min_face_size
        self.camera_worker.blur_threshold = self.cfg.blur_threshold

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

/* ---------------- CHECKBOX ---------------- */
QCheckBox {
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #3A3A3A;
    border-radius: 4px;
    background-color: #1C1C1C;
}

QCheckBox::indicator:checked {
    background-color: #2979FF;
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

