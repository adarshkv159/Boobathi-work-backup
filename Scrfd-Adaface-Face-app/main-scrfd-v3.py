import os
import glob
import time
import argparse
import numpy as np
import cv2 as cv

import tflite_runtime.interpreter as tflite
from skimage.transform import SimilarityTransform

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
# SCRFD detector (from your float32-tflite.py) [file:94]
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
        input_shape = input_cfg["shape"]  # [1,H,W,3]
        self.input_size = (int(input_shape[2]), int(input_shape[1]))  # (W,H)

        outputs = self.interpreter.get_output_details()
        self.batched = len(outputs[0]["shape"]) == 3

        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True

        # from your file [file:94]
        self.output_map = {
            "stride_32": {"bbox_idx": 7, "score_idx": 1, "kps_idx": 2},
            "stride_16": {"bbox_idx": 0, "score_idx": 6, "kps_idx": 5},
            "stride_8":  {"bbox_idx": 3, "score_idx": 4, "kps_idx": 8},
        }

        print(f"[SCRFD] Input size (W,H): {self.input_size}")

    def forward(self, img, thresh):
        scores_list, bboxes_list, kpss_list = [], [], []
        input_size = tuple(img.shape[0:2][::-1])  # (W,H)

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

        scores = scores.reshape(-1, 1).astype(np.float32)
        kpss = np.concatenate(valid_kpss) / det_scale if len(valid_kpss) > 0 else None

        pre_det = np.hstack((bboxes, scores)).astype(np.float32)
        order = pre_det[:, 4].argsort()[::-1]
        pre_det = pre_det[order, :]

        keep = self.nms(pre_det)
        det = pre_det[keep, :]

        if kpss is not None:
            kpss = kpss[order][keep]

        # only limit if max_num > 0 [file:94]
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
# DB: multi-template
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
# Dataset
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
                               det_thresh: float):
    det, kpss = detector.detect(img_bgr, thresh=det_thresh, input_size=detector.input_size, max_num=1)
    if kpss is None or len(kpss) == 0:
        return None
    kpt = kpss[0].astype(np.float32)
    face = norm_crop_image(img_bgr, kpt, 112)
    emb = l2_normalize(recognizer(face).astype(np.float32))
    return emb


def build_db_from_dataset(dataset_dir: str, db_path: str,
                          detector: SCRFD, recognizer: FaceRecognitionTFLiteInt8,
                          det_thresh: float):
    db = db_empty()
    total = 0
    kept = 0

    for person, path in iter_dataset_images(dataset_dir):
        total += 1
        img = cv.imread(path)
        if img is None:
            print(f"[SKIP] cannot read: {path}")
            continue

        emb = compute_embedding_from_bgr(img, detector, recognizer, det_thresh)
        if emb is None:
            print(f"[SKIP] no face: {path}")
            continue

        db_add_embedding(db, person, emb, path)
        kept += 1
        print(f"[OK] {person}: {os.path.basename(path)}")

    db_save(db_path, db)
    print(f"Saved DB: {db_path} | persons={len(db['person_names'])} templates={db['embeddings'].shape[0]} (kept {kept}/{total})")


def capture_dataset_images(dataset_dir: str, person_name: str, cam: int,
                           detector: SCRFD, det_thresh: float):
    out_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam}")

    print("Capture dataset images (SPACE save, q quit)")
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det, _ = detector.detect(frame, thresh=det_thresh, input_size=detector.input_size, max_num=0)
        view = frame.copy()
        for i in range(len(det)):
            x1, y1, x2, y2 = det[i][:4].astype(int)
            cv.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv.putText(view, f"Capture {person_name} saved={idx} (SPACE)", (20, 40),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (50, 255, 50), 2)
        cv.imshow("Capture Dataset", view)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            fn = os.path.join(out_dir, f"{int(time.time())}_{idx}.jpg")
            cv.imwrite(fn, frame)
            print(f"Saved: {fn}")
            idx += 1

    cap.release()
    cv.destroyAllWindows()


# ---------------------------
# Multi-face live recognition (FIXED)
# ---------------------------
def recognize_live(db_path: str, cam: int,
                   detector: SCRFD, recognizer: FaceRecognitionTFLiteInt8,
                   threshold: float, det_thresh: float, max_faces: int = 0):
    db = db_load(db_path)
    if db["embeddings"].shape[0] == 0:
        raise RuntimeError(f"DB empty: {db_path} (build_db first)")

    cap = cv.VideoCapture(cam)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {cam}")

    print("Recognize live multi-face: press q to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        det, kpss = detector.detect(
            frame,
            thresh=det_thresh,
            input_size=detector.input_size,
            max_num=max_faces  # 0=all, or set e.g. 5
        )

        if kpss is not None and len(kpss) > 0:
            for i in range(len(det)):
                x1, y1, x2, y2 = det[i][:4].astype(int)
                det_score = float(det[i][4])

                kpt = kpss[i].astype(np.float32)
                face = norm_crop_image(frame, kpt, 112)
                emb = l2_normalize(recognizer(face).astype(np.float32))

                name, sim, best_idx = match_identity(db, emb, threshold)

                # bbox
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # label position for THIS face
                ty = y1 - 10
                if ty < 20:
                    ty = y1 + 20

                txt1 = f"{name} sim={sim:.2f}" if sim is not None else name
                txt2 = f"det={det_score:.2f}"

                cv.putText(frame, txt1, (x1, ty),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)
                cv.putText(frame, txt2, (x1, ty + 22),
                           cv.FONT_HERSHEY_SIMPLEX, 0.55, (50, 255, 50), 2)

        cv.imshow("Live Recognition", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--vx_delegate", default="/usr/lib/libvx_delegate.so")
    ap.add_argument("--scrfd_model", default="model_float32.tflite")
    ap.add_argument("--rec_model", default="recognition_full_integer_quant.tflite")

    ap.add_argument("--scrfd_npu", action="store_true")
    ap.add_argument("--rec_delegate", default="")
    ap.add_argument("--det_thresh", type=float, default=0.5)

    pc = sub.add_parser("capture_dataset")
    pc.add_argument("--dataset", default="dataset")
    pc.add_argument("--name", required=True)
    pc.add_argument("--cam", type=int, default=0)

    pb = sub.add_parser("build_db")
    pb.add_argument("--dataset", default="dataset")
    pb.add_argument("--db", default=DB_DEFAULT)

    pr = sub.add_parser("recognize")
    pr.add_argument("--db", default=DB_DEFAULT)
    pr.add_argument("--cam", type=int, default=0)
    pr.add_argument("--threshold", type=float, default=0.55)
    pr.add_argument("--max_faces", type=int, default=0, help="0=all faces, else limit")

    args = ap.parse_args()

    detector = SCRFD(
        model_file=args.scrfd_model,
        nms_thresh=0.4,
        use_npu=args.scrfd_npu,
        delegate_path=args.vx_delegate,
        num_threads=args.threads,
    )

    rec_delegate = try_load_delegate(args.rec_delegate)
    recognizer = FaceRecognitionTFLiteInt8(
        args.rec_model,
        num_threads=args.threads,
        delegate=rec_delegate,
        rgb=True,
    )

    print("SCRFD backend:", "NPU" if args.scrfd_npu else "CPU")
    print("SCRFD input  :", detector.input_size, "(W,H)")
    print("REC backend  :", args.rec_delegate if rec_delegate else "CPU")

    if args.cmd == "capture_dataset":
        capture_dataset_images(args.dataset, args.name, args.cam, detector, args.det_thresh)
    elif args.cmd == "build_db":
        build_db_from_dataset(args.dataset, args.db, detector, recognizer, args.det_thresh)
    elif args.cmd == "recognize":
        recognize_live(args.db, args.cam, detector, recognizer,
                       threshold=args.threshold, det_thresh=args.det_thresh, max_faces=args.max_faces)


if __name__ == "__main__":
    main()

