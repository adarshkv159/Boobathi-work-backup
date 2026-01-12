#!/usr/bin/env python3
"""
Simple TFLite people-tracking demo.

This script loads a TFLite detection model (SSD-like) and runs inference on a
camera or video file, filters detections for the "person" class, and tracks
them using norfair.

Usage examples:
  python people_tracker_tflite.py --model ssdlite_mobiledet_dsp_320x320_coco_20200519.tflite
  python people_tracker_tflite.py --model /path/to/model.tflite --source 0

Notes / assumptions:
- The TFLite model should follow the common TFLite Detection PostProcess
  output signatures (boxes, classes, scores, num_detections). The loader tries
  to detect these outputs robustly but may need adjustments for custom models.
- Default person class id is 0. If your model uses 1 for 'person', pass
  --person_id 1.

"""
import argparse
import time
import numpy as np
import cv2

try:
    import tflite_runtime.interpreter as tflite
except Exception:
    try:
        from tensorflow.lite import Interpreter as tflite
    except Exception as e:
        raise RuntimeError(
            "No TFLite runtime found. Install tflite-runtime or TensorFlow.")

# Prefer norfair for tracking; fall back to a tiny centroid tracker if it's
# not installed so the script remains runnable in minimal environments.
try:
    from norfair import Detection, Tracker
    NORFAIR_AVAILABLE = True
except Exception:
    NORFAIR_AVAILABLE = False
    # Minimal Detection compatibility shim
    class Detection:
        def __init__(self, points, scores=None, label=None):
            # points expected as array-like of shape (N,2)
            self.points = np.asarray(points)
            self.scores = scores
            self.label = label

    # Very small centroid-based tracker that exposes a compatible `update`
    # method returning objects with `.estimate` (np.array Nx2) and `.id`.
    class _SimpleTrackedObject:
        def __init__(self, oid, estimate):
            self.id = oid
            self.estimate = np.asarray(estimate)

    class Tracker:
        def __init__(self, distance_threshold=50, **kwargs):
            # distance_threshold in pixels (euclidean)
            self.distance_threshold = distance_threshold
            self.next_id = 0
            # map id -> (x,y)
            self.objects = {}

        def update(self, detections=None):
            # detections: list of Detection-like objects with .points
            new_objects = {}
            tracked = []
            if detections is None:
                return tracked

            det_centroids = []
            for d in detections:
                pts = np.asarray(d.points)
                # compute centroid of points
                try:
                    cx = float(np.mean(pts[:, 0]))
                    cy = float(np.mean(pts[:, 1]))
                except Exception:
                    continue
                det_centroids.append((cx, cy))

            # naive greedy matching: for each detection find nearest existing obj
            used_ids = set()
            for (cx, cy) in det_centroids:
                best_id = None
                best_dist = None
                for oid, (ox, oy) in self.objects.items():
                    if oid in used_ids:
                        continue
                    d2 = (ox - cx) * (ox - cx) + (oy - cy) * (oy - cy)
                    if best_dist is None or d2 < best_dist:
                        best_dist = d2
                        best_id = oid

                if best_id is None or best_dist is None or best_dist > (self.distance_threshold ** 2):
                    oid = self.next_id
                    self.next_id += 1
                else:
                    oid = best_id

                used_ids.add(oid)
                new_objects[oid] = (cx, cy)
                tracked.append(_SimpleTrackedObject(oid, [[cx, cy]]))

            # replace objects with new_objects (simple lifetime handling)
            self.objects = new_objects
            return tracked


def load_interpreter(model_path):
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    return interp


def get_io_details(interp):
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    return input_details, output_details


def preprocess(frame, input_shape, dtype):
    # expect input_shape like [1, H, W, C]
    h = int(input_shape[1])
    w = int(input_shape[2])
    img = cv2.resize(frame, (w, h))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if dtype == np.uint8:
        inp = img_rgb.astype(np.uint8)
    else:
        inp = img_rgb.astype(np.float32) / 255.0
    inp = np.expand_dims(inp, axis=0)
    return inp


def parse_outputs(output_dict):
    # Try to find boxes, classes, scores, num_detections
    boxes = None
    classes = None
    scores = None
    num = None

    for name, arr in output_dict.items():
        a = np.asarray(arr)
        if a.ndim == 3 and a.shape[-1] == 4:
            boxes = a
        elif a.ndim == 2 and a.shape[0] == 1 and a.shape[1] > 0:
            # could be classes or scores
            # determine by range: classes often near integers, scores in [0,1]
            if np.all((a >= 0) & (a <= 1.0)):
                # likely scores
                if scores is None:
                    scores = a
                else:
                    # ambiguous - keep first as scores
                    pass
            else:
                if classes is None:
                    classes = a
        elif a.ndim == 1 and a.size == 1:
            num = a

    # fallback: use positional ordering if above didn't find everything
    if boxes is None or scores is None or classes is None:
        arrays = list(output_dict.values())
        arrays = [np.asarray(a) for a in arrays]
        # common pattern: [1,num,4], [1,num], [1,num], [1]
        by_ndim = sorted(arrays, key=lambda x: x.ndim, reverse=True)
        if len(by_ndim) >= 3:
            boxes = by_ndim[0]
            # find 1D or 2D arrays of shape (1,N)
            rest = [a for a in by_ndim[1:] if a.ndim == 2 or a.ndim == 1]
            if len(rest) >= 2:
                classes = rest[0]
                scores = rest[1]
            if len(by_ndim) >= 4:
                num = by_ndim[3]

    return boxes, classes, scores, num


def detections_from_outputs(boxes, classes, scores, num, frame_shape, score_thresh=0.5, person_id=0):
    # convert outputs to a list of dicts with bbox coords in pixel space
    h, w = frame_shape[0], frame_shape[1]
    dets = []
    if boxes is None or scores is None or classes is None:
        return dets

    boxes = np.squeeze(boxes)
    scores = np.squeeze(scores)
    classes = np.squeeze(classes)

    # if num provided, clamp
    count = None
    if num is not None:
        try:
            count = int(np.squeeze(num))
        except Exception:
            count = None

    N = boxes.shape[0]
    if count is not None:
        N = min(N, count)

    for i in range(N):
        score = float(np.squeeze(scores[i]))
        class_id = int(np.squeeze(classes[i]))
        if score < score_thresh:
            continue
        # filter person class
        if class_id != person_id:
            continue

        # boxes from TFLite detection usually are [ymin, xmin, ymax, xmax] normalized
        box = boxes[i]
        if box.max() <= 1.01:  # normalized
            ymin, xmin, ymax, xmax = box
            x1 = int(xmin * w)
            y1 = int(ymin * h)
            x2 = int(xmax * w)
            y2 = int(ymax * h)
        else:
            # absolute coords
            ymin, xmin, ymax, xmax = box
            x1, y1, x2, y2 = int(xmin), int(ymin), int(xmax), int(ymax)

        dets.append({
            "bbox": [x1, y1, x2, y2],
            "score": score,
            "class_id": class_id,
        })

    return dets


def draw_detections(frame, dets, tracked_objects=None):
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 200, 10), 2)
        cv2.putText(frame, "P:{:.2f}".format(d["score"]), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    # draw tracks
    if tracked_objects is not None:
        for tobj in tracked_objects:
            # norfair tracked object has .points and .id
            pts = tobj.estimate
            # estimate may be shape (N, 2) or (2,) etc.
            try:
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.mean(pts[:, 1]))
            except Exception:
                continue
            cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)
            cv2.putText(frame, f"ID:{tobj.id}", (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .tflite model")
    parser.add_argument("--source", default=0, help="Video source (0 for webcam or path to video file)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Detection score threshold")
    parser.add_argument("--person_id", type=int, default=0, help="Class id used for 'person' in the model (default 0)")
    parser.add_argument("--show_fps", action="store_true", help="Show FPS on display")
    args = parser.parse_args()

    interp = load_interpreter(args.model)
    input_details, output_details = get_io_details(interp)

    input_shape = input_details[0]["shape"]
    input_dtype = np.dtype(input_details[0]["dtype"])

    cap = None
    try:
        src = int(args.source)
    except Exception:
        src = args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    tracker = Tracker(distance_function="iou", distance_threshold=0.7)

    prev = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        inp = preprocess(frame, input_shape, input_dtype)
        interp.set_tensor(input_details[0]["index"], inp)
        interp.invoke()

        outputs = {}
        for od in output_details:
            outputs[od.get("name", str(od.get("index")))] = interp.get_tensor(od["index"])

        boxes, classes, scores, num = parse_outputs(outputs)
        dets = detections_from_outputs(boxes, classes, scores, num, (h, w), score_thresh=args.threshold, person_id=args.person_id)

        # convert to norfair detections (two points: top-left and bottom-right)
        nf_dets = []
        for d in dets:
            x1, y1, x2, y2 = d["bbox"]
            points = np.array([[x1, y1], [x2, y2]], dtype=np.float32)
            scores = np.array([d["score"], d["score"]], dtype=np.float32)
            nf_dets.append(Detection(points=points, scores=scores, label=d["class_id"]))

        tracked = tracker.update(detections=nf_dets)

        out = frame.copy()
        out = draw_detections(out, dets, tracked)

        if args.show_fps:
            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / (now - prev)) if prev != 0 else 0.0
            prev = now
            cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        cv2.imshow("People Tracker", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
