#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import time
import argparse
import os
import cv2 as cv
import numpy as np

# Use tflite-runtime instead of tensorflow
try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite-runtime")
except ImportError:
    import tensorflow.lite as tflite
    print("tflite-runtime not found, falling back to tensorflow.lite")


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box."""
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
    """Decode distance prediction to keypoints."""
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
    def __init__(self, model_file, nms_thresh=0.4, 
                 use_npu=False, delegate_path="/usr/lib/libvx_delegate.so"):
        self.model_file = model_file
        self.interpreter = None
        self.use_npu = use_npu
        self.delegate_path = delegate_path
        self.delegate = None
        self.center_cache = {}
        self.nms_thresh = nms_thresh
        
        self._create_interpreter()
        self._init_vars()

    def _create_interpreter(self):
        """Create interpreter with optional NPU delegate"""
        delegates = []
        
        if self.use_npu:
            try:
                if os.path.exists(self.delegate_path):
                    print(f"Loading NPU delegate from {self.delegate_path}")
                    os.environ['USE_GPU_INFERENCE'] = '0'
                    ext_delegate_options = {}
                    
                    self.delegate = tflite.load_delegate(
                        self.delegate_path,
                        ext_delegate_options
                    )
                    delegates.append(self.delegate)
                    print("✓ NPU delegate loaded successfully")
                else:
                    print(f"✗ NPU delegate not found at {self.delegate_path}")
                    print("Falling back to CPU")
            except Exception as e:
                print(f"✗ Failed to load NPU delegate: {e}")
                print("Falling back to CPU")
        
        if delegates:
            self.interpreter = tflite.Interpreter(
                model_path=self.model_file,
                experimental_delegates=delegates
            )
            print("✓ Interpreter created with NPU delegate")
        else:
            self.interpreter = tflite.Interpreter(
                model_path=self.model_file,
                num_threads=4
            )
            print("✓ Interpreter created for CPU execution")
        
        self.interpreter.allocate_tensors()

    def _init_vars(self):
        input_cfg = self.interpreter.get_input_details()[0]
        input_shape = input_cfg['shape']
        self.input_size = tuple(input_shape[1:3][::-1])
        outputs = self.interpreter.get_output_details()
        
        self.batched = len(outputs[0]['shape']) == 3
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2
        self.use_kps = True
        
        print(f"✓ Model: {os.path.basename(self.model_file)}")
        print(f"  Input size: {self.input_size}")
        print(f"  FPN levels: {self.fmc}, Strides: {self._feat_stride_fpn}")
        print(f"  Anchors per location: {self._num_anchors}")

    def forward(self, img, thresh):
        scores_list = []
        bboxes_list = []
        kpss_list = []
        input_size = tuple(img.shape[0:2][::-1])
        
        # Preprocess
        blob = cv.dnn.blobFromImage(
            img, 1.0 / 128, input_size, (127.5, 127.5, 127.5), swapRB=True)
        blob = blob.transpose(0, 2, 3, 1)
        
        # Inference
        input_details = self.interpreter.get_input_details()
        self.interpreter.set_tensor(input_details[0]['index'], blob.astype(np.float32))
        self.interpreter.invoke()
        
        output_details = self.interpreter.get_output_details()
        input_height, input_width = blob.shape[1], blob.shape[2]
        
        # Output mapping: [2400,4], [600,1], [600,10], [9600,4], [9600,1], [2400,10], [2400,1], [600,4], [9600,10]
        output_map = {
            'stride_32': {'bbox_idx': 7, 'score_idx': 1, 'kps_idx': 2},
            'stride_16': {'bbox_idx': 0, 'score_idx': 6, 'kps_idx': 5},
            'stride_8': {'bbox_idx': 3, 'score_idx': 4, 'kps_idx': 8},
        }
        
        for stride in self._feat_stride_fpn:
            mapping = output_map[f'stride_{stride}']
            
            # Get outputs
            scores = self.interpreter.get_tensor(output_details[mapping['score_idx']]['index'])
            bbox_preds = self.interpreter.get_tensor(output_details[mapping['bbox_idx']]['index'])
            kps_preds = self.interpreter.get_tensor(output_details[mapping['kps_idx']]['index'])
            
            if self.batched:
                scores, bbox_preds, kps_preds = scores[0], bbox_preds[0], kps_preds[0]
            
            bbox_preds *= stride
            kps_preds *= stride
            
            if len(scores.shape) == 2 and scores.shape[1] == 1:
                scores = scores.flatten()
            
            # Generate anchors
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
            
            # Decode predictions
            pos_inds = np.where(scores >= thresh)[0]
            bboxes = distance2bbox(anchor_centers, bbox_preds)
            scores_list.append(scores[pos_inds])
            bboxes_list.append(bboxes[pos_inds])
            
            if self.use_kps:
                kpss = distance2kps(anchor_centers, kps_preds)
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                kpss_list.append(kpss[pos_inds])
        
        return scores_list, bboxes_list, kpss_list

    def detect(self, img, thresh=0.5, input_size=None, max_num=0):
        input_size = self.input_size if input_size is None else input_size
        
        # Resize with aspect ratio
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
        
        # Detect
        scores_list, bboxes_list, kpss_list = self.forward(det_img, thresh)
        
        # Handle empty
        valid_scores = [s for s in scores_list if len(s) > 0]
        valid_bboxes = [b for b in bboxes_list if len(b) > 0]
        valid_kpss = [k for k in kpss_list if len(k) > 0]
        
        if len(valid_scores) == 0:
            return np.zeros((0, 5)), None
        
        # Post-process
        scores = np.concatenate(valid_scores)
        bboxes = np.concatenate(valid_bboxes) / det_scale
        
        if len(scores.shape) == 1:
            scores = scores.reshape(-1, 1)
        
        if len(valid_kpss) > 0:
            kpss = np.concatenate(valid_kpss) / det_scale
        else:
            kpss = None
        
        pre_det = np.hstack((bboxes, scores)).astype(np.float32)
        order = pre_det[:, 4].argsort()[::-1]
        pre_det = pre_det[order, :]
        
        # NMS
        keep = self.nms(pre_det)
        det = pre_det[keep, :]
        
        if kpss is not None:
            kpss = kpss[order][keep]
        
        # Limit faces
        if max_num > 0 and det.shape[0] > max_num:
            area = (det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1])
            img_center = img.shape[0] // 2, img.shape[1] // 2
            offsets = np.vstack([(det[:, 0] + det[:, 2]) / 2 - img_center[1],
                                (det[:, 1] + det[:, 3]) / 2 - img_center[0]])
            values = area - np.sum(np.power(offsets, 2.0), 0) * 2.0
            bindex = np.argsort(values)[::-1][:max_num]
            det = det[bindex]
            if kpss is not None:
                kpss = kpss[bindex]
        
        return det, kpss

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


def get_args():
    parser = argparse.ArgumentParser(description='SCRFD Face Detection')
    parser.add_argument("--device", type=int, default=0, help="Camera device ID")
    parser.add_argument("--movie", type=str, default=None, help="Video file path")
    parser.add_argument("--model", type=str, default='model_float32.tflite', help="Model path")
    parser.add_argument("--input_size", type=str, default='640,480', help="Input size W,H")
    parser.add_argument("--score_th", type=float, default=0.5, help="Score threshold")
    parser.add_argument("--nms_th", type=float, default=0.4, help="NMS threshold")
    parser.add_argument("--use_npu", action="store_true", help="Use NPU acceleration")
    parser.add_argument("--delegate_path", type=str, default="/usr/lib/libvx_delegate.so")
    return parser.parse_args()


def main():
    args = get_args()
    input_size = [int(i) for i in args.input_size.split(',')]

    # Initialize camera
    cap = cv.VideoCapture(args.movie if args.movie else args.device)
    if not cap.isOpened():
        print("Error: Cannot open camera/video")
        return
    
    frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {frame_width}x{frame_height}\n")

    # Load model
    print(f"Loading model: {args.model}")
    detector = SCRFD(
        model_file=args.model,
        nms_thresh=args.nms_th,
        use_npu=args.use_npu,
        delegate_path=args.delegate_path
    )
    print("Model loaded successfully\n")

    # Performance tracking
    frame_count = 0
    total_time = 0
    fps_display = 0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        debug_image = copy.deepcopy(frame)

        # Detect faces
        bboxes, keypoints = detector.detect(
            frame,
            args.score_th,
            input_size=(input_size[0], input_size[1]),
        )

        elapsed_time = time.time() - start_time
        
        # Update FPS
        frame_count += 1
        total_time += elapsed_time
        if frame_count % 10 == 0:
            fps_display = 10 / total_time
            total_time = 0

        # Draw detections
        for index, bbox in enumerate(bboxes):
            x1, y1, x2, y2, score = bbox.astype(int)
            
            # Bounding box
            cv.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Score label
            label = f'{score:.2f}'
            label_size, _ = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv.rectangle(debug_image, (x1, y1 - label_size[1] - 4), 
                        (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv.putText(debug_image, label, (x1, y1 - 4),
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)

            # Keypoints
            if keypoints is not None and len(keypoints) > index:
                for keypoint in keypoints[index]:
                    kp_x, kp_y = keypoint.astype(int)
                    cv.circle(debug_image, (kp_x, kp_y), 3, (0, 0, 255), -1)

        # Display info
        info_text = [
            f"FPS: {fps_display:.1f}" if frame_count > 10 else "FPS: calculating...",
            f"Inference: {elapsed_time * 1000:.1f}ms",
            f"Faces: {len(bboxes)}",
            f"Backend: {'NPU' if args.use_npu else 'CPU'}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv.putText(debug_image, text, (10, y_offset),
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv.LINE_AA)
            y_offset += 30

        cv.imshow('SCRFD Face Detection', debug_image)
        
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        elif key == ord('s'):
            filename = f'screenshot_{int(time.time())}.jpg'
            cv.imwrite(filename, debug_image)
            print(f"Screenshot saved: {filename}")

    cap.release()
    cv.destroyAllWindows()
    print(f"\nTotal frames: {frame_count}")


if __name__ == '__main__':
    main()

