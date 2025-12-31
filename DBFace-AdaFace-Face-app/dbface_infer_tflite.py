import common
import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
from argparse import ArgumentParser
from tensorflow.lite.python.interpreter import Interpreter


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


def detect(interpreter, input_details, output_details, image, threshold=0.4, nms_iou=0.5):
    
    # Quantize input: uint8 input, no quantization params needed (direct pixel values)
    input_data = image.astype(np.uint8)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get quantized outputs
    lm_quant = interpreter.get_tensor(output_details[0]['index'])  # Identity:0 - uint8[1,120,160,10]
    hm_quant = interpreter.get_tensor(output_details[1]['index'])  # Identity_2:0 - uint8[1,120,160,1]
    box_quant = interpreter.get_tensor(output_details[2]['index'])  # Identity_1:0 - uint8[1,120,160,4]
    
    # Dequantize outputs using quantization parameters
    # Identity:0 (landmark): scale * (q - zero_point)
    lm_scale, lm_zero_point = output_details[0]['quantization']
    lm = (lm_quant.astype(np.float32) - lm_zero_point) * lm_scale
    
    # Identity_2:0 (heatmap): scale * (q - zero_point)
    hm_scale, hm_zero_point = output_details[1]['quantization']
    hm = (hm_quant.astype(np.float32) - hm_zero_point) * hm_scale
    
    # Identity_1:0 (box): scale * (q - zero_point)
    box_scale, box_zero_point = output_details[2]['quantization']
    box = (box_quant.astype(np.float32) - box_zero_point) * box_scale
    
    # Transpose to match expected format: (1,h,w,c) -> (1,c,h,w)
    lm = lm.transpose((0, 3, 1, 2))
    hm = hm.transpose((0, 3, 1, 2))
    box = box.transpose((0, 3, 1, 2))
    
    # Convert to torch tensors
    hm = torch.from_numpy(hm)
    box = torch.from_numpy(box)
    landmark = torch.from_numpy(lm)
    
    # Apply sigmoid to heatmap (face detection confidence)
    hm = torch.sigmoid(hm)
    
    # NMS pooling
    hm_pool = F.max_pool2d(hm, 3, 1, 1)
    scores, indices = ((hm == hm_pool).float() * hm).view(1, -1).cpu().topk(1000)
    hm_height, hm_width = hm.shape[2:]

    scores = scores.squeeze()
    indices = indices.squeeze()
    ys = list((indices / hm_width).int().data.numpy())
    xs = list((indices % hm_width).int().data.numpy())
    scores = list(scores.data.numpy())
    box = box.cpu().squeeze().data.numpy()
    landmark = landmark.cpu().squeeze().data.numpy()

    stride = 4
    objs = []
    for cx, cy, score in zip(xs, ys, scores):
        if score < threshold:
            break

        x, y, r, b = box[:, cy, cx]
        xyrb = (np.array([cx, cy, cx, cy]) + [-x, -y, r, b]) * stride
        x5y5 = landmark[:, cy, cx]
        x5y5 = (common.exp(x5y5 * 4) + ([cx]*5 + [cy]*5)) * stride
        box_landmark = list(zip(x5y5[:5], x5y5[5:]))
        objs.append(common.BBox(0, xyrb=xyrb, score=score, landmark=box_landmark))
    return nms(objs, iou=nms_iou)


def camera_demo():
    interpreter = Interpreter(model_path='model_full_integer_quant.tflite', num_threads=4)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model details for verification
    print("Input Details:")
    print(f"  Name: {input_details[0]['name']}")
    print(f"  Shape: {input_details[0]['shape']}")
    print(f"  Type: {input_details[0]['dtype']}")
    print(f"  Quantization: {input_details[0]['quantization']}")
    
    print("\nOutput Details:")
    for i, detail in enumerate(output_details):
        print(f"  Output {i}: {detail['name']}")
        print(f"    Shape: {detail['shape']}")
        print(f"    Type: {detail['dtype']}")
        print(f"    Quantization: {detail['quantization']}")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        
        # Resize to model input size: 640x480 (width x height)
        img = cv2.resize(frame, (640, 480))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img[np.newaxis, :, :, :]  # Add batch dimension: [1, 480, 640, 3]
        
        objs = detect(interpreter, input_details, output_details, img,threshold=0.63, nms_iou=0.5)

        for obj in objs:
            common.drawbbox(frame, obj)

        cv2.imshow("demo DBFace", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    camera_demo()

