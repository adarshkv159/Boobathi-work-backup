import cv2
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

# For int8 quantized model
model_path = "models/model_full_integer_quant.tflite"
model_type = ModelType.TUSIMPLE

# Initialize lane detection model with NPU support and int8 quantization
# Set use_npu=False to force CPU execution
# Set model_dtype='float32' for float32 models
lane_detector = UltrafastLaneDetector(model_path, model_type, use_npu=True, model_dtype='int8')

# Initialize webcam
cap = cv2.VideoCapture("input.mp4")
cv2.namedWindow("Detected lanes", cv2.WINDOW_NORMAL)

while(True):
    ret, frame = cap.read()
    if not ret:
        break
        
    # Detect the lanes
    output_img = lane_detector.detect_lanes(frame)
    
    cv2.imshow("Detected lanes", output_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
