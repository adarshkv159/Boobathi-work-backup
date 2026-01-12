import argparse
import cv2 as cv
import numpy as np
import tensorflow as tf


def run_inference(interpreter, input_size, image):
    # Preprocess
    input_image = cv.resize(image, (input_size[1], input_size[0]))
    input_image = input_image.astype(np.float32)
    input_image = np.expand_dims(input_image, axis=0)

    # Inference
    input_details = interpreter.get_input_details()
    interpreter.set_tensor(input_details[0]['index'], input_image)
    interpreter.invoke()

    output_details = interpreter.get_output_details()
    embedding = interpreter.get_tensor(output_details[0]['index'])[0]

    return embedding


def cosine_similarity(vec1, vec2):
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    return np.dot(vec1, vec2)


def is_same_person(emb1, emb2, threshold=0.7):
    similarity = cosine_similarity(emb1, emb2)
    same = similarity >= threshold
    return same, similarity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model_float16_quant.tflite")
    parser.add_argument("--input_size", type=str, default="256,128")
    parser.add_argument("--threshold", type=float, default=0.7)

    args = parser.parse_args()
    input_size = [int(i) for i in args.input_size.split(",")]

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    # Read images
    img1 = cv.imread("image-1.jpeg")
    img2 = cv.imread("image-2.jpeg")

    if img1 is None or img2 is None:
        print("❌ Error: Unable to read input images")
        return

    # Extract embeddings
    emb1 = run_inference(interpreter, input_size, img1)
    emb2 = run_inference(interpreter, input_size, img2)

    # Compare
    same_person, similarity = is_same_person(emb1, emb2, args.threshold)

    print(f"🔹 Cosine Similarity: {similarity:.4f}")

    if same_person:
        print("✅ Same person")
    else:
        print("❌ Different persons")


if __name__ == "__main__":
    main()

