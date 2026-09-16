import sys
import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = "models/plant_disease_mobilenetv2.keras"

CLASS_NAMES = [
    "early_blight",
    "healthy",
    "late_blight"
]

IMAGE_SIZE = (224, 224)


def predict_image(image_path):

    model = tf.keras.models.load_model(MODEL_PATH)

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_index]

    predicted_class = CLASS_NAMES[predicted_index]

    return predicted_class, confidence


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python src/predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    disease, confidence = predict_image(image_path)

    print()
    print("Plant Disease Detection")
    print("-----------------------")
    print("Image:", image_path)
    print("Prediction:", disease)
    print("Confidence:", f"{confidence * 100:.2f}%")