import os

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request
from nlp_classifier import predict_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "plant_disease_mobilenetv2.keras"
)

CLASS_NAMES = [
    "early_blight",
    "healthy",
    "late_blight"
]

IMAGE_SIZE = (224, 224)


# Load the model once when Flask starts
model = tf.keras.models.load_model(MODEL_PATH)


def predict_image(image):

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image, dtype=np.float32)

    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )

    image_array = np.expand_dims(image_array, axis=0)

    predictions = model.predict(image_array, verbose=0)

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[0][predicted_index]

    return predicted_class, confidence


@app.route("/", methods=["GET", "POST"])
def home():

    image_prediction = None
    image_confidence = None

    text_prediction = None
    text_confidence = None

    if request.method == "POST":

        # Image prediction
        if "image" in request.files:

            image_file = request.files["image"]

            if image_file.filename != "":

                image = Image.open(image_file)

                image_prediction, image_confidence = predict_image(
                    image
                )

                image_confidence = round(
                    float(image_confidence) * 100,
                    2
                )

        # NLP prediction
        symptoms = request.form.get("symptoms", "").strip()

        if symptoms:

            text_prediction, text_confidence = predict_text(
                symptoms
            )

            text_confidence = round(
                float(text_confidence) * 100,
                2
            )

    return render_template(
        "index.html",
        image_prediction=image_prediction,
        image_confidence=image_confidence,
        text_prediction=text_prediction,
        text_confidence=text_confidence
    )


if __name__ == "__main__":
    app.run(debug=True)