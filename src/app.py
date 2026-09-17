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

    predictions = model.predict(image_array, verbose=0)[0]

    predicted_index = np.argmax(predictions)

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = predictions[predicted_index]

    probabilities = {
        class_name: float(probability)
        for class_name, probability in zip(
            CLASS_NAMES,
            predictions
        )
    }

    return predicted_class, confidence, probabilities

def fuse_predictions(image_probabilities, text_probabilities):

    IMAGE_WEIGHT = 0.7
    TEXT_WEIGHT = 0.3

    final_probabilities = {}

    for class_name in CLASS_NAMES:

        image_probability = image_probabilities.get(
            class_name,
            0.0
        )

        text_probability = text_probabilities.get(
            class_name,
            0.0
        )

        final_probability = (
            IMAGE_WEIGHT * image_probability
            + TEXT_WEIGHT * text_probability
        )

        final_probabilities[class_name] = final_probability

    final_class = max(
        final_probabilities,
        key=final_probabilities.get
    )

    final_confidence = final_probabilities[final_class]

    return final_class, final_confidence, final_probabilities


@app.route("/", methods=["GET", "POST"])
def home():

    image_prediction = None
    image_confidence = None
    image_probabilities = None

    text_prediction = None
    text_confidence = None
    text_probabilities = None

    final_prediction = None
    final_confidence = None

    if request.method == "POST":

        # Image prediction
        if "image" in request.files:

            image_file = request.files["image"]

            if image_file.filename != "":

                image = Image.open(image_file)

                (
                    image_prediction,
                    image_confidence,
                    image_probabilities
                ) = predict_image(image)

                image_confidence = round(
                    float(image_confidence) * 100,
                    2
                )

        # NLP prediction
        symptoms = request.form.get(
            "symptoms",
            ""
        ).strip()

        if symptoms:

            (
                text_prediction,
                text_confidence,
                text_probabilities
            ) = predict_text(symptoms)

            text_confidence = round(
                float(text_confidence) * 100,
                2
            )

        # Multimodal fusion
        if image_probabilities and text_probabilities:

            (
                final_prediction,
                final_confidence,
                _
            ) = fuse_predictions(
                image_probabilities,
                text_probabilities
            )

            final_confidence = round(
                float(final_confidence) * 100,
                2
            )

    return render_template(
        "index.html",
        image_prediction=image_prediction,
        image_confidence=image_confidence,
        text_prediction=text_prediction,
        text_confidence=text_confidence,
        final_prediction=final_prediction,
        final_confidence=final_confidence
    )

if __name__ == "__main__":
    app.run(debug=False)