import os

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "nlp",
    "symptom_data.csv"
)


CLASS_NAMES = [
    "early_blight",
    "healthy",
    "late_blight"
]


def train_nlp_model():

    data = pd.read_csv(DATA_PATH)

    X = data["text"]
    y = data["label"]

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2)
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ])

    model.fit(X, y)

    return model


# Train the NLP model once when the application starts
nlp_model = train_nlp_model()


def predict_text(text):

    prediction = nlp_model.predict([text])[0]

    probabilities = nlp_model.predict_proba([text])[0]

    probability_dict = {
        class_name: float(probability)
        for class_name, probability in zip(
            nlp_model.classes_,
            probabilities
        )
    }

    predicted_index = list(
        nlp_model.classes_
    ).index(prediction)

    confidence = probabilities[predicted_index]

    return prediction, confidence, probability_dict