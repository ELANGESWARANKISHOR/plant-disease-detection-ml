import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix


DATA_PATH = "data/nlp/symptom_data.csv"


# Load dataset
data = pd.read_csv(DATA_PATH)

X = data["text"]
y = data["label"]


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# Create NLP pipeline
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


# Train
model.fit(X_train, y_train)


# Evaluate
predictions = model.predict(X_test)

print("\nNLP Model Evaluation")
print("--------------------")

print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))


# Test example
example = [
    "The potato leaves have dark brown spots and the damage is spreading"
]

prediction = model.predict(example)
probabilities = model.predict_proba(example)

print("\nExample Prediction")
print("------------------")
print("Text:", example[0])
print("Prediction:", prediction[0])

print("\nClass probabilities:")

for class_name, probability in zip(
    model.classes_,
    probabilities[0]
):
    print(f"{class_name}: {probability:.2%}")