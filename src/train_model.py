from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline


TRAIN_PATH = "data/mental_heath_unbanlanced.csv"
TEST_PATH = "data/mental_health_combined_test.csv"
MODEL_PATH = "models/text_classifier.joblib"


def load_data(path):
    data = pd.read_csv(path)

    required = {"text", "status"}
    if not required.issubset(data.columns):
        raise ValueError(f"{path} must contain {required}")

    data = data.dropna(subset=["text", "status"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data["status"] = data["status"].astype(str).str.strip()
    data = data[data["text"] != ""]
    data = data[data["status"] != ""]

    return data


def main():
    train = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)

    # Remove repeated training texts after normalizing case and whitespace.
    train["normalized_text"] = (
        train["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    )
    before = len(train)
    train = train.drop_duplicates(
        subset=["normalized_text"], keep="first"
    ).copy()

    print(f"Training rows before deduplication: {before}")
    print(f"Training rows after deduplication: {len(train)}")
    print(f"Removed duplicate rows: {before - len(train)}")
    print(f"Test rows: {len(test)}")

    # Report overlap rather than silently hiding it.
    test_normalized = (
        test["text"].str.lower().str.replace(r"\s+", " ", regex=True)
    )
    overlap = set(train["normalized_text"]) & set(test_normalized)
    print(f"Train/test text overlap: {len(overlap)}")
    print("Warning: overlapping texts may inflate test scores.")

    labels = sorted(train["status"].unique())
    if len(labels) < 2:
        raise ValueError("Training data must contain at least two classes.")

    if not set(test["status"].unique()).issubset(set(labels)):
        raise ValueError("Test data contains a label absent from training.")

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                strip_accents="unicode",
                ngram_range=(1, 2),
                max_features=100000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    print("\nTraining the model...")
    model.fit(train["text"], train["status"])

    predictions = model.predict(test["text"])

    print("\n=== TEST RESULTS ===")
    print(f"Accuracy: {accuracy_score(test['status'], predictions):.4f}")
    print("\nClassification report:")
    print(classification_report(
        test["status"],
        predictions,
        labels=labels,
        zero_division=0,
    ))

    print("Confusion matrix (rows=true, columns=predicted):")
    print(labels)
    print(confusion_matrix(
        test["status"], predictions, labels=labels
    ))

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")
    print("These are research metrics, not clinical validation.")


if __name__ == "__main__":
    main()
