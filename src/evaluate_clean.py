
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
REPORT_DIR = Path("reports")

LABELS = ["Anxiety", "Depression", "Normal", "Suicidal"]


def normalize_text(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    train = train.dropna(subset=["text", "status"]).copy()
    test = test.dropna(subset=["text", "status"]).copy()

    train["text"] = train["text"].astype(str).str.strip()
    test["text"] = test["text"].astype(str).str.strip()

    train = train[train["text"] != ""].copy()
    test = test[test["text"] != ""].copy()

    train["normalized"] = normalize_text(train["text"])
    test["normalized"] = normalize_text(test["text"])

    # Remove duplicate training sentences.
    train = train.drop_duplicates(
        subset="normalized",
        keep="first",
    ).copy()

    # Remove training sentences that overlap with the test set.
    test_texts = set(test["normalized"])
    before = len(train)

    train = train[
        ~train["normalized"].isin(test_texts)
    ].copy()

    removed = before - len(train)

    print("Training rows after cleaning:", len(train))
    print("Training rows removed due to test overlap:", removed)
    print("Test rows:", len(test))

    # Build the TF-IDF + Logistic Regression pipeline.
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

    print("\nTraining clean-evaluation model...")
    model.fit(train["text"], train["status"])

    predictions = model.predict(test["text"])
    accuracy = accuracy_score(test["status"], predictions)

    # Generate reports in a consistent label order.
    report_dict = classification_report(
        test["status"],
        predictions,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )

    report_text = classification_report(
        test["status"],
        predictions,
        labels=LABELS,
        zero_division=0,
    )

    matrix = confusion_matrix(
        test["status"],
        predictions,
        labels=LABELS,
    )

    # Save metrics as JSON.
    metrics = {
        "accuracy": float(accuracy),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "weighted_f1": float(report_dict["weighted avg"]["f1-score"]),
        "test_rows": int(len(test)),
        "training_rows_after_cleaning": int(len(train)),
        "training_rows_removed_due_to_test_overlap": int(removed),
        "labels": LABELS,
        "classification_report": report_dict,
        "evaluation_note": (
            "Evaluated on the supplied test file after removing "
            "normalized text overlaps from training. This does not "
            "guarantee independence across source datasets. Results "
            "are not clinical validation and must not be used for "
            "diagnosis or individual risk assessment."
        ),
    }

    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    # Save a readable text report.
    (REPORT_DIR / "classification_report.txt").write_text(
        f"Accuracy: {accuracy:.4f}\n"
        f"Macro F1: {report_dict['macro avg']['f1-score']:.4f}\n\n"
        f"{report_text}",
        encoding="utf-8",
    )

    # Plot and save the confusion matrix.
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax)

    ax.set(
        xticks=range(len(LABELS)),
        yticks=range(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix - Clean Evaluation",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    threshold = matrix.max() / 2 if matrix.size else 0

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row, col] > threshold
                    else "black"
                ),
            )

    fig.tight_layout()
    fig.savefig(
        REPORT_DIR / "confusion_matrix.png",
        dpi=160,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n=== CLEANED EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {report_dict['macro avg']['f1-score']:.4f}")
    print("\nClassification report:")
    print(report_text)

    print("\nSaved evaluation files:")
    print(REPORT_DIR / "classification_report.txt")
    print(REPORT_DIR / "metrics.json")
    print(REPORT_DIR / "confusion_matrix.png")


if __name__ == "__main__":
    main()
