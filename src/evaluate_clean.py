import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


TRAIN_PATH = "data/mental_heath_unbanlanced.csv"
TEST_PATH = "data/mental_health_combined_test.csv"


def normalize_text(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
    )


def main():
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
        subset="normalized", keep="first"
    ).copy()

    # Exclude every training sentence also present in the test set.
    test_texts = set(test["normalized"])
    before = len(train)
    train = train[~train["normalized"].isin(test_texts)].copy()

    print("Training rows after duplicate and overlap removal:", len(train))
    print("Training rows removed due to test overlap:", before - len(train))
    print("Test rows:", len(test))

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

    print("\n=== CLEANED EVALUATION ===")
    print(f"Accuracy: {accuracy_score(test['status'], predictions):.4f}")
    print(
        classification_report(
            test["status"],
            predictions,
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
