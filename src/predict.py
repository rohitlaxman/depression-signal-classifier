import joblib

MODEL_PATH = "models/text_classifier.joblib"


def main():
    model = joblib.load(MODEL_PATH)

    print("\nSentence-Level Text Classifier")
    print("Categories: Anxiety, Depression, Normal, Suicidal")
    print("Type 'exit' to quit.")
    print("Research prototype only — not a diagnostic tool.")

    while True:
        sentence = input("\nEnter a sentence: ").strip()

        if sentence.lower() == "exit":
            break

        if not sentence:
            print("Please enter a sentence.")
            continue

        prediction = model.predict([sentence])[0]
        probabilities = model.predict_proba([sentence])[0]
        classes = model.named_steps["classifier"].classes_

        print(f"\nPredicted category: {prediction}")
        print("Model scores:")

        for label, score in sorted(
            zip(classes, probabilities),
            key=lambda item: item[1],
            reverse=True,
        ):
            print(f"  {label}: {score:.1%}")

        print("\nThese scores are not clinical probabilities or a diagnosis.")


if __name__ == "__main__":
    main()
