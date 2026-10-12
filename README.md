# Sentence-Level Mental Health Text Classifier

## Overview

This project uses Natural Language Processing (NLP) and machine learning to classify a single text sentence into one of four dataset categories:

- Anxiety
- Depression
- Normal
- Suicidal

It is an educational research prototype, not a clinical diagnostic or risk-assessment tool.

## Technology Stack

- Python
- pandas
- scikit-learn
- TF-IDF text vectorization
- Logistic Regression
- Streamlit
- joblib

## Model Workflow

1. Load labelled text data from CSV files.
2. Remove missing and duplicate training records.
3. Convert text into numerical features using TF-IDF.
4. Train a Logistic Regression classifier.
5. Evaluate predictions using a separate test dataset.
6. Save the trained pipeline.
7. Use a Streamlit interface to classify new sentences.

## Dataset

The project uses the Mental Health Text Classification Dataset (4-Class) published on Hugging Face:

https://huggingface.co/datasets/ourafla/Mental-Health_Text-Classification_Dataset

The dataset combines previously published text corpora. Its labels are derived from source datasets and are not equivalent to clinical diagnoses.

## Evaluation

The initial evaluation reported 79.03% accuracy on 992 test examples, with 496 normalized text entries overlapping between training and test sets.

After removing duplicate training entries and excluding training texts overlapping with the test set, the evaluation reported:

- Accuracy: 70.97%
- Macro F1-score: 0.71

These are preliminary results. The dataset's construction and labels may introduce bias, and further independent validation is needed.

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the Web Application

From the project root, run:

```bash
streamlit run app.py
```

Open the local URL printed in the terminal, usually:

http://localhost:8501

## Train and Evaluate

Train the model and generate the initial evaluation:

```bash
python src/train_model.py
```

Run the overlap-controlled evaluation:

```bash
python src/evaluate_clean.py
```

Run the terminal prediction interface:

```bash
python src/predict.py
```

## Limitations and Ethics

- The model classifies text patterns, not people or medical conditions.
- Prediction scores are model outputs, not clinical probabilities.
- Social-media language may not generalize to other contexts.
- The dataset may contain biases, ambiguous labels, and distressing content.
- This prototype must not be used for diagnosis, crisis intervention, or consequential decisions.

## License and Attribution

Review the dataset's CC BY 4.0 terms and cite the dataset and its original sources when distributing or presenting this project. The dataset's licence does not automatically determine the licence of this project's code.
