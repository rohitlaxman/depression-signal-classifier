import joblib
import streamlit as st

MODEL_PATH = "models/text_classifier.joblib"

st.set_page_config(
    page_title="Mental Health Text Classifier",
    page_icon="🧠",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("Sentence-Level Mental Health Text Classifier")

st.write(
    "Enter one sentence to classify its text patterns "
    "into one of four dataset categories."
)

st.warning(
    "Research prototype only. This tool does not diagnose "
    "mental health conditions or assess immediate risk."
)

sentence = st.text_area(
    "Enter your sentence",
    placeholder="Type a sentence here...",
    height=120,
)

if st.button("Classify sentence", type="primary"):
    if not sentence.strip():
        st.error("Please enter a sentence first.")
    else:
        prediction = model.predict([sentence.strip()])[0]
        scores = model.predict_proba([sentence.strip()])[0]
        classes = model.named_steps["classifier"].classes_

        st.subheader("Predicted text category")
        st.success(prediction)

        st.subheader("Model scores")

        for label, score in sorted(
            zip(classes, scores),
            key=lambda item: item[1],
            reverse=True,
        ):
            st.write(f"**{label}:** {score:.1%}")
            st.progress(float(score))

        st.caption(
            "Scores reflect the model's outputs, not the probability "
            "that a person has a condition. Predictions may be wrong."
        )
