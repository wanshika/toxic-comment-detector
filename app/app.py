import streamlit as st

# ✅ Must be FIRST Streamlit command
st.set_page_config(
    page_title="Toxic Comment Classifier",
    layout="centered",
    page_icon="🛡️"
)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import numpy as np
import os
import matplotlib.pyplot as plt

# Labels
LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

# Get absolute paths to models
APP_DIR = os.path.dirname(__file__)
VECTORIZER_PATH = os.path.join(APP_DIR, "../models/tfidf_vectorizer.pkl")
LOGREG_MODEL_PATH = os.path.join(APP_DIR, "../models/logistic_models.pkl")


# Load TF-IDF + Logistic Regression model
vectorizer = joblib.load(VECTORIZER_PATH)
logreg_models = joblib.load(LOGREG_MODEL_PATH)

# Load Toxic BERT
@st.cache_resource
def load_bert():
    model_name = "unitary/toxic-bert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model

tokenizer_bert, model_bert = load_bert()

# ---------- UI ----------
st.markdown("## 🛡️ Toxic Comment Detection App")
st.write("Check whether a comment contains toxic or harmful language.")

comment = st.text_area("Paste a comment to analyze", height=150)

model_choice = st.radio(
    "Choose model for analysis:",
    ("Logistic Regression", "Toxic-BERT"),
    horizontal=True
)

if st.button("Analyze") and comment.strip():

    st.markdown("### 🔍 Prediction Results")

    if model_choice == "Logistic Regression":
        X_input = vectorizer.transform([comment])
        probs = [model.predict_proba(X_input)[0][1] for model in logreg_models.values()]
    else:
        encoded = tokenizer_bert(
            [comment],
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        with torch.no_grad():
            output = model_bert(**encoded)
            probs = torch.sigmoid(output.logits).cpu().numpy().flatten()

    threshold = 0.5  # You can tweak this for stricter detection

    for label, prob in zip(LABELS, probs):
        confidence_pct = f"{prob * 100:.1f}%"
        if prob >= threshold:
            st.success(f"✅ **{label}** – {confidence_pct}")
        else:
            st.info(f"⚪ **{label}** – {confidence_pct}")


    # ---------- Visualization ----------
    st.markdown("### 📊 Confidence Scores")
    fig, ax = plt.subplots()
    ax.bar(LABELS, probs, color="skyblue")
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    st.pyplot(fig)
