# 🛡️ Toxic Comment Classifier
# 🛠️ Trigger Streamlit redeploy

import os
import joblib
import numpy as np
import streamlit as st

# ✅ Must be FIRST Streamlit command
st.set_page_config(page_title="Toxic Comment Classifier", layout="centered", page_icon="🛡️")

# -----------------------------
# Constants / Labels
# -----------------------------
# Jigsaw multilabel heads used by most toxic-bert variants
BERT_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_attack"]
BINARY_LABEL = ["toxic"]  # for your TF-IDF + Logistic Regression model

VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
LOGREG_MODEL_PATH = "models/logreg_model.pkl"

# -----------------------------
# Load classical (binary) model
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_binary_pipeline():
    vectorizer = joblib.load(VECTORIZER_PATH)
    if not hasattr(vectorizer, "idf_"):
        raise ValueError("Vectorizer is not fitted. Expected a fitted TF-IDF vectorizer with idf_.")
    logreg_model = joblib.load(LOGREG_MODEL_PATH)
    return vectorizer, logreg_model

try:
    vectorizer, logreg_model = load_binary_pipeline()
except Exception as e:
    st.error(f"❌ Failed to load classic pipeline: {e}")
    st.stop()

# -----------------------------
# Lazy HF load (optional)
# -----------------------------
def _lazy_import_hf():
    import torch  # noqa
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # noqa
    return torch, AutoTokenizer, AutoModelForSequenceClassification

@st.cache_resource(show_spinner=False)
def load_bert():
    """Load BERT only when needed; return (tokenizer, model) or (None, None) on failure."""
    try:
        torch, AutoTokenizer, AutoModelForSequenceClassification = _lazy_import_hf()
        model_name = os.environ.get("TOXIC_MODEL_ID", "unitary/toxic-bert")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.warning(f"⚠️ Could not load Hugging Face model (falling back to Logistic Regression): {e}")
        return None, None

# -----------------------------
# UI
# -----------------------------
st.title("🧹 Toxic Comment Detector")
st.caption("Demo app to score comments for likely toxicity. For evaluation only—use with human review.")

example = st.selectbox(
    "Try an example",
    [
        "I completely disagree with your point but appreciate the discussion.",
        "You're an idiot — nobody wants to hear this.",
        "Let's keep this thread respectful and on-topic.",
        "Go back to where you came from.",
    ],
)
text = st.text_area("Paste a comment to analyze", value=example, height=140)

col1, col2 = st.columns([1, 1])
with col1:
    model_choice = st.radio("Model", ("Logistic Regression (binary)", "Toxic-BERT (multilabel)"), horizontal=False)
with col2:
    threshold = st.slider("Label threshold", 0.1, 0.9, 0.5, 0.05)

analyze = st.button("Analyze", type="primary")

# -----------------------------
# Inference
# -----------------------------
if analyze:
    if not text.strip():
        st.warning("Please enter text to analyze.")
        st.stop()

    if model_choice.startswith("Logistic"):
        # Binary overall toxicity
        X = vectorizer.transform([text])
        prob = float(logreg_model.predict_proba(X)[0][1])  # P(toxic)
        label = "Toxic" if prob >= threshold else "Not toxic"

        st.subheader("Result")
        st.metric(label="Overall toxicity score", value=f"{prob:.2f}")
        st.write(f"**Label:** {label}")
        st.caption(f"Model: Logistic Regression (TF-IDF) • Threshold: {threshold:.2f}")

        # Simple bar with one metric to avoid misleading per-label bars
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(BINARY_LABEL, [prob])
        ax.set_ylim(0, 1)
        ax.set_ylabel("Confidence")
        st.pyplot(fig)

    else:
        # BERT multilabel
        tokenizer, model = load_bert()
        if tokenizer is None or model is None:
            st.info("Using fallback Logistic Regression since BERT didn't load.")
            X = vectorizer.transform([text])
            prob = float(logreg_model.predict_proba(X)[0][1])
            label = "Toxic" if prob >= threshold else "Not toxic"

            st.subheader("Result")
            st.metric(label="Overall toxicity score", value=f"{prob:.2f}")
            st.write(f"**Label:** {label}")
            st.caption(f"Model: Logistic Regression (fallback) • Threshold: {threshold:.2f}")

        else:
            import torch
            encoded = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**encoded).logits
                probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()

            st.subheader("Result (Per label)")
            any_flag = False
            for label_name, p in zip(BERT_LABELS, probs):
                pct = f"{p*100:.1f}%"
                if p >= threshold:
                    any_flag = True
                    st.success(f"✅ **{label_name}** — {pct}")
                else:
                    st.info(f"⚪ **{label_name}** — {pct}")

            if not any_flag:
                st.info("No labels crossed the threshold; likely **Not toxic** overall.")

            st.caption(f"Model: toxic-bert • Threshold: {threshold:.2f}")

            # Bar chart for all labels
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.bar(BERT_LABELS, probs)
            ax.set_ylim(0, 1)
            ax.set_ylabel("Confidence")
            fig.autofmt_xdate(rotation=45)
            st.pyplot(fig)

    st.divider()
    with st.expander("Notes & limitations"):
        st.write(
            "- This is a demo, not a production moderation system.\n"
            "- Models can be biased or make mistakes; always include human review.\n"
            "- Threshold tuning and feedback collection improve performance over time."
        )
