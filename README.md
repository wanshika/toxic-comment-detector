# 🛡️ Toxic Comment Detection App

This project is a multi-label **Toxic Comment Classifier** that detects whether a user comment contains toxic, severe toxic, obscene, threat, insult, or identity hate language.  
It uses both traditional ML (TF-IDF + Logistic Regression) and deep learning (Toxic-BERT from Hugging Face).

---

## 🧠 Project Highlights

| Feature                     | Details |
|----------------------------|---------|
| 🔤 Input                   | Raw text comments |
| 📦 Models                  | Logistic Regression, BERT (Hugging Face `unitary/toxic-bert`) |
| 🧪 Tasks                   | Multi-label classification (6 classes) |
| 📊 Evaluation              | Precision, Recall, F1-Score |
| 📈 Visuals                 | Classification reports + ROC & AUC graphs |
| 🎛️ Model Toggle           | Compare predictions between Logistic Regression and BERT |
| 🌐 Interface               | Built using Streamlit |
| ✅ Real-time inference     | Paste text → get predictions + confidence scores |

---

## 🚀 Demo

⚡ Hosted on Streamlit:  
🔗 [YOUR DEPLOYMENT URL WILL GO HERE]

---

## 📂 Folder Structure

toxic-comment-detector/ ├── app/ # Streamlit app │ └── app.py ├── models/ # Saved vectorizer and models │ ├── tfidf_vectorizer.pkl │ └── logistic_regression_model.pkl ├── outputs/ # Model predictions for comparison │ ├── logreg_probs.npy │ └── toxic_bert_preds.npy ├── notebooks/ # All training and comparison notebooks │ ├── 1_baseline_model.ipynb │ ├── 2_toxic_bert_inference.ipynb │ └── 3_model_comparison_and_visualization.ipynb ├── data/ │ └── train.csv # Toxic comment dataset ├── requirements.txt └── README.md

yaml
Copy
Edit

---

## 🧪 Evaluation Summary

| Category       | Logistic Regression | Toxic-BERT |
|----------------|---------------------|------------|
| Toxic          | 0.85                | 0.94       |
| Severe Toxic   | 0.31                | 0.50       |
| Obscene        | 0.85                | 0.88       |
| Threat         | 0.15                | 0.58       |
| Insult         | 0.81                | 0.83       |
| Identity Hate  | 0.21                | 0.69       |

---

## ⚙️ Setup & Run

1. Clone this repo  
```bash
git clone https://github.com/your-username/toxic-comment-detector.git
cd toxic-comment-detector
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the Streamlit app

bash
Copy
Edit
streamlit run app/app.py
📚 Dataset
Used the Jigsaw Toxic Comment Classification dataset from Kaggle.

🙌 Author
Wanshika Patro and Nikhil Gokhale
🎓 MS in Data Science, University at Buffalo