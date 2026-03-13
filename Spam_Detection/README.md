# 📩 SMS Spam Classifier

A complete traditional NLP machine learning pipeline for SMS spam detection,
built with classical NLP techniques and deployed as a web application.

---

## 🎯 Project Overview

This project classifies SMS messages as **spam** or **ham** (legitimate)
using a traditional NLP pipeline without deep learning.

---

## 🗂️ Project Structure

```
spam-classifier/
├── app.py               ← Streamlit web application
├── requirements.txt     ← Python dependencies
├── vectorizer.pkl       ← Trained TF-IDF vectorizer
├── model.pkl            ← Trained Linear SVM model
└── README.md
```

---

## 🔬 Pipeline Steps

| Step | Description |
|------|-------------|
| 1️⃣ Data Loading | SMS Spam Collection Dataset (5169 messages after cleaning) |
| 2️⃣ EDA | Class distribution, message length, word clouds |
| 3️⃣ Preprocessing | Lowercase → Remove punctuation → Tokenize → Stopwords → Lemmatize |
| 4️⃣ Feature Extraction | BOW (CountVectorizer) and TF-IDF (TfidfVectorizer) |
| 5️⃣ Model Training | Logistic Regression, Naive Bayes, Linear SVM |
| 6️⃣ Evaluation | Accuracy, Precision, Recall, F1 Score, Confusion Matrix |
| 7️⃣ Save Artifacts | vectorizer.pkl, model.pkl |
| 8️⃣ Deployment | Streamlit web app |

---

## 🏆 Model Results

| Model | Vectorizer | F1 Score |
|-------|------------|----------|
| **Linear SVM** | **TF-IDF** | **0.9786** 🥇 |
| Logistic Regression | TF-IDF | 0.9749 🥈 |
| Naive Bayes | BOW | 0.9729 🥉 |
| Naive Bayes | TF-IDF | 0.9618 |
| Linear SVM | BOW | 0.9521 |
| Logistic Regression | BOW | 0.9317 |

### Best Model Performance (Linear SVM + TF-IDF)

```
              precision    recall  f1-score

         ham       0.99      0.99      0.99
        spam       0.93      0.90      0.91

    accuracy                           0.98
```

---

## 🧹 Preprocessing Pipeline

```
Raw Text
   ↓ Lowercase
   ↓ Remove punctuation & numbers (regex)
   ↓ Tokenize (NLTK word_tokenize)
   ↓ Remove stopwords (NLTK — 198 words)
   ↓ Lemmatize (WordNetLemmatizer)
Clean Text
```

---

## 📊 Dataset

- **Source:** [SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Total messages:** 5,169 (after removing 403 duplicates)
- **Classes:** Ham (4,516) · Spam (653)
- **Split:** 80% train / 20% test (stratified)

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOURUSERNAME/spam-classifier.git
cd spam-classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## 🌐 Live Demo

👉 [spam-classifier.streamlit.app](https://spam-classifier.streamlit.app)

---

## 🛠️ Tech Stack

- **Language:** Python 3.x
- **NLP:** NLTK
- **ML:** Scikit-learn
- **UI:** Streamlit
- **Serialization:** Joblib

---

## 📝 Key Findings

- **TF-IDF outperformed BOW** in 2/3 models by amplifying rare spam signals
- **Naive Bayes works better with BOW** — mathematically designed for raw word counts
- **Imbalanced dataset** handled with `class_weight='balanced'` on LR and SVM
- **F1 Score used** as primary metric (more reliable than accuracy for imbalanced data)
