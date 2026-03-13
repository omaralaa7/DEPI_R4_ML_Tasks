# 📧 Spam Detection Classifier

A complete NLP pipeline that classifies SMS/email messages as **Spam** or **Ham (Not Spam)**, built with scikit-learn and deployed with Streamlit.

## 🌐 Live Demo

👉 [omarator-spam-detection.streamlit.app](https://omarator-spam-detection.streamlit.app)

---

## 📌 Project Overview

This project covers the full ML lifecycle:
- **Data Exploration** with visualizations
- **Text Preprocessing** (cleaning, stopword removal, stemming)
- **Feature Extraction** using Bag of Words & TF-IDF
- **Model Training** across 6 classifiers
- **Model Evaluation** with accuracy, precision, recall, F1, ROC-AUC
- **Web UI** deployed on Streamlit Cloud

---

## 📁 Project Structure

```
Spam_Detection/
│
├── app.py                    # Streamlit web app
├── nlp_workshop_Omar.ipynb   # Main Jupyter notebook
├── vectorizer.pkl            # Saved TF-IDF vectorizer
├── preprocess.pkl            # Saved preprocessing pipeline
├── model.pkl                 # Best trained model
├── spam.csv                  # Dataset
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 🧠 Models Trained

| Model               | Notes            |
|---------------------|------------------|
| Logistic Regression | ✅ High accuracy |
| Naive Bayes         | ✅ High accuracy |
| Random Forest       | ✅ High accuracy |
| SVM                 | ✅ High accuracy |
| XGBoost             | ✅ High accuracy |
| Decision Tree       | ✅ Moderate      |

> Best model saved as `model.pkl` and used in the web app.

---

## 🔧 Tech Stack

- **Language:** Python 3.x
- **ML:** scikit-learn, XGBoost
- **NLP:** NLTK, TF-IDF, Bag of Words
- **UI:** Streamlit
- **Deployment:** Streamlit Cloud
- **Serialization:** Joblib

---

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/omarator2004/DEPI_R4_ML_Tasks.git
cd DEPI_R4_ML_Tasks/Spam_Detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📊 EDA Highlights

- Class distribution (spam vs. ham)
- Message length distribution
- Most common words in spam vs. ham
- Word cloud visualizations

---

## 👨‍💻 Author

**Omar Alaa**
- GitHub: [@omarator2004](https://github.com/omarator2004)
- Project: DEPI ML Engineer Track — NLP Workshop

---

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
