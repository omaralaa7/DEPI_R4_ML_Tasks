# 🧪 Ensemble Learning Lab: Bagging, Boosting & Stacking

A hands-on classification project comparing ensemble learning techniques on the
[Kaggle Playground Series S6E2 — Predicting Heart Disease](https://www.kaggle.com/competitions/playground-series-s6e2/overview) dataset.

---

## 📊 Dataset

- **Source:** Kaggle Playground Series — Season 6, Episode 2
- **Task:** Binary Classification (Heart Disease: Presence / Absence)
- **Size:** 630,000 training samples, 13 features
- **Class Balance:** 55.2% Absence / 44.8% Presence

---

## 🧠 Models Implemented

| Model | Type | ROC-AUC | Train Time |
|-------|------|---------|------------|
| BaggingClassifier | Bagging | 0.9513 | 154s |
| AdaBoostClassifier | Boosting | 0.9544 | 60s |
| GradientBoostingClassifier | Boosting | 0.9557 | 178s |
| XGBClassifier | Boosting | **0.9558** ✅ | **9s** ✅ |
| StackingClassifier | Stacking | 0.9546 | 900s |

---

## 🏆 Key Findings

- **XGBoost** achieved the best ROC-AUC (0.9558) and best Recall (0.8841) while being 18x faster than GradientBoosting
- **Stacking** was the most expensive (900s) yet did not outperform XGBoost — diminishing returns when base learners are already strong
- **False Negatives** (missed sick patients) consistently improved: 7,993 → 7,860 → 7,385 → 6,548 across Bagging → AdaBoost → GBM → XGBoost

---

## 📁 Project Structure

    ├── notebook.ipynb       # Full executable Colab notebook
    ├── submission.csv       # Final Kaggle submission
    └── README.md

---

## 🚀 How to Run

1. Open the notebook in Google Colab:
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AWLuBQIoUIdi0FSqBBEjRy5KTKtG7TeT?usp=sharing)

2. Upload your `kaggle.json` to the Colab session

3. Run all cells sequentially

---

## ⚙️ Dependencies

    pip install kaggle xgboost scikit-learn pandas numpy matplotlib seaborn

---

## 📈 Evaluation Metrics

All models evaluated on: Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix

> Primary metric: **ROC-AUC** (as per Kaggle competition scoring)

---

## 🎓 Learning Outcomes

- Practical difference between **bagging** (parallel, reduces variance) and **boosting** (sequential, reduces bias)
- Why **XGBoost dominates** in speed and accuracy over vanilla GBM
- When **stacking adds value** — and when it doesn't
- How `scale_pos_weight` handles class imbalance in XGBoost
- Medical context: why **Recall matters more than Precision** (missing a sick patient is worse than a false alarm)

---

## 👤 Author

**Omar Alaa** — ML Engineer Track (DEPI)
