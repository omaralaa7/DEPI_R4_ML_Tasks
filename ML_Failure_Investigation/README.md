# 🔍 ML Failure Investigation Lab

> **"Why is the model failing, and how can we prove it?"**

A hands-on machine learning workshop focused on **reasoning, diagnosis, and evidence-based decision making** using the Credit Card Fraud Detection dataset. This lab teaches you how to think like an ML engineer — not just how to train a model, but how to understand *why* it fails and *how* to fix it responsibly.

---

## 📌 Overview

Most ML courses evaluate you on accuracy. This lab evaluates you on:

- 🧠 **Reasoning** — why does the model behave this way?
- 🔎 **Diagnosis** — what is the actual root cause of failure?
- ⚖️ **Tradeoff understanding** — what do you gain and lose with each fix?
- 🧪 **Evidence-based improvements** — every change must be justified

---

## 📂 Repository Structure

```
ml-failure-investigation-lab/
│
├── ml_failure_investigation_lab_COMPLETED.ipynb   # Main notebook (fully implemented)
├── README.md                                       # This file
└── assets/
    ├── failure_analysis.png     # Caught vs. missed fraud visualization
    └── pr_curve.png             # Precision-Recall curve
```

---

## 📊 Dataset

**Credit Card Fraud Detection** — ULB Machine Learning Group

| Property | Value |
|---|---|
| Rows | 284,807 transactions |
| Features | 28 PCA components (V1–V28) + Time + Amount |
| Target | `Class` — 0 = Legitimate, 1 = Fraud |
| Imbalance ratio | ~577:1 (0.17% fraud) |
| Missing values | None |

The dataset is automatically downloaded via `kagglehub`. You can also download it manually from:

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🧱 Lab Structure

The notebook walks through 8 tasks + 3 bonus challenges:

| Task | Topic |
|---|---|
| Task 1 | Dataset investigation — shape, nulls, class balance |
| Task 2 | Stratified train/test split — why `stratify` matters |
| Task 3 | Scaling decisions — which models need it, which don't |
| Task 4 | Baseline model — Logistic Regression without imbalance handling |
| Task 5 | Error analysis — confusion matrix, FN vs FP costs |
| Task 6 | Model improvement — `class_weight="balanced"` with full justification |
| Task 7 | Tradeoff investigation — recall ↑ vs precision ↓ |
| Task 8 | Failure case analysis — patterns in missed fraud |
| Bonus 1 | Threshold tuning — 0.1 to 0.7 sweep + PR curve |
| Bonus 2 | Random Forest comparison — scale-invariance, non-linearity |
| Bonus 3 | Leakage trap — target leakage and why it's dangerous |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/ml-failure-investigation-lab.git
cd ml-failure-investigation-lab
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib kagglehub
```

### 3. Run the notebook

```bash
jupyter notebook ml_failure_investigation_lab_COMPLETED.ipynb
```

> The dataset is downloaded automatically in the first code cell via `kagglehub`. No manual setup required.

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | Model training, evaluation, preprocessing |
| `matplotlib` | Visualizations |
| `kagglehub` | Automatic dataset download |

Python **3.8+** recommended.

---

## 🔑 Key Concepts Covered

### The Accuracy Trap

With a 577:1 class imbalance, a model that always predicts "not fraud" achieves **~99.83% accuracy** while catching **zero fraud cases**. This lab demonstrates why accuracy is a misleading metric for imbalanced datasets.

### Why Recall Matters More Than Accuracy

In fraud detection, the cost of errors is asymmetric:

- **False Negative** (missed fraud) → financial loss, legal liability, customer harm
- **False Positive** (false alarm) → minor inconvenience, one phone call

This asymmetry means **Recall** and **F1-score** are the metrics that actually reflect model quality here.

### Class Weighting

`class_weight="balanced"` instructs the optimizer to upweight minority class samples proportionally to their rarity. Each fraud sample receives ~289× the loss contribution of a legitimate sample, forcing the decision boundary to shift toward fraud detection.

### The Precision-Recall Tradeoff

Improving recall (catching more fraud) always comes at the cost of precision (more false alarms). The optimal threshold depends on business context — specifically the **relative cost of FN vs FP**.

### Data Leakage

Creating a feature derived from the target variable inflates evaluation metrics to near-perfect during training, but the feature doesn't exist at prediction time in production → the model fails completely in the real world.

---

## 📈 Results Summary

| Model | Precision | Recall | F1 |
|---|---|---|---|
| Logistic Regression (baseline) | High | Low | Low |
| Logistic Regression (balanced) | Moderate | High | Moderate |
| Random Forest (default) | High | Moderate–High | High |

> Exact values will vary based on your environment. The key learning is the **direction of change**, not specific numbers.

---

## 💡 Key Takeaways

1. **Never trust accuracy on imbalanced datasets** — always check per-class metrics
2. **Every modeling decision needs justification** — what problem does it solve and what tradeoff does it introduce?
3. **Threshold tuning is free improvement** — calibrate the operating point to your business cost model
4. **Data leakage is catastrophic** — always ask "would this feature exist at prediction time?"
5. **Tree-based models are scale-invariant** — feature scaling is not required for Random Forest or XGBoost
6. **Evaluation must simulate production** — random splits on time-ordered data can hide real-world performance degradation

---

## 🎓 Grading Rubric

This lab grades you on **reasoning quality**, NOT just final metrics:

- ✅ Reasoning quality
- ✅ Justification quality
- ✅ Investigation depth
- ✅ Evidence usage
- ✅ Interpretation of results
- ✅ Tradeoff understanding

---

## 📝 License

This project is for educational purposes. The Credit Card Fraud Detection dataset is licensed under the [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/1.0/) by the ULB Machine Learning Group.

---

## 🙌 Acknowledgements

- Dataset: [ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Workshop design: DEPI Machine Learning Engineering Track
