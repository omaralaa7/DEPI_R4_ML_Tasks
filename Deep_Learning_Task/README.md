# 🧠 Deep Learning Training Techniques — Practical Study

A hands-on exploration of key deep learning training decisions using a tabular
binary classification dataset. This notebook covers optimizers, batch sizes,
regularization methods, and early stopping — with full visualizations and
a final reflection on best practices.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nYmNzgbwK2Dz5yz3dKEYzvcUWs8ZirYV?usp=sharing)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Parts Summary](#parts-summary)
- [Key Results](#key-results)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Conclusions](#conclusions)

---

## 📌 Overview

This project systematically studies the impact of different training choices
on a deep learning model's performance, focusing on:

- Which **optimizer** converges fastest and generalizes best
- How **batch size** affects training stability and accuracy
- Whether **L2 or Dropout** regularization better prevents overfitting
- How **early stopping** saves training time while improving generalization

---

## 🗂️ Project Structure

```
├── notebook.ipynb          # Main Colab notebook
├── README.md               # This file
└── plots/
    ├── optimizer_curves.png
    ├── batch_size_curves.png
    ├── regularization_curves.png
    └── early_stopping_curves.png
```

---

## 📚 Parts Summary

### Part 0 — Data Preparation
- Loaded tabular binary classification dataset
- Applied train / validation / test split (70% / 15% / 15%)
- Standardized features using `StandardScaler`

### Part 1 — Optimizers
Compared **SGD**, **RMSprop**, and **Adam** over 50 epochs.

| Optimizer | Val Accuracy | Test Accuracy |
|-----------|-------------|---------------|
| SGD       | ~83%        | ~83%          |
| RMSprop   | ~84%        | ~84%          |
| **Adam**  | **~84.5%**  | **~84.5%**    |

✅ **Adam** selected as the best optimizer.

### Part 2 — Batch Size
Tested batch sizes **16, 32, 64, 128, 256** using Adam optimizer.

| Batch Size | Behavior |
|------------|----------|
| 16         | Noisy but good generalization |
| **32**     | **Best balance of speed & accuracy** |
| 256        | Fastest but lower val accuracy |

✅ **Batch size 32** selected as optimal.

### Part 3 — Regularization
Compared **No Regularization**, **L2**, and **Dropout** over 50 epochs.

| Method            | Train  | Val    | Test      | Gap      |
|-------------------|--------|--------|-----------|----------|
| No Regularization | 92.0%  | 81.9%  | 83.3%     | 🔴 10.1% |
| L2                | 85.3%  | 84.3%  | 84.6%     | ✅ 0.96% |
| **Dropout**       | 85.9%  | 84.4%  | **84.9%** | ✅ 1.54% |

✅ **Dropout** gave the best test accuracy with minimal overfitting.

### Part 4 — Early Stopping
Trained with and without early stopping (`patience=10`, `monitor=val_loss`).

| Method             | Epochs | Val Acc    | Test Acc   |
|--------------------|--------|------------|------------|
| No Early Stopping  | 50     | 83.43%     | 84.14%     |
| **Early Stopping** | **17** | **84.02%** | **84.52%** |

✅ Early stopping used **66% fewer epochs** and still achieved **better accuracy**.

### Part 5 — Reflection
Final recommended setup for any new tabular dataset:

| Decision       | Choice                               |
|----------------|--------------------------------------|
| Optimizer      | Adam                                 |
| Batch Size     | 32                                   |
| Regularization | Dropout (0.3–0.5)                    |
| Early Stopping | ✅ patience=10, restore_best_weights  |
| Data Split     | 70% Train / 15% Val / 15% Test       |

---

## 📊 Key Results

> **Best configuration achieved ~84.5% test accuracy** using:
> Adam optimizer + batch size 32 + Dropout regularization + Early Stopping

---

## ⚙️ Requirements

```bash
tensorflow >= 2.x
numpy
pandas
matplotlib
scikit-learn
```

Install with:
```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

---

## 🚀 How to Run

1. Open the notebook directly in Google Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1nYmNzgbwK2Dz5yz3dKEYzvcUWs8ZirYV?usp=sharing)

2. Or clone the repo and run locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
   jupyter notebook notebook.ipynb
   ```

---

## 💡 Conclusions

- **Adam** is the safest default optimizer for tabular data
- **Batch size 32** balances noise and convergence effectively
- **Dropout** outperforms L2 on test accuracy while both prevent overfitting
- **Early stopping** is a free regularizer — always use it
- Proper **train/val/test splitting** is essential for honest evaluation

---

## 👤 Author

**Omar Alaa**  
DEPI Microsoft ML Engineer Track
