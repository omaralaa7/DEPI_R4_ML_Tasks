# 🏆 Kaggle Playground Series S5E8 — RandomForest Baseline (Colab)

[![🚀 Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Wuj1XhszmuFNk6dh5GQqWAZHFjcqu5sK?usp=sharing)

In this notebook, I built an end-to-end baseline solution for a tabular **binary classification** Kaggle task using a preprocessing + model pipeline.

## ✅ What I did
- 📥 Loaded `train.csv`, `test.csv`, and `sample_submission.csv`.
- 🎯 Selected the target column and prepared features (dropped `id` from inputs).
- 🧹 Built preprocessing for numeric + categorical columns (imputation + encoding).
- 🌲 Trained a `RandomForestClassifier` baseline with `n_estimators=300`.
- 📊 Generated probability predictions (`predict_proba`) for the test set.
- 📄 Created `submission.csv` matching the exact format of `sample_submission.csv`.
- ☁️ (Optional) Submitted the file to Kaggle using the Kaggle API.
