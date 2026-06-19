# Natural Language Processing with Disaster Tweets

A machine learning project for the Kaggle **NLP Getting Started** competition, where the goal is to classify tweets as referring to a real disaster (`1`) or not (`0`). The competition provides about 10,000 labeled tweets and uses F1 score for evaluation [web:16][web:21].

## Project Overview

This notebook builds a complete baseline NLP pipeline in Google Colab using:

- Data loading from Kaggle competition files.
- Basic exploratory data analysis (EDA).
- Missing value handling for `keyword` and `location`.
- Text preprocessing with regular expressions.
- TF-IDF feature extraction using unigrams and bigrams.
- Logistic Regression for binary classification.
- Validation using F1 score, classification report, and confusion matrix.
- Submission file generation for Kaggle.

## Dataset

The competition dataset includes these main columns [web:16][web:17]:

- `id`: unique tweet identifier.
- `keyword`: disaster-related keyword, sometimes missing.
- `location`: user-provided location, often missing or noisy.
- `text`: the tweet content.
- `target`: label in the training set, where `1` means disaster and `0` means not disaster.

## Notebook Workflow

### 1. Setup and download
The notebook installs the Kaggle API, uploads `kaggle.json`, downloads the competition files, and extracts them into the Colab environment.

### 2. Import libraries
The workflow uses `pandas`, `numpy`, `matplotlib`, `seaborn`, `re`, and several `scikit-learn` modules for modeling and evaluation.

### 3. Load data
The notebook reads:

- `train.csv`
- `test.csv`
- `sample_submission.csv`

### 4. Handle missing values
Missing values are handled as follows:

- `keyword` is filled with `'none'`.
- `location` is filled with `'unknown'`.
- `id` is excluded from model training because it is only an identifier, but it remains necessary in the final submission file.

### 5. Exploratory data analysis
EDA includes:

- Checking data types and missing values.
- Examining class balance.
- Visualizing tweet length distributions.
- Inspecting the most frequent keywords by class.

### 6. Text preprocessing
Tweets are cleaned by:

- Lowercasing text.
- Removing URLs.
- Removing `@mentions`.
- Removing special characters and extra spaces.
- Preserving hashtag words by removing only the `#` symbol.

### 7. Feature extraction
TF-IDF converts the cleaned tweets into numeric vectors that can be used by a machine learning model. The notebook uses:

- `max_features=10000`
- `ngram_range=(1, 2)`
- `sublinear_tf=True`

### 8. Model training
The baseline classifier is Logistic Regression, trained on TF-IDF features with a train/validation split.

### 9. Evaluation
Model performance is checked with:

- F1 score.
- Classification report.
- Confusion matrix heatmap.

### 10. Submission
Predictions for the test set are saved into `submission.csv` using the Kaggle submission format.

## Repository Structure

```bash
.
├── README.md
├── notebook.ipynb
└── submission.csv
```

Update the file names if the notebook or submission file uses a different name in the repository.

## How to Run

### In Google Colab

1. Open the notebook in Colab.
2. Upload your `kaggle.json` file when prompted.
3. Run all cells from top to bottom.
4. Train the model and generate `submission.csv`.
5. Submit the file to Kaggle from Colab or through the Kaggle website.

## Baseline Model

The baseline is intentionally simple and strong for a first submission:

- TF-IDF text representation.
- Logistic Regression classifier.
- F1-based validation.

This is a common and effective starting point for short-text classification tasks such as tweets.

## Possible Improvements

Several upgrades can improve the baseline:

- Use `class_weight='balanced'` to handle mild class imbalance.
- Tune the classification threshold using validation probabilities.
- Add `keyword` and `location` as engineered text features.
- Try oversampling methods such as RandomOverSampler or SMOTE.
- Replace Logistic Regression with linear SVM, Naive Bayes, or transformer-based models.
- Fine-tune `distilbert-base-uncased` or `bert-base-uncased` on Colab GPU.

## Notes

- The `id` column should not be used as a training feature, but it must be present in the final Kaggle submission file.
- `location` contains many missing values and noisy free-text entries, so it may or may not help the model.
- Since the competition is evaluated with F1 score, improving recall and precision together matters more than raw accuracy [web:21].

## Competition Link

- [Kaggle: Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)

## License

This repository is for educational and competition purposes. Check Kaggle competition rules for dataset usage and submission requirements.
