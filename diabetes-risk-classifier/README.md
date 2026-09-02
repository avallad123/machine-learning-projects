# Diabetes Risk Classifier

A random forest classifier predicting diabetes risk from the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) — diagnostic measurements (glucose, BMI, blood pressure, age, etc.) for a population of women, with a binary outcome for diabetes diagnosis.

This project was inspired by a general classification exercise from a data science / machine learning course, but the code, tuning methodology, and analysis here are original — built independently rather than reproducing any graded assignment.

> **Note:** This is a smaller-scale, self-directed exercise built to practice a proper classification workflow (data cleaning, hyperparameter search, cross-validation, interpretability) rather than a from-scratch model implementation — the underlying `scikit-learn` estimator is standard.

## What's different from a basic fixed-hyperparameter exercise

- **Automated hyperparameter search** (`GridSearchCV` over criterion, max depth, and split/leaf sizes) combined with stratified cross-validation, rather than a single hand-picked set of hyperparameters or a manual loop over fold counts — the grid search handles both at once and reports the best combination found.
- **Sentinel-value cleaning**: several columns in this dataset (glucose, blood pressure, skin thickness, insulin, BMI) use `0` as an implausible placeholder for missing data rather than a real measurement. These are detected and replaced with the column median instead of being treated as valid zero readings, which a naive null-check alone would miss.
- **Feature importance analysis** using the trained forest's built-in importances — not part of a metrics-only exercise, and useful for understanding which measurements actually drive the model's predictions.
- **A test suite** built on synthetic data with a known, engineered signal (glucose and BMI driving the outcome), checking that the model both outperforms random guessing and correctly identifies the true signal-bearing features.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib pytest
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or the original UCI source.

## Running

```bash
cd src
python3 diabetes_classifier.py --data path/to/diabetes.csv
```

Example output:
```
Best hyperparameters found via grid search + stratified cross-validation:
  criterion: gini
  max_depth: None
  min_samples_leaf: 1
  min_samples_split: 2

Cross-validated F1 (weighted): 0.7638 (+/- 0.0240)

Held-out test set performance:
  Accuracy:  0.7576
  Precision: 0.7532
  Recall:    0.7576
  F1:        0.7544
  AUC:       0.8251

Feature importance (most predictive first):
  Glucose                      0.2447
  BMI                          0.1566
  Age                          0.1443
  DiabetesPedigreeFunction     0.1237
  BloodPressure                0.0906
  Insulin                      0.0876
  SkinThickness                0.0768
  Pregnancies                  0.0756
```

Generate the ROC curve:

```bash
python3 plot_roc.py --data path/to/diabetes.csv
```

Run the tests:

```bash
cd ..
python3 -m pytest tests/ -v
```

## Interpreting the results

Glucose level is by far the strongest predictor of diabetes risk in this model, consistent with its clinical role as a direct diagnostic marker. BMI and age follow as the next most influential features, while pregnancy count contributes the least — a sensible finding, since pregnancy count is a much less direct physiological signal than the other measurements.

## Concepts demonstrated

- Detecting and handling implicit missing-value sentinels in real-world tabular data
- Combining hyperparameter search with cross-validation via `GridSearchCV` and `StratifiedKFold`
- Evaluating a classifier with accuracy, precision, recall, F1, and ROC-AUC
- Interpreting a random forest via built-in feature importances
- Writing tests against synthetic data with a known ground-truth signal to validate both predictive performance and interpretability
