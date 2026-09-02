import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn import metrics


TARGET_COLUMN = "Outcome"

# Rather than hand-picking one fixed set of hyperparameters, search over
# a small grid and let cross-validation pick the best combination. This
# also removes the need to separately loop over fold counts afterward --
# the grid search itself uses cross-validation internally.
PARAM_GRID = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 10],
    "min_samples_leaf": [1, 4],
}


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    best_params: dict
    cv_accuracy_mean: float
    cv_accuracy_std: float


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.dropna()

    # Several columns in this dataset use 0 as an implicit "missing"
    # sentinel for physiologically impossible values (e.g. 0 blood
    # pressure or 0 BMI). Replace those with the column median rather
    # than treating them as real zero measurements.
    zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    for col in zero_as_missing_cols:
        if col in df.columns:
            median_val = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, median_val)

    return df


def tune_and_train(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5, random_state: int = 1) -> GridSearchCV:
    """Search PARAM_GRID with stratified cross-validation and return the
    fitted GridSearchCV object (best estimator accessible via .best_estimator_)."""
    skfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    base_clf = RandomForestClassifier(max_features="sqrt", random_state=random_state)
    search = GridSearchCV(base_clf, PARAM_GRID, cv=skfold, scoring="f1_weighted", n_jobs=-1)
    search.fit(X_train, y_train)

    return search


def evaluate(search: GridSearchCV, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_probs = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, y_probs)
    auc_score = metrics.auc(fpr, tpr)

    cv_results = search.cv_results_
    best_index = search.best_index_

    return EvaluationResult(
        accuracy=metrics.accuracy_score(y_test, y_pred),
        precision=metrics.precision_score(y_test, y_pred, average="weighted"),
        recall=metrics.recall_score(y_test, y_pred, average="weighted"),
        f1=metrics.f1_score(y_test, y_pred, average="weighted"),
        auc=auc_score,
        best_params=search.best_params_,
        cv_accuracy_mean=float(cv_results["mean_test_score"][best_index]),
        cv_accuracy_std=float(cv_results["std_test_score"][best_index]),
    )


def feature_importance(search: GridSearchCV, feature_names: list) -> pd.Series:
    best_model = search.best_estimator_
    importances = pd.Series(best_model.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)


def run_pipeline(csv_path: str, test_size: float = 0.3, random_state: int = 0) -> tuple:
    df = load_and_clean(csv_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    search = tune_and_train(X_train, y_train)
    result = evaluate(search, X_test, y_test)
    importances = feature_importance(search, list(X.columns))

    return result, importances, search


def print_summary(result: EvaluationResult, importances: pd.Series) -> None:
    print("Best hyperparameters found via grid search + stratified cross-validation:")
    for k, v in result.best_params.items():
        print(f"  {k}: {v}")
    print()

    print(f"Cross-validated F1 (weighted): {result.cv_accuracy_mean:.4f} (+/- {result.cv_accuracy_std:.4f})")
    print()

    print("Held-out test set performance:")
    print(f"  Accuracy:  {result.accuracy:.4f}")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  F1:        {result.f1:.4f}")
    print(f"  AUC:       {result.auc:.4f}")
    print()

    print("Feature importance (most predictive first):")
    for feature, score in importances.items():
        print(f"  {feature:28s} {score:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Random forest diabetes risk classifier with hyperparameter search.")
    parser.add_argument("--data", required=True, help="Path to the diabetes CSV file.")
    args = parser.parse_args()

    result, importances, _ = run_pipeline(args.data)
    print_summary(result, importances)


if __name__ == "__main__":
    main()
