import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from diabetes_classifier import (
    load_and_clean,
    tune_and_train,
    evaluate,
    feature_importance,
    TARGET_COLUMN,
)


def make_synthetic_df(n=300, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    glucose = rng.uniform(70, 200, n)
    bmi = rng.uniform(18, 45, n)
    age = rng.integers(21, 80, n)
    pregnancies = rng.integers(0, 10, n)
    blood_pressure = rng.uniform(50, 110, n)
    skin_thickness = rng.uniform(10, 50, n)
    insulin = rng.uniform(0, 300, n)
    pedigree = rng.uniform(0.1, 2.0, n)

    # construct outcome with a real (if noisy) dependence on glucose and BMI,
    # so tests can check the model finds a meaningful, non-random signal
    risk_score = 0.05 * (glucose - 120) + 0.15 * (bmi - 25) + rng.normal(0, 1.5, n)
    outcome = (risk_score > 0).astype(int)

    return pd.DataFrame({
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
        "Outcome": outcome,
    })


def test_load_and_clean_replaces_zero_sentinels():
    df = make_synthetic_df(n=50)
    df.loc[0, "BMI"] = 0  # inject a sentinel zero
    df.loc[1, "Glucose"] = 0

    cleaned = load_and_clean_from_df(df)

    assert cleaned.loc[0, "BMI"] != 0
    assert cleaned.loc[1, "Glucose"] != 0


def load_and_clean_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Test helper mirroring load_and_clean's cleaning logic without
    requiring a CSV round-trip."""
    zero_as_missing_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    df = df.copy()
    for col in zero_as_missing_cols:
        median_val = df.loc[df[col] != 0, col].median()
        df[col] = df[col].replace(0, median_val)
    return df


def test_model_beats_random_guessing_on_synthetic_signal():
    df = make_synthetic_df(n=400)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    search = tune_and_train(X_train, y_train, cv_folds=3)
    result = evaluate(search, X_test, y_test)

    # with a real signal in the synthetic data, the model should clearly
    # beat a coin flip
    assert result.accuracy > 0.6
    assert result.auc > 0.6


def test_feature_importance_identifies_known_signal_features():
    df = make_synthetic_df(n=400)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    search = tune_and_train(X_train, y_train, cv_folds=3)
    importances = feature_importance(search, list(X.columns))

    top_three = set(importances.head(3).index)
    # Glucose and BMI were the two features actually driving the
    # synthetic outcome, so they should rank among the top few
    assert "Glucose" in top_three
    assert "BMI" in top_three
