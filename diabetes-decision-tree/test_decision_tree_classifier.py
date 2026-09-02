import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from decision_tree_classifier import (
    find_best_pruning_alpha,
    evaluate,
    extract_rules,
    TARGET_COLUMN,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def make_synthetic_df(n=400, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    glucose = rng.uniform(70, 200, n)
    bmi = rng.uniform(18, 45, n)
    age = rng.integers(21, 80, n)
    pregnancies = rng.integers(0, 10, n)
    blood_pressure = rng.uniform(50, 110, n)
    skin_thickness = rng.uniform(10, 50, n)
    insulin = rng.uniform(0, 300, n)
    pedigree = rng.uniform(0.1, 2.0, n)

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


def test_pruning_path_produces_at_least_one_candidate():
    df = make_synthetic_df()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    results = find_best_pruning_alpha(X_train, y_train, cv_folds=3)

    assert len(results) > 0
    for r in results:
        assert r.alpha >= 0
        assert 0 <= r.cv_accuracy_mean <= 1
        assert r.tree_depth >= 0
        assert r.tree_leaf_count >= 1


def test_more_pruning_reduces_or_maintains_tree_size():
    df = make_synthetic_df()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    results = find_best_pruning_alpha(X_train, y_train, cv_folds=3)
    results_sorted = sorted(results, key=lambda r: r.alpha)

    # as alpha increases (more pruning), leaf count should not increase
    leaf_counts = [r.tree_leaf_count for r in results_sorted]
    assert leaf_counts[-1] <= leaf_counts[0]


def test_model_beats_random_guessing_on_synthetic_signal():
    df = make_synthetic_df()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    clf = DecisionTreeClassifier(criterion="entropy", random_state=100, ccp_alpha=0.01)
    clf.fit(X_train, y_train)

    result = evaluate(clf, X_test, y_test)

    assert result.accuracy > 0.6
    assert result.auc > 0.6


def test_extract_rules_mentions_known_signal_features():
    df = make_synthetic_df()
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    clf = DecisionTreeClassifier(criterion="entropy", random_state=100, max_depth=3)
    clf.fit(X_train, y_train)

    rules_text = extract_rules(clf, list(X.columns))

    # Glucose and BMI drive the synthetic outcome, so the extracted rules
    # should reference at least one of them
    assert "Glucose" in rules_text or "BMI" in rules_text
