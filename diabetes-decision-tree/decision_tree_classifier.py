import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn import metrics


TARGET_COLUMN = "Outcome"
ZERO_AS_MISSING_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


@dataclass
class PruningResult:
    alpha: float
    cv_accuracy_mean: float
    cv_accuracy_std: float
    tree_depth: int
    tree_leaf_count: int


@dataclass
class EvaluationResult:
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.dropna()

    for col in ZERO_AS_MISSING_COLS:
        if col in df.columns:
            median_val = df.loc[df[col] != 0, col].median()
            df[col] = df[col].replace(0, median_val)

    return df


def find_best_pruning_alpha(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5, random_state: int = 100) -> list:
    """Rather than fixing max_depth and min_samples_leaf by hand, grow a
    full tree, extract its cost-complexity pruning path (the sequence of
    effective alphas at which the tree would be pruned back), and
    cross-validate each resulting pruned tree to find the alpha that
    generalizes best. This is a technique specific to trees (it doesn't
    apply to ensemble methods like random forests), so it's a more
    tree-appropriate way to control overfitting than an arbitrary fixed
    depth limit.
    """
    full_tree = DecisionTreeClassifier(criterion="entropy", random_state=random_state)
    path = full_tree.cost_complexity_pruning_path(X_train, y_train)
    alphas = path.ccp_alphas

    # very small/negative alphas at the start of the path are numerically
    # uninteresting (they barely prune anything); sample a reasonable
    # number of candidate points across the range instead of every alpha
    if len(alphas) > 20:
        indices = np.linspace(0, len(alphas) - 1, 20).astype(int)
        alphas = alphas[indices]

    skfold = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    results = []

    for alpha in alphas:
        tree = DecisionTreeClassifier(criterion="entropy", random_state=random_state, ccp_alpha=alpha)
        scores = cross_val_score(tree, X_train, y_train, cv=skfold)

        # fit once on the full training set just to report depth/leaf count
        tree.fit(X_train, y_train)

        results.append(PruningResult(
            alpha=float(alpha),
            cv_accuracy_mean=float(scores.mean()),
            cv_accuracy_std=float(scores.std()),
            tree_depth=tree.get_depth(),
            tree_leaf_count=tree.get_n_leaves(),
        ))

    return results


def evaluate(clf: DecisionTreeClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> EvaluationResult:
    y_pred = clf.predict(X_test)
    y_probs = clf.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = metrics.roc_curve(y_test, y_probs)
    auc_score = metrics.auc(fpr, tpr)

    return EvaluationResult(
        accuracy=metrics.accuracy_score(y_test, y_pred),
        precision=metrics.precision_score(y_test, y_pred, average="weighted"),
        recall=metrics.recall_score(y_test, y_pred, average="weighted"),
        f1=metrics.f1_score(y_test, y_pred, average="weighted"),
        auc=auc_score,
    )


def extract_rules(clf: DecisionTreeClassifier, feature_names: list) -> str:
    """Return a human-readable text representation of the tree's decision
    rules -- a form of interpretability a random forest can't offer as
    directly, since it's an ensemble of many trees rather than one."""
    return export_text(clf, feature_names=feature_names)


def run_pipeline(csv_path: str, test_size: float = 0.2, random_state: int = 1) -> tuple:
    df = load_and_clean(csv_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pruning_results = find_best_pruning_alpha(X_train, y_train)
    best_pruning = max(pruning_results, key=lambda r: r.cv_accuracy_mean)

    final_clf = DecisionTreeClassifier(criterion="entropy", random_state=100, ccp_alpha=best_pruning.alpha)
    final_clf.fit(X_train, y_train)

    result = evaluate(final_clf, X_test, y_test)

    return final_clf, result, best_pruning, pruning_results, list(X.columns)


def print_summary(clf, result: EvaluationResult, best_pruning: PruningResult, feature_names: list) -> None:
    print("Cost-complexity pruning selected the following tree:")
    print(f"  alpha:            {best_pruning.alpha:.5f}")
    print(f"  cross-val acc:    {best_pruning.cv_accuracy_mean:.4f} (+/- {best_pruning.cv_accuracy_std:.4f})")
    print(f"  tree depth:       {best_pruning.tree_depth}")
    print(f"  number of leaves: {best_pruning.tree_leaf_count}")
    print()

    print("Held-out test set performance:")
    print(f"  Accuracy:  {result.accuracy:.4f}")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall:    {result.recall:.4f}")
    print(f"  F1:        {result.f1:.4f}")
    print(f"  AUC:       {result.auc:.4f}")
    print()

    print("Decision rules learned by the pruned tree:")
    print(extract_rules(clf, feature_names))


def main():
    parser = argparse.ArgumentParser(description="Interpretable decision tree classifier with cost-complexity pruning.")
    parser.add_argument("--data", required=True, help="Path to the diabetes CSV file.")
    args = parser.parse_args()

    clf, result, best_pruning, _, feature_names = run_pipeline(args.data)
    print_summary(clf, result, best_pruning, feature_names)


if __name__ == "__main__":
    main()
