import argparse
import os

import matplotlib.pyplot as plt
from sklearn import metrics

from diabetes_classifier import run_pipeline


def plot_roc(csv_path: str, output_path: str) -> None:
    result, _, search = run_pipeline(csv_path)

    # Re-derive fpr/tpr for plotting (evaluate() only returns the scalar AUC)
    from diabetes_classifier import load_and_clean, TARGET_COLUMN
    from sklearn.model_selection import train_test_split

    df = load_and_clean(csv_path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    best_model = search.best_estimator_
    y_probs = best_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="darkorange", label=f"AUC = {result.auc:.2f}")
    ax.plot([0, 1], [0, 1], "b--", label="Random guess")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Diabetes Risk Classifier")
    ax.legend(loc="lower right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved ROC curve to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot the ROC curve for the tuned diabetes classifier.")
    parser.add_argument("--data", required=True, help="Path to the diabetes CSV file.")
    parser.add_argument("--output", default="outputs/roc_curve.png", help="Output image path.")
    args = parser.parse_args()

    plot_roc(args.data, args.output)


if __name__ == "__main__":
    main()
