import argparse
import os

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from decision_tree_classifier import run_pipeline


def plot_pruning_curve(pruning_results: list, output_path: str) -> None:
    alphas = [r.alpha for r in pruning_results]
    accuracies = [r.cv_accuracy_mean for r in pruning_results]
    stds = [r.cv_accuracy_std for r in pruning_results]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(alphas, accuracies, yerr=stds, marker="o", color="darkorange", capsize=3)
    ax.set_xlabel("Effective alpha (cost-complexity pruning parameter)")
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title("Accuracy vs. Pruning Strength")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved pruning curve to {output_path}")


def plot_tree_structure(clf, feature_names: list, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(clf, feature_names=feature_names, class_names=["No Diabetes", "Diabetes"],
              filled=True, rounded=True, fontsize=8, ax=ax)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved tree structure to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the pruning curve and final tree structure.")
    parser.add_argument("--data", required=True, help="Path to the diabetes CSV file.")
    parser.add_argument("--pruning-output", default="outputs/pruning_curve.png")
    parser.add_argument("--tree-output", default="outputs/tree_structure.png")
    args = parser.parse_args()

    clf, _, _, pruning_results, feature_names = run_pipeline(args.data)

    plot_pruning_curve(pruning_results, args.pruning_output)
    plot_tree_structure(clf, feature_names, args.tree_output)


if __name__ == "__main__":
    main()
