# Diabetes Decision Tree

An interpretable decision tree classifier predicting diabetes risk from the [Pima Indians Diabetes dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database), built around cost-complexity pruning and human-readable rule extraction.

This project was inspired by a general classification exercise from a data science / machine learning course, but the code, tuning methodology, and analysis here are original — built independently rather than reproducing any graded assignment. It uses the same public dataset as a separate [random forest project](../diabetes-risk-classifier) in this repo, deliberately, to highlight what a single interpretable tree offers that an ensemble model doesn't.

> **Note:** This is a smaller-scale, self-directed exercise built to practice tree-specific analysis techniques (pruning, rule extraction) rather than a from-scratch model implementation — the underlying `scikit-learn` estimator is standard.

## What's different from a basic fixed-hyperparameter exercise

- **Cost-complexity pruning** instead of a hand-picked `max_depth`/`min_samples_leaf`: a full tree is grown, its pruning path (the sequence of effective alphas at which subtrees would be pruned away) is extracted, and each candidate pruned tree is cross-validated to find the alpha that generalizes best. This is a technique specific to individual trees — it doesn't have a direct equivalent for ensemble methods like random forests — so it's a more tree-appropriate way to control overfitting than an arbitrary fixed depth limit.
- **Human-readable rule extraction**: the final pruned tree's decision logic is printed as plain-language if/else rules (via `export_text`), something a single tree can offer directly that a forest of many trees cannot.
- **Pruning curve and tree structure visualizations**, showing how cross-validated accuracy changes as the tree is pruned more aggressively, and a plotted diagram of the final tree itself.
- **A test suite** built on synthetic data with a known, engineered signal, checking that pruning behaves correctly (more pruning never increases tree size), that the model beats random guessing, and that the extracted rules reference the actual signal-bearing features.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib pytest
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) or the original UCI source.

## Running

```bash
cd src
python3 decision_tree_classifier.py --data path/to/diabetes.csv
```

Example output:
```
Cost-complexity pruning selected the following tree:
  alpha:            0.01376
  cross-val acc:    0.7621 (+/- 0.0567)
  tree depth:       4
  number of leaves: 8

Held-out test set performance:
  Accuracy:  0.7468
  Precision: 0.7669
  Recall:    0.7468
  F1:        0.7516
  AUC:       0.8260

Decision rules learned by the pruned tree:
|--- Glucose <= 127.50
|   |--- Age <= 28.50
|   |   |--- BMI <= 31.30
|   |   |   |--- class: 0
...
```

Generate the pruning curve and tree structure diagrams:

```bash
python3 plot_tree_analysis.py --data path/to/diabetes.csv
```

Run the tests:

```bash
cd ..
python3 -m pytest tests/ -v
```

## Interpreting the results

The pruned tree splits first on glucose level, then on age or BMI depending on the branch — consistent with glucose being the dominant diabetes risk signal, with age and body mass as secondary factors. Because the tree is shallow (a handful of levels deep after pruning), the entire decision logic can be read directly as a small set of threshold rules, which is the main practical advantage a single decision tree has over a less transparent ensemble model: a clinician or analyst could sanity-check every branch of this model by eye.

## Concepts demonstrated

- Growing a full decision tree and extracting its cost-complexity pruning path
- Using cross-validation to select a pruning strength (alpha) that balances underfitting and overfitting
- Extracting and reading a trained tree's decision logic as human-readable rules
- Visualizing both a bias/variance-style tradeoff curve (accuracy vs. pruning strength) and the tree structure itself
- Writing tests against synthetic data with a known ground-truth signal to validate pruning behavior, predictive performance, and interpretability
