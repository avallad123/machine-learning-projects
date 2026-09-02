# Housing Price Regression

A linear regression exploration of the [UCI Real Estate Valuation dataset](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set) (Yeh & Hsu, 2018) — housing transactions in Sindian District, New Taipei City, Taiwan. Predicts price per unit area from house age, distance to the nearest MRT (transit) station, and the number of nearby convenience stores.

This project was inspired by a general linear regression exercise from a data science / machine learning course, but the code, evaluation methodology, and analysis here are original — built independently rather than reproducing any graded assignment.

> **Note:** This is a smaller-scale, self-directed exercise built to practice a proper regression workflow (cross-validation, standardized feature importance, testing) rather than a from-scratch model implementation — the underlying `scikit-learn` estimator is standard.

## What's different from a basic train/test-split exercise

- **K-fold cross-validation** instead of comparing a few manually chosen random train/test splits — gives a more statistically stable estimate of model performance (mean and standard deviation of RMSE across folds) rather than picking "the best" of three arbitrary seeds.
- **Standardized coefficients for feature importance** — features are scaled to zero mean/unit variance before fitting a second model purely for importance ranking, so a feature measured in meters (distance) is fairly comparable to one measured in unit counts (convenience stores), rather than comparing raw, differently-scaled coefficients directly.
- **A small test suite** using a synthetic dataset with a known linear relationship, checking that the model recovers the expected signs and that metrics are internally consistent (e.g. RMSE ≈ √MSE).
- **A visualization script** producing a predicted-vs-actual scatter plot, which isn't part of a basic metrics-only exercise.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib pytest
```

Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set) (or any CSV with the same column layout).

## Running

```bash
cd src
python3 housing_regression.py --data path/to/real_estate.csv
```

Example output:
```
Cross-validated performance (5-fold):
  Mean MAE:  6.545
  Mean MSE:  86.384
  Mean RMSE: 9.190  (+/- 1.386)

Standardized feature importance (most influential first):
  distance_to_transit_m        -6.781
  nearby_convenience_stores    +3.817
  house_age_years              -2.877

Full-data model intercept: 42.98
```

Generate the predicted-vs-actual plot:

```bash
python3 plot_results.py --data path/to/real_estate.csv
```

Run the tests:

```bash
cd ..
python3 -m pytest tests/ -v
```

## Interpreting the results

Distance to the nearest MRT station is the strongest (negative) predictor of price per unit area once features are placed on a comparable scale — properties farther from transit are systematically cheaper. Nearby convenience store count is a positive predictor (a rough proxy for neighborhood walkability/density), and house age has a smaller negative effect.

## Concepts demonstrated

- Structuring a regression workflow into clear, testable functions rather than a single top-to-bottom script
- K-fold cross-validation with `scikit-learn`'s `cross_validate`
- Feature scaling with `StandardScaler` inside a `Pipeline`, and why standardized coefficients are more interpretable for comparing feature importance across differently-scaled inputs
- Writing unit tests against a synthetic dataset with a known ground-truth relationship
- Basic result visualization with `matplotlib`
