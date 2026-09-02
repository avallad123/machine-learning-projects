import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


COLUMN_RENAME_MAP = {
    "No": "record_id",
    "X1 transaction date": "transaction_date",
    "X2 house age": "house_age_years",
    "X3 distance to the nearest MRT station": "distance_to_transit_m",
    "X4 number of convenience stores": "nearby_convenience_stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
    "Y house price of unit area": "price_per_unit_area",
}

FEATURE_COLUMNS = ["house_age_years", "distance_to_transit_m", "nearby_convenience_stores"]
TARGET_COLUMN = "price_per_unit_area"


@dataclass
class FoldMetrics:
    """Aggregated cross-validation performance for one model configuration."""
    mean_mae: float
    mean_mse: float
    mean_rmse: float
    std_rmse: float


def load_and_clean(csv_path: str) -> pd.DataFrame:
    """Load the raw CSV, apply readable column names, and drop any rows
    with missing values in the columns we actually use."""
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_RENAME_MAP)

    relevant_cols = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_before = df[relevant_cols].isnull().sum().sum()
    if missing_before > 0:
        df = df.dropna(subset=relevant_cols)

    return df


def cross_validated_performance(df: pd.DataFrame, n_splits: int = 5, random_state: int = 42) -> FoldMetrics:
    """Evaluate a linear regression model with k-fold cross-validation
    rather than a handful of manually chosen train/test splits, giving
    a more stable read on expected performance."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    model = LinearRegression()

    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
    }
    results = cross_validate(model, X, y, cv=kfold, scoring=scoring)

    mae_scores = -results["test_mae"]
    mse_scores = -results["test_mse"]
    rmse_scores = np.sqrt(mse_scores)

    return FoldMetrics(
        mean_mae=float(np.mean(mae_scores)),
        mean_mse=float(np.mean(mse_scores)),
        mean_rmse=float(np.mean(rmse_scores)),
        std_rmse=float(np.std(rmse_scores)),
    )


def standardized_feature_importance(df: pd.DataFrame) -> pd.Series:
    """Fit a linear model on standardized features (zero mean, unit
    variance) so coefficient magnitudes are directly comparable across
    features measured in very different units, then return them sorted
    by absolute influence on the target, most influential first."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    pipeline = make_pipeline(StandardScaler(), LinearRegression())
    pipeline.fit(X, y)

    model = pipeline.named_steps["linearregression"]
    coefficients = pd.Series(model.coef_, index=FEATURE_COLUMNS)
    return coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)


def fit_full_model(df: pd.DataFrame) -> LinearRegression:
    """Fit a plain (unstandardized) linear model on the full dataset,
    useful for reporting an interpretable intercept and per-unit
    coefficients."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    model = LinearRegression()
    model.fit(X, y)
    return model


def summarize(df: pd.DataFrame) -> None:
    metrics = cross_validated_performance(df)
    print("Cross-validated performance (5-fold):")
    print(f"  Mean MAE:  {metrics.mean_mae:.3f}")
    print(f"  Mean MSE:  {metrics.mean_mse:.3f}")
    print(f"  Mean RMSE: {metrics.mean_rmse:.3f}  (+/- {metrics.std_rmse:.3f})")
    print()

    importance = standardized_feature_importance(df)
    print("Standardized feature importance (most influential first):")
    for feature, coef in importance.items():
        print(f"  {feature:28s} {coef:+.3f}")
    print()

    model = fit_full_model(df)
    print(f"Full-data model intercept: {model.intercept_:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Linear regression exploration on housing valuation data.")
    parser.add_argument("--data", required=True, help="Path to the real estate valuation CSV file.")
    args = parser.parse_args()

    df = load_and_clean(args.data)
    summarize(df)


if __name__ == "__main__":
    main()
