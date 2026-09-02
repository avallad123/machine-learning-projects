import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from housing_regression import (
    cross_validated_performance,
    standardized_feature_importance,
    fit_full_model,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def make_synthetic_df(n=100, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    house_age = rng.uniform(0, 40, n)
    distance = rng.uniform(50, 2000, n)
    stores = rng.integers(0, 10, n)

    # construct a target with a known linear relationship plus noise,
    # so we can sanity-check the model recovers roughly the right signs
    price = 50 - 0.3 * house_age - 0.01 * distance + 1.5 * stores + rng.normal(0, 2, n)

    return pd.DataFrame({
        "house_age_years": house_age,
        "distance_to_transit_m": distance,
        "nearby_convenience_stores": stores,
        "price_per_unit_area": price,
    })


def test_cross_validated_performance_returns_reasonable_metrics():
    df = make_synthetic_df()
    metrics = cross_validated_performance(df, n_splits=5)

    assert metrics.mean_mae >= 0
    assert metrics.mean_mse >= 0
    assert metrics.mean_rmse >= 0
    # RMSE should be roughly sqrt(MSE)
    assert abs(metrics.mean_rmse ** 2 - metrics.mean_mse) < 1.0


def test_feature_importance_recovers_expected_signs():
    df = make_synthetic_df(n=500)
    importance = standardized_feature_importance(df)

    assert set(importance.index) == set(FEATURE_COLUMNS)
    # distance has a negative true relationship with price
    assert importance["distance_to_transit_m"] < 0
    # convenience stores has a positive true relationship with price
    assert importance["nearby_convenience_stores"] > 0


def test_fit_full_model_produces_finite_intercept():
    df = make_synthetic_df()
    model = fit_full_model(df)

    assert np.isfinite(model.intercept_)
    assert len(model.coef_) == len(FEATURE_COLUMNS)
