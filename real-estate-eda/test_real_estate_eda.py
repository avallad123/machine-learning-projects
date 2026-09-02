import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from real_estate_eda import (
    impute_missing_with_median,
    remove_outliers_iqr,
    convert_price_to_usd_per_sqm,
    normalize_numeric_columns,
    classify_house_age,
    NTD_PER_10K_UNIT,
    NTD_TO_USD_RATE,
    M2_PER_PING,
)


def make_synthetic_df(n=100, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "house_age_years": rng.uniform(0, 40, n),
        "distance_to_transit_m": rng.uniform(50, 2000, n),
        "nearby_convenience_stores": rng.integers(0, 10, n),
        "price_10k_ntd_per_ping": rng.uniform(10, 60, n),
    })
    return df


def test_impute_missing_with_median_fills_all_nans():
    df = make_synthetic_df()
    df.loc[0, "house_age_years"] = np.nan
    df.loc[5, "price_10k_ntd_per_ping"] = np.nan

    result = impute_missing_with_median(df)

    assert result.isnull().sum().sum() == 0


def test_remove_outliers_iqr_removes_extreme_values():
    df = make_synthetic_df(n=200)
    # inject a clear outlier
    df.loc[0, "price_10k_ntd_per_ping"] = 10_000

    cleaned, bounds_report = remove_outliers_iqr(df, ["price_10k_ntd_per_ping"])

    assert len(cleaned) < len(df)
    assert 10_000 not in cleaned["price_10k_ntd_per_ping"].values
    assert bounds_report[0].n_removed >= 1


def test_price_conversion_matches_expected_formula():
    df = pd.DataFrame({"price_10k_ntd_per_ping": [10.0]})
    result = convert_price_to_usd_per_sqm(df)

    expected = 10.0 * NTD_PER_10K_UNIT * NTD_TO_USD_RATE / M2_PER_PING
    assert np.isclose(result["price_usd_per_sqm"].iloc[0], expected)


def test_normalize_numeric_columns_scales_to_unit_range():
    df = make_synthetic_df()
    columns = ["house_age_years", "distance_to_transit_m"]

    normalized = normalize_numeric_columns(df, columns)

    for col in columns:
        assert normalized[col].min() >= 0
        assert normalized[col].max() <= 1
        assert np.isclose(normalized[col].min(), 0)
        assert np.isclose(normalized[col].max(), 1)


def test_classify_house_age_produces_roughly_equal_tiers():
    df = make_synthetic_df(n=300)
    classified = classify_house_age(df)

    tier_counts = classified["age_tier"].value_counts()
    # quantile-based tiers should be roughly balanced (each ~1/3 of 300 = 100)
    for count in tier_counts:
        assert 80 <= count <= 120
