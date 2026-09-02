import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


COLUMN_RENAME_MAP = {
    "No": "record_id",
    "X1 transaction date": "transaction_date",
    "X2 house age": "house_age_years",
    "X3 distance to the nearest MRT station": "distance_to_transit_m",
    "X4 number of convenience stores": "nearby_convenience_stores",
    "X5 latitude": "latitude",
    "X6 longitude": "longitude",
    "Y house price of unit area": "price_10k_ntd_per_ping",
}

# Conversion constants: the raw target is in units of 10,000 New Taiwan
# Dollars per "Ping" (a local area unit). To express price per square
# meter in USD: multiply by 10,000 to get NTD/Ping, multiply by the
# NTD->USD rate to get USD/Ping, then divide by (m^2 per Ping) to get
# USD/m^2 -- dividing, not multiplying, since a Ping is the larger unit
# and converting to a smaller unit (m^2) means a smaller price per unit.
NTD_PER_10K_UNIT = 10_000
NTD_TO_USD_RATE = 0.03
M2_PER_PING = 3.3


@dataclass
class MissingValueReport:
    counts_by_column: pd.Series
    total_missing: int


@dataclass
class OutlierBounds:
    column: str
    lower: float
    upper: float
    n_removed: int


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_RENAME_MAP)
    return df


def missing_value_report(df: pd.DataFrame) -> MissingValueReport:
    counts = df.isnull().sum()
    return MissingValueReport(counts_by_column=counts, total_missing=int(counts.sum()))


def impute_missing_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values with each column's median, rather than
    dropping rows outright -- preserves sample size, which matters more
    on a dataset this small (~400 rows) than it would on a larger one."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df = df.copy()
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list, iqr_multiplier: float = 1.5) -> tuple:
    """Remove outliers using the standard IQR rule (values beyond
    Q1 - k*IQR or Q3 + k*IQR) for each specified column, rather than
    hand-picked absolute thresholds. This adapts to the actual
    distribution of the data instead of relying on manually chosen
    cutoff values that would need to be re-picked for a different
    dataset or population."""
    df = df.copy()
    bounds_report = []

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - iqr_multiplier * iqr
        upper = q3 + iqr_multiplier * iqr

        before = len(df)
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        removed = before - len(df)

        bounds_report.append(OutlierBounds(column=col, lower=lower, upper=upper, n_removed=removed))

    return df, bounds_report


def convert_price_to_usd_per_sqm(df: pd.DataFrame, price_column: str = "price_10k_ntd_per_ping") -> pd.DataFrame:
    df = df.copy()
    df["price_usd_per_sqm"] = (
        df[price_column] * NTD_PER_10K_UNIT * NTD_TO_USD_RATE / M2_PER_PING
    )
    return df


def normalize_numeric_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    df = df.copy()
    scaler = MinMaxScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df


def classify_house_age(df: pd.DataFrame, age_column: str = "house_age_years") -> pd.DataFrame:
    """Bucket house age into quantile-based tiers (roughly equal-sized
    groups) rather than fixed cutoffs like <10/<30, so the categories
    adapt to the actual age distribution in the data instead of assuming
    cutoffs that happen to make sense for one particular dataset."""
    df = df.copy()
    df["age_tier"] = pd.qcut(df[age_column], q=3, labels=["Newer Third", "Middle Third", "Older Third"])
    return df


def price_by_convenience_stores(df: pd.DataFrame, age_max: float = 9, price_min: float = 27) -> pd.Series:
    subset = df[(df["house_age_years"] <= age_max) & (df["price_usd_per_sqm"] > price_min)]
    return subset.groupby("nearby_convenience_stores")["price_usd_per_sqm"].mean().round(2)


def run_pipeline(csv_path: str) -> dict:
    df = load_data(csv_path)
    missing_before = missing_value_report(df)

    df_imputed = impute_missing_with_median(df)

    outlier_columns = ["price_10k_ntd_per_ping", "distance_to_transit_m", "longitude"]
    df_clean, outlier_bounds = remove_outliers_iqr(df_imputed, outlier_columns)

    df_converted = convert_price_to_usd_per_sqm(df_clean)
    df_classified = classify_house_age(df_converted)

    age_tier_counts = df_classified["age_tier"].value_counts().sort_index()

    numeric_cols = df_converted.select_dtypes(include=[np.number]).columns.tolist()
    df_normalized = normalize_numeric_columns(df_converted, numeric_cols)

    grouped = price_by_convenience_stores(df_converted)

    return {
        "missing_before": missing_before,
        "outlier_bounds": outlier_bounds,
        "df_clean": df_converted,
        "df_classified": df_classified,
        "age_tier_counts": age_tier_counts,
        "df_normalized": df_normalized,
        "grouped_price_by_stores": grouped,
    }


def print_summary(results: dict) -> None:
    mv = results["missing_before"]
    print(f"Missing values before imputation: {mv.total_missing} total")
    print(mv.counts_by_column[mv.counts_by_column > 0])
    print()

    print("Outlier removal (IQR method):")
    for b in results["outlier_bounds"]:
        print(f"  {b.column}: kept [{b.lower:.2f}, {b.upper:.2f}], removed {b.n_removed} rows")
    print()

    print("House age tiers (quantile-based):")
    print(results["age_tier_counts"])
    print()

    print("Normalized price column mean:", round(results["df_normalized"]["price_usd_per_sqm"].mean(), 4))
    print()

    print("Mean price (USD/m^2) by nearby convenience stores, for newer/higher-priced homes:")
    print(results["grouped_price_by_stores"])


def main():
    parser = argparse.ArgumentParser(description="Real estate EDA and data wrangling pipeline.")
    parser.add_argument("--data", required=True, help="Path to the real estate valuation CSV file.")
    args = parser.parse_args()

    results = run_pipeline(args.data)
    print_summary(results)


if __name__ == "__main__":
    main()
