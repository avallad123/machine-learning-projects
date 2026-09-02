import argparse
import os

import matplotlib.pyplot as plt

from real_estate_eda import run_pipeline


def plot_price_by_convenience_stores(grouped: "pd.Series", output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    grouped.plot.bar(ax=ax, color="steelblue")
    ax.set_title("Mean House Price by Convenience Store Proximity")
    ax.set_xlabel("Number of nearby convenience stores")
    ax.set_ylabel("Price (USD/m^2)")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved bar chart to {output_path}")


def plot_age_tier_distribution(age_tier_counts: "pd.Series", output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    age_tier_counts.plot.bar(ax=ax, color="darkorange")
    ax.set_title("House Count by Age Tier (Quantile-Based)")
    ax.set_xlabel("Age tier")
    ax.set_ylabel("Number of houses")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved age tier chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize the real estate EDA results.")
    parser.add_argument("--data", required=True, help="Path to the real estate valuation CSV file.")
    parser.add_argument("--price-output", default="outputs/price_by_stores.png")
    parser.add_argument("--age-output", default="outputs/age_tier_distribution.png")
    args = parser.parse_args()

    results = run_pipeline(args.data)
    plot_price_by_convenience_stores(results["grouped_price_by_stores"], args.price_output)
    plot_age_tier_distribution(results["age_tier_counts"], args.age_output)


if __name__ == "__main__":
    main()
