import argparse
import os

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from housing_regression import load_and_clean, FEATURE_COLUMNS, TARGET_COLUMN


def plot_predicted_vs_actual(csv_path: str, output_path: str) -> None:
    df = load_and_clean(csv_path)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")

    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="Perfect prediction")

    ax.set_xlabel("Actual price per unit area")
    ax.set_ylabel("Predicted price per unit area")
    ax.set_title("Predicted vs. Actual Housing Prices")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot predicted vs. actual housing prices.")
    parser.add_argument("--data", required=True, help="Path to the real estate valuation CSV file.")
    parser.add_argument("--output", default="outputs/predicted_vs_actual.png", help="Output image path.")
    args = parser.parse_args()

    plot_predicted_vs_actual(args.data, args.output)


if __name__ == "__main__":
    main()
