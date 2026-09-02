import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from customer_segmentation import (
    select_clustering_features,
    search_cluster_counts,
    best_k_by_silhouette,
    fit_final_model,
    profile_clusters,
)
from sklearn.preprocessing import StandardScaler


def make_synthetic_customers(n_per_cluster=50, seed=0):
    """Generate synthetic customer data with 4 well-separated clusters
    placed on a grid (so the two informative features are naturally
    close to uncorrelated, mirroring real income/spending data), plus a
    third column that's weakly correlated with income (as age loosely is
    in real demographic data) so it isn't trivially "least correlated"
    just by being pure noise.
    """
    rng = np.random.default_rng(seed)

    # four cluster centers arranged in a grid, not a diagonal, so income
    # and spending don't pick up spurious correlation from cluster geometry
    centers = [(30, 30), (30, 80), (80, 30), (80, 80)]
    cluster_ids = []
    income_vals = []
    spending_vals = []

    for cluster_id, (income_center, spending_center) in enumerate(centers):
        income_vals.append(rng.normal(income_center, 5, n_per_cluster))
        spending_vals.append(rng.normal(spending_center, 5, n_per_cluster))
        cluster_ids.extend([cluster_id] * n_per_cluster)

    income = np.concatenate(income_vals)
    spending = np.concatenate(spending_vals)

    # age has a mild, real-ish relationship with income but no relationship
    # with spending, so it's more correlated overall than pure noise would be,
    # and should lose out to the (near-zero correlation) income/spending pair
    age = 20 + 0.3 * (income - income.mean()) + rng.normal(0, 15, len(income))

    df = pd.DataFrame({
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending,
        "Age": age,
    })
    return df, np.array(cluster_ids)


def test_feature_selection_prefers_low_correlation_columns():
    df, _ = make_synthetic_customers()
    selected = select_clustering_features(
        df, ["Age", "Annual Income (k$)", "Spending Score (1-100)"], top_n=2
    )

    assert len(selected) == 2
    assert "Annual Income (k$)" in selected
    assert "Spending Score (1-100)" in selected


def test_silhouette_selection_recovers_known_cluster_count():
    df, _ = make_synthetic_customers()
    X = df[["Annual Income (k$)", "Spending Score (1-100)"]].values
    X_scaled = StandardScaler().fit_transform(X)

    search_results = search_cluster_counts(X_scaled, range(2, 8))
    best_k = best_k_by_silhouette(search_results)

    # the synthetic data has exactly 4 well-separated blobs
    assert best_k == 4


def test_cluster_profiles_sum_to_total_rows():
    df, _ = make_synthetic_customers()
    feature_columns = ["Annual Income (k$)", "Spending Score (1-100)"]
    X_scaled = StandardScaler().fit_transform(df[feature_columns].values)

    model = fit_final_model(X_scaled, k=4)
    labels = model.predict(X_scaled)

    profiles = profile_clusters(df, feature_columns, labels)

    assert len(profiles) == 4
    assert sum(p.size for p in profiles) == len(df)
