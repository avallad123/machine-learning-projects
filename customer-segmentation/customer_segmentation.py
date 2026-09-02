import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


@dataclass
class ClusterSearchResult:
    k: int
    wcss: float
    silhouette: float


@dataclass
class ClusterProfile:
    cluster_id: int
    size: int
    feature_means: pd.Series


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = df.isnull().sum().sum()
    if missing > 0:
        df = df.dropna()

    return df


def select_clustering_features(df: pd.DataFrame, candidate_columns: list, top_n: int = 2) -> list:
    """Rather than hand-picking two columns by eyeballing a correlation
    heatmap, automatically choose the pair (or top_n set) of numeric
    columns with the lowest mutual correlation -- the columns that carry
    the most distinct, non-redundant information for clustering."""
    numeric_df = df[candidate_columns].select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()

    # sum of absolute correlation with all other candidate columns;
    # lower total means the column is more "independent" of the rest
    redundancy_score = corr_matrix.sum() - 1  # subtract self-correlation of 1
    selected = redundancy_score.sort_values().index[:top_n].tolist()
    return selected


def search_cluster_counts(X_scaled: np.ndarray, k_range: range, random_state: int = 42) -> list:
    """Fit KMeans across a range of cluster counts and record both WCSS
    (for an elbow-style view) and silhouette score (for a quantitative,
    non-eyeballed signal of cluster quality) at each k."""
    results = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)

        wcss = kmeans.inertia_
        # silhouette score is undefined for k=1 (needs at least 2 clusters)
        sil = silhouette_score(X_scaled, labels) if k >= 2 else float("nan")

        results.append(ClusterSearchResult(k=k, wcss=wcss, silhouette=sil))

    return results


def best_k_by_silhouette(search_results: list) -> int:
    """Pick the cluster count with the highest silhouette score, rather
    than reading an elbow chart by eye and hardcoding the chosen value."""
    valid = [r for r in search_results if not np.isnan(r.silhouette)]
    best = max(valid, key=lambda r: r.silhouette)
    return best.k


def fit_final_model(X_scaled: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=random_state)
    kmeans.fit(X_scaled)
    return kmeans


def profile_clusters(df: pd.DataFrame, feature_columns: list, labels: np.ndarray) -> list:
    """Summarize each cluster by its size and mean feature values in the
    original (unscaled) units, so the clusters can actually be
    interpreted as customer segments rather than left as abstract
    centroid coordinates."""
    profiled = df[feature_columns].copy()
    profiled["cluster"] = labels

    profiles = []
    for cluster_id, group in profiled.groupby("cluster"):
        profiles.append(ClusterProfile(
            cluster_id=int(cluster_id),
            size=len(group),
            feature_means=group[feature_columns].mean(),
        ))

    return sorted(profiles, key=lambda p: p.cluster_id)


def run_pipeline(csv_path: str, k_search_range: range = range(2, 11)) -> tuple:
    df = load_and_clean(csv_path)

    candidate_columns = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    feature_columns = select_clustering_features(df, candidate_columns, top_n=2)

    X = df[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    search_results = search_cluster_counts(X_scaled, k_search_range)
    best_k = best_k_by_silhouette(search_results)

    final_model = fit_final_model(X_scaled, best_k)
    labels = final_model.predict(X_scaled)

    profiles = profile_clusters(df, feature_columns, labels)

    return final_model, feature_columns, search_results, best_k, profiles, X, labels


def print_summary(feature_columns: list, search_results: list, best_k: int, profiles: list) -> None:
    print(f"Selected clustering features (lowest mutual correlation): {feature_columns}")
    print()

    print("Cluster count search (silhouette score):")
    for r in search_results:
        marker = "  <-- selected" if r.k == best_k else ""
        sil_str = f"{r.silhouette:.4f}" if not np.isnan(r.silhouette) else "n/a"
        print(f"  k={r.k:2d}  WCSS={r.wcss:10.2f}  silhouette={sil_str}{marker}")
    print()

    print(f"Cluster profiles (k={best_k}):")
    for p in profiles:
        means_str = ", ".join(f"{col}={val:.1f}" for col, val in p.feature_means.items())
        print(f"  Cluster {p.cluster_id}: n={p.size:3d}  {means_str}")


def main():
    parser = argparse.ArgumentParser(description="Customer segmentation via KMeans with automated cluster selection.")
    parser.add_argument("--data", required=True, help="Path to the customers CSV file.")
    args = parser.parse_args()

    _, feature_columns, search_results, best_k, profiles, _, _ = run_pipeline(args.data)
    print_summary(feature_columns, search_results, best_k, profiles)


if __name__ == "__main__":
    main()
