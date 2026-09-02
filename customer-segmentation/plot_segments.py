import argparse
import os

import matplotlib.pyplot as plt

from customer_segmentation import run_pipeline


def plot_elbow_and_silhouette(search_results: list, best_k: int, output_path: str) -> None:
    ks = [r.k for r in search_results]
    wcss = [r.wcss for r in search_results]
    sils = [r.silhouette for r in search_results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(ks, wcss, marker="o", color="red")
    ax1.set_title("Elbow Method (WCSS)")
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("WCSS")

    ax2.plot(ks, sils, marker="o", color="darkorange")
    ax2.axvline(best_k, color="gray", linestyle="--", label=f"selected k={best_k}")
    ax2.set_title("Silhouette Score by k")
    ax2.set_xlabel("Number of clusters (k)")
    ax2.set_ylabel("Silhouette score")
    ax2.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved elbow/silhouette comparison to {output_path}")


def plot_clusters(model, feature_columns: list, X, labels, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 7))

    n_clusters = len(set(labels))
    cmap = plt.get_cmap("tab10")

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        ax.scatter(X[mask, 0], X[mask, 1], color=cmap(cluster_id), label=f"Cluster {cluster_id}", alpha=0.7)

    # centroids are in scaled space; approximate original-space centroids
    # via the per-cluster feature means instead of inverse-transforming,
    # so this plot doesn't require carrying the scaler around separately
    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        centroid_x = X[mask, 0].mean()
        centroid_y = X[mask, 1].mean()
        ax.scatter(centroid_x, centroid_y, color="black", marker="X", s=150,
                   edgecolor="white", linewidth=1.5, zorder=5)

    ax.set_xlabel(feature_columns[0])
    ax.set_ylabel(feature_columns[1])
    ax.set_title("Customer Segments")
    ax.legend(loc="upper right")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved cluster scatter plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize customer segmentation results.")
    parser.add_argument("--data", required=True, help="Path to the customers CSV file.")
    parser.add_argument("--elbow-output", default="outputs/elbow_silhouette.png")
    parser.add_argument("--cluster-output", default="outputs/customer_clusters.png")
    args = parser.parse_args()

    model, feature_columns, search_results, best_k, _, X, labels = run_pipeline(args.data)

    plot_elbow_and_silhouette(search_results, best_k, args.elbow_output)
    plot_clusters(model, feature_columns, X, labels, args.cluster_output)


if __name__ == "__main__":
    main()
