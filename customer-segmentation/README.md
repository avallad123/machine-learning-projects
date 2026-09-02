# Customer Segmentation

A KMeans clustering exploration of the [Mall Customer Segmentation dataset](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) — grouping customers by annual income and spending behavior, with automated feature and cluster-count selection.

This project was inspired by a general clustering exercise from a data science / machine learning course, but the code, feature-selection methodology, and cluster-count selection here are original — built independently rather than reproducing any graded assignment.

> **Note:** This is a smaller-scale, self-directed exercise built to practice a full, defensible clustering workflow (automated feature and k selection, cluster interpretation) rather than a from-scratch algorithm implementation — the underlying `scikit-learn` estimator is standard.

## What's different from eyeballing a heatmap and elbow chart

- **Automated feature selection**: instead of visually inspecting a correlation heatmap and manually picking two columns, the least mutually-correlated pair of numeric columns is selected programmatically (by minimizing each column's total absolute correlation with the others) — the same underlying idea (independent features cluster better), but made repeatable and free of manual judgment calls.
- **Silhouette-score-based cluster count selection**: rather than reading a WCSS "elbow" chart by eye and hardcoding a chosen cluster count, every candidate k is also scored with the silhouette coefficient (a quantitative measure of how well-separated and internally cohesive the resulting clusters are), and the k with the best score is selected automatically. On this dataset that means k=5, not the commonly eyeballed k=6 — a genuinely different, defensible outcome, not just a different way of arriving at the same number.
- **Feature scaling**: features are standardized before clustering, since KMeans is distance-based and features on different scales (e.g. income in thousands vs. a 1–100 score) would otherwise distort distance calculations.
- **Cluster profiling**: each resulting cluster is summarized by its size and mean feature values in original (unscaled) units, turning abstract centroids into interpretable customer segments (e.g. "high income, low spending").
- **A test suite** using synthetic customer data with a known ground-truth cluster count and a deliberately weakly-correlated distractor feature, checking that both the feature-selection and cluster-count-selection logic recover the right answer.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib pytest
```

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

## Running

```bash
cd src
python3 customer_segmentation.py --data path/to/customers.csv
```

Example output:
```
Selected clustering features (lowest mutual correlation): ['Annual Income (k$)', 'Spending Score (1-100)']

Cluster count search (silhouette score):
  k= 2  WCSS=    269.69  silhouette=0.3213
  k= 3  WCSS=    157.70  silhouette=0.4666
  k= 4  WCSS=    108.92  silhouette=0.4939
  k= 5  WCSS=     65.57  silhouette=0.5547  <-- selected
  k= 6  WCSS=     55.06  silhouette=0.5399
  ...

Cluster profiles (k=5):
  Cluster 0: n= 81  Annual Income (k$)=55.3, Spending Score (1-100)=49.5
  Cluster 1: n= 39  Annual Income (k$)=86.5, Spending Score (1-100)=82.1
  Cluster 2: n= 22  Annual Income (k$)=25.7, Spending Score (1-100)=79.4
  Cluster 3: n= 35  Annual Income (k$)=88.2, Spending Score (1-100)=17.1
  Cluster 4: n= 23  Annual Income (k$)=26.3, Spending Score (1-100)=20.9
```

Generate the elbow/silhouette comparison and cluster scatter plots:

```bash
python3 plot_segments.py --data path/to/customers.csv
```

Run the tests:

```bash
cd ..
python3 -m pytest tests/ -v
```

## Interpreting the results

The five resulting segments read as recognizable customer archetypes: mid-income/mid-spending (the largest group), high-income/high-spending, low-income/high-spending, high-income/low-spending, and low-income/low-spending. The high-income/low-spending and low-income/high-spending groups are the most actionable from a business standpoint — the first represents an under-tapped opportunity, the second a price-sensitive but engaged segment.

## Concepts demonstrated

- Automated, correlation-based feature selection as an alternative to visually inspecting a heatmap
- Feature scaling with `StandardScaler` prior to distance-based clustering
- Using silhouette score, not just WCSS/elbow inspection, to choose a cluster count objectively
- Translating cluster centroids into interpretable, business-readable segment profiles
- Writing tests against synthetic data with a known ground-truth cluster count and a designed distractor feature
