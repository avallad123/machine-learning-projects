# Machine Learning Lab

A collection of small, self-directed machine learning exercises covering regression, classification, clustering, and data wrangling on public tabular datasets — with an emphasis on defensible methodology (cross-validation, statistically-grounded cleaning, automated hyperparameter and model selection) over just calling `.fit()`.

These projects were inspired by concepts from data science / machine learning coursework, but each one is an original, independently designed exercise — not a reproduction of any graded assignment. Each project uses a different, well-known public dataset and applies a meaningfully different methodology than a basic fixed-parameter walkthrough.

## Projects

### [`housing-price-regression/`](./housing-price-regression)
Linear regression on the UCI Real Estate Valuation dataset, predicting price per unit area from house age, transit distance, and nearby amenities. Uses k-fold cross-validation for a stable performance estimate and standardized coefficients so feature importance is fairly comparable across differently-scaled inputs.

### [`diabetes-risk-classifier/`](./diabetes-risk-classifier)
A random forest classifier on the Pima Indians Diabetes dataset. Combines automated hyperparameter search (`GridSearchCV`) with stratified cross-validation, detects and corrects implicit missing-value sentinels in the raw data, and ranks features by their trained importance.

### [`diabetes-decision-tree/`](./diabetes-decision-tree)
A single decision tree on the same diabetes dataset, deliberately built around what a tree offers that an ensemble doesn't: cost-complexity pruning to select tree size via cross-validation, and full extraction of the model's decision logic as human-readable rules.

### [`customer-segmentation/`](./customer-segmentation)
KMeans clustering on the Mall Customer Segmentation dataset. Selects clustering features automatically by minimizing mutual correlation, and picks the number of clusters using silhouette score rather than eyeballing an elbow chart — arriving at a different, defensible cluster count as a result.

### [`real-estate-eda/`](./real-estate-eda)
A data cleaning and exploratory analysis pipeline on the same real estate dataset used for regression. Replaces hand-picked outlier thresholds with the IQR statistical rule, corrects a unit-conversion error found in reasoning through the math independently, and documents a real limitation uncovered by the automated approach (a statistical outlier rule misfiring on bimodal geographic data).

## Setup

All projects are standalone Python scripts using standard, well-known libraries (`pandas`, `numpy`, `scikit-learn`, `matplotlib, tensorflow`) and public datasets (linked in each project's own README). Each subfolder includes its own setup, run, and test instructions.
