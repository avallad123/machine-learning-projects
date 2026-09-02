# Real Estate Data Wrangling & EDA

A data cleaning and exploratory analysis pipeline for the [UCI Real Estate Valuation dataset](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set) (Taiwan housing transactions): missing value handling, statistically-driven outlier removal, unit conversion, feature normalization, and grouped price analysis.

This project was inspired by a general pandas data-wrangling exercise from a data science / machine learning course, but the code and cleaning methodology here are original — built independently rather than reproducing any graded assignment. It uses the same public dataset as a separate [regression project](../housing-price-regression) in this repo, since data wrangling and modeling are naturally different tasks on the same source data.

> **Note:** This is a smaller-scale, self-directed exercise built to practice a defensible data-cleaning workflow rather than to showcase novel algorithms — the cleaning techniques (IQR outlier detection, median imputation, quantile binning) are standard, well-established methods.

## What's different from hardcoded cleaning steps

- **IQR-based outlier removal** instead of hand-picked absolute thresholds (e.g. "drop anything above 80"): bounds are computed per-column from the data's own quartiles (`Q1 - 1.5*IQR` to `Q3 + 1.5*IQR`), so the method adapts to whatever data it's given rather than assuming cutoffs that happen to work for one specific dataset.
- **A corrected unit conversion**: converting price from (10,000 NTD/Ping) to (USD/m²) requires *dividing* by the square-meters-per-Ping conversion factor, not multiplying — since converting a price to a smaller area unit means a smaller price per unit, not a larger one. This implementation gets that direction right, with the reasoning spelled out in code comments.
- **Quantile-based age tiers** instead of fixed cutoffs like "<10 years = New": houses are split into three roughly equal-sized groups (thirds) based on the actual age distribution in the data, which stays meaningful even if applied to a dataset with a very different age range.
- **A test suite** validating each cleaning step independently (imputation, outlier bounds, the unit conversion formula, normalization range, and tier balance) against synthetic data.

## A worthwhile finding from this approach

Applying the IQR method to longitude specifically surfaces a real limitation worth calling out: this dataset's longitude values are bimodal (two geographic clusters of properties, not one smooth distribution), so a statistical outlier rule ends up flagging an entire legitimate second neighborhood as "outliers" rather than just extreme individual points. A fixed, domain-informed threshold can sometimes outperform a general statistical rule precisely because a human already knows something about the data's structure that the statistic doesn't. Both approaches are shown here on purpose — automating a decision doesn't always mean automating it *correctly* for every column, and it's worth checking.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib pytest
```

Download the dataset from the [UCI repository](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set).

## Running

```bash
cd src
python3 real_estate_eda.py --data path/to/real_estate.csv
```

Generate the bar charts:

```bash
python3 plot_eda_results.py --data path/to/real_estate.csv
```

Run the tests:

```bash
cd ..
python3 -m pytest tests/ -v
```

## Concepts demonstrated

- Missing value diagnosis and median imputation
- Statistically-grounded outlier detection with the IQR rule, and its limitations on multimodal data
- Careful, verifiable unit conversion with reasoning documented in code
- Quantile-based binning as an alternative to fixed-threshold categorization
- Feature normalization with `MinMaxScaler`
- Grouped aggregation and analysis with `pandas.groupby`
- Writing tests that validate each data transformation step independently
