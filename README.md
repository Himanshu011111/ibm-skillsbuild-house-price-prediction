# California House Price Prediction using Machine Learning

**AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026 - BharatCares**
**Author:** Himanshu Namdev - B.Tech CSE (Final Year), Samrat Ashok Rajkiya Engineering College, Mirzapur (AKTU, Lucknow)

## Project Description

An end-to-end machine-learning regression project that predicts the **median house value of a California neighbourhood (census block group)** from its demographics, housing characteristics and location, and explains which factors drive price the most.

The notebook covers the full analytics life-cycle: data loading and quality checks -> exploratory data analysis -> feature engineering (per-household ratios and distances to major cities) -> leak-free preprocessing pipeline -> comparison of 8 regression algorithms with 5-fold cross-validation -> hyper-parameter tuning -> evaluation on a held-out test set -> error analysis -> interpretation (permutation importance, partial dependence) -> saving the model and predicting new data.

**Final model:** `HistGradientBoosting (tuned)` - test R² = **0.866**, MAE = **$27,329**, RMSE = **$41,718** (a 36% lower RMSE than linear regression).

## Dataset

**California Housing Prices** (1990 US Census; Pace & Barry, 1997) - 20,640 rows, 10 columns (9 features + target `median_house_value`).

- Kaggle page: <https://www.kaggle.com/datasets/camnugent/california-housing-prices>
- Direct CSV used by the notebook: <https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv>

| Column | Description |
|---|---|
| `longitude`, `latitude` | Location of the block group |
| `housing_median_age` | Median age of the houses (years) |
| `total_rooms`, `total_bedrooms` | Total rooms / bedrooms in the block group |
| `population`, `households` | Residents / households |
| `median_income` | Median household income (tens of thousands of US$) |
| `ocean_proximity` | Categorical distance to the ocean |
| **`median_house_value`** | **Target** - median house value (US$) |

The notebook looks for `housing.csv` next to itself and **downloads it automatically** if it is missing, so no manual download is needed.

## Technologies Used

Python 3.10+ - pandas - NumPy - scikit-learn - matplotlib - seaborn - SciPy - joblib - Jupyter Notebook

Results in this README were produced with Python 3.13.16, pandas 3.0.5, NumPy 2.5.3, scikit-learn 1.9.1, matplotlib 3.11.2, seaborn 0.13.2.

## Project Structure

```
HimanshuNamdev_HousePricePrediction.ipynb    # complete, executed code (main file)
requirements.txt                             # Python dependencies
HimanshuNamdev_ProjectReport.docx            # full project report
README.md                                    # this file
housing.csv                                  # dataset (auto-downloaded if absent)
figures/                                     # charts saved by the notebook
outputs/                                     # metrics, tables and the trained model
```

## Setup and Run Instructions

```bash
# 1. (optional) create a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows        (Linux/macOS: source venv/bin/activate)

# 2. install dependencies
pip install -r requirements.txt

# 3. launch Jupyter and open the notebook
jupyter notebook HimanshuNamdev_HousePricePrediction.ipynb
```

In Jupyter choose **Kernel -> Restart & Run All**. A full run takes about 10 minutes on a 2-core machine; set `FAST_MODE = True` in the configuration cell for a ~3x faster run with lighter tuning. The notebook also runs unchanged on **Google Colab** (upload the notebook, all libraries are pre-installed).

All charts are saved to `figures/`; metrics, comparison tables and the trained model (`house_price_model.joblib`) are saved to `outputs/`. A fixed random seed (42) makes results reproducible.

## Methodology

1. **Data quality checks** - 207 missing `total_bedrooms` values (median-imputed inside the pipeline), target capped at $500,001 for 965 rows (kept and analysed), no duplicates.
2. **EDA** - target distribution, feature histograms, price map of California, correlation heatmap, income vs price, price by ocean proximity.
3. **Feature engineering** - `rooms_per_household`, `bedrooms_per_room`, `population_per_household`, haversine distance to Los Angeles / San Francisco / San Diego / San Jose / Sacramento and to the nearest of them.
4. **Pipeline** - median imputation + standard scaling for numeric columns, one-hot encoding for `ocean_proximity`; 80/20 split stratified by income category.
5. **Models** - dummy baseline, Linear, Ridge, Lasso, KNN, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoosting (5-fold CV).
6. **Tuning** - randomised search on HistGradientBoosting and Random Forest.
7. **Evaluation** - RMSE, MAE, R², MAPE on the untouched test set; error analysis by region and price band; permutation importance; partial dependence.

## Key Results

**Cross-validation on the training set (US$)**

| Model | CV RMSE | CV MAE | CV R² |
|---|---|---|---|
| HistGradientBoosting (tuned) | 44,074 | 28,222 | 0.855 |
| Random Forest (tuned) | 44,428 | 28,325 | 0.852 |
| HistGradientBoosting | 45,936 | 30,151 | 0.842 |
| Gradient Boosting | 47,596 | 31,548 | 0.831 |
| Random Forest | 47,795 | 30,538 | 0.829 |
| KNN (k=10) | 58,216 | 38,630 | 0.747 |
| Decision Tree | 59,220 | 39,074 | 0.738 |
| Ridge | 66,931 | 47,644 | 0.665 |
| Lasso | 66,934 | 47,643 | 0.665 |
| Linear Regression | 66,935 | 47,643 | 0.665 |
| Dummy (mean baseline) | 115,699 | 91,302 | -0.000 |

**Test set (US$)**

| Model | MAE | RMSE | R² | MAPE |
|---|---|---|---|---|
| Dummy baseline | 90,807 | 114,165 | -0.000 | 63.0% |
| Linear Regression | 47,390 | 64,887 | 0.677 | 27.7% |
| HistGradientBoosting (tuned) | 27,329 | 41,718 | 0.866 | 15.2% |

Selected hyper-parameters: `min_samples_leaf=10`, `max_leaf_nodes=31`, `max_iter=800`, `max_depth=14`, `learning_rate=0.08`, `l2_regularization=10.0`.

**Main findings**

- Non-linear ensembles are far more accurate than linear regression, which in turn clearly beats the mean-value baseline.
- The most important features (permutation importance) are `latitude`, `longitude`, `median_income`, `population_per_household`, `ocean_proximity`: **location and neighbourhood income dominate house prices**.
- The model under-predicts the most expensive neighbourhoods and over-predicts the cheapest ones; part of this is caused by the $500,001 cap in the data.

## Limitations

- The data are from 1990 - absolute price levels are outdated; the method transfers, the numbers do not.
- `median_house_value` is capped at $500,001 and rows are block-group aggregates, not individual houses.
- Random train/test splits can be slightly optimistic because neighbouring areas are similar (spatial autocorrelation).

## Output Files

`outputs/house_price_model.joblib` (trained pipeline), `outputs/metrics.json`, `outputs/model_comparison_cv.csv`, `outputs/test_metrics.csv`, `outputs/permutation_importance.csv`, `outputs/error_by_*.csv`, and 12 PNG charts in `figures/`.

To predict new data load the pipeline with `joblib.load` and call `predict_house_price(raw_df)` defined in section 14 of the notebook (it adds the engineered features and calls the model).

## Reference

Pace, R. K. & Barry, R. (1997). Sparse spatial autoregressions. *Statistics & Probability Letters*, 33(3), 291-297.
