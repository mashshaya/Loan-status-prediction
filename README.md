# Loan Status Prediction

Credit scoring pet project for predicting whether a loan application is approved.

The current version keeps the original exploratory notebook and adds reusable
Python pipelines for loading data, preprocessing features, comparing models,
choosing decision thresholds, and exporting explainability reports.

## Data

The project expects the dataset at:

```text
Data/loan_data.csv
```

The CSV contains 45,000 rows and 14 columns, including the target column
`loan_status`.

Raw data and trained model binaries are not committed to Git. See
`docs/data_management.md` for the local data workflow.

## Project Structure

```text
.
├── Data/                         # Raw dataset used by the notebook and scripts
├── docs/                         # Data policy and analysis notes
├── examples/                     # Small synthetic inference examples
├── models/                       # Saved model artifacts
├── notebooks/                    # Exploratory analysis notebooks
├── reports/                      # Generated figures, tables, and reports
├── src/loan_status_prediction/   # Reusable project code
├── requirements.txt
└── README.md
```

## Quick Start

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run the logistic regression baseline:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.train_baseline
```

Optionally save the fitted model:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.train_baseline --save-model
```

Compare Logistic Regression, Random Forest, and XGBoost:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.train_models
```

Save the best business-cost model for later inference:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.train_models --save-best-model
```

Generate predictions for a CSV with feature columns:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.predict \
  --input-path Data/loan_data.csv \
  --output-path reports/predictions.csv
```

Run leakage diagnostics:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.leakage
```

Run fairness diagnostics for the saved best model:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.fairness
```

Run repeated stratified cross-validation:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.validation
```

Run probability calibration diagnostics:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.calibration
```

Create SVG model report plots:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.report_plots
```

Create SHAP explainability report for the saved XGBoost model:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.shap_analysis
```

Run tests:

```bash
PYTHONPATH=src python3 -m pytest
```

Or use the Makefile shortcuts:

```bash
make test
make train-save
make reports
make app
```

The comparison writes:

- `reports/model_comparison.csv`
- `reports/model_comparison.json`
- `reports/logistic_regression_explainability.csv`
- `reports/random_forest_explainability.csv`
- `reports/xgboost_explainability.csv`
- `reports/fairness_report.csv`
- `reports/cross_validation_report.csv`
- `reports/calibration_curve.csv`
- `reports/roc_curve.svg`
- `reports/precision_recall_curve.svg`
- `reports/confusion_matrix.svg`
- `reports/threshold_cost_curve.svg`
- `reports/feature_importance.svg`
- `reports/xgboost_shap_importance.csv`
- `models/best_model.joblib`, when `--save-best-model` is used
- `models/best_model_metadata.json`, when `--save-best-model` is used

## Baseline

The first reproducible baseline uses:

- median imputation and standard scaling for numeric features
- most-frequent imputation and one-hot encoding for categorical features
- logistic regression with class balancing
- stratified train/test split
- accuracy, precision, recall, F1, ROC-AUC, and classification report

This baseline is intentionally simple: it gives the project a clean benchmark
before improving Random Forest, XGBoost, threshold tuning, and explainability.

## Model Comparison

The model comparison workflow uses the same train/test split and metric format
for all models. Random Forest and XGBoost are tuned with cross-validation, and
SMOTE is placed inside the `imblearn` pipeline so resampling is fit separately
inside each training fold.

The current business-cost assumption is:

- false positive: `5.0`, approving a loan that should have been rejected
- false negative: `1.0`, rejecting a loan that could have been approved

The latest short tuning run selected XGBoost as the best business-cost model:

| Model | ROC-AUC | Best F1 | Cost threshold | Business cost |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.9753 | 0.8359 | 0.95 | 884 |
| Random Forest | 0.9731 | 0.8276 | 0.82 | 936 |
| Logistic Regression | 0.9562 | 0.7778 | 0.95 | 1458 |

## Production Inference

The training workflow can save the best model according to business cost. The
saved metadata stores the selected model name, threshold, ROC-AUC, F1, and cost
so predictions use the same decision rule that was selected during evaluation.

The prediction command accepts any CSV with the required feature columns. If the
CSV also contains `loan_status`, it is preserved in the output but is not used
for inference.

## Data And Leakage Notes

Raw datasets, trained model binaries, and full prediction exports are ignored by
Git. See `docs/data_management.md` for the recommended workflow.

EDA lives in `notebooks/01_eda.ipynb`. The main leakage concern is
`previous_loan_defaults_on_file`: in the current dataset, one category maps to a
0.0 positive target rate. See `docs/leakage_check.md` and run the leakage
diagnostic command before treating the model as final.

## Validation, Fairness, And Calibration

The project includes post-training checks for model reliability. `fairness`
reports target rate, approval rate, precision, recall, and F1 by gender,
education, home ownership, and age band. `validation` runs repeated stratified
cross-validation instead of relying only on one train/test split. `calibration`
checks whether predicted probabilities match observed positive rates and reports
Brier score and log loss.

See `docs/model_reliability.md` for details.

## Reporting And SHAP

`report_plots` writes lightweight SVG charts without depending on a graphical
matplotlib backend: ROC curve, precision-recall curve, confusion matrix,
threshold-vs-cost curve, and feature-importance bars.

`shap_analysis` computes mean absolute SHAP values for the saved XGBoost model
on a sampled test set. This complements built-in feature importance with a more
model-aware explanation of which transformed features drive predictions.

## Documentation

- `docs/developer_guide.md` explains setup, commands, and the local workflow.
- `docs/data_management.md` explains why raw data and model binaries are ignored.
- `docs/leakage_check.md` documents the main leakage risk.
- `docs/model_reliability.md` documents fairness, validation, calibration, plots, and SHAP.
- `MODEL_CARD.md` summarizes intended use, limitations, ethics, and next steps.

## Demo App

Run the Streamlit demo locally after saving a model:

```bash
make train-save
make app
```

The app is for portfolio demonstration only and requires local model artifacts.
