# Model Reliability Checks

The project separates three reliability checks from ordinary model training.

## Fairness

Run:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.fairness
```

The command evaluates the saved best model on the test split and writes:

- `reports/fairness_report.csv`
- `reports/fairness_gaps.json`

Tracked groups:

- `person_gender`
- `person_education`
- `person_home_ownership`
- derived `age_band`

## Stricter Validation

Run:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.validation
```

The command runs repeated stratified cross-validation and writes:

- `reports/cross_validation_report.csv`
- `reports/cross_validation_report.json`

This helps avoid overtrusting a single train/test split.

## Calibration

Run:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.calibration
```

The command evaluates probability calibration for the saved best model and
writes:

- `reports/calibration_curve.csv`
- `reports/calibration_curve.json`

Calibration matters because credit scoring decisions often depend on probability
thresholds, not just class labels.

Add `--plot` to also create `reports/calibration_curve.png` when a working
matplotlib backend is available.

## Report Plots

Run:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.report_plots
```

The command writes SVG plots that are easy to view in GitHub:

- `reports/roc_curve.svg`
- `reports/precision_recall_curve.svg`
- `reports/confusion_matrix.svg`
- `reports/threshold_cost_curve.svg`
- `reports/feature_importance.svg`

## SHAP

Run:

```bash
PYTHONPATH=src python3 -m loan_status_prediction.shap_analysis
```

The command writes:

- `reports/xgboost_shap_importance.csv`
- `reports/xgboost_shap_importance.json`
- `reports/xgboost_shap_importance.svg`

SHAP is intentionally a separate command because it is slower and requires the
extra `shap` dependency.
