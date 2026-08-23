# Model Card: Loan Status Prediction

## Intended Use

This project is a portfolio and learning project for binary loan-status
prediction. It demonstrates data validation, preprocessing, model comparison,
threshold tuning, reporting, explainability, and reliability checks.

It should not be used as a real credit decision system without additional data
governance, legal review, monitoring, and fairness validation.

## Data

The current local dataset contains 45,000 rows and 14 columns. The target is
`loan_status`. Raw data is intentionally excluded from Git.

## Model

The current best model is XGBoost selected by validation business cost. The
production-style path uses the `no_leakage` feature set, excluding
`previous_loan_defaults_on_file`.

- test ROC-AUC: `0.9325`
- test F1 at selected threshold: `0.7247`
- selected threshold: `0.88`
- validation business cost: `1048`
- test business cost: `1032`

Repeated cross-validation produced:

- XGBoost ROC-AUC mean: `0.9348`
- XGBoost F1 mean: `0.7657`

## Key Factors

SHAP analysis highlights the strongest drivers:

- `loan_int_rate`
- `loan_percent_income`
- `person_income`
- `person_home_ownership`
- `loan_intent`

## Limitations

The feature `previous_loan_defaults_on_file` is a major leakage or business-rule
risk. In the current dataset, category `Yes` maps to a `0.0` positive target
rate. It is excluded from the default model, but full-feature experiments remain
useful for understanding the dataset.

Fairness diagnostics show large gaps for `person_home_ownership`, especially
recall. This requires deeper review before any real-world interpretation.

## Ethical Considerations

Credit scoring can materially affect people. Any production version would need:

- clear definition of target and decision policy
- protected-class and proxy-feature analysis
- probability calibration review
- human review and appeal process
- monitoring for drift and disparate impact

## Recommended Next Steps

Use this repository as a portfolio-quality ML project. For real deployment,
collect production-like data, verify feature timing, perform legal/fairness
review, and add ongoing monitoring.
