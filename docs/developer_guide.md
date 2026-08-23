# Developer Guide

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Common Commands

```bash
make test
make train-save
make reports
make predict
```

## Data

The raw dataset is expected at `Data/loan_data.csv`, but CSV data is ignored by
Git. See `docs/data_management.md`.

## Model Workflow

1. Run `make train-save` to compare models and save the best business-cost model.
2. Run `make reports` to refresh leakage, fairness, validation, calibration,
   SVG plot, and SHAP reports.
3. Review `docs/leakage_check.md`, `docs/model_reliability.md`, and
   `MODEL_CARD.md` before publishing conclusions.

## Streamlit Demo

```bash
PYTHONPATH=src streamlit run app.py
```

The app requires local model artifacts in `models/`. These artifacts are ignored
by Git and should be regenerated locally.
