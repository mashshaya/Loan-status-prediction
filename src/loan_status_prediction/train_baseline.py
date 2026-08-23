from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from loan_status_prediction.config import DATA_PATH, MODELS_DIR
from loan_status_prediction.data import load_loan_data, make_train_test_split
from loan_status_prediction.preprocessing import build_logistic_baseline_pipeline


def evaluate_baseline(data_path: str | Path = DATA_PATH, feature_set: str = "no_leakage") -> tuple[dict, object]:
    df = load_loan_data(data_path)
    X_train, X_test, y_train, y_test = make_train_test_split(df, feature_set=feature_set)

    model = build_logistic_baseline_pipeline(feature_set)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "rows": int(len(df)),
        "features": int(X_train.shape[1]),
        "feature_set": feature_set,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    return metrics, model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the logistic regression baseline.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to loan_data.csv.")
    parser.add_argument("--save-model", action="store_true", help="Save the fitted model to models/.")
    parser.add_argument("--feature-set", default="no_leakage", choices=["full", "no_leakage"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, model = evaluate_baseline(args.data_path, args.feature_set)
    print(json.dumps(metrics, indent=2))

    if args.save_model:
        MODELS_DIR.mkdir(exist_ok=True)
        model_path = MODELS_DIR / "logistic_baseline.joblib"
        joblib.dump(model, model_path)
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()
