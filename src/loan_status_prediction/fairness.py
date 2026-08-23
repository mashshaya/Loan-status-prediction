from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from loan_status_prediction.artifacts import load_model_artifact, load_model_metadata
from loan_status_prediction.config import BEST_MODEL_METADATA_PATH, BEST_MODEL_PATH, DATA_PATH, REPORTS_DIR, project_relative
from loan_status_prediction.data import load_loan_data, make_train_test_split
from loan_status_prediction.evaluation import predict_with_threshold


GROUP_FEATURES = ["person_gender", "person_education", "person_home_ownership"]


def add_age_band(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["age_band"] = pd.cut(
        result["person_age"],
        bins=[0, 25, 35, 50, np.inf],
        labels=["<=25", "26-35", "36-50", "51+"],
        right=True,
    ).astype(str)
    return result


def group_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    groups: pd.Series,
    group_feature: str,
) -> list[dict[str, float | int | str]]:
    rows = []
    for group_value in sorted(groups.dropna().unique()):
        mask = groups == group_value
        positive_count = int(y_true[mask].sum())
        predicted_positive_count = int(y_pred[mask].sum())
        rows.append(
            {
                "group_feature": group_feature,
                "group_value": str(group_value),
                "rows": int(mask.sum()),
                "positive_count": positive_count,
                "predicted_positive_count": predicted_positive_count,
                "target_rate": round(float(y_true[mask].mean()), 4),
                "approval_rate": round(float(y_pred[mask].mean()), 4),
                "avg_score": round(float(y_proba[mask].mean()), 4),
                "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 4),
                "precision": round(float(precision_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "recall": round(float(recall_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
                "f1": round(float(f1_score(y_true[mask], y_pred[mask], zero_division=0)), 4),
            }
        )
    return rows


def summarize_gaps(metrics: pd.DataFrame) -> list[dict[str, float | str]]:
    gap_rows = []
    for feature, feature_metrics in metrics.groupby("group_feature"):
        for metric in ["approval_rate", "recall", "f1"]:
            gap_rows.append(
                {
                    "group_feature": str(feature),
                    "metric": metric,
                    "min": round(float(feature_metrics[metric].min()), 4),
                    "max": round(float(feature_metrics[metric].max()), 4),
                    "gap": round(float(feature_metrics[metric].max() - feature_metrics[metric].min()), 4),
                }
            )
    return gap_rows


def run_fairness_report(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = BEST_MODEL_PATH,
    metadata_path: str | Path = BEST_MODEL_METADATA_PATH,
    output_path: str | Path = REPORTS_DIR / "fairness_report.csv",
) -> dict:
    df = load_loan_data(data_path)
    model = load_model_artifact(model_path)
    metadata = load_model_metadata(metadata_path)
    threshold = float(metadata["threshold"])
    feature_set = str(metadata.get("feature_set", "full"))
    _, X_test, _, y_test = make_train_test_split(df, feature_set=feature_set)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = predict_with_threshold(y_proba, threshold)
    X_test_with_bands = add_age_band(X_test)

    metric_rows = []
    for feature in [*GROUP_FEATURES, "age_band"]:
        metric_rows.extend(group_metrics(y_test, y_pred, y_proba, X_test_with_bands[feature], feature))

    metrics = pd.DataFrame(metric_rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_path, index=False)

    gap_path = output_path.with_name("fairness_gaps.json")
    gaps = summarize_gaps(metrics)
    gap_path.write_text(json.dumps(gaps, indent=2), encoding="utf-8")

    return {
        "threshold": threshold,
        "feature_set": feature_set,
        "model": metadata.get("model", str(model_path)),
        "fairness_report_path": project_relative(output_path),
        "fairness_gaps_path": project_relative(gap_path),
        "largest_gap": max(gaps, key=lambda row: row["gap"]) if gaps else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create group-level fairness diagnostics.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to loan_data.csv.")
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH), help="Path to a saved .joblib model.")
    parser.add_argument("--metadata-path", default=str(BEST_MODEL_METADATA_PATH), help="Path to metadata JSON.")
    parser.add_argument("--output-path", default=str(REPORTS_DIR / "fairness_report.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_fairness_report(args.data_path, args.model_path, args.metadata_path, args.output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
