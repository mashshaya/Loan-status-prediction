from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from loan_status_prediction.config import DATA_PATH, RANDOM_STATE, REPORTS_DIR
from loan_status_prediction.data import load_loan_data, split_features_target
from loan_status_prediction.modeling import candidate_pipelines
from loan_status_prediction.train_models import class_ratio


SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "precision": "precision",
    "recall": "recall",
}


def summarize_cv_scores(scores: dict[str, list[float]], model_name: str) -> dict[str, float | str]:
    summary: dict[str, float | str] = {"model": model_name}
    for key, values in scores.items():
        if not key.startswith("test_"):
            continue
        metric = key.replace("test_", "")
        series = pd.Series(values)
        summary[f"{metric}_mean"] = round(float(series.mean()), 4)
        summary[f"{metric}_std"] = round(float(series.std(ddof=0)), 4)
    return summary


def run_cross_validation(
    data_path: str | Path = DATA_PATH,
    model_names: list[str] | None = None,
    cv_splits: int = 3,
    repeats: int = 2,
    output_path: str | Path = REPORTS_DIR / "cross_validation_report.csv",
) -> dict:
    df = load_loan_data(data_path)
    X, y = split_features_target(df)
    selected_model_names = model_names or ["logistic_regression", "xgboost"]
    pipelines = candidate_pipelines(scale_pos_weight=class_ratio(y))
    cv = RepeatedStratifiedKFold(n_splits=cv_splits, n_repeats=repeats, random_state=RANDOM_STATE)

    summaries = []
    for model_name in selected_model_names:
        scores = cross_validate(
            pipelines[model_name],
            X,
            y,
            scoring=SCORING,
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )
        summaries.append(summarize_cv_scores(scores, model_name))

    report_df = pd.DataFrame(summaries).sort_values("roc_auc_mean", ascending=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output_path, index=False)

    json_path = output_path.with_suffix(".json")
    report = {
        "rows": int(len(df)),
        "cv_splits": cv_splits,
        "repeats": repeats,
        "folds_total": cv_splits * repeats,
        "cross_validation_report_path": str(output_path),
        "models": report_df.to_dict(orient="records"),
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repeated stratified cross-validation.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to loan_data.csv.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "xgboost"],
        choices=["logistic_regression", "random_forest", "xgboost"],
    )
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output-path", default=str(REPORTS_DIR / "cross_validation_report.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_cross_validation(args.data_path, args.models, args.cv_splits, args.repeats, args.output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
