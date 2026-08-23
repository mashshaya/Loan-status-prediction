from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_status_prediction.config import DATA_PATH, RANDOM_STATE, REPORTS_DIR
from loan_status_prediction.data import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    load_loan_data,
    make_train_test_split,
)


SUSPECT_FEATURES = ["previous_loan_defaults_on_file"]


def target_rate_by_feature(df: pd.DataFrame, feature: str) -> list[dict[str, float | int | str]]:
    summary = (
        df.groupby(feature, dropna=False)[TARGET_COLUMN]
        .agg(rows="count", target_rate="mean")
        .reset_index()
        .sort_values(feature)
    )
    summary["target_rate"] = summary["target_rate"].round(4)
    return summary.to_dict(orient="records")


def build_baseline_for_features(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_logistic_feature_set(
    df: pd.DataFrame,
    excluded_features: list[str] | None = None,
) -> dict[str, float | int | list[str]]:
    excluded_features = excluded_features or []
    numeric_features = [feature for feature in NUMERIC_FEATURES if feature not in excluded_features]
    categorical_features = [feature for feature in CATEGORICAL_FEATURES if feature not in excluded_features]
    selected_features = numeric_features + categorical_features

    X_train, X_test, y_train, y_test = make_train_test_split(df)
    model = build_baseline_for_features(numeric_features, categorical_features)
    model.fit(X_train[selected_features], y_train)
    y_proba = model.predict_proba(X_test[selected_features])[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    return {
        "excluded_features": excluded_features,
        "feature_count": len(selected_features),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
    }


def run_leakage_check(
    data_path: str | Path = DATA_PATH,
    output_path: str | Path = REPORTS_DIR / "leakage_check.json",
) -> dict:
    df = load_loan_data(data_path)
    REPORTS_DIR.mkdir(exist_ok=True)

    report = {
        "rows": int(len(df)),
        "target_rate": round(float(df[TARGET_COLUMN].mean()), 4),
        "suspect_features": {
            feature: target_rate_by_feature(df, feature) for feature in SUSPECT_FEATURES
        },
        "baseline_with_all_features": evaluate_logistic_feature_set(df),
        "baseline_without_suspect_features": evaluate_logistic_feature_set(df, SUSPECT_FEATURES),
        "notes": [
            "A category with a near-perfect target split can indicate leakage or an overly dominant business rule.",
            "Confirm that suspect features are available before the loan decision in the real process.",
        ],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check likely feature leakage in the loan dataset.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to loan_data.csv.")
    parser.add_argument(
        "--output-path",
        default=str(REPORTS_DIR / "leakage_check.json"),
        help="Where to write leakage diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_leakage_check(args.data_path, args.output_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
