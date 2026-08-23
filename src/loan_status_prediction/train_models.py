from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from loan_status_prediction.artifacts import save_best_model_artifact
from loan_status_prediction.config import DATA_PATH, MODELS_DIR, RANDOM_STATE, REPORTS_DIR, project_relative
from loan_status_prediction.data import get_feature_columns, load_loan_data, make_train_validation_test_split
from loan_status_prediction.evaluation import BusinessCosts, classification_metrics, find_best_thresholds, predict_with_threshold
from loan_status_prediction.explainability import logistic_odds_ratios, tree_feature_importance
from loan_status_prediction.modeling import candidate_pipelines, tuning_spaces


def class_ratio(y_train: pd.Series) -> float:
    negative = int((y_train == 0).sum())
    positive = int((y_train == 1).sum())
    return negative / positive


def tune_model(name: str, pipeline, X_train, y_train, n_iter: int, cv_splits: int):
    spaces = tuning_spaces()
    if name not in spaces:
        pipeline.fit(X_train, y_train)
        return pipeline, None

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=spaces[name],
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        n_jobs=1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, {
        "best_score": round(float(search.best_score_), 4),
        "best_params": search.best_params_,
    }


def save_explainability(name: str, model, reports_dir: Path) -> str | None:
    if name == "logistic_regression":
        explanation = logistic_odds_ratios(model)
    elif name in {"random_forest", "xgboost"}:
        explanation = tree_feature_importance(model)
    else:
        return None

    path = reports_dir / f"{name}_explainability.csv"
    explanation.to_csv(path, index=False)
    return project_relative(path)


def run_comparison(
    data_path: str | Path = DATA_PATH,
    n_iter: int = 4,
    cv_splits: int = 3,
    save_models: bool = False,
    save_best_model: bool = False,
    feature_set: str = "no_leakage",
) -> dict:
    df = load_loan_data(data_path)
    X_train, X_validation, X_test, y_train, y_validation, y_test = make_train_validation_test_split(
        df,
        feature_set=feature_set,
    )
    scale_pos_weight = class_ratio(y_train)

    REPORTS_DIR.mkdir(exist_ok=True)
    if save_models:
        MODELS_DIR.mkdir(exist_ok=True)

    costs = BusinessCosts()
    summaries = []
    tuning = {}
    explainability_paths = {}
    fitted_models = {}

    for name, pipeline in candidate_pipelines(scale_pos_weight=scale_pos_weight, feature_set=feature_set).items():
        model, tuning_summary = tune_model(name, pipeline, X_train, y_train, n_iter, cv_splits)
        fitted_models[name] = model
        validation_proba = model.predict_proba(X_validation)[:, 1]
        threshold_summary = find_best_thresholds(y_validation, validation_proba, costs)
        selected_threshold = threshold_summary["best_cost_threshold"]
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = predict_with_threshold(test_proba, selected_threshold)
        test_metrics = classification_metrics(y_test, test_pred, test_proba, selected_threshold, costs)
        summaries.append(
            {
                "model": name,
                "feature_set": feature_set,
                "validation_best_f1_threshold": threshold_summary["best_f1_threshold"],
                "validation_best_f1": threshold_summary["best_f1"],
                "validation_best_cost_threshold": threshold_summary["best_cost_threshold"],
                "validation_best_business_cost": threshold_summary["best_business_cost"],
                **{f"test_{key}": value for key, value in test_metrics.items()},
                "best_cost_threshold": selected_threshold,
                "best_cost_business_cost": test_metrics["business_cost"],
                "best_f1": test_metrics["f1"],
                "default_roc_auc": test_metrics["roc_auc"],
            }
        )
        if tuning_summary is not None:
            tuning[name] = tuning_summary

        explainability_path = save_explainability(name, model, REPORTS_DIR)
        if explainability_path is not None:
            explainability_paths[name] = explainability_path

        if save_models:
            joblib.dump(model, MODELS_DIR / f"{name}.joblib")

    comparison = pd.DataFrame(summaries).sort_values("validation_best_business_cost")
    comparison_path = REPORTS_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    report = {
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_validation)),
        "test_rows": int(len(X_test)),
        "feature_set": feature_set,
        "feature_columns": get_feature_columns(feature_set),
        "business_costs": {
            "false_positive": costs.false_positive,
            "false_negative": costs.false_negative,
        },
        "model_comparison_path": project_relative(comparison_path),
        "explainability_paths": explainability_paths,
        "tuning": tuning,
        "models": comparison.to_dict(orient="records"),
    }

    if save_best_model:
        best_row = comparison.iloc[0].to_dict()
        artifact_metadata = {
            "model": best_row["model"],
            "selection_metric": "validation_best_business_cost",
            "threshold": best_row["best_cost_threshold"],
            "validation_business_cost": best_row["validation_best_business_cost"],
            "test_business_cost": best_row["best_cost_business_cost"],
            "roc_auc": best_row["default_roc_auc"],
            "f1": best_row["best_f1"],
            "feature_set": feature_set,
            "feature_columns": get_feature_columns(feature_set),
            "data_path": project_relative(data_path),
            "rows": int(len(df)),
        }
        report["best_model_artifact"] = save_best_model_artifact(
            fitted_models[best_row["model"]],
            artifact_metadata,
        )

    json_path = REPORTS_DIR / "model_comparison.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare loan status prediction models.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to loan_data.csv.")
    parser.add_argument("--n-iter", type=int, default=4, help="Random search iterations for tuned models.")
    parser.add_argument("--cv-splits", type=int, default=3, help="CV folds for tuned models.")
    parser.add_argument("--save-models", action="store_true", help="Save fitted models to models/.")
    parser.add_argument("--feature-set", default="no_leakage", choices=["full", "no_leakage"])
    parser.add_argument(
        "--save-best-model",
        action="store_true",
        help="Save the best business-cost model as models/best_model.joblib.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_comparison(
        data_path=args.data_path,
        n_iter=args.n_iter,
        cv_splits=args.cv_splits,
        save_models=args.save_models,
        save_best_model=args.save_best_model,
        feature_set=args.feature_set,
    )
    compact = {
        "model_comparison_path": report["model_comparison_path"],
        "business_costs": report["business_costs"],
        "models": [
            {
                "model": row["model"],
                "feature_set": row["feature_set"],
                "best_cost_threshold": row["best_cost_threshold"],
                "best_cost_business_cost": row["best_cost_business_cost"],
                "best_f1": row["best_f1"],
                "default_roc_auc": row["default_roc_auc"],
            }
            for row in report["models"]
        ],
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
