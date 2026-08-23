from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from scipy import sparse

from loan_status_prediction.artifacts import load_model_artifact, load_model_metadata
from loan_status_prediction.config import BEST_MODEL_METADATA_PATH, BEST_MODEL_PATH, DATA_PATH, RANDOM_STATE, REPORTS_DIR
from loan_status_prediction.data import load_loan_data, make_train_test_split
from loan_status_prediction.explainability import get_feature_names
from loan_status_prediction.svg_reports import bar_chart_svg


def transformed_sample(model, X: pd.DataFrame, sample_size: int) -> np.ndarray:
    sample = X.sample(n=min(sample_size, len(X)), random_state=RANDOM_STATE)
    transformed = model.named_steps["preprocessor"].transform(sample)
    if sparse.issparse(transformed):
        return transformed.toarray()
    return np.asarray(transformed)


def compute_shap_importance(model, X: pd.DataFrame, sample_size: int = 500, top_n: int = 25) -> pd.DataFrame:
    try:
        import shap
    except ImportError as exc:
        raise ImportError("Install shap to run this command: python3 -m pip install shap") from exc

    X_transformed = transformed_sample(model, X, sample_size)
    classifier = model.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]

    feature_names = get_feature_names(model)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }
    )
    return result.sort_values("mean_abs_shap", ascending=False).head(top_n)


def run_shap_report(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = BEST_MODEL_PATH,
    metadata_path: str | Path = BEST_MODEL_METADATA_PATH,
    output_path: str | Path = REPORTS_DIR / "xgboost_shap_importance.csv",
    sample_size: int = 500,
    top_n: int = 25,
) -> dict:
    df = load_loan_data(data_path)
    _, X_test, _, _ = make_train_test_split(df)
    model = load_model_artifact(model_path)
    metadata = load_model_metadata(metadata_path)

    if metadata.get("model") and metadata["model"] != "xgboost":
        raise ValueError(f"SHAP report expects an XGBoost model, got {metadata['model']!r}.")

    importance = compute_shap_importance(model, X_test, sample_size=sample_size, top_n=top_n)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_path, index=False)
    svg_path = bar_chart_svg(
        importance.head(15),
        "feature",
        "mean_abs_shap",
        "Top XGBoost SHAP Importances",
        output_path.with_suffix(".svg"),
    )

    report = {
        "model": metadata.get("model", str(model_path)),
        "sample_size": min(sample_size, len(X_test)),
        "top_n": top_n,
        "shap_importance_path": str(output_path),
        "shap_importance_svg": svg_path,
        "top_features": importance.head(10).to_dict(orient="records"),
    }
    json_path = output_path.with_suffix(".json")
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SHAP explainability report for the saved XGBoost model.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to loan_data.csv.")
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH), help="Path to a saved .joblib model.")
    parser.add_argument("--metadata-path", default=str(BEST_MODEL_METADATA_PATH), help="Path to metadata JSON.")
    parser.add_argument("--output-path", default=str(REPORTS_DIR / "xgboost_shap_importance.csv"))
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--top-n", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_shap_report(
        data_path=args.data_path,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        sample_size=args.sample_size,
        top_n=args.top_n,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
