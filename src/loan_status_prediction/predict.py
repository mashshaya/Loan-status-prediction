from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

from loan_status_prediction.artifacts import load_model_artifact, load_model_metadata
from loan_status_prediction.config import BEST_MODEL_METADATA_PATH, BEST_MODEL_PATH, PROJECT_ROOT
from loan_status_prediction.data import get_feature_columns, validate_feature_data
from loan_status_prediction.evaluation import predict_with_threshold


def predict_rows(model, features: pd.DataFrame, threshold: float, feature_set: str = "full") -> pd.DataFrame:
    validate_feature_data(features, feature_set)
    probabilities = np.asarray(model.predict_proba(features[get_feature_columns(feature_set)]))[:, 1]
    predictions = predict_with_threshold(probabilities, threshold)
    output = features.copy()
    output["loan_status_probability"] = probabilities.round(6)
    output["loan_status_prediction"] = predictions
    return output


def resolve_threshold(cli_threshold: float | None, metadata: dict) -> float:
    if cli_threshold is not None:
        return cli_threshold
    if "threshold" not in metadata:
        raise ValueError("Model metadata must include a decision threshold.")
    return float(metadata["threshold"])


def run_batch_prediction(
    input_path: str | Path,
    output_path: str | Path,
    model_path: str | Path = BEST_MODEL_PATH,
    metadata_path: str | Path = BEST_MODEL_METADATA_PATH,
    threshold: float | None = None,
) -> dict[str, str | float | int]:
    metadata = load_model_metadata(metadata_path)
    resolved_threshold = resolve_threshold(threshold, metadata)
    feature_set = str(metadata.get("feature_set", "full"))
    model = load_model_artifact(model_path)
    input_df = pd.read_csv(input_path)
    output_df = predict_rows(model, input_df, resolved_threshold, feature_set=feature_set)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "model_path": str(model_path),
        "threshold": resolved_threshold,
        "feature_set": feature_set,
        "rows": int(len(output_df)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate loan status predictions for a CSV file.")
    parser.add_argument("--input-path", required=True, help="CSV with feature columns.")
    parser.add_argument(
        "--output-path",
        default=str(PROJECT_ROOT / "reports" / "predictions.csv"),
        help="Where to write predictions.",
    )
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH), help="Path to a saved .joblib model.")
    parser.add_argument(
        "--metadata-path",
        default=str(BEST_MODEL_METADATA_PATH),
        help="Path to model metadata JSON with the default threshold.",
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override the saved decision threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_batch_prediction(
        input_path=args.input_path,
        output_path=args.output_path,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
