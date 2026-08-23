from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from loan_status_prediction.artifacts import load_model_artifact, load_model_metadata
from loan_status_prediction.config import BEST_MODEL_METADATA_PATH, BEST_MODEL_PATH, DATA_PATH, REPORTS_DIR
from loan_status_prediction.data import load_loan_data, make_train_test_split


def run_calibration_report(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = BEST_MODEL_PATH,
    metadata_path: str | Path = BEST_MODEL_METADATA_PATH,
    output_path: str | Path = REPORTS_DIR / "calibration_curve.csv",
    n_bins: int = 10,
    save_plot: bool = False,
) -> dict:
    df = load_loan_data(data_path)
    _, X_test, _, y_test = make_train_test_split(df)
    model = load_model_artifact(model_path)
    metadata = load_model_metadata(metadata_path)

    y_proba = np.asarray(model.predict_proba(X_test))[:, 1]
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins, strategy="quantile")
    curve = pd.DataFrame(
        {
            "mean_predicted_probability": prob_pred.round(6),
            "observed_positive_rate": prob_true.round(6),
        }
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(output_path, index=False)

    plot_path = None
    if save_plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_path = output_path.with_suffix(".png")
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
        plt.plot(prob_pred, prob_true, marker="o", label=str(metadata.get("model", "model")))
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Observed positive rate")
        plt.title("Calibration curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    json_path = output_path.with_suffix(".json")
    report = {
        "model": metadata.get("model", str(model_path)),
        "rows": int(len(df)),
        "test_rows": int(len(X_test)),
        "n_bins": n_bins,
        "brier_score": round(float(brier_score_loss(y_test, y_proba)), 4),
        "log_loss": round(float(log_loss(y_test, y_proba)), 4),
        "calibration_curve_path": str(output_path),
        "calibration_plot_path": str(plot_path) if plot_path else None,
    }
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create probability calibration diagnostics.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to loan_data.csv.")
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH), help="Path to a saved .joblib model.")
    parser.add_argument("--metadata-path", default=str(BEST_MODEL_METADATA_PATH), help="Path to metadata JSON.")
    parser.add_argument("--output-path", default=str(REPORTS_DIR / "calibration_curve.csv"))
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--plot", action="store_true", help="Also save a PNG calibration plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_calibration_report(
        data_path=args.data_path,
        model_path=args.model_path,
        metadata_path=args.metadata_path,
        output_path=args.output_path,
        n_bins=args.n_bins,
        save_plot=args.plot,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
