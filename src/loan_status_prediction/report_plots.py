from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from loan_status_prediction.artifacts import load_model_artifact, load_model_metadata
from loan_status_prediction.config import BEST_MODEL_METADATA_PATH, BEST_MODEL_PATH, DATA_PATH, REPORTS_DIR, project_relative
from loan_status_prediction.data import load_loan_data, make_train_test_split
from loan_status_prediction.evaluation import BusinessCosts, business_cost, predict_with_threshold
from loan_status_prediction.explainability import tree_feature_importance
from loan_status_prediction.svg_reports import bar_chart_svg, confusion_matrix_svg, line_chart_svg


def threshold_cost_curve(y_true, y_proba, costs: BusinessCosts) -> pd.DataFrame:
    rows = []
    for threshold in [round(x / 100, 2) for x in range(5, 96)]:
        y_pred = predict_with_threshold(y_proba, threshold)
        rows.append(
            {
                "threshold": threshold,
                "business_cost": business_cost(y_true, y_pred, costs),
            }
        )
    return pd.DataFrame(rows)


def run_report_plots(
    data_path: str | Path = DATA_PATH,
    model_path: str | Path = BEST_MODEL_PATH,
    metadata_path: str | Path = BEST_MODEL_METADATA_PATH,
    output_dir: str | Path = REPORTS_DIR,
) -> dict:
    df = load_loan_data(data_path)
    model = load_model_artifact(model_path)
    metadata = load_model_metadata(metadata_path)
    threshold = float(metadata["threshold"])
    feature_set = str(metadata.get("feature_set", "full"))
    _, X_test, _, y_test = make_train_test_split(df, feature_set=feature_set)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = predict_with_threshold(y_proba, threshold)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr})
    roc_csv = output_dir / "roc_curve.csv"
    roc_data.to_csv(roc_csv, index=False)
    roc_svg = line_chart_svg(
        roc_data,
        "false_positive_rate",
        "true_positive_rate",
        "ROC Curve",
        output_dir / "roc_curve.svg",
        "False positive rate",
        "True positive rate",
        baseline_diagonal=True,
    )

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_data = pd.DataFrame({"recall": recall, "precision": precision})
    pr_csv = output_dir / "precision_recall_curve.csv"
    pr_data.to_csv(pr_csv, index=False)
    pr_svg = line_chart_svg(
        pr_data,
        "recall",
        "precision",
        "Precision-Recall Curve",
        output_dir / "precision_recall_curve.svg",
        "Recall",
        "Precision",
    )

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist()
    cm_svg = confusion_matrix_svg(matrix, f"Confusion Matrix at threshold {threshold}", output_dir / "confusion_matrix.svg")

    costs = BusinessCosts()
    threshold_data = threshold_cost_curve(y_test, y_proba, costs)
    threshold_csv = output_dir / "threshold_cost_curve.csv"
    threshold_data.to_csv(threshold_csv, index=False)
    threshold_svg = line_chart_svg(
        threshold_data,
        "threshold",
        "business_cost",
        "Threshold vs Business Cost",
        output_dir / "threshold_cost_curve.svg",
        "Threshold",
        "Business cost",
    )

    importance = tree_feature_importance(model, top_n=15)
    importance_svg = bar_chart_svg(
        importance,
        "feature",
        "importance",
        "Top XGBoost Feature Importances",
        output_dir / "feature_importance.svg",
    )

    report = {
        "model": metadata.get("model", str(model_path)),
        "feature_set": feature_set,
        "threshold": threshold,
        "roc_curve_csv": project_relative(roc_csv),
        "roc_curve_svg": project_relative(roc_svg),
        "precision_recall_curve_csv": project_relative(pr_csv),
        "precision_recall_curve_svg": project_relative(pr_svg),
        "confusion_matrix_svg": project_relative(cm_svg),
        "threshold_cost_curve_csv": project_relative(threshold_csv),
        "threshold_cost_curve_svg": project_relative(threshold_svg),
        "feature_importance_svg": project_relative(importance_svg),
    }
    json_path = output_dir / "plot_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create SVG report plots for the saved best model.")
    parser.add_argument("--data-path", default=str(DATA_PATH), help="Path to loan_data.csv.")
    parser.add_argument("--model-path", default=str(BEST_MODEL_PATH), help="Path to a saved .joblib model.")
    parser.add_argument("--metadata-path", default=str(BEST_MODEL_METADATA_PATH), help="Path to metadata JSON.")
    parser.add_argument("--output-dir", default=str(REPORTS_DIR), help="Where to write plot reports.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_report_plots(args.data_path, args.model_path, args.metadata_path, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
