from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from loan_status_prediction.config import DEFAULT_BUSINESS_COSTS


@dataclass(frozen=True)
class BusinessCosts:
    false_positive: float = DEFAULT_BUSINESS_COSTS["false_positive"]
    false_negative: float = DEFAULT_BUSINESS_COSTS["false_negative"]


def predict_with_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    return (y_proba >= threshold).astype(int)


def business_cost(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    costs: BusinessCosts = BusinessCosts(),
) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(fp * costs.false_positive + fn * costs.false_negative)


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    costs: BusinessCosts = BusinessCosts(),
) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "business_cost": round(business_cost(y_true, y_pred, costs), 4),
    }


def find_best_thresholds(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    costs: BusinessCosts = BusinessCosts(),
    thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.01)

    scored_thresholds = []
    for threshold in thresholds:
        y_pred = predict_with_threshold(y_proba, threshold)
        scored_thresholds.append(
            {
                "threshold": float(threshold),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "business_cost": business_cost(y_true, y_pred, costs),
            }
        )

    best_f1 = max(scored_thresholds, key=lambda row: row["f1"])
    best_cost = min(scored_thresholds, key=lambda row: row["business_cost"])
    return {
        "best_f1_threshold": round(best_f1["threshold"], 4),
        "best_f1": round(best_f1["f1"], 4),
        "best_cost_threshold": round(best_cost["threshold"], 4),
        "best_business_cost": round(best_cost["business_cost"], 4),
    }


def evaluate_model(
    model_name: str,
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    costs: BusinessCosts = BusinessCosts(),
) -> dict[str, float | str]:
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold_summary = find_best_thresholds(y_test, y_proba, costs)

    default_pred = predict_with_threshold(y_proba, 0.5)
    f1_pred = predict_with_threshold(y_proba, threshold_summary["best_f1_threshold"])
    cost_pred = predict_with_threshold(y_proba, threshold_summary["best_cost_threshold"])

    result = {
        "model": model_name,
        **{f"default_{k}": v for k, v in classification_metrics(y_test, default_pred, y_proba, 0.5, costs).items()},
        **{f"best_f1_{k}": v for k, v in classification_metrics(y_test, f1_pred, y_proba, threshold_summary["best_f1_threshold"], costs).items()},
        **{f"best_cost_{k}": v for k, v in classification_metrics(y_test, cost_pred, y_proba, threshold_summary["best_cost_threshold"], costs).items()},
    }
    result.update(threshold_summary)
    result["best_f1"] = result["best_f1_f1"]
    result["best_business_cost"] = result["best_cost_business_cost"]
    return result
