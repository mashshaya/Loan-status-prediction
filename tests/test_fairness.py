import numpy as np
import pandas as pd

from loan_status_prediction.fairness import add_age_band, group_metrics, summarize_gaps


def test_add_age_band_creates_expected_buckets():
    df = pd.DataFrame({"person_age": [22, 30, 40, 70]})

    result = add_age_band(df)

    assert result["age_band"].tolist() == ["<=25", "26-35", "36-50", "51+"]


def test_summarize_gaps_calculates_metric_ranges():
    metrics = pd.DataFrame(
        {
            "group_feature": ["person_gender", "person_gender"],
            "approval_rate": [0.2, 0.5],
            "recall": [0.6, 0.9],
            "f1": [0.4, 0.7],
        }
    )

    gaps = summarize_gaps(metrics)

    assert {"group_feature": "person_gender", "metric": "approval_rate", "min": 0.2, "max": 0.5, "gap": 0.3} in gaps


def test_group_metrics_returns_one_row_per_group():
    rows = group_metrics(
        y_true=pd.Series([0, 1, 1, 0]),
        y_pred=np.array([0, 1, 0, 0]),
        y_proba=np.array([0.1, 0.8, 0.4, 0.2]),
        groups=pd.Series(["A", "A", "B", "B"]),
        group_feature="segment",
    )

    assert [row["group_value"] for row in rows] == ["A", "B"]
