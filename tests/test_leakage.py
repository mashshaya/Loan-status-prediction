import pandas as pd

from loan_status_prediction.leakage import target_rate_by_feature


def test_target_rate_by_feature_returns_group_counts_and_rates():
    df = pd.DataFrame(
        {
            "previous_loan_defaults_on_file": ["No", "No", "Yes"],
            "loan_status": [1, 0, 0],
        }
    )

    result = target_rate_by_feature(df, "previous_loan_defaults_on_file")

    assert result == [
        {"previous_loan_defaults_on_file": "No", "rows": 2, "target_rate": 0.5},
        {"previous_loan_defaults_on_file": "Yes", "rows": 1, "target_rate": 0.0},
    ]
