import pandas as pd
import pytest

from loan_status_prediction.data import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    get_feature_columns,
    split_features_target,
    validate_loan_data,
)


def test_validate_loan_data_rejects_missing_columns():
    df = pd.DataFrame({TARGET_COLUMN: [0, 1]})

    with pytest.raises(ValueError, match="Missing required feature columns"):
        validate_loan_data(df)


def test_split_features_target_returns_expected_columns():
    df = pd.DataFrame(
        {
            "person_age": [30],
            "person_income": [80000],
            "person_emp_exp": [5],
            "loan_amnt": [12000],
            "loan_int_rate": [10.5],
            "loan_percent_income": [0.15],
            "cb_person_cred_hist_length": [4],
            "credit_score": [700],
            "person_gender": ["female"],
            "person_education": ["Bachelor"],
            "person_home_ownership": ["RENT"],
            "loan_intent": ["EDUCATION"],
            "previous_loan_defaults_on_file": ["No"],
            TARGET_COLUMN: [1],
        }
    )

    X, y = split_features_target(df)

    assert list(X.columns) == FEATURE_COLUMNS
    assert y.tolist() == [1]


def test_no_leakage_feature_set_excludes_previous_defaults():
    assert "previous_loan_defaults_on_file" not in get_feature_columns("no_leakage")
    assert "previous_loan_defaults_on_file" in get_feature_columns("full")
