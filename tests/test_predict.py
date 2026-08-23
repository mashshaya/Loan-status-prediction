import pandas as pd
import pytest

from loan_status_prediction.predict import predict_rows, resolve_threshold


class FakeModel:
    def predict_proba(self, X):
        return [[0.8, 0.2], [0.3, 0.7]]


def sample_features():
    return pd.DataFrame(
        {
            "person_age": [30, 45],
            "person_income": [80000, 50000],
            "person_emp_exp": [5, 12],
            "loan_amnt": [12000, 20000],
            "loan_int_rate": [10.5, 15.2],
            "loan_percent_income": [0.15, 0.4],
            "cb_person_cred_hist_length": [4, 10],
            "credit_score": [700, 610],
            "person_gender": ["female", "male"],
            "person_education": ["Bachelor", "High School"],
            "person_home_ownership": ["RENT", "MORTGAGE"],
            "loan_intent": ["EDUCATION", "MEDICAL"],
            "previous_loan_defaults_on_file": ["No", "Yes"],
        }
    )


def test_resolve_threshold_prefers_cli_value():
    assert resolve_threshold(0.4, {"threshold": 0.8}) == 0.4


def test_resolve_threshold_requires_metadata_threshold():
    with pytest.raises(ValueError, match="decision threshold"):
        resolve_threshold(None, {})


def test_predict_rows_adds_probability_and_prediction_columns():
    output = predict_rows(FakeModel(), sample_features(), threshold=0.5)

    assert output["loan_status_probability"].tolist() == [0.2, 0.7]
    assert output["loan_status_prediction"].tolist() == [0, 1]


def test_predict_rows_supports_no_leakage_feature_set():
    output = predict_rows(FakeModel(), sample_features(), threshold=0.5, feature_set="no_leakage")

    assert output["loan_status_prediction"].tolist() == [0, 1]
