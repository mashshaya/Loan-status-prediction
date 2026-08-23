from pathlib import Path

import pandas as pd

from loan_status_prediction.calibration import run_calibration_report


class FakeModel:
    def predict_proba(self, X):
        return [[0.9, 0.1], [0.7, 0.3], [0.2, 0.8], [0.1, 0.9]]


def test_run_calibration_report_writes_outputs(tmp_path, monkeypatch):
    df = pd.DataFrame(
        {
            "person_age": [30, 45, 29, 52],
            "person_income": [80000, 50000, 90000, 70000],
            "person_emp_exp": [5, 12, 6, 20],
            "loan_amnt": [12000, 20000, 10000, 15000],
            "loan_int_rate": [10.5, 15.2, 9.1, 13.4],
            "loan_percent_income": [0.15, 0.4, 0.11, 0.21],
            "cb_person_cred_hist_length": [4, 10, 5, 14],
            "credit_score": [700, 610, 740, 660],
            "person_gender": ["female", "male", "female", "male"],
            "person_education": ["Bachelor", "High School", "Master", "Associate"],
            "person_home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT"],
            "loan_intent": ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL"],
            "previous_loan_defaults_on_file": ["No", "Yes", "No", "No"],
            "loan_status": [0, 0, 1, 1],
        }
    )
    data_path = tmp_path / "loan_data.csv"
    output_path = tmp_path / "calibration.csv"
    metadata_path = tmp_path / "metadata.json"
    df.to_csv(data_path, index=False)
    metadata_path.write_text('{"model": "fake"}', encoding="utf-8")
    monkeypatch.setattr("loan_status_prediction.calibration.load_model_artifact", lambda _: FakeModel())
    monkeypatch.setattr(
        "loan_status_prediction.calibration.make_train_test_split",
        lambda loaded_df, **_: (None, loaded_df.drop(columns=["loan_status"]), None, loaded_df["loan_status"]),
    )

    report = run_calibration_report(
        data_path=data_path,
        model_path=Path("fake.joblib"),
        metadata_path=metadata_path,
        output_path=output_path,
        n_bins=2,
        save_plot=False,
    )

    assert output_path.exists()
    assert report["brier_score"] == 0.0375
