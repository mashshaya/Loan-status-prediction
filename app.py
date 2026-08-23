from __future__ import annotations

import pandas as pd
import streamlit as st

from loan_status_prediction.artifacts import load_model_artifact, load_model_metadata
from loan_status_prediction.config import BEST_MODEL_METADATA_PATH, BEST_MODEL_PATH
from loan_status_prediction.demo_data import load_sample_applications
from loan_status_prediction.predict import predict_rows


st.set_page_config(page_title="Loan Status Prediction", layout="wide")

st.title("Loan Status Prediction")
st.caption("Portfolio demo for credit-risk model inference and explainability.")


@st.cache_resource
def get_model():
    return load_model_artifact(BEST_MODEL_PATH), load_model_metadata(BEST_MODEL_METADATA_PATH)


def build_sidebar_inputs(sample: pd.Series) -> pd.DataFrame:
    st.sidebar.header("Loan application")
    person_age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=int(sample["person_age"]))
    person_income = st.sidebar.number_input("Annual income", min_value=0, value=int(sample["person_income"]), step=1000)
    person_emp_exp = st.sidebar.number_input("Employment experience", min_value=0, max_value=80, value=int(sample["person_emp_exp"]))
    loan_amnt = st.sidebar.number_input("Loan amount", min_value=0, value=int(sample["loan_amnt"]), step=500)
    loan_int_rate = st.sidebar.number_input("Interest rate", min_value=0.0, max_value=100.0, value=float(sample["loan_int_rate"]))
    loan_percent_income = st.sidebar.number_input(
        "Loan percent income",
        min_value=0.0,
        max_value=2.0,
        value=float(sample["loan_percent_income"]),
        step=0.01,
    )
    cb_person_cred_hist_length = st.sidebar.number_input(
        "Credit history length",
        min_value=0,
        max_value=80,
        value=int(sample["cb_person_cred_hist_length"]),
    )
    credit_score = st.sidebar.number_input("Credit score", min_value=300, max_value=900, value=int(sample["credit_score"]))

    return pd.DataFrame(
        [
            {
                "person_age": person_age,
                "person_gender": st.sidebar.selectbox("Gender", ["female", "male"], index=0),
                "person_education": st.sidebar.selectbox(
                    "Education",
                    ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
                    index=2,
                ),
                "person_income": person_income,
                "person_emp_exp": person_emp_exp,
                "person_home_ownership": st.sidebar.selectbox(
                    "Home ownership",
                    ["RENT", "MORTGAGE", "OWN", "OTHER"],
                    index=0,
                ),
                "loan_amnt": loan_amnt,
                "loan_intent": st.sidebar.selectbox(
                    "Loan intent",
                    ["EDUCATION", "MEDICAL", "VENTURE", "PERSONAL", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
                    index=0,
                ),
                "loan_int_rate": loan_int_rate,
                "loan_percent_income": loan_percent_income,
                "cb_person_cred_hist_length": cb_person_cred_hist_length,
                "credit_score": credit_score,
                "previous_loan_defaults_on_file": st.sidebar.selectbox("Previous defaults", ["No", "Yes"], index=0),
            }
        ]
    )


try:
    model, metadata = get_model()
    threshold = float(metadata["threshold"])
    feature_set = str(metadata.get("feature_set", "full"))
    sample_df = load_sample_applications()
    application = build_sidebar_inputs(sample_df.iloc[0])
    prediction = predict_rows(model, application, threshold, feature_set=feature_set)
    probability = float(prediction["loan_status_probability"].iloc[0])
    decision = int(prediction["loan_status_prediction"].iloc[0])

    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted probability", f"{probability:.1%}")
    col2.metric("Decision threshold", f"{threshold:.2f}")
    col3.metric("Predicted class", "Approve / positive" if decision else "Reject / negative")
    st.caption(f"Loaded model feature set: `{feature_set}`")

    st.subheader("Application")
    st.dataframe(application, width="stretch")

    st.subheader("Prediction output")
    st.dataframe(prediction, width="stretch")

    st.info(
        "This demo is for portfolio use only. The repository documents leakage, fairness, "
        "calibration, and validation checks that should be reviewed before any real decision use."
    )
except FileNotFoundError:
    st.warning(
        "Saved model artifacts were not found. Run `make train-save` locally first, "
        "then start the app with `PYTHONPATH=src streamlit run app.py`."
    )
