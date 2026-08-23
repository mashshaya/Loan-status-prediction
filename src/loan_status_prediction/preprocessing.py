from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_status_prediction.config import RANDOM_STATE
from loan_status_prediction.data import get_categorical_features, get_numeric_features


def build_preprocessor(feature_set: str = "full") -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, get_numeric_features(feature_set)),
            ("categorical", categorical_pipeline, get_categorical_features(feature_set)),
        ]
    )


def build_logistic_baseline_pipeline(feature_set: str = "full") -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_set)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
