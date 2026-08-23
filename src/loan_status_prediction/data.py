from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from loan_status_prediction.config import DATA_PATH, RANDOM_STATE


TARGET_COLUMN = "loan_status"

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_exp",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
    "credit_score",
]

CATEGORICAL_FEATURES = [
    "person_gender",
    "person_education",
    "person_home_ownership",
    "loan_intent",
    "previous_loan_defaults_on_file",
]

REQUIRED_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
SUSPECT_LEAKAGE_FEATURES = ["previous_loan_defaults_on_file"]
FEATURE_SETS = {
    "full": FEATURE_COLUMNS,
    "no_leakage": [feature for feature in FEATURE_COLUMNS if feature not in SUSPECT_LEAKAGE_FEATURES],
}


def load_loan_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the raw loan dataset and validate the expected schema."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    validate_loan_data(df)
    return df


def validate_loan_data(df: pd.DataFrame) -> None:
    validate_feature_data(df)

    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    invalid_targets = sorted(set(df[TARGET_COLUMN].dropna().unique()) - {0, 1})
    if invalid_targets:
        raise ValueError(f"{TARGET_COLUMN} must be binary 0/1; got {invalid_targets}")


def get_feature_columns(feature_set: str = "full") -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; expected one of {sorted(FEATURE_SETS)}")
    return FEATURE_SETS[feature_set]


def get_numeric_features(feature_set: str = "full") -> list[str]:
    selected_features = set(get_feature_columns(feature_set))
    return [feature for feature in NUMERIC_FEATURES if feature in selected_features]


def get_categorical_features(feature_set: str = "full") -> list[str]:
    selected_features = set(get_feature_columns(feature_set))
    return [feature for feature in CATEGORICAL_FEATURES if feature in selected_features]


def validate_feature_data(df: pd.DataFrame, feature_set: str = "full") -> None:
    missing_columns = sorted(set(get_feature_columns(feature_set)) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required feature columns: {missing_columns}")


def split_features_target(df: pd.DataFrame, feature_set: str = "full") -> tuple[pd.DataFrame, pd.Series]:
    return df[get_feature_columns(feature_set)].copy(), df[TARGET_COLUMN].copy()


def make_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    feature_set: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X, y = split_features_target(df, feature_set)
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def make_train_validation_test_split(
    df: pd.DataFrame,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    feature_set: str = "full",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train_validation, X_test, y_train_validation, y_test = make_train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        feature_set=feature_set,
    )
    validation_fraction = validation_size / (1 - test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_validation,
        y_train_validation,
        test_size=validation_fraction,
        random_state=random_state,
        stratify=y_train_validation,
    )
    return X_train, X_validation, X_test, y_train, y_validation, y_test
