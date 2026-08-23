from __future__ import annotations

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from loan_status_prediction.config import RANDOM_STATE
from loan_status_prediction.preprocessing import build_preprocessor


def build_logistic_pipeline(feature_set: str = "full") -> ImbPipeline:
    return ImbPipeline(
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


def build_random_forest_pipeline(feature_set: str = "full") -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_set)),
            ("sampler", SMOTE(random_state=RANDOM_STATE)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    class_weight="balanced_subsample",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                ),
            ),
        ]
    )


def build_xgboost_pipeline(scale_pos_weight: float = 1.0, feature_set: str = "full") -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_set)),
            (
                "classifier",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=RANDOM_STATE,
                    n_jobs=1,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )


def candidate_pipelines(scale_pos_weight: float = 1.0, feature_set: str = "full") -> dict[str, ImbPipeline]:
    return {
        "logistic_regression": build_logistic_pipeline(feature_set),
        "random_forest": build_random_forest_pipeline(feature_set),
        "xgboost": build_xgboost_pipeline(scale_pos_weight=scale_pos_weight, feature_set=feature_set),
    }


def tuning_spaces() -> dict[str, dict[str, list]]:
    return {
        "random_forest": {
            "classifier__n_estimators": [150, 250],
            "classifier__max_depth": [8, 14, None],
            "classifier__min_samples_leaf": [1, 3],
        },
        "xgboost": {
            "classifier__n_estimators": [150, 250],
            "classifier__max_depth": [3, 5],
            "classifier__learning_rate": [0.03, 0.08],
            "classifier__subsample": [0.8, 1.0],
            "classifier__colsample_bytree": [0.8, 1.0],
        },
    }
