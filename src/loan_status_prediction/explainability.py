from __future__ import annotations

import numpy as np
import pandas as pd


def get_feature_names(model) -> np.ndarray:
    preprocessor = model.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out()


def logistic_odds_ratios(model, top_n: int = 25) -> pd.DataFrame:
    feature_names = get_feature_names(model)
    classifier = model.named_steps["classifier"]
    coefs = classifier.coef_[0]
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefs,
            "odds_ratio": np.exp(coefs),
            "abs_coefficient": np.abs(coefs),
        }
    )
    return result.sort_values("abs_coefficient", ascending=False).head(top_n)


def tree_feature_importance(model, top_n: int = 25) -> pd.DataFrame:
    feature_names = get_feature_names(model)
    classifier = model.named_steps["classifier"]
    importances = classifier.feature_importances_
    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    return result.sort_values("importance", ascending=False).head(top_n)
