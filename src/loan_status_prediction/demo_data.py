from __future__ import annotations

from pathlib import Path

import pandas as pd

from loan_status_prediction.config import PROJECT_ROOT


SAMPLE_DATA_PATH = PROJECT_ROOT / "examples" / "sample_applications.csv"


def load_sample_applications(path: str | Path = SAMPLE_DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(path)
