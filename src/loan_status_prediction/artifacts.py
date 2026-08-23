from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from loan_status_prediction.config import BEST_MODEL_METADATA_PATH, BEST_MODEL_PATH, MODELS_DIR, project_relative


def save_best_model_artifact(model: Any, metadata: dict[str, Any]) -> dict[str, str]:
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, BEST_MODEL_PATH)
    BEST_MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "model_path": project_relative(BEST_MODEL_PATH),
        "metadata_path": project_relative(BEST_MODEL_METADATA_PATH),
    }


def load_model_artifact(model_path: str | Path = BEST_MODEL_PATH) -> Any:
    return joblib.load(model_path)


def load_model_metadata(metadata_path: str | Path = BEST_MODEL_METADATA_PATH) -> dict[str, Any]:
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"Model metadata not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
