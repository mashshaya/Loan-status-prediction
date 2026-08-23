from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "Data" / "loan_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
BEST_MODEL_METADATA_PATH = MODELS_DIR / "best_model_metadata.json"
RANDOM_STATE = 42

# False positives mean approving a loan that should have been rejected.
# False negatives mean rejecting a loan that could have been approved.
DEFAULT_BUSINESS_COSTS = {
    "false_positive": 5.0,
    "false_negative": 1.0,
}
