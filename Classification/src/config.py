"""Configuration loader for Phase 4 ML Classification.

Single loader for all environment variables via python-dotenv.
No other module should call os.getenv() directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root if present
load_dotenv()

# ── Cross-Validation ─────────────────────────────────────────
CV_FOLDS: int = int(os.getenv("CV_FOLDS", "5"))

# ── Threshold Search (Calibration) ───────────────────────────
THRESHOLD_SEARCH_START: float = float(os.getenv("THRESHOLD_SEARCH_START", "0.50"))
THRESHOLD_SEARCH_END: float = float(os.getenv("THRESHOLD_SEARCH_END", "0.99"))
THRESHOLD_SEARCH_STEP: float = float(os.getenv("THRESHOLD_SEARCH_STEP", "0.01"))
FBETA_SCORE_BETA: float = float(os.getenv("FBETA_SCORE_BETA", "0.5"))

# ── Optuna HPO ───────────────────────────────────────────────
OPTUNA_N_TRIALS: int = int(os.getenv("OPTUNA_N_TRIALS", "100"))
_raw_timeout = os.getenv("OPTUNA_TIMEOUT_SECONDS", "3600")
OPTUNA_TIMEOUT_SECONDS: float | None = float(_raw_timeout) if _raw_timeout and float(_raw_timeout) > 0 else None
OPTUNA_SEED: int = int(os.getenv("OPTUNA_SEED", "42"))

# ── XGBoost Training ─────────────────────────────────────────
EARLY_STOPPING_ROUNDS: int = int(os.getenv("EARLY_STOPPING_ROUNDS", "50"))
XGB_MAX_N_ESTIMATORS: int = int(os.getenv("XGB_MAX_N_ESTIMATORS", "5000"))
XGB_TREE_METHOD: str = os.getenv("XGB_TREE_METHOD", "hist")
XGB_DEVICE: str = os.getenv("XGB_DEVICE", "cpu")
TARGET_ENCODER_CV: int = int(os.getenv("TARGET_ENCODER_CV", "5"))
XGB_NEGATIVE_SUBSAMPLE_RATIO: float = float(os.getenv("XGB_NEGATIVE_SUBSAMPLE_RATIO", "1.0"))
XGB_SCALE_POS_WEIGHT: float = float(os.getenv("XGB_SCALE_POS_WEIGHT", "-1"))

# ── Model Family ─────────────────────────────────────────────
MODEL_FAMILY: str = os.getenv("MODEL_FAMILY", "xgboost")

# ── Paths ────────────────────────────────────────────────────
MODEL_OUTPUT_PATH: str = os.getenv("MODEL_OUTPUT_PATH", "models/xgb_model.pkl")
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output/")
LOGS_DIR: str = os.getenv("LOGS_DIR", "logs/")
METRICS_DIR: str = os.getenv("METRICS_DIR", "metrics/phase4/")

def get_config_dict() -> dict[str, object]:
    """Return a dictionary snapshot of current configuration values."""
    return {
        "CV_FOLDS": CV_FOLDS,
        "THRESHOLD_SEARCH_START": THRESHOLD_SEARCH_START,
        "THRESHOLD_SEARCH_END": THRESHOLD_SEARCH_END,
        "THRESHOLD_SEARCH_STEP": THRESHOLD_SEARCH_STEP,
        "FBETA_SCORE_BETA": FBETA_SCORE_BETA,
        "OPTUNA_N_TRIALS": OPTUNA_N_TRIALS,
        "OPTUNA_TIMEOUT_SECONDS": OPTUNA_TIMEOUT_SECONDS,
        "OPTUNA_SEED": OPTUNA_SEED,
        "EARLY_STOPPING_ROUNDS": EARLY_STOPPING_ROUNDS,
        "XGB_MAX_N_ESTIMATORS": XGB_MAX_N_ESTIMATORS,
        "XGB_TREE_METHOD": XGB_TREE_METHOD,
        "XGB_DEVICE": XGB_DEVICE,
        "TARGET_ENCODER_CV": TARGET_ENCODER_CV,
        "XGB_NEGATIVE_SUBSAMPLE_RATIO": XGB_NEGATIVE_SUBSAMPLE_RATIO,
        "XGB_SCALE_POS_WEIGHT": XGB_SCALE_POS_WEIGHT,
        "MODEL_FAMILY": MODEL_FAMILY,
        "MODEL_OUTPUT_PATH": MODEL_OUTPUT_PATH,
        "OUTPUT_DIR": OUTPUT_DIR,
        "LOGS_DIR": LOGS_DIR,
        "METRICS_DIR": METRICS_DIR,
    }
