"""Smoke test for Phase 4 ML Classification training pipeline."""

import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config
from src.train import run_training


def test_smoke_end_to_end_training(synthetic_train_df: pd.DataFrame, tmp_path: Path):
    """Run full training entrypoint end-to-end at smoke scale."""
    run_id = "smoke_test"

    # Direct outputs to tmp_path
    model_path = tmp_path / "models" / "xgb_model.pkl"
    logs_dir = tmp_path / "logs"
    metrics_dir = tmp_path / "metrics"

    # Temporarily redirect paths
    original_model_path = config.MODEL_OUTPUT_PATH
    original_logs_dir = config.LOGS_DIR
    original_metrics_dir = config.METRICS_DIR

    config.MODEL_OUTPUT_PATH = str(model_path)
    config.LOGS_DIR = str(logs_dir)
    config.METRICS_DIR = str(metrics_dir)

    try:
        # Run with 2 trials and 2 folds for ultra-fast execution (< 5s)
        artifact = run_training(
            df=synthetic_train_df,
            run_id=run_id,
            n_trials=2,
            cv_folds=2,
            timeout_seconds=60,
        )

        # 1. Confirm returned artifact has required keys and types
        assert isinstance(artifact, dict)
        assert "model" in artifact
        assert "threshold" in artifact
        assert "best_params" in artifact
        assert "feature_names" in artifact

        assert isinstance(artifact["model"], Pipeline)
        assert 0.0 <= artifact["threshold"] <= 1.0
        assert len(artifact["feature_names"]) == 26

        # 2. Confirm persisted model on disk
        assert model_path.exists()
        loaded = joblib.load(model_path)
        assert isinstance(loaded["model"], Pipeline)
        assert loaded["threshold"] == artifact["threshold"]

        # 3. Confirm log file is produced and non-empty
        log_file = logs_dir / f"phase4_train_{run_id}.log"
        assert log_file.exists(), f"Log file not found at {log_file}"
        assert log_file.stat().st_size > 0, "Log file is empty!"

        # 4. Confirm run summary JSON is produced and contains all required sections
        summary_file = metrics_dir / f"run_summary_{run_id}.json"
        assert summary_file.exists(), f"Summary JSON not found at {summary_file}"
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        assert "run_id" in summary_data
        assert "config" in summary_data
        assert "dataset" in summary_data
        assert "phase_a_hpo" in summary_data
        assert "phase_b_clean_eval" in summary_data
        assert "phase_c_calibration" in summary_data
        assert "phase_d_final_fit" in summary_data

        assert summary_data["phase_a_hpo"]["total_trials"] == 2
        assert len(summary_data["phase_c_calibration"]["threshold_sweep"]) > 0

    finally:
        config.MODEL_OUTPUT_PATH = original_model_path
        config.LOGS_DIR = original_logs_dir
        config.METRICS_DIR = original_metrics_dir
