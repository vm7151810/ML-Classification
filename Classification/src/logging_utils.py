"""Logging and run summary utilities for Phase 4 ML Classification.

Implements Decision 13:
- Dual console and run-scoped file logger.
- Comprehensive machine-readable JSON run summary at the end of the run.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def setup_logger(log_dir: str = "logs", run_id: str | None = None) -> tuple[logging.Logger, Path]:
    """Set up console and file logger for the training run.

    Returns:
        tuple[logging.Logger, Path]: Configured logger and path to the log file.
    """
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / f"phase4_train_{run_id}.log"

    logger = logging.getLogger(f"phase4_{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Clear existing handlers if re-initialized
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console Handler
    c_handler = logging.StreamHandler(sys.stdout)
    c_handler.setLevel(logging.INFO)
    c_handler.setFormatter(formatter)
    logger.addHandler(c_handler)

    # File Handler
    f_handler = logging.FileHandler(log_file, encoding="utf-8")
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(formatter)
    logger.addHandler(f_handler)

    return logger, log_file


class RunSummary:
    """Collects and serializes a structured JSON summary of the entire run."""

    def __init__(self, run_id: str | None = None) -> None:
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = datetime.now().isoformat()
        self.config_snapshot: dict[str, Any] = {}
        self.dataset_summary: dict[str, Any] = {}
        self.phase_a_hpo: dict[str, Any] = {}
        self.phase_b_clean_eval: dict[str, Any] = {}
        self.phase_c_calibration: dict[str, Any] = {}
        self.phase_d_final_fit: dict[str, Any] = {}

    def set_config(self, cfg: dict[str, Any]) -> None:
        self.config_snapshot = cfg

    def set_dataset(
        self,
        total_rows: int,
        n_features: int,
        n_pos: int,
        n_neg: int,
        scale_pos_weight: float,
        unique_s1_entities: int,
    ) -> None:
        self.dataset_summary = {
            "total_rows": total_rows,
            "n_features": n_features,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "scale_pos_weight": scale_pos_weight,
            "unique_s1_entities": unique_s1_entities,
        }

    def set_hpo_results(
        self,
        total_trials: int,
        completed_trials: int,
        pruned_trials: int,
        best_trial_number: int,
        best_score: float,
        best_params: dict[str, Any],
        duration_seconds: float,
        trials_detail: list[dict[str, Any]],
    ) -> None:
        self.phase_a_hpo = {
            "total_trials": total_trials,
            "completed_trials": completed_trials,
            "pruned_trials": pruned_trials,
            "best_trial_number": best_trial_number,
            "best_score": best_score,
            "best_params": best_params,
            "duration_seconds": duration_seconds,
            "trials": trials_detail,
        }

    def set_clean_eval_results(
        self,
        n_folds: int,
        fold_scores: list[float],
        oof_macro_fbeta: float,
        hpo_winning_score: float,
        duration_seconds: float,
    ) -> None:
        self.phase_b_clean_eval = {
            "n_folds": n_folds,
            "fold_scores": fold_scores,
            "oof_macro_fbeta": oof_macro_fbeta,
            "hpo_winning_score": hpo_winning_score,
            "score_difference": oof_macro_fbeta - hpo_winning_score,
            "duration_seconds": duration_seconds,
        }

    def set_calibration_results(
        self,
        optimal_threshold: float,
        best_score: float,
        precision_at_optimal: float,
        recall_at_optimal: float,
        threshold_sweep: list[dict[str, float]],
        duration_seconds: float,
    ) -> None:
        self.phase_c_calibration = {
            "optimal_threshold": optimal_threshold,
            "best_score": best_score,
            "precision_at_optimal": precision_at_optimal,
            "recall_at_optimal": recall_at_optimal,
            "threshold_sweep": threshold_sweep,
            "duration_seconds": duration_seconds,
        }

    def set_final_fit_results(
        self,
        artifact_path: str,
        artifact_keys: list[str],
        initial_scale_pos_weight: float,
        final_scale_pos_weight: float,
        weights_matched: bool,
        duration_seconds: float,
    ) -> None:
        self.phase_d_final_fit = {
            "artifact_path": artifact_path,
            "artifact_keys": artifact_keys,
            "initial_scale_pos_weight": initial_scale_pos_weight,
            "final_scale_pos_weight": final_scale_pos_weight,
            "weights_matched": weights_matched,
            "duration_seconds": duration_seconds,
        }

    def save(self, metrics_dir: str = "metrics/phase4") -> Path:
        out_dir = Path(metrics_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"run_summary_{self.run_id}.json"

        data = {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "config": self.config_snapshot,
            "dataset": self.dataset_summary,
            "phase_a_hpo": self.phase_a_hpo,
            "phase_b_clean_eval": self.phase_b_clean_eval,
            "phase_c_calibration": self.phase_c_calibration,
            "phase_d_final_fit": self.phase_d_final_fit,
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return summary_path
