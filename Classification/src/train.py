"""Main training module for Phase 4 ML Classification.

Orchestrates the 4 training phases:
- Phase A: Hyperparameter optimization using Optuna + GroupKFold.
- Phase B: Clean cross-validated out-of-fold evaluation with fixed best parameters.
- Phase C: Threshold calibration sweeping macro F-beta over clean OOF probabilities.
- Phase D: Final fit on 100% labeled data and model serialization.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna_integration import XGBoostPruningCallback
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config
from src.logging_utils import RunSummary, setup_logger
from src.metrics import compute_scale_pos_weight, entity_macro_fbeta
from src.model import (
    COUNTRY_COL,
    FEATURE_COLS,
    NUMERICAL_COLS,
    build_pipeline,
    manual_encode_for_eval,
    standardize_input_dataframe,
)

# Silence noisy Optuna logs below WARNING for root logger
optuna.logging.set_verbosity(optuna.logging.WARNING)

import warnings
warnings.filterwarnings("ignore", message=".*The reported value is ignored because this.*")


def _extract_config_val(config_obj: Any, key: str, default: Any) -> Any:
    """Helper to safely extract a config attribute or key, falling back to default."""
    if config_obj is None:
        return default
    if isinstance(config_obj, dict):
        return config_obj.get(key, default)
    if hasattr(config_obj, key):
        return getattr(config_obj, key)
    return default


def run_training(
    df: Any,
    model_output_path: str | Path | None = None,
    config_obj: Any = None,
    run_id: str | None = None,
    n_trials: int | None = None,
    cv_folds: int | None = None,
    timeout_seconds: float | None = None,
    negative_subsample_ratio: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Execute the end-to-end 4-phase training pipeline.

    Supports both Phase 4 standalone invocation and the upstream Orchestrator convention:
        run_training(labeled_df, model_out, config)

    Args:
        df: Input DataFrame (Polars/Pandas or path) containing grouping key, candidate ID, 26 features, and label.
        model_output_path: Optional destination filepath for the fitted pipeline artifact.
        config_obj: Optional configuration object or dictionary.
        run_id: Optional unique identifier for this run.
        n_trials: Override for Optuna trial count.
        cv_folds: Override for cross-validation fold count.
        timeout_seconds: Override for Optuna wall-clock timeout in seconds.
        negative_subsample_ratio: Optional fraction in (0.0, 1.0] of negative rows to retain.
        **kwargs: Extra parameters for forward compatibility.

    Returns:
        dict: Loaded artifact dictionary containing model, threshold, best_params, feature_names.
    """
    # Guard against positional ambiguity if run_id was passed as 2nd positional arg
    if (
        isinstance(model_output_path, str)
        and not (model_output_path.endswith((".pkl", ".joblib", ".bin")) or "/" in model_output_path or "\\" in model_output_path)
        and run_id is None
        and config_obj is None
    ):
        run_id = model_output_path
        model_output_path = None

    # Resolve paths and directories
    target_model_path = model_output_path or _extract_config_val(config_obj, "MODEL_OUTPUT_PATH", config.MODEL_OUTPUT_PATH)
    actual_logs_dir = _extract_config_val(config_obj, "LOGS_DIR", config.LOGS_DIR)
    actual_metrics_dir = _extract_config_val(config_obj, "METRICS_DIR", config.METRICS_DIR)

    logger, log_file = setup_logger(log_dir=actual_logs_dir, run_id=run_id)
    summary = RunSummary(run_id=run_id)

    # Resolve overrides vs config
    actual_trials = n_trials if n_trials is not None else _extract_config_val(config_obj, "OPTUNA_N_TRIALS", config.OPTUNA_N_TRIALS)
    actual_folds = cv_folds if cv_folds is not None else _extract_config_val(config_obj, "CV_FOLDS", config.CV_FOLDS)
    actual_timeout = timeout_seconds if timeout_seconds is not None else _extract_config_val(config_obj, "OPTUNA_TIMEOUT_SECONDS", config.OPTUNA_TIMEOUT_SECONDS)
    subsample_ratio = negative_subsample_ratio if negative_subsample_ratio is not None else _extract_config_val(config_obj, "XGB_NEGATIVE_SUBSAMPLE_RATIO", config.XGB_NEGATIVE_SUBSAMPLE_RATIO)
    actual_fbeta = float(_extract_config_val(config_obj, "FBETA_SCORE_BETA", config.FBETA_SCORE_BETA))
    actual_model_family = str(_extract_config_val(config_obj, "MODEL_FAMILY", config.MODEL_FAMILY))
    actual_target_encoder_cv = int(_extract_config_val(config_obj, "TARGET_ENCODER_CV", config.TARGET_ENCODER_CV))
    xgb_device = str(_extract_config_val(config_obj, "XGB_DEVICE", config.XGB_DEVICE))
    xgb_tree_method = str(_extract_config_val(config_obj, "XGB_TREE_METHOD", config.XGB_TREE_METHOD))

    cfg_snapshot = config.get_config_dict()
    cfg_snapshot["ACTUAL_N_TRIALS"] = actual_trials
    cfg_snapshot["ACTUAL_CV_FOLDS"] = actual_folds
    cfg_snapshot["ACTUAL_TIMEOUT"] = actual_timeout
    cfg_snapshot["ACTUAL_FBETA"] = actual_fbeta
    cfg_snapshot["MODEL_OUTPUT_PATH"] = str(target_model_path)
    summary.set_config(cfg_snapshot)

    logger.info("=" * 70)
    logger.info("PHASE 4: ML CLASSIFICATION TRAINING PIPELINE")
    logger.info("=" * 70)
    logger.info("Log file: %s", log_file)
    logger.info("Configuration snapshot: %s", cfg_snapshot)

    # ── Input Standardization & Contract Verification ─────────
    df = standardize_input_dataframe(df)

    # ── Optional Negative Subsampling ────────────────────────
    if subsample_ratio is not None and 0.0 < float(subsample_ratio) < 1.0:
        ratio = float(subsample_ratio)
        pos_df = df[df["label"] == 1]
        neg_df = df[df["label"] == 0]
        if len(neg_df) > 0:
            n_neg_sample = max(1, int(np.round(len(neg_df) * ratio)))
            neg_sampled = neg_df.sample(n=n_neg_sample, random_state=config.OPTUNA_SEED)
            df = pd.concat([pos_df, neg_sampled], ignore_index=True)
            logger.info(
                "Applied negative subsampling (ratio=%.4f): retained %d positives, subsampled negatives from %d to %d (total %d rows)",
                ratio, len(pos_df), len(neg_df), n_neg_sample, len(df)
            )

    X_df = df[FEATURE_COLS]
    y = df["label"].to_numpy(dtype=int)
    s1_ids = df["s1_id"].to_numpy()
    cand_ids = df["cand_id"].to_numpy()

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    cfg_spw = _extract_config_val(config_obj, "XGB_SCALE_POS_WEIGHT", config.XGB_SCALE_POS_WEIGHT)
    if cfg_spw is not None and float(cfg_spw) > 0:
        scale_pos_weight = float(cfg_spw)
    else:
        scale_pos_weight = compute_scale_pos_weight(y)

    unique_entities = len(np.unique(s1_ids))
    if actual_folds > unique_entities:
        logger.warning(
            "Requested cv_folds=%d exceeds unique entities=%d. Capping cv_folds to %d.",
            actual_folds, unique_entities, unique_entities
        )
        actual_folds = unique_entities

    summary.set_dataset(
        total_rows=len(df),
        n_features=len(FEATURE_COLS),
        n_pos=n_pos,
        n_neg=n_neg,
        scale_pos_weight=scale_pos_weight,
        unique_s1_entities=unique_entities,
    )

    logger.info(
        "Dataset loaded: %d rows, %d features, %d unique S1 entities",
        len(df), len(FEATURE_COLS), unique_entities
    )
    logger.info(
        "Class balance: Positive=%d, Negative=%d, Effective scale_pos_weight=%.4f",
        n_pos, n_neg, scale_pos_weight
    )

    # ── Phase A: Hyperparameter Optimization (Optuna) ────────
    logger.info("-" * 70)
    logger.info("PHASE A: HYPERPARAMETER OPTIMIZATION (Optuna)")
    logger.info("Number of trials: %d, Timeout: %s, CV Folds: %d", actual_trials, actual_timeout, actual_folds)
    phase_a_start = time.time()

    trials_detail: list[dict[str, Any]] = []

    def objective(trial: optuna.Trial) -> float:
        trial_start = time.time()
        max_n_est = int(_extract_config_val(config_obj, "XGB_MAX_N_ESTIMATORS", config.XGB_MAX_N_ESTIMATORS))
        high_n_est = max(10, min(3000, max_n_est))
        low_n_est = min(300, high_n_est)

        # Decision 10: Parameter names prefixed with 'classifier__'
        params = {
            "classifier__n_estimators": trial.suggest_int("classifier__n_estimators", low_n_est, high_n_est),
            "classifier__max_depth": trial.suggest_int("classifier__max_depth", 3, 9),
            "classifier__learning_rate": trial.suggest_float("classifier__learning_rate", 1e-3, 0.3, log=True),
            "classifier__subsample": trial.suggest_float("classifier__subsample", 0.5, 1.0),
            "classifier__colsample_bytree": trial.suggest_float("classifier__colsample_bytree", 0.5, 1.0),
            "classifier__min_child_weight": trial.suggest_int("classifier__min_child_weight", 1, 20),
            "classifier__reg_alpha": trial.suggest_float("classifier__reg_alpha", 1e-8, 10.0, log=True),
            "classifier__reg_lambda": trial.suggest_float("classifier__reg_lambda", 1e-8, 10.0, log=True),
        }

        gkf = GroupKFold(n_splits=actual_folds)
        fold_scores: list[float] = []

        for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_df, y, groups=s1_ids)):
            X_train_f, y_train_f = X_df.iloc[train_idx], y[train_idx]
            X_val_f, y_val_f = X_df.iloc[val_idx], y[val_idx]
            s1_val_f = s1_ids[val_idx]

            # Decision 2: Manual encoding for pruning validation eval set
            X_val_enc = manual_encode_for_eval(
                X_train_f,
                y_train_f,
                X_val_f,
                cv=config.TARGET_ENCODER_CV,
                random_state=config.OPTUNA_SEED,
            )

            pipeline = build_pipeline(
                model_family=actual_model_family,
                scale_pos_weight=scale_pos_weight,
                cv=actual_target_encoder_cv,
                random_state=config.OPTUNA_SEED,
                device=xgb_device,
                tree_method=xgb_tree_method,
            )
            pipeline.set_params(**params)

            # Fit with pruning callback watching validation_0-logloss
            pruning_cb = XGBoostPruningCallback(trial, "validation_0-logloss")
            pipeline.set_params(classifier__callbacks=[pruning_cb])
            pipeline.fit(
                X_train_f,
                y_train_f,
                classifier__eval_set=[(X_val_enc, y_val_f)],
                classifier__verbose=False,
            )

            val_probs = pipeline.predict_proba(X_val_f)[:, 1]
            # Decision 6: Evaluated at fixed threshold (0.50)
            score = entity_macro_fbeta(
                y_val_f,
                val_probs,
                s1_val_f,
                threshold=0.50,
                beta=actual_fbeta,
            )
            fold_scores.append(score)

        mean_score = float(np.mean(fold_scores))
        duration = time.time() - trial_start

        trials_detail.append({
            "trial_number": trial.number,
            "params": params,
            "fold_scores": fold_scores,
            "mean_score": mean_score,
            "state": "COMPLETE",
            "duration_seconds": duration,
        })

        logger.info(
            "Trial %d complete | Mean F_%.1f: %.4f | Duration: %.2fs",
            trial.number, actual_fbeta, mean_score, duration
        )
        return mean_score

    sampler = TPESampler(seed=config.OPTUNA_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        objective,
        n_trials=actual_trials,
        timeout=actual_timeout,
        catch=(optuna.TrialPruned,),
    )

    phase_a_duration = time.time() - phase_a_start
    best_trial = study.best_trial
    best_params = best_trial.params

    completed_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    summary.set_hpo_results(
        total_trials=len(study.trials),
        completed_trials=completed_count,
        pruned_trials=pruned_count,
        best_trial_number=best_trial.number,
        best_score=best_trial.value,
        best_params=best_params,
        duration_seconds=phase_a_duration,
        trials_detail=trials_detail,
    )

    logger.info("Phase A finished in %.2fs. Completed: %d, Pruned: %d", phase_a_duration, completed_count, pruned_count)
    logger.info("Best Trial #%d with F_%.1f: %.4f", best_trial.number, actual_fbeta, best_trial.value)
    logger.info("Best parameters: %s", best_params)

    # ── Phase B: Clean Cross-Validated Evaluation ────────────
    logger.info("-" * 70)
    logger.info("PHASE B: CLEAN CROSS-VALIDATED EVALUATION (Fixed Best Params)")
    phase_b_start = time.time()

    gkf = GroupKFold(n_splits=actual_folds)
    oof_probs = np.zeros(len(df), dtype=float)
    oof_fold_scores: list[float] = []

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_df, y, groups=s1_ids)):
        fold_start = time.time()
        X_train_f, y_train_f = X_df.iloc[train_idx], y[train_idx]
        X_val_f, y_val_f = X_df.iloc[val_idx], y[val_idx]
        s1_val_f = s1_ids[val_idx]

        fold_pipeline = build_pipeline(
            model_family=actual_model_family,
            scale_pos_weight=scale_pos_weight,
            cv=actual_target_encoder_cv,
            random_state=config.OPTUNA_SEED,
            device=xgb_device,
            tree_method=xgb_tree_method,
        )
        fold_pipeline.set_params(**best_params)

        fold_pipeline.fit(X_train_f, y_train_f, classifier__verbose=False)
        probs = fold_pipeline.predict_proba(X_val_f)[:, 1]
        oof_probs[val_idx] = probs

        f_score = entity_macro_fbeta(
            y_val_f,
            probs,
            s1_val_f,
            threshold=0.50,
            beta=actual_fbeta,
        )
        oof_fold_scores.append(f_score)
        fold_dur = time.time() - fold_start

        logger.info(
            "Fold %d/%d: Train size=%d, Val size=%d | F_%.1f@0.5: %.4f | Duration: %.2fs",
            fold_idx + 1, actual_folds, len(train_idx), len(val_idx), actual_fbeta, f_score, fold_dur
        )

    oof_macro_score = entity_macro_fbeta(
        y,
        oof_probs,
        s1_ids,
        threshold=0.50,
        beta=actual_fbeta,
    )
    phase_b_duration = time.time() - phase_b_start

    summary.set_clean_eval_results(
        n_folds=actual_folds,
        fold_scores=oof_fold_scores,
        oof_macro_fbeta=oof_macro_score,
        hpo_winning_score=best_trial.value,
        duration_seconds=phase_b_duration,
    )

    logger.info(
        "Phase B clean OOF Macro F_%.1f@0.5: %.4f (HPO winning trial was %.4f, diff: %+.4f)",
        actual_fbeta, oof_macro_score, best_trial.value, oof_macro_score - best_trial.value
    )

    # ── Phase C: Threshold Calibration ───────────────────────
    logger.info("-" * 70)
    logger.info("PHASE C: THRESHOLD CALIBRATION")
    threshold_start = float(_extract_config_val(config_obj, "THRESHOLD_SEARCH_START", config.THRESHOLD_SEARCH_START))
    threshold_end = float(_extract_config_val(config_obj, "THRESHOLD_SEARCH_END", config.THRESHOLD_SEARCH_END))
    threshold_step = float(_extract_config_val(config_obj, "THRESHOLD_SEARCH_STEP", config.THRESHOLD_SEARCH_STEP))
    logger.info(
        "Sweeping threshold in [%.2f, %.2f] with step %.2f",
        threshold_start, threshold_end, threshold_step
    )
    phase_c_start = time.time()

    thresholds = np.arange(
        threshold_start,
        threshold_end + (threshold_step / 2.0),
        threshold_step,
    )

    threshold_sweep: list[dict[str, float]] = []
    best_threshold = 0.50
    best_calib_score = -1.0

    for t in thresholds:
        t_val = round(float(t), 4)
        score = entity_macro_fbeta(
            y,
            oof_probs,
            s1_ids,
            threshold=t_val,
            beta=actual_fbeta,
        )
        threshold_sweep.append({"threshold": t_val, "score": score})
        if score > best_calib_score:
            best_calib_score = score
            best_threshold = t_val

    # Compute overall precision and recall at winning threshold
    y_preds_opt = (oof_probs > best_threshold).astype(int)
    tp = int(np.sum((y == 1) & (y_preds_opt == 1)))
    fp = int(np.sum((y == 0) & (y_preds_opt == 1)))
    fn = int(np.sum((y == 1) & (y_preds_opt == 0)))
    precision_opt = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_opt = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    phase_c_duration = time.time() - phase_c_start
    summary.set_calibration_results(
        optimal_threshold=best_threshold,
        best_score=best_calib_score,
        precision_at_optimal=precision_opt,
        recall_at_optimal=recall_opt,
        threshold_sweep=threshold_sweep,
        duration_seconds=phase_c_duration,
    )

    logger.info(
        "Optimal threshold: %.4f | Best OOF F_%.1f: %.4f | Precision: %.4f, Recall: %.4f",
        best_threshold, actual_fbeta, best_calib_score, precision_opt, recall_opt
    )

    # ── Phase D: Final Fit and Persistence ───────────────────
    logger.info("-" * 70)
    logger.info("PHASE D: FINAL FIT AND PERSISTENCE")
    phase_d_start = time.time()

    final_scale_pos_weight = compute_scale_pos_weight(y) if (cfg_spw is None or float(cfg_spw) <= 0) else float(cfg_spw)
    weights_match = abs(final_scale_pos_weight - scale_pos_weight) < 1e-6
    if not weights_match:
        logger.warning(
            "scale_pos_weight mismatch! Initial: %.6f, Final: %.6f",
            scale_pos_weight, final_scale_pos_weight
        )
    else:
        logger.info("scale_pos_weight sanity check passed: %.4f", final_scale_pos_weight)

    # Decision 10: Strip 'classifier__' prefix for final classifier params
    xgb_params = {k.replace("classifier__", ""): v for k, v in best_params.items()}
    final_pipeline = build_pipeline(
        model_family=actual_model_family,
        scale_pos_weight=final_scale_pos_weight,
        cv=actual_target_encoder_cv,
        random_state=config.OPTUNA_SEED,
        device=xgb_device,
        tree_method=xgb_tree_method,
        **xgb_params,
    )

    logger.info("Fitting final pipeline on 100%% of labeled data (%d rows)...", len(df))
    final_pipeline.fit(X_df, y, classifier__verbose=False)
    phase_d_duration = time.time() - phase_d_start

    artifact_path = Path(target_model_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": final_pipeline,
        "threshold": best_threshold,
        "best_params": best_params,
        "feature_names": FEATURE_COLS,
    }

    joblib.dump(artifact, artifact_path)
    logger.info("Model artifact successfully saved to: %s", artifact_path)

    summary.set_final_fit_results(
        artifact_path=str(artifact_path),
        artifact_keys=list(artifact.keys()),
        initial_scale_pos_weight=scale_pos_weight,
        final_scale_pos_weight=final_scale_pos_weight,
        weights_matched=weights_match,
        duration_seconds=phase_d_duration,
    )

    summary_file = summary.save(metrics_dir=actual_metrics_dir)
    logger.info("Run summary JSON written to: %s", summary_file)
    logger.info("=" * 70)
    logger.info("PHASE 4 TRAINING COMPLETE!")
    logger.info("=" * 70)

    return artifact
