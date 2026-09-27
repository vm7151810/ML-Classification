"""Pipeline connection and end-to-end integration verification tests for Phase 4.

Verifies the full pipeline integration between:
1. Upstream Preprocessing & Feature Layer outputs -> Labeled DataFrame.
2. Training entrypoint: run_training(labeled_df, model_out, config) / ml_layer.trainer.
3. Downstream Orchestrator infer-mode verification: verify_model_feature_alignment().
4. Online inference scoring: model.predict_proba() > threshold on seen and unseen countries.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.pipeline import Pipeline

from ml_layer.trainer import run_training
from src import config
from src.metrics import compute_scale_pos_weight, entity_macro_fbeta
from src.model import (
    COUNTRY_COL,
    FEATURE_COLS,
    NUMERICAL_COLS,
    standardize_input_dataframe,
)


def verify_model_feature_alignment(model_artifact: dict) -> None:
    """Exact feature alignment check from Orchestrator_1.0.md (lines 306-335).

    Compares model_artifact['feature_names'] (saved at train time)
    against the live FEATURE_COLS constant.
    """
    saved_cols = model_artifact.get("feature_names", [])
    live_cols = FEATURE_COLS
    if saved_cols == live_cols:
        return
    missing_in_live = set(saved_cols) - set(live_cols)
    extra_in_live = set(live_cols) - set(saved_cols)
    order_diffs = [
        (i, s, l) for i, (s, l) in enumerate(zip(saved_cols, live_cols)) if s != l
    ]
    raise RuntimeError(
        f"Feature column mismatch between saved model and live feature_extractor.py!\n"
        f"  Missing in live code : {missing_in_live}\n"
        f"  Extra in live code   : {extra_in_live}\n"
        f"  Order diffs (idx, saved, live): {order_diffs}\n"
        "Fix feature_extractor.py or retrain the model before running infer."
    )


def test_pipeline_input_connection_with_upstream_schema(tmp_path: Path):
    """Verify that the upstream output format feeds cleanly into run_training.

    Simulates the exact train-mode pipeline flow:
    1. Upstream candidate generation (seen_dict) + feature extractor produces train_df (Polars).
    2. Left-join with ground truth produces labeled_df.
    3. run_training(labeled_df, model_out, config) is called.
    """
    n_entities = 12
    candidates_per_entity = 10
    total_rows = n_entities * candidates_per_entity

    s1_ids = []
    cand_ids = []
    for i in range(n_entities):
        s1 = f"S1-{i:05d}"
        for j in range(candidates_per_entity):
            s1_ids.append(s1)
            cand_ids.append(f"S2-{i:03d}_{j:02d}")

    # Generate 26 upstream features using exact dim_X_... names from feature layer specs
    rng = np.random.RandomState(42)
    feature_dict: dict[str, Any] = {
        "s1_id": s1_ids,
        "cand_id": cand_ids,
        "dim_1_name_jaro_winkler": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_2_name_monge_elkan": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_3_name_levenshtein_norm": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_4_name_exact_match": rng.choice([0, 1], total_rows).astype(np.int32),
        "dim_5_phonetic_match": rng.choice([0, 1], total_rows).astype(np.int32),
        "dim_6_name_acronym_match": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_7_semantic_cosine": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_8_addr_jaccard": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_9_addr_containment": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_10_addr_exact_match": rng.choice([0, 1], total_rows).astype(np.int32),
        "dim_11_num_jaccard": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_12_num_exact_match": rng.choice([0, 1], total_rows).astype(np.int32),
        "dim_13_num_presence": rng.choice([0, 1], total_rows).astype(np.int32),
        "dim_14_name_bm25_norm": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_15_addr_bm25_norm": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_16_rrf_score": rng.uniform(0.0, 0.5, total_rows).astype(np.float32),
        "dim_17_overlap_count": rng.choice([1, 2, 3], total_rows).astype(np.int32),
        "dim_18_name_len_ratio": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_19_addr_len_ratio": rng.uniform(0.0, 1.0, total_rows).astype(np.float32),
        "dim_20_s1_name_missing": rng.choice([0, 1], total_rows, p=[0.9, 0.1]).astype(np.int32),
        "dim_21_cand_name_missing": rng.choice([0, 1], total_rows, p=[0.9, 0.1]).astype(np.int32),
        "dim_22_s1_addr_missing": rng.choice([0, 1], total_rows, p=[0.8, 0.2]).astype(np.int32),
        "dim_23_cand_addr_missing": rng.choice([0, 1], total_rows, p=[0.8, 0.2]).astype(np.int32),
        "dim_24_cross_script": rng.choice([0, 1], total_rows, p=[0.8, 0.2]).astype(np.int32),
        "dim_25_source_origin": rng.choice([0, 1], total_rows).astype(np.int32),
        "dim_26_country_raw": rng.choice(["India", "US"], total_rows),
    }
    train_df = pl.DataFrame(feature_dict)

    # Ground truth: assign 1 true match for first 8 entities, entities 8..11 are singletons (0 matches)
    gt_rows = []
    for i in range(8):
        s1 = f"S1-{i:05d}"
        cand = f"S2-{i:03d}_00"  # Candidate 0 is true match
        gt_rows.append({"source1_entity_id": s1, "matched_entity_ids": cand})
    for i in range(8, 12):
        s1 = f"S1-{i:05d}"
        gt_rows.append({"source1_entity_id": s1, "matched_entity_ids": ""})

    gt_df = pl.DataFrame(gt_rows)

    # Label assignment via left-join (Orchestrator lines 555-567)
    gt_exploded = (
        gt_df
        .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
        .explode("matched_list")
        .rename({"source1_entity_id": "s1_id", "matched_list": "cand_id"})
        .filter(pl.col("cand_id") != "")
        .with_columns(pl.lit(1).alias("label").cast(pl.Int8))
    )
    labeled_df = (
        train_df.join(gt_exploded[["s1_id", "cand_id", "label"]],
                      on=["s1_id", "cand_id"], how="left")
        .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
    )

    assert len(labeled_df) == total_rows
    assert labeled_df["label"].sum() == 8

    # Define mock config passed by Orchestrator
    class OrchestratorConfig:
        CV_FOLDS = 3
        OPTUNA_N_TRIALS = 3
        OPTUNA_TIMEOUT_SECONDS = 30
        OPTUNA_SEED = 42
        FBETA_SCORE_BETA = 0.5
        MODEL_FAMILY = "xgboost"
        XGB_DEVICE = "cpu"
        XGB_TREE_METHOD = "hist"
        XGB_NEGATIVE_SUBSAMPLE_RATIO = 1.0
        XGB_SCALE_POS_WEIGHT = -1
        LOGS_DIR = str(tmp_path / "logs")
        METRICS_DIR = str(tmp_path / "metrics")

    mock_cfg = OrchestratorConfig()
    model_output_path = tmp_path / "models" / "xgb_model.pkl"

    # Execute training via Orchestrator call
    artifact = run_training(labeled_df, model_output_path, mock_cfg)

    # Verify model artifact produced
    assert model_output_path.exists()
    assert isinstance(artifact, dict)
    assert set(artifact.keys()) == {"model", "threshold", "best_params", "feature_names"}
    assert isinstance(artifact["model"], Pipeline)
    assert 0.0 <= artifact["threshold"] <= 1.0


def test_pipeline_output_connection_and_orchestrator_infer(tmp_path: Path):
    """Verify that the saved model artifact is completely compatible with Orchestrator infer mode."""
    # 1. Train model on small synthetic data
    from tests.conftest import generate_synthetic_data

    train_data = generate_synthetic_data(n_entities=10, cands_per_entity=8, random_seed=42)
    model_path = tmp_path / "infer_test_model.pkl"

    artifact = run_training(
        train_data,
        model_output_path=model_path,
        n_trials=2,
        cv_folds=2,
        timeout_seconds=30,
    )

    # 2. Test verify_model_feature_alignment() as run at Orchestrator startup
    loaded_artifact = joblib.load(model_path)
    # Must not raise RuntimeError
    verify_model_feature_alignment(loaded_artifact)

    # 3. Simulate Orchestrator infer-mode candidate scoring
    pipeline = loaded_artifact["model"]
    threshold = loaded_artifact["threshold"]
    feature_names = loaded_artifact["feature_names"]

    # Candidate batch including a seen country (US) and UNSEEN country (France)
    infer_rows = generate_synthetic_data(n_entities=4, cands_per_entity=5, random_seed=99)
    infer_rows[COUNTRY_COL[0]] = ["France"] * 10 + ["US"] * 10

    # Ensure batch contains canonical FEATURE_COLS
    X_batch = infer_rows[feature_names]
    probs = pipeline.predict_proba(X_batch)[:, 1]

    assert len(probs) == len(X_batch)
    assert not np.isnan(probs).any(), "Inference probabilities contain NaN!"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Verify binary acceptance via threshold
    accepted = probs > threshold
    assert isinstance(accepted, np.ndarray)
    assert accepted.dtype == bool


def test_extreme_imbalance_and_singleton_handling(tmp_path: Path):
    """Confirm the pipeline gracefully handles extreme class imbalance and singleton clusters."""
    # Create dataset with 1 positive and 100 negatives across 10 entities
    rows = []
    for i in range(10):
        s1 = f"S1-{i:04d}"
        for j in range(10):
            # Only entity 0, candidate 0 is a match; all other 9 entities are singletons
            is_match = 1 if (i == 0 and j == 0) else 0
            row = {
                "source1_entity_id": s1,
                "candidate_entity_id": f"S2-{i:03d}_{j:02d}",
                "country": "India",
                "label": is_match,
            }
            for col in NUMERICAL_COLS:
                row[col] = 0.9 if is_match else 0.1
            rows.append(row)

    imbalanced_df = pd.DataFrame(rows)
    spw = compute_scale_pos_weight(imbalanced_df["label"].to_numpy())
    assert spw == 99.0  # 99 negatives / 1 positive

    model_path = tmp_path / "extreme_imbalance_model.pkl"
    artifact = run_training(
        imbalanced_df,
        model_output_path=model_path,
        n_trials=2,
        cv_folds=2,
        timeout_seconds=30,
        negative_subsample_ratio=0.5,  # Subsample 50% of negatives
    )
    assert model_path.exists()
    assert 0.0 <= artifact["threshold"] <= 1.0
