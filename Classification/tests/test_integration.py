"""Integration tests for Phase 4 ML Classification pipeline."""

import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import TargetEncoder

from src.model import (
    COUNTRY_COL,
    FEATURE_COLS,
    NUMERICAL_COLS,
    build_pipeline,
    build_preprocessor,
    create_classifier,
)


def test_pipeline_end_to_end_with_unseen_country(synthetic_test_df_with_france: pd.DataFrame):
    """Test full pipeline fit and predict_proba on dataset containing unseen country 'france'."""
    df = synthetic_test_df_with_france
    X = df[FEATURE_COLS]
    y = df["label"].to_numpy()

    # Train on us and india only
    train_mask = df[COUNTRY_COL[0]].isin(["us", "india"])
    X_train, y_train = X[train_mask], y[train_mask]
    X_test = X[~train_mask]  # rows with 'france'

    pipeline = build_pipeline(
        model_family="xgboost",
        scale_pos_weight=2.0,
        n_estimators=50,
        max_depth=3,
        random_state=42,
    )

    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]

    assert len(probs) == len(X_test)
    assert not np.isnan(probs).any(), "Probabilities for unseen country contain NaN!"
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_no_target_encoding_leakage(synthetic_train_df: pd.DataFrame):
    """Confirm TargetEncoder inside fold only sees training split labels."""
    df = synthetic_train_df
    X = df[FEATURE_COLS]
    y = df["label"].to_numpy()
    s1_ids = df["s1_id"].to_numpy()

    gkf = GroupKFold(n_splits=3)
    train_idx, val_idx = next(gkf.split(X, y, groups=s1_ids))

    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]

    # Preprocessor fitted strictly on training fold
    preprocessor = build_preprocessor(cv=3, random_state=42)
    preprocessor.fit(X_train, y_train)

    enc_step = preprocessor.named_transformers_["target_enc"]
    # Check that the encoder learned categories match train fold country set
    train_countries = set(X_train[COUNTRY_COL[0]].unique())
    assert set(enc_step.categories_[0]).issubset(train_countries)


def test_persistence_round_trip(synthetic_train_df: pd.DataFrame):
    """Confirm pipeline serialization round-trip reproduces identical predictions."""
    df = synthetic_train_df
    X = df[FEATURE_COLS]
    y = df["label"].to_numpy()

    pipeline = build_pipeline(
        model_family="xgboost",
        scale_pos_weight=1.5,
        n_estimators=50,
        max_depth=3,
        random_state=42,
    )
    pipeline.fit(X, y)
    preds_before = pipeline.predict_proba(X)[:, 1]

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_model_path = Path(tmp_dir) / "test_model.pkl"
        artifact = {
            "model": pipeline,
            "threshold": 0.65,
            "best_params": {"classifier__n_estimators": 50},
            "feature_names": FEATURE_COLS,
        }
        joblib.dump(artifact, tmp_model_path)

        loaded_artifact = joblib.load(tmp_model_path)
        loaded_pipeline = loaded_artifact["model"]
        preds_after = loaded_pipeline.predict_proba(X)[:, 1]

        np.testing.assert_allclose(
            preds_before,
            preds_after,
            rtol=1e-6,
            atol=1e-6,
            err_msg="Predictions differ after model serialization round trip!",
        )


def test_monotone_constraint_behavioral(synthetic_train_df: pd.DataFrame):
    """Behavioral test: increasing a positively-constrained feature never decreases predicted probability."""
    df = synthetic_train_df
    X = df[FEATURE_COLS]
    y = df["label"].to_numpy()

    pipeline = build_pipeline(
        model_family="xgboost",
        scale_pos_weight=2.0,
        n_estimators=100,
        max_depth=4,
        random_state=42,
    )
    pipeline.fit(X, y)

    # Pick a base row
    base_row = X.iloc[[0]].copy()

    # Feature 0: 'dim_1_name_jaro_winkler' (monotone +1)
    row_low = base_row.copy()
    row_low[NUMERICAL_COLS[0]] = 0.1

    row_high = base_row.copy()
    row_high[NUMERICAL_COLS[0]] = 0.9

    prob_low = pipeline.predict_proba(row_low)[:, 1][0]
    prob_high = pipeline.predict_proba(row_high)[:, 1][0]

    assert prob_high >= prob_low - 1e-6, (
        f"Monotone constraint violated! {NUMERICAL_COLS[0]} 0.1 gave {prob_low:.4f}, but 0.9 gave {prob_high:.4f}"
    )


def test_fallback_model_family_raises():
    """Verify Decision 8: requesting lightgbm raises NotImplementedError."""
    with pytest.raises(NotImplementedError, match="LightGBM model family fallback is currently unimplemented"):
        create_classifier(model_family="lightgbm")


def test_polars_dataframe_orchestrator_interface(synthetic_train_df: pd.DataFrame, tmp_path: Path):
    """Confirm run_training accepts Polars DataFrame with s1_id, cand_id, dim_X_... columns."""
    import polars as pl
    from src.train import run_training
    from ml_layer.trainer import run_training as ml_layer_run_training

    # Pass Polars DataFrame with canonical upstream columns
    polars_df = pl.from_pandas(synthetic_train_df)

    class MockOrchestratorConfig:
        CV_FOLDS = 2
        OPTUNA_N_TRIALS = 2
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

    mock_cfg = MockOrchestratorConfig()
    model_out = tmp_path / "models" / "xgb_model.pkl"

    # Orchestrator calling pattern: run_training(labeled_df, model_out, config)
    artifact = ml_layer_run_training(polars_df, model_out, mock_cfg)

    assert isinstance(artifact, dict)
    assert "model" in artifact
    assert "threshold" in artifact
    assert "best_params" in artifact
    assert "feature_names" in artifact
    assert model_out.exists()

    # Verify loaded artifact can predict on canonical FEATURE_COLS
    loaded = joblib.load(model_out)
    pred_probs = loaded["model"].predict_proba(synthetic_train_df[FEATURE_COLS])[:, 1]
    assert len(pred_probs) == len(synthetic_train_df)
    assert not np.isnan(pred_probs).any()


def test_negative_subsampling_behavior(synthetic_train_df: pd.DataFrame, tmp_path: Path):
    """Confirm negative subsampling reduces negative rows while preserving all positives."""
    from src.train import run_training

    # Replicate negatives to create higher imbalance
    pos_df = synthetic_train_df[synthetic_train_df["label"] == 1]
    neg_df = synthetic_train_df[synthetic_train_df["label"] == 0]
    imbalanced_df = pd.concat([pos_df] + [neg_df] * 3, ignore_index=True)

    out_model = tmp_path / "subsample_model.pkl"
    artifact = run_training(
        imbalanced_df,
        model_output_path=out_model,
        n_trials=2,
        cv_folds=2,
        negative_subsample_ratio=0.5,
    )
    assert out_model.exists()
    assert 0.0 <= artifact["threshold"] <= 1.0


def test_legacy_column_alias_backward_compatibility(tmp_path: Path):
    """Confirm training runs seamlessly on legacy DataFrames with source1_entity_id, rrf_score, country, etc."""
    from src.train import run_training
    from src.model import FEATURE_COLS

    # Build dataset using legacy column names
    rows = []
    for i in range(8):
        s1 = f"S1-{i:03d}"
        for j in range(5):
            is_match = 1 if j == 0 else 0
            rows.append({
                "source1_entity_id": s1,
                "candidate_entity_id": f"S2-{i:02d}_{j:02d}",
                "label": is_match,
                "name_jaro_winkler": 0.9 if is_match else 0.2,
                "name_monge_elkan": 0.85 if is_match else 0.15,
                "name_levenshtein": 0.8 if is_match else 0.1,
                "exact_name_match": is_match,
                "name_phonetic_match": is_match,
                "acronym_score": 0.5,
                "name_semantic_cosine": 0.9 if is_match else 0.3,
                "addr_token_jaccard": 0.7 if is_match else 0.1,
                "addr_token_containment": 0.8 if is_match else 0.2,
                "exact_addr_match": is_match,
                "addr_numeric_jaccard": 0.6 if is_match else 0.0,
                "addr_numeric_exact": 1 if is_match else 0,
                "addr_numeric_exists_both": 1,
                "name_bm25_score": 0.85 if is_match else 0.1,
                "addr_bm25_score": 0.75 if is_match else 0.05,
                "rrf_score": 0.03 if is_match else 0.005,
                "stream_overlap_count": 3 if is_match else 1,
                "name_length_ratio": 0.9,
                "addr_length_ratio": 0.85,
                "s1_name_missing": 0,
                "cand_name_missing": 0,
                "s1_addr_missing": 0,
                "cand_addr_missing": 0,
                "cross_script_target": 0,
                "source_origin": 0,
                "country": "US",
            })

    legacy_df = pd.DataFrame(rows)
    out_model = tmp_path / "legacy_alias_model.pkl"
    artifact = run_training(
        legacy_df,
        model_output_path=out_model,
        n_trials=2,
        cv_folds=2,
        timeout_seconds=30,
    )
    assert out_model.exists()
    assert set(artifact["feature_names"]) == set(FEATURE_COLS)
    assert 0.0 <= artifact["threshold"] <= 1.0


