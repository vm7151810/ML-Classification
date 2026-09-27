"""Model and Preprocessor Architecture for Phase 4 ML Classification.

Implements:
- Authoritative FEATURE_COLS (26 features) and MONOTONE_VECTOR.
- ColumnTransformer with Decision 1 fix (numeric passthrough first, TargetEncoder second).
- Manual encoding for HPO pruning callback per Decision 2.
- Model factory with Decision 8 fallback guard.
- Full scikit-learn Pipeline construction.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier

from src import config

# ── Feature Definitions (Authoritative 26-Dim Feature Space from Feature Layer) ─
NUMERICAL_COLS: list[str] = [
    "dim_1_name_jaro_winkler",      # Dim 1  (+1)
    "dim_2_name_monge_elkan",       # Dim 2  (+1)
    "dim_3_name_levenshtein_norm",  # Dim 3  (+1)
    "dim_4_name_exact_match",       # Dim 4  (+1)
    "dim_5_phonetic_match",         # Dim 5  (+1)
    "dim_6_name_acronym_match",     # Dim 6  (+1)
    "dim_7_semantic_cosine",        # Dim 7  (+1)
    "dim_8_addr_jaccard",           # Dim 8  (+1)
    "dim_9_addr_containment",       # Dim 9  (+1)
    "dim_10_addr_exact_match",      # Dim 10 (+1)
    "dim_11_num_jaccard",           # Dim 11 (+1)
    "dim_12_num_exact_match",       # Dim 12 (0)
    "dim_13_num_presence",          # Dim 13 (0)
    "dim_14_name_bm25_norm",        # Dim 14 (+1)
    "dim_15_addr_bm25_norm",        # Dim 15 (+1)
    "dim_16_rrf_score",             # Dim 16 (+1)
    "dim_17_overlap_count",         # Dim 17 (+1)
    "dim_18_name_len_ratio",        # Dim 18 (0)
    "dim_19_addr_len_ratio",        # Dim 19 (0)
    "dim_20_s1_name_missing",       # Dim 20 (0)
    "dim_21_cand_name_missing",     # Dim 21 (0)
    "dim_22_s1_addr_missing",       # Dim 22 (0)
    "dim_23_cand_addr_missing",     # Dim 23 (0)
    "dim_24_cross_script",          # Dim 24 (0)
    "dim_25_source_origin",         # Dim 25 (0)
]

COUNTRY_COL: list[str] = ["dim_26_country_raw"]  # Dim 26 (0) - raw string, target-encoded

FEATURE_COLS: list[str] = NUMERICAL_COLS + COUNTRY_COL

# Monotonic constraint vector strictly matching FEATURE_COLS order
# (+1 = monotonically increasing, 0 = unconstrained)
MONOTONE_VECTOR: tuple[int, ...] = (
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0
)

ID_COLS: list[str] = ["s1_id", "cand_id"]

# Column aliases mapping upstream blocking/orchestrator naming to Phase 4 canonical names
COLUMN_ALIASES: dict[str, str] = {
    # Entity ID aliases
    "source1_entity_id": "s1_id",
    "candidate_entity_id": "cand_id",
    "is_match": "label",
    "y": "label",
    # Legacy flat names to canonical dim_N_* names
    "name_jaro_winkler": "dim_1_name_jaro_winkler",
    "name_monge_elkan": "dim_2_name_monge_elkan",
    "name_levenshtein": "dim_3_name_levenshtein_norm",
    "name_levenshtein_norm": "dim_3_name_levenshtein_norm",
    "dim_3_name_levenshtein": "dim_3_name_levenshtein_norm",
    "exact_name_match": "dim_4_name_exact_match",
    "name_phonetic_match": "dim_5_phonetic_match",
    "dim_5_name_phonetic_match": "dim_5_phonetic_match",
    "acronym_score": "dim_6_name_acronym_match",
    "dim_6_acronym_score": "dim_6_name_acronym_match",
    "name_semantic_cosine": "dim_7_semantic_cosine",
    "dim_7_name_semantic_cosine": "dim_7_semantic_cosine",
    "addr_token_jaccard": "dim_8_addr_jaccard",
    "dim_8_addr_token_jaccard": "dim_8_addr_jaccard",
    "addr_token_containment": "dim_9_addr_containment",
    "dim_9_addr_token_containment": "dim_9_addr_containment",
    "exact_addr_match": "dim_10_addr_exact_match",
    "addr_numeric_jaccard": "dim_11_num_jaccard",
    "dim_11_addr_numeric_jaccard": "dim_11_num_jaccard",
    "addr_numeric_exact": "dim_12_num_exact_match",
    "dim_12_addr_numeric_exact": "dim_12_num_exact_match",
    "addr_numeric_exists_both": "dim_13_num_presence",
    "dim_13_addr_numeric_exists_both": "dim_13_num_presence",
    "name_bm25_score": "dim_14_name_bm25_norm",
    "dim_14_name_bm25_score": "dim_14_name_bm25_norm",
    "addr_bm25_score": "dim_15_addr_bm25_norm",
    "dim_15_addr_bm25_score": "dim_15_addr_bm25_norm",
    "rrf_score": "dim_16_rrf_score",
    "dim_16_rrf": "dim_16_rrf_score",
    "dim_17_stream_overlap_count": "dim_17_overlap_count",
    "stream_overlap_count": "dim_17_overlap_count",
    "name_length_ratio": "dim_18_name_len_ratio",
    "dim_18_name_length_ratio": "dim_18_name_len_ratio",
    "addr_length_ratio": "dim_19_addr_len_ratio",
    "dim_19_addr_length_ratio": "dim_19_addr_len_ratio",
    "s1_name_missing": "dim_20_s1_name_missing",
    "cand_name_missing": "dim_21_cand_name_missing",
    "s1_addr_missing": "dim_22_s1_addr_missing",
    "cand_addr_missing": "dim_23_cand_addr_missing",
    "cross_script_target": "dim_24_cross_script",
    "dim_24_cross_script_target": "dim_24_cross_script",
    "source_origin": "dim_25_source_origin",
    "country": "dim_26_country_raw",
    "dim_26_country": "dim_26_country_raw",
}


def standardize_input_dataframe(data: Any) -> pd.DataFrame:
    """Standardize input data from Polars/Pandas/File to canonical Phase 4 DataFrame.

    - Accepts Polars DataFrame, Pandas DataFrame, or filepath (Parquet/TSV/CSV/Directory).
    - Translates column aliases (e.g. s1_id -> source1_entity_id, dim_14_... -> name_bm25_score).
    - Downcasts numeric feature columns to float32 for memory efficiency.
    - Validates required contract columns.
    """
    from pathlib import Path

    # 1. Handle file paths / directories
    if isinstance(data, (str, Path)):
        p = Path(data)
        if p.is_dir():
            # Directory with chunks (e.g. train_buffer_chunk_*.parquet)
            chunk_files = sorted(p.glob("*.parquet"))
            if not chunk_files:
                raise FileNotFoundError(f"No parquet chunks found in directory: {p}")
            dfs = [pd.read_parquet(f) for f in chunk_files]
            df = pd.concat(dfs, ignore_index=True)
        elif p.suffix == ".parquet":
            df = pd.read_parquet(p)
        elif p.suffix in (".tsv", ".txt"):
            df = pd.read_csv(p, sep="\t")
        elif p.suffix == ".csv":
            df = pd.read_csv(p)
        else:
            raise ValueError(f"Unsupported file format: {p}")
    # 2. Handle Polars DataFrame
    elif hasattr(data, "to_pandas"):
        try:
            df = data.to_pandas()
        except Exception:
            # Fallback if pyarrow is not installed for Polars .to_pandas()
            df = pd.DataFrame(data.to_dict(as_series=False))
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Unsupported input data type: {type(data)}")

    # 3. Rename known column aliases
    rename_map = {col: COLUMN_ALIASES[col] for col in df.columns if col in COLUMN_ALIASES and col != COLUMN_ALIASES[col]}
    if rename_map:
        df = df.rename(columns=rename_map)

    # 4. Verify required columns
    required = ID_COLS + ["label"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Input DataFrame is missing required columns: {missing}.\n"
            f"Available columns: {list(df.columns)}"
        )

    # 5. Downcast numeric columns to float32
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype(np.float32)

    df["label"] = df["label"].astype(int)
    df[COUNTRY_COL[0]] = df[COUNTRY_COL[0]].astype(str)

    return df


def build_preprocessor(
    cv: int = config.TARGET_ENCODER_CV,
    random_state: int = config.OPTUNA_SEED,
) -> ColumnTransformer:
    """Build ColumnTransformer for feature preprocessing.

    Decision 1: Numerical passthrough MUST be declared FIRST and TargetEncoder
    SECOND so that the output matrix has numerical columns in positions 0..24
    and encoded country in position 25, correctly aligning with MONOTONE_VECTOR.
    """
    return ColumnTransformer(
        transformers=[
            ("passthrough", "passthrough", NUMERICAL_COLS),
            (
                "target_enc",
                TargetEncoder(
                    cv=cv,
                    target_type="continuous",
                    random_state=random_state,
                ),
                COUNTRY_COL,
            ),
        ],
        remainder="drop",
    )


def manual_encode_for_eval(
    X_train: pd.DataFrame,
    y_train: np.ndarray | pd.Series,
    X_val: pd.DataFrame,
    cv: int = config.TARGET_ENCODER_CV,
    random_state: int = config.OPTUNA_SEED,
) -> np.ndarray:
    """Manually encode validation set for HPO pruning callback.

    Decision 2: Mirrors the exact column order of the fitted preprocessor:
    [NUMERICAL_COLS, target_enc(country)].
    Decision 11: Uses the same fixed random seed.
    """
    enc = TargetEncoder(
        cv=cv,
        target_type="continuous",
        random_state=random_state,
    )
    # Fit target encoder on training country and transform validation country
    enc.fit(X_train[COUNTRY_COL], y_train)
    val_country_enc = enc.transform(X_val[COUNTRY_COL])

    val_num = X_val[NUMERICAL_COLS].to_numpy(dtype=np.float32)
    val_country_enc = np.asarray(val_country_enc, dtype=np.float32).reshape(-1, 1)

    return np.hstack([val_num, val_country_enc])


def create_classifier(
    model_family: str = config.MODEL_FAMILY,
    scale_pos_weight: float = 1.0,
    random_state: int = config.OPTUNA_SEED,
    **kwargs: Any,
) -> XGBClassifier:
    """Construct classifier based on requested model family.

    Decision 8: Primary path (XGBoost) implemented; fallback path (LightGBM)
    present but deliberately raises NotImplementedError.
    Decision 5: eval_metric='logloss' configured as a constant setting.
    Decision 12: Classifier randomness pinned to fixed random seed.
    """
    family = model_family.lower()
    if family == "xgboost":
        device = kwargs.pop("device", config.XGB_DEVICE)
        tree_method = kwargs.pop("tree_method", config.XGB_TREE_METHOD)
        params: dict[str, Any] = {
            "monotone_constraints": MONOTONE_VECTOR,
            "scale_pos_weight": scale_pos_weight,
            "tree_method": tree_method,
            "device": device,
            "eval_metric": "logloss",
            "random_state": random_state,
        }
        params.update(kwargs)
        try:
            return XGBClassifier(**params)
        except Exception:
            if device == "cuda":
                params["device"] = "cpu"
                return XGBClassifier(**params)
            raise
    elif family == "lightgbm":
        raise NotImplementedError(
            "LightGBM model family fallback is currently unimplemented per Decision 8."
        )
    else:
        raise ValueError(f"Unsupported model family: '{model_family}'. Supported: 'xgboost'.")


def build_pipeline(
    model_family: str = config.MODEL_FAMILY,
    scale_pos_weight: float = 1.0,
    cv: int = config.TARGET_ENCODER_CV,
    random_state: int = config.OPTUNA_SEED,
    **classifier_kwargs: Any,
) -> Pipeline:
    """Construct full scikit-learn Pipeline with preprocessor and classifier."""
    preprocessor = build_preprocessor(cv=cv, random_state=random_state)
    classifier = create_classifier(
        model_family=model_family,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        **classifier_kwargs,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
