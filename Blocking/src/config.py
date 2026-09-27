import os
from pathlib import Path
from dotenv import load_dotenv

# Automatically load the .env file from the src directory
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

def get_env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default)

def get_env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_float(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default

def get_env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key, str(default)).lower()
    return val in ('true', '1', 't', 'y', 'yes')

def get_env_list(key: str, default: list = None) -> list:
    val = os.getenv(key, "")
    if not val:
        return default if default is not None else []
    return [item.strip() for item in val.split(',')]


class Config:
    # ── SYSTEM ──────────────────────────────────────────
    RANDOM_SEED = get_env_int("RANDOM_SEED", 42)
    PIPELINE_MODE = get_env_str("PIPELINE_MODE", "infer")
    
    # ── PATHS ───────────────────────────────────────────
    DATA_DIR_TRAIN = get_env_str("DATA_DIR_TRAIN", "dataset/train")
    DATA_DIR_TEST = get_env_str("DATA_DIR_TEST", "dataset/test")
    GROUND_TRUTH_PATH = get_env_str("GROUND_TRUTH_PATH", "dataset/train/train_ground_truth.tsv")
    PROCESSED_DATA_DIR = get_env_str("PROCESSED_DATA_DIR", "data/processed")
    MODEL_OUTPUT_PATH = get_env_str("MODEL_OUTPUT_PATH", "models/xgb_model.pkl")
    OUTPUT_DIR = get_env_str("OUTPUT_DIR", "output/")

    # ── PHASE 0: PREPROCESSING ──────────────────────────
    NULL_LITERALS = get_env_list("NULL_LITERALS", ["n/a", "na", "null", "none", "-"])
    NULLNAME_PREFIX = get_env_str("NULLNAME_PREFIX", "nullname")
    NULLADDR_PREFIX = get_env_str("NULLADDR_PREFIX", "nulladdr")
    ENABLE_TRANSLITERATION = get_env_bool("ENABLE_TRANSLITERATION", True)
    TRANSLITERATION_SCHEME = get_env_str("TRANSLITERATION_SCHEME", "itrans")
    SYMBOL_AT_REPLACEMENT = get_env_str("SYMBOL_AT_REPLACEMENT", " at ")
    SYMBOL_AMP_REPLACEMENT = get_env_str("SYMBOL_AMP_REPLACEMENT", " and ")
    PARQUET_COMPRESSION = get_env_str("PARQUET_COMPRESSION", "snappy")

    # ── PHASE 1-2: BLOCKING - FAISS ──────────────────────
    SENTENCE_TRANSFORMER_MODEL = get_env_str("SENTENCE_TRANSFORMER_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")
    FAISS_EMBED_BATCH_SIZE = get_env_int("FAISS_EMBED_BATCH_SIZE", 256)
    FAISS_INDEX_TYPE = get_env_str("FAISS_INDEX_TYPE", "IndexFlatIP")
    FAISS_IVF_NLIST = get_env_int("FAISS_IVF_NLIST", 1024)
    FAISS_IVF_NPROBE = get_env_int("FAISS_IVF_NPROBE", 64)
    FAISS_HNSW_M = get_env_int("FAISS_HNSW_M", 32)
    FAISS_METRIC = get_env_str("FAISS_METRIC", "METRIC_INNER_PRODUCT")
    FAISS_OMP_NUM_THREADS = get_env_int("FAISS_OMP_NUM_THREADS", 1)

    # ── PHASE 1-2: BLOCKING - BM25 ───────────────────────
    BM25_NGRAM_SIZE = get_env_int("BM25_NGRAM_SIZE", 3)
    BM25_K1 = get_env_float("BM25_K1", 1.5)
    BM25_B = get_env_float("BM25_B", 0.75)
    BM25_EPSILON = get_env_float("BM25_EPSILON", 0.25)

    # ── PHASE 1-2: BLOCKING - ORCHESTRATOR ───────────────
    DAG_K_BASE = get_env_int("DAG_K_BASE", 20)
    DAG_MAX_SEEN = get_env_int("DAG_MAX_SEEN", 90)
    DAG_MIN_CYCLES = get_env_int("DAG_MIN_CYCLES", 3)

    # ── PHASE 1-2: BLOCKING - SEEN DICTIONARY ────────────
    BATCH_SIZE = get_env_int("BATCH_SIZE", 10000)

    # ── PHASE 3: FEATURE ENGINEERING ─────────────────────
    RRF_K_BM25_NAME = get_env_int("RRF_K_BM25_NAME", 60)
    RRF_K_BM25_ADDR = get_env_int("RRF_K_BM25_ADDR", 60)
    RRF_K_FAISS = get_env_int("RRF_K_FAISS", 60)
    BM25_SELF_SCORE_FLOOR = get_env_float("BM25_SELF_SCORE_FLOOR", 1e-9)
    JARO_WINKLER_PREFIX_WEIGHT = get_env_float("JARO_WINKLER_PREFIX_WEIGHT", 0.1)
    TARGET_ENCODER_CV = get_env_int("TARGET_ENCODER_CV", 5)
    COUNTRY_ENCODER_TYPE = get_env_str("COUNTRY_ENCODER_TYPE", "TargetEncoder")

    # ── PHASE 4: ML CLASSIFICATION ───────────────────────
    ML_BACKEND = get_env_str("ML_BACKEND", "xgboost")
    
    OPTUNA_N_TRIALS = get_env_int("OPTUNA_N_TRIALS", 100)
    OPTUNA_TIMEOUT_SECONDS = get_env_int("OPTUNA_TIMEOUT_SECONDS", 3600)
    OPTUNA_SEED = get_env_int("OPTUNA_SEED", 42)

    XGB_N_ESTIMATORS = get_env_int("XGB_N_ESTIMATORS", 1000)
    XGB_LEARNING_RATE = get_env_float("XGB_LEARNING_RATE", 0.05)
    XGB_MAX_DEPTH = get_env_int("XGB_MAX_DEPTH", 6)
    XGB_MIN_CHILD_WEIGHT = get_env_int("XGB_MIN_CHILD_WEIGHT", 5)
    XGB_SUBSAMPLE = get_env_float("XGB_SUBSAMPLE", 0.8)
    XGB_COLSAMPLE_BYTREE = get_env_float("XGB_COLSAMPLE_BYTREE", 0.8)
    XGB_REG_ALPHA = get_env_float("XGB_REG_ALPHA", 0.1)
    XGB_REG_LAMBDA = get_env_float("XGB_REG_LAMBDA", 1.0)
    XGB_NEGATIVE_SUBSAMPLE_RATIO = get_env_float("XGB_NEGATIVE_SUBSAMPLE_RATIO", 1.0)
    XGB_SCALE_POS_WEIGHT = get_env_float("XGB_SCALE_POS_WEIGHT", -1)
    XGB_EARLY_STOPPING_ROUNDS = get_env_int("XGB_EARLY_STOPPING_ROUNDS", 50)
    XGB_DEVICE = get_env_str("XGB_DEVICE", "cuda")
    XGB_EVAL_METRIC = get_env_str("XGB_EVAL_METRIC", "logloss")
    XGB_TREE_METHOD = get_env_str("XGB_TREE_METHOD", "hist")

    LGBM_N_ESTIMATORS = get_env_int("LGBM_N_ESTIMATORS", 1000)
    LGBM_LEARNING_RATE = get_env_float("LGBM_LEARNING_RATE", 0.05)
    LGBM_MAX_DEPTH = get_env_int("LGBM_MAX_DEPTH", 6)
    LGBM_NUM_LEAVES = get_env_int("LGBM_NUM_LEAVES", 63)
    LGBM_MIN_CHILD_SAMPLES = get_env_int("LGBM_MIN_CHILD_SAMPLES", 20)
    LGBM_SUBSAMPLE = get_env_float("LGBM_SUBSAMPLE", 0.8)
    LGBM_COLSAMPLE_BYTREE = get_env_float("LGBM_COLSAMPLE_BYTREE", 0.8)
    LGBM_REG_ALPHA = get_env_float("LGBM_REG_ALPHA", 0.1)
    LGBM_REG_LAMBDA = get_env_float("LGBM_REG_LAMBDA", 1.0)
    LGBM_SCALE_POS_WEIGHT = get_env_float("LGBM_SCALE_POS_WEIGHT", -1)
    LGBM_EARLY_STOPPING_ROUNDS = get_env_int("LGBM_EARLY_STOPPING_ROUNDS", 50)
    LGBM_DEVICE = get_env_str("LGBM_DEVICE", "cpu")
    LGBM_EVAL_METRIC = get_env_str("LGBM_EVAL_METRIC", "binary_logloss")

    CV_FOLDS = get_env_int("CV_FOLDS", 5)
    THRESHOLD_SEARCH_START = get_env_float("THRESHOLD_SEARCH_START", 0.50)
    THRESHOLD_SEARCH_END = get_env_float("THRESHOLD_SEARCH_END", 0.99)
    THRESHOLD_SEARCH_STEP = get_env_float("THRESHOLD_SEARCH_STEP", 0.01)
    FBETA_SCORE_BETA = get_env_float("FBETA_SCORE_BETA", 0.5)

config = Config()
