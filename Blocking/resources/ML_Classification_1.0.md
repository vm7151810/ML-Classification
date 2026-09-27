# Phase 4: ML Classification (v2.0)

## Goal
Take the 26-dimensional numerical feature vectors produced in Phase 3 and train an XGBoost classification model to predict whether a candidate S2/S3 entity is a true match for the S1 entity (`1`) or a non-match (`0`). The model must operate seamlessly within the 3-Stream Cyclic DAG, adhering strictly to `.env` configuration and Optuna-driven hyperparameter tuning.

---

## Section 0: Configuration Architecture

All tunable numeric parameters and constants are managed via a `.env` file to separate configuration from code and enable rapid experimentation.

**`.env` Schema (canonical — these are the exact key names used by `config.py` and all downstream modules):**
```dotenv
# ── Cross-Validation ─────────────────────────────────────────
CV_FOLDS=5                        # GroupKFold k (used in both Optuna objective and Phase 2 OOF pass)

# ── Threshold Search (Phase 3 of trainer.py) ─────────────────
THRESHOLD_SEARCH_START=0.50
THRESHOLD_SEARCH_END=0.99
THRESHOLD_SEARCH_STEP=0.01
FBETA_SCORE_BETA=0.5              # Beta for fbeta_score (0.5 = F_0.5, precision-heavy)

# ── Optuna HPO ───────────────────────────────────────────────
OPTUNA_N_TRIALS=100               # Number of Optuna trials
OPTUNA_TIMEOUT_SECONDS=3600       # Wall-clock cap in seconds (0 = no cap)
OPTUNA_SEED=42

# ── XGBoost Training ─────────────────────────────────────────
EARLY_STOPPING_ROUNDS=50
XGB_MAX_N_ESTIMATORS=5000         # Fixed upper bound; early stopping determines actual tree count
XGB_TREE_METHOD=hist              # Tree algorithm: hist (required for GPU/CPU)
XGB_DEVICE=cpu                    # Execution device: cpu or cuda
TARGET_ENCODER_CV=5               # Internal CV folds inside TargetEncoder (anti-leakage)

# ── Blocking Tuning ──────────────────────────────────────────
CANDIDATES_PER_STREAM=20          # k per stream per cycle
MAX_SEEN_PER_ENTITY=90            # Global stopping ceiling
MIN_CYCLES_BEFORE_STOP=3          # Min cycles before 0-match stop

# ── Paths ────────────────────────────────────────────────────
MODEL_OUTPUT_PATH=models/xgb_model.pkl
OUTPUT_DIR=output/
```

> [!WARNING]
> **Never commit `.env` to git or the submission zip.** It contains experiment-specific values that will differ between machines. Instead, commit a `.env.example` file (copy of the above schema with default values) and document in `README.md` that the user must run `cp .env.example .env` before executing the pipeline. The `.env` file must be listed in `.gitignore`.

**Implementation Rule:** `config.py` is the **single loader** for all `.env` values using `python-dotenv`. No other file is permitted to call `os.getenv()` directly. All modules must import variables from `config.py`.

```python
# config.py — canonical loader (partial)
from dotenv import load_dotenv
import os
load_dotenv()

CV_FOLDS                = int(os.getenv("CV_FOLDS", 5))
THRESHOLD_SEARCH_START  = float(os.getenv("THRESHOLD_SEARCH_START", 0.50))
THRESHOLD_SEARCH_END    = float(os.getenv("THRESHOLD_SEARCH_END", 0.99))
THRESHOLD_SEARCH_STEP   = float(os.getenv("THRESHOLD_SEARCH_STEP", 0.01))
FBETA_SCORE_BETA        = float(os.getenv("FBETA_SCORE_BETA", 0.5))
OPTUNA_N_TRIALS         = int(os.getenv("OPTUNA_N_TRIALS", 100))
OPTUNA_TIMEOUT_SECONDS  = float(os.getenv("OPTUNA_TIMEOUT_SECONDS", 0)) or None
OPTUNA_SEED             = int(os.getenv("OPTUNA_SEED", 42))
EARLY_STOPPING_ROUNDS   = int(os.getenv("EARLY_STOPPING_ROUNDS", 50))
XGB_MAX_N_ESTIMATORS    = int(os.getenv("XGB_MAX_N_ESTIMATORS", 5000))
XGB_TREE_METHOD         = os.getenv("XGB_TREE_METHOD", "hist")
XGB_DEVICE              = os.getenv("XGB_DEVICE", "cpu")
TARGET_ENCODER_CV       = int(os.getenv("TARGET_ENCODER_CV", 5))
CANDIDATES_PER_STREAM   = int(os.getenv("CANDIDATES_PER_STREAM", 20))
MAX_SEEN_PER_ENTITY     = int(os.getenv("MAX_SEEN_PER_ENTITY", 90))
MIN_CYCLES_BEFORE_STOP  = int(os.getenv("MIN_CYCLES_BEFORE_STOP", 3))
MODEL_OUTPUT_PATH       = os.getenv("MODEL_OUTPUT_PATH", "models/xgb_model.pkl")
OUTPUT_DIR              = os.getenv("OUTPUT_DIR", "output/")
```

---

## Section 1: Training Data Preparation & Class Imbalance

**Label Assignment:**
When the Cyclic DAG is run in **train mode**, it operates for exactly 3 cycles per entity (by design, `new_matches_this_cycle` remains 0 because inference is bypassed). After the DAG completes, the accumulated `(s1_id, cand_id, feature_vector)` buffer is left-joined against `train_ground_truth.tsv`.
- If the pair exists in the ground truth, assign label `y=1`.
- Otherwise, assign label `y=0`.

**Class Imbalance Handling:**
The negative class (non-matches) will heavily outnumber the positive class. The model compensates using the `scale_pos_weight` parameter:
- `scale_pos_weight = count(negatives) / count(positives)`
- This is an **exact auto-computed formula** based on the final labeled dataset, NOT a tuned hyperparameter.

---

## Section 2: Model Selection & Hard Negatives

**Model Priority:**
- **Primary:** XGBoost Classifier (tree method configurable via `.env`)
- **Fallback:** LightGBM

**Feature Space:**
The model consumes strictly **26-dimensional feature vectors** (23 core + 3 meta-features) as defined in `Feature_Engineering_1.0.md` (v7.0).

**Hard Negatives:**
Hard negative mining arises **naturally** from the BM25/FAISS blocking stage. Because the blocking systems retrieve the mathematically closest non-matching records, the model is automatically trained on high-quality hard negatives. No supplementary mining step is required.

---

## Section 3: Cross-Validation & Target Encoding Leakage Guard

**Cross-Validation:**
- Use `GroupKFold(n_splits=config.CV_FOLDS)` grouping by `source1_entity_id`. This guarantees that all candidates for a given S1 entity remain strictly in either the train or validation fold, preventing data leakage.

**Country Target Encoding (Dim 26) — Pipeline Ownership:**
- `feature_extractor.py` outputs Dim 26 as a **raw country string** (e.g., `"us"`, `"india"`). It has no access to target labels `y` and cannot apply `TargetEncoder`.
- `trainer.py` wraps the entire model in a sklearn **`Pipeline`**:

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder

NUMERICAL_COLS = FEATURE_COLS[:25]   # Dims 1–25 (float)
COUNTRY_COL    = ["country"]          # Dim 26 (raw string)

preprocessor = ColumnTransformer([
    ("target_enc", TargetEncoder(cv=config.TARGET_ENCODER_CV,
                                  target_type="continuous"), COUNTRY_COL),
    ("passthrough", "passthrough", NUMERICAL_COLS),
], remainder="drop")

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   XGBClassifier(
                         monotone_constraints=MONOTONE_VECTOR,
                         scale_pos_weight=scale_pos_weight,
                         **best_params)),
])
```

- `Pipeline.fit(X_train_fold, y_train_fold)` automatically ensures `TargetEncoder` only sees training-fold labels — **zero leakage by construction**.
- At inference time, `pipeline.predict_proba(X)` encodes the `country` column automatically. France (unseen country) receives the global mean fallback from `TargetEncoder(target_type='continuous')`.

---

## Section 4: Monotonic Constraints

To leverage domain knowledge, we apply monotonic constraints. A constraint of `+1` means that as the feature value increases (e.g., higher similarity), the probability of a match must monotonically increase or stay flat. A constraint of `0` means no constraint.

**MONOTONE_VECTOR is a hardcoded architectural constant** (NOT in `.env`).

| Dim | Feature | Constraint |
| :--- | :--- | :--- |
| 1 | Name Jaro-Winkler | `+1` |
| 2 | Name Monge-Elkan (Symmetric) | `+1` |
| 3 | Name Levenshtein Similarity | `+1` |
| 4 | Exact Name Match | `+1` |
| 5 | Name Phonetic Match | `+1` |
| 6 | Acronym Score | `+1` |
| 7 | Name Semantic Cosine | `+1` |
| 8 | Addr Token Jaccard | `+1` |
| 9 | Addr Token Containment | `+1` |
| 10 | Exact Address Match | `+1` |
| 11 | Addr Numeric Jaccard | `+1` |
| 12 | Addr Numeric Exact (Ternary: -1/0/1) | `0` (non-monotone ordinal) |
| 13 | Addr Numeric Exists Both | `0` (binary flag, not similarity) |
| 14 | Name BM25 Score | `+1` |
| 15 | Addr BM25 Score | `+1` |
| 16 | RRF Score (3-stream) | `+1` |
| 17 | Stream Overlap Count | `+1` |
| 18 | Name Length Ratio | `0` (both very short and very long names can match) |
| 19 | Addr Length Ratio | `0` (same reason) |
| 20 | S1 Name Is Missing | `0` |
| 21 | Cand Name Is Missing | `0` |
| 22 | S1 Addr Is Missing | `0` |
| 23 | Cand Addr Is Missing | `0` |
| 24 | Cross-Script Target | `0` |
| 25 | Source Origin (S2=0, S3=1) | `0` |
| 26 | Country Target Encoded | `0` |

*The exact vector passed to the model is:* `(1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0)`

---

## Section 5: Optuna HPO Loop

Hyperparameter tuning is fully automated using an Optuna TPE loop, evaluated over the GroupKFold cross-validation splits.

**Study Configuration:**
- `direction='maximize'`
- `sampler=TPESampler(seed=config.OPTUNA_SEED)`
- `n_trials=config.OPTUNA_N_TRIALS`
- `timeout=config.OPTUNA_TIMEOUT_SECONDS`

**Objective uses `config.CV_FOLDS`** (same key as Phase 2 OOF pass — guarantees the same fold structure is used across both phases for reproducibility).

**Search Space:**
- `n_estimators`: `[300, 3000]`
- `max_depth`: `[3, 9]`
- `learning_rate`: `[1e-3, 0.3]` (log scale)
- `subsample`: `[0.5, 1.0]`
- `colsample_bytree`: `[0.5, 1.0]`
- `min_child_weight`: `[1, 20]`
- `reg_alpha`: `[1e-8, 10.0]` (log scale)
- `reg_lambda`: `[1e-8, 10.0]` (log scale)

> [!IMPORTANT]
> **Pipeline Parameter Naming:** Because the model is a sklearn `Pipeline`, Optuna must use the `classifier__` prefix when suggesting XGBoost parameters:
> ```python
> params = {
>     "classifier__n_estimators"    : trial.suggest_int("n_estimators", 300, 3000),
>     "classifier__max_depth"       : trial.suggest_int("max_depth", 3, 9),
>     "classifier__learning_rate"   : trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
>     "classifier__subsample"       : trial.suggest_float("subsample", 0.5, 1.0),
>     "classifier__colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
>     "classifier__min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
>     "classifier__reg_alpha"       : trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
>     "classifier__reg_lambda"      : trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
> }
> pipeline.set_params(**params)
> pipeline.fit(X_train_fold, y_train_fold)
> ```

**Objective & Callbacks:**
- The objective function must return the **mean entity-level F₀.₅ score** across all CV folds (using `entity_macro_f05()`), directly optimizing the competition metric.
- **NO `early_stopping_rounds` in Optuna trials.** Because `n_estimators` is in the search space, Optuna controls tree count directly. Using early stopping would override Optuna's suggestion, creating an inconsistency between evaluated and saved params.
- Instead, use **`XGBoostPruningCallback`** to prune clearly bad trials mid-run based on `logloss` on the validation fold:

```python
from optuna.integration import XGBoostPruningCallback

# Inside each GroupKFold fold, within the Optuna objective:
# 1. Apply TargetEncoder on train fold manually to produce eval_set for pruning
enc = TargetEncoder(cv=config.TARGET_ENCODER_CV, target_type="continuous")
X_train_enc = enc.fit_transform(X_train_fold, y_train_fold)
X_val_enc   = enc.transform(X_val_fold)

# 2. Set params and fit with pruning callback
pipeline.set_params(**params)
pipeline.fit(
    X_train_fold, y_train_fold,
    classifier__eval_set=[(X_val_enc, y_val_fold)],
    classifier__callbacks=[XGBoostPruningCallback(trial, "validation_0-logloss")],
    classifier__verbose=False,
)
# 3. Optuna objective returns entity-level F0.5 (not logloss)
oof_probs_fold = pipeline.predict_proba(X_val_fold)[:, 1]
fold_score = entity_macro_f05(y_val_fold, oof_probs_fold, s1_ids_val, threshold=0.5)
```

> **Why two metrics?** `logloss` is used only by `XGBoostPruningCallback` to prune bad trials early — it is a strictly proper scoring rule that strongly correlates with downstream F₀.₅. The Optuna objective returns F₀.₅ computed from `predict_proba()` output. These serve entirely different purposes and there is no conflict.

> [!IMPORTANT]
> Optuna trials must **NOT** cache or return OOF predictions. Each trial's objective function only computes and returns the mean F₀.₅ score across folds. OOF predictions are NOT stored per trial. This is an explicit memory guard.

---

## Section 6: Threshold Calibration & Model Persistence

### Calibration Architecture — Option B (Industry Standard)

Threshold calibration uses a clean, separate GroupKFold pass after Optuna completes — not the Optuna trial OOF. This is the standard industry pattern and eliminates the circular feedback bias where the same OOF predictions that selected hyperparams are also used to calibrate the threshold.

**4-Phase `trainer.py` Execution Flow:**

```
Phase 1 — Optuna Study:
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=config.OPTUNA_SEED))
    study.optimize(objective, n_trials=config.OPTUNA_N_TRIALS, timeout=config.OPTUNA_TIMEOUT_SECONDS)
    best_params = study.best_params
    → best_params (dict, all XGBoost params including n_estimators, with 'classifier__' prefix)

Phase 2 — Clean OOF Generation (dedicated pass):
    Re-run GroupKFold(n_splits=config.CV_FOLDS) with FIXED best_params.
    GroupKFold has NO random_state — do NOT add one. It is deterministic
    given consistent DataFrame row order from Phase 0 parquet files.
    For each fold:
        pipeline.set_params(**best_params)
        pipeline.fit(X_train_fold, y_train_fold)      # NO early stopping
        Append pipeline.predict_proba(X_val_fold)[:, 1] to oof_probs[:]
        Append s1_ids from val_fold indices to oof_s1_ids[:]
    → oof_probs   (np.array, shape: [n_train_pairs])   # float probabilities
      oof_labels  (np.array, shape: [n_train_pairs])   # 0/1 ground truth
      oof_s1_ids  (list,     shape: [n_train_pairs])   # required by entity_macro_f05()

Phase 3 — Threshold Calibration:
    Grid search t in [config.THRESHOLD_SEARCH_START, config.THRESHOLD_SEARCH_END]
                                                      step=config.THRESHOLD_SEARCH_STEP
    For each t:
        preds = (oof_probs > t).astype(int)
        # BUG FIX: average='macro' gives equal weight to class-0 F0.5 (trivially ~1.0)
        # and class-1 F0.5, masking the real signal. Use entity-level macro instead.
        score = entity_macro_f05(oof_labels, oof_probs, oof_s1_ids, threshold=t)
    optimal_threshold = t at max(score)
    → optimal_threshold (float)

def entity_macro_f05(oof_labels, oof_probs, oof_s1_ids, threshold):
    """Exact competition metric: per-entity F0.5, then macro-average across entities."""
    from collections import defaultdict
    groups = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for s1_id, label, prob in zip(oof_s1_ids, oof_labels, oof_probs):
        groups[s1_id]["y_true"].append(label)
        groups[s1_id]["y_pred"].append(int(prob > threshold))
    entity_scores = []
    for entry in groups.values():
        y_true, y_pred = entry["y_true"], entry["y_pred"]
        tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred))
        fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred))
        fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred))
        # Singleton: no true matches, no predictions → perfect 1.0
        if sum(y_true) == 0 and sum(y_pred) == 0:
            entity_scores.append(1.0)
        # Singleton: no true matches, but predicted something → 0.0
        elif sum(y_true) == 0 and sum(y_pred) > 0:
            entity_scores.append(0.0)
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            denom = 0.25 * precision + recall
            entity_scores.append((1.25 * precision * recall) / denom if denom > 0 else 0.0)
    return sum(entity_scores) / len(entity_scores)

Phase 4 — Final Fit on 100% Data:
    scale_pos_weight = count(y_all == 0) / count(y_all == 1)
    # Strip 'classifier__' prefix to get raw XGBoost params for final pipeline
    xgb_params = {k.replace('classifier__', ''): v for k, v in best_params.items()}
    final_pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("target_enc", TargetEncoder(cv=config.TARGET_ENCODER_CV,
                                          target_type="continuous"), ["country"]),
            ("passthrough", "passthrough", NUMERICAL_COLS),
        ], remainder="drop")),
        ("classifier", XGBClassifier(
            monotone_constraints=MONOTONE_VECTOR,
            scale_pos_weight=scale_pos_weight,
            **xgb_params
        )),
    ])
    # NO early stopping — fit on 100% data. TargetEncoder fitted on full y_all.
    final_pipeline.fit(X_all, y_all)
    joblib.dump({
        'model'         : final_pipeline,   # Pipeline: TargetEncoder + XGBClassifier
        'threshold'     : optimal_threshold,
        'best_params'   : best_params,
        'feature_names' : FEATURE_COLS,     # 25 float cols + 'country' (raw string)
    }, config.MODEL_OUTPUT_PATH)
```

> [!NOTE]
> **Why Phase 2 is not redundant with Phase 1:** Phase 1 Optuna trials each run on a different hyperparameter combination and use `XGBoostPruningCallback` for trial pruning — the OOF predictions from any single trial reflect suboptimal params. Phase 2 runs one dedicated clean pass using only the single best hyperparameter set from `best_params`, producing OOF predictions that are uncontaminated by the HPO process and on which threshold calibration (Phase 3) is performed.

> [!WARNING]
> **Column ordering:** `FEATURE_COLS` is the authoritative 26-column ordered list defined below. It must be explicitly passed when constructing feature matrices. At inference time, `orchestrator.py` must construct the feature matrix with columns in exactly this order before calling `model.predict_proba()`.

```python
# The authoritative feature order required for trainer.py and orchestrator.py:
FEATURE_COLS = [
    "name_jaro_winkler", "name_monge_elkan", "name_levenshtein",
    "exact_name_match", "name_phonetic_match", "acronym_score",
    "name_semantic_cosine", "addr_token_jaccard", "addr_token_containment",
    "exact_addr_match", "addr_numeric_jaccard", "addr_numeric_exact",
    "addr_numeric_exists_both", "name_bm25_score", "addr_bm25_score",
    "rrf_score", "stream_overlap_count", "name_length_ratio",
    "addr_length_ratio", "s1_name_missing", "cand_name_missing",
    "s1_addr_missing", "cand_addr_missing", "cross_script_target",
    "source_origin", "country"   # Dim 26: raw string — TargetEncoder applied by Pipeline
]
```

---

## Section 7: Online Inference (Integration with `orchestrator.py`)

At inference time, the model is integrated directly into the Cyclic DAG via `pipeline/orchestrator.py`.

**Inference Flow (mode == `infer` only):**
1. **Model Load:** At startup, load `config.MODEL_OUTPUT_PATH` via `joblib.load()`. Verify that the artifact's `feature_names` list matches the 26-column order of the live feature matrix before any inference begins. Crash loudly if there is a mismatch.
2. **Batch Size:** On each cycle, the orchestrator passes a union of up to **60 unique candidates** (3 streams × `config.CANDIDATES_PER_STREAM`, minus blacklisted entities) to the ML model.
3. **Scoring:** The model evaluates the 26-dim features: `probs = model.predict_proba(features)[:, 1]`
4. **Acceptance:** If `prob > threshold`, the candidate is marked as `accepted=True` inside the global `seen_dict`.
   - **NO Bipartite Post-Processing:** The problem statement permits S1 → many matches. Greedy 1:1 bipartite resolution would drop valid true matches and is explicitly prohibited.
   - **NO Global Probability Cache:** `seen_dict` is the sole accumulator. We do not maintain a separate `potential_matches` cache.
5. **Output (infer mode only):** After ALL entities across ALL countries are processed, call `write_all_outputs(seen_dict)` exactly once to flush `matching_results.tsv`, `candidate_pairs.tsv`, and `candidate_pairs_per_cycle.tsv`. This function is **NOT called in train mode** — in train mode, the post-loop step is label assignment + `trainer.py` invocation (see `Blocking_1.0.md` Section 5).

---

## Requirements Additions

Add the following to `requirements.txt`:
```
xgboost>=2.0.0
lightgbm>=4.3.0
optuna>=3.5.0
python-dotenv>=1.0.0
joblib>=1.3.0
scikit-learn>=1.4.0   # for GroupKFold, TargetEncoder, fbeta_score
```

> [!NOTE]
> `scikit-learn>=1.4.0` is required because `TargetEncoder` was added in v1.3 but its `target_type='continuous'` parameter (required for the global mean fallback for France) was stabilized in v1.4. Pin accordingly.
