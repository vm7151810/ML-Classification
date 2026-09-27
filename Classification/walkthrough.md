# Walkthrough: Phase 4 — ML Classification Implementation

We have completed the implementation and automated verification of **Phase 4 (ML Classification)** of the Business Entity Resolution pipeline.

All 15 architectural decisions in [02_phase4_decisions.md](file:///home/mivikev/Desktop/AMC/Classification/.agent/Phase4/02_phase4_decisions.md), the "Phase 4: ML Classification (v2.0)" design document, and the canonical 26-dimension feature table have been realized in `Classification/`.

---

## Changes Made

### 1. Project Dependencies & Configuration
- **[`requirements.txt`](file:///home/mivikev/Desktop/AMC/Classification/requirements.txt)**: Pinned dependencies including `xgboost>=2.0.0`, `lightgbm>=4.3.0`, `optuna>=3.5.0`, `optuna-integration>=3.5.0`, `scikit-learn>=1.4.0`, `joblib`, `pandas`, `numpy`, `python-dotenv`, and `pytest`.
- **[`.env.example`](file:///home/mivikev/Desktop/AMC/Classification/.env.example)**: Canonical environment schema with standard defaults for CV folds, Optuna trials, threshold sweep ranges, and file paths.
- **[`src/config.py`](file:///home/mivikev/Desktop/AMC/Classification/src/config.py)**: Single configuration loader ensuring no other module calls `os.getenv()` directly.

### 2. Core Modules
- **[`src/metrics.py`](file:///home/mivikev/Desktop/AMC/Classification/src/metrics.py)**:
  - `compute_scale_pos_weight()`: Calculates $\frac{N_{\text{neg}}}{N_{\text{pos}}}$ on the complete dataset; raises `ValueError` if zero positives exist.
  - `entity_macro_fbeta()`: Evaluates per-entity precision, recall, and weighted $F_\beta$ (strict inequality $\text{prob} > \text{threshold}$), awarding $1.0$ to correctly empty singletons, $0.0$ to false merges on singletons, and averaging unweighted across all entities.
- **[`src/model.py`](file:///home/mivikev/Desktop/AMC/Classification/src/model.py)**:
  - Canonical 26-dimension feature list: 25 numeric features + 1 raw `country` string.
  - 26-dimension `MONOTONE_VECTOR`: `(1,1,1,1,1,1,1,1,1,1,1,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0)`.
  - **Decision 1 Fix**: `build_preprocessor()` explicitly declares the 25 numeric passthrough features *first* and `TargetEncoder` *second*, ensuring output columns $0..24$ are numeric and column $25$ is encoded country.
  - **Decision 2 Fix**: `manual_encode_for_eval()` mirrors this exact column alignment for Optuna pruning.
  - **Decision 8 Guard**: `create_classifier()` instantiates `XGBClassifier` with pinned seeds and constant `eval_metric="logloss"`; raises `NotImplementedError` if `lightgbm` is requested.
  - Full scikit-learn `Pipeline` composition.
- **[`src/logging_utils.py`](file:///home/mivikev/Desktop/AMC/Classification/src/logging_utils.py)**:
  - Dual console and file logger (`logs/phase4_train_<timestamp>.log`).
  - Structured `RunSummary` builder exporting full run metrics to `metrics/phase4/run_summary_<timestamp>.json`.
- **[`src/train.py`](file:///home/mivikev/Desktop/AMC/Classification/src/train.py)**:
  - **Phase A**: GroupKFold cross-validation on `source1_entity_id`, Optuna hyperparameter optimization with `classifier__` prefixed parameters, and `XGBoostPruningCallback` on validation `logloss`. Evaluates trials at fixed threshold $0.50$.
  - **Phase B**: Clean out-of-fold cross-validation pass using fixed winning hyperparameters without early stopping.
  - **Phase C**: Full threshold sweep ($t \in [0.50, 0.99]$, step $0.01$) over clean OOF probabilities to determine `optimal_threshold`.
  - **Phase D**: Final fit on 100% data, `scale_pos_weight` consistency verification, and artifact serialization containing `model`, `threshold`, `best_params`, and `feature_names`.

---

## Verification Results

### Automated Test Suite (`tests/`)
All 16 unit, regression, integration, and smoke tests were executed using `pytest`:

```text
tests/test_integration.py::test_pipeline_end_to_end_with_unseen_country PASSED [  6%]
tests/test_integration.py::test_no_target_encoding_leakage PASSED        [ 12%]
tests/test_integration.py::test_persistence_round_trip PASSED            [ 18%]
tests/test_integration.py::test_monotone_constraint_behavioral PASSED    [ 25%]
tests/test_integration.py::test_fallback_model_family_raises PASSED      [ 31%]
tests/test_metrics.py::test_scale_pos_weight_known_ratio PASSED          [ 37%]
tests/test_metrics.py::test_scale_pos_weight_zero_positives_raises PASSED [ 43%]
tests/test_metrics.py::test_entity_macro_fbeta_singleton_correct PASSED  [ 50%]
tests/test_metrics.py::test_entity_macro_fbeta_singleton_false_merge PASSED [ 56%]
tests/test_metrics.py::test_entity_macro_fbeta_hand_computed_mixed_group PASSED [ 62%]
tests/test_metrics.py::test_entity_macro_fbeta_strict_inequality PASSED  [ 68%]
tests/test_metrics.py::test_entity_macro_fbeta_macro_average PASSED      [ 75%]
tests/test_metrics.py::test_entity_macro_fbeta_alternate_beta PASSED     [ 81%]
tests/test_preprocessor_regression.py::test_preprocessor_column_order_regression PASSED [ 87%]
tests/test_preprocessor_regression.py::test_monotone_vector_alignment_contract PASSED [ 93%]
tests/test_smoke.py::test_smoke_end_to_end_training PASSED               [100%]

============================== 16 passed in 4.66s ==============================
```

### Key Decision Verifications
1. **Decision 1 Regression Test (`test_preprocessor_column_order_regression`)**: Passed. Proves that `ColumnTransformer` outputs numeric features in columns $0..24$ and encoded country in column $25$.
2. **Behavioral Monotonic Constraint Test (`test_monotone_constraint_behavioral`)**: Passed. Proves that increasing a positively-constrained feature (`name_jaro_winkler`) never decreases the predicted probability.
3. **Unseen Country Fallback (`test_pipeline_end_to_end_with_unseen_country`)**: Passed. Evaluates records with `"france"` without `NaN`s, using `TargetEncoder(target_type='continuous')` global mean fallback.
4. **Smoke End-to-End Pipeline (`test_smoke_end_to_end_training`)**: Passed. Verifies that all 4 phases run to completion, saving the model artifact, structured logs, and complete JSON summary.
