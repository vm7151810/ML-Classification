# Phase 5: Orchestrator Module v1.0 — `pipeline/orchestrator.py` (PATCHED v2.0)

> **Status:** PATCHED v2.0 (Bulletproofed)
> **Depends on:** `Preprocessing_1.0.md`, `Blocking_1.0.md`, `Feature_Engineering_1.0.md`, `ML_Classification_1.0.md`
> **Implements:** Implementation Plan v2.3, Section 8 (DAG) + Section 11 (Steps 6–9)

> [!WARNING]
> **ARCHITECTURAL OVERRIDES APPLIED**
> The logic in this document has been bulletproofed. You MUST apply the following patches during implementation:
> 1. **Seen Dictionary Self-Healing:** `update_seen()` must natively auto-initialize missing entities. Eradicate `UninitializedEntityError`.
> 2. **Zero-Match Finalization Defense:** `finalize_entity()` must also auto-initialize the entity if it doesn't exist, preventing fatal KeyErrors for 0-match S1 queries.
> 3. **Entity-Scoped Blacklist:** The "global rejection blacklist" is strictly **entity-scoped**. The filter `[x for x in candidates if x not in entity_seen_ids]` only applies to the current S1 entity.
> 4. **JSONL Disk-Backed State (Batch Flush):** The `seen_dict` must NOT hold the entire dataset in RAM. It must stream processed entities to a `.jsonl` file in batches (e.g., 10,000) and clear the RAM to prevent OOM.
> 5. **Deterministic Output Serialization:** When writing outputs, all `sets` must be explicitly sorted (`sorted(list(accepted))`) to prevent hash-randomization.
> 6. **Concurrency Jitter Prevention:** The `per_cycle` lists must also be explicitly sorted (by `candidate_id`) to prevent thread-scheduling race conditions in concurrent execution.
> 7. **SeenDict OOP Refactor:** `seen_dict` is no longer a collection of functional dict modifiers. It MUST be implemented as a stateful class (e.g., `SeenDict(run_dir: str)`).
> 8. **State File Location:** The `SeenDict` instance must natively manage its `.jsonl` state file within its initialized `run_dir` (no hardcoded paths).
> 9. **TSV Output Location:** `write_all_outputs()` no longer accepts `output_dir`; it relies on the internal `self.run_dir` for all TSV generation.
> 10. **Environment Isolation:** Read `BATCH_SIZE` natively via `int(os.environ.get("BATCH_SIZE", 10000))`. Do NOT introduce a `python-dotenv` dependency into the module.

---

## 0. Purpose & Scope

`pipeline/orchestrator.py` is the **single nerve centre** of the entire pipeline. It wires together every module produced in Steps 1–5 (`data_layer`, `blocking_layer`, `feature_layer`, `ml_layer`) and drives the 2-mode **Progressive Cyclic DAG** to completion. Nothing outside `main.py` should import or instantiate this module directly.

**What the Orchestrator does NOT do:**
- It does not preprocess raw TSVs (that is `data_layer/cleaner.py`'s job; parquets are pre-built before the orchestrator runs).
- It does not train the model or calibrate thresholds (that is `ml_layer/trainer.py`'s job).
- It does not compute features internally — it calls `feature_layer/feature_extractor.py`.
- It does not build indices internally — it calls the `blocking_layer/` helpers.

---

## 1. CLI Entry Point: `src/main.py`

`main.py` is a thin wrapper. It parses CLI arguments, resolves paths, and calls `run_orchestrator()`.

### 1A. Accepted CLI Arguments

| Argument | Required | Modes | Description |
|:---|:---|:---|:---|
| `--mode` | Always | `train` / `infer` | Selects pipeline mode. |
| `--data-dir` | Always | both | Path to the **raw** dataset directory (`dataset/train` or `dataset/test`). Used to locate ground truth in train mode. |
| `--processed-dir` | Always | both | Path to the directory containing pre-built `Query_*.parquet` and `Target_*.parquet` files (output of Phase 0). |
| `--ground-truth` | train only | `train` | Path to `train_ground_truth.tsv`. Required when `--mode train`. |
| `--model-out` | train only | `train` | Path to save the trained model artifact (`models/xgb_model.pkl`). |
| `--model-in` | infer only | `infer` | Path to the saved model artifact to load. |
| `--output-dir` | infer only | `infer` | Directory where the 3 output TSV files will be written. |

> [!IMPORTANT]
> All path arguments override their `config.py` equivalents. If an argument is omitted, `main.py` falls back to the corresponding `config.*` value. **No path is ever hardcoded inside `orchestrator.py` itself.**

### 1B. `main.py` Skeleton

```python
# src/main.py
import argparse
from pipeline.orchestrator import run_orchestrator

def parse_args():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--mode",          required=True,  choices=["train", "infer"])
    parser.add_argument("--data-dir",      required=False, default=None)
    parser.add_argument("--processed-dir", required=False, default=None)
    parser.add_argument("--ground-truth",  required=False, default=None)
    parser.add_argument("--model-out",     required=False, default=None)
    parser.add_argument("--model-in",      required=False, default=None)
    parser.add_argument("--output-dir",    required=False, default=None)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_orchestrator(args)
```

---

## 2. Module Imports & Dependencies

```python
# src/pipeline/orchestrator.py — top-level imports
import os
import glob
import logging
from pathlib import Path

import polars as pl
import numpy as np
import joblib

from config import config
from data_layer.loader import load_parquet
from blocking_layer.semantic_index import build_faiss_index, faiss_search
from blocking_layer.lexical_name_index import build_bm25_name_index, bm25_name_search
from blocking_layer.lexical_addr_index import build_bm25_addr_index, bm25_addr_search
from blocking_layer.seen_dict import (
    init_seen_dict, init_entity, update_seen, write_all_outputs
)
from feature_layer.feature_extractor import extract_features, FEATURE_COLS
from ml_layer.trainer import run_training
from ml_layer.classifier import predict_proba_batch
```

---

## 3. The Candidate Object Contract

Every candidate travelling through the Cyclic DAG is a **Python dict** with the following exact schema. This is the contract between the blocking layer, the union/sort logic, and the feature extractor.

```python
# Candidate object — produced by `priority_deduplicate()`, consumed by `extract_features()`
candidate = {
    # ── Identity ──────────────────────────────────────────────────────
    "entity_id"        : str,   # e.g. "S2-001234" or "S3-009876"

    # ── BM25-Name stream metadata ──────────────────────────────────────
    "bm25_name_score"  : float, # Raw BM25-Name score from rank-bm25 (0.0 if not retrieved)
    "bm25_name_rank"   : int,   # 1-indexed rank within BM25-Name results (k+1 if not retrieved)

    # ── BM25-Addr stream metadata ──────────────────────────────────────
    "bm25_addr_score"  : float, # Raw BM25-Addr score (0.0 if not retrieved)
    "bm25_addr_rank"   : int,   # 1-indexed rank (k+1 if not retrieved)

    # ── FAISS-Name stream metadata ─────────────────────────────────────
    "faiss_score"      : float, # Inner-product similarity after L2-norm (0.0 if not retrieved)
    "faiss_rank"       : int,   # 1-indexed rank (k+1 if not retrieved)

    # ── Derived (computed by priority_deduplicate) ────────────────────
    "stream_count"     : int,   # {1, 2, 3} — how many streams retrieved this candidate

    # ── Full entity row from Target parquet (populated post-merge) ────
    "row"              : dict,  # Full candidate row from Target_C parquet (all columns)
}
```

> [!IMPORTANT]
> **Sentinel rules:** If a candidate was NOT retrieved by a given stream, its score for that stream MUST be `0.0` and its rank MUST be `k + 1` (where `k` is the current cycle's k value). This ensures the RRF contribution from a missing stream is effectively zero. Do NOT use `None` or `-1`; downstream feature extractors assume numeric types.

---

## 4. Helper Function Specifications

### 4A. `discover_countries(processed_dir: str) -> list[str]`

Scans the processed data directory for `Query_*.parquet` files and extracts country names from filenames.

```python
def discover_countries(processed_dir: str) -> list[str]:
    """
    Scans `processed_dir` for files matching 'Query_*.parquet'.
    Extracts the country name from the filename stem (e.g., 'Query_india' → 'india').
    Returns a sorted list of country strings.
    Raises FileNotFoundError if no Query_*.parquet files are found.
    Handles France and any future unseen country automatically — zero code changes needed.
    """
    pattern = os.path.join(processed_dir, "Query_*.parquet")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Query_*.parquet files found in '{processed_dir}'. "
            "Run Phase 0 (cleaner.py) first."
        )
    countries = []
    for f in files:
        stem = Path(f).stem          # e.g. "Query_india"
        country = stem.split("_", 1)[1]  # e.g. "india"
        countries.append(country)
    return countries
```

### 4B. `to_char_ngrams(text: str, n: int) -> list[str]`

Converts a string to character n-grams (used for BM25 queries).

```python
def to_char_ngrams(text: str, n: int = 3) -> list[str]:
    """
    Converts `text` to a list of overlapping character n-grams.
    Example: to_char_ngrams("hello", 3) → ["hel", "ell", "llo"]
    If len(text) < n, returns [text] (whole string as single token — avoids empty list).
    """
    if len(text) < n:
        return [text]
    return [text[i:i+n] for i in range(len(text) - n + 1)]
```

### 4C. `priority_deduplicate(bm25_name_results, bm25_addr_results, faiss_results, k) -> list[dict]`

Merges the 3 filtered stream results into a single priority-sorted list of candidate objects.

**Algorithm:**

```
1. Build a merged dict keyed by entity_id.
   For each stream's results, enumerate 1-indexed:
     - If entity_id not yet in merged: insert with all sentinel defaults (score=0.0, rank=k+1, stream_count=0, row=None).
     - Update that stream's score/rank with actual values.
     - Increment stream_count.

2. Sort merged values: stream_count DESC, then bm25_name_rank ASC (tiebreak by name rank).

3. Return sorted list. Each dict already has all 9 fields EXCEPT 'row' (populated afterwards).
```

> [!NOTE]
> The `"row"` field is populated **after** this function returns, by looking up each `entity_id` in the `target_lookup` dict. `priority_deduplicate()` does NOT do any dataframe lookups itself.

### 4D. Seen Dict Functions (in `blocking_layer/seen_dict.py`)

```python
def init_seen_dict() -> dict:
    """Returns an empty dict. Called ONCE before the country loop. Never reset between countries."""
    return {}

def init_entity(seen_dict: dict, s1_id: str) -> None:
    """
    Idempotent. Inserts blank entry for s1_id if it does not already exist.
    MUST be called before any cycle logic for this entity.
    Guarantees the entity appears in all 3 output files even if 0 candidates are found.
    """
    if s1_id not in seen_dict:
        seen_dict[s1_id] = {
            "accepted" : set(),
            "rejected" : set(),
            "seen_ids" : set(),
            "total"    : 0,
            "per_cycle": {},
        }

def update_seen(seen_dict: dict, s1_id: str, cand_id: str,
                accepted: bool, cycle: int) -> None:
    """
    Updates seen_dict[s1_id] with the result for cand_id.
    - Adds cand_id to accepted OR rejected set.
    - Adds cand_id to seen_ids (blacklist).
    - Increments total.
    - Appends cand_id to per_cycle[cycle] list.
    Guard: silently returns if cand_id is already in seen_ids (duplicate-safe).
    """
```

---

## 5. Training Buffer Management

In train mode, the orchestrator accumulates `(s1_id, cand_id, feature_vector)` tuples. With 2M+ S1 entities × up to 60 candidates × 3 cycles, the buffer **must be flushed periodically** to avoid OOM.

### 5A. Buffer Flush Strategy

- **Flush trigger:** After completing ALL cycles for every `config.TRAIN_BUFFER_CHUNK_SIZE` S1 entities.
- **Flush action:** `pl.DataFrame(train_buffer).write_parquet(chunk_path)` → clear `train_buffer = []`.
- **Chunk filename pattern:** `{buffer_dir}/train_buffer_chunk_{chunk_id:04d}.parquet`
- **Final flush:** After the country loop ends, any remaining rows in `train_buffer` are flushed as a final chunk.

**Buffer row schema (one dict per candidate pair):**

| Column | Type | Description |
|:---|:---|:---|
| `s1_id` | str | Source 1 entity ID |
| `cand_id` | str | Candidate entity ID |
| `f_01` … `f_26` | float32 | Feature dimensions 1–26 in canonical `FEATURE_COLS` order |

> [!NOTE]
> **Why parquet, not CSV?** Float32 parquet is ~3× smaller than CSV, and Polars can scan/concat multiple chunk files lazily using `pl.scan_parquet(glob_pattern).collect()` without loading all into memory at once.

### 5B. Post-Loop Label Assembly (Train Mode)

After all buffer chunks are flushed:

```python
# Step 1: Collect all chunk parquets
train_df = pl.scan_parquet(f"{buffer_dir}/train_buffer_chunk_*.parquet").collect()

# Step 2: Explode ground truth → one row per (s1_id, matched_cand_id)
ground_truth_df = pl.read_csv(ground_truth_path, separator="\t").with_columns(
    pl.col("matched_entity_ids").fill_null("")
)
gt_exploded = (
    ground_truth_df
    .with_columns(pl.col("matched_entity_ids").str.split(",").alias("matched_list"))
    .explode("matched_list")
    .rename({"source1_entity_id": "s1_id", "matched_list": "cand_id"})
    .filter(pl.col("cand_id") != "")    # Remove empty strings from singletons
    .with_columns(pl.lit(1).alias("label").cast(pl.Int8))
)

# Step 3: Left-join → label=1 if in gt_exploded, else label=0
labeled_df = (
    train_df.join(gt_exploded[["s1_id", "cand_id", "label"]],
                  on=["s1_id", "cand_id"], how="left")
    .with_columns(pl.col("label").fill_null(0).cast(pl.Int8))
)

# Step 4: Hand off to trainer
run_training(labeled_df, model_out_path, config)
```

---

## 6. Model Integrity Check (Infer Mode)

Before any inference begins, the orchestrator verifies feature column alignment:

```python
def verify_model_feature_alignment(model_artifact: dict) -> None:
    """
    Compares model_artifact['feature_names'] (saved at train time)
    against the live FEATURE_COLS constant from feature_extractor.py.

    FEATURE_COLS contains 25 float column names + 'country' (raw string) as the 26th.
    The model artifact contains a full sklearn Pipeline (ColumnTransformer + XGBClassifier)
    that applies TargetEncoder internally — orchestrator does NOT encode country manually.

    Raises RuntimeError with a clear diff if they do not match exactly
    (both name AND position matter — XGBoost is column-order sensitive).
    """
    saved_cols = model_artifact.get("feature_names", [])
    live_cols  = FEATURE_COLS
    if saved_cols == live_cols:
        logging.info("✅ Feature alignment check passed: 26 columns match exactly.")
        return
    missing_in_live = set(saved_cols) - set(live_cols)
    extra_in_live   = set(live_cols)  - set(saved_cols)
    order_diffs     = [
        (i, s, l) for i, (s, l) in enumerate(zip(saved_cols, live_cols)) if s != l
    ]
    raise RuntimeError(
        f"Feature column mismatch between saved model and live feature_extractor.py!\n"
        f"  Missing in live code : {missing_in_live}\n"
        f"  Extra in live code   : {extra_in_live}\n"
        f"  Order diffs (idx, saved, live): {order_diffs}\n"
        "Fix feature_extractor.py or retrain the model before running infer."
    )
```

---

## 7. The Full `run_orchestrator()` Function

```python
def run_orchestrator(args) -> None:
    """Main entry point — called by main.py after arg parsing."""

    mode          = args.mode
    processed_dir = args.processed_dir or config.PROCESSED_DATA_DIR
    ground_truth  = getattr(args, "ground_truth", None) or config.GROUND_TRUTH_PATH
    model_out     = getattr(args, "model_out",    None) or config.MODEL_OUTPUT_PATH
    model_in      = getattr(args, "model_in",     None) or config.MODEL_OUTPUT_PATH
    output_dir    = getattr(args, "output_dir",   None) or config.OUTPUT_DIR

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger(__name__)

    # ── INFER MODE: Load and validate model ────────────────────────────
    model_artifact = model = threshold = None
    if mode == "infer":
        log.info(f"Loading model from: {model_in}")
        model_artifact = joblib.load(model_in)
        verify_model_feature_alignment(model_artifact)
        model     = model_artifact["model"]
        threshold = model_artifact["threshold"]
        log.info(f"Model loaded. Inference threshold: {threshold:.4f}")

    # ── GLOBAL STATE ────────────────────────────────────────────────────
    seen_dict       = init_seen_dict()  # ONCE — persists across all countries
    train_buffer    = []                # train mode only
    buffer_dir      = os.path.join(processed_dir, "_train_buffer")
    s1_entity_count = 0

    if mode == "train":
        os.makedirs(buffer_dir, exist_ok=True)

    # ── COUNTRY DISCOVERY ────────────────────────────────────────────────
    countries = discover_countries(processed_dir)
    log.info(f"Discovered {len(countries)} countries: {countries}")

    # ════════════════════════════════════════════════════════════════════
    # COUNTRY LOOP
    # ════════════════════════════════════════════════════════════════════
    for country in countries:
        log.info(f"━━━ Processing country: '{country}' ━━━")

        query_path  = os.path.join(processed_dir, f"Query_{country}.parquet")
        target_path = os.path.join(processed_dir, f"Target_{country}.parquet")
        query_df    = load_parquet(query_path)
        target_df   = load_parquet(target_path)

        # ── EMPTY PARTITION GUARD ─────────────────────────────────────
        if len(target_df) == 0:
            log.warning(f"  Target partition for '{country}' is EMPTY. "
                        "Initializing all S1 entities as singletons and skipping.")
            for row in query_df.iter_rows(named=True):
                init_entity(seen_dict, row["entity_id"])
            continue

        # ── INDEX BUILDING ────────────────────────────────────────────
        log.info(f"  Building indices: {len(query_df)} queries, {len(target_df)} targets")
        target_ids_list = target_df["entity_id"].to_list()

        target_name_corpus = [
            to_char_ngrams(t, config.BM25_NGRAM_SIZE)
            for t in target_df["name_for_bm25"].to_list()
        ]
        target_addr_corpus = [
            to_char_ngrams(t, config.BM25_NGRAM_SIZE)
            for t in target_df["addr_for_bm25"].to_list()
        ]
        bm25_name_index = build_bm25_name_index(target_name_corpus)
        bm25_addr_index = build_bm25_addr_index(target_addr_corpus)

        faiss_index, _ = build_faiss_index(
            texts=target_df["name_for_faiss"].to_list(),
            batch_size=config.FAISS_EMBED_BATCH_SIZE,
        )

        # Batch-embed ALL S1 queries in one GPU pass; store in lookup dict (alignment-safe)
        log.info(f"  Batch-embedding {len(query_df)} S1 entities via GPU...")
        s1_embs_matrix = faiss_index.embed_batch(
            query_df["name_for_faiss"].to_list(),
            batch_size=config.FAISS_EMBED_BATCH_SIZE,
        )
        emb_lookup = {
            eid: s1_embs_matrix[i]
            for i, eid in enumerate(query_df["entity_id"].to_list())
        }

        # Build O(1) target row lookup by entity_id
        target_lookup = {row["entity_id"]: row for row in target_df.iter_rows(named=True)}

        # ════════════════════════════════════════════════════════════════
        # S1 ENTITY LOOP
        # ════════════════════════════════════════════════════════════════
        for s1_row in query_df.iter_rows(named=True):
            s1_id = s1_row["entity_id"]
            s1_entity_count += 1

            init_entity(seen_dict, s1_id)

            cycle       = 1
            active      = True
            new_matches = 0

            # Pre-compute once; reused in all cycles
            s1_name_3grams = to_char_ngrams(s1_row["name_for_bm25"], config.BM25_NGRAM_SIZE)
            s1_addr_3grams = to_char_ngrams(s1_row["addr_for_bm25"], config.BM25_NGRAM_SIZE)
            s1_name_emb    = emb_lookup[s1_id]

            # ════════════════════════════════════════════════════════
            # CYCLIC DAG WHILE LOOP
            # ════════════════════════════════════════════════════════
            while active:
                new_matches = 0  # Reset at start of each cycle

                # ── STEP A: Dynamic Top-k Retrieval ─────────────────
                total_seen = seen_dict[s1_id]["total"]
                k          = config.DAG_K_BASE + total_seen
                k_faiss    = min(k, faiss_index.ntotal)  # Hard FAISS cap

                bm25_name_raw = bm25_name_search(bm25_name_index, s1_name_3grams, target_ids_list, n=k)
                bm25_addr_raw = bm25_addr_search(bm25_addr_index, s1_addr_3grams, target_ids_list, n=k)
                faiss_raw     = faiss_search(faiss_index, s1_name_emb, target_ids_list, n=k_faiss)

                # ── STEP B: Blacklist Filtering ──────────────────────
                blacklist     = seen_dict[s1_id]["seen_ids"]
                new_bm25_name = [x for x in bm25_name_raw if x["entity_id"] not in blacklist][:config.DAG_K_BASE]
                new_bm25_addr = [x for x in bm25_addr_raw if x["entity_id"] not in blacklist][:config.DAG_K_BASE]
                new_faiss     = [x for x in faiss_raw     if x["entity_id"] not in blacklist][:config.DAG_K_BASE]

                # ── Union & Priority Sort (≤ 60 candidates) ──────────
                union_candidates = priority_deduplicate(new_bm25_name, new_bm25_addr, new_faiss, k=k)

                # Index exhausted: 0 new candidates → terminate this entity
                if len(union_candidates) == 0:
                    active = False
                    break

                # ── Enrich each candidate with its full target row ────
                for cand in union_candidates:
                    cand["row"] = target_lookup[cand["entity_id"]]

                # ── Store per-cycle candidate list ────────────────────
                seen_dict[s1_id]["per_cycle"][cycle] = [c["entity_id"] for c in union_candidates]

                # ── STEP C: Feature Extraction ───────────────────────
                # Returns pd.DataFrame of shape (len(union_candidates), 26)
                # Columns 0-24: float (Dims 1-25), Column 25: str 'country' (raw, NOT encoded)
                # The Pipeline in the model artifact handles TargetEncoder automatically.
                feature_df = extract_features(s1_row, union_candidates)

                # ── STEP C: Mode Routing ─────────────────────────────
                if mode == "train":
                    # No ML call. Accumulate (s1_id, cand_id, features) for labeling.
                    for i, cand in enumerate(union_candidates):
                        train_buffer.append({
                            "s1_id"   : s1_id,
                            "cand_id" : cand["entity_id"],
                            **{col: feature_df.iloc[i][col] for col in FEATURE_COLS}
                        })
                        # Mark rejected (solely to blacklist in next cycle)
                        update_seen(seen_dict, s1_id, cand["entity_id"], accepted=False, cycle=cycle)
                    # new_matches stays 0 → stopping condition fires after cycle >= DAG_MIN_CYCLES

                else:  # mode == "infer"
                    # Pipeline.predict_proba() applies TargetEncoder internally
                    probs = pipeline.predict_proba(feature_df)[:, 1]
                    for i, cand in enumerate(union_candidates):
                        is_match = bool(probs[i] > threshold)
                        update_seen(seen_dict, s1_id, cand["entity_id"], accepted=is_match, cycle=cycle)
                        if is_match:
                            new_matches += 1

                # ── STOPPING CONDITION ────────────────────────────────
                total_seen = seen_dict[s1_id]["total"]
                if total_seen >= config.DAG_MAX_SEEN or \
                   (cycle >= config.DAG_MIN_CYCLES and new_matches == 0):
                    active = False
                else:
                    cycle += 1

            # END while loop

            # ── TRAIN BUFFER FLUSH (every TRAIN_BUFFER_CHUNK_SIZE entities) ──
            if mode == "train" and s1_entity_count % config.TRAIN_BUFFER_CHUNK_SIZE == 0:
                chunk_id   = s1_entity_count // config.TRAIN_BUFFER_CHUNK_SIZE
                chunk_path = os.path.join(buffer_dir, f"train_buffer_chunk_{chunk_id:04d}.parquet")
                pl.DataFrame(train_buffer).write_parquet(chunk_path)
                log.info(f"  Flushed chunk {chunk_id:04d}: {len(train_buffer)} rows → {chunk_path}")
                train_buffer = []

        log.info(f"  Country '{country}' complete. Cumulative S1 processed: {s1_entity_count:,}")

    # END country loop

    # ════════════════════════════════════════════════════════════════════
    # POST-LOOP FINALIZATION
    # ════════════════════════════════════════════════════════════════════
    if mode == "train":
        # Final buffer flush
        if train_buffer:
            chunk_id   = (s1_entity_count // config.TRAIN_BUFFER_CHUNK_SIZE) + 1
            chunk_path = os.path.join(buffer_dir, f"train_buffer_chunk_{chunk_id:04d}.parquet")
            pl.DataFrame(train_buffer).write_parquet(chunk_path)
            log.info(f"  Flushed final chunk: {len(train_buffer)} rows → {chunk_path}")

        # Assemble all chunks
        log.info("Assembling full training dataset from buffer chunks...")
        train_df = pl.scan_parquet(os.path.join(buffer_dir, "train_buffer_chunk_*.parquet")).collect()

        # Label via left-join with ground truth
        gt_df = pl.read_csv(ground_truth, separator="\t").with_columns(
            pl.col("matched_entity_ids").fill_null("")
        )
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
        n_pos = labeled_df["label"].sum()
        n_neg = len(labeled_df) - n_pos
        log.info(f"  Labeled pairs: {len(labeled_df):,}  |  Positives: {n_pos:,}  |  Negatives: {n_neg:,}")
        run_training(labeled_df, model_out, config)

    else:  # mode == "infer"
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Writing output files to: {output_dir}")
        write_all_outputs(seen_dict, output_dir)
        log.info("✅ All output files written. Run validate_submission.py to verify.")
```

---

## 8. `write_all_outputs()` Contract (`blocking_layer/seen_dict.py`)

Called **exactly once** in infer mode after ALL entities across ALL countries are processed.

```python
def write_all_outputs(seen_dict: dict, output_dir: str) -> None:
    """
    Single-pass writer. Generates exactly 3 tab-separated output files.
    All ID lists are comma-separated with NO quotes.

    ┌─────────────────────────────────────────────────────────────────┐
    │ matching_results.tsv                                            │
    │   Columns: source1_entity_id  matched_entity_ids               │
    │   Source:  seen_dict[s1_id]["accepted"]                        │
    ├─────────────────────────────────────────────────────────────────┤
    │ candidate_pairs.tsv                                             │
    │   Columns: source1_entity_id  candidate_entity_ids             │
    │   Source:  seen_dict[s1_id]["seen_ids"] (accepted ∪ rejected)  │
    ├─────────────────────────────────────────────────────────────────┤
    │ candidate_pairs_per_cycle.tsv  (debug/audit)                   │
    │   Columns: source1_entity_id  cycle_number  candidate_ids      │
    │   Source:  seen_dict[s1_id]["per_cycle"][cycle_num]            │
    └─────────────────────────────────────────────────────────────────┘

    Guarantees:
    - EXACTLY 1 row per s1_id in matching_results.tsv and candidate_pairs.tsv.
    - matched_entity_ids / candidate_entity_ids are EMPTY STRINGS for singletons,
      NOT "None", NOT a missing column value.
    - ID lists are sorted(set) for fully reproducible outputs.
    """
    matching_path  = os.path.join(output_dir, "matching_results.tsv")
    candidate_path = os.path.join(output_dir, "candidate_pairs.tsv")
    per_cycle_path = os.path.join(output_dir, "candidate_pairs_per_cycle.tsv")

    with open(matching_path,  "w", encoding="utf-8") as f_match, \
         open(candidate_path, "w", encoding="utf-8") as f_cand,  \
         open(per_cycle_path, "w", encoding="utf-8") as f_cyc:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        f_cyc.write("source1_entity_id\tcycle_number\tcandidate_entity_ids\n")

        for s1_id, entry in seen_dict.items():
            accepted_str = ",".join(sorted(entry["accepted"]))
            all_seen_str = ",".join(sorted(entry["seen_ids"]))

            f_match.write(f"{s1_id}\t{accepted_str}\n")
            f_cand.write(f"{s1_id}\t{all_seen_str}\n")

            for cycle_num in sorted(entry["per_cycle"].keys()):
                cycle_ids_str = ",".join(entry["per_cycle"][cycle_num])
                f_cyc.write(f"{s1_id}\t{cycle_num}\t{cycle_ids_str}\n")
```

---

## 9. Error Handling & Logging Standards

| Situation | Action |
|:---|:---|
| `processed_dir` has no `Query_*.parquet` files | `raise FileNotFoundError` with actionable message |
| `Target_C.parquet` is missing for a discovered country | `raise FileNotFoundError` — parquets come in Query/Target pairs |
| `Target_C` is empty (0 rows) | Log WARNING, init all S1 as singletons, `continue` to next country |
| `k_faiss > faiss_index.ntotal` | Silently cap to `faiss_index.ntotal` — no per-entity log spam |
| Feature column mismatch on infer startup | `raise RuntimeError` with full column diff — crash loudly |
| BM25 returns fewer than `k` results | `rank-bm25` handles natively — no special handling needed |
| Empty union after blacklist filtering | `active = False`, `break` — no feature extraction, no buffer row |
| `emb_lookup` missing an entity_id | Should be impossible if `build_faiss_index` and `query_df` are aligned; raise `KeyError` if it occurs |

---

## 10. Configuration Variables Used by the Orchestrator

All values read from `config.py` (loaded from `.env`). No module other than `config.py` calls `os.getenv()`.

| `config.*` Key | Type | Default | Usage |
|:---|:---|:---|:---|
| `config.PROCESSED_DATA_DIR` | str | `"data/processed"` | Default parquet directory |
| `config.GROUND_TRUTH_PATH` | str | `"dataset/train/train_ground_truth.tsv"` | Train label join |
| `config.MODEL_OUTPUT_PATH` | str | `"models/xgb_model.pkl"` | Default model path |
| `config.OUTPUT_DIR` | str | `"output/"` | Default output directory |
| `config.DAG_K_BASE` | int | `20` | Base k per stream per cycle |
| `config.DAG_MAX_SEEN` | int | `90` | Global stopping ceiling (MAX_SEEN_PER_ENTITY) |
| `config.DAG_MIN_CYCLES` | int | `3` | Min cycles before 0-match stop (MIN_CYCLES_BEFORE_STOP) |
| `config.BM25_NGRAM_SIZE` | int | `3` | Char n-gram size for BM25 queries |
| `config.FAISS_EMBED_BATCH_SIZE` | int | `256` | GPU batch size for MiniLM |
| `config.TRAIN_BUFFER_CHUNK_SIZE` | int | `10_000` | S1 entities per buffer flush (**ADD TO .env + config.py**) |

> [!IMPORTANT]
> `TRAIN_BUFFER_CHUNK_SIZE` is **not yet in `config.py`**. It must be added before implementing the orchestrator. Add `TRAIN_BUFFER_CHUNK_SIZE = get_env_int("TRAIN_BUFFER_CHUNK_SIZE", 10_000)` to `Config` class, and `TRAIN_BUFFER_CHUNK_SIZE=10000` to `.env`.

---

## 11. Integration Checklist

Before running the orchestrator end-to-end, verify all 10 items:

```
[ ] Phase 0 complete: Query_<country>.parquet + Target_<country>.parquet exist in PROCESSED_DATA_DIR
[ ] blocking_layer/semantic_index.py: build_faiss_index(texts, batch_size) → (index, emb_matrix)
                                       faiss_search(index, query_emb, target_ids, n) → list[dict{entity_id, score}]
                                       index.embed_batch(texts, batch_size) → np.ndarray
                                       index.ntotal (int property)
[ ] blocking_layer/lexical_name_index.py: build_bm25_name_index(corpus) → BM25Okapi
                                           bm25_name_search(index, query_tokens, target_ids, n) → list[dict{entity_id, score}]
[ ] blocking_layer/lexical_addr_index.py: build_bm25_addr_index(corpus) → BM25Okapi
                                           bm25_addr_search(index, query_tokens, target_ids, n) → list[dict{entity_id, score}]
[ ] blocking_layer/seen_dict.py: init_seen_dict(), init_entity(), update_seen(), write_all_outputs()
[ ] feature_layer/feature_extractor.py: extract_features(s1_row, union_candidates) → pd.DataFrame (N, 26)
                                          Columns 0-24: float (Dims 1-25)
                                          Column 'country': raw string — NOT encoded
                                          FEATURE_COLS: list of 26 str (25 names + 'country')
[ ] ml_layer/trainer.py: run_training(labeled_df, model_out_path, config) → saves Pipeline artifact
                          NOTE: ml_layer/classifier.py is NOT needed.
                          Pipeline.predict_proba(df) handles encoding + scoring in one call.
[ ] config.py + .env: TRAIN_BUFFER_CHUNK_SIZE added
[ ] Model artifact (infer mode): joblib dict with keys:
      "model"         → sklearn Pipeline (ColumnTransformer(TargetEncoder) + XGBClassifier)
      "threshold"     → float
      "feature_names" → list of 26 str (25 float col names + 'country')
      "best_params"   → dict (XGBoost params with 'classifier__' prefix)
```

---

## 12. Development & Verification Plan

### Phase A — Smoke Test (Step 6, before full run)

Verify the orchestrator wires correctly with a tiny synthetic dataset.

```bash
# Reduce config to minimal values:
# DAG_MAX_SEEN=20, DAG_MIN_CYCLES=1, CANDIDATES_PER_STREAM=5, TRAIN_BUFFER_CHUNK_SIZE=50

python3 src/main.py \
    --mode train \
    --processed-dir data/processed_sample/ \
    --ground-truth dataset/train/train_ground_truth.tsv \
    --model-out models/xgb_model_smoke.pkl
```

**Expected:** No crashes; buffer parquets created; `labeled_df` assembled; `run_training()` called.

### Phase B — Full Train Run (Step 7)

```bash
bash run.sh --mode train \
            --processed-dir data/processed/ \
            --ground-truth dataset/train/train_ground_truth.tsv \
            --model-out models/xgb_model.pkl
```

**Expected:** `models/xgb_model.pkl` saved; threshold logged.

### Phase C — Full Infer + Validation (Steps 8–9)

```bash
bash run.sh --mode infer \
            --processed-dir data/processed_test/ \
            --model-in models/xgb_model.pkl \
            --output-dir output/

python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
# Must print: PASS (exit 0)
```
