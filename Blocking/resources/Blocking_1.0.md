# Blocking 2.7: Progressive Cyclic DAG Pipeline (PATCHED v2.0)

> [!WARNING]
> **ARCHITECTURAL OVERRIDES APPLIED**
> The logic in this document has been bulletproofed. You MUST apply the following patches during implementation:
> 1. **BM25 Math Isolation:** BM25 math must be extracted to a standalone `bm25_utils.py` module.
> 2. **OOV Safety Guard:** Use `.get(token, 0.0)` for OOV handling in the self-score calculation.
> 3. **Stateless Zero-Division Protection:** Protect `avgdl <= 0.0` with a floor value of `1e-9`.
> 4. **Retrieval Optimization (Pre-Filter):** Explicitly filter scores `> 0.0` *before* sorting to achieve O(M log M) speed.
> 5. **Stable Descending Sort:** Use `valid_idx[np.argsort(-scores[valid_idx], kind='stable')[:k]]` to prevent RRF jitter.
> 6. **Generator OOM Defense:** Pass a lazy generator `(to_char_3grams(t) for t in texts)` instead of a list comprehension to `rank-bm25`.
> 7. **Memory Leak Prevention:** Drop the raw `texts` list immediately after building the index.
> 8. **Zero-Vocabulary Self-Healing:** Eradicate `EmptyCorpusError`. Set `self._is_empty = True` and return `[]` if corpus is empty.
> 9. **Raw Score Guarantee:** Indices must return pure, raw scores to prevent Double-Normalization poisoning.
> 10. **Duck-Typing Parity:** `SemanticIndex` must implement `get_self_score(query_embedding)` returning a constant `1.0` (or `1e-9`).
> 11. **Symmetric Normalization:** `SemanticIndex.search()` MUST explicitly L2-normalize the `query_embedding`.
> 12. **FAISS Thread-Thrashing:** Mandate `faiss.omp_set_num_threads(1)` to disable internal multithreading.
> 13. **Native Index Caching:** Both indices must implement `save(dir_path)` and `load(dir_path)`. `SemanticIndex` must use `faiss.write_index`.
> 14. **FAISS C++ Segfault Defense:** Cast query embeddings to `np.float32` and `.reshape(1, -1)` before `search()`.
> 15. **JSONL Micro-Batch State Tracker:** `seen_dict` must stream finalized entities to a `.jsonl` file in batches to prevent RAM OOM, reading existing lines into a `completed_ids` set on startup for True Resumability.
> 16. **State Anti-Corruption:** Throw `AlreadyFinalizedError` on duplicate processing and `AlreadyActiveError` on double-initialization. Force-flush the stranded JSONL buffer before writing TSVs.
> 17. **Redundancy Elimination:** Prune `seen_ids` and `total` from the `seen_dict` schema. Evaluate state dynamically via `candidate in accepted or candidate in rejected`.
> 18. **Missing Match Fallback:** Output empty right-hand sides for zero-match entities across ALL generated TSVs.
> 19. **Generator-Safe EAFP Pattern:** To prevent generator exhaustion while self-healing zero-vocabulary lexical indices, wrap the `BM25Okapi(generator)` initialization in `try...except Exception:` and execute a post-instantiation check (`if len(self._bm25.idf) == 0`) to dynamically set `self._is_empty = True`.

This document defines the final, robust architecture for Candidate Generation and Evaluation. It perfectly aligns with the Amazon ML Challenge constraints (multilingual, dynamic country scaling) and introduces a state-of-the-art **Progressive Cyclic DAG (Directed Acyclic Graph)**. 

Instead of overwhelming the ML model with hundreds of candidates per entity upfront, this pipeline intelligently retrieves candidates in cycles across 3 independent streams, evaluates them, memorizes rejections globally, and stops automatically when an entity is exhausted.

---

## 0. Inputs from Phase 0 (Pre-built Parquets)
*Goal: Load the pre-processed, country-partitioned datasets produced by Phase 0 to optimize the search direction and handle unseen countries (like `France`) seamlessly.*

The blocking layer (`orchestrator.py`) does not process raw TSVs. It receives pre-partitioned `.parquet` files produced by `data_layer/cleaner.py`. 

For each country `C` discovered in the input directories, the orchestrator loads:
1.  **`Query_C.parquet`**: Cleaned Source 1 entities for country `C`.
2.  **`Target_C.parquet`**: Combined and cleaned Source 2 and Source 3 entities for country `C`.

---

## 1. 3-Stream Feature Extraction
*Goal: Capture exact character matches and semantic synonyms across any language natively using specialized pre-processed representations.*

We use 3 distinct indexing streams, relying on the preprocessed columns from Phase 0:

1.  **BM25-Name (Lexical Name Stream):** Uses the `name_for_bm25` column. Converts to **Character 3-Gram BM25Okapi**. This mathematically catches typos, abbreviations, and exact character overlaps.
2.  **BM25-Addr (Lexical Address Stream):** Uses the `addr_for_bm25` column. Converts to **Character 3-Gram BM25Okapi**. Addresses are structured; this dedicated index prevents long addresses from drowning out name signals and natively catches numeric code overlaps.
3.  **FAISS-Name (Semantic Name Stream):** Uses the `name_for_faiss` column. Converts to dense vectors using a pre-trained **Multilingual Bi-Encoder** (e.g., `paraphrase-multilingual-MiniLM-L12-v2`). This flawlessly handles cross-lingual semantics (e.g., `Société` vs `SA`) while preserving original scripts. (FAISS is NOT used on addresses).

---

## 2. Dynamic Indexing, 3-Stream Retrieval & Union (The Base Engine)
*Goal: Execute the search efficiently and combine the strengths of all 3 streams.*

For each dynamically discovered country `C`:
1.  **Empty Partition Guard:** Before proceeding, check `len(Target_C) == 0`. If the target dataset for this country is empty (possible for test-set countries like `France` where only S1 exists but S2/S3 were dropped or empty), immediately call `init_entity(seen_dict, s1_id)` for all entities in `Query_C` and skip index building. This prevents hard crashes in `rank-bm25` or `faiss` and ensures singletons receive output rows.
2.  **Batch Embedding (Crucial Performance Step):** Batch encode the `name_for_faiss` column for the entire `Query_C` dataset using the GPU to yield a 2D matrix (`s1_name_embs`). **Alignment Guard:** Immediately map this matrix into a lookup dictionary keyed by the entity ID (`emb_lookup = {s1_id: s1_name_embs[i] for i, s1_id in enumerate(Query_C['entity_id'])}`). Do not re-embed strings in the inner loop, and do not rely on positional loop counters.
3.  **Index Building:** 
    *   Build a **BM25-Name Index** using `name_for_bm25` of `Target_C`.
    *   Build a **BM25-Addr Index** using `addr_for_bm25` of `Target_C`.
    *   Build a **FAISS-Name Index** using semantic embeddings of `name_for_faiss` from `Target_C`.
4.  **3-Stream Retrieval Execution:** For every query record (Source 1):
    *   Query BM25-Name to retrieve Top-`k` lexical name matches.
    *   Query BM25-Addr to retrieve Top-`k` lexical address matches.
    *   Query FAISS-Name to retrieve Top-`k` semantic name matches.
5.  **Union:** Merge candidates from all 3 streams. This yields a maximum of `3 * k` unique candidates per Source 1 entity (often fewer due to overlap), guaranteeing almost zero true matches are missed before evaluation.

---

## 3. The Cyclic DAG Retrieval & Evaluation Process
*Goal: Actively retrieve candidates in cycles, maintaining a global rejection blacklist, to save compute and maximize precision.*

> **Pre-condition:** This section is only entered if `Target_C` is non-empty. If the Empty Partition Guard in Section 2 fired for this country, all `Query_C` entities were already initialized via `init_entity()` and Section 3 is skipped entirely for that country. Do NOT duplicate the entity loop.

Before processing any countries or queries, we initialize a global dictionary `seen_dict` as the single source of truth. **This dictionary is initialized exactly ONCE at the start of the entire pipeline run (before the country loop) and persists across all countries. It is never reset.**

```python
seen_dict["S1-00001"] = {
    "accepted"  : set(),   # Confirmed matches
    "rejected"  : set(),   # ML-rejected candidates (blacklisted)
    "per_cycle" : {},      # dict: { cycle_num (int): [candidate_ids (str)] }
}
```

For every record in `Query_C` (Source 1), we initialize:
1.  **`init_entity(seen_dict, s1_id)`:** Unconditionally create the entity's blank entry in `seen_dict` FIRST. This guarantees the entity appears in all 3 output files even if zero candidates are ever found.
2.  `cycle_count = 1`
3.  `active_search = True`
4.  **Pre-compute Query Representations:** Compute `s1_name_3grams` and `s1_addr_3grams` once. Fetch the pre-computed `s1_name_emb` from the dictionary (`emb_lookup[s1_id]`). These are reused identically in every cycle to save massive compute and guarantee O(1) alignment.

While `active_search == True`:

### Step A: Dynamic Top-K Retrieval
We ensure that the ML model always receives new, unseen candidates per stream.
1.  **Reset:** Set `new_matches_this_cycle = 0` at the start of each cycle.
2.  Set `k = config.DAG_K_BASE + seen_dict[s1_id]["total"]`.
3.  **FAISS Cap (Critical):** Set `k_faiss = min(k, faiss_index.ntotal)`. FAISS's C++ core enforces a hard assertion `k <= index.ntotal` and will crash the process if violated. BM25 (`rank-bm25`) gracefully returns all documents when `k > corpus_size`, so no capping is needed for BM25 queries.
4.  Retrieve Top-`k` candidates from BM25-Name and BM25-Addr using `k`. Retrieve Top-`k_faiss` candidates from FAISS-Name using `k_faiss`. Use the pre-computed query representations for all 3 queries.

### Step B: Post-Filtering & Ordering (The Blacklist)
1.  Filter out any candidate ID that currently exists in `seen_dict[s1_id]["accepted"]` or `seen_dict[s1_id]["rejected"]` from all 3 retrieved lists.
2.  This leaves **up to `config.DAG_K_BASE`** new, unseen candidates per stream (fewer if the index is smaller than `k` or the target partition is nearly exhausted).
3.  Combine these lists into a final union of **<= `3 * config.DAG_K_BASE` candidates** (depending on overlap).
4.  **Ordering Strategy:** Sort this combined list prioritizing candidates found in **all 3 streams**, then **2 streams**, then **1 stream only**.
5.  **RRF Metadata Preservation:** Each candidate object in the union must carry its original rank (position in the 1-indexed retrieval list) from each stream it appeared in. For streams where a candidate was NOT retrieved, assign a sentinel rank of `k+1` (giving a near-zero RRF contribution). This rank metadata is required by the feature extractor to compute the 3-stream RRF Score (Feature Dim 16).
6.  **Index Exhaustion Override:** If the union is empty (0 candidates after filtering), immediately set `active_search = False` and skip Step C for this entity.

### Step C: Processing & Candidate Saving (Train vs Infer Mode)
The behaviour here depends entirely on the `--mode` flag of the orchestrator.
1.  **Store Per-Cycle Output:** Store this cycle's union candidates in `seen_dict[s1_id]["per_cycle"][cycle_count]`.
2.  Pass all `<= 3 * config.DAG_K_BASE` unseen candidates through the **Feature Engineering** phase to extract the **26-Dimensional Feature Vector** (including `name_bm25_score`, `addr_bm25_score`, 3-stream RRF Score, Stream Overlap Count, and string similarities).
3.  **Mode Routing:**
    *   **If Mode == `train`:**
        *   The ML model is NOT called (it doesn't exist yet).
        *   Store `(s1_id, cand_id, feature_vector)` tuples in a training buffer.
        *   Mark all candidates as **Rejected** (`update_seen(..., accepted=False)`) solely to add them to the blacklist for the next cycle.
        *   *Note: `new_matches_this_cycle` will safely remain 0, meaning training entities systematically trigger the stopping condition after exactly 3 cycles.*
    *   **If Mode == `infer`:**
        *   Pass the 18-dim features into the **trained XGBoost ML Classifier**.
        *   If `XGBoost Confidence > Threshold`: Mark as **Matched** (`update_seen(..., accepted=True)`). Increment `new_matches_this_cycle`.
        *   If `XGBoost Confidence <= Threshold`: Mark as **Rejected** (`update_seen(..., accepted=False)`).

---

## 4. The Local Stopping Metric
*Goal: Prevent infinite loops and save compute by halting searches for entities that have exhausted their true matches.*

After Step C completes, we evaluate the stopping condition for this specific S1 entity:

1.  **The Metric:** 
    *   If `(len(seen_dict[s1_id]["accepted"]) + len(seen_dict[s1_id]["rejected"])) >= config.DAG_MAX_SEEN` OR `(cycle_count >= config.DAG_MIN_CYCLES AND new_matches_this_cycle == 0)`: Set `active_search = False`. Finalize the S1 entity and flush it to the JSONL buffer. Move to the next S1 entity.
    *   Else: Increment `cycle_count += 1` and loop back to Step A.

---

## 5. Step D: Post-Loop Finalization
*Goal: Ensure exact compliance with output file constraints and training dataset preparation.*

**After ALL countries and ALL S1 entities in the dataset have been processed**, the outer loop finishes. We finalize based on the mode:

### If Mode == `train`:
Join the accumulated training buffer with `train_ground_truth.tsv`. If a pair exists in the ground truth, label it `1`, else `0`. Pass this dataset to `trainer.py` to train the XGBoost model.

### If Mode == `infer`:
Call `write_all_outputs(seen_dict)` **exactly once** to generate the 3 output files in a single pass. The module will execute a force-flush of the JSONL buffer, then parse the `.jsonl` file to generate the TSVs. The files are tab-separated (`\t`), and ID lists are mathematically sorted and comma-separated with NO quotes. Empty match fallbacks apply across all files.

1.  **`matching_results.tsv`** (Uploaded to leaderboard)
    *   Rows: Exactly 1 row per S1 entity.
    *   Columns: `source1_entity_id` and `matched_entity_ids` (from `accepted`).
2.  **`candidate_pairs.tsv`** (Submitted for blocking evaluation)
    *   Rows: Exactly 1 row per S1 entity.
    *   Columns: `source1_entity_id` and `candidate_entity_ids` (from `accepted ∪ rejected`).
3.  **`candidate_pairs_per_cycle.tsv`** (Internal debug/audit)
    *   Rows: Multiple rows. Flattens the `per_cycle` dictionary.
    *   Columns: `source1_entity_id`, `cycle_number`, `candidate_entity_ids` (comma-separated list of candidates found in that specific cycle).

---

## 6. Why This Architecture is 'Bulletproof'
1.  **Guarantees Output Compliance:** Single-pass writing post-loop guarantees exactly one row per S1 entity with correct exact column names, aggregating results safely across all country partitions.
2.  **No Wasted Compute:** The local stopping metric (`config.DAG_MAX_SEEN`, or `config.DAG_MIN_CYCLES` with no matches) halts semantic searches immediately when matches are exhausted. The strict cycle reset of `new_matches_this_cycle` prevents infinite loops.
3.  **Memory Safe & Highly Precise (Active Rejection):** Global O(1) `seen_dict` blacklisting guarantees we always feed only **new, unseen** candidates from each stream to the heavy ML model.
4.  **Mode Parity:** The train and infer modes share the exact same cyclic retrieval logic, guaranteeing that the distribution of candidates the model trains on perfectly matches what it will see in production.

---

## 7. Execution & Data Flowchart

```mermaid
graph TD
    %% Initial State Data Stores
    GlobalStart["Initialize JSONL Micro-Batch & completed_ids ONCE"]
    S[("JSONL state.jsonl & completed_ids")]
    GlobalStart -.-> S
    
    %% Country Loop
    GlobalStart --> CountryLoop{For each Country C}
    
    CountryLoop -->|Load Data| LoadC["Load Query_C and Target_C"]
    LoadC --> EmptyGuard{"len(Target_C) == 0?"}
    EmptyGuard -->|Yes| SkipIdx["init_entity() for all S1<br>Skip Country"]
    SkipIdx --> NextCountry
    EmptyGuard -->|No| BuildIdx["Batch Embed Query_C (GPU)<br>Build BM25 & FAISS Indices"]
    
    %% S1 Entity Loop
    BuildIdx --> EntityLoop{For each S1 Entity}
    
    %% Cycle Initialization
    EntityLoop -->|1. init_entity() FIRST<br>2. Pre-compute Query Reps| Start["Initialize:<br>cycle=1, active=True"]
    Start --> Loop{While active == True}
    
    %% Dynamic Retrieval
    Loop -->|Reset new_matches=0<br>k = 20 + total_seen| R_BN["Retrieve Top-k (BM25-Name)"]
    Loop -->|Retrieve| R_BA["Retrieve Top-k (BM25-Addr)"]
    Loop -->|Retrieve| R_F["Retrieve Top-k (FAISS-Name)"]
    
    %% Post-Filtering Isolation
    R_BN --> F_BN["Filter against accepted/rejected"]
    R_BA --> F_BA["Filter against accepted/rejected"]
    R_F --> F_F["Filter against accepted/rejected"]
    S -.->|Blacklist IDs (O(1))| F_BN
    S -.->|Blacklist IDs (O(1))| F_BA
    S -.->|Blacklist IDs (O(1))| F_F
    
    %% The <= 60 Combination
    F_BN -->|Up to 20 New| Union["Union Combine<br>(Yields <= 60 Candidates)"]
    F_BA -->|Up to 20 New| Union
    F_F -->|Up to 20 New| Union
    
    %% Ordering Strategy & Early Stop
    Union --> Sort["Sort: 3 Streams > 2 Streams > 1 Stream"]
    Sort --> StopEarlyCheck{"Union is empty?"}
    StopEarlyCheck -->|Yes| End["Set active = False"]
    StopEarlyCheck -->|No| StoreP["Store in per_cycle dict"]
    
    %% ML Evaluation & Mode Split
    StoreP --> Eval["Extract 26-Dim Features"]
    Eval --> ModeCheck{Mode == 'infer'?}
    
    %% Train Mode
    ModeCheck -->|No (train)| StoreBuf["Store in Train Buffer"]
    StoreBuf --> RejectTrain["Mark All Rejected<br>(For Blacklist Only)"]
    RejectTrain --> UpdateS
    
    %% Infer Mode
    ModeCheck -->|Yes (infer)| XGB["XGBoost Probability Score"]
    XGB -->|Score > Threshold| Match["Mark Matched<br>Increment new_matches"]
    XGB -->|Score <= Threshold| Reject["Mark Rejected"]
    
    Match -->|Add ID| UpdateS["Update seen_dict<br>accepted/seen_ids"]
    Reject -->|Add ID| UpdateS
    
    %% Stopping Metric
    UpdateS --> StopCheck{"(len(acc) + len(rej)) >= 90 OR<br>(cycle >= 3 AND 0 new matches)?"}
    StopCheck -->|Yes| End
    StopCheck -->|No| Inc["cycle += 1"]
    Inc --> Loop
    
    %% Loop Returns
    End --> NextEntity["Next S1 Entity"]
    NextEntity --> EntityLoop
    
    EntityLoop -.->|No more entities| NextCountry["Next Country"]
    NextCountry --> CountryLoop
    
    %% Final Write
    CountryLoop -.->|After ALL Countries Done| Write["write_all_outputs(seen_dict)"]
    Write --> C_OUT[("candidate_pairs.tsv")]
    Write --> M_OUT[("matching_results.tsv")]
    Write --> P_OUT[("candidate_pairs_per_cycle.tsv")]
```
