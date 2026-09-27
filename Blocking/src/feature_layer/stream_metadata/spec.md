# Stream Metadata Feature Spec (Dims 14-26)

## 1. Specifications & Schemas
**Inputs:**
- `name_a`, `name_b`, `addr_a`, `addr_b` (Strings)
- `bm25_name_score`, `bm25_addr_score` (Floats)
- `bm25_name_self_score`, `bm25_addr_self_score` (Floats, passed from Orchestrator)
- `bm25_name_rank`, `bm25_addr_rank`, `faiss_name_rank` (Ints or None) - **1-indexed**
- `meta_is_cross_script_a`, `meta_is_cross_script_b` (Int)
- `meta_source` (String: "S2" or "S3")
- `meta_country` (String)

**Outputs:**
- `dim_14_name_bm25_norm` (Float)
- `dim_15_addr_bm25_norm` (Float)
- `dim_16_rrf_score` (Float)
- `dim_17_overlap_count` (Int): {0, 1, 2, 3}
- `dim_18_name_len_ratio` (Float): [0.0, 1.0]
- `dim_19_addr_len_ratio` (Float): [0.0, 1.0]
- `dim_20_s1_name_missing` (Int): {0, 1}
- `dim_21_cand_name_missing` (Int): {0, 1}
- `dim_22_s1_addr_missing` (Int): {0, 1}
- `dim_23_cand_addr_missing` (Int): {0, 1}
- `dim_24_cross_script` (Int): {0, 1}
- `dim_25_source_origin` (Int): {0, 1}
- `dim_26_country_raw` (String)

## 2. Requirements (Business Logic)
- **Dims 14, 15 (BM25 Normalization):** `bm25_score / bm25_self_score`. (BM25 self-score and 1e-9 clamping logic is fully decoupled and provided by `bm25_utils.py` upstream).
- **Dim 16 (RRF):** `sum(1/(K + rank))` for all non-None ranks. The constants MUST be dynamically imported: `config.RRF_K_BM25_NAME`, `config.RRF_K_BM25_ADDR`, and `config.RRF_K_FAISS`. Ranks are **1-indexed**.
- **Dim 17 (Overlap):** Count of streams where candidate rank is not None.
- **Dims 18, 19 (Length Ratios):** `min(len(A), len(B)) / max(len(A), len(B), 1)` (character units). The denominator guard `max(..., 1)` prevents zero division if upstream leaks an empty string.
- **Dims 20-23 (Missing Flags):** Binary flags testing if the string `.startswith("nullname")` or `.startswith("nulladdr")`.
- **Dim 24 (Cross Script):** `max(meta_is_cross_script_a, meta_is_cross_script_b)`.
- **Dim 25 (Source Origin):** Strictly a single binary scalar: `0` for S2, `1` for S3 (prevents perfect collinearity / dummy variable trap).
- **Dim 26 (Country):** Return raw string (e.g. `"us"`, `"france"`). TargetEncoder is applied later in ML pipeline.

## 3. Assumptions
- Upstream `bm25_utils.py` handles the analytical `get_self_score()` calculation.
- Retrieval models return **1-indexed** ranks.

## 4. Edge-Cases & Errors
- **Length Ratios on Sentinels (Dims 18 & 19):** If either string `.startswith("nullname")` or `.startswith("nulladdr")`, the ratio MUST explicitly return `0.0` instead of calculating a ratio against the arbitrary sentinel string. This ensures a clean split for XGBoost.
- **Missing Rankings:** Candidates may not be found in all 3 streams. Missing ranks contribute `0.0` to the RRF score.
