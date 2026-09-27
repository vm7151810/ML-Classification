# Phase 3: Feature Engineering v7.0 (26-Dim Final Frozen Edition) (PATCHED v2.0)

> [!WARNING]
> **ARCHITECTURAL OVERRIDES APPLIED**
> The logic in this document has been bulletproofed. You MUST apply the following patches during implementation:
> 1. **BM25 Math Decoupling:** The `compute_bm25_self_score()` pseudocode in this document is DEPRECATED. BM25 math has been strictly extracted to `src/blocking_layer/bm25_utils.py`. The Orchestrator queries `index.get_self_score()`, and passes the normalization denominator to the feature layer.
> 2. **Length Ratio Null Sentinel Guard:** For Dims 18 and 19, if either string `.startswith("nullname")` or `.startswith("nulladdr")`, the length ratio MUST explicitly return `0.0`. Do not compute character length ratios on arbitrary sentinel flags.
> 3. **Dynamic RRF Constants:** `config.RRF_K_BM25_NAME`, `config.RRF_K_BM25_ADDR`, and `config.RRF_K_FAISS` must be dynamically imported from `config.py`. Do NOT hardcode `60`.
> 4. **Source Origin Encoding (Dim 25):** The "One-hot" terminology is imprecise. Because there are only two secondary sources, Dim 25 MUST be a single binary scalar (S2 = `0`, S3 = `1`) to perfectly avoid collinearity (the dummy variable trap).

> **Changelog:**
> - v1.0 → v2.0: Aligned to 3-stream blocking; added `is_cross_script` meta-feature; fixed 2-stream RRF to 3-stream.
> - v2.0 → v3.0: Added Addr Token Containment (Dim 9); split Missing Flags (Dims 19–20); added Phonetic Set Intersection fix.
> - v3.0 → v4.0: Fixed BM25 normalization to Analytical Self-Score; fixed phonetic encoding to word-level; specified Acronym Score edge cases; added Country Encoding leakage guard.
> - v4.0 → v5.0: Fixed Monge-Elkan asymmetry (now symmetric average); added `compute_bm25_self_score()` pseudocode; added France/unseen-country fallback for target encoding; split 2 missing-data flags into 4 distinct flags; added `Addr_Numeric_Exists_Both` companion flag; added canonical 26-row summary table.
> - v5.0 → v6.0: Fixed Levenshtein to be a similarity metric (1.0 - dist/max_len) for monotonic constraint compatibility; fixed `ZeroDivisionError` in Addr Numeric Jaccard; explicitly defined `is_cross_script` aggregation (`max`); mandated L2-normalization for Semantic Cosine; specified `.split()` tokenizer for address sets.
> - v6.0 → v7.0: **Fixed `pyphonetics` API mismatch (primary code only); added `max(0.0, score)` clamping to Semantic Cosine to guarantee `[0, 1]` range; explicitly defined character-level units for Length Ratio features.**

---

In Phase 2 (Blocking 2.6), we generated a manageable, highly precise union of <= 60 candidate pairs per entity across 3 independent streams. In **Phase 3 (Feature Engineering)**, the goal is to transform every candidate pair into a strict **26-Dimensional Feature Vector** (23 core features + 3 meta features) so the final XGBoost/LightGBM model can make a highly informed binary decision (Match vs. Non-Match).

This version is fully aligned with `Preprocessing 2.0` and `Blocking 2.6`. All runtime crash risks, math-safety issues, and model-quality gaps have been resolved.

---

## 1. Feature Representation Rules (Crucial — Read First)

Before calculating any metrics, the feature extractor must route the correct string representations (produced by Phase 0 Preprocessing) to the correct distance algorithms:

| Feature Category | Column to Use | Reason |
| :--- | :--- | :--- |
| **Semantic** (cosine) | `name_for_faiss` | MiniLM requires the original script (Devanagari, Tamil, etc.) to extract meaning. |
| **Lexical & Phonetic** | `name_for_bm25` | Double Metaphone, Jaro-Winkler, Levenshtein expect Latin characters. These columns are safely ITRANS-transliterated. |
| **Address features** | `addr_for_bm25` | Same Latin-only guarantee; includes transliterated Indic tokens and expanded abbreviations. |
| **Missing Data Flags** | `name_for_bm25`, `addr_for_bm25` | Check `.startswith("nullname")` / `.startswith("nulladdr")`. Preprocessing v2.0 guarantees no standard null values — missingness is always flagged by the `nullname`/`nulladdr` prefix sentinel. |

---

## 2. Name-Based Features (Lexical, Semantic & Phonetic)

### A. String Distance Metrics (all computed on `name_for_bm25`)

*   **[Dim 1] Jaro-Winkler Similarity:** Heavily weights prefix matches. Captures abbreviation variants like `Amazon Inc` vs `Amazon Retail`.

*   **[Dim 2] Monge-Elkan Distance (Symmetric):** An industrial standard for multi-word strings that handles word-reordering combined with typos (e.g., `Batra Retail Inc` vs `Retail Bata`).
    > **Implementation Rule (Symmetry):** Monge-Elkan is asymmetric — `ME(A,B) ≠ ME(B,A)`. Always compute both directions and return the average:
    > `score = (ME(A, B) + ME(B, A)) / 2.0`

*   **[Dim 3] Normalized Levenshtein Similarity:** 
    > **Implementation Rule:** Must be a *similarity* (1.0 = identical) to maintain XGBoost monotonic constraints. 
    > `1.0 - (jellyfish.levenshtein_distance(A, B) / max(len(A), len(B)))`

*   **[Dim 4] Exact Name Match:** Binary `1` or `0` if the `name_for_bm25` strings are a perfect 1:1 identical match.

### B. Phonetic Encoding (computed on `name_for_bm25`)

*   **[Dim 5] Name Phonetic Match (Word-Level Set Intersection):**
    > **Implementation Rule:** Double Metaphone operates on **single words**, not phrases. Tokenize first.
    > 1. Tokenize `name_A_bm25` → list of words → compute Double Metaphone hash (primary code only via `pyphonetics.DoubleMetaphone().phonetics(word)`) per word → form `setA`. *Note: Secondary code is omitted due to library API constraints.*
    > 2. Tokenize `name_B_bm25` → same process → form `setB`.
    > 3. Return `1` if `setA & setB` is non-empty, else `0`.

### C. Acronym Match Score (computed on `name_for_bm25`)

*   **[Dim 6] Acronym Score:**
    > **Implementation Rules:**
    > 1. If `len(tokens_of_longer_string) <= 1`: return `0.0` (no meaningful acronym from a single word).
    > 2. If both strings have the same number of words (equal length tie): compute initials of A vs B AND initials of B vs A; return `max(jw(initials_A, B), jw(initials_B, A))`.
    > 3. Normal case: return `jw(initials_of_longer, shorter_string)`.

### D. Semantic Embedding Score

*   **[Dim 7] Name Semantic Cosine:** Dot product / cosine similarity between the two pre-computed `name_for_faiss` embedding vectors from Phase 2 (MiniLM). Reuses existing embeddings — no re-inference needed.
    > **Implementation Rule:** The FAISS `IndexFlatIP` dot product only equals cosine similarity if vectors are L2-normalized. Embeddings stored in FAISS and used in Dim 7 **must be L2-normalized** via `faiss.normalize_L2(embeddings)` before index insertion or dot product calculation.
    > **Safety Rule:** To preserve the `[0, 1]` feature range guarantee across all metrics, clamp negative cosine similarities to 0: `score = max(0.0, dot_product)`.

---

## 3. Address-Based Features (Lexical & Numeric)

All address features computed on `addr_for_bm25`.

### A. Structural Address Overlap & Containment

> **Implementation Rule (Tokenizer):** For all word-level set operations in this section, tokenize using Python's `.split()` on the whitespace-cleaned `addr_for_bm25` string. Preprocessing v2.0 guarantees the string is already punctuation-stripped and space-normalized.

> **Sentinel Guard:** If either address `.startswith("nulladdr")`, the module MUST immediately return `0.0` for all 3 dimensions (Dims 8, 9, 10). This strictly prevents the `nulladdr` prefix from falsely intersecting and generating artificially high Jaccard/Containment scores (e.g., 50% overlap).

*   **[Dim 8] Addr Token Jaccard:** Word-level `|intersection| / max(|union|, 1)`. Robust to reordered components but penalizes partial addresses. The `max(..., 1)` math clamp permanently eliminates zero-division risks.

*   **[Dim 9] Addr Token Containment (Subset Score):** Word-level `|intersection| / max(min(|A|, |B|), 1)`. Ensures a strict partial address (e.g., `123 Main St` vs `123 Main St, New York NY`) scores `1.0`.

*   **[Dim 10] Exact Address Match:** Binary `1` or `0` if the `addr_for_bm25` strings are a perfect 1:1 identical match.

### B. Numeric Overlap Features (Country-Agnostic)

We cannot hardcode Zip Code / PIN code regexes because France follows different postal rules. Use a generic approach.

*   **The Approach:** Use `re.findall(r'\d+', addr_bm25)` to extract all contiguous numeric strings into sets for both Address A and Address B.
    > **Sentinel Guard:** If either address `.startswith("nulladdr")`, the module MUST immediately return `0.0` for Dim 11, `-1` for Dim 12, and `0` for Dim 13. This prevents extracting the sentinel `<id>` as a legitimate address token.

*   **[Dim 11] Addr Numeric Jaccard:** `|intersection| / |union|` of the extracted number sets.
    > **Safety Rule:** If both addresses contain no numbers (`len(union) == 0`), explicitly return `0.0` to avoid a `ZeroDivisionError`.

*   **[Dim 12] Addr Numeric Exact (Ternary Feature):**
    > **Note:** This is a 3-value ordinal feature — explicitly `{1, 0, -1}`. XGBoost handles this natively via splits at `0.5` and `-0.5`.
    > - `1`: Both sets are identical AND non-empty.
    > - `0`: Both sets are non-empty but different.
    > - `-1`: At least one address contains no numbers (common for India landmark-based addresses like *"Near SBI ATM, MG Road"*).

*   **[Dim 13] Addr Numeric Exists Both (Companion Binary Flag):**
    Binary `1` if both addresses contain at least one number, `0` otherwise. Decouples the "do numbers exist" question from the "do they match" question, giving XGBoost a clean root split for the `-1` case above.

---

## 4. Stream Metadata Features

### A. BM25 Scores (Analytical Self-Normalized)

*   **[Dim 14] Name BM25 Score:** `name_bm25_score / max(compute_bm25_self_score(query_tokens, bm25_name_index), 1e-9)`

*   **[Dim 15] Addr BM25 Score:** `addr_bm25_score / max(compute_bm25_self_score(query_tokens, bm25_addr_index), 1e-9)`

    > **`compute_bm25_self_score()` — Required Implementation:**
    > This function does NOT exist in `rank-bm25`. It must be implemented in `feature_extractor.py` using the `BM25Okapi` object's internal parameters. It computes the score the corpus's BM25 index would assign if the query document were itself present as a target — anchoring normalization to query complexity, not to corpus best-match.
    >
    > ```python
    > from collections import Counter
    >
    > def compute_bm25_self_score(bm25_index, query_tokens: list[str]) -> float:
    >     """
    >     Computes the virtual BM25 self-score of the query using the
    >     target corpus's IDF values (from BM25Okapi internals).
    >     Uses: bm25_index.idf, bm25_index.k1, bm25_index.b, bm25_index.avgdl
    >     """
    >     if not query_tokens:
    >         return 1e-9
    >     score = 0.0
    >     doc_len = len(query_tokens)
    >     tf_counter = Counter(query_tokens)
    >     for token in set(query_tokens):
    >         if token not in bm25_index.idf:
    >             continue  # OOV token: contributes 0
    >         idf = bm25_index.idf[token]
    >         tf = tf_counter[token]
    >         numerator = tf * (bm25_index.k1 + 1)
    >         denominator = tf + bm25_index.k1 * (
    >             1 - bm25_index.b + bm25_index.b * doc_len / bm25_index.avgdl
    >         )
    >         score += idf * (numerator / denominator)
    >     return max(score, 1e-9)
    > ```

*   **[Dim 16] 3-Stream RRF Score:** `1/(config.RRF_K_BM25_NAME + bm25_name_rank) + 1/(config.RRF_K_BM25_ADDR + bm25_addr_rank) + 1/(config.RRF_K_FAISS + faiss_rank)`. If a candidate was not retrieved by a given stream, that stream contributes `0` to the sum.

*   **[Dim 17] Stream Overlap Count:** Integer `{0, 1, 2, 3}` — how many distinct streams retrieved this candidate pair during Phase 2.

---

## 5. Structural & Missing Data Features

> **Implementation Rule (Units):** For Dims 18 and 19, use **character-level length** (`len(string)`), not word count. Character ratio explicitly signals acronym patterns (e.g., `len("IBM") / len("International Business Machines")`).

*   **[Dim 18] Name Length Ratio:** `min(len(A), len(B)) / max(len(A), len(B))`

*   **[Dim 19] Addr Length Ratio:** Same formula for addresses.

> **Missing Data Flags — Design Note:** The two combined flags from v4.0 have been split into four independent flags. This is critical because "S1 name is missing" is a property of the *query* (same value for all 60 candidates of that entity), while "Candidate name is missing" is a property of the *specific candidate pair*. Conflating these forces XGBoost to waste tree depth deducing which source is missing.

*   **[Dim 20] S1 Name Is Missing:** Binary `1` if the S1 entity's `name_for_bm25` `.startswith("nullname")`, else `0`.

*   **[Dim 21] Cand Name Is Missing:** Binary `1` if the candidate's `name_for_bm25` `.startswith("nullname")`, else `0`.

*   **[Dim 22] S1 Addr Is Missing:** Binary `1` if the S1 entity's `addr_for_bm25` `.startswith("nulladdr")`, else `0`.

*   **[Dim 23] Cand Addr Is Missing:** Binary `1` if the candidate's `addr_for_bm25` `.startswith("nulladdr")`, else `0`.

---

## 6. Meta-Features

*   **[Meta 24] Cross-Script Target:** 
    > **Implementation Rule:** Inherits `is_cross_script` from Phase 0 Preprocessing. Computed as `max(S1.is_cross_script, Cand.is_cross_script)`. This correctly fires `1` if *either* entity in the pair is originally in a non-Latin script, teaching XGBoost that transliterated matches might have inherently lower string similarities.

*   **[Meta 25] Source Origin:** One-hot encoded (S2 = `0`, S3 = `1`). Allows the tree model to learn source-specific noise levels.

*   **[Meta 26] Country (Raw String — Encoded by `trainer.py` Pipeline):**
    > **Critical Architectural Note:** `feature_extractor.py` has **no access to target labels `y`**. `TargetEncoder` is a supervised encoder — it physically cannot be applied inside the Cyclic DAG. This feature must be emitted as a **raw country string** (e.g., `"us"`, `"india"`, `"france"`) from `feature_extractor.py`.
    >
    > **Ownership:** `trainer.py` wraps encoding inside a sklearn `Pipeline([ColumnTransformer([TargetEncoder(...)]), XGBClassifier(...)])`. `Pipeline.fit(X_train_fold, y_train_fold)` automatically ensures the encoder only ever sees training-fold labels — zero leakage, by construction.
    >
    > **Unseen Country Fallback (France):** `TargetEncoder(target_type='continuous', cv=config.TARGET_ENCODER_CV)` automatically applies the global mean match rate for any unseen country. France requires zero special-casing.

---

## 7. Canonical Feature Summary Table (26-Dim)

| Dim | Feature Name | Input Column | Library / Method |
| :--- | :--- | :--- | :--- |
| 1 | Name Jaro-Winkler Similarity | `name_for_bm25` | `jellyfish.jaro_winkler_similarity` |
| 2 | Name Monge-Elkan Distance (Symmetric Avg) | `name_for_bm25` | `textdistance.MongeElkan` — average ME(A,B) and ME(B,A) |
| 3 | Name Levenshtein Similarity | `name_for_bm25` | `1.0 - (jellyfish.levenshtein_distance / max_len)` |
| 4 | Exact Name Match | `name_for_bm25` | String equality |
| 5 | Name Phonetic Match (Word-Level Set Intersection) | `name_for_bm25` | `pyphonetics` Double Metaphone (primary code) per word → set intersect |
| 6 | Acronym Score (Symmetric, Edge-Case Safe) | `name_for_bm25` | `jellyfish.jaro_winkler_similarity(initials, other)` |
| 7 | Name Semantic Cosine | `name_for_faiss` | L2-normalized MiniLM embedding dot product (clamped `max(0, x)`) |
| 8 | Addr Token Jaccard | `addr_for_bm25` | `.split()` tokenize → `\|intersect\| / \|union\|` |
| 9 | Addr Token Containment | `addr_for_bm25` | `.split()` tokenize → `\|intersect\| / min(\|A\|, \|B\|)` |
| 10 | Exact Address Match | `addr_for_bm25` | String equality |
| 11 | Addr Numeric Jaccard | `addr_for_bm25` | `re.findall(r'\d+')` → set IoU (guard `0/0`) |
| 12 | Addr Numeric Exact (Ternary: 1/0/-1) | `addr_for_bm25` | Set comparison with sentinel |
| 13 | Addr Numeric Exists Both | `addr_for_bm25` | Binary: both sets non-empty? |
| 14 | Name BM25 Score (Self-Normalized) | `name_for_bm25` | `rank-bm25` score / `compute_bm25_self_score()` |
| 15 | Addr BM25 Score (Self-Normalized) | `addr_for_bm25` | `rank-bm25` score / `compute_bm25_self_score()` |
| 16 | 3-Stream RRF Score | — | `1/(config.RRF_K_BM25_NAME+r_name) + ...` |
| 17 | Stream Overlap Count | — | Integer `{0,1,2,3}` from blocking metadata |
| 18 | Name Length Ratio | `name_for_bm25` | `min(len_A, len_B) / max(len_A, len_B)` (character units) |
| 19 | Addr Length Ratio | `addr_for_bm25` | Same formula |
| 20 | S1 Name Is Missing | `name_for_bm25` (S1) | `.startswith("nullname")` |
| 21 | Cand Name Is Missing | `name_for_bm25` (Cand) | `.startswith("nullname")` |
| 22 | S1 Addr Is Missing | `addr_for_bm25` (S1) | `.startswith("nulladdr")` |
| 23 | Cand Addr Is Missing | `addr_for_bm25` (Cand) | `.startswith("nulladdr")` |
| 24 (Meta) | Cross-Script Target | `is_cross_script` | `max(S1.is_cross_script, Cand.is_cross_script)` |
| 25 (Meta) | Source Origin | `source` | One-hot: S2=0, S3=1 |
| 26 (Meta) | Country (raw string) | `country` | Raw string output from `feature_extractor.py` — `TargetEncoder` owned by `trainer.py` `Pipeline` |
