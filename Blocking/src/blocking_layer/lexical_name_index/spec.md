# Lexical Name Index Module Specification

## 1. Description
This module is responsible for exact lexical matching on entity names using the BM25Okapi algorithm. It captures typos and abbreviation variants. Crucially, this module fully encapsulates BM25 indexing, its own internal ID mapping, and provides raw scoring interfaces to strictly adhere to the Progressive Cyclic DAG orchestration architecture.

## 2. EARS Requirements

### Ubiquitous Requirements
*   **REQ-LNI-01**: The `lexical_name_index` module shall implement the `BM25Okapi` algorithm (from `rank-bm25`) for scoring name candidates. The module MUST expose an `__init__(self, k1=None, b=None)` method. If `k1` or `b` are not explicitly passed, they MUST be loaded from `.env` variables. These hyperparameters must be explicitly passed into the `BM25Okapi` constructor to allow ML hyperparameter tuning.
*   **REQ-LNI-02**: The `lexical_name_index` module shall internally manage the mapping between the BM25 integer indices and the actual string `entity_id`s, ensuring absolute alignment.
*   **REQ-LNI-03**: The `lexical_name_index` module shall guarantee that all returned candidate ranks are strictly **1-indexed** (1, 2, 3...) to support the downstream 3-stream RRF Score calculation.
*   **REQ-LNI-04**: The `lexical_name_index` module shall expose a `get_self_score(query_tokens: list[str]) -> float` method that calculates the BM25 score of a query against itself. **Zero-Vocabulary Defense**: If the internal `_is_empty` flag is set, this method must immediately return a minimum floor of `1e-9`. Otherwise, it MUST delegate this to `calculate_self_score` by dynamically passing `self._bm25.idf`, `self._bm25.avgdl`, `self._bm25.k1`, and `self._bm25.b` to guarantee absolute mathematical parity, returning a minimum floor of `1e-9`.

### Event-Driven Requirements
*   **REQ-LNI-05**: When initialized via a `build(ids, texts)` method with target entities, the module shall strictly consume the `to_char_3grams` function (imported from `blocking_utils.py`) to tokenize the `texts` array only during the `build()` method. The `search()` method must explicitly accept a pre-computed `query_tokens: list[str]` and MUST NOT attempt to re-tokenize the query. The module must *not* implement any bespoke tokenization or edge-case string handling internally. It shall store the `ids` list internally. **Generator OOM Defense**: The developer MUST pass a lazy Python generator expression (e.g., `(to_char_3grams(t) for t in texts)`) directly to the `BM25Okapi` constructor to prevent materializing a massive list of tokenized documents in RAM. **Memory Leak Defense**: The module MUST NOT retain the raw `texts` list as an internal state variable (e.g., `self.texts = texts`) after the `build()` method completes, as this would permanently anchor a massive object in memory. **Zero-Vocabulary Defense (Generator-Safe Pattern)**: Because inspecting a generator for emptiness exhausts it, the developer MUST pass the generator to `BM25Okapi` inside a `try...except` block. If instantiation throws an exception (e.g., `ZeroDivisionError`), OR if instantiation succeeds but `len(self._bm25.idf) == 0`, the module MUST set an internal `_is_empty = True` flag and cleanly bypass further BM25 operations. Otherwise, it shall ensure `_is_empty = False`.
*   **REQ-LNI-06**: When queried via a `search(query_tokens, k)` method with an S1 query (provided as a pre-computed `list[str]` of 3-grams), the module shall return a standardized list of candidate dictionaries. **Empty Query Trap Defense**: If `query_tokens` is empty, the method MUST immediately short-circuit and return `[]` to prevent executing useless $O(N \log N)$ numpy allocations and sorts. **Zero-Vocabulary Defense**: If the `_is_empty` flag is set, this method must immediately return `[]`. Otherwise, to bypass `rank-bm25`'s limitation (where `get_top_n` discards scores), this MUST be implemented by calling `.get_scores(query)` over the entire corpus. **Argsort Bottleneck Defense**: Crucially, to prevent sorting millions of zeros, the method MUST strictly filter the scores using a boolean mask (e.g., `valid_idx = np.where(scores > 0.0)[0]`) *BEFORE* applying `np.argsort(-valid_scores, kind='stable')[:k]`. **Reproducibility Defense**: The use of negative valid scores with `kind='stable'` is strictly mandated to ensure that exact score ties are deterministically broken by their original Target_C corpus order, completely preventing non-deterministic RRF feature jitter across runs. This explicitly maps only the non-zero top candidates to their scores and 1-indexed ranks, ensuring `[]` is naturally returned for zero-match queries.

### State-Driven Requirements
*   **REQ-LNI-07**: While a query is being executed, the module shall limit the number of returned candidates to a maximum of `k`.

### Unwanted Behavior Requirements
*   **REQ-LNI-08**: If the requested `k` exceeds the total number of valid matching documents in the corpus, the module must gracefully return all available items. The developer MUST rely on the native Out-Of-Bounds safety of NumPy slice notation (i.e., `np.argsort(-valid_scores, kind='stable')[:k]`) which handles `k > len(valid_scores)` automatically. The downstream ID-mapping logic must dynamically iterate over the resulting slice length rather than hardcoding a loop to `k`.
*   **REQ-LNI-09**: If `build(ids, texts)` is called where `len(ids) != len(texts)`, then the module shall raise an `AlignmentError` to prevent silent misalignments.

## 3. Schemas

### Input Schema (Initialization - `__init__(k1, b)`)
*   **k1**: `float` (BM25 k1 hyperparameter. If not explicitly provided, MUST be loaded from `.env` variables).
*   **b**: `float` (BM25 b hyperparameter. If not explicitly provided, MUST be loaded from `.env` variables).

### Input Schema (Build Index - `build(ids, texts)`)
*   **ids**: `list[string]` (The `entity_id`s of the Target_C dataset)
*   **texts**: `list[string]` (The raw `name_for_bm25` strings of the Target_C dataset)

### Input Schema (Query - `search(query_tokens, k)`)
*   **query_tokens**: `list[string]` (The pre-computed 3-gram tokens of the S1 query `name_for_bm25`)
*   **k**: `int` (The maximum number of results to return)

### Output Schema (Query Results - `search()`)
```json
[
  {
    "id": "string",
    "score": "float",
    "rank": "int" 
  }
]
```
*(Note: `rank` is strictly 1-indexed)*

### Output Schema (Self Score - `get_self_score()`)
*   **score**: `float` (The raw BM25 self-score (to be used downstream as a normalization denominator), bounded to a minimum of 1e-9)

## 4. Errors & Exceptions
*   `AlignmentError`: Raised during `build()` if the length of `ids` does not perfectly match the length of `texts`.
*   `InvalidKError`: Raised if `k` is zero or negative during a query.
