# BM25 Utils Specification (`blocking_layer/bm25_utils`)

## 1. Description
This module houses pure mathematical operations related to BM25 scoring, ensuring that complex mathematical formulas are strictly isolated from the stateful index wrappers and the feature extraction pipeline.

## 2. EARS Requirements

### Ubiquitous Requirements
*   **REQ-BMU-01**: The `bm25_utils` module shall expose a stateless `calculate_self_score(query_tokens, idf_dict, avgdl, k1, b) -> float` function. These parameters (`idf_dict`, `avgdl`, `k1`, `b`) must be dynamically extracted from the `rank-bm25` instance to guarantee exact mathematical parity with the internal corpus scoring, avoiding any hardcoded constants.

### Event-Driven Requirements
*   **REQ-BMU-02**: When invoked, the function shall calculate the hypothetical BM25 score of the query document against itself, utilizing the provided corpus IDF dictionary and parameters. To guarantee mathematical parity with `rank-bm25`, the calculation loop MUST iterate over every un-deduplicated token in `query_tokens` (or if iterating over unique tokens, must explicitly multiply the resulting term score by the query term frequency). The exact formula must be:
    `term_score = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (len(query_tokens) / avgdl)))`
    *(where `freq` is the term frequency of the token within the `query_tokens` list).*

### Unwanted Behavior Requirements
*   **REQ-BMU-03**: If the `query_tokens` list is empty, the function shall return a minimum floor value of `1e-9`.
*   **REQ-BMU-04**: If the calculated score is `0.0` or negative, the function shall clamp the return value to `1e-9`.
*   **REQ-BMU-05**: If the `query_tokens` contain Out-Of-Vocabulary (OOV) tokens that are not found in the `idf_dict`, the function MUST gracefully default to an IDF of `0.0` (e.g., `idf_dict.get(token, 0.0)`) to prevent `KeyError` crashes and perfectly preserve the zero-contribution mathematical parity.
*   **REQ-BMU-06**: If `avgdl <= 0.0`, the function MUST gracefully return the minimum floor value of `1e-9` to completely protect the stateless mathematical execution against `ZeroDivisionError` crashes.
*   **REQ-BMU-07**: **O(V) Bottleneck Defense**: The function MUST NOT iterate over the entire `idf_dict` (e.g., `for k, v in idf_dict.items()`) for type-checking or validation. Because this function is called inside a hot retrieval loop for every S1 entity, iterating over the global vocabulary would degrade the time complexity per query from $O(L)$ (where $L$ is the query length) to $O(V)$ (where $V$ is the global vocabulary size), causing a catastrophic pipeline freeze. The implementation MUST assume the `idf_dict` is well-formed and restrict all iteration strictly to the `query_tokens` array.

## 3. Schemas

### Input Schema
*   **query_tokens**: `list[str]`
*   **idf_dict**: `dict[str, float]`
*   **avgdl**: `float`
*   **k1**: `float`
*   **b**: `float`

### Output Schema
*   **score**: `float`
