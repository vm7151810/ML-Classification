# Semantic Index Module Specification

## 1. Description
This module captures semantic synonyms natively using a pre-trained Multilingual Bi-Encoder (`paraphrase-multilingual-MiniLM-L12-v2`) and FAISS dense vector search. It strictly encapsulates ID mapping and memory safety constraints to adhere to the cyclic DAG architecture.

## 2. EARS Requirements

### Ubiquitous Requirements
*   **REQ-SEM-01**: The `semantic_index` module shall manage a `faiss.IndexFlatIP` (Inner Product) index for similarity search.
*   **REQ-SEM-02**: The `semantic_index` module shall internally manage the mapping between the FAISS integer indices and the actual string `entity_id`s, ensuring absolute alignment.
*   **REQ-SEM-03**: The `semantic_index` module shall guarantee that all returned candidate ranks are strictly **1-indexed** (1, 2, 3...) to support the downstream 3-stream RRF Score calculation.
*   **REQ-SEM-04**: **Symmetric Normalization Math**: The `semantic_index` module MUST strictly enforce L2 normalization on all corpus embeddings prior to FAISS ingestion AND MUST strictly apply L2 normalization to the `query_embedding` during `search()` to mathematically guarantee that the FAISS inner product equates exactly to bounded cosine similarity.
*   **REQ-SEM-12**: **Thread Thrashing Defense**: Upon initialization, the module MUST explicitly execute `faiss.omp_set_num_threads(1)` to strictly disable FAISS internal OpenMP multithreading, allowing the Orchestrator's higher-level multiprocessing DAG to safely scale without catastrophically freezing the host server.
*   **REQ-SEM-11**: To maintain strict duck-typing interface parity with the Lexical indices for downstream scaling, the module MUST expose a `get_self_score(query_embedding) -> float` method. Because L2-normalized inner products equate to Cosine Similarity, this method MUST return a hardcoded constant `1.0` (or `1e-9` if empty) to satisfy the DAG interface without executing complex math.

### Event-Driven Requirements
*   **REQ-SEM-05**: When initialized via `build(ids, texts)` with target entities, the module shall batch embed the texts and inject them into FAISS, storing the `ids` array internally. **Zero-Vocabulary Defense**: If the corpus is empty, it MUST set `_is_empty = True`, completely bypass FAISS instantiation, and seamlessly allow the DAG to organically skip the stream. **Memory Leak Defense**: The module MUST strictly forbid retaining the raw `texts` list as an internal state variable to prevent anchoring gigabytes of strings in RAM.
*   **REQ-SEM-15**: **VRAM Hoarding Defense**: The `SentenceTransformer` model (`paraphrase-multilingual-MiniLM-L12-v2`) MUST be instantiated strictly as a local variable INSIDE the `build()` method. It MUST NOT be instantiated in `__init__()`. Because `search()` accepts pre-computed embeddings and `load()` simply reads a pre-built FAISS index, instantiating the transformer in `__init__()` would permanently permanently hog hundreds of megabytes of GPU VRAM during the inference phase when the model is no longer needed.
*   **REQ-SEM-13**: When `save(dir_path)` is called, the module MUST use `faiss.write_index()` to persist the C++ FAISS index and standard I/O to persist the internal `ids` array, bypassing Python `pickle` limitations.
*   **REQ-SEM-14**: When `load(dir_path)` is called, the module MUST natively reconstruct the state using `faiss.read_index()` to instantly bypass the multi-hour embedding bottleneck upon pipeline restart.
*   **REQ-SEM-06**: When calling `sentence-transformers` `.encode()`, the module shall explicitly use `convert_to_numpy=True` to flush embeddings from VRAM to system CPU RAM, preventing out-of-memory (OOM) errors.
*   **REQ-SEM-07**: When queried via `search(query_embedding, k)` with a pre-computed S1 query vector, the module shall return a standardized list of candidate dictionaries. **Empty Query Trap Defense**: If `query_embedding` is all zeros or `None`, the method MUST immediately short-circuit and return `[]`, completely preventing fatal `NaN` normalization division errors and bypassing massive useless FAISS computations on empty strings. **FAISS C++ Segfault Defense**: The module MUST aggressively cast the `query_embedding` to `np.float32` and safely enforce a `.reshape(1, -1)` before querying FAISS, mathematically guaranteeing absolute C++ memory safety regardless of upstream Orchestrator NumPy shape or type jitter.

### State-Driven Requirements
*   **REQ-SEM-08**: While querying FAISS, the module shall strictly enforce `k <= faiss_index.ntotal` to prevent C++ core assertion crashes.

### Unwanted Behavior Requirements
*   **REQ-SEM-09**: If `build(ids, texts)` is called where `len(ids) != len(texts)`, then the module shall raise an `AlignmentError` to prevent silent misalignments.
*   **REQ-SEM-10**: If an empty string `""` or pure whitespace is provided in the `texts` array during `build()`, the module MUST strictly rely on the native default behavior of `SentenceTransformer.encode()` (which deterministically outputs a valid non-zero embedding for the `[CLS]` token). The developer MUST NOT manually intercept empty strings to generate random or zero vectors, as doing so would cause fatal `NaN` errors during L2 normalization and break the mathematical determinism of the pipeline.

## 3. Schemas

### Input Schema (Build Index - `build(ids, texts)`)
*   **ids**: `list[string]` (The `entity_id`s of the Target_C dataset)
*   **texts**: `list[string]` (The raw `name_for_faiss` strings of the Target_C dataset)

### Input Schema (Disk I/O - `save(dir_path)` / `load(dir_path)`)
*   **dir_path**: `string` (Path to the directory containing `index.faiss` and `ids.json`/`ids.npy`)

### Input Schema (Query - `search(query_embedding, k)`)
*   **query_embedding**: `numpy.ndarray` (The pre-computed dense vector for the S1 query entity, shape `(1, D)`)
*   **k**: `int` (The maximum number of results to return)

### Output Schema (Query Results - `search()`)
```json
[
  {
    "id": "string",
    "score": "float (The raw cosine similarity score, bounded between [-1.0, 1.0])",
    "rank": "int" 
  }
]
```
*(Note: `rank` is strictly 1-indexed)*

## 4. Errors & Exceptions
*   `AlignmentError`: Raised during `build()` if the length of `ids` does not perfectly match the length of `texts`.
*   `FAISSAssertionError`: Prevented, but theoretically raised if `k` exceeds `ntotal`.
