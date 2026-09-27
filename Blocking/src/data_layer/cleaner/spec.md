# Cleaner Module Specification (`data_layer/cleaner`)

## 1. Description
This module performs aggressive multilingual cleaning and normalizes the data into specialized dual representations (one for FAISS, one for BM25). It also partitions the unified dataset by country to prepare for strictly segregated Blocking Layer execution.

## 2. EARS Requirements

### Ubiquitous Requirements
*   **REQ-CLN-01**: The `cleaner` module shall utilize Polars string functions (`.str.replace_all()`, etc.) for all text transformations.
*   **REQ-CLN-02**: The `cleaner` module shall detect Indic scripts via regex (`[\u0900-\u097F]`) and transliterate them to ITRANS phonetic Latin via the `indic-transliteration` library.
*   **REQ-CLN-03**: The `cleaner` module shall generate a binary `is_cross_script` column (`1` if Indic script detected, else `0`).

### Event-Driven Requirements
*   **REQ-CLN-04**: When generating `name_for_faiss`, the module shall preserve the original script, convert to lowercase, and collapse multiple spaces, avoiding any ASCII coercion or punctuation stripping.
*   **REQ-CLN-05**: When generating `name_for_bm25` and `addr_for_bm25`, the module shall apply transliteration, Unicode NFKD ASCII coercion, lowercase conversion, and punctuation stripping (protecting acronyms and mapping specific symbols like `@`).
*   **REQ-CLN-06**: When processing addresses, the module shall expand known abbreviations (e.g., `st` -> `street`) via word-boundary regex.

### State-Driven Requirements
*   **REQ-CLN-07**: While processing any representation, if the resulting string is completely empty `""`, the module shall replace it with the sentinel format `nullname <entity_id>` or `nulladdr <entity_id>` to ensure global uniqueness.

### Unwanted Behavior Requirements
*   **REQ-CLN-08**: If partitioning datasets, the module shall strictly segregate records such that a country-specific partition contains only entities from that country.

## 3. Schemas

### Input Schema
*   **dataframe**: `polars.DataFrame` (Output from loader)

### Output Schema (Partitioned Parquet Files)
*   **entity_id**: String
*   **country**: Categorical
*   **source**: String (`"S1"`, `"S2"`, or `"S3"`)
*   **name_original**: String
*   **addr_original**: String
*   **name_for_faiss**: String
*   **name_for_bm25**: String
*   **addr_for_bm25**: String
*   **is_cross_script**: Int8 (`0` or `1`)

## 4. Errors & Exceptions
*   `PartitioningError`: Raised if the country column contains unexpected or unmapped values.
