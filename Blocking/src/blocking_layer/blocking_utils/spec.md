# Blocking Utils Specification (`blocking_layer/blocking_utils`)

## 1. Description
This module houses shared stateless utilities for the blocking layer, specifically the tokenizer required for BM25 processing.

## 2. EARS Requirements

### Ubiquitous Requirements
*   **REQ-BUTIL-01**: The `blocking_utils` module shall expose a `to_char_3grams(text: str) -> list[str]` function.

### Event-Driven Requirements
*   **REQ-BUTIL-02**: When provided a string of length >= 3, the `to_char_3grams` function shall return a list of overlapping 3-character substrings.

### Unwanted Behavior Requirements
*   **REQ-BUTIL-03**: The module must expose a `to_char_3grams` function. It must implement three defense mechanisms: (1) Defensive fallback for empty strings returning `[]`, (2) Explicit interception of `nullname/nulladdr` returning `[]` to prevent index pollution, and (3) A fallback returning `[text]` if `len(text) < 3` to protect short queries.

## 3. Schemas

### Input Schema
*   **text**: `str`

### Output Schema
*   **tokens**: `list[str]`
