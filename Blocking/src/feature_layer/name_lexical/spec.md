# Name Lexical Feature Spec (Dims 1-4, 6)

## 1. Specifications & Schemas
**Inputs:**
- `name_a` (String): The `name_for_bm25` of Entity A.
- `name_b` (String): The `name_for_bm25` of Entity B.

**Outputs:**
- `dim_1_name_jaro_winkler` (Float): [0.0, 1.0]
- `dim_2_name_monge_elkan` (Float): [0.0, 1.0]
- `dim_3_name_levenshtein_norm` (Float): [0.0, 1.0]
- `dim_4_name_exact_match` (Int): {0, 1}
- `dim_6_name_acronym_match` (Float): [0.0, 1.0]

## 2. Requirements (Business Logic)
- **Sentinel Short-Circuit:** If `name_a.startswith("nullname")` or `name_b.startswith("nullname")`, the module MUST immediately return `0.0` (or `0` for ints) for ALL 5 dimensions. This prevents massive false-positive lexical matches on missing data sentinels.
- **Dim 1 (Jaro-Winkler):** Computed via `jellyfish.jaro_winkler_similarity(A, B)`.
- **Dim 2 (Monge-Elkan):** Both strings must be tokenized into lists of words. 
  - **Empty Token Guard:** Must explicitly verify `if not A_tokens or not B_tokens:` and return `0.0` to prevent library errors on empty arrays.
  - Computed via `textdistance.MongeElkan`. Must calculate the symmetric average: `(ME(A, B) + ME(B, A)) / 2.0`.
- **Dim 3 (Normalized Levenshtein):** Computed as `1.0 - (Levenshtein(A, B) / max(len(A), len(B), 1))`. The `max(..., 1)` guard ensures absolute immunity against `ZeroDivisionError` if upstream leaks an empty string. Use `jellyfish.levenshtein_distance`.
- **Dim 4 (Exact Match):** Return `1` if `A == B`, else `0`.
- **Dim 6 (Acronym Match):**
  - If `max(len(A_tokens), len(B_tokens)) <= 1`: return `0.0`.
  - Extract initials of `A_tokens` into `initials_A`, and `B_tokens` into `initials_B`.
  - If `len(A_tokens) == len(B_tokens)`: return `max(jw(initials_A, B), jw(initials_B, A))`.
  - Else: return `jw(initials_of_longer, shorter_string)`.

## 3. Assumptions
- Inputs are already lowercased and cleaned of special characters (handled by Preprocessing phase).
- `name_for_bm25` strings use standard space delimiters.

## 4. Edge-Cases & Errors
- **Sentinel Similarity Poisoning:** Calculating edit distances on two `nullname` sentinels (e.g., `"nullname 123"` and `"nullname 456"`) yields artificially high similarities. The Sentinel Short-Circuit prevents this poisoning.
- **Levenshtein Zero-Guard:** Even though `nullname` theoretically prevents empty strings, pipeline leaks happen. `max(..., 1)` guarantees absolute math safety.
- **Empty Token Guard:** External distance libraries sometimes lack safety checks for empty inputs. Checking `not A_tokens` prevents unexpected crashes.
