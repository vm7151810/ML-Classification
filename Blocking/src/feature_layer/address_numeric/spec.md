# Address Numeric Feature Spec (Dims 11-13)

## 1. Specifications & Schemas
**Inputs:**
- `addr_a` (String): The `addr_for_bm25` of Entity A.
- `addr_b` (String): The `addr_for_bm25` of Entity B.

**Outputs:**
- `dim_11_num_jaccard` (Float): [0.0, 1.0]
- `dim_12_num_exact_match` (Int): {-1, 0, 1}
- `dim_13_num_presence` (Int): {0, 1}

## 2. Requirements (Business Logic)
- **Sentinel Short-Circuit:** If `addr_a.startswith("nulladdr")` or `addr_b.startswith("nulladdr")`, the module MUST immediately return `0.0` for Dim 11, `-1` for Dim 12, and `0` for Dim 13. This explicitly prevents extracting sentinel IDs (e.g., `"0042"`) as valid numeric tokens.
- **Token Extraction:** For both addresses, use `re.findall(r'\d+', addr)` to extract all contiguous numeric strings into sets for both Address A and Address B. Form sets `num_set_a` and `num_set_b`.
- **Dim 11 (Numeric Jaccard):** Return `len(intersect) / max(len(union), 1)`. The `max(..., 1)` guards against `ZeroDivisionError` on numberless addresses.
- **Dim 12 (Numeric Exact Match) [Ternary]:**
  - If `len(num_set_a) == 0 or len(num_set_b) == 0`: Return `-1` (At least one address has no numbers).
  - Else if `num_set_a == num_set_b`: Return `1` (Both non-empty and identical).
  - Else: Return `0` (Both non-empty but different).
- **Dim 13 (Numeric Presence):** Return `1` if `len(num_set_a) > 0` AND `len(num_set_b) > 0`, else `0`.

## 3. Assumptions
- Inputs are lowercased and cleaned.

## 4. Edge-Cases & Errors
- **Sentinel ID Poisoning:** Sentinels take the form `"nulladdr <id>"`. Without the short-circuit, the `<id>` string would be extracted as a legitimate address token, massively poisoning XGBoost splits with false positives.
- **ZeroDivisionError (Math Clamping):** Purely alphabetical addresses have empty numeric sets, guaranteeing a zero union length. `max(len(union), 1)` mathematically eliminates crash risks.
- **Empty Set Exact Match:** `set() == set()` evaluates to True. Dim 12 explicitly employs a `-1` ternary flag if either set is empty, preventing XGBoost from conflating "missing numbers" with "exactly matching numbers".
