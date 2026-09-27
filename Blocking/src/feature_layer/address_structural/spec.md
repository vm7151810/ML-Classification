# Address Structural Feature Spec (Dims 8-10)

## 1. Specifications & Schemas
**Inputs:**
- `addr_a` (String): The `addr_for_bm25` of Entity A.
- `addr_b` (String): The `addr_for_bm25` of Entity B.

**Outputs:**
- `dim_8_addr_jaccard` (Float): [0.0, 1.0]
- `dim_9_addr_containment` (Float): [0.0, 1.0]
- `dim_10_addr_exact_match` (Int): {0, 1}

## 2. Requirements (Business Logic)
- **Sentinel Short-Circuit:** If `addr_a.startswith("nulladdr")` or `addr_b.startswith("nulladdr")`, the module MUST immediately return `0.0` (or `0` for ints) for ALL 3 dimensions. This absolutely prevents the shared `"nulladdr"` token from artificially intersecting.
- **Dim 8 (Jaccard):**
  - Tokenize both addresses using exactly `set(addr.split())` (no arguments) to natively drop contiguous whitespace.
  - Return `len(intersect) / max(len(union), 1)`. The `max(..., 1)` guards against `ZeroDivisionError`.
- **Dim 9 (Containment):**
  - Tokenize both addresses using exactly `set(addr.split())`.
  - Return `len(intersect) / max(min(len(set_a), len(set_b)), 1)`.
- **Dim 10 (Exact Match):** Return `1` if `addr_a == addr_b`, else `0`.

## 3. Assumptions
- Inputs are lowercased and cleaned of special characters (handled by Preprocessing phase).

## 4. Edge-Cases & Errors
- **Sentinel Token Collision:** The sentinels `nulladdr <id>` share the `nulladdr` prefix. Without the short-circuit, token sets would perfectly intersect on `"nulladdr"`, generating 50% false-positive containment scores.
- **ZeroDivisionError:** If upstream leaks an empty string, the token sets are empty, resulting in a zero denominator. The explicit `max(..., 1)` clamping ensures mathematical immunity.
- **Whitespace Token Injection:** Using `addr.split(' ')` preserves empty strings `""` on double-spaces, causing false empty-string intersections. The spec strictly mandates `set(addr.split())` to inherently purge whitespace.
