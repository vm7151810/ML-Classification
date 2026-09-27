# Name Phonetic Feature Spec (Dim 5)

## 1. Specifications & Schemas
**Inputs:**
- `name_a` (String): The `name_for_bm25` of Entity A (S1).
- `name_b` (String): The `name_for_bm25` of Entity B (Candidate).

**Outputs:**
- `dim_5_phonetic_match` (Int): {0, 1}

## 2. Requirements (Business Logic)
- **Dim 5:** Double Metaphone word-level set intersection.
  - **Sentinel Short-Circuit:** If `name_a.startswith("nullname")` or `name_b.startswith("nullname")`, immediately return `0` (prevents false positive matches on missing data).
  - Tokenize both strings into lists of words.
  - Apply `pyphonetics.DoubleMetaphone().phonetics(word)[0]` to extract **only the primary code** for each word.
  - Form sets of phonetic codes for A and B.
  - **Empty Hash Purging:** Explicitly discard any empty strings (`""`) from both sets (prevents false positive matches on numbers/symbols).
  - Return `1` if the sets intersect (`len(setA & setB) > 0`), else `0`.

## 3. Assumptions
- Both inputs are already Latin-transliterated; Double Metaphone is undefined for non-Latin characters.
- Secondary phonetic codes from `DoubleMetaphone` are deliberately ignored per architectural constraints.

## 4. Edge-Cases & Errors
- **Sentinel Phonetic Collisions:** Hashing the `nullname` sentinel produces a phonetic hash. Without the short-circuit, two missing names would produce a spurious correlation.
- **Empty Hash Collisions:** Double Metaphone yields `""` for pure numbers. Without discarding `""`, two completely different numbers (e.g., "123" and "999") would falsely trigger a phonetic match.
- **No valid phonetic codes:** If words contain only numbers or un-hashable chars, filtering `""` will leave empty sets. `setA & setB` will correctly evaluate to empty and return `0`.
- **Tuple Indexing:** The `pyphonetics` API returns a tuple `(primary, secondary)`. The specification explicitly mandates `[0]` indexing to avoid pushing the tuple object into the sets.
