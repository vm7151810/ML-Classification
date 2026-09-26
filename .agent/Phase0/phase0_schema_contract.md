# Data Schema Contract — Query / Target Parquet Files

Authoritative schema for `Query_<country>.parquet` and `Target_<country>.parquet`,
produced by Phase 0 and consumed by every later phase. If a later phase needs a new
derived field, add it there and update this contract — don't retrofit it into Phase 0
without updating this document first.

| Column | Type | Description | Notes |
|---|---|---|---|
| `entity_id` | string | Original entity ID, prefix intact | Primary key within a file |
| `source` | string (`S1`/`S2`/`S3`) | Derived from entity_id prefix | Redundant with entity_id, kept for convenience |
| `country` | string | Original country label | Open-set; drives file partitioning |
| `name_original` | string | Raw `business_name`, untouched | For debugging / feature engineering that needs the original |
| `addr_original` | string | Raw `business_address`, untouched | Same |
| `is_cross_script` | bool | True if `business_name` is not (predominantly) Latin script | Open-set detection via Unicode Script property, not a hardcoded script list |
| `script` | string | Detected dominant script name | e.g. `Latin`, `Devanagari`, `Tamil` — whatever the detector returns, not a fixed enum |
| `has_address` | bool | True if raw address had real content | Computed pre-cleaning; treats NULL/N/A literals as absent |
| `name_for_faiss` | string | Cleaned name, native script | Feeds the dense/embedding channel |
| `name_for_bm25` | string | Cleaned name; transliterated via a script-family handler when one is registered (OPTITRANS for Indic scripts today), otherwise native-script passthrough | Feeds the lexical/char-3gram channel |
| `addr_for_bm25` | string | Cleaned address | Feeds the lexical address channel; no structural parsing |

## Deliberately absent (by design, not oversight)
- `split` (train/val) — assigned later via a separate manifest, never mixed into this file
- Any city/state/PIN column — a Phase 1 concern
- `addr_for_faiss` — address gets no embedding channel (justified by the ~99.4%
  Latin-script rate on addresses even for cross-script name records)

## Invariants any writer of this schema must uphold
- `entity_id` unique within a file
- No row appears in more than one country partition
- `name_for_faiss` / `name_for_bm25` are never null or empty
- Query and Target files for the same country/split share this exact column set — no
  drift between the two
