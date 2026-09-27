# Phase 0 — Preprocessing & Country Partitioning: Implementation Plan

## Objective
Clean and partition the raw Source 1/2/3 TSV records into a uniform, country-partitioned
schema that Phase 1 (candidate generation) can consume directly. This phase does **not**
touch ground truth, does **not** compute candidate pairs, does **not** build any index
(FAISS/BM25), and does **not** parse city/state/PIN from addresses. Those belong to later
phases — see "Explicitly out of scope" below.

## Inputs
- `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`
- `test_source1.tsv`, `test_source2.tsv`, `test_source3.tsv`
- Columns: `entity_id`, `business_name`, `business_address`, `country` (tab-separated)
- See attached `data_audit_eda.md` for baseline statistics (row counts, missingness rates,
  noise-pattern frequencies, script distribution). Several validation checks compare
  against these numbers.
- See attached transliteration validation results for the OPTITRANS scheme decision and
  its measured behavior at the word-token vs. char-trigram level.

## Outputs
Written to `data/processed/{train,test}/`:
- `Query_<country>.parquet` — cleaned Source 1 records for that country
- `Target_<country>.parquet` — cleaned Source 2 + Source 3 records for that country, unioned

`<country>` is **never** a fixed/hardcoded list — partition dynamically over whatever
distinct values appear in the `country` column of the file being processed. This must
keep working unchanged if a new country shows up in future data.

Query and Target files share an **identical schema**. See `phase0_schema_contract.md` for
the authoritative column definitions — do not add a `split` column or any
city/state/PIN-derived columns here.

## Processing steps
(Conceptual sequence — the implementer has latitude on internals, library choices, and
exact regex/logic for each cleaning sub-step.)

1. **Load.** Lazy/streaming read per source file. Confirm the row count read matches the
   known count for that file (from the EDA) before proceeding — a cheap parse-integrity
   check, not a design decision.
2. **Derive `source`.** Parse the S1-/S2-/S3- prefix from `entity_id`. Assert it's
   consistent with which file the row came from.
3. **Detect script.** Classify `business_name` by Unicode Script property (open-set — do
   not hardcode a Devanagari/Tamil/Telugu/... enum). Produce `is_cross_script`
   (Latin vs. not) and `script` (the detected script name).
4. **Flag address presence.** Compute `has_address` from the *raw* `business_address`,
   before any cleaning — it should reflect whether the source supplied real data, not
   whether cleaning later emptied the field. Treat literal `NULL`/`N/A` strings and true
   nulls/empty strings the same way here.
5. **Clean name (shared base pass).** Normalize Unicode (NFKC), case-fold, strip the noise
   patterns catalogued in the EDA (§7.1): bracket/paren wrapping, leading punctuation runs,
   embedded URLs, consecutive duplicate tokens, stray symbols. This shared cleaned text
   feeds both derived name fields below. Do **not** strip legal-form suffixes
   (Ltd/Pvt/Inc/SARL/etc.) — that's a retrieval-time IDF concern for Phase 1, not cleaning.
6. **`name_for_faiss`.** The shared cleaned name, native script, unchanged further. Feeds
   the dense/embedding channel (BGE-M3) — no transliteration needed, multilingual
   embedding models take native script directly.
7. **`name_for_bm25`.** The shared cleaned name for Latin-script rows, unchanged. For
   `is_cross_script = true` rows, dispatch to a transliteration handler by detected
   **script family**, rather than calling one library unconditionally:
   - **Indic-family scripts** (Devanagari, Tamil, Telugu, Kannada, Bengali, Gujarati,
     Gurmukhi — the families actually present in this data) → transliterate via
     `indic-transliteration` (OPTITRANS scheme), then re-run the whitespace/case
     normalization pass once more, since transliteration output can introduce its own
     artifacts.
   - **Any other script family** (e.g. Arabic, Cyrillic, CJK — not present in train/test
     today, but the registry must not assume they never will be) → no handler yet; pass
     the cleaned native-script text through unchanged, and record it as an
     unhandled-script row (see `phase0_metrics_and_logging.md`). This is a deliberate,
     logged gap, not a silent one — it's the mechanism by which a future script gets
     noticed and a handler added later, without touching detection logic or this
     dispatch structure.

   This field is intended to be consumed as character-3gram tokens downstream
   (transliteration helps at the trigram level, not the word-token level) — Phase 0
   doesn't need to pre-tokenize into trigrams, just hand over clean, comparison-ready
   text.
8. **Clean address.** Normalize `NULL`/`N/A` literals to true empty, strip `#`/`##`
   house-number prefixes, case-fold, NFKC, collapse whitespace. No reordering, no
   city/state/PIN extraction, no state-abbreviation normalization — all deferred to
   Phase 1. Produce `addr_for_bm25` only; there is no `addr_for_faiss`.
9. **Assemble and partition.** Group rows by the country value actually present; write one
   Query or Target parquet per country, per split (train/test).
10. **Emit metrics.** See `phase0_metrics_and_logging.md`.
11. **Run validation checks.** See `phase0_validation_checks.md`. Phase 0 isn't done until
    these pass at the stated thresholds.

## Design principles to preserve
- Country and script handling must be open-set — no `{US, India}` or
  `{Devanagari, Tamil, ...}` hardcoding anywhere.
- Query and Target schemas stay identical, so downstream code treats rows uniformly
  regardless of role.
- Legal-suffix handling is deliberately *not* done here — it's a retrieval-time (IDF)
  concern.
- Per-country BM25 indices in Phase 1 should compute IDF from their own partition's
  corpus, not a shared/global one — this is why the France partition doesn't need a
  hardcoded French legal-suffix list. Keep Phase 0 output structured so that stays
  possible (never merge countries together anywhere in the output).
- Transliteration is scoped only to `is_cross_script = true` rows — do not run it over
  the full corpus. Within that, only invoke a transliteration library for script
  families that have a registered handler; unhandled families pass through unchanged
  rather than erroring or being force-fit through a mismatched library.

## Explicitly out of scope for Phase 0
- Ground truth joining or any label usage
- Candidate pair generation, FAISS/BM25 index construction
- Train/validation split assignment
- City/state/PIN extraction or address component parsing
- Any model training or inference
