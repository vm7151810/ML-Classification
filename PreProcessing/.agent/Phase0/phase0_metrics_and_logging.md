# Phase 0 — Metrics & Logging

Goal: every run of Phase 0 should leave behind a comparable record of what happened, so a
later change to the cleaning logic can be checked against the previous run rather than
just eyeballed. Keep this lightweight — plain structured files, no heavyweight tracking
infrastructure needed at this stage.

## What to record, per run
Write one JSON record per pipeline run (e.g. `metrics/phase0/<run_id>.json`), and append a
copy to a running `metrics/phase0/history.jsonl` for easy diffing across runs:

- `run_id` (timestamp or short hash) and `pipeline_version` — bump the version string
  whenever cleaning logic changes meaningfully; it's the anchor for comparing runs.
- Per source file: input row count, output row count (should match — Phase 0 must not
  drop rows).
- Per country partition: row count for Query and Target.
- `is_cross_script` rate, broken down by country and by source (S1/S2/S3).
- `script` distribution (counts per detected script), by country.
- `has_address` rate, by country and by source.
- Transliteration coverage: of all `is_cross_script` rows, how many had `name_for_bm25`
  actually overwritten by a registered handler vs. passed through unhandled — broken
  down by `script`. This is the signal for noticing a new script family needs a handler
  added; an unhandled family should never go unnoticed between runs. Also track how many
  overwritten rows produced an empty/degenerate output (should be ~0).
- Name/address length percentiles (P25/P50/P75) post-cleaning, by country — for
  comparing against the EDA's pre-cleaning numbers to catch over-aggressive stripping.
- Wall-clock time per stage (load / clean / partition / write) — useful once this needs
  to scale or be optimized.

## Why this shape
- Per-country and per-source breakdowns matter more than a single aggregate number,
  because the known risks (cross-script gap, missing addresses) are concentrated in
  specific slices, not spread evenly — an aggregate could look fine while a slice is
  broken.
- `pipeline_version` plus the history log is what makes "easy to upgrade later" actually
  true: when the cleaning logic changes, you can see exactly which numbers moved and by
  how much, instead of re-deriving everything from scratch.
- This is intentionally separate from the pass/fail validation checks (see
  `phase0_validation_checks.md`) — metrics describe what happened, validation judges
  whether it's acceptable. Keeping them apart lets you inspect drift across many
  iterations without every run needing to "pass" anything.
