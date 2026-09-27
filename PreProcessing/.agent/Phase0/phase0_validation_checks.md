# Phase 0 — Validation & Test Checklist

Three tiers, in increasing order of subjectivity. Tier 1 failures block moving to
Phase 1. Tier 2 failures should warn loudly but can be judgment calls. Tier 3 has no
automated pass/fail — it's a required manual step.

## Tier 1 — Structural correctness (hard fail, zero tolerance)
- Sum of per-country output row counts equals the input file's row count, for every
  source file. No silent row drops.
- `entity_id` is unique within every output file.
- No `entity_id` appears in more than one country partition.
- `name_for_faiss` and `name_for_bm25` are never null/empty for any row (since
  `business_name` has 0% missingness in the raw data per the EDA — an empty output here
  means the cleaning step over-stripped a real name down to nothing).
- No replacement characters (U+FFFD) or obvious encoding corruption introduced by the
  pipeline that wasn't present in the raw text.

## Tier 2 — Distributional regression checks (compare against EDA baselines, tolerance bands)
- `is_cross_script` rate for India rows should land near the EDA's measured ~5–10% (S2) /
  ~3–5% (S3) — flag if it falls well outside that, since it likely means the script
  detector regressed. This check only applies where a baseline exists (US/India); France
  has no baseline to compare against and should just be inspected qualitatively.
- `has_address` missing-rate for S2/S3 should land near the EDA's ~3.3% — flag if
  materially different.
- Name length percentiles post-cleaning should be close to the EDA's pre-cleaning
  percentiles (§5) — a large drop in median length suggests the cleaning regex is too
  aggressive and eating real content, not just noise.
- Transliteration should touch only `is_cross_script = true` rows, and only those whose
  detected `script` has a registered handler — it must never touch a Latin-script row,
  and must never silently skip or error on a handled-family row. The count of overwritten
  rows should equal the count of cross-script rows *with* a registered handler for their
  `script` (not the full cross-script count) — any gap must be entirely accounted for by
  scripts with no handler yet, and that gap should show up in
  `phase0_metrics_and_logging.md`, not just disappear.

Suggested default tolerance: flag anything that moves more than ~20% relative to the EDA
baseline for manual review. Treat this as a starting point to tighten or loosen once
you've seen a few runs, not a fixed law.

## Tier 3 — Manual quality spot-check (required every run, not automatable)
- Build a small golden sample by pulling real example strings directly from the EDA's own
  noise catalog (§7.1, §7.2 — brackets, leading punctuation, URL-in-name, repeated words,
  NULL/N/A literals, `#`/`##` prefixes) and confirm the pipeline cleans each pattern the
  way a human would expect. This is cheap to build (the EDA already lists the exact
  strings) and should be re-run and extended every time the cleaning logic changes, so it
  doubles as a regression suite.
- Export ~30–50 random `is_cross_script = true` rows with `name_original`,
  `name_for_faiss`, and `name_for_bm25` side by side, and read through them — this is the
  only way to sanity-check transliteration quality, since there's no automatic ground
  truth for "is this a good transliteration."

## Minimum bar to consider Phase 0 done
All Tier 1 checks pass exactly. Tier 2 checks are within tolerance, or any deviation has
an understood explanation (e.g., France naturally differing from the India baseline).
Tier 3 manual review has been done at least once, even if it's a confidence check rather
than a hard pass/fail.
