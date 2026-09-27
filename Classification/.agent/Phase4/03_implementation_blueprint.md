# Phase 4 — Implementation Blueprint

*Part 3 of 4. Previous: `02_phase4_decisions.md`. Next: `04_metrics_logging_testing.md`.*

---

## What this file is, and isn't

This is a blueprint: it describes the shape of what you're building, how the pieces
relate, and what must be true when you're done. It deliberately does **not**
prescribe function signatures, internal helper boundaries, exact naming, or code
structure below the module level — those are your decisions to make as you
implement. Where a specific mechanism is mandated (because it directly implements
one of the numbered decisions in `02_phase4_decisions.md`), that's called out
explicitly and should be followed exactly; everything else here is describing
*what must happen*, not *how to write it*.

---

## Assumed input contract — verify before building around it

Phase 4 consumes a single table, one row per `(S1 entity, candidate)` pair,
containing:

- a grouping key identifying which S1 entity the row belongs to (used for
  cross-validation splitting and for entity-level metric aggregation)
- a separate identifier for the candidate itself, carried along per Decision 9 —
  never fed to the model, only for traceability
- the fixed set of 26 features named and ordered in the canonical feature table (25
  numeric columns plus one raw-string country column)
- a binary match label

**This is an assumption, not a confirmed fact about the real upstream output.**
Before writing the pipeline construction code, inspect a real sample of what the
upstream feature-extraction step actually produces (per the just-in-time attachment
guidance in `00_START_HERE.md`) and confirm the column names and types match. If
something differs, adapt the column-name references only — do not change anything
about the pipeline design, transformer ordering, or constraint-vector alignment to
work around a mismatch. If the mismatch is more than a naming difference, stop and
flag it rather than improvising a fix.

---

## Module layout

A reasonable grouping — feel free to adjust exact filenames or split points, this is
a suggested organization, not a mandate:

- **A metrics module** — the two pure evaluation functions (entity-level scoring,
  class-imbalance weighting), fully specified behaviorally in
  `04_metrics_logging_testing.md`. These should exist in exactly one place and be
  imported everywhere they're needed, never reimplemented inline.
- **A model construction module** — the small selection point described in Decision
  8, returning a constructed classifier for a requested model family.
- **A logging/summary module** — sets up the run logger and writes the final
  machine-readable summary, detailed in `04_metrics_logging_testing.md`.
- **The main training module** — orchestrates the four phases described below and
  produces the final persisted artifact.
- **A tests directory**, mirroring the modules above, detailed in
  `04_metrics_logging_testing.md`.

Extend, rather than replace, whatever configuration loader, environment file, and
dependency manifest already exist in the project — the new settings this phase needs
are already fully enumerated in the original design document's configuration
section; add them there rather than inventing a parallel configuration mechanism.

---

## The four training phases

These are described behaviorally — what must happen and what must be true
afterward — not as a sequence of code to transcribe.

### Phase A — Hyperparameter search

A search process explores the hyperparameter space defined in the original design
document, evaluated via grouped cross-validation (grouped by S1 entity, so a given
entity's candidates never split across train and validation within a fold). Each
trial:

- constructs a model using the fixed, non-tuned settings established in Decisions
  4, 5, 11, and 12, plus that trial's sampled tunable hyperparameters — named per
  Decision 10 from the moment they're first sampled
- trains on the fold's training split
- is evaluated for early abandonment using a validation loss signal, requiring the
  manual encoding described in Decision 2
- reports back the entity-level score (Decision 6: at the fixed threshold, using
  the configurable weighting from Decision 3) as its result, averaged across folds

Nothing beyond each trial's final scalar score needs to be retained once a trial
finishes — no per-trial predictions should accumulate in memory across the whole
search. At the end, the single best-performing hyperparameter set is what carries
forward — and because of Decision 10, it should already be in the exact form needed
by Phase B with no translation step.

### Phase B — Clean cross-validated evaluation

A second, separate cross-validation pass — same grouping, same fold structure
philosophy — this time with the hyperparameters fixed to the Phase A result. No
loss-based early abandonment here; each fold trains to completion. The purpose is to
produce a set of out-of-fold predictions that reflect the chosen hyperparameters
cleanly, uncontaminated by the search process itself (which evaluated many
different, mostly worse, hyperparameter sets along the way). These out-of-fold
predictions — together with their true labels, their grouping keys, and the
candidate identifiers from Decision 9 — are what the next phase calibrates against.

### Phase C — Threshold calibration

Using only the clean out-of-fold predictions from Phase B, sweep the decision
threshold across the range defined in configuration, scoring each candidate
threshold with the same entity-level metric used throughout. The threshold that
scores highest is the one that ships with the final model. This phase should record
the *entire* sweep, not just the winner — the full curve is part of what gets logged
and summarized, per `04_metrics_logging_testing.md`.

### Phase D — Final fit and persistence

Using the same fixed hyperparameters from Phase A, fit one final model on the entire
labeled dataset — no train/validation split at this point, no early abandonment.
Recompute the class-imbalance weighting from the full dataset as a sanity check
against the value computed at the very start of the run (Decision 4) — they should
match; log a warning, not a crash, if they don't. Persist a single artifact
containing: the fitted model itself, the calibrated threshold from Phase C, the
hyperparameters used, and the exact feature ordering the model expects. This last
piece matters — whatever loads this artifact later needs to build its feature matrix
in exactly this order, so it must be recorded precisely and unambiguously.

---

## What's explicitly left to your judgment

- Exact function/class boundaries within each module
- Whether phases A–D are separate functions, a single orchestrating function, or a
  small internal state object — whatever reads clearly
- Variable naming, docstring style, internal helper structure
- Whether to parallelize the cross-validation folds, and how
- Exact error-handling and messaging style, beyond the specific "fail loudly, don't
  silently default" requirements called out in the decisions file (Decision 8's
  unimplemented fallback, Decision 4's imbalance ratio guard, Phase D's mismatch
  warning)

Anything not explicitly constrained by a numbered decision in
`02_phase4_decisions.md` is fair territory for your own judgment. When in doubt,
prefer the simplest implementation that satisfies the behavioral requirements above
and the test requirements in the next file.
