# Phase 4 — Scope & Context

*Part 1 of 4. Read alongside `README.md` (the problem statement). Next: `02_phase4_decisions.md`.*

---

## The overall challenge, briefly

Business records arrive from three independent, noisy sources (S1, S2, S3), sharing
no common identifiers. Source 1 is the deduplicated reference set; the task is to
find every matching S2/S3 record for each S1 entity, where a match count of zero,
one, or many is all valid. Output is scored per-S1-entity using a precision-weighted
F₀.₅ metric, macro-averaged across entities, with singletons (no true matches)
scored 1.0 for a correctly-empty prediction and 0.0 for any false merge. Full formula
and output format are in `README.md` — read that for the actual rules; this file only
gives the shape of the problem so the rest of this package makes sense.

## The pipeline as a sequence of phases

This project is being built in phases. The ones before Phase 4 are already
implemented — you will not see their code in this handoff package, and you should
treat them as a black box you consume through a defined contract, not something to
inspect or reason about internally:

- **Phase 0 — Preprocessing.** Raw TSVs are cleaned, nulls imputed, and multilingual
  text is normalized into dual representations (one for semantic search, one for
  lexical search).
- **Phases 1–2 — Blocking.** For each S1 entity, candidate S2/S3 records are
  retrieved via a combination of lexical and semantic search, run as a cyclic,
  budget-limited loop that expands the candidate pool across multiple passes.
- **Phase 3 — Feature Engineering.** Each retained `(S1 entity, candidate)` pair is
  turned into a fixed-length numeric feature vector capturing name similarity,
  address similarity, retrieval-signal strength, and a handful of metadata flags.

**Phase 4 — ML Classification — is what this handoff package covers.** It takes the
labeled feature vectors produced by Phase 3 (during training) and trains a
classifier that will later decide, for each candidate pair, whether it's a true
match. The deliverable is a trained, calibrated, persisted model — not a live
inference integration.

**After Phase 4** comes integration of the trained model back into the blocking loop
for live inference (a component tentatively called the orchestrator, plus a small
model-loading wrapper). **That integration is not part of this task.** Treat it as
not-yet-existing. Do not design Phase 4's code around assumptions of how it will be
called later beyond the one thing that matters: it must be loadable from a single
saved artifact and expose a standard `predict_proba`-style interface, which any
fitted scikit-learn `Pipeline` already does.

## What Phase 4 concretely receives and must produce

**Receives:** a table of labeled `(S1 entity, candidate)` pairs, one row per pair,
carrying a fixed set of numeric/categorical features and a binary match label. (The
exact assumed shape of this table is detailed in `03_implementation_blueprint.md` —
not needed yet at this stage.)

**Must produce:** a single persisted artifact containing a fitted model, a
calibrated decision threshold, the hyperparameters that were selected, and the
feature ordering the model expects — plus a complete, inspectable record (logs and a
machine-readable summary) of how the training run arrived at that artifact.

## Explicit boundaries for this task

**In scope:** everything needed to go from the labeled feature table to that
persisted artifact — hyperparameter search, cross-validated evaluation, threshold
calibration, final model fit, and the logging/testing infrastructure around all of
it.

**Out of scope:** anything upstream of the feature table (preprocessing, blocking,
feature extraction — already built, not to be modified or even read in detail) and
anything downstream of the saved artifact (loading it for live inference, wiring it
into the candidate-scoring loop — not yet built, not this task).

## What happens next

Proceed to `02_phase4_decisions.md`, alongside the original Phase 4 design document
and the canonical feature table, as described in `00_START_HERE.md`.
