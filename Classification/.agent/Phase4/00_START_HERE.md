# Phase 4 — ML Classification: Reading Order & Handoff Index

This is a staged handoff package for implementing Phase 4 (ML Classification) of the
entity resolution pipeline. It is split into 4 files on purpose, so the implementing
agent only loads what it needs at each point, instead of one large document up front.

**Read and attach in stages, in order. Do not skip ahead. There is one explicit
checkpoint (end of Stage 2) where you should stop and ask questions before continuing.**

---

## Stage 1 — Understand the overall scope

**Attach:** `README.md` (the competition problem statement, provided separately) +
`01_scope_and_context.md`

**Goal:** Understand the end-to-end entity resolution challenge, and — more
importantly — understand exactly where "Phase 4" sits inside a larger, already
partially-built pipeline, and what its boundaries are. This is orientation only,
no phase-4-specific technical detail yet.

Nothing else is needed at this stage. Deliberately — the broader implementation plan
this project was originally scoped from is intentionally *not* included here; it
covers phases that are already built and contains some stale details that would only
add noise at the orientation stage.

---

## Stage 2 — Understand the Phase 4 decisions

**Attach:** `02_phase4_decisions.md` + the original **"Phase 4: ML Classification
(v2.0)" design document** + the **canonical 26-dimension feature summary table**

**Goal:** The original design document is the source of truth for *what* Phase 4
should do. `02_phase4_decisions.md` resolves a handful of places where that document
was ambiguous, internally inconsistent, or contained an actual bug, and explains why
each resolution was made. Read the design doc first, then the decisions file, so the
"why" makes sense against the original text.

> **CHECKPOINT.** Before moving to Stage 3: if anything in the decisions file seems
> to contradict the design document, or a decision seems to require information not
> given anywhere in these files, **stop and ask before proceeding.** Do not guess or
> silently resolve an inconsistency on your own at this stage — that's exactly what
> this checkpoint is for.

---

## Stage 3 — Read the implementation blueprint

**Attach:** `03_implementation_blueprint.md`

(Keep the design doc and decisions file available for reference — you don't need to
re-read them in full, but the blueprint will point back to specific decisions by
number rather than re-explaining them.)

**Goal:** Understand the shape of what you're building — module boundaries, data
flow through the four training phases, the assumed input contract. This file is a
blueprint, not a spec of literal code — it deliberately does not dictate function
signatures, internal helper structure, or naming. Those are yours to decide as you
implement.

---

## Stage 4 — Read metrics, logging & testing details

**Attach:** `04_metrics_logging_testing.md`

**Goal:** The precise behavioral spec for the two metric functions, what must be
logged and when, and the testing plan (unit, integration, smoke) with priority on
integration and smoke coverage.

---

## Stage 5 — Begin implementation

At this point you have everything conceptual you need. From here, pull in real
repository content **just-in-time, only as actually needed**, rather than attaching
it all up front:

- The current `config.py` — to extend, not overwrite
- The current `requirements.txt` — to append to, not duplicate
- The current `.env.example` — to extend
- A schema sample (column names + dtypes, or a handful of real rows) of the actual
  training buffer produced upstream — to verify the assumed input contract in
  `03_implementation_blueprint.md` before you build around it

**Never attach, at any stage, for this task:** the full v2.3 implementation plan,
`orchestrator.py`, any `blocking_layer/*` file, `feature_layer/feature_extractor.py`
in full, `classifier.py`, `validate_submission.py`, or `Documentation_template.md`.
None of these are in scope for Phase 4, and pulling them in risks contaminating the
task with details from phases that are either already finished or not yet begun.

---

## Final checklist before calling Phase 4 complete

- [ ] Every decision in `02_phase4_decisions.md` is reflected in the code, not just
      the document
- [ ] The regression test and behavioral test called out in
      `04_metrics_logging_testing.md` for the column-alignment fix (decision #1) both
      pass — these are the direct proof that the most important correctness fix in
      this phase actually landed
- [ ] All unit, integration, and smoke tests pass
- [ ] A run (even at smoke scale) produces a log file, a complete JSON summary, and a
      model artifact with all required keys
- [ ] No code written in this phase imports from or references `orchestrator.py`,
      `blocking_layer/*`, or `classifier.py`
