# Phase 4 — Design Decisions

*Part 2 of 4. Read alongside the original "Phase 4: ML Classification (v2.0)" design
document and the canonical 26-dimension feature summary table. Previous:
`01_scope_and_context.md`. Next: `03_implementation_blueprint.md`.*

---

## Why this file exists

The original design document is the source of truth for what Phase 4 should do, and
it is detailed and mostly self-consistent. But a handful of places needed a decision
that either wasn't fully specified, or — in one case — was internally contradictory
in a way that would prevent the documented code from actually running. This file
records those decisions and the reasoning behind each, so the intent is preserved
even though the exact mechanism differs slightly from what the original document's
illustrative code snippets show.

**Nothing here changes what Phase 4 is trying to achieve.** Every decision below is
about *how* to correctly implement the original intent, not a change of scope or
goal.

---

## Decision 1 — Column order feeding the monotone constraint vector

**The issue:** The design document builds the preprocessing step by declaring the
country target-encoder transformer before the numeric passthrough transformer. A
`ColumnTransformer` concatenates its outputs in the order the transformers are
*declared*, not the order columns appear in the source table — so as written, the
matrix actually reaching the classifier would have the encoded country column
*first*, with the 25 numeric feature columns shifted after it. The monotone
constraint vector, however, is defined and ordered assuming the numeric columns come
first and country comes last (matching the documented `FEATURE_COLS` order). Left
as-is, every constraint would silently bind to the wrong feature — a real,
undetectable-from-metrics-alone correctness bug.

**Decision:** Declare the numeric passthrough transformer first and the country
target-encoder second, so the transformed matrix column order is numeric features
followed by encoded country — matching `FEATURE_COLS` and the monotone vector
exactly, with no other change needed anywhere else.

**Why it matters:** This is the single most important correctness fix in this phase.
Everything downstream (Optuna search, calibration, final fit) inherits whatever
alignment the preprocessing step establishes.

## Decision 2 — Manual encoding used for HPO pruning must mirror the real pipeline

**The issue:** During hyperparameter search, a validation set needs to be encoded
by hand (outside the full pipeline) purely so a pruning callback can watch its
loss. If that manual encoding uses a different column order or logic than the real
pipeline's preprocessing step, the loss it's pruning on doesn't actually reflect
what the real fold model would see.

**Decision:** The manual encoding used for pruning must reproduce the exact column
order and encoding logic established in Decision 1 — no separate or shortcut
encoding scheme.

## Decision 3 — The competition's precision/recall weighting must be configurable, not hardcoded

**The issue:** The core evaluation formula (the entity-level, precision-weighted
score used throughout calibration and hyperparameter search) has a fixed weighting
between precision and recall. The design document's `.env` schema exposes this
weighting as a tunable value, but the accompanying formula in the document hardcodes
it.

**Decision:** The weighting parameter is threaded through as a real function
argument, sourced from config, not hardcoded inside the formula. The current default
value matches the competition's own official metric, so behavior is unchanged out of
the box — this decision is about keeping the door open for experimentation, not
about changing today's default behavior.

## Decision 4 — The class-imbalance weighting is fixed, not tuned

**The issue:** It could be read either way — some pipelines do treat this kind of
weighting as tunable — but the design document is explicit that it's meant to be an
exact, formula-derived value rather than something the hyperparameter search
chooses, and the documented search space doesn't include it.

**Decision:** Compute the class-imbalance weighting once, from the complete labeled
dataset, before hyperparameter search begins. Reuse that exact same fixed value
everywhere a model is constructed for the rest of the run — every trial during
search, the later clean re-evaluation pass, and the final model fit. Do not
recompute it per-fold and do not let the search choose a different value.

## Decision 5 — Two small additions needed to make the pruning callback actually work

**The issue:** The design document references a pruning callback that watches
validation loss mid-fit to abandon clearly bad hyperparameter trials early. Two
things it depends on aren't explicitly set up anywhere: (a) the callback needs an
extra small library alongside the core search library, since modern versions ship it
separately; (b) the callback needs the classifier to actually be tracking the loss
metric it watches, which isn't otherwise configured.

**Decision:** Add the small companion package needed for the callback. Fix the loss
metric the classifier tracks as a constant, non-tuned setting on every model
construction — same treatment as the class-imbalance weighting in Decision 4.

## Decision 6 — Hyperparameter search evaluates at a fixed decision threshold

**The issue:** Should the search itself sweep decision thresholds to find each
trial's own best operating point, or evaluate every trial at one fixed threshold and
leave threshold selection entirely to the later, dedicated calibration step?

**Decision:** Fixed threshold (the metric's natural midpoint) during search,
threshold selection deferred entirely to the dedicated calibration phase described
in the blueprint. This keeps the two searches decoupled — search picks
hyperparameters, calibration picks the operating point — and avoids letting a noisy,
small per-trial evaluation set overfit a threshold choice that then contaminates
which hyperparameters look best.

## Decision 7 — Scope boundary: this phase stops at a saved, loadable model

**Decision:** This task delivers the training pipeline and the artifact it produces.
The small wrapper that will later load that artifact for live scoring inside the
candidate-generation loop is a separate, later task and is not part of this handoff.

## Decision 8 — A fallback model family is named in the design but not built yet

**The issue:** The design document names a second model family as a fallback option
without specifying when or how it gets implemented.

**Decision:** Structure model construction behind a small selection point — one
path fully implemented now, the fallback path present but deliberately
unimplemented (it should fail loudly and clearly if ever selected) — so that adding
it later doesn't require touching the surrounding search/calibration/persistence
logic at all.

## Decision 9 — Keep a non-feature identifier column alongside the feature table

**Decision:** The candidate's own identifier (distinct from the grouping key used
for cross-validation) should ride along through the entire pipeline as a
passthrough column — present in the data, excluded from what actually reaches the
model — so that any individual prediction can be traced back to the exact pair it
came from during later debugging or error analysis. This costs nothing during
training and pays for itself the first time something needs investigating.

## Decision 10 — Hyperparameter names must be defined consistently from the start

**The issue:** The design document's hyperparameter-search code names each
parameter one way when asking the search library to suggest a value, but then feeds
the search library's own recorded "best parameters" back into the model as if they
had been named a different, prefixed way (a scikit-learn `Pipeline` requires nested
parameters to be referenced with a prefix identifying which pipeline step they
belong to). As written, the step that reuses the best parameters after search
completes would fail outright, because the names don't actually match what the
search library recorded.

**Decision:** Use the fully-prefixed name from the very first point a parameter is
named — when asking the search library to suggest a value for it. This way, the
value the search library reports back as "best" is already in exactly the form
needed everywhere downstream, with no translation step required, and the design
document's own later logic (which strips the prefix for the truly final model
construction) continues to work exactly as documented.

## Decision 11 — Encoding must be reproducible across separate fit calls

**Decision:** Wherever the country encoding step is constructed — inside the real
pipeline, and separately inside the manual pruning-callback encoding described in
Decision 2 — it must be given the same fixed random seed used elsewhere in this run,
so two separately-fit copies of it behave as identically as the underlying library
allows.

## Decision 12 — The classifier itself needs a fixed seed too

**Decision:** The classifier's own randomness (used internally during training) is
pinned to the same fixed seed used everywhere else in the run. Without this, two
runs given the exact same selected hyperparameters are not guaranteed to produce
identical trained models, which undermines reproducibility of the later clean
re-evaluation pass and the final fit.

## Decision 13 — Logging approach

**Decision:** Use straightforward structured logging (console plus a run-scoped log
file) and a separate, complete, machine-readable run summary — not a dedicated
experiment-tracking platform. This is a single well-defined four-phase run per
invocation, not an ongoing multi-experiment research workflow, so the added
infrastructure of a full tracking platform isn't earning its cost right now. Full
detail in `04_metrics_logging_testing.md`.

## Decision 14 — Hyperparameter search state is not persisted to disk mid-run

**Decision:** The search runs in memory only; a crash partway through loses
completed trials rather than resuming from a checkpoint. Accepted trade-off given
the run length involved — simplicity now, revisit only if run failures in practice
turn out to be costly.

## Decision 15 — Testing priority

**Decision:** Integration and smoke-level tests carry the most weight and should be
built first and most thoroughly; a focused set of unit tests covers the pure-logic
pieces cheaply. Full detail, including an explicit regression test tied directly to
Decision 1, is in `04_metrics_logging_testing.md`.

---

## Reproducibility checklist (cross-reference)

Every one of these should trace back to one fixed, shared seed value from
configuration:

- [ ] The search library's own sampling process (Decision — search determinism)
- [ ] The country encoder, everywhere it's constructed (Decision 11)
- [ ] The classifier itself (Decision 12)
- [ ] The cross-validation splitter is *intentionally* left without a seed — it's
      deterministic given a consistent row order in the input data, and adding a
      seed where the design document doesn't call for one would be an unrequested
      change, not a fix

---

## CHECKPOINT before continuing

If anything above seems to contradict the original design document, or applying a
decision requires information not available in this package (for example, actual
column names that might differ from what's assumed), **stop and ask now.** Once you
proceed to `03_implementation_blueprint.md`, these decisions are treated as settled.
