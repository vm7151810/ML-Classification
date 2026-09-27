# Phase 4 — Metrics, Logging & Testing

*Part 4 of 4. Previous: `03_implementation_blueprint.md`.*

---

## Metrics specification

### Class-imbalance weighting

A single number: the count of negative-labeled rows divided by the count of
positive-labeled rows, across whatever dataset it's asked to summarize (per Decision
4, this should only ever be called on the *full* labeled dataset, once). If there
are zero positive rows, this is a real data problem and should fail clearly rather
than default to some fallback value silently.

### Entity-level score

This is the evaluation metric used throughout the phase — during hyperparameter
search (at a fixed threshold, per Decision 6), during threshold calibration (swept
across many thresholds), and anywhere else a single number is needed to judge a set
of predictions. It takes: the true labels, the predicted probabilities, the grouping
key identifying which S1 entity each row belongs to, a decision threshold, and the
precision/recall weighting parameter (per Decision 3, not hardcoded).

Behavior:

1. Group all rows by their grouping key.
2. Within each group, apply the threshold to the probabilities to get binary
   predictions, then compute that group's own precision, recall, and weighted
   F-score exactly as the competition defines it (weighting precision more heavily
   than recall — see the original problem statement for the exact formula and
   worked example).
3. Two special cases apply within a group: if the group has no true positive labels
   at all and the predictions also correctly predict nothing, that group scores a
   perfect 1.0 (correctly identifying a "no match" case is worth full credit). If
   the group has no true positive labels but predictions still include something,
   that group scores 0.0 (a false merge on a true singleton is fully penalized).
4. The final result is the unweighted mean of every group's score — a true
   macro-average across entities, not a pooled average across all rows.

Get the boundary condition right: the comparison between a predicted probability and
the threshold should be strict (above the threshold, not at-or-above), matching how
it's described in the original design document.

---

## Logging specification

### What gets set up

A run should produce two artifacts beyond the model itself: a plain-text log
(console plus a run-scoped file) and one complete, machine-readable summary written
once at the end of the run. The log is for humans watching a run happen or debugging
after the fact; the summary is for anything that needs to consume "what happened in
this run" programmatically or wants a single place to check the full story without
reading a log file.

### What to log, by phase

- **At the start of a run:** every relevant configuration value in effect, the shape
  of the input dataset, the class balance, and the computed class-imbalance
  weighting.
- **During hyperparameter search:** for every trial — its sampled hyperparameters,
  its per-fold scores and their mean, whether it was abandoned early or ran to
  completion, and how long it took. When the search finishes: which trial won, its
  hyperparameters and score, and overall totals (trials completed, trials abandoned
  early, total time).
- **During the clean evaluation pass:** per fold, the split sizes, how long it took,
  and its score at the fixed threshold. At the end: the overall out-of-fold score at
  the fixed threshold, and a rough consistency comparison against the winning
  trial's own fold scores from the search phase — logged as a signal to eyeball, not
  something that should halt the run if it doesn't match exactly.
- **During threshold calibration:** the complete sweep — every threshold tried and
  its score — not just the winner, plus the winning threshold's own precision and
  recall at that operating point.
- **During the final fit:** the imbalance-weighting consistency check described in
  the blueprint's Phase D, how long the fit took, and confirmation of what was
  written to the persisted artifact and where.

### The run summary

Should contain everything needed to answer "what happened in this run and why did it
end up like this" without needing to re-read the log file line by line. At minimum,
it should capture: the configuration snapshot, the dataset summary, the full
hyperparameter search outcome (winner, its score, trial counts, timing), the clean
evaluation outcome, the complete threshold-vs-score curve and the chosen operating
point, and the final fit's timing and artifact location. Whatever structured,
easily-parseable format you choose for this file is fine — the important thing is
completeness, not a specific schema.

---

## Testing specification

Priority order, per Decision 15: **integration and smoke tests first and most
thoroughly; unit tests for the cheap, high-value pure-logic pieces.**

### Unit-level coverage

- The entity-level score function: both special-case branches (correctly-empty
  singleton scoring 1.0; falsely-populated singleton scoring 0.0), a hand-computed
  mixed group to confirm precision/recall/weighting arithmetic is right, the
  strict-inequality threshold boundary, correct behavior across multiple groups
  averaged together, and at least one alternate weighting value to confirm the
  weighting parameter is actually load-bearing and not silently ignored.
- The class-imbalance weighting function: a known ratio computed correctly, and a
  clear failure on zero positive rows.
- Configuration loading: the new settings this phase adds resolve to their correct
  default values when unset, and correctly pick up overrides when set.
- **A direct regression test for Decision 1.** Build the actual preprocessing +
  model pipeline on a small synthetic dataset, fit it, and explicitly assert that
  the fitted preprocessor's output places the numeric features before the encoded
  country column — not just that the pipeline runs without error, but that the
  column order is actually correct. This is the single most important test in this
  phase, since Decision 1 fixes a bug that would otherwise produce a pipeline that
  runs, trains, and reports plausible-looking metrics while being silently wrong.

### Integration-level coverage

Using a small synthetic dataset with realistic shape (multiple grouping keys, a mix
of countries including at least one that wouldn't have appeared during training —
to exercise the fallback behavior of the country encoding on genuinely unseen
values):

- Fit the real pipeline end-to-end and generate predictions — not mocked at any
  level.
- Confirm no leakage: the country encoding step, when fit inside a cross-validation
  split, must never have visibility into that split's held-out labels.
- Confirm persistence round-trips correctly: save the fitted pipeline, reload it,
  and confirm predictions on a held-out batch are identical before and after the
  round trip.
- **Confirm the monotone constraints actually bind to the intended features** — not
  just that column order is correct in isolation (the unit test above), but that the
  trained model's actual predicted behavior respects a constraint end to end.
  Construct a pair of synthetic rows differing only in one feature that's supposed
  to be constrained to increase match probability, with everything else held equal,
  and confirm the model's predicted probability doesn't decrease when that feature
  increases. This is the strongest possible proof that Decision 1 was implemented
  correctly, since it tests actual model behavior rather than just data shape.

### Smoke-level coverage

Run the entire training entrypoint end-to-end, with the trial count and
cross-validation fold count both forced down to their smallest reasonable values so
the whole thing completes in seconds rather than the real configured runtime.
Confirm: all four phases complete without error, a log file is produced and
non-empty, the run summary is produced and contains every section described above,
and the persisted artifact — when reloaded — contains exactly the required pieces
(fitted model, threshold, hyperparameters, feature ordering) with sensible types and
the correct feature count.

### Shared test data

However you organize the test files, build the synthetic dataset generation logic
once and reuse it everywhere it's needed — the unit-level pipeline-alignment test,
the integration tests, and the smoke test should all be able to draw on the same
generator with different sizes/parameters, rather than three separate ad hoc data
setups drifting apart from each other over time.
