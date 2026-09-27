"""Metrics implementation for Phase 4 ML Classification.

Contains:
- compute_scale_pos_weight: Class-imbalance ratio on full dataset.
- entity_macro_fbeta: Macro-averaged entity-level F-beta score.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Sequence
import numpy as np


def compute_scale_pos_weight(y: Sequence[int] | np.ndarray) -> float:
    """Compute negative_count / positive_count for class imbalance handling.

    Args:
        y: Binary ground-truth labels (0 or 1).

    Returns:
        float: The scale_pos_weight ratio.

    Raises:
        ValueError: If there are zero positive labels in the dataset.
    """
    arr = np.asarray(y)
    n_pos = int(np.sum(arr == 1))
    n_neg = int(np.sum(arr == 0))

    if n_pos == 0:
        raise ValueError("Cannot compute scale_pos_weight: zero positive labels in dataset.")

    return float(n_neg / n_pos)


def entity_macro_fbeta(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    s1_ids: Sequence[Any],
    threshold: float,
    beta: float = 0.5,
) -> float:
    """Compute the macro-averaged entity-level F_beta score.

    Entities (S1) are grouped by their identifier. For each entity:
    - Probability > threshold is treated as a predicted match (strict inequality).
    - If an entity has no true matches and predicts no matches (singleton), score = 1.0.
    - If an entity has no true matches but predicts matches, score = 0.0.
    - Otherwise, compute precision, recall, and weighted F_beta.

    The final score is the unweighted arithmetic mean across all unique entities.

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_prob: Predicted match probabilities in [0.0, 1.0].
        s1_ids: S1 entity grouping identifiers.
        threshold: Decision threshold for classification (strict >).
        beta: Beta parameter for F_beta weighting (default 0.5).

    Returns:
        float: Macro-averaged F_beta across all entities.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_prob)

    if len(s1_ids) == 0:
        return 0.0

    groups: dict[Any, dict[str, list[int]]] = defaultdict(lambda: {"y_true": [], "y_pred": []})
    for s1_id, label, prob in zip(s1_ids, y_t, y_p):
        groups[s1_id]["y_true"].append(int(label))
        groups[s1_id]["y_pred"].append(int(prob > threshold))

    entity_scores: list[float] = []
    beta_sq = beta ** 2
    weight_mult = 1.0 + beta_sq

    for entry in groups.values():
        y_true_grp = entry["y_true"]
        y_pred_grp = entry["y_pred"]

        sum_true = sum(y_true_grp)
        sum_pred = sum(y_pred_grp)

        # Singleton special cases
        if sum_true == 0:
            if sum_pred == 0:
                entity_scores.append(1.0)
            else:
                entity_scores.append(0.0)
            continue

        tp = sum(t == 1 and p == 1 for t, p in zip(y_true_grp, y_pred_grp))
        fp = sum(t == 0 and p == 1 for t, p in zip(y_true_grp, y_pred_grp))
        fn = sum(t == 1 and p == 0 for t, p in zip(y_true_grp, y_pred_grp))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        denom = beta_sq * precision + recall
        if denom > 0.0:
            score = (weight_mult * precision * recall) / denom
        else:
            score = 0.0
        entity_scores.append(score)

    return float(np.mean(entity_scores)) if entity_scores else 0.0
