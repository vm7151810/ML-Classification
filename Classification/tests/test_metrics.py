"""Unit tests for metrics module."""

import pytest
import numpy as np

from src.metrics import compute_scale_pos_weight, entity_macro_fbeta


def test_scale_pos_weight_known_ratio():
    """Test compute_scale_pos_weight on known negative/positive ratio."""
    y = [0, 0, 0, 0, 1]  # 4 negatives, 1 positive -> 4.0
    assert compute_scale_pos_weight(y) == 4.0

    y2 = [0] * 75 + [1] * 25  # 75 negatives, 25 positives -> 3.0
    assert compute_scale_pos_weight(y2) == 3.0


def test_scale_pos_weight_zero_positives_raises():
    """Test that compute_scale_pos_weight raises ValueError if zero positives."""
    y = [0, 0, 0]
    with pytest.raises(ValueError, match="zero positive labels"):
        compute_scale_pos_weight(y)


def test_entity_macro_fbeta_singleton_correct():
    """Singleton with no true matches and no predicted matches scores 1.0."""
    y_true = [0, 0, 0]
    y_prob = [0.1, 0.2, 0.4]  # all below threshold 0.5
    s1_ids = ["S1-001", "S1-001", "S1-001"]

    score = entity_macro_fbeta(y_true, y_prob, s1_ids, threshold=0.5, beta=0.5)
    assert score == 1.0


def test_entity_macro_fbeta_singleton_false_merge():
    """Singleton with no true matches but at least one predicted match scores 0.0."""
    y_true = [0, 0, 0]
    y_prob = [0.1, 0.6, 0.2]  # item 1 > threshold 0.5 -> false merge
    s1_ids = ["S1-001", "S1-001", "S1-001"]

    score = entity_macro_fbeta(y_true, y_prob, s1_ids, threshold=0.5, beta=0.5)
    assert score == 0.0


def test_entity_macro_fbeta_hand_computed_mixed_group():
    """Test exact arithmetic of precision, recall, and F0.5 on a mixed entity."""
    # S1-001: 2 true matches out of 3 candidates
    y_true = [1, 1, 0]
    y_prob = [0.9, 0.4, 0.8]
    s1_ids = ["S1-001", "S1-001", "S1-001"]

    # At threshold 0.5:
    # Pred = [1, 0, 1]
    # TP = 1 (idx 0), FP = 1 (idx 2), FN = 1 (idx 1)
    # Precision = 1/2 = 0.5, Recall = 1/2 = 0.5
    # F_0.5 = (1.25 * 0.5 * 0.5) / (0.25 * 0.5 + 0.5) = 0.3125 / 0.625 = 0.5
    score = entity_macro_fbeta(y_true, y_prob, s1_ids, threshold=0.5, beta=0.5)
    assert pytest.approx(score, rel=1e-5) == 0.5

    # Test problem statement worked example:
    # S1-00001: 2 true matches out of 3 candidates
    # Predicted matches: 3 candidates (idx 0, 1, 2)
    # True matches: idx 0, 2. (idx 1 is non-match)
    y_true_ex = [1, 0, 1]
    y_prob_ex = [0.8, 0.7, 0.9]  # all predicted matched
    s1_ids_ex = ["S1-00001", "S1-00001", "S1-00001"]
    # Precision = 2/3, Recall = 2/2 = 1.0
    # F_0.5 = (1.25 * 0.6667 * 1.0) / (0.25 * 0.6667 + 1.0) = 0.8333 / 1.1667 = 0.7142857
    score_ex = entity_macro_fbeta(y_true_ex, y_prob_ex, s1_ids_ex, threshold=0.5, beta=0.5)
    assert pytest.approx(score_ex, rel=1e-4) == 0.7143


def test_entity_macro_fbeta_strict_inequality():
    """Verify strict inequality: prob == threshold is treated as 0 (not match)."""
    y_true = [1]
    y_prob = [0.50]  # Exactly equal to threshold
    s1_ids = ["S1-001"]

    score = entity_macro_fbeta(y_true, y_prob, s1_ids, threshold=0.50, beta=0.5)
    # Pred = [0], so TP=0, FP=0, FN=1 -> Recall=0 -> Score = 0.0
    assert score == 0.0

    y_prob_above = [0.50001]
    score_above = entity_macro_fbeta(y_true, y_prob_above, s1_ids, threshold=0.50, beta=0.5)
    # Pred = [1], TP=1, FP=0, FN=0 -> Precision=1, Recall=1 -> Score = 1.0
    assert score_above == 1.0


def test_entity_macro_fbeta_macro_average():
    """Verify that multiple entities are macro-averaged unweighted."""
    # Entity 1: perfect singleton -> 1.0
    # Entity 2: false merge singleton -> 0.0
    y_true = [0, 0]
    y_prob = [0.2, 0.8]
    s1_ids = ["S1-001", "S1-002"]

    score = entity_macro_fbeta(y_true, y_prob, s1_ids, threshold=0.5, beta=0.5)
    assert score == 0.5


def test_entity_macro_fbeta_alternate_beta():
    """Verify that changing beta changes the score when precision != recall."""
    # S1-001: Precision = 2/3, Recall = 1.0
    y_true = [1, 0, 1]
    y_prob = [0.9, 0.8, 0.9]
    s1_ids = ["S1-001", "S1-001", "S1-001"]

    score_beta_05 = entity_macro_fbeta(y_true, y_prob, s1_ids, threshold=0.5, beta=0.5)
    score_beta_10 = entity_macro_fbeta(y_true, y_prob, s1_ids, threshold=0.5, beta=1.0)
    score_beta_20 = entity_macro_fbeta(y_true, y_prob, s1_ids, threshold=0.5, beta=2.0)

    # F1 score for P=2/3, R=1: 2 * (2/3) * 1 / (2/3 + 1) = (4/3) / (5/3) = 0.8
    assert pytest.approx(score_beta_10, rel=1e-4) == 0.8000
    assert pytest.approx(score_beta_05, rel=1e-4) == 0.7143
    # Different betas produce different scores
    assert score_beta_05 != score_beta_10
    assert score_beta_10 != score_beta_20
