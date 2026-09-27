"""
Test suite for the bm25_utils module in the blocking layer.
Validates all requirements from REQ-BMU-01 to REQ-BMU-06, including strict type adherence.
"""

import pytest
from src.blocking_layer.bm25_utils.bm25_utils import calculate_self_score


def test_calculate_self_score_standard_calculation():
    """
    Test standard calculation of BM25 self score (REQ-BMU-01, REQ-BMU-02).
    Ensures that frequency explicitly affects the final score according to the exact math.
    """
    query_tokens = ["a", "b", "a"]
    idf_dict = {"a": 1.5, "b": 2.0}
    avgdl = 2.5
    k1 = 1.5
    b = 0.75
    
    # Expected manual calculation:
    # 'a' frequency = 2
    # term_score_a = 1.5 * (2 * 2.5) / (2 + 1.5 * (1 - 0.75 + 0.75 * (3 / 2.5)))
    # term_score_a = 7.5 / (2 + 1.5 * 1.15) = 7.5 / 3.725 = 2.0134228187919465
    # total contribution for 'a' over the loop (2 occurrences) = 4.026845637583893
    #
    # 'b' frequency = 1
    # term_score_b = 2.0 * (1 * 2.5) / (1 + 1.5 * 1.15) = 5.0 / 2.725 = 1.834862385321101
    # total contribution for 'b' over the loop (1 occurrence) = 1.834862385321101
    #
    # final_score = 4.026845637583893 + 1.834862385321101 = 5.861708022904994
    expected_score = 5.861708022904994
    
    score = calculate_self_score(query_tokens, idf_dict, avgdl, k1, b)
    assert pytest.approx(score, rel=1e-9) == expected_score


def test_calculate_self_score_all_unique():
    """
    Test calculation when all query tokens are unique (REQ-BMU-02).
    """
    query_tokens = ["a", "b", "c"]
    idf_dict = {"a": 1.0, "b": 2.0, "c": 3.0}
    avgdl = 3.0
    k1 = 1.2
    b = 0.75
    
    # len=3, avgdl=3 -> len/avgdl = 1.0
    # denominator = 1 + 1.2 * (1 - 0.75 + 0.75 * 1.0) = 2.2
    # term_a = 1.0 * (1 * 2.2) / 2.2 = 1.0
    # term_b = 2.0 * (1 * 2.2) / 2.2 = 2.0
    # term_c = 3.0 * (1 * 2.2) / 2.2 = 3.0
    # Total = 6.0
    score = calculate_self_score(query_tokens, idf_dict, avgdl, k1, b)
    assert pytest.approx(score, rel=1e-9) == 6.0


def test_empty_query_tokens_floor():
    """
    Test that an empty query_tokens list returns 1e-9 (REQ-BMU-03).
    """
    score = calculate_self_score([], {"a": 1.5}, 2.5, 1.5, 0.75)
    assert score == 1e-9


def test_negative_or_zero_score_clamping():
    """
    Test that a 0.0 or negative calculated score is clamped to 1e-9 (REQ-BMU-04).
    """
    # Negative IDF yields negative score
    score_negative = calculate_self_score(["a"], {"a": -1.5}, 2.5, 1.5, 0.75)
    assert score_negative == 1e-9

    # Zero IDF yields zero score
    score_zero = calculate_self_score(["a"], {"a": 0.0}, 2.5, 1.5, 0.75)
    assert score_zero == 1e-9


def test_oov_tokens_default_idf():
    """
    Test that OOV tokens gracefully default to IDF 0.0 and don't raise KeyError (REQ-BMU-05).
    """
    query_tokens = ["a", "oov_token"]
    idf_dict = {"a": 1.0}
    avgdl = 2.0
    k1 = 1.2
    b = 0.75
    
    # denom = 1 + 1.2 * (1 - 0.75 + 0.75 * (2/2)) = 2.2
    # term_a = 1.0 * (1 * 2.2) / 2.2 = 1.0
    # term_oov = 0.0
    # Total expected = 1.0
    score = calculate_self_score(query_tokens, idf_dict, avgdl, k1, b)
    assert pytest.approx(score, rel=1e-9) == 1.0

    # All OOV
    score_all_oov = calculate_self_score(["x", "y"], {"a": 1.0}, 2.0, 1.2, 0.75)
    assert score_all_oov == 1e-9  # Clamped from 0.0


def test_invalid_avgdl():
    """
    Test that avgdl <= 0.0 gracefully returns 1e-9 (REQ-BMU-06).
    """
    score_zero = calculate_self_score(["a"], {"a": 1.0}, 0.0, 1.2, 0.75)
    assert score_zero == 1e-9

    score_negative = calculate_self_score(["a"], {"a": 1.0}, -2.5, 1.2, 0.75)
    assert score_negative == 1e-9


def test_output_schema_type():
    """
    Ensure the output is strictly a float, even in clamped scenarios.
    """
    score = calculate_self_score(["a"], {"a": 1.0}, 2.5, 1.5, 0.75)
    assert type(score) is float
    
    score_clamped = calculate_self_score([], {"a": 1.0}, 2.5, 1.5, 0.75)
    assert type(score_clamped) is float


