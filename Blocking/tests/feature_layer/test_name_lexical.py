import pytest
import math
from src.feature_layer.name_lexical import compute_name_lexical_features

def test_exact_match():
    """Dim 4 (Exact Match) should be 1, and all float similarities should be 1.0"""
    name_a = "amazon retail inc"
    name_b = "amazon retail inc"
    
    result = compute_name_lexical_features(name_a, name_b)
    
    assert result["dim_4_name_exact_match"] == 1
    assert math.isclose(result["dim_1_name_jaro_winkler"], 1.0)
    assert math.isclose(result["dim_2_name_monge_elkan"], 1.0)
    assert math.isclose(result["dim_3_name_levenshtein_norm"], 1.0)

def test_sentinel_short_circuit():
    """If either string starts with 'nullname', all dimensions MUST be exactly 0.0 or 0."""
    cases = [
        ("nullname 123", "amazon retail inc"),
        ("amazon retail inc", "nullname 456"),
        ("nullname 123", "nullname 456")
    ]
    
    for name_a, name_b in cases:
        result = compute_name_lexical_features(name_a, name_b)
        assert result["dim_1_name_jaro_winkler"] == 0.0
        assert result["dim_2_name_monge_elkan"] == 0.0
        assert result["dim_3_name_levenshtein_norm"] == 0.0
        assert result["dim_4_name_exact_match"] == 0
        assert result["dim_6_name_acronym_match"] == 0

def test_empty_token_guard():
    """Empty strings or strings with only spaces should gracefully return 0.0."""
    cases = [
        ("", "amazon retail"),
        ("   ", "amazon retail"),
        ("", "")
    ]
    
    for name_a, name_b in cases:
        result = compute_name_lexical_features(name_a, name_b)
        assert result["dim_2_name_monge_elkan"] == 0.0
        # For levenshtein zero-guard
        assert result["dim_3_name_levenshtein_norm"] == 0.0

def test_levenshtein_similarity():
    """Ensure Dim 3 is 1.0 - (dist / max_len) and not raw distance."""
    # "kitten" vs "sitting"
    # levenshtein distance is 3
    # max length is 7
    # similarity is 1.0 - (3/7) = 4/7
    result = compute_name_lexical_features("kitten", "sitting")
    expected = 1.0 - (3.0 / 7.0)
    assert math.isclose(result["dim_3_name_levenshtein_norm"], expected)

def test_monge_elkan_symmetry():
    """Verify that Monge-Elkan is computed as the symmetric average: (ME(A,B) + ME(B,A)) / 2.0"""
    name_a = "batra retail inc"
    name_b = "retail bata"
    
    result_ab = compute_name_lexical_features(name_a, name_b)
    result_ba = compute_name_lexical_features(name_b, name_a)
    
    # The symmetric implementation must return the exact same score regardless of argument order
    assert math.isclose(result_ab["dim_2_name_monge_elkan"], result_ba["dim_2_name_monge_elkan"])
    
    # We also sanity check that it's > 0 and < 1
    assert 0.0 < result_ab["dim_2_name_monge_elkan"] < 1.0

def test_acronym_match_logic():
    """Verify Dim 6 logic for acronyms."""
    # Standard acronym vs full name
    # ibm -> initials of longer string ("international business machines") -> ibm
    res1 = compute_name_lexical_features("ibm", "international business machines")
    assert res1["dim_6_name_acronym_match"] == 1
    
    res2 = compute_name_lexical_features("international business machines", "ibm")
    assert res2["dim_6_name_acronym_match"] == 1
    
    # No meaningful acronym from a single word
    res3 = compute_name_lexical_features("amazon", "inc")
    assert res3["dim_6_name_acronym_match"] == 0
    


def test_return_schema():
    """Ensure the function returns a flat dict with the exact expected keys."""
    result = compute_name_lexical_features("amazon", "amazon")
    
    expected_keys = {
        "dim_1_name_jaro_winkler",
        "dim_2_name_monge_elkan",
        "dim_3_name_levenshtein_norm",
        "dim_4_name_exact_match",
        "dim_6_name_acronym_match"
    }
    
    assert set(result.keys()) == expected_keys
