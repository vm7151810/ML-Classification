import pytest
from unittest.mock import patch, MagicMock

# The module to be implemented by the developing agent
# Assuming it will be located at src/feature_layer/name_phonetic/name_phonetic.py
# with a function named `compute_name_phonetic_match`
try:
    from src.feature_layer.name_phonetic.name_phonetic import compute_name_phonetic_match
except ImportError:
    # Dummy placeholder for TDD - tests will fail until implemented
    def compute_name_phonetic_match(name_a: str, name_b: str) -> int:
        raise NotImplementedError("compute_name_phonetic_match is not implemented yet.")


class TestNamePhoneticMatch:
    """
    Test suite for Dim 5: Name Phonetic Match
    """

    def test_exact_match(self):
        """Test with identical names."""
        assert compute_name_phonetic_match("amazon inc", "amazon inc") == 1

    def test_phonetic_intersection_match(self):
        """Test where names are different but share at least one phonetic token."""
        # 'batra' and 'bata' might share a phonetic code depending on DoubleMetaphone
        # We test a known phonetic equivalence like 'smith' and 'smyth'
        assert compute_name_phonetic_match("smith retail", "smyth corp") == 1

    def test_no_phonetic_intersection(self):
        """Test where names share absolutely no phonetic tokens."""
        assert compute_name_phonetic_match("apple", "microsoft") == 0

    def test_sentinel_short_circuit_a(self):
        """Test Sentinel Short-Circuit on name_a."""
        assert compute_name_phonetic_match("nullname_123", "amazon inc") == 0

    def test_sentinel_short_circuit_b(self):
        """Test Sentinel Short-Circuit on name_b."""
        assert compute_name_phonetic_match("amazon inc", "nullname_456") == 0

    def test_sentinel_short_circuit_both(self):
        """Test Sentinel Short-Circuit on both, ensuring no phonetic collision on 'nullname'."""
        assert compute_name_phonetic_match("nullname_1", "nullname_2") == 0

    def test_empty_hash_purging_numbers(self):
        """
        Test Empty Hash Collisions:
        Double Metaphone yields "" for pure numbers. Discarding "" prevents 
        false positive matches on completely different numbers.
        """
        assert compute_name_phonetic_match("123", "999") == 0

    def test_empty_hash_purging_mixed_with_numbers(self):
        """Test strings with numbers and words to ensure number parts don't falsely match."""
        # If '123' and '999' give empty hashes, they shouldn't trigger a match
        # 'apple' and 'banana' have no intersection.
        assert compute_name_phonetic_match("apple 123", "banana 999") == 0

    @patch("pyphonetics.DoubleMetaphone")
    def test_primary_code_usage_and_tuple_indexing(self, mock_double_metaphone):
        """
        Mock pyphonetics to ensure ONLY the primary code (index 0) is used,
        and secondary codes are ignored per architectural constraints.
        """
        mock_instance = MagicMock()
        mock_double_metaphone.return_value = mock_instance
        
        # Word 'testA' -> primary 'A1', secondary 'A2'
        # Word 'testB' -> primary 'B1', secondary 'A2' (secondary matches)
        # Because we only use primary code, these should NOT match (return 0).
        def mock_phonetics(word):
            if word == "testA":
                return ("A1", "A2")
            if word == "testB":
                return ("B1", "A2")
            return ("", "")
            
        mock_instance.phonetics.side_effect = mock_phonetics
        
        assert compute_name_phonetic_match("testA", "testB") == 0
        
        # Verify the mock was actually called to ensure the logic runs
        assert mock_instance.phonetics.call_count == 2
        mock_instance.phonetics.assert_any_call("testA")
        mock_instance.phonetics.assert_any_call("testB")

    @patch("pyphonetics.DoubleMetaphone")
    def test_primary_code_match(self, mock_double_metaphone):
        """
        Mock pyphonetics to ensure a match on primary code returns 1.
        """
        mock_instance = MagicMock()
        mock_double_metaphone.return_value = mock_instance
        
        def mock_phonetics(word):
            if word == "match1":
                return ("M1", "M2")
            if word == "match2":
                return ("M1", "X2")
            return ("", "")
            
        mock_instance.phonetics.side_effect = mock_phonetics
        
        assert compute_name_phonetic_match("match1", "match2") == 1

    def test_casing_and_punctuation_handling(self):
        """
        Test that casing doesn't affect the phonetic match (assuming inputs might have mixed casing 
        although preprocessing usually handles it, the feature extractor should be safe).
        """
        assert compute_name_phonetic_match("Smith", "SMITH") == 1
