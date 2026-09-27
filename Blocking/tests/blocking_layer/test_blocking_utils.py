import pytest
from src.blocking_layer.blocking_utils import to_char_3grams

class TestBlockingUtils:

    def test_to_char_3grams_basic(self):
        """REQ-BUTIL-02: Normal behavior for strings with length >= 3."""
        assert to_char_3grams("abc") == ["abc"]
        assert to_char_3grams("abcd") == ["abc", "bcd"]
        assert to_char_3grams("hello") == ["hel", "ell", "llo"]
        assert to_char_3grams("a b") == ["a b"]
        assert to_char_3grams("a b c") == ["a b", " b ", "b c"]

    def test_to_char_3grams_short_strings(self):
        """REQ-BUTIL-03 (3): Fallback returning [text] if len(text) < 3."""
        assert to_char_3grams("a") == ["a"]
        assert to_char_3grams("ab") == ["ab"]
        assert to_char_3grams(" ") == [" "]
        assert to_char_3grams("  ") == ["  "]

    def test_to_char_3grams_empty_string(self):
        """REQ-BUTIL-03 (1): Defensive fallback for empty strings returning []."""
        assert to_char_3grams("") == []

    def test_to_char_3grams_nullname_nulladdr_interception(self):
        """REQ-BUTIL-03 (2): Explicit interception of nullname/nulladdr returning [].
        Should be case-insensitive and ignore leading/trailing whitespace."""
        intercept_cases = [
            "nullname", "nulladdr", 
            "NULLNAME", "NULLADDR", 
            "NullName", "NullAddr", 
            " nullname ", "  nulladdr", 
            "nullname  ", "\t nulladdr \n",
            "\nnullname\r"
        ]
        for case in intercept_cases:
            assert to_char_3grams(case) == [], f"Failed to intercept null value: {repr(case)}"

    def test_to_char_3grams_nullname_substring(self):
        """Ensure that nullname/nulladdr as a substring is NOT intercepted."""
        assert to_char_3grams("anullnamea") == ['anu', 'nul', 'ull', 'lln', 'lna', 'nam', 'ame', 'mea']
        assert to_char_3grams("nullnames") == ['nul', 'ull', 'lln', 'lna', 'nam', 'ame', 'mes']
        assert to_char_3grams("prefix nulladdr") == ['pre', 'ref', 'efi', 'fix', 'ix ', 'x n', ' nu', 'nul', 'ull', 'lla', 'lad', 'add', 'ddr']

    def test_to_char_3grams_invalid_types(self):
        """Schema check / Error handling: Invalid inputs should return [] instead of crashing."""
        invalid_cases = [
            None,
            123,
            45.67,
            True,
            ["abc"],
            {"key": "value"}
        ]
        for case in invalid_cases:
            assert to_char_3grams(case) == [], f"Failed on invalid type: {type(case)} with value {repr(case)}"
