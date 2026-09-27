import pytest

# Since we are using TDD and the implementation isn't there yet,
# we will mock the function signature as expected by the spec.
# The developing agent will implement this function.

# Expected function signature:
# def extract_address_numeric_features(addr_a: str, addr_b: str) -> dict:
#     """Returns dict with keys: dim_11_num_jaccard, dim_12_num_exact_match, dim_13_num_presence"""
#     pass

try:
    from src.feature_layer.address_numeric.address_numeric import extract_address_numeric_features
except ImportError:
    # TDD Placeholder: tests will fail until this is implemented correctly
    def extract_address_numeric_features(addr_a: str, addr_b: str) -> dict:
        raise NotImplementedError("Developer needs to implement this.")

class TestAddressNumericFeatures:
    
    def test_sentinel_short_circuit(self):
        """
        If either address starts with 'nulladdr', the module MUST immediately return 
        0.0 for Dim 11, -1 for Dim 12, and 0 for Dim 13.
        """
        res1 = extract_address_numeric_features("nulladdr 0042", "123 main st")
        assert res1["dim_11_num_jaccard"] == 0.0
        assert res1["dim_12_num_exact_match"] == -1
        assert res1["dim_13_num_presence"] == 0

        res2 = extract_address_numeric_features("456 broadway", "nulladdr 9999")
        assert res2["dim_11_num_jaccard"] == 0.0
        assert res2["dim_12_num_exact_match"] == -1
        assert res2["dim_13_num_presence"] == 0
        
        res3 = extract_address_numeric_features("nulladdr 123", "nulladdr 456")
        assert res3["dim_11_num_jaccard"] == 0.0
        assert res3["dim_12_num_exact_match"] == -1
        assert res3["dim_13_num_presence"] == 0

    def test_token_extraction_with_regex(self):
        """
        Ensures that contiguous numeric strings are extracted correctly, 
        ignoring alphabetical characters (e.g. apt 4 b vs apt4b).
        """
        # Both addresses essentially have "123" and "4" as numeric tokens.
        addr_a = "123 main st, apt 4 b"
        addr_b = "123 main st, apt4b"
        res = extract_address_numeric_features(addr_a, addr_b)
        
        assert res["dim_11_num_jaccard"] == 1.0
        assert res["dim_12_num_exact_match"] == 1
        assert res["dim_13_num_presence"] == 1

    def test_dim_11_numeric_jaccard(self):
        """
        Dim 11: len(intersect) / max(len(union), 1)
        """
        # A: {1, 2, 3}, B: {2, 3, 4} -> Intersect: {2, 3} (len 2), Union: {1, 2, 3, 4} (len 4) -> 2/4 = 0.5
        res = extract_address_numeric_features("1 st 2 ave 3 blvd", "2 ave 3 blvd 4 rd")
        assert res["dim_11_num_jaccard"] == 0.5
        
        # A: {10}, B: {20} -> Intersect: {}, Union: {10, 20} -> 0.0
        res2 = extract_address_numeric_features("10 main", "20 main")
        assert res2["dim_11_num_jaccard"] == 0.0

    def test_dim_12_exact_match_ternary(self):
        """
        Dim 12 (Numeric Exact Match) [Ternary]:
        - If len(num_set_a) == 0 or len(num_set_b) == 0: Return -1
        - Else if num_set_a == num_set_b: Return 1
        - Else: Return 0
        """
        # Identical and non-empty (1)
        res_ident = extract_address_numeric_features("123 A st", "123 B st")
        assert res_ident["dim_12_num_exact_match"] == 1

        # Non-empty but different (0)
        res_diff = extract_address_numeric_features("123 A st", "456 B st")
        assert res_diff["dim_12_num_exact_match"] == 0

        # At least one empty (-1)
        res_empty1 = extract_address_numeric_features("main st", "123 main st")
        assert res_empty1["dim_12_num_exact_match"] == -1
        
        res_empty2 = extract_address_numeric_features("123 main st", "main st")
        assert res_empty2["dim_12_num_exact_match"] == -1
        
        # Both empty (-1)
        res_both_empty = extract_address_numeric_features("main st", "broadway")
        assert res_both_empty["dim_12_num_exact_match"] == -1

    def test_dim_13_presence(self):
        """
        Dim 13 (Numeric Presence): Return 1 if len(num_set_a) > 0 AND len(num_set_b) > 0, else 0.
        """
        # Both have numbers
        res_both = extract_address_numeric_features("123 st", "456 ave")
        assert res_both["dim_13_num_presence"] == 1
        
        # One has numbers
        res_one = extract_address_numeric_features("123 st", "main st")
        assert res_one["dim_13_num_presence"] == 0
        
        # None have numbers
        res_none = extract_address_numeric_features("first st", "second ave") # no digits
        assert res_none["dim_13_num_presence"] == 0

    def test_zero_division_guard(self):
        """
        ZeroDivisionError (Math Clamping): Purely alphabetical addresses have empty numeric sets.
        max(len(union), 1) mathematically eliminates crash risks.
        """
        res = extract_address_numeric_features("near sbi atm, mg road", "mg road")
        assert res["dim_11_num_jaccard"] == 0.0 # Without max(..., 1), this would crash with ZeroDivisionError
