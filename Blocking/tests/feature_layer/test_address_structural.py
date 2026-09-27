import pytest
from src.feature_layer.address_structural.feature import calculate_address_structural

class TestAddressStructuralFeature:
    @pytest.mark.parametrize("addr_a, addr_b, expected", [
        # Sentinel Short-Circuit tests
        ("nulladdr_123", "123 main st", (0.0, 0.0, 0)),
        ("123 main st", "nulladdr_456", (0.0, 0.0, 0)),
        ("nulladdr", "nulladdr", (0.0, 0.0, 0)),
        ("nulladdr 1", "nulladdr 1", (0.0, 0.0, 0)),

        # Exact Match (Dim 10) and Jaccard/Containment (Dim 8, 9)
        ("123 main st", "123 main st", (1.0, 1.0, 1)),
        ("apple st", "apple st", (1.0, 1.0, 1)),

        # Disjoint Sets
        ("123 main st", "456 broadway ave", (0.0, 0.0, 0)),

        # Partial Overlap (Dim 8 and Dim 9)
        # addr_a: {"123", "main", "st"} -> len 3
        # addr_b: {"123", "main", "st", "apt", "4"} -> len 5
        # intersect: {"123", "main", "st"} -> len 3
        # union: {"123", "main", "st", "apt", "4"} -> len 5
        # Jaccard (Dim 8): 3 / 5 = 0.6
        # Containment (Dim 9): 3 / min(3, 5) = 1.0
        # Exact (Dim 10): 0
        ("123 main st", "123 main st apt 4", (0.6, 1.0, 0)),

        # Another Partial Overlap
        # addr_a: {"hello", "world"} -> len 2
        # addr_b: {"world", "peace"} -> len 2
        # intersect: {"world"} -> len 1
        # union: {"hello", "world", "peace"} -> len 3
        # Jaccard: 1 / 3
        # Containment: 1 / min(2, 2) = 0.5
        ("hello world", "world peace", (1/3, 0.5, 0)),

        # Whitespace token injection prevention
        # addr.split() drops contiguous spaces.
        # addr_a: {"a", "b"} -> len 2
        # addr_b: {"a", "b"} -> len 2
        ("a b", "a    b", (1.0, 1.0, 0)), # Not an exact string match, but tokens perfectly match
        ("  leading", "leading  ", (1.0, 1.0, 0)), # Different strings, same tokens
        
        # ZeroDivisionError guards
        # Empty string cases
        ("", "", (0.0, 0.0, 1)), # Exact match 1, intersection 0, union 0 -> max(0, 1) -> 0.0 for dim 8 & 9
        ("  ", "   ", (0.0, 0.0, 0)), # Empty token sets, exact match 0
        ("", "123 main st", (0.0, 0.0, 0)),
    ])
    def test_address_structural_metrics(self, addr_a, addr_b, expected):
        dim8, dim9, dim10 = calculate_address_structural(addr_a, addr_b)
        
        expected_dim8, expected_dim9, expected_dim10 = expected
        
        # Use pytest.approx for floating point comparisons
        assert dim8 == pytest.approx(expected_dim8, abs=1e-9), f"Dim 8 (Jaccard) failed for '{addr_a}' and '{addr_b}'"
        assert dim9 == pytest.approx(expected_dim9, abs=1e-9), f"Dim 9 (Containment) failed for '{addr_a}' and '{addr_b}'"
        assert dim10 == expected_dim10, f"Dim 10 (Exact Match) failed for '{addr_a}' and '{addr_b}'"

    def test_return_types(self):
        """Ensure the function returns the correct types according to the schema."""
        dim8, dim9, dim10 = calculate_address_structural("123 st", "123 st")
        assert isinstance(dim8, float)
        assert isinstance(dim9, float)
        assert isinstance(dim10, int)
