import pytest
import numpy as np
from src.feature_layer.name_semantic.feature import compute_dim_7_semantic_cosine

class TestNameSemanticFeature:

    def test_valid_cosine_similarity(self):
        # Two perfectly identical vectors
        emb_a = np.array([0.6, 0.8])
        emb_b = np.array([0.6, 0.8])
        # Dot product = 0.36 + 0.64 = 1.0
        result = compute_dim_7_semantic_cosine(emb_a, emb_b)
        assert np.isclose(result, 1.0)

        # Orthogonal vectors
        emb_c = np.array([1.0, 0.0])
        emb_d = np.array([0.0, 1.0])
        result = compute_dim_7_semantic_cosine(emb_c, emb_d)
        assert np.isclose(result, 0.0)

        # Arbitrary vectors
        emb_e = np.array([0.5, 0.5, 0.5, 0.5])
        emb_f = np.array([0.5, 0.5, 0.5, 0.5])
        result = compute_dim_7_semantic_cosine(emb_e, emb_f)
        assert np.isclose(result, 1.0)

    def test_upper_bound_clamping(self):
        # Floating point inaccuracy can cause dot product to slightly exceed 1.0
        # We simulate this by providing un-normalized vectors that result in > 1.0
        # The function should clamp it to 1.0
        emb_a = np.array([1.1, 0.0])
        emb_b = np.array([1.1, 0.0])
        result = compute_dim_7_semantic_cosine(emb_a, emb_b)
        assert result == 1.0

    def test_lower_bound_clamping(self):
        # Opposite direction vectors will have negative dot product.
        # Must be clamped to 0.0
        emb_a = np.array([1.0, 0.0])
        emb_b = np.array([-1.0, 0.0])
        result = compute_dim_7_semantic_cosine(emb_a, emb_b)
        assert result == 0.0

    def test_missing_embeddings_raises_error(self):
        # Spec states: If either embedding is None, crash loudly
        expected_msg = "Semantic embeddings cannot be None. Upstream leakage detected."
        
        with pytest.raises(ValueError, match=expected_msg):
            compute_dim_7_semantic_cosine(None, np.array([1.0, 0.0]))
            
        with pytest.raises(ValueError, match=expected_msg):
            compute_dim_7_semantic_cosine(np.array([1.0, 0.0]), None)
            
        with pytest.raises(ValueError, match=expected_msg):
            compute_dim_7_semantic_cosine(None, None)

    def test_dimension_mismatch_raises_error(self):
        # Spec states: assert emb_a.shape == emb_b.shape
        emb_a = np.array([1.0, 0.0, 0.0])
        emb_b = np.array([1.0, 0.0])
        
        with pytest.raises(AssertionError):
            compute_dim_7_semantic_cosine(emb_a, emb_b)

    def test_not_1d_array_raises_error(self):
        # Spec states: assert emb_a.ndim == 1 and emb_b.ndim == 1
        emb_a = np.array([[1.0, 0.0]])
        emb_b = np.array([1.0, 0.0])
        
        with pytest.raises(AssertionError):
            compute_dim_7_semantic_cosine(emb_a, emb_b)

    def test_nan_values_raises_error(self):
        # The user specifically requested tests for NaN values.
        # NaNs can poison the downstream ML model, so it should raise a ValueError.
        emb_a = np.array([np.nan, 0.0])
        emb_b = np.array([1.0, 0.0])
        
        with pytest.raises(ValueError, match="NaN"):
            compute_dim_7_semantic_cosine(emb_a, emb_b)

    def test_inf_values_raises_error(self):
        # The user specifically requested tests for Inf values.
        emb_a = np.array([np.inf, 0.0])
        emb_b = np.array([1.0, 0.0])
        
        with pytest.raises(ValueError, match="Inf|inf"):
            compute_dim_7_semantic_cosine(emb_a, emb_b)

    @pytest.mark.xfail(reason="FLAGGED: Empty arrays handling is not defined in Feature_Engineering_1.0.md or spec.md")
    def test_empty_arrays(self):
        # FLAGGED behavior: empty arrays of shape (0,)
        # The spec and Feature_Engineering_1.0.md do not mention empty arrays.
        # This test ensures we flag it. In a robust system, this should likely raise a ValueError.
        emb_a = np.array([])
        emb_b = np.array([])
        
        with pytest.raises(ValueError, match="empty"):
            compute_dim_7_semantic_cosine(emb_a, emb_b)
