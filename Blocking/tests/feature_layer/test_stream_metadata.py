import pytest
from typing import Dict, Any

# We assume the developing agent will create a function with this signature.
# If they use a class or different function name, they should update the import.
from src.feature_layer.stream_metadata import extract_stream_metadata_features
from src import config

@pytest.fixture
def default_valid_input() -> Dict[str, Any]:
    return {
        "name_a": "John Doe",
        "name_b": "John Doe",
        "addr_a": "123 Main St",
        "addr_b": "123 Main St",
        "bm25_name_score": 10.5,
        "bm25_addr_score": 8.0,
        "bm25_name_self_score": 12.0,
        "bm25_addr_self_score": 10.0,
        "bm25_name_rank": 1,
        "bm25_addr_rank": 2,
        "faiss_name_rank": 3,
        "meta_is_cross_script_a": 0,
        "meta_is_cross_script_b": 0,
        "meta_source": "S2",
        "meta_country": "us"
    }

class TestStreamMetadataFeatures:

    def test_dims_14_15_bm25_normalization_standard(self, default_valid_input):
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_14_name_bm25_norm"] == pytest.approx(10.5 / 12.0)
        assert output["dim_15_addr_bm25_norm"] == pytest.approx(8.0 / 10.0)



    def test_dim_16_rrf_score_all_ranks_present(self, default_valid_input, monkeypatch):
        # Mock config values for predictable testing
        monkeypatch.setattr(config, "RRF_K_BM25_NAME", 60)
        monkeypatch.setattr(config, "RRF_K_BM25_ADDR", 60)
        monkeypatch.setattr(config, "RRF_K_FAISS", 60)
        
        output = extract_stream_metadata_features(default_valid_input)
        expected_rrf = (1.0 / (60 + 1)) + (1.0 / (60 + 2)) + (1.0 / (60 + 3))
        assert output["dim_16_rrf_score"] == pytest.approx(expected_rrf)

    def test_dim_16_rrf_score_missing_ranks(self, default_valid_input, monkeypatch):
        monkeypatch.setattr(config, "RRF_K_BM25_NAME", 60)
        monkeypatch.setattr(config, "RRF_K_BM25_ADDR", 60)
        monkeypatch.setattr(config, "RRF_K_FAISS", 60)
        
        default_valid_input["bm25_name_rank"] = None
        default_valid_input["bm25_addr_rank"] = 5
        default_valid_input["faiss_name_rank"] = None
        
        output = extract_stream_metadata_features(default_valid_input)
        expected_rrf = 1.0 / (60 + 5)
        assert output["dim_16_rrf_score"] == pytest.approx(expected_rrf)

    def test_dim_17_overlap_count(self, default_valid_input):
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_17_overlap_count"] == 3
        
        default_valid_input["bm25_addr_rank"] = None
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_17_overlap_count"] == 2
        
        default_valid_input["bm25_name_rank"] = None
        default_valid_input["faiss_name_rank"] = None
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_17_overlap_count"] == 0

    def test_dims_18_19_length_ratio_standard(self, default_valid_input):
        default_valid_input["name_a"] = "John" # len 4
        default_valid_input["name_b"] = "John Doe" # len 8
        default_valid_input["addr_a"] = "123 Main Street" # len 15
        default_valid_input["addr_b"] = "123 Main St" # len 11
        
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_18_name_len_ratio"] == pytest.approx(4.0 / 8.0)
        assert output["dim_19_addr_len_ratio"] == pytest.approx(11.0 / 15.0)

    def test_dims_18_19_length_ratio_empty_strings(self, default_valid_input):
        default_valid_input["name_a"] = ""
        default_valid_input["name_b"] = "John"
        default_valid_input["addr_a"] = ""
        default_valid_input["addr_b"] = ""
        
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_18_name_len_ratio"] == 0.0 # 0 / max(0, 4, 1)
        assert output["dim_19_addr_len_ratio"] == 0.0 # 0 / max(0, 0, 1) = 0 / 1

    def test_dims_18_19_length_ratio_sentinel_handling(self, default_valid_input):
        # If either string starts with sentinel, ratio MUST explicitly return 0.0
        default_valid_input["name_a"] = "nullname_123"
        default_valid_input["name_b"] = "John"
        default_valid_input["addr_a"] = "123 Main St"
        default_valid_input["addr_b"] = "nulladdr_xyz"
        
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_18_name_len_ratio"] == 0.0
        assert output["dim_19_addr_len_ratio"] == 0.0

    def test_dims_20_to_23_missing_flags(self, default_valid_input):
        default_valid_input["name_a"] = "nullname_s1"
        default_valid_input["name_b"] = "Valid Name"
        default_valid_input["addr_a"] = "Valid Addr"
        default_valid_input["addr_b"] = "nulladdr_cand"
        
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_20_s1_name_missing"] == 1
        assert output["dim_21_cand_name_missing"] == 0
        assert output["dim_22_s1_addr_missing"] == 0
        assert output["dim_23_cand_addr_missing"] == 1

    def test_dim_24_cross_script(self, default_valid_input):
        default_valid_input["meta_is_cross_script_a"] = 0
        default_valid_input["meta_is_cross_script_b"] = 1
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_24_cross_script"] == 1
        
        default_valid_input["meta_is_cross_script_a"] = 1
        default_valid_input["meta_is_cross_script_b"] = 0
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_24_cross_script"] == 1

    def test_dim_25_source_origin(self, default_valid_input):
        default_valid_input["meta_source"] = "S2"
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_25_source_origin"] == 0
        
        default_valid_input["meta_source"] = "S3"
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_25_source_origin"] == 1

    def test_dim_26_country_raw(self, default_valid_input):
        default_valid_input["meta_country"] = "france"
        output = extract_stream_metadata_features(default_valid_input)
        assert output["dim_26_country_raw"] == "france"
