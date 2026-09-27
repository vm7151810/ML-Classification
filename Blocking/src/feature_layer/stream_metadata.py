from typing import Dict, Any
from src import config

def extract_stream_metadata_features(data: Dict[str, Any]) -> Dict[str, Any]:
    # Dim 14 & 15: Self-Normalized BM25 Scores
    # The denominator is guaranteed by the index to be >= 1e-9.
    denom_name = data.get("bm25_name_self_score", 1e-9)
    if denom_name == 0.0:
        denom_name = 1e-9
    dim_14 = data.get("bm25_name_score", 0.0) / denom_name

    denom_addr = data.get("bm25_addr_self_score", 1e-9)
    if denom_addr == 0.0:
        denom_addr = 1e-9
    dim_15 = data.get("bm25_addr_score", 0.0) / denom_addr

    # Dim 16: RRF Score
    rrf = 0.0
    rank_n = data.get("bm25_name_rank")
    if rank_n is not None:
        rrf += 1.0 / (config.RRF_K_BM25_NAME + rank_n)
    
    rank_a = data.get("bm25_addr_rank")
    if rank_a is not None:
        rrf += 1.0 / (config.RRF_K_BM25_ADDR + rank_a)
        
    rank_f = data.get("faiss_name_rank")
    if rank_f is not None:
        rrf += 1.0 / (config.RRF_K_FAISS + rank_f)
    
    dim_16 = rrf

    # Dim 17: Stream Overlap Count
    dim_17 = 0
    if rank_n is not None: dim_17 += 1
    if rank_a is not None: dim_17 += 1
    if rank_f is not None: dim_17 += 1

    # Dims 18 & 19: Length Ratios
    name_a = data.get("name_a", "")
    name_b = data.get("name_b", "")
    addr_a = data.get("addr_a", "")
    addr_b = data.get("addr_b", "")

    if name_a.startswith("nullname") or name_b.startswith("nullname"):
        dim_18 = 0.0
    else:
        len_na, len_nb = len(name_a), len(name_b)
        max_n = max(len_na, len_nb)
        dim_18 = min(len_na, len_nb) / max_n if max_n > 0 else 0.0

    if addr_a.startswith("nulladdr") or addr_b.startswith("nulladdr"):
        dim_19 = 0.0
    else:
        len_aa, len_ab = len(addr_a), len(addr_b)
        max_a = max(len_aa, len_ab)
        dim_19 = min(len_aa, len_ab) / max_a if max_a > 0 else 0.0

    # Dims 20 - 23: Missing Flags
    dim_20 = 1 if name_a.startswith("nullname") else 0
    dim_21 = 1 if name_b.startswith("nullname") else 0
    dim_22 = 1 if addr_a.startswith("nulladdr") else 0
    dim_23 = 1 if addr_b.startswith("nulladdr") else 0

    # Dim 24: Cross-Script Target
    is_cs_a = data.get("meta_is_cross_script_a", 0)
    is_cs_b = data.get("meta_is_cross_script_b", 0)
    dim_24 = max(is_cs_a, is_cs_b)

    # Dim 25: Source Origin (S2=0, S3=1)
    src = data.get("meta_source", "")
    dim_25 = 1 if src == "S3" else 0

    # Dim 26: Country Raw
    dim_26 = data.get("meta_country", "")

    return {
        "dim_14_name_bm25_norm": dim_14,
        "dim_15_addr_bm25_norm": dim_15,
        "dim_16_rrf_score": dim_16,
        "dim_17_overlap_count": dim_17,
        "dim_18_name_len_ratio": dim_18,
        "dim_19_addr_len_ratio": dim_19,
        "dim_20_s1_name_missing": dim_20,
        "dim_21_cand_name_missing": dim_21,
        "dim_22_s1_addr_missing": dim_22,
        "dim_23_cand_addr_missing": dim_23,
        "dim_24_cross_script": dim_24,
        "dim_25_source_origin": dim_25,
        "dim_26_country_raw": dim_26
    }
