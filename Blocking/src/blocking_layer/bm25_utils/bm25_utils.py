from typing import List, Dict
import collections
from src.config import config

def calculate_self_score(
    query_tokens: List[str],
    idf_dict: Dict[str, float],
    avgdl: float,
    k1: float,
    b: float
) -> float:
    """
    Calculate the hypothetical BM25 score of a query document against itself.
    
    This function implements a stateless execution of the BM25 formula ensuring
    mathematical parity with rank-bm25 without any hardcoded constants, by taking
    all necessary parameters as inputs.
    
    Args:
        query_tokens (List[str]): The list of tokens in the query document.
        idf_dict (Dict[str, float]): A mapping of token to its inverse document frequency (IDF) in the corpus.
        avgdl (float): The average document length in the corpus.
        k1 (float): The BM25 term frequency saturation parameter.
        b (float): The BM25 length normalization parameter.
        
    Returns:
        float: The calculated self BM25 score, clamped to a minimum of BM25_SELF_SCORE_FLOOR.
    """
    # Clean handling of schemas by explicitly checking input types
    if not isinstance(query_tokens, (list, tuple)):
        raise TypeError(f"query_tokens must be a list or tuple, got {type(query_tokens)}")
    for t in query_tokens:
        if not isinstance(t, str):
            raise TypeError(f"query_tokens elements must be str, got {type(t)}")
            
    if not isinstance(idf_dict, dict):
        raise TypeError(f"idf_dict must be a dictionary, got {type(idf_dict)}")
            
    if not isinstance(avgdl, (float, int)):
        raise TypeError(f"avgdl must be a float or int, got {type(avgdl)}")
    if not isinstance(k1, (float, int)):
        raise TypeError(f"k1 must be a float or int, got {type(k1)}")
    if not isinstance(b, (float, int)):
        raise TypeError(f"b must be a float or int, got {type(b)}")
        
    try:
        # Check if query_tokens is not empty, if it is not valid, or if avgdl is non-positive.
        # REQ-BMU-03: If the query_tokens list is empty, the function shall return a minimum floor value.
        if len(query_tokens) == 0:
            return config.BM25_SELF_SCORE_FLOOR
            
        # REQ-BMU-06: If avgdl <= 0.0, the function MUST gracefully return the minimum floor value 
        # to completely protect the stateless mathematical execution against ZeroDivisionError crashes.
        if float(avgdl) <= 0.0:
            return config.BM25_SELF_SCORE_FLOOR

        # We need the length of query_tokens to determine the document length for BM25 formula
        doc_len = len(query_tokens)
        
        # We calculate the frequency of each term in the query_tokens list
        # This gives us the `freq` variable needed in the term_score formula
        freqs = collections.Counter(query_tokens)
        
        # Initialize the total score
        total_score = 0.0
        
        # Iterate over the unique tokens and their corresponding frequencies
        # REQ-BMU-02: To guarantee mathematical parity, we iterate over unique tokens and 
        # explicitly multiply the resulting term score by the query term frequency.
        for token, freq in freqs.items():
            
            # REQ-BMU-05: If the query_tokens contain Out-Of-Vocabulary (OOV) tokens that are not found 
            # in the idf_dict, the function MUST gracefully default to an IDF of 0.0
            idf = float(idf_dict.get(token, 0.0))
            
            # Calculate the term score according to the provided formula
            # term_score = idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * (len(query_tokens) / avgdl)))
            numerator = float(freq * (k1 + 1.0))
            
            # Denominator calculates the normalization component based on document length and term frequency
            denominator = float(freq + k1 * (1.0 - b + b * (doc_len / avgdl)))
            
            # The base term score for a single occurrence of the token
            term_score = (idf * numerator) / denominator
            
            # Multiply the term score by the term frequency as per REQ-BMU-02 when iterating over unique tokens
            total_score += (term_score * freq)
            
        # REQ-BMU-04: If the calculated score is 0.0 or negative, the function shall clamp the return value.
        if total_score <= 0.0:
            return config.BM25_SELF_SCORE_FLOOR
            
        # Return the strictly calculated, strictly positive BM25 self score
        return float(total_score)
        
    except (ValueError, ArithmeticError):
        # A specific fallback for mathematical errors (like overflow), but let TypeError propagate
        return config.BM25_SELF_SCORE_FLOOR
