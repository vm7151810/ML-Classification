import numpy as np
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

from src.config import config
from src.blocking_layer.blocking_utils.blocking_utils import to_char_3grams
from src.blocking_layer.bm25_utils.bm25_utils import calculate_self_score

class AlignmentError(Exception):
    """Raised during build() if the length of ids does not match the length of texts."""
    pass

class InvalidKError(Exception):
    """Raised if k is zero or negative during a query."""
    pass

class LexicalNameIndex:
    """
    Exact lexical matching on entity names using the BM25Okapi algorithm.
    """
    
    def __init__(self, k1: float = None, b: float = None):
        """
        Initializes the BM25 parameters, loading them from environment variables if not provided.
        """
        # REQ-LNI-01
        if k1 is None:
            k1 = config.BM25_K1
        if b is None:
            b = config.BM25_B
            
        self.k1 = float(k1)
        self.b = float(b)
        
        self._bm25 = None
        self._is_empty = True
        self.ids: List[str] = []

    def build(self, ids: List[str], texts: List[str]) -> None:
        """
        Builds the internal BM25Okapi index.
        """
        # REQ-LNI-09: Alignment Error
        if len(ids) != len(texts):
            raise AlignmentError(f"Length mismatch: {len(ids)} ids vs {len(texts)} texts")
            
        # REQ-LNI-05: Store the ids list internally.
        self.ids = ids
        
        # Zero-Vocabulary Defense & Generator OOM Defense: 
        # Pass lazy Python generator expression and use EAFP pattern for protection.
        tokenized_corpus = (to_char_3grams(t) for t in texts)
        
        try:
            self._bm25 = BM25Okapi(tokenized_corpus, k1=self.k1, b=self.b)
            
            # Perform the post-instantiation vocabulary check
            if len(self._bm25.idf) == 0:
                self._is_empty = True
            else:
                self._is_empty = False
                
        except Exception: # Catch ZeroDivisionError or any rank-bm25 internal crash
            self._is_empty = True
            self._bm25 = None

    def search(self, query_tokens: List[str], k: int) -> List[Dict[str, Any]]:
        """
        Queries the BM25 index and returns a list of candidate dictionaries.
        """
        if k <= 0:
            raise InvalidKError("k must be greater than zero")
            
        # Empty Query Trap Defense
        if not query_tokens:
            return []
            
        # Zero-Vocabulary Defense
        if self._is_empty:
            return []
            
        # Compute scores using .get_scores(query)
        scores = self._bm25.get_scores(query_tokens)
        
        # Argsort Bottleneck Defense: Filter scores using a boolean mask BEFORE applying argsort
        # Convert to numpy array for boolean masking
        scores_array = np.array(scores)
        valid_idx = np.where(scores_array > 0.0)[0]
        
        if len(valid_idx) == 0:
            return []
            
        valid_scores = scores_array[valid_idx]
        
        # Reproducibility Defense: Apply stable sorting on negative valid scores
        sorted_indices = np.argsort(-valid_scores, kind='stable')[:k]
        
        # Build results ensuring strictly 1-indexed ranks (REQ-LNI-03)
        # Handle REQ-LNI-08 native Out-Of-Bounds safety dynamically by iterating slice length
        results = []
        for rank_idx, sorted_idx in enumerate(sorted_indices):
            original_idx = valid_idx[sorted_idx]
            results.append({
                "id": self.ids[original_idx],
                "score": float(valid_scores[sorted_idx]),
                "rank": rank_idx + 1
            })
            
        return results

    def get_self_score(self, query_tokens: List[str]) -> float:
        """
        Calculates the BM25 score of a query against itself.
        """
        # REQ-LNI-04: Zero-Vocabulary Defense
        if self._is_empty:
            return 1e-9
            
        # REQ-LNI-04: Delegate to calculate_self_score by dynamically passing arguments
        return calculate_self_score(
            query_tokens=query_tokens,
            idf_dict=self._bm25.idf,
            avgdl=self._bm25.avgdl,
            k1=self._bm25.k1,
            b=self._bm25.b
        )
