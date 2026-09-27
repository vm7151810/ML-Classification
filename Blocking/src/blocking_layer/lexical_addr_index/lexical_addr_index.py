import logging
import numpy as np
from rank_bm25 import BM25Okapi

from src.config import config
from src.blocking_layer.blocking_utils.blocking_utils import to_char_3grams
from src.blocking_layer.bm25_utils.bm25_utils import calculate_self_score

logger = logging.getLogger(__name__)

class AlignmentError(Exception):
    """Raised when the lengths of ids and texts do not match."""
    pass

class InvalidKError(Exception):
    """Raised when k is zero or negative."""
    pass

class LexicalAddrIndex:
    """
    Exact lexical matching on entity addresses using the BM25Okapi algorithm.
    """
    
    def __init__(self, k1=None, b=None):
        """
        Initialize the LexicalAddrIndex.
        
        Args:
            k1 (float, optional): BM25 term frequency saturation parameter.
            b (float, optional): BM25 length normalization parameter.
        """
        if k1 is not None:
            self.k1 = float(k1)
        else:
            # REQ-LAI-01: Load from config if not provided
            self.k1 = float(config.BM25_K1)
            
        if b is not None:
            self.b = float(b)
        else:
            # REQ-LAI-01: Load from config if not provided
            self.b = float(config.BM25_B)
            
        self._is_empty = True
        self._bm25 = None
        self._id_mapping = []  # Maps internal integer index to actual entity_id
        
    def build(self, ids: list[str], texts: list[str]):
        """
        Build the BM25 index from a target corpus.
        
        Args:
            ids (list[str]): The entity_ids of the Target_C dataset.
            texts (list[str]): The raw addr_for_bm25 strings of the Target_C dataset.
        """
        # REQ-LAI-10: AlignmentError
        if len(ids) != len(texts):
            raise AlignmentError(f"Length of ids ({len(ids)}) must match length of texts ({len(texts)}).")
            
        # REQ-LAI-02: Internally manage the mapping
        self._id_mapping = list(ids)
        
        # REQ-LAI-06: Lazy generator expression to prevent OOM
        # REQ-LAI-05: Strictly consume to_char_3grams
        token_gen = (to_char_3grams(t) for t in texts)
        
        # Zero-Vocabulary Defense using EAFP pattern
        try:
            self._bm25 = BM25Okapi(token_gen, k1=self.k1, b=self.b)
        except Exception as e:
            logger.warning(f"Failed to instantiate BM25Okapi (possibly empty vocabulary): {e}")
            self._is_empty = True
            return
            
        # Post-instantiation check for empty vocabulary
        if len(self._bm25.idf) == 0:
            self._is_empty = True
        else:
            self._is_empty = False
            
    def search(self, query_tokens: list[str], k: int) -> list[dict]:
        """
        Search for top-k matching addresses using the built BM25 index.
        
        Args:
            query_tokens (list[str]): Pre-computed 3-gram tokens of the S1 query.
            k (int): Maximum number of results to return.
            
        Returns:
            list[dict]: Standardized list of candidate dictionaries.
        """
        # Invalid K check
        if k <= 0:
            raise InvalidKError(f"k must be greater than 0, got {k}")
            
        # REQ-LAI-07: Empty Query Trap Defense
        if not query_tokens:
            return []
            
        # REQ-LAI-07: Zero-Vocabulary Defense
        if self._is_empty or self._bm25 is None:
            return []
            
        # Call .get_scores over entire corpus
        scores = self._bm25.get_scores(query_tokens)
        
        # REQ-LAI-07: Filter the scores using boolean mask BEFORE applying argsort
        valid_idx = np.where(scores > 0.0)[0]
        
        if len(valid_idx) == 0:
            return []
            
        valid_scores = scores[valid_idx]
        
        # Stable sort the valid scores descending
        # Negative valid scores with kind='stable' for deterministic tie-breaking
        # slice notation handles k > len(valid_scores) automatically
        top_k_indices_of_valid = np.argsort(-valid_scores, kind='stable')[:k]
        
        results = []
        # Rank is strictly 1-indexed
        for rank, local_idx in enumerate(top_k_indices_of_valid, start=1):
            original_corpus_idx = valid_idx[local_idx]
            entity_id = self._id_mapping[original_corpus_idx]
            score = float(valid_scores[local_idx])
            results.append({
                "id": entity_id,
                "score": score,
                "rank": rank
            })
            
        return results
        
    def get_self_score(self, query_tokens: list[str]) -> float:
        """
        Calculates the BM25 score of a query against itself.
        
        Args:
            query_tokens (list[str]): Tokens of the query.
            
        Returns:
            float: BM25 score, bounded to a minimum of 1e-9.
        """
        # REQ-LAI-04: Zero-Vocabulary Defense
        if self._is_empty or self._bm25 is None:
            return 1e-9
            
        return calculate_self_score(
            query_tokens=query_tokens,
            idf_dict=self._bm25.idf,
            avgdl=self._bm25.avgdl,
            k1=self._bm25.k1,
            b=self._bm25.b
        )
