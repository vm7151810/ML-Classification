import os
import json
import numpy as np
import faiss
from src.config import config

# For mocking in tests
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

class AlignmentError(Exception):
    """Raised if the length of ids does not match the length of texts during index building."""
    pass

class SemanticIndex:
    def __init__(self):
        """
        Initializes the SemanticIndex state.
        Does NOT instantiate the SentenceTransformer here to prevent VRAM hoarding.
        """
        self._faiss_index = None
        self._ids = []
        self._is_empty = True
        
        # REQ-SEM-12: Thread Thrashing Defense
        # Explicitly disable FAISS internal OpenMP multithreading by default,
        # but expose it via the FAISS_OMP_NUM_THREADS environment variable.
        faiss.omp_set_num_threads(config.FAISS_OMP_NUM_THREADS)
        
    def build(self, ids: list, texts: list):
        """
        Builds the FAISS index by batch embedding the provided texts.
        """
        # REQ-SEM-09: Prevent silent misalignments
        if len(ids) != len(texts):
            raise AlignmentError(f"Length of ids ({len(ids)}) must match length of texts ({len(texts)}).")
            
        # REQ-SEM-05: Zero-Vocabulary Defense
        if not ids:
            self._is_empty = True
            return
            
        self._is_empty = False
        
        # REQ-SEM-15: Local instantiation to prevent VRAM hoarding.
        # Uses the global SentenceTransformer reference so it can be mocked in tests.
        model = SentenceTransformer(config.SENTENCE_TRANSFORMER_MODEL)
        
        # REQ-SEM-06: Explicitly use convert_to_numpy=True to flush embeddings from VRAM.
        # Expose the FAISS_EMBED_BATCH_SIZE environment variable to control GPU VRAM usage.
        embeddings = model.encode(texts, batch_size=config.FAISS_EMBED_BATCH_SIZE, convert_to_numpy=True)
        
        # Cast to float32 safely
        embeddings = np.asarray(embeddings, dtype=np.float32)
        
        # REQ-SEM-04: Symmetric Normalization Math
        # Strictly enforce L2 normalization on all corpus embeddings prior to FAISS ingestion
        # Manual numpy normalization to ensure mathematical correctness even if faiss is mocked in tests
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        embeddings = embeddings / norms
        
        # Also call faiss normalize just to satisfy any downstream duck-typing if needed, though redundant
        try:
            faiss.normalize_L2(embeddings)
        except AttributeError:
            pass
        
        d = embeddings.shape[1]
        
        # REQ-SEM-01: Manage a faiss.IndexFlatIP (Inner Product) index
        # Expose FAISS_INDEX_TYPE environment variable with fallback to IndexFlatIP
        if hasattr(faiss, config.FAISS_INDEX_TYPE):
            self._faiss_index = getattr(faiss, config.FAISS_INDEX_TYPE)(d)
        else:
            self._faiss_index = faiss.IndexFlatIP(d)
        
        # Inject the embeddings into the FAISS index
        self._faiss_index.add(embeddings)
        
        # REQ-SEM-02: Manage the mapping between FAISS integer indices and entity_ids
        self._ids = ids
        
        # Note: REQ-SEM-05 Memory Leak Defense is naturally satisfied as `texts`
        # is a local parameter and we do not store it in `self`.
        
    def save(self, dir_path: str):
        """
        Persists the FAISS index and the IDs array to disk.
        """
        os.makedirs(dir_path, exist_ok=True)
        
        # Persist the internal ids array bypassing Python pickle limitations
        ids_path = os.path.join(dir_path, 'ids.json')
        with open(ids_path, 'w', encoding='utf-8') as f:
            json.dump(self._ids, f)
            
        # REQ-SEM-13: Use faiss.write_index() to persist the C++ FAISS index
        if not self._is_empty and self._faiss_index is not None:
            faiss_path = os.path.join(dir_path, 'index.faiss')
            faiss.write_index(self._faiss_index, faiss_path)
            
    def load(self, dir_path: str):
        """
        Loads the FAISS index and the IDs array from disk.
        """
        ids_path = os.path.join(dir_path, 'ids.json')
        with open(ids_path, 'r', encoding='utf-8') as f:
            self._ids = json.load(f)
            
        # REQ-SEM-14: Natively reconstruct the state using faiss.read_index()
        if not self._ids:
            self._is_empty = True
            self._faiss_index = None
        else:
            self._is_empty = False
            faiss_path = os.path.join(dir_path, 'index.faiss')
            self._faiss_index = faiss.read_index(faiss_path)
            
    def search(self, query_embedding, k: int):
        """
        Searches the FAISS index for the k most similar candidate entities.
        """
        # REQ-SEM-07: Empty Query Trap Defense
        if query_embedding is None:
            return []
            
        # Create a copy so we don't modify the caller's array during normalization
        query_embedding = np.array(query_embedding, dtype=np.float32, copy=True)
        
        # If the embedding is all zeros, short-circuit
        if not np.any(query_embedding):
            return []
            
        # Organic skip if empty corpus
        if self._is_empty or self._faiss_index is None:
            return []
            
        # REQ-SEM-07: FAISS C++ Segfault Defense
        # Aggressively cast to float32 (already done above) and safely enforce reshape
        query_embedding = query_embedding.reshape(1, -1)
        
        # REQ-SEM-04: Symmetric Normalization Math
        # Strictly apply L2 normalization to the query_embedding
        norms = np.linalg.norm(query_embedding, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        query_embedding = query_embedding / norms
        
        try:
            faiss.normalize_L2(query_embedding)
        except AttributeError:
            pass
        
        # REQ-SEM-08: Strictly enforce k <= faiss_index.ntotal
        actual_k = min(k, self._faiss_index.ntotal)
        
        if actual_k == 0:
            return []
            
        distances, indices = self._faiss_index.search(query_embedding, actual_k)
        
        results = []
        for i in range(actual_k):
            dist = distances[0][i]
            idx = indices[0][i]
            
            # REQ-SEM-03: Guarantee that returned candidate ranks are strictly 1-indexed
            rank = i + 1
            
            # Ensure the cosine similarity score is strictly bounded between [-1.0, 1.0]
            score = max(-1.0, min(1.0, float(dist)))
            
            entity_id = self._ids[idx]
            
            results.append({
                "id": entity_id,
                "score": score,
                "rank": rank
            })
            
        return results
        
    def get_self_score(self, query_embedding) -> float:
        """
        Returns a hardcoded constant to satisfy the DAG interface duck-typing.
        """
        # REQ-SEM-11: Maintain duck-typing interface parity with Lexical indices
        if query_embedding is None or not np.any(query_embedding):
            return 1e-9
        return 1.0
