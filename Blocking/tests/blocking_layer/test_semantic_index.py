import pytest
import numpy as np
import os
import json
from unittest.mock import MagicMock, patch

# Note: The developer should implement SemanticIndex and AlignmentError in this module.
from src.blocking_layer.semantic_index import SemanticIndex, AlignmentError

@pytest.fixture
def mock_faiss():
    with patch('src.blocking_layer.semantic_index.faiss') as mock:
        mock_index = MagicMock()
        mock.IndexFlatIP.return_value = mock_index
        yield mock

@pytest.fixture
def mock_sentence_transformer():
    with patch('src.blocking_layer.semantic_index.SentenceTransformer') as mock:
        mock_model = MagicMock()
        mock.return_value = mock_model
        
        # Mock encode to return a numpy array of embeddings (e.g., size 384)
        def mock_encode(texts, **kwargs):
            return np.random.rand(len(texts), 384).astype(np.float32)
            
        mock_model.encode.side_effect = mock_encode
        yield mock

def test_req_sem_12_thread_thrashing_defense(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-12: explicit faiss.omp_set_num_threads(1) upon initialization."""
    index = SemanticIndex()
    mock_faiss.omp_set_num_threads.assert_any_call(1)

def test_req_sem_09_build_alignment_error(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-09: raise AlignmentError if len(ids) != len(texts)."""
    index = SemanticIndex()
    with pytest.raises(AlignmentError):
        index.build(ids=["id1", "id2"], texts=["text1"])

def test_req_sem_05_zero_vocabulary_defense(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-05: Empty corpus bypasses FAISS instantiation and handles state properly."""
    index = SemanticIndex()
    index.build(ids=[], texts=[])
    
    assert getattr(index, "_is_empty", False) is True
    mock_faiss.IndexFlatIP.assert_not_called()
    assert getattr(index, "texts", None) is None
    
    # Searching an empty index should safely return []
    query = np.random.rand(1, 384).astype(np.float32)
    results = index.search(query, k=5)
    assert results == []

def test_req_sem_05_memory_leak_defense(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-05: Memory Leak Defense - texts must not be retained."""
    index = SemanticIndex()
    index.build(ids=["id1"], texts=["text1"])
    assert not hasattr(index, "texts") or index.texts is None

def test_req_sem_01_index_initialization(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-01: manages a faiss.IndexFlatIP index."""
    index = SemanticIndex()
    index.build(ids=["id1"], texts=["text1"])
    mock_faiss.IndexFlatIP.assert_called_once()

def test_req_sem_06_convert_to_numpy(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-06: encode uses convert_to_numpy=True explicitly."""
    index = SemanticIndex()
    mock_model = mock_sentence_transformer.return_value
    
    index.build(ids=["id1"], texts=["text1"])
    
    encode_call_kwargs = mock_model.encode.call_args.kwargs
    assert encode_call_kwargs.get("convert_to_numpy") is True

def test_req_sem_10_empty_string_handling(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-10: empty string is deterministically replaced/handled."""
    index = SemanticIndex()
    mock_model = mock_sentence_transformer.return_value
    
    index.build(ids=["id1", "id2"], texts=["text1", ""])
    
    encode_call_texts = mock_model.encode.call_args.args[0]
    # Check that "" was passed natively to sentence-transformers
    assert "" in encode_call_texts

def test_req_sem_07_empty_query_defense(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-07: Zeros or None query safely short-circuits to [] without computation."""
    index = SemanticIndex()
    index.build(ids=["id1"], texts=["text1"])
    
    mock_index = mock_faiss.IndexFlatIP.return_value
    mock_index.search.reset_mock()
    
    assert index.search(None, k=5) == []
    assert index.search(np.zeros((1, 384)), k=5) == []
    mock_index.search.assert_not_called()

def test_req_sem_07_query_casting_and_shape(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-07: Query embedding is cast to np.float32 and safely reshaped to (1, -1)."""
    index = SemanticIndex()
    index.build(ids=["id1"], texts=["text1"])
    
    mock_index = mock_faiss.IndexFlatIP.return_value
    mock_index.ntotal = 1
    mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
    
    # 1D array of float64
    query = np.random.rand(384).astype(np.float64)
    index.search(query, k=5)
    
    search_args = mock_index.search.call_args.args
    processed_query = search_args[0]
    
    assert processed_query.dtype == np.float32
    assert processed_query.shape == (1, 384)

def test_req_sem_02_and_03_rank_and_mapping(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-02 & REQ-SEM-03: Mapping between FAISS IDs and entity_ids, strictly 1-indexed ranks."""
    index = SemanticIndex()
    index.build(ids=["idA", "idB", "idC"], texts=["tA", "tB", "tC"])
    
    mock_index = mock_faiss.IndexFlatIP.return_value
    mock_index.ntotal = 3
    # Mocks FAISS returning the 3rd item (index 2) and 1st item (index 0)
    mock_index.search.return_value = (np.array([[0.95, 0.85]]), np.array([[2, 0]]))
    
    query = np.random.rand(1, 384).astype(np.float32)
    results = index.search(query, k=2)
    
    assert len(results) == 2
    
    # Rank 1 -> idC (index 2)
    assert results[0]["id"] == "idC"
    assert results[0]["score"] == 0.95
    assert results[0]["rank"] == 1
    
    # Rank 2 -> idA (index 0)
    assert results[1]["id"] == "idA"
    assert results[1]["score"] == 0.85
    assert results[1]["rank"] == 2

def test_req_sem_08_k_bound(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-08: Ensure k <= faiss_index.ntotal before querying FAISS."""
    index = SemanticIndex()
    index.build(ids=["id1"], texts=["t1"])
    
    mock_index = mock_faiss.IndexFlatIP.return_value
    mock_index.ntotal = 1
    mock_index.search.return_value = (np.array([[0.9]]), np.array([[0]]))
    
    query = np.random.rand(1, 384).astype(np.float32)
    index.search(query, k=10)  # User requests 10, but ntotal is 1
    
    search_args = mock_index.search.call_args.args
    actual_k_passed_to_faiss = search_args[1]
    
    assert actual_k_passed_to_faiss <= mock_index.ntotal
    assert actual_k_passed_to_faiss == 1

def test_req_sem_11_get_self_score(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-11: get_self_score returns 1.0 (or 1e-9 if empty)."""
    index = SemanticIndex()
    query = np.random.rand(1, 384).astype(np.float32)
    
    assert index.get_self_score(query) == 1.0
    assert index.get_self_score(None) == 1e-9
    assert index.get_self_score(np.zeros((1, 384))) == 1e-9

def test_req_sem_04_l2_normalization(mock_faiss, mock_sentence_transformer):
    """REQ-SEM-04: L2 Normalization on corpus embeddings and query embedding."""
    index = SemanticIndex()
    
    mock_model = mock_sentence_transformer.return_value
    # Encode returns a non-normalized vector
    mock_model.encode.return_value = np.array([[3.0, 4.0, 0.0]], dtype=np.float32)
    
    index.build(ids=["id1"], texts=["t1"])
    
    mock_index = mock_faiss.IndexFlatIP.return_value
    add_args = mock_index.add.call_args.args
    corpus_embeddings_passed = add_args[0]
    
    # Vector length of (3,4,0) is 5. Normalized should be 1.0
    assert np.isclose(np.linalg.norm(corpus_embeddings_passed[0]), 1.0)
    
    # Now for search
    mock_index.ntotal = 1
    mock_index.search.return_value = (np.array([[1.0]]), np.array([[0]]))
    
    query = np.array([[0.0, 10.0, 0.0]], dtype=np.float32)
    index.search(query, k=1)
    
    search_args = mock_index.search.call_args.args
    query_passed = search_args[0]
    
    assert np.isclose(np.linalg.norm(query_passed[0]), 1.0)

def test_req_sem_13_and_14_save_and_load(mock_faiss, mock_sentence_transformer, tmp_path):
    """REQ-SEM-13 & 14: save() uses faiss.write_index, load() uses faiss.read_index and standard I/O for ids."""
    index = SemanticIndex()
    ids = ["id1", "id2"]
    index.build(ids=ids, texts=["t1", "t2"])
    
    # Ensure save handles directories correctly
    save_dir = str(tmp_path / "semantic_index_data")
    os.makedirs(save_dir, exist_ok=True)
    
    index.save(save_dir)
    
    mock_faiss.write_index.assert_called_once()
    faiss_write_index_arg = mock_faiss.write_index.call_args[0][0]
    faiss_write_path_arg = mock_faiss.write_index.call_args[0][1]
    
    assert faiss_write_index_arg == mock_faiss.IndexFlatIP.return_value
    assert save_dir in faiss_write_path_arg
    
    # Verify IDs are saved
    assert os.path.exists(os.path.join(save_dir, "ids.json")) or os.path.exists(os.path.join(save_dir, "ids.npy"))
    
    new_index = SemanticIndex()
    new_index.load(save_dir)
    
    mock_faiss.read_index.assert_called_once()
    faiss_read_path_arg = mock_faiss.read_index.call_args[0][0]
    assert save_dir in faiss_read_path_arg
    
    # Verify state reconstructed
    new_index.faiss_index = mock_faiss.read_index.return_value
    new_index.faiss_index.ntotal = 2
    new_index.faiss_index.search.return_value = (np.array([[1.0]]), np.array([[1]]))
    
    # Because internal names (ids vs _ids) are up to the implementation, rely on public interface search()
    query = np.random.rand(1, 384).astype(np.float32)
    results = new_index.search(query, k=1)
    
    assert len(results) == 1
    assert results[0]["id"] == "id2"
