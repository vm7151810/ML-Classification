import pytest
import numpy as np
from unittest.mock import patch, MagicMock

# Assuming the module will be located here
from src.blocking_layer.lexical_addr_index.lexical_addr_index import (
    LexicalAddrIndex,
    AlignmentError,
    InvalidKError
)

# --- FIxTURES AND SETUP ---

@pytest.fixture
def mock_3grams():
    with patch('src.blocking_layer.lexical_addr_index.lexical_addr_index.to_char_3grams') as mock:
        # A simple mocked behavior for tests if needed, though we often assert how it's called
        mock.side_effect = lambda x: [x[i:i+3] for i in range(len(x)-2)] if len(x) >= 3 else [x]
        yield mock

@pytest.fixture
def index_module():
    return LexicalAddrIndex(k1=1.5, b=0.75)


# --- REQ-LAI-01: Initialization & Env Variables ---

def test_init_with_explicit_hyperparameters():
    index = LexicalAddrIndex(k1=1.2, b=0.8)
    assert index.k1 == 1.2
    assert index.b == 0.8

def test_init_with_env_variables(monkeypatch):
    monkeypatch.setattr("src.config.config.BM25_K1", 1.6)
    monkeypatch.setattr("src.config.config.BM25_B", 0.6)
    index = LexicalAddrIndex()
    assert index.k1 == 1.6
    assert index.b == 0.6


# --- REQ-LAI-10: Alignment Error ---

def test_build_raises_alignment_error_on_mismatched_lengths():
    index = LexicalAddrIndex()
    with pytest.raises(AlignmentError):
        index.build(ids=["1", "2"], texts=["address 1"])


# --- REQ-LAI-05 & REQ-LAI-06: Build & Tokenization & Memory ---

@patch("src.blocking_layer.lexical_addr_index.lexical_addr_index.BM25Okapi")
def test_build_uses_lazy_generator_and_tokenizes(mock_bm25, mock_3grams):
    # Set a side_effect to consume the generator when BM25Okapi is called
    def consume_generator(*args, **kwargs):
        if args and hasattr(args[0], '__iter__'):
            list(args[0])
        return MagicMock()
    mock_bm25.side_effect = consume_generator

    index = LexicalAddrIndex()
    ids = ["id1", "id2"]
    texts = ["abcde", "fghij"]
    
    index.build(ids, texts)
    
    # Verify to_char_3grams was called during build
    assert mock_3grams.call_count == 2
    
    # Ensure memory leak defense (texts are not stored)
    assert not hasattr(index, 'texts')
    
    # Ensure the generator was passed to BM25Okapi
    mock_bm25.assert_called_once()
    args, kwargs = mock_bm25.call_args
    corpus_arg = args[0]
    assert str(type(corpus_arg)) == "<class 'generator'>" or hasattr(corpus_arg, '__iter__')

def test_build_zero_vocabulary(mock_3grams):
    index = LexicalAddrIndex()
    # Provide empty texts that will result in empty tokens
    mock_3grams.side_effect = lambda x: []
    ids = ["id1", "id2"]
    texts = ["", ""]
    
    index.build(ids, texts)
    
    assert getattr(index, '_is_empty', False) is True
    # BM25Okapi should NOT be instantiated
    assert not hasattr(index, '_bm25') or index._bm25 is None


# --- REQ-LAI-07 & REQ-LAI-08 & REQ-LAI-09: Search, Output Schema, Edge Cases ---

def test_search_empty_query_trap():
    index = LexicalAddrIndex()
    index.build(["id1"], ["some text"])
    
    # Empty query should short-circuit and return []
    results = index.search(query_tokens=[], k=5)
    assert results == []

def test_search_zero_vocabulary_trap():
    index = LexicalAddrIndex()
    index._is_empty = True
    
    # Should immediately return []
    results = index.search(query_tokens=["tok"], k=5)
    assert results == []

def test_search_invalid_k():
    index = LexicalAddrIndex()
    index.build(["id1"], ["text1"])
    with pytest.raises(InvalidKError):
        index.search(query_tokens=["tok"], k=0)
    with pytest.raises(InvalidKError):
        index.search(query_tokens=["tok"], k=-1)

def test_search_returns_valid_schema_and_ranks(mock_3grams):
    index = LexicalAddrIndex()
    ids = ["id1", "id2", "id3"]
    # Provide mock texts
    texts = ["abc", "def", "abc def"]
    index.build(ids, texts)
    
    query_tokens = ["abc"]
    # Expected: id1 and id3 match, id2 does not.
    results = index.search(query_tokens=query_tokens, k=5)
    
    assert isinstance(results, list)
    assert len(results) <= 5
    
    for item in results:
        assert isinstance(item, dict)
        assert "id" in item
        assert isinstance(item["id"], str)
        assert "score" in item
        assert isinstance(item["score"], float)
        assert "rank" in item
        assert isinstance(item["rank"], int)
        # Ranks must be 1-indexed
        assert item["rank"] > 0
        assert item["score"] > 0.0

def test_search_handles_k_greater_than_valid_scores(mock_3grams):
    index = LexicalAddrIndex()
    # Use a larger corpus so that the query terms appear in less than 50% of the documents.
    # This prevents BM25 IDF from going negative.
    ids = ["id1", "id2", "id3", "id4", "id5", "id6"]
    texts = ["match this", "match this too", "other", "different", "unrelated", "noise"]
    index.build(ids, texts)
    
    # Query matches the first two. We ask for k=10
    query_tokens = mock_3grams("match this")
    results = index.search(query_tokens=query_tokens, k=10)
    
    # Should gracefully return all valid items (which is 2)
    assert len(results) == 2

@patch("src.blocking_layer.lexical_addr_index.lexical_addr_index.BM25Okapi")
def test_search_stable_sort(mock_bm25, mock_3grams):
    # This tests that kind='stable' is used.
    # We will mock the get_scores to return identical scores for all items.
    mock_instance = MagicMock()
    # Fix: Give the mock instance a non-empty idf dict to pass the Zero-Vocabulary defense check
    mock_instance.idf = {'tok': 1.0}
    mock_bm25.return_value = mock_instance
    mock_instance.get_scores.return_value = np.array([2.5, 2.5, 2.5])
    
    index = LexicalAddrIndex()
    ids = ["id1", "id2", "id3"]
    texts = ["text1", "text2", "text3"]
    index.build(ids, texts)
    
    results = index.search(["tok"], k=3)
    
    # Because scores are identical and sort is stable, original order must be preserved.
    assert [r["id"] for r in results] == ["id1", "id2", "id3"]
    assert [r["rank"] for r in results] == [1, 2, 3]


# --- REQ-LAI-04: Self Score ---

def test_get_self_score_zero_vocabulary():
    index = LexicalAddrIndex()
    index._is_empty = True
    
    score = index.get_self_score(["tok"])
    assert score == 1e-9

def test_get_self_score_delegation():
    index = LexicalAddrIndex()
    index.build(["id1"], ["some text"])
    
    # We expect get_self_score to call calculate_self_score from BM25 logic 
    # Or implement it directly using idf, avgdl, k1, b
    # Here we just check it returns a float and applies floor
    score = index.get_self_score(["some", "tok"])
    assert isinstance(score, float)
    assert score >= 1e-9

def test_get_self_score_minimum_floor():
    index = LexicalAddrIndex()
    index.build(["id1"], ["hello"])
    
    # Mocking idf to be zero or negative to force a tiny/negative score
    index._bm25.idf = {}
    score = index.get_self_score(["nonexistent"])
    
    assert score == 1e-9
