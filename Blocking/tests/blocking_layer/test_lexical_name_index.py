import pytest
import os
from unittest.mock import patch, MagicMock

# Assuming the implementation is in src.blocking_layer.lexical_name_index.lexical_name_index
from src.blocking_layer.lexical_name_index.lexical_name_index import LexicalNameIndex, AlignmentError, InvalidKError

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setattr("src.config.config.BM25_K1", 1.2, raising=False)
    monkeypatch.setattr("src.config.config.BM25_B", 0.8, raising=False)

def test_initialization_with_explicit_params():
    index = LexicalNameIndex(k1=2.0, b=0.5)
    assert index.k1 == 2.0
    assert index.b == 0.5

def test_initialization_with_env_params(mock_env):
    index = LexicalNameIndex()
    assert index.k1 == 1.2
    assert index.b == 0.8

def test_initialization_with_defaults(monkeypatch):
    monkeypatch.setattr("src.config.config.BM25_K1", 1.5, raising=False)
    monkeypatch.setattr("src.config.config.BM25_B", 0.75, raising=False)
    index = LexicalNameIndex()
    assert index.k1 == 1.5
    assert index.b == 0.75

def test_build_alignment_error():
    index = LexicalNameIndex()
    with pytest.raises(AlignmentError):
        index.build(ids=["1", "2"], texts=["only one"])

@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_build_stores_ids_and_sets_up_bm25(mock_to_char_3grams):
    mock_to_char_3grams.side_effect = lambda x: [x[i:i+3] for i in range(len(x)-2)] if len(x) >= 3 else [x]
    
    index = LexicalNameIndex()
    ids = ["id1", "id2"]
    texts = ["apple", "banana"]
    
    index.build(ids=ids, texts=texts)
    
    assert index.ids == ids
    # REQ-LNI-05: Memory Leak Defense - must not retain the raw texts list
    assert not hasattr(index, 'texts') or getattr(index, 'texts') is None, "texts list must not be retained"
    assert index._is_empty is False
    assert index._bm25 is not None

@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_build_zero_vocabulary(mock_to_char_3grams):
    mock_to_char_3grams.return_value = []
    
    index = LexicalNameIndex()
    index.build(ids=["id1"], texts=[""])
    
    assert index._is_empty is True
    assert not hasattr(index, '_bm25') or getattr(index, '_bm25') is None

@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_search_valid_query(mock_to_char_3grams):
    mock_to_char_3grams.side_effect = lambda x: [x[i:i+3] for i in range(len(x)-2)] if len(x) >= 3 else [x]
    
    index = LexicalNameIndex()
    index.build(ids=["id1", "id2", "d1", "d2", "d3"], texts=["apple", "appletree", "x", "y", "z"])
    
    query_tokens = ["app", "ppl", "ple"]
    results = index.search(query_tokens=query_tokens, k=2)
    
    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
    assert "id" in results[0]
    assert "score" in results[0]
    
@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_search_empty_query(mock_to_char_3grams):
    mock_to_char_3grams.side_effect = lambda x: [x]
    index = LexicalNameIndex()
    index.build(ids=["id1"], texts=["apple"])
    
    results = index.search(query_tokens=[], k=5)
    assert results == []

@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_search_zero_vocabulary_flag(mock_to_char_3grams):
    mock_to_char_3grams.return_value = []
    index = LexicalNameIndex()
    index.build(ids=["id1"], texts=[""])
    
    results = index.search(query_tokens=["any"], k=5)
    assert results == []

def test_search_invalid_k():
    index = LexicalNameIndex()
    with pytest.raises(InvalidKError):
        index.search(query_tokens=["abc"], k=0)
    with pytest.raises(InvalidKError):
        index.search(query_tokens=["abc"], k=-1)

@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_search_k_greater_than_valid_documents(mock_to_char_3grams):
    mock_to_char_3grams.side_effect = lambda x: [x[i:i+3] for i in range(len(x)-2)] if len(x) >= 3 else [x]
    index = LexicalNameIndex()
    index.build(ids=["id1", "d1", "d2"], texts=["apple", "x", "y"])
    
    results = index.search(query_tokens=["app", "ppl", "ple"], k=5)
    assert len(results) == 1
    
@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_search_ties_broken_by_original_order(mock_to_char_3grams):
    # Setup tokens so both texts yield the exact same score for a query
    mock_to_char_3grams.side_effect = lambda x: ["abc"] if x == "same" else [x]
    index = LexicalNameIndex()
    index.build(ids=["id1", "id2", "id3", "d1", "d2", "d3", "d4"], texts=["same", "same", "same", "x", "y", "z", "w"])
    
    results = index.search(query_tokens=["abc"], k=3)
    assert len(results) == 3
    # stable sort should keep original order for ties: id1, id2, id3
    assert results[0]["id"] == "id1"
    assert results[1]["id"] == "id2"
    assert results[2]["id"] == "id3"

@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_get_self_score_valid(mock_to_char_3grams):
    mock_to_char_3grams.side_effect = lambda x: [x[i:i+3] for i in range(len(x)-2)] if len(x) >= 3 else [x]
    index = LexicalNameIndex()
    index.build(ids=["id1"], texts=["apple"])
    
    score = index.get_self_score(["app", "ppl", "ple"])
    assert isinstance(score, float)
    assert score >= 1e-9

@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_get_self_score_zero_vocabulary(mock_to_char_3grams):
    mock_to_char_3grams.return_value = []
    index = LexicalNameIndex()
    index.build(ids=["id1"], texts=[""])
    
    score = index.get_self_score(["app"])
    assert score == 1e-9
    
@patch('src.blocking_layer.lexical_name_index.lexical_name_index.to_char_3grams')
def test_search_filters_zero_scores(mock_to_char_3grams):
    def fake_to_char_3grams(text):
        return [text]
    mock_to_char_3grams.side_effect = fake_to_char_3grams
    
    index = LexicalNameIndex()
    index.build(ids=["id1", "id2", "d1", "d2"], texts=["match", "nomatch", "x", "y"])
    
    results = index.search(query_tokens=["match"], k=5)
    assert len(results) == 1
    assert results[0]["id"] == "id1"
