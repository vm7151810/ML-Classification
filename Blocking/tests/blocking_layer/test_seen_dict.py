"""
Test suite for the SeenStateManager in the seen_dict module.
Validates all EARS requirements, error boundaries, schemas, and strict type checking.
"""

import os
import json
import pytest
from unittest.mock import patch, mock_open, call

# Assuming the developing agent will structure the code as follows:
from src.blocking_layer.seen_dict.seen_dict import (
    SeenStateManager,
    AlreadyActiveError,
    AlreadyFinalizedError,
    FileWriteError
)


@pytest.fixture
def temp_env(monkeypatch):
    """Fixture to reset env vars for testing."""
    monkeypatch.setenv("BATCH_SIZE", "2")
    yield
    monkeypatch.delenv("BATCH_SIZE", raising=False)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provides a temporary directory for file output tests."""
    return str(tmp_path)


def test_init_entity_success(temp_env):
    """REQ-SD-02: verify init_entity sets up a template dictionary."""
    manager = SeenStateManager(run_dir="/tmp")
    manager.init_entity("S1-001")
    
    # Internal state verification (assuming standard active memory structure)
    assert "S1-001" in manager.active_memory
    assert manager.active_memory["S1-001"]["accepted"] == set()
    assert manager.active_memory["S1-001"]["rejected"] == set()
    assert manager.active_memory["S1-001"]["per_cycle"] == {}


def test_init_entity_already_active(temp_env):
    """REQ-SD-02: Active State Wipe Defense."""
    manager = SeenStateManager(run_dir="/tmp")
    manager.init_entity("S1-001")
    with pytest.raises(AlreadyActiveError):
        manager.init_entity("S1-001")


def test_init_entity_already_finalized(temp_env):
    """REQ-SD-09: Violent prevention of duplicate state corruption."""
    manager = SeenStateManager(run_dir="/tmp")
    manager.completed_ids.add("S1-001")
    with pytest.raises(AlreadyFinalizedError):
        manager.init_entity("S1-001")


def test_update_seen_success(temp_env):
    """REQ-SD-03: update_seen adds to per_cycle and accepted/rejected sets."""
    manager = SeenStateManager(run_dir="/tmp")
    # REQ-SD-07: Auto-init triggers since S1-001 doesn't exist
    manager.update_seen("S1-001", "C-01", 1, True)
    manager.update_seen("S1-001", "C-02", 1, False)
    manager.update_seen("S1-001", "C-03", 2, True)
    
    assert "C-01" in manager.active_memory["S1-001"]["accepted"]
    assert "C-02" in manager.active_memory["S1-001"]["rejected"]
    assert "C-03" in manager.active_memory["S1-001"]["accepted"]
    
    # Check per_cycle schema
    assert manager.active_memory["S1-001"]["per_cycle"] == {
        "1": ["C-01", "C-02"],
        "2": ["C-03"]
    }


def test_update_seen_duplicate_ignore(temp_env):
    """REQ-SD-06: Ignored if already seen."""
    manager = SeenStateManager(run_dir="/tmp")
    manager.update_seen("S1-001", "C-01", 1, True)
    # This should be ignored
    manager.update_seen("S1-001", "C-01", 1, True)
    manager.update_seen("S1-001", "C-01", 2, False) # Same candidate, different cycle/flag
    
    assert len(manager.active_memory["S1-001"]["accepted"]) == 1
    assert len(manager.active_memory["S1-001"]["rejected"]) == 0
    assert manager.active_memory["S1-001"]["per_cycle"] == {"1": ["C-01"]}


def test_update_seen_already_finalized(temp_env):
    """REQ-SD-09: Prevents updates on finalized entities."""
    manager = SeenStateManager(run_dir="/tmp")
    manager.completed_ids.add("S1-001")
    with pytest.raises(AlreadyFinalizedError):
        manager.update_seen("S1-001", "C-01", 1, True)


def test_has_been_seen_o1_lookup(temp_env):
    """REQ-SD-05: O(1) unified state lookup."""
    manager = SeenStateManager(run_dir="/tmp")
    manager.update_seen("S1-001", "C-01", 1, True)
    manager.update_seen("S1-001", "C-02", 1, False)
    
    assert manager.has_been_seen("S1-001", "C-01") is True
    assert manager.has_been_seen("S1-001", "C-02") is True
    assert manager.has_been_seen("S1-001", "C-03") is False
    
    # REQ-SD-07 check: has_been_seen should not auto-init, just return False if entity missing
    assert manager.has_been_seen("S1-MISSING", "C-01") is False


def test_finalize_entity_auto_init(temp_env):
    """REQ-SD-07: Auto-init triggers for zero-match entities during finalization."""
    manager = SeenStateManager(run_dir="/tmp")
    # Mock flush to isolate this behavior
    with patch.object(manager, '_flush_buffer') as mock_flush:
        manager.finalize_entity("S1-000")
        assert "S1-000" not in manager.active_memory
        assert "S1-000" in manager.completed_ids


def test_finalize_entity_already_finalized(temp_env):
    """REQ-SD-09: AlreadyFinalizedError on duplicate finalize."""
    manager = SeenStateManager(run_dir="/tmp")
    manager.completed_ids.add("S1-001")
    with pytest.raises(AlreadyFinalizedError):
        manager.finalize_entity("S1-001")


def test_micro_batch_flush_trigger(temp_env, temp_output_dir):
    """
    REQ-SD-08: Tests BATCH_SIZE trigger, jsonl append, sorted sets, and memory clear.
    """
    state_file = os.path.join(temp_output_dir, "state.jsonl")
    manager = SeenStateManager(run_dir=temp_output_dir)
    
    # BATCH_SIZE is 2 from fixture
    manager.update_seen("S1-001", "C-02", 1, True)
    manager.update_seen("S1-001", "C-01", 1, False)
    manager.finalize_entity("S1-001")
    
    # Not flushed yet
    assert "S1-001" not in manager.active_memory
    assert len(manager.buffer) == 1
    assert not os.path.exists(state_file)
    
    manager.update_seen("S1-002", "C-03", 1, True)
    manager.finalize_entity("S1-002")
    
    # Flush should have triggered!
    assert "S1-001" not in manager.active_memory
    assert "S1-002" not in manager.active_memory
    assert os.path.exists(state_file)
    
    # Check JSON Set Serialization Defense (sets must be mathematically sorted lists)
    with open(state_file, "r") as f:
        lines = f.readlines()
        
    assert len(lines) == 2
    state_1 = json.loads(lines[0])
    
    # Validate State Schema from REQ-SD-08 / Section 3
    assert "S1-001" in state_1
    assert state_1["S1-001"]["accepted"] == ["C-02"]
    # Sets cast to sorted lists explicitly: "C-01"
    assert state_1["S1-001"]["rejected"] == ["C-01"]
    assert state_1["S1-001"]["per_cycle"] == {"1": ["C-02", "C-01"]}


def test_startup_parsing_completed_ids(temp_env, temp_output_dir):
    """REQ-SD-08: Parsing existing .jsonl to populate completed_ids on startup."""
    state_file = os.path.join(temp_output_dir, "state.jsonl")
    with open(state_file, "w") as f:
        f.write(json.dumps({"S1-001": {"accepted": [], "rejected": [], "per_cycle": {}}}) + "\n")
        f.write(json.dumps({"S1-002": {"accepted": [], "rejected": [], "per_cycle": {}}}) + "\n")
        
    manager = SeenStateManager(run_dir=temp_output_dir)
    assert manager.completed_ids == {"S1-001", "S1-002"}


def test_write_all_outputs(temp_env, temp_output_dir):
    """
    REQ-SD-04: Force flush, parse jsonl, Missing Match Fallback, deterministic sorted outputs.
    """
    state_file = os.path.join(temp_output_dir, "state.jsonl")
    manager = SeenStateManager(run_dir=temp_output_dir)
    
    # Entity 1: standard match
    manager.update_seen("S1-001", "B", 1, True)
    manager.update_seen("S1-001", "A", 1, True)
    manager.finalize_entity("S1-001")
    
    # Entity 2: zero match (Missing Match Fallback)
    manager.finalize_entity("S1-002") # Auto-inits pristine state
    
    # Trigger write all outputs (which force flushes the 2 entities in buffer)
    manager.write_all_outputs()
    
    # Verify candidate_pairs.tsv
    with open(os.path.join(temp_output_dir, "candidate_pairs.tsv"), "r") as f:
        cp_lines = f.read().splitlines()
    assert "S1-001\tA,B" in cp_lines # Mathematical sorting A,B
    assert "S1-002\t" in cp_lines    # Empty RHS fallback
    
    # Verify matching_results.tsv (only accepted ones)
    with open(os.path.join(temp_output_dir, "matching_results.tsv"), "r") as f:
        mr_lines = f.read().splitlines()
    assert "S1-001\tA,B" in mr_lines
    assert "S1-002\t" in mr_lines
    
    # Verify candidate_pairs_per_cycle.tsv
    with open(os.path.join(temp_output_dir, "candidate_pairs_per_cycle.tsv"), "r") as f:
        cppc_lines = f.read().splitlines()
    assert "S1-001\t1\tA,B" in cppc_lines
    # For zero matches, what cycle number should it have? The spec says empty RHS across all.
    # We expect something like "S1-002\\t\\t" or "S1-002\\t1\\t" based on implementation, 
    # but strictly checking empty RHS at the end.
    assert any(line.startswith("S1-002\t") and line.endswith("\t") for line in cppc_lines)


def test_file_write_error(temp_env, tmp_path):
    """Errors: FileWriteError on un-writable output dir."""
    bad_dir = tmp_path / "bad_dir"
    bad_dir.mkdir(mode=0o444) # Read only
    
    manager = SeenStateManager(run_dir=str(bad_dir))
    manager.update_seen("S1-001", "C-01", 1, True)
    manager.finalize_entity("S1-001")
    
    with pytest.raises(FileWriteError):
        manager.write_all_outputs()
        
    bad_dir.chmod(0o777) # Cleanup


# --- STRICT TYPE CHECKING TESTS ---

def test_strict_types_init_entity():
    manager = SeenStateManager(run_dir="/tmp")
    with pytest.raises(TypeError):
        manager.init_entity(123) # type: ignore


def test_strict_types_update_seen():
    manager = SeenStateManager(run_dir="/tmp")
    with pytest.raises(TypeError):
        manager.update_seen(123, "C", 1, True) # type: ignore
    with pytest.raises(TypeError):
        manager.update_seen("S1", 123, 1, True) # type: ignore
    with pytest.raises(TypeError):
        manager.update_seen("S1", "C", "1", True) # type: ignore
    with pytest.raises(TypeError):
        manager.update_seen("S1", "C", 1, "True") # type: ignore


def test_strict_types_finalize_entity():
    manager = SeenStateManager(run_dir="/tmp")
    with pytest.raises(TypeError):
        manager.finalize_entity(123) # type: ignore


def test_strict_types_has_been_seen():
    manager = SeenStateManager(run_dir="/tmp")
    with pytest.raises(TypeError):
        manager.has_been_seen(123, "C") # type: ignore
    with pytest.raises(TypeError):
        manager.has_been_seen("S1", 123) # type: ignore


def test_strict_types_write_all_outputs():
    manager = SeenStateManager(run_dir="/tmp")
    with pytest.raises(TypeError):
        manager.write_all_outputs(123) # type: ignore
