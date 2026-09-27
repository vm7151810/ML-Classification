import os
import json
from src.config import config

class AlreadyActiveError(Exception):
    """Raised if init_entity is called on an entity currently active in memory, preventing state wipes."""
    pass

class AlreadyFinalizedError(Exception):
    """Raised if an operation is attempted on an entity that has already been flushed to disk."""
    pass

class FileWriteError(Exception):
    """Raised if destination output paths are not writable or the disk is full."""
    pass

class SeenStateManager:
    def __init__(self, run_dir: str = None):
        if run_dir is not None and not isinstance(run_dir, str):
            raise TypeError("run_dir must be a string")
        # Defaulting to current dir if not provided, just in case tests don't provide it
        self.run_dir = run_dir if run_dir is not None else "."
        self.state_file = os.path.join(self.run_dir, "state.jsonl")
        self.completed_ids = set()
        self.active_memory = {}
        self.buffer = []
        self.BATCH_SIZE = config.BATCH_SIZE
        
        # Parse existing .jsonl file if it exists to populate completed_ids
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        for entity_id in record.keys():
                            self.completed_ids.add(entity_id)
                    except json.JSONDecodeError:
                        continue
                        
    def get_completed_ids(self) -> set:
        """Returns the set of entity IDs that have been finalized and flushed."""
        return self.completed_ids

    def init_entity(self, entity_id: str):
        """Initializes a template dictionary for a new S1 entity_id."""
        if not isinstance(entity_id, str):
            raise TypeError("entity_id must be a string")
        if entity_id in self.completed_ids:
            raise AlreadyFinalizedError(f"Entity {entity_id} is already finalized.")
        if entity_id in self.active_memory:
            raise AlreadyActiveError(f"Entity {entity_id} is already active.")
        
        self.active_memory[entity_id] = {
            "accepted": set(),
            "rejected": set(),
            "per_cycle": {}
        }

    def update_seen(self, entity_id: str, candidate_id: str, cycle: int, is_accepted: bool):
        """Updates the state for a given entity and candidate."""
        if not isinstance(entity_id, str):
            raise TypeError("entity_id must be a string")
        if not isinstance(candidate_id, str):
            raise TypeError("candidate_id must be a string")
        if not isinstance(cycle, int):
            raise TypeError("cycle must be an integer")
        if not isinstance(is_accepted, bool):
            raise TypeError("is_accepted must be a boolean")
            
        if entity_id in self.completed_ids:
            raise AlreadyFinalizedError(f"Entity {entity_id} is already finalized.")
            
        if entity_id not in self.active_memory:
            self.init_entity(entity_id)
            
        state = self.active_memory[entity_id]
        
        # Check if candidate has already been seen
        if candidate_id in state["accepted"] or candidate_id in state["rejected"]:
            return
            
        if is_accepted:
            state["accepted"].add(candidate_id)
        else:
            state["rejected"].add(candidate_id)
            
        cycle_str = str(cycle)
        if cycle_str not in state["per_cycle"]:
            state["per_cycle"][cycle_str] = []
        state["per_cycle"][cycle_str].append(candidate_id)

    def has_been_seen(self, entity_id: str, candidate_id: str) -> bool:
        """Checks if a candidate has already been seen for a given entity."""
        if not isinstance(entity_id, str):
            raise TypeError("entity_id must be a string")
        if not isinstance(candidate_id, str):
            raise TypeError("candidate_id must be a string")
            
        if entity_id not in self.active_memory:
            return False
            
        state = self.active_memory[entity_id]
        return candidate_id in state["accepted"] or candidate_id in state["rejected"]

    def finalize_entity(self, entity_id: str):
        """Finalizes an entity, moving it to the micro-batch buffer."""
        if not isinstance(entity_id, str):
            raise TypeError("entity_id must be a string")
            
        if entity_id in self.completed_ids:
            raise AlreadyFinalizedError(f"Entity {entity_id} is already finalized.")
            
        if entity_id not in self.active_memory:
            self.init_entity(entity_id)
            
        state = self.active_memory.pop(entity_id)
        self.completed_ids.add(entity_id)
        
        # Explicitly cast sets to mathematically sorted lists for JSON serialization
        clean_state = {
            "accepted": sorted(list(state["accepted"])),
            "rejected": sorted(list(state["rejected"])),
            "per_cycle": state["per_cycle"]
        }
        
        self.buffer.append({entity_id: clean_state})
        
        if len(self.buffer) >= self.BATCH_SIZE:
            self._flush_buffer()

    def _flush_buffer(self):
        """Flushes the micro-batch buffer to the state file."""
        if not self.buffer:
            return
            
        try:
            # Ensure run_dir exists if it's not current dir
            if self.run_dir and self.run_dir != ".":
                os.makedirs(self.run_dir, exist_ok=True)
            with open(self.state_file, 'a', encoding='utf-8') as f:
                for item in self.buffer:
                    f.write(json.dumps(item) + '\n')
            self.buffer.clear()
        except Exception as e:
            raise FileWriteError(f"Failed to write to {self.state_file}") from e

    def write_all_outputs(self):
        """Force-flushes buffer and parses the state file to generate output TSVs."""
        self._flush_buffer()
        
        matching_results_path = os.path.join(self.run_dir, "matching_results.tsv")
        candidate_pairs_path = os.path.join(self.run_dir, "candidate_pairs.tsv")
        per_cycle_path = os.path.join(self.run_dir, "candidate_pairs_per_cycle.tsv")
        
        try:
            with open(matching_results_path, 'w', encoding='utf-8') as f_match, \
                 open(candidate_pairs_path, 'w', encoding='utf-8') as f_pairs, \
                 open(per_cycle_path, 'w', encoding='utf-8') as f_cycle:
                
                if not os.path.exists(self.state_file):
                    return
                    
                with open(self.state_file, 'r', encoding='utf-8') as f_state:
                    for line in f_state:
                        if not line.strip():
                            continue
                        try:
                            record = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        for entity_id, state in record.items():
                            accepted = state.get("accepted", [])
                            rejected = state.get("rejected", [])
                            per_cycle = state.get("per_cycle", {})
                            
                            # Matching results
                            if not accepted:
                                f_match.write(f"{entity_id}\t\n")
                            else:
                                matched_sorted = sorted(list(set(accepted)))
                                f_match.write(f"{entity_id}\t{','.join(matched_sorted)}\n")
                                
                            # Candidate pairs
                            all_cands = set(accepted).union(set(rejected))
                            if not all_cands:
                                f_pairs.write(f"{entity_id}\t\n")
                            else:
                                cands_sorted = sorted(list(all_cands))
                                f_pairs.write(f"{entity_id}\t{','.join(cands_sorted)}\n")
                                
                            # Per cycle
                            if not per_cycle:
                                f_cycle.write(f"{entity_id}\t\t\n")
                            else:
                                # Ensure we write out deterministically by sorting cycles
                                for cycle in sorted(per_cycle.keys(), key=lambda x: int(x)):
                                    cands_for_cycle = per_cycle[cycle]
                                    if not cands_for_cycle:
                                        f_cycle.write(f"{entity_id}\t{cycle}\t\n")
                                    else:
                                        cycle_cands_sorted = sorted(list(set(cands_for_cycle)))
                                        f_cycle.write(f"{entity_id}\t{cycle}\t{','.join(cycle_cands_sorted)}\n")

        except Exception as e:
            raise FileWriteError("Failed to write TSV outputs.") from e
