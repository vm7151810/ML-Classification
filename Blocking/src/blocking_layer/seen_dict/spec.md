# Seen Dictionary State Manager Specification

## 1. Description
This module handles global state tracking, cycle management, and cyclic output finalization. It guarantees O(1) lookups for the entity-scoped rejection blacklist and ensures exact output compliance through single-pass writing.

## 2. EARS Requirements

### Ubiquitous Requirements
*   **REQ-SD-01**: The `seen_dict` module shall maintain a global nested dictionary mapped by S1 `entity_id`.

### Event-Driven Requirements
*   **REQ-SD-02**: When an `init_entity` call is made for a new S1 `entity_id`, the module shall initialize a template dictionary containing `accepted` (set), `rejected` (set), and `per_cycle` (dict). **Active State Wipe Defense**: If `init_entity` is called for an entity that already exists in the active memory dictionary, it MUST safely ignore the call or throw an `AlreadyActiveError` to prevent silently wiping out previously evaluated candidates.
*   **REQ-SD-03**: When an `update_seen` call is made, the module shall append the candidate to `per_cycle` for the given cycle count, and add the candidate to either `accepted` or `rejected` based on the provided boolean flag.
*   **REQ-SD-08**: **JSONL Micro-Batch State Tracker**: The `seen_dict` MUST be instantiated with a `run_dir: str` argument provided by the Orchestrator to support dynamic per-run directories. It MUST use this directory for its append-only state file (`run_dir/state.jsonl`). The in-memory batch buffer MUST be configured by reading `os.environ.get("BATCH_SIZE", 10000)` natively to avoid unnecessary `python-dotenv` module dependencies. On initialization, the module MUST parse the existing `.jsonl` file to populate a lightweight `completed_ids: set[str]` in RAM, exposing a `get_completed_ids()` method for the Orchestrator to natively skip processed entities. When an S1 entity is finalized, it enters the buffer and its ID is added to `completed_ids`. When full, the buffer MUST sequentially append the entity JSON strings to disk and instantly clear their detailed state from RAM. **JSON Set Serialization Defense**: Before invoking `json.dumps()` during the flush, all internal `set` objects (`accepted`, `rejected`) MUST be explicitly cast to mathematically sorted lists to prevent a fatal `TypeError: Object of type set is not JSON serializable` and guarantee determinism. This micro-batch design completely prevents OOM crashes, ensures high-speed I/O, and natively provides pipeline resumability.
*   **REQ-SD-04**: When `write_all_outputs` is invoked at the end of the pipeline, the module MUST explicitly execute a mandatory force-flush of any remaining stranded entities in the buffer to the `.jsonl` file *before* it begins parsing the file to generate the TSVs. It shall then generate `matching_results.tsv`, `candidate_pairs.tsv`, and `candidate_pairs_per_cycle.tsv` directly into the `run_dir` by parsing the `.jsonl` file. **Missing Match Fallback**: If an entity has zero candidates for a given output, it MUST output a line with an empty right-hand side (e.g., `S1-00001 \t `) across ALL generated TSV files to ensure all S1 entities are flawlessly represented. **Reproducibility Defense**: The module MUST mathematically sort all candidate groupings (including lists and set unions) before comma-string joining to guarantee 100% deterministic TSV outputs regardless of upstream concurrency.

### State-Driven Requirements
*   **REQ-SD-05**: While filtering new candidate retrievals, the module shall provide O(1) lookup against the unified state (`candidate in accepted or candidate in rejected`) for a given S1 entity to determine if it has been seen.

### Unwanted Behavior Requirements
*   **REQ-SD-06**: If an attempt is made to update a candidate that has already been seen (`candidate in accepted or candidate in rejected`), then the module shall safely ignore the addition to prevent double-counting.
*   **REQ-SD-07**: If an `entity_id` passed to `update_seen` or `finalize_entity` does not exist in the dictionary, then the module MUST automatically invoke `init_entity` to auto-initialize it with a pristine empty state. This strictly prevents fatal `KeyError` crashes for zero-match entities and guarantees they are correctly flushed to disk.
*   **REQ-SD-09**: If `update_seen`, `init_entity`, or `finalize_entity` is called for an ID that is already present in the `completed_ids` set, the module MUST throw a fatal `AlreadyFinalizedError` to violently prevent duplicate state corruption.

## 3. Schemas

### State Schema (Internal structure per S1 entity)
```json
{
  "S1-00001": {
    "accepted": ["S2-0005", "S3-0100"],
    "rejected": ["S2-0099", "S2-0402"],
    "per_cycle": {
      "1": ["S2-0005", "S2-0099"],
      "2": ["S3-0100", "S2-0402"]
    }
  }
}
```

### File Output Schemas
*   **matching_results.tsv**: `source1_entity_id \t matched_entity_ids (comma-separated)`
*   **candidate_pairs.tsv**: `source1_entity_id \t candidate_entity_ids (comma-separated)`
*   **candidate_pairs_per_cycle.tsv**: `source1_entity_id \t cycle_number \t candidate_entity_ids (comma-separated)`

## 4. Errors & Exceptions
*   `AlreadyActiveError`: Raised if `init_entity` is called on an entity currently active in memory, preventing state wipes.
*   `AlreadyFinalizedError`: Raised if an operation is attempted on an entity that has already been flushed to disk.
*   `FileWriteError`: Raised if destination output paths are not writable or the disk is full.
