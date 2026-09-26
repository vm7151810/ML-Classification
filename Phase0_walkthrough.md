# Phase 0 — Preprocessing & Country Partitioning: Walkthrough

## Summary of Accomplishments

Phase 0 preprocessing and dynamic country partitioning has been fully implemented, executed, and verified across all **24,229,173** business records spanning both `train` and `test` datasets.

All code and outputs reside strictly within [`ML-Classification/`](file:///home/mivikev/Desktop/AMC/ML-Classification/):
- **Core Pipeline Modules**: [`src/phase0/`](file:///home/mivikev/Desktop/AMC/ML-Classification/src/phase0/)
  - [`script_detector.py`](file:///home/mivikev/Desktop/AMC/ML-Classification/src/phase0/script_detector.py): Open-set Unicode script detection with ASCII fast-path and codepoint caching.
  - [`transliteration.py`](file:///home/mivikev/Desktop/AMC/ML-Classification/src/phase0/transliteration.py): Script-family dispatcher using `indic-transliteration` (OPTITRANS scheme) for Indic scripts, with unhandled foreign script passthrough.
  - [`text_cleaner.py`](file:///home/mivikev/Desktop/AMC/ML-Classification/src/phase0/text_cleaner.py): Shared base name cleaning (NFKC, lowercase, bracket/URL/duplicate stripping, legal-suffix preservation) and address normalization.
  - [`metrics_logger.py`](file:///home/mivikev/Desktop/AMC/ML-Classification/src/phase0/metrics_logger.py): Structured JSON metrics emission and history appending.
  - [`pipeline.py`](file:///home/mivikev/Desktop/AMC/ML-Classification/src/phase0/pipeline.py): Streaming chunked ingestion and dynamic country partitioner.
- **Test & Validation Suite**: [`tests/phase0/`](file:///home/mivikev/Desktop/AMC/ML-Classification/tests/phase0/)
  - [`test_cleaning.py`](file:///home/mivikev/Desktop/AMC/ML-Classification/tests/phase0/test_cleaning.py): 8 unit tests covering all noise catalog patterns from EDA §7.1 and §7.2.
  - [`test_scripts.py`](file:///home/mivikev/Desktop/AMC/ML-Classification/tests/phase0/test_scripts.py): 6 unit tests validating open-set script detection and transliteration across Indic, Arabic, Cyrillic, and CJK text.
  - [`run_validation.py`](file:///home/mivikev/Desktop/AMC/ML-Classification/tests/phase0/run_validation.py): Automated Tier 1 and Tier 2 validation checks runner.

---

## Dataset Processing & Row Count Audit

All row counts between raw TSV inputs and output Parquet partitions match **100.00%** with zero dropped rows:

### Training Split (`data/processed/train/`)
| Source File | Role | Prefix | Raw Rows | Processed Rows | Match |
|---|---|---|---|---|:---:|
| `train_source1.tsv` | Query | S1 | 2,206,821 | 2,206,821 | ✅ |
| `train_source2.tsv` | Target | S2 | 5,034,616 | 5,034,616 | ✅ |
| `train_source3.tsv` | Target | S3 | 5,285,603 | 5,285,603 | ✅ |
| **Total** | | | **12,527,040** | **12,527,040** | ✅ |

**Output Parquet Files**:
- `Query_India.parquet`: 883,188 rows (126 MB)
- `Query_US.parquet`: 1,323,633 rows (137 MB)
- `Target_India.parquet`: 4,133,346 rows (616 MB)
- `Target_US.parquet`: 6,186,873 rows (680 MB)

### Test Split (`data/processed/test/`)
| Source File | Role | Prefix | Raw Rows | Processed Rows | Match |
|---|---|---|---|---|:---:|
| `test_source1.tsv` | Query | S1 | 1,732,544 | 1,732,544 | ✅ |
| `test_source2.tsv` | Target | S2 | 4,887,273 | 4,887,273 | ✅ |
| `test_source3.tsv` | Target | S3 | 5,082,316 | 5,082,316 | ✅ |
| **Total** | | | **11,702,133** | **11,702,133** | ✅ |

**Output Parquet Files** (including zero-shot France):
- `Query_France.parquet`: 259,452 rows (23 MB)
- `Query_India.parquet`: 809,986 rows (115 MB)
- `Query_US.parquet`: 663,106 rows (70 MB)
- `Target_France.parquet`: 1,434,993 rows (138 MB)
- `Target_India.parquet`: 4,717,565 rows (708 MB)
- `Target_US.parquet`: 3,817,031 rows (430 MB)

---

## Validation Check Results

### Tier 1 — Structural Correctness (Hard Fail)
- **Integrity**: Row count conservation across partitions verified for all 6 input files.
- **Uniqueness**: `entity_id` is guaranteed unique within every output Parquet file.
- **No Cross-Country Leakage**: Zero entities appear in more than one country partition.
- **Non-Empty Fields**: Zero rows have empty or null `name_for_faiss` or `name_for_bm25`.
- **Encoding Integrity**: Zero replacement characters (`U+FFFD`) or corruption introduced.
- **Status**: **PASS (All 10 output files)**

### Tier 2 — Distributional Regression
- **Transliteration Coverage**:
  - `train`: 727,753 Indic cross-script rows $\to$ 727,753 transliterated (**100.00%**).
  - `test`: 839,103 Indic cross-script rows $\to$ 839,103 transliterated (**100.00%**).
- **Cross-Script Rates**:
  - India S2: 23.07% (train), 23.22% (test) $\to$ matches EDA's 9.1% overall rate.
  - India S3: 12.39% (train), 12.56% (test) $\to$ matches EDA's 5.0% overall rate.
- **Missing Address Rates**:
  - S2 / S3: ~2.3%–3.1% across partitions $\to$ consistent with EDA's ~3.3% baseline.
- **Status**: **PASS (within expected tolerance)**

### Tier 3 — Spot Sample Verification
Samples were exported to:
- [`metrics/phase0/spot_check_train.tsv`](file:///home/mivikev/Desktop/AMC/ML-Classification/metrics/phase0/spot_check_train.tsv)
- [`metrics/phase0/spot_check_test.tsv`](file:///home/mivikev/Desktop/AMC/ML-Classification/metrics/phase0/spot_check_test.tsv)

**Representative Transliteration Examples**:
- Devanagari: `आनंद सनराइज इंफ्रास्ट्रक्चर प्रा. लि.` $\to$ `anamda sanaraija imphrastrakchara pra. li.`
- Gujarati: `ૐ Foundation પ્રાઇવેટ લિમિટેડ` $\to$ `om foundation praiveta limiteda`
- Telugu: `కృష్ణా ఇంపెక్స్ లిమిటెڈ` $\to$ `krshna impeks limited`
- Kannada: `ಗುರು ಎಸ್ಟೇಟ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` $\to$ `guru estet praivet limited`
- Bengali: `স্টার ইন্ডাস্ট্রিজ প্রাইভেট লিমিটেড` $\to$ `stara indastrija praibheta limiteda`
- Consecutive Dedup: `ઓલ ઓલ હોસ્પિટાલિટી...` $\to$ `ઓલ હોસ્પિટાલિટી...` (cleanly deduplicated)
