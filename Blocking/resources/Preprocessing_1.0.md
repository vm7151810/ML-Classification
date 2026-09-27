# Phase 0: Preprocessing & Partitioning v2.0 (Polars Engine) (PATCHED v2.1)

> [!WARNING]
> **ARCHITECTURAL OVERRIDES APPLIED**
> The logic in this document has been bulletproofed. You MUST apply the following patches during implementation:
> 1. **Tokenizer Index Pollution Defense:** The `to_char_3grams` tokenizer (used in BM25 indices downstream) must explicitly intercept `nulladdr` or `nullname` imputation flags and immediately return `[]`. This prevents garbage 3-grams from polluting the BM25 IDF mathematics.

In Phase 0, our primary goal is to load the raw massive datasets (`train_source1.tsv`, etc.), handle multilingual text (Indic scripts) safely, aggressively clean and standardize the text into specialized representations for downstream FAISS and BM25 indices, and partition the datasets by country to prepare for the Phase 1 Blocking phase.

By utilizing **Polars**, we ensure that memory-intensive string operations execute natively in Rust across all available CPU threads, preventing OOM crashes.

---

## 1. High-Speed Data Loading & Imputation
*   **Engine:** `polars.read_csv(separator='\t')`
*   **Literal Null Parsing:** Map common fake-null string literals (`["n/a", "na", "null", "none", "-"]`) to `""`.
*   **Missing Values:** Immediately fill nulls with empty strings `""` to prevent `NoneType` errors.
*   **Data Types:** `country` will be cast to a Categorical type to save memory.

---

## 2. Multilingual Script Detection & Transliteration
Our EDA revealed that **7.10%** of all true match pairs in the ground truth are cross-script (Latin S1 ↔ Indic S2/S3). The previous `NFKD + ASCII encode` pipeline silently destroyed these.

*   **Script Detection:** We will use regex patterns (`[\u0900-\u097F]` for Devanagari, etc.) to detect Indic scripts (Devanagari, Bengali, Tamil, Telugu, Gujarati, Kannada, Malayalam, Gurmukhi).
*   **Transliteration:** If an Indic script is detected, we will use the `indic-transliteration` library to convert it to the **ITRANS** Roman scheme (phonetic Latin). This allows BM25 char 3-grams to partially overlap with the Latin S1 queries.

---

## 3. Dual Representation Text Normalization Pipeline
Since FAISS (MiniLM) and BM25 have fundamentally different requirements for text matching, we will generate **two independent representations** for `business_name`.

### A. Name for FAISS (`name_for_faiss`)
MiniLM is a multilingual model natively supporting 50+ languages. We must preserve the original script.
*   **Process:** Lowercase -> Trim leading/trailing whitespace -> Collapse multiple spaces into one.
*   **Crucial Rule:** NO NFKD/ASCII encoding, NO transliteration, NO punctuation stripping. (Preserves meaning for semantic embeddings).

### B. Name for BM25 (`name_for_bm25`)
BM25 relies on exact character 3-gram overlaps. It requires an aggressively sanitized Latin-only representation.
*   **Transliteration:** If Indic, transliterate to ITRANS.
*   **Unicode Normalization:** Safe `NFKD` encoding and ASCII coercion (`encode("ascii", "ignore").decode()`).
*   **Case & Space:** Lowercase and strip.
*   **Punctuation Stripping:** 
    *   Protect acronyms: Remove periods `.` with no space replacement (`I.B.M.` -> `IBM`).
    *   Map `@` -> ` at ` and `&` -> ` and `.
    *   Remove all remaining non-alphanumeric characters.
*   **Abbreviation Expansion:** Use `\b` anchored regex to expand `corp`->`corporation`, `inc`->`incorporated`, etc.

### C. Address Normalization (`addr_for_bm25`)
FAISS is not used on addresses. Addresses are highly structured, so we only need a BM25 representation.
*   **Token-Level Transliteration:** Split the address into words. Transliterate only the Indic tokens, keeping Latin tokens (like PIN codes or English city names) intact.
*   **Standardization:** Apply the same pipeline as `name_for_bm25` (ASCII coercion, lowercase, symbol mapping, punctuation stripping).
*   **Address Abbreviations:** Expand `st`->`street`, `rd`->`road`, etc., using `\b` anchors.

### D. Empty String Safety Fallback
*   If any resulting representation is an empty string `""`, we replace it with `f"nullname {entity_id}"` (or `nulladdr {entity_id}`). This creates a globally unique singleton string, avoiding artificial clustering in indices.

---

## 4. Country-Wise Partitioning & Target Union
An entity in India will never match an entity in the US.
*   **Country Normalization:** Clean the `country` column (lowercase, map variations like `usa` -> `us`).
*   **Unioning Targets:** Vertically concatenate `S2` and `S3` into a single target dataset. Add a `source` column (`"S2"` or `"S3"`) to trace origin.
*   **Partitioning:** Split the queries (S1) and targets (S2+S3) into strict country-specific DataFrames.

---

## 5. Detailed Output Expectations

The preprocessing script will output serialized files (e.g., `.parquet` for maximum I/O speed into Phase 1) partitioned by country and split into query vs. target sets. 

### Expected Output Files (per country chunk)
Assuming the dataset contains countries `us` and `india`:
1.  **`Query_us.parquet`** (Cleaned S1 entities for US)
2.  **`Target_us.parquet`** (Cleaned S2 + S3 entities for US)
3.  **`Query_india.parquet`** (Cleaned S1 entities for India)
4.  **`Target_india.parquet`** (Cleaned S2 + S3 entities for India)

### Row Structure / Columns Expected (Schema)

Each record in these output files will contain the exact following columns:

| Column Name | Data Type | Description & Expected Values |
| :--- | :--- | :--- |
| `entity_id` | String | Original ID (e.g., `"S1-925783039"`, `"S2-166376419"`). Primary key. |
| `country` | Categorical/String | Normalized country name (e.g., `"us"`, `"india"`). |
| `source` | String | Indicator of origin: `"S1"`, `"S2"`, or `"S3"`. |
| `name_original` | String | The raw, untouched `business_name` from the TSV. |
| `addr_original` | String | The raw, untouched `business_address` from the TSV. |
| `name_for_faiss` | String | Lowercased, original script preserved. Used directly by MiniLM. (e.g., `"एसएस फूड प्राइवेट लिमिटेड"`). |
| `name_for_bm25` | String | Latin-only, transliterated, fully normalized, no punctuation. Used for char 3-grams. (e.g., `"esas phuud praaiive t limite d"`). |
| `addr_for_bm25` | String | Latin-only, transliterated, expanded abbreviations. Used for char 3-grams. (e.g., `"af 0684 ghaziabad 9487203 uttar pradesh"`). |
| `is_cross_script` | Int8 (0 or 1) | `1` if the original name contained Indic scripts, `0` if purely Latin. Used as an ML feature downstream. |

### Characteristics of Entries
- **Missing Data:** No nulls. Missing names/addresses will be replaced by `nullname <entity_id>` or `nulladdr <entity_id>`.
- **Row Counts:** The sum of rows across all `Query_<country>.parquet` files will exactly equal the row count of `train_source1.tsv`. The sum of all `Target_<country>.parquet` files will exactly equal `train_source2.tsv` + `train_source3.tsv`.
- **Integrity:** Every `entity_id` is preserved. `country` casing is uniform. All string columns are guaranteed to be UTF-8 safe (for FAISS) or ASCII-safe (for BM25).
