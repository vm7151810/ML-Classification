# Approach 2: Hybrid Entity Resolution Architecture

This document details the refined end-to-end Entity Resolution pipeline. By inverting the naive approach, this architecture strictly follows the industry standard for resolving millions of entities: **Retrieve (Block) -> Extract Features -> Score (ML).**

---

## High-Level Pipeline

1. **Preprocessing:** Standardize the text.
2. **Blocking (Candidate Generation):** Rapidly narrow down the trillions of possible combinations to a manageable subset (Top-K) using lexical search.
3. **Feature Engineering:** Calculate pairwise similarity metrics for the generated candidates.
4. **Machine Learning Classification:** Use a Gradient Boosted Tree to predict the probability of a match.
5. **Thresholding:** Optimize the classification boundary for the F_0.5 metric.

---

## Phase 1: Data Preprocessing & Standardization

Before any matching occurs, both the reference dataset (Source 1) and the noisy datasets (Source 2 & 3) must be aggressively standardized.

*   **Case & Whitespace:** Convert all strings to lowercase. Strip leading, trailing, and redundant internal whitespaces.
*   **Punctuation:** Remove all non-alphanumeric characters (e.g., `,`, `.`, `-`, `&` -> `and`).
*   **Normalization:** Expand common abbreviations using a predefined dictionary.
    *   *Examples:* `corp` -> `corporation`, `st` -> `street`, `pvt` -> `private`.
*   **Concatenation:** Create a single text blob for each record: `[normalized_business_name] + " " + [normalized_business_address]`.

---

## Phase 2: Blocking (Candidate Generation)

**Goal:** Avoid the Cartesian product (comparing 2.2M rows against millions of rows). 

Instead of passing everything into an ML model, we act like a search engine. We build an index of the Source 1 records and query it using Source 2 and 3 records.

*   **Mechanism:** TF-IDF Vectorization with **BM25** (Best Matching 25) or fast Cosine Similarity using character n-grams (e.g., 2-grams or 3-grams). Character n-grams are highly robust against typos and OCR errors.
*   **Execution:** 
    1. Build a fast retrieval index (like FAISS or a sparse TF-IDF matrix) on Source 1's concatenated text blob.
    2. For every record in Source 2 and Source 3, query the index.
    3. Retrieve the **Top-20 most similar Source 1 records**.
*   **Hard Constraint:** Only retrieve candidates that share the exact same `country`. 

> [!TIP]
> This phase outputs the exact data required for the `candidate_pairs.tsv` submission file.

---

## Phase 3: Feature Engineering

**Goal:** For every candidate pair `(Source 1 Record, Source 2/3 Record)` generated in Phase 2, extract a rich set of numerical features that describe *how* similar they are.

For a given pair, compute the following features:
1.  **Name Distances:** 
    *   Jaro-Winkler distance (excellent for short strings like names).
    *   Levenshtein (Edit) distance.
2.  **Address Distances:**
    *   Token overlap percentage (Jaccard similarity of words).
3.  **BM25 / TF-IDF Retrieval Score:** The raw similarity score obtained from Phase 2.
4.  **Numerical Extraction Match:** 
    *   Extract digits (often Zip Codes or Building Numbers) from both addresses. 
    *   Create a binary feature: `1` if the extracted numbers match exactly, `0` if they mismatch, `-1` if missing.
5.  **Length Differentials:** The absolute difference in character length between the two names.

---

## Phase 4: Machine Learning Classification

**Goal:** Use the numerical features to make a final, unified decision.

*   **Model:** **XGBoost** or **LightGBM** (Binary Classifier).
*   **Training Data Setup:**
    *   **Positive Samples (Label 1):** Candidate pairs that exist in `train_ground_truth.tsv`.
    *   **Negative Samples (Label 0):** Candidate pairs generated in Phase 2 that do *not* exist in the ground truth (these act as "hard negatives" which makes the model very robust).
*   **Inference:** The trained XGBoost model takes the feature vector of an unknown candidate pair (from the test set) and outputs a probability score between `0.0` and `1.0`.

> [!NOTE]
> Unlike the earlier proposal, there is no "Address Check Fallback" or "NLP System Fallback". The XGBoost model sees all features at once and learns internally how to balance a weak address match against a strong name match.

---

## Phase 5: Thresholding & Post-Processing

**Goal:** Maximize the F_0.5 metric, which favors Precision (avoiding False Positives) twice as much as Recall.

*   **Threshold Calibration:** We do not simply use `> 0.50` as the cutoff. Using a validation set, we scan probability thresholds (e.g., `0.5`, `0.6`, `...`, `0.95`) to find the exact threshold that maximizes the F_0.5 score.
*   **Typical Behavior:** For F_0.5, the optimal threshold is usually high (e.g., `> 0.80` or `> 0.90`).
*   **Final Output Formatting:** 
    *   If a candidate pair scores above the optimal threshold, it is considered a Match.
    *   Group the matches by `source1_entity_id` and format them as comma-separated lists to generate `matching_results.tsv`.
    *   If no candidates for a Source 1 entity pass the threshold, it is correctly left empty (identifying a singleton).
