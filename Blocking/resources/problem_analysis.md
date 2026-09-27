# Amazon ML Challenge 2026: Business Entity Resolution Analysis

## 1. Problem Statement Overview
The core challenge is **Entity Resolution (ER)**, which is the process of determining whether multiple records from different, independent data sources refer to the exact same real-world entity (in this case, businesses). 

You are provided with data from 3 distinct sources:
*   **Source 1:** A clean, deduplicated reference source.
*   **Source 2 & Source 3:** Two external data sources containing noisy, fragmented, and inconsistent representations of businesses.

> [!NOTE]
> There are no shared keys or universal identifiers across the sources. The matching must be performed purely on the content of the records (Name, Address, Country).

## 2. Task Breakdown
Your objective is to find all matching business records from Source 2 and Source 3 for **every single entity** in Source 1.
*   A Source 1 entity may have **multiple matches** from Source 2 & 3.
*   A Source 1 entity may have **no matches** (singleton).
*   You must generate matches for the test set and output them in a specific format (`matching_results.tsv`). You also need to supply the candidate pairs generated during your blocking stage (`candidate_pairs.tsv`).

> [!IMPORTANT]
> **Strict Constraint**: You are **STRICTLY PROHIBITED** from using external APIs, lookups, or databases (like Google Maps API, Government registries, or commercial ER tools). Your model must learn purely from the provided training set.

## 3. Raw Data Available

All data files are **Tab-Separated (`.tsv`)**. 

### 3.1. Columns
*   `entity_id`: A unique identifier (Prefixed with `S1-`, `S2-`, or `S3-`).
*   `business_name`: The name of the business (contains abbreviations, DBAs, typos).
*   `business_address`: Address string (contains varying formats, missing pin codes, landmark references).
*   `country`: Country code.
    *   **Train Data**: Only `US` and `India`.
    *   **Test Data**: Includes a **new unseen country**, `France`. Your pipeline must dynamically handle new country patterns without hardcoded filters.

### 3.2. Data Volumes
Based on the file sizes in the `dataset/train/` and `dataset/test/` directories:
*   **Train Source 1**: ~2.2 Million records (210 MB)
*   **Train Source 2**: ~489 MB
*   **Train Source 3**: ~503 MB
*   **Train Ground Truth**: ~127 MB mapping file.
*   **Test Sets**: Roughly similar proportions (S1: 175 MB, S2: 509 MB, S3: 506 MB).

*(Comparing ~2M records from S1 against millions in S2 and S3 makes a naive NxM Cartesian product computationally impossible. An efficient Candidate Generation/Blocking strategy is mandatory.)*

## 4. Evaluation Metric
The challenge uses the **Macro-Averaged F_0.5 Score**. 
*   **Formula**: `F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`
*   **Why F_0.5?** This heavily penalizes **False Positives** (merging two different businesses) while being slightly forgiving on **False Negatives** (missing a match). Precision is weighted 2x over Recall. 
*   **Singletons Matter**: If a Source 1 entity has no true matches, correctly predicting an empty list gives you a perfect 1.0 for that entity. Guessing a false match gives you a 0.0.

## 5. Recommended Technical Approach (Moving Forward)

To tackle this challenge effectively, we should build a classic 3-stage Entity Resolution ML Pipeline:

### Phase 1: Data Preprocessing & Standardization
*   **Lowercasing & Cleaning**: Strip punctuation, special characters, and extra whitespaces.
*   **Normalization**: Standardize common abbreviations (e.g., "Corp" -> "Corporation", "St" -> "Street", "Rd" -> "Road").
*   **Tokenization**: Split names and addresses into N-grams (bi-grams, tri-grams) to prepare for robust text matching.

### Phase 2: Candidate Generation (Blocking)
Since a Cartesian product is impossible, we need to reduce the search space to a manageable subset of "highly probable" candidate pairs while keeping Recall as close to 100% as possible.
*   **Strategy 1 (Exact Match/Hashing)**: Group by exact matching normalized names and zip codes.
*   **Strategy 2 (TF-IDF + Cosine Similarity / MinHash LSH)**: Create TF-IDF vectors for `business_name` + `business_address`. Use fast Approximate Nearest Neighbor (ANN) search like FAISS or MinHash LSH to fetch the top-K closest Source 2/Source 3 neighbors for each Source 1 entity.
*   **Constraint**: Block strictly within the same `country`. 

### Phase 3: Feature Engineering
For every candidate pair (S1, S2) and (S1, S3) generated in Phase 2, compute pairwise similarity features:
*   **String Distances**: Levenshtein distance, Jaro-Winkler, and Jaccard similarity between names and addresses.
*   **Token Overlap**: Percentage of shared words.
*   **Numerical Extraction**: Extract and match street numbers or ZIP codes from the address string.
*   **Length Ratios**: Ratio of lengths between the two strings.

### Phase 4: Machine Learning Matching Classifier
*   **Model**: Train a gradient boosted tree model (like **XGBoost** or **LightGBM**) on the engineered features. The target label is `1` if the pair exists in the ground truth, and `0` otherwise.
*   **Loss Function / Thresholding**: Since the metric is F_0.5, we will need to tune the probability threshold to heavily favor Precision (e.g., only predicting a match if the model's confidence is > 0.75 or 0.85).
