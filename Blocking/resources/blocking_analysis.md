# Deep Dive: Phase 2 - Blocking (Candidate Generation)

In Entity Resolution (ER), **Blocking** (or Candidate Generation) is widely considered the most critical phase for scaling. If you have 2.2 million entities in Source 1 and 1 million entities in Source 2, comparing them all yields **2.2 Trillion pairs**. 

Blocking safely ignores the 99.99% of pairs that are obviously not matches, allowing the heavy Machine Learning models (in Phase 3 & 4) to focus only on highly plausible candidates.

---

## 1. The Core Metrics of Blocking

A blocking algorithm is evaluated on two competing metrics:
1.  **Reduction Ratio (Efficiency):** How much did you reduce the search space? (e.g., going from 2.2 Trillion pairs down to 22 Million is a 99.99% reduction ratio).
2.  **Recall Ceiling (Effectiveness):** Did you accidentally block a true match? If a true match is never generated as a candidate in Phase 2, the ML model in Phase 4 will never see it. **Blocking sets the absolute ceiling for your final Recall.**

---

## 2. Best Industrial Standards (The Workhorses)

In production environments (like Amazon, Meta, or Stripe), blocking needs to be blazingly fast. Industry heavily relies on inverted indices and hashing rather than deep learning for this specific phase.

### A. TF-IDF + Cosine Similarity with FAISS
*   **How it works:** Concatenate the entity name and address into a single document. Vectorize it using TF-IDF. To capture typos, use **Character N-Grams** (e.g., 3-grams: `amazon` -> `ama, maz, azo, zon`) instead of word tokens. 
*   **Scale:** Use Facebook's **FAISS** (Facebook AI Similarity Search) to perform highly optimized Approximate Nearest Neighbor (ANN) searches, retrieving the Top-20 nearest vectors in milliseconds.
*   **Pros:** The industry gold standard for text with typos.

### B. MinHash LSH (Locality Sensitive Hashing)
*   **How it works:** Instead of comparing all documents, LSH hashes similar documents into the same "buckets" with high probability. 
*   **Pros:** Extremely fast and highly distributed (often used with Apache Spark).
*   **Cons:** Can be difficult to tune the number of permutations and bands to balance precision/recall.

### C. Standard Inverted Index (BM25)
*   **How it works:** Using search engines like Elasticsearch or highly optimized local implementations. BM25 is a non-linear TF-IDF variation that prevents extremely long addresses from skewing the similarity score.

---

## 3. State-of-the-Art / Novel Academic Approaches

Recent papers (2020–2024) have shifted away from purely lexical methods toward **Semantic and Learned Blocking**.

### A. DeepBlocker / AutoBlock (Embedding-Based Blocking)
*   **Concept:** Instead of sparse TF-IDF vectors, these papers use Deep Learning (Autoencoders or Bi-directional RNNs/Transformers) to learn a dense embedding of a data tuple. 
*   **Novelty:** It understands semantics. For example, it learns that "Corp" and "Corporation" are identical without needing a manual mapping dictionary.
*   **Execution:** The dense vectors are then fed into FAISS for fast retrieval.

### B. Contrastive Learning with Bi-Encoders
*   **Concept:** Adapted from modern NLP (like Sentence-BERT), a Bi-Encoder framework passes Source 1 records through a neural network, and Source 2 records through an identical network. The model is trained using a contrastive loss function (e.g., InfoNCE) to push true matches close together in vector space and pull non-matches apart.
*   **Drawback for ER:** Often struggles with numbers (e.g., "123 Main St" vs "124 Main St"). Lexical methods usually outperform semantic models on raw addresses.

### C. Meta-Blocking (Graph-Based Pruning)
*   **Concept:** Proposed extensively by the team behind *JedAI*. After an initial blocking phase (like TF-IDF), you create a massive graph where nodes are entities and edges are candidate pairs.
*   **Novelty:** Meta-blocking traverses this graph to prune redundant edges based on node degrees and local graph topology, significantly increasing the Reduction Ratio without sacrificing Recall.

---

## 4. Key Open-Source Repositories (State of the Art)

If you are looking to integrate existing frameworks rather than writing from scratch, the academic and open-source communities recommend:

1.  **[Awesome Entity Resolution](https://github.com/OlivierBinette/Awesome-Entity-Resolution):** The premier curated list of papers and ER frameworks.
2.  **[PyJedAI](https://github.com/scify/JedAIToolkit):** The Python implementation of the Java Entity Data Integration (JedAI) toolkit. It contains state-of-the-art token blocking, LSH, and Meta-Blocking algorithms. Highly recommended.
3.  **[SparkER](https://github.com/vinh-n/SparkER):** For massive, distributed datasets. Implements blocking and meta-blocking natively in Apache Spark.
4.  **[DeepMatcher](https://github.com/anhaidgroup/deepmatcher):** While more focused on the matching phase, its data processing pipeline is a standard for deep learning in ER.

---

## 5. Recommendation for Our specific ML Challenge

Given our constraints (No external APIs, 8B parameter max for models, heavily noisy addresses/names with abbreviations):

1.  **Do NOT use pure LLMs / NLP Embeddings for Blocking:** Dense embeddings notoriously fail at character-level typos and numerical matching (e.g., misreading a Zip Code).
2.  **The Optimal Strategy:** Build a **Hybrid Index**. 
    *   Create a **Character 3-Gram TF-IDF matrix** of the combined Name + Address string.
    *   Use **BM25 or FAISS** to retrieve the Top-20 matches.
    *   Apply a strict **Country hard-filter** (only retrieve records where `country` matches exactly). 

This approach will give us >99% Recall Ceiling while reducing the search space from trillions to just a few million pairs—perfectly setting up the XGBoost classifier in the next phase.
