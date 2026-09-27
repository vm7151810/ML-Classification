# Data Audit & EDA — Business Entity Resolution
> Scope: **Sub-problem 1 — Candidate Pair Generation (Blocking)**
> Status: All training files available; all EDA probes complete.

---

## 1. Dataset

### Scale

| File | Rows | Size |
|---|---|---|
| `train_source1.tsv` | **2,206,821** | 210 MB |
| `train_source2.tsv` | **5,034,616** | 489 MB |
| `train_source3.tsv` | **5,285,603** | 504 MB |
| `train_ground_truth.tsv` | 2,206,821 (1-to-1 with S1) | 127 MB |
| `test_source1.tsv` | 1,732,544 | 175 MB |
| `test_source2.tsv` | 4,887,273 | 509 MB |
| `test_source3.tsv` | 5,082,316 | 506 MB |

**Total candidate index (train):** ~10.3M records (S2 + S3).
**GT integrity:** Perfect 1-to-1 mapping between S1 and ground truth — zero orphans.

### Schema

All files: tab-separated, 4 columns — `entity_id` (prefixed `S1-`/`S2-`/`S3-`), `business_name`, `business_address`, `country`.

### Country Distribution

| Source | US | India | France |
|---|---|---|---|
| train (all sources) | ~60% | ~40% | 0% |
| test (all sources) | ~38% | ~47% | **~15%** |

> **France is zero-shot** — appears only in test (~730K–1.4M records/source). No training GT for France.

---

## 2. Data Quality

### Missing Addresses

| Source | Missing `business_address` |
|---|---|
| train_source1 | **0** (0.00%) — cleanest source |
| train_source2 | **168,967** (3.36%) |
| train_source3 | **175,916** (3.33%) |

`NULL`/`N/A` literals also appear in non-empty address cells (~2.5% of S2/S3). All `business_name` fields populated across all sources.

### Impact on Ground Truth (full scan of 7,638,365 GT pairs)

| Metric | Count | % |
|---|---|---|
| Matched candidates with **empty address** | **337,018** | **4.41%** |
| S1 entities with ≥1 match having empty address | **312,600** | ~15% of matched S1 |
| S1 entities where **all** matches have empty address | **6,225** | ~0.3% |

> **CRITICAL: Address cannot be a required blocking condition.** Making address overlap mandatory would lose 337K true positive matches (4.41% recall) at blocking stage. Address must be additive: present → boost; absent → do not penalise.

**Sample GT pairs with empty candidate address:**
```
S1: "Maure Williams Colombier Inc"  | 85 Wayne Ave, Ticonderoga, NY
 ↔ S2: "Maure Wilblims Colombier Inc"       | address: ''   ← typo variant, no address
 ↔ S3: "Maure Williams Inc Center"          | address: ''   ← reordered, no address

S1: "Orellana Investments LLC"      | 728 A Quail Ave, Geneva, IA
 ↔ S2: "Orellana Investments Investments Llc" | address: ''  ← duplication, no address
```

### Field Length (chars)

| | Name P25/Median/P75 | Address P25/Median/P75 |
|---|---|---|
| Source1 | 18 / 24 / 30 | 33 / 41 / 70 |
| Source2 | 19 / 25 / 31 | 31 / 37 / 63 |
| Source3 | 18 / 25 / 31 | 35 / 42 / 55 |

---

## 3. Noise Patterns

### Business Name Noise

| Pattern | Source1 | Source2 | Source3 |
|---|---|---|---|
| `[brackets]` / `(parens)` | 2.1% | 7.8% | 8.0% |
| Leading punct (`--`, `*`, `<<`) | 0.1% | 6.1% | 4.4% |
| URL in name (`.com`, `www.`) | 0.0% | 4.0% | 4.0% |
| Repeated word | 0.1% | 2.0% | 2.0% |

Source1 is the clean reference; S2 and S3 share nearly identical noise profiles.

### Address Noise

| Pattern | Source1 | Source2 | Source3 |
|---|---|---|---|
| `NULL` literal | 0.0% | 2.5% | 2.5% |
| `N/A` literal | 0.0% | 0.8% | 0.8% |
| State-first inversion | 3.8% | 3.8% | 1.5% |
| `#`/`##` house prefix | 0.8% | 7.7% | 10.3% |

**S3-specific:** Uses full US state names (`"Texas"`, `"North Carolina"`) while S1/S2 use 2-letter codes (`"TX"`, `"NC"`). Only **7.1%** of S3 US records have abbreviations.

### Multi-Script Distribution (sample of 300K rows)

| Script | Source1 | Source2 | Source3 |
|---|---|---|---|
| Latin | **100.0%** | 90.9% | 95.0% |
| Devanagari (Hindi) | — | 5.2% | 2.8% |
| Telugu / Kannada / Tamil | — | 0.6–0.8% each | 0.3–0.4% each |
| Bengali / Gujarati | — | 0.6% each | 0.3% each |

---

## 4. Ground Truth Analysis

### Match Cardinality

| # Matches | Count | % |
|---|---|---|
| **0 — Singleton** | **123,247** | **5.58%** |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6+ | 252,292 | 11.43% |

94.4% have ≥1 match; mean = **3.67** matches/entity; max = 11.

### Cross-Source Split

| | Count | % |
|---|---|---|
| Both S2 + S3 | **1,776,047** | **80.5%** |
| S3-only | 164,498 | 7.5% |
| S2-only | 143,029 | 6.5% |
| Singleton | 123,247 | 5.6% |

Mean: ~1.8 S2 + ~1.9 S3 matches per matched S1 entity.

---

## 5. Blocking Scale

**~22.8 Trillion** naive pairs (S1 × S2 + S1 × S3). **Goal:** >99.99% reduction recalling all ~7.64M true matches.

---

## 6. EDA Probes

### 6.1 Country Partition — Lossless ✅

Scanned all 7,638,365 GT pairs: **zero cross-country matches** (US→US: 4,578,522; India→India: 3,059,843).
Country = perfectly safe hard partition (~2.5× reduction at zero recall cost).

---

### 6.2 Name Similarity (50K GT pairs)

| Metric | P25 | P50 | P75 | Mean | **Zero-overlap** |
|---|---|---|---|---|---|
| Token Jaccard (legal stripped) | 0.333 | 0.750 | 1.000 | 0.669 | **14.9%** |
| Char Trigram Jaccard | 0.467 | 0.667 | 0.828 | 0.623 | **8.2%** |

- **52%** TokJacc > 0.75 — trivially retrievable
- **14.9%** zero token overlap — entirely missed by TF-IDF/MinHash
- Trigram recovers ~6.7pp → **8.2%** remain (cross-script India)
- **Recall ceilings:** TF-IDF ~85%; +char trigram ~7%; remaining ~8% needs embedding

---

### 6.3 Cross-Script Analysis (2,000 GT pairs)

Latin S1 ↔ Devanagari/Tamil/Telugu/Kannada S2/S3: **97.2% zero-overlap** under all 3 normalisation strategies (raw, NFKD→ASCII, Latin-only).

```
"Ram Maa Logistics Private Limited"  ↔  "राम मां लॉजिस्टिक्स प्राइवेट लिमिटेड"  TokJacc=0.000
"Raj Investments LLP"                ↔  "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"        TokJacc=0.000
```

> A multilingual bi-encoder is **mandatory** for ~10% of India S2 and ~5% of S3 records.

---

### 6.4 Address Stability (50K GT pairs)

| Metric | P25 | P50 | Mean | Zero-overlap |
|---|---|---|---|---|
| Address alpha-token Jaccard | 0.429 | 0.625 | 0.635 | **5.0%** |

- **95%** of matched pairs share ≥1 address alpha-token — strong secondary signal
- **26.9%** share a 4–6 digit numeric code (PIN/ZIP)
- Cannot be primary key: 3.3% records missing + 4.41% of GT matches have empty candidate address

---

### 6.5 France Zero-Shot Probe (100 S1 records, 1.43M candidates)

Top-1 TokJacc = 1.0 for **100/100** records. NFKD accent folding + case folding sufficient. Risk: generic names create false positives — address re-ranking required.
```
"ZNB Club SARL"       → "znb club sarl"          TokJ=1.0 ✓
"Thermal & Fils SASU" → "Thermal & Fils SARL"    TokJ=1.0 (suffix swap)
"Association de Pena" → "Association De Pension"  TokJ=0.0, TriJ=0.78 ← FALSE POSITIVE
```

---

### 6.6 Singleton Structural Profiling

123,247 singletons vs 2,083,574 matched entities — **delta = 0** on every axis (name length, address length, token count, noise rates, script, country).

> Pre-filtering singletons is not feasible. Handle at match-score threshold, not at blocking stage.

---

### 6.7 Legal Suffix Vocabulary (IDF-Driven, N=11.8M)

| Token | Doc% | IDF | Category |
|---|---|---|---|
| `limited` | 14.9% | 1.91 | India legal |
| `private` | 13.8% | 1.98 | India legal |
| `llc` | 12.3% | 2.10 | US legal |
| `inc` | 9.2% | 2.39 | US legal |
| `ltd` | 8.2% | 2.50 | Both |
| `pvt` | 4.5% | 3.10 | India legal |
| `center`/`partners`/`group` | 3–4% | 3.2–3.6 | Generic |

India honorifics (`sri`, `shri`, `dr`, `mr`, `smt`) each ~1.3% — treat as stopwords for India MinHash keys.

**Architecture by retrieval method:**
- **TF-IDF**: IDF auto-handles legal suffixes — no hard list needed
- **MinHash / char-n-gram keys**: strip tokens with IDF < 3.5
- **French bootstrap**: `{sarl, sas, sasu, eurl, sci, sa, snc}` — IDF > 12 in training data (unseen), must hard-code
- **BGE-M3 sparse mode**: learns weights end-to-end — no stopword list needed

---

### 6.8 Address Component Extractability

| Component | S1 | S2 | S3 |
|---|---|---|---|
| US — ZIP (5-digit) | 10.8% | 10.5% | 10.5% |
| US — 2-letter state | **100.0%** | **100.0%** | **7.1%** |
| US — city (comma-parse) | ~100% | ~100% | ~100% |
| India — PIN (6-digit) | **0.0%** | **0.0%** | **0.0%** |
| India — city (comma-parse) | ~100% | 99.5% | 99.6% |

City token = only reliable geo-key (~100% coverage, all sources).

**City extractor V2** (walk backwards, skip known state names):

| | Recall (cross-script) | Bucket mean |
|---|---|---|
| V1 (last segment → state) | 36.2% | 124,980 |
| **V2 (state-skip → city)** | **52.0%** | **41,746** |

Residual issues: inverted S3 format (~10%); <1% candidate addresses in local script.

---

### 6.9 TF-IDF False Positive Analysis (k=500, US partition)

Index: 6.19M US S2+S3. 500 S1 US records with matches.

| Metric | Value |
|---|---|
| **Recall@500 (mean)** | **77.5%** |
| Perfect recall (1.0) | 50.0% |
| Zero recall (0.0) | **6.8%** |
| **Precision@500** | **1.05%** (~5.3 true matches / 500 candidates) |
| **Reduction ratio** | **99.9922%** |

Bimodal recall: 50% perfect, 6.8% zero (cross-script + severely garbled names). Matching model receives ~500 candidates/S1 with ~5 true positives — well-posed binary classification.

---

### 6.10 Bi-Encoder Recall on Cross-Script Pairs

**Model:** `paraphrase-multilingual-MiniLM-L12-v2` (118M params, CPU)
**Setup:** 500 cross-script GT pairs ranked against 5,050 non-Latin Indian candidates.

| k | MiniLM Recall | TF-IDF |
|---|---|---|
| 10 | 12.0% | 0% |
| 50 | 23.0% | 0% |
| **100** | **28.8%** | **0%** |

True pair cosine mean = **0.405** vs random = **0.263** (+54% gap — real signal, poor rank discrimination).

**Root causes for low recall:**
1. **Dravidian blind spot**: Tamil pair cosine = 0.112 (rank 2558); Kannada pair cosine = 0.268. MiniLM undertrained on Dravidian scripts.
2. **Generic name ambiguity**: "New Solutions" cosine = 0.441 with true match, but 315 similar "solutions" businesses score nearly identically.

**Address rescue:** 99.4% of non-Latin-name candidates have Latin addresses. V2 city-token blocking achieves **52% recall** on cross-script pairs independently.

**Combined recall (union):**
```
MiniLM@k=100 ∪ V2 city-token: 52% + (28.8% × 48%) ≈ 66%
BGE-M3       ∪ V2 city-token: ~70–80% (estimated)
```

**BGE-M3 (BAAI, 568M params) — recommended production model:**
- **Dense**: stronger Dravidian alignment
- **Sparse**: learned token weights → replaces TF-IDF channel (no stopword list needed)
- **ColBERT**: token-level matching → addresses generic-name disambiguation
- Single model covers Channels A + C in the architecture below

---

## 7. Final Architecture

```
INPUT: S1 record (country, name, address)
  │
  ▼
[HARD PARTITION by country]     ← 100% lossless; ~2.5× reduction
  US | India | France (zero-shot)
  │
  ├─ A: TF-IDF / MinHash LSH
  │     Ceiling: ~85% | Use for: Latin names (US, France)
  │     Strip (MinHash keys): IDF < 3.5 tokens + India honorifics
  │     France: add bootstrap {sarl, sas, sasu, eurl, sci, sa, snc}
  │
  ├─ B: Char Trigram / n-gram cosine  (+~7% recall)
  │     Handles: typos, abbreviations, partial token overlap
  │
  ├─ C: Multilingual Bi-Encoder — BGE-M3 (preferred) / LaBSE
  │     +~8–15% recall | Mandatory for: cross-script India
  │     BGE-M3 sparse mode also replaces channel A
  │
  └─ D: V2 City-token blocking (soft, secondary)
        52% recall on cross-script pairs | NEVER a hard gate
        (4.41% of GT matches have empty candidate address)

UNION A + B + C + D → ~500 candidates/S1
  └─ ML matching model: classify ~5 true positives out of 500
```

---

## 8. Key Decisions

| Decision | Evidence | Status |
|---|---|---|
| Country = hard partition | 100.0% same-country in 7.64M GT pairs | ✅ Confirmed |
| Address = additive only, never hard gate | 337K GT matches have empty candidate address (4.41%) | ✅ Confirmed |
| Bi-encoder mandatory | 97.2% zero token overlap on cross-script pairs | ✅ Confirmed |
| France = zero-shot token blocking safe | 100/100 French records TokJacc=1.0 | ✅ Confirmed |
| Singleton pre-filter = not viable | Zero structural difference from matched entities | ✅ Confirmed |
| IDF auto-handles legal suffixes in TF-IDF | `llc` IDF=1.64, `limited` IDF=0.90 | ✅ Confirmed |
| City token = best geo-key | ZIP 10.5%; India PIN 0%; S3 state abbrev 7.1% | ✅ Confirmed |
| BGE-M3 preferred over MiniLM | Dravidian coverage; ColBERT; sparse mode | ✅ Recommended |
| Union of channels | No single channel >85% recall; cross-script needs embedding | ✅ Confirmed |

---

## 9. Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| Cross-script names (10% India S2): zero token recall | **CRITICAL** | Mandatory BGE-M3 bi-encoder |
| MiniLM Dravidian blind spot (Tamil/Kannada cosine ≈0.1–0.27) | **HIGH** | Use BGE-M3 (Dravidian-trained) |
| Generic name ambiguity (rank 300+ at cosine 0.44) | **HIGH** | BGE-M3 ColBERT; V2 city-token as intersection signal |
| Missing addresses (4.41% of GT matches have empty candidate addr) | **MEDIUM** | Name-only fallback; no hard address gate |
| French generic name false positives | **MEDIUM** | Address re-ranking; ≥2 content tokens required |
| S3 inverted address format (~10% state-first) | **MEDIUM** | V2 extractor partially handles; residual ~10% |
| Singletons (5.6%) falsely matched | **MEDIUM** | Match-score threshold; F₀.₅ penalises FP 2× |
| Address component reordering | **LOW** | Token Jaccard is order-invariant |
| Repeated word duplication in names (2% S2/S3) | **LOW** | Dedup tokens before hashing |
