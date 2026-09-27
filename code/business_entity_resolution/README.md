# Amazon ML Challenge 2026 — Business Entity Resolution (V0→V5 Pipeline)

An end-to-end, high-precision, scalable Business Entity Resolution system designed for the Amazon ML Challenge 2026.

---

## 1. Problem Overview

In large-scale commercial platforms, business records arrive from multiple independent, noisy sources without shared primary keys:
- **Source 1 ($S_1$):** Deduplicated reference source. For every $S_1$ entity, find all matching records from $S_2$ and $S_3$. Matches can be zero (singleton), one, or many.
- **Source 2 ($S_2$) and Source 3 ($S_3$):** Noisy external sources containing name abbreviations, legal suffix variations, missing address components, typos, transliteration, and landmark descriptions.
- **Evaluation Metric:** Per-entity Macro $F_{0.5}$ score:
  $$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$
  Singletons score $1.0$ if predicted empty, and $0.0$ if any match is predicted. Precision is weighted 2× over recall.

---

## 2. Integrated V0→V5 Architecture

The entire production pipeline is implemented in a single self-contained, high-performance module (`main.py`):

```
RAW SOURCES (S1, S2, S3)
       │
       ▼
1. MULTI-VIEW CONSERVATIVE NORMALIZATION
   ├── NFKD Unicode decomposition & whitespace compaction
   ├── Legal suffix canonicalization (corp, inc, pvt, ltd, sarl, sas, etc.)
   ├── Address abbreviation canonicalization (st, rd, ave, blvd, etc.)
   └── Numeric & PIN/postal code extraction
       │
       ▼
2. HIGH-RECALL MULTI-PASS CANDIDATE BLOCKING (with Provenance)
   ├── Route 1: Exact normalized name inverted index
   ├── Route 2: Word-order-sorted name inverted index
   ├── Route 3: Exact normalized address inverted index
   ├── Route 4: Joint (name, address) inverted index
   ├── Route 5: Numeric anchor (first name token + primary building number)
   ├── Route 6: Rare token inverted index (IDF frequency bounded)
   └── Route 7: Name character-prefix anchor (4-gram + building number)
       │
       ▼
3. PAIRWISE FEATURE ENGINE (33-Dimensional Feature Vector)
   ├── Name similarities: Exact match, Token Jaccard, Containment, Levenshtein ratio, Token Sort ratio
   ├── Address similarities: Exact match, Token Jaccard, Containment, Levenshtein ratio, Number Jaccard
   ├── Missingness flags: S1 missing, Target missing, Either missing
   ├── Cross-field interactions: Product, Min, Mean, Both-high, Asymmetric strength
   └── Retrieval provenance: Route indicators, active route count, source tags
       │
       ▼
4. TWO-STAGE GBDT MATCHER & HARD-NEGATIVE MINING
   ├── Stage 1: Base GBDT (LightGBM with XGBoost fallback) trained on candidate pairs
   ├── Stage 2: Hard-negative mining discovers high-confidence false merges
   └── Retraining on hard-negative augmented candidate sets
       │
       ▼
5. RECIPROCAL EVIDENCE & LOCAL S1-CENTERED GRAPH
   ├── Cross-source witness support (S1-S2 supported by S1-S3 candidate)
   └── Triangle agreement bonus
       │
       ▼
6. PRECISION-CONTROLLED ENTITY DECISION & SINGLETON POLICY
   ├── Top-candidate confidence thresholding (singleton detection)
   ├── Score-gap thresholding (bounding multi-match ambiguity)
   └── Validation Macro F0.5 grid-search threshold calibration
       │
       ▼
7. COMPETITION OUTPUTS & VALIDATION
   ├── output/candidate_pairs.tsv (Final candidate set fed to model)
   ├── output/matching_results.tsv (Strict subset of candidates)
   └── Automated check via official validate_submission.py
```

---

## 3. Directory Layout

```
amazon-ml-Challenge/
├── main.py                     # Complete integrated pipeline
├── config.yaml                 # Centralized configuration
├── requirements.txt            # Pinned dependencies
├── README.md                   # Documentation
├── data/
│   └── raw/
│       ├── train/              # train_source1/2/3.tsv, train_ground_truth.tsv
│       └── test/               # test_source1/2/3.tsv
├── artifacts/
│   ├── models/                 # Serialized GBDT matcher
│   └── reports/                # Forensics, validation, and diagnostics reports
├── output/
│   ├── matching_results.tsv    # Final match predictions
│   └── candidate_pairs.tsv     # Final candidate pairs
└── challenge/
    └── validate_submission.py  # Official submission validator
```

---

## 4. Installation

Python 3.10+ (tested on Python 3.14.x Windows & Linux/Colab):

```bash
pip install -r requirements.txt
```

Core dependencies: `pandas`, `numpy`, `scipy`, `scikit-learn`, `lightgbm`, `xgboost`, `rapidfuzz`, `joblib`, `pyyaml`.

---

## 5. Execution Commands

### A. Data Profiling & Forensics
Analyzes row counts, missingness, country distribution, and true match cardinality:
```bash
python main.py --mode profile
```

### B. Model Training & Validation Calibration
Builds multi-pass candidates, extracts features, trains base GBDT, mines hard negatives, retrains, and calibrates decision thresholds for optimal S1 Macro $F_{0.5}$:
```bash
python main.py --mode train
```

For quick smoke testing with a smaller S1 entity sample:
```bash
python main.py --mode train --sample-size 5000
```

### C. Validation Diagnostics
Evaluates the model against held-out validation entities and outputs cohort-level diagnostics:
```bash
python main.py --mode validate
```

### D. Streaming Test Inference
Runs country-partitioned streaming candidate generation, feature extraction, scoring, and decision writing for all test entities (including France, US, India). Automatically invokes `validate_submission.py`:
```bash
python main.py --mode test
```

### E. Full End-to-End Execution
Runs profile $\to$ train $\to$ validate $\to$ test in a single command:
```bash
python main.py --mode full
```

---

## 6. Official Competition Invariants & Compliance

1. **Submission Invariants Enforced:**
   - Every Source 1 entity in `test_source1.tsv` appears exactly once in `matching_results.tsv`.
   - `candidate_pairs.tsv` represents the exact set of candidates evaluated by the ML model.
   - `matched_entity_ids` $\subseteq$ `candidate_entity_ids` is mathematically guaranteed.
   - Singletons are represented as an empty string after the tab (`S1-xxxxx\t`).
   - No self-matches (no S1 IDs in match sets) and no duplicate IDs in any list.
2. **Open-Set Country Handling:**
   - France (`France`) is fully processed and preserved in test outputs alongside `US` and `India`.
3. **Fair Play & Model Constraints:**
   - Strictly 100% within the provided challenge datasets.
   - Zero external lookups, zero business registries, zero geocoding APIs.
   - Model is classical GBDT (LightGBM/XGBoost) with MIT / Apache 2.0 license and under 100K parameters (well within the 8B limit).