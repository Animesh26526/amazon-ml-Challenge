# System Architecture — Business Entity Resolution Engine

## Overview

This repository implements an offline, high-precision Business Entity Resolution system designed for the Amazon ML Challenge 2026. The primary task is to identify all Source 2 (`S2-`) and Source 3 (`S3-`) business records corresponding to every Source 1 (`S1-`) reference entity.

The primary optimization metric is **macro-averaged $F_{0.5}$** across all $S1$ entities:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Because $F_{0.5}$ weights precision twice as heavily as recall, false merges (joining distinct businesses) are severely penalized, while correctly predicting an empty match set for a true singleton receives a full $1.0$ score.

---

## High-Level Pipeline Architecture

```
 Raw Data (S1, S2, S3)
          │
          ▼
 Multi-View Normalization
 (Raw + Normalized + Tokens + Char N-Grams + Numeric Anchors)
          │
          ▼
 Multi-Pass Candidate Generation / Blocking
 (Exact Name, Exact Address, Char N-Grams, TF-IDF, Rare Tokens, Phonetic, Numeric)
          │
          ▼
 candidate_pairs.tsv (Verified subset input to model)
          │
          ▼
 Pairwise Feature Engineering
 (Name Similarities, Address Similarities, Missingness Indicators, Interactions, Retrieval Provenance)
          │
          ▼
 Pairwise Matcher (Gradient Boosted Decision Trees: XGBoost / CatBoost)
          │
          ▼
 S1-Centered Local Cross-Source Witness / Graph Evidence
 (Direct S1-S2, Direct S1-S3, Witness S2-S3, Triangle Support, Contradiction Analysis)
          │
          ▼
 Entity-Level Decision Engine
 (Singleton / Empty Match, Ambiguity Detection, Score Margins, Cardinality Capping)
          │
          ▼
 matching_results.tsv (Final predicted matches)
```

---

## Core Invariants

1. **Subset Invariant**:
   $$\text{predicted\_matches}(S_1) \subseteq \text{candidate\_pairs}(S_1) \quad \forall S_1 \in \text{Test}$$
2. **Entity Coverage**: Every $S_1$ entity present in `test_source1.tsv` must appear exactly once in both output files.
3. **Prefix Discipline**: Matches and candidates may only reference entities prefixed with `S2-` or `S3-`. Self-matches to `S1-` are strictly prohibited.
4. **Data Isolation**: Test data is strictly inference-only. No labels, features, or thresholds may be fitted on the test partition.
5. **Conservative Normalization**: Raw attribute values are preserved alongside transformed views.
6. **Missingness vs. Contradiction**: Missing address information is not treated as a contradiction; it is encoded explicitly as missingness indicators.

---

## Directory Organization

```
amazon-ml-challenge-2026/
├── PRD.md                       # Authoritative engineering specification
├── challenge/
│   ├── problem_statement.md     # Competition rules, requirements, metric formula
│   └── validate_submission.py   # Official organizer-provided submission validator
├── configs/
│   └── default.yaml             # Central pipeline configuration
├── data/
│   ├── raw/                     # Raw competition datasets (gitignored)
│   └── samples/                 # Small synthetic fixtures for fast unit testing
├── src/
│   ├── io.py                    # Memory-safe loading, writers, schema validation
│   ├── normalization.py         # Multi-view text normalization and anchor extraction
│   ├── blocking/                # Multi-pass candidate generation routes
│   ├── features/                # Pairwise feature engineering
│   ├── model/                   # Model training, inference, and calibration
│   ├── graph/                   # S1-centered witness and cross-source evidence
│   ├── decision/                # Entity-level decision engine, singletons, ambiguity
│   └── evaluation/              # Macro F_0.5, candidate recall, and cohort diagnostics
├── scripts/                     # Modular CLI entrypoints for each pipeline stage
├── experiments/
│   └── experiment_log.csv       # Tracked experiment benchmark log
├── tests/                       # Automated fast unit tests
└── .github/workflows/           # Continuous integration workflows
```
