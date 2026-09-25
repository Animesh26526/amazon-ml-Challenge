# Amazon ML Challenge 2026 — Business Entity Resolution

[![Unit Tests](https://github.com/Animesh26526/amazon-ml-Challenge/actions/workflows/tests.yml/badge.svg)](https://github.com/Animesh26526/amazon-ml-Challenge/actions/workflows/tests.yml)
[![Repository Sanity](https://github.com/Animesh26526/amazon-ml-Challenge/actions/workflows/repo-check.yml/badge.svg)](https://github.com/Animesh26526/amazon-ml-Challenge/actions/workflows/repo-check.yml)

## 1. Project Overview

This repository contains the engineering codebase for the **Amazon ML Challenge 2026: Business Entity Resolution** competition. 

The objective is to resolve business entities across three independent, noisy data sources:
- **Source 1 ($S_1$)**: Deduplicated reference entity source.
- **Source 2 ($S_2$)**: Noisy business records with missing fields and format variations.
- **Source 3 ($S_3$)**: Noisy business records with abbreviations, typos, and format variations.

For each Source 1 reference record, our system determines the complete set of corresponding $S_2$ and $S_3$ records (an $S_1$ entity may have zero, one, or multiple matches).

The official evaluation metric is **macro-averaged $F_{0.5}$** computed across all $S_1$ entities:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Because $F_{0.5}$ weights precision twice as heavily as recall, false merges severely penalize performance, and correctly predicting an empty match set for a true singleton receives a full $1.0$ score.

---

## 2. Authoritative Project Materials

- **[PRD.md](file:///C:/Users/Lenovo/Desktop/Projects/Amazon%20ML%20Challenge/amazon-ml-Challenge/PRD.md)**: The authoritative engineering specification at the repository root. All design decisions, experiment ladders, and architecture principles originate here.
- **[challenge/problem_statement.md](file:///C:/Users/Lenovo/Desktop/Projects/Amazon%20ML%20Challenge/amazon-ml-Challenge/challenge/problem_statement.md)**: Official organizer problem statement, competition rules, and evaluation format.
- **[challenge/validate_submission.py](file:///C:/Users/Lenovo/Desktop/Projects/Amazon%20ML%20Challenge/amazon-ml-Challenge/challenge/validate_submission.py)**: Official organizer-provided submission validator (preserved unchanged).
- **[docs/architecture.md](file:///C:/Users/Lenovo/Desktop/Projects/Amazon%20ML%20Challenge/amazon-ml-Challenge/docs/architecture.md)**: Engineering architecture, invariant definitions, and data flow.

---

## 3. Repository Structure

```
amazon-ml-challenge-2026/
├── PRD.md                       # Authoritative engineering specification
├── challenge/
│   ├── problem_statement.md     # Organizer problem statement and rules
│   └── validate_submission.py   # Official submission validator
├── configs/
│   └── default.yaml             # Central pipeline configuration (paths, hyperparameters)
├── data/
│   ├── raw/                     # Competition datasets (train & test, gitignored)
│   └── samples/                 # Small synthetic test datasets
├── src/
│   ├── io.py                    # Memory-safe loading, writers, schema validation
│   ├── normalization.py         # Multi-view text normalization and anchor extraction
│   ├── blocking/                # Multi-pass candidate generation routes
│   │   ├── exact.py             # Exact name, address, and name+address blocking
│   │   ├── char_ngram.py        # Character n-gram index retrieval
│   │   ├── tfidf.py             # TF-IDF sparse cosine similarity retrieval
│   │   ├── phonetic.py          # Soundex phonetic encoding
│   │   ├── rare_tokens.py       # Inverted index on discriminative rare tokens
│   │   └── numeric.py           # Address numeric anchor blocking
│   ├── features/                # Pairwise similarity feature engineering
│   │   ├── name.py              # Name lexical, token, edit, n-gram similarities
│   │   ├── address.py           # Address similarities and explicit missing indicators
│   │   ├── interactions.py      # Cross-field interactions
│   │   └── retrieval.py         # Route provenance metadata features
│   ├── model/                   # Pairwise GBDT matcher (training, inference, calibration)
│   ├── graph/                   # S1-centered witness and cross-source evidence
│   ├── decision/                # Entity-level decision engine, singletons, ambiguity
│   └── evaluation/              # Macro F_0.5, candidate recall, and cohort diagnostics
├── scripts/                     # Modular CLI entrypoints for each pipeline stage
├── experiments/
│   └── experiment_log.csv       # Tracked experiment benchmark log
├── tests/                       # Fast unit test suite
└── .github/workflows/           # Automated CI workflows
```

---

## 4. Local-Only vs. Tracked Directories

| Directory / File | Status | Description |
|---|---|---|
| `data/raw/` | **Gitignored** | Full competition raw data (~2.9 GB). Never committed. |
| `artifacts/` | **Gitignored** | Checkpoints, indexes, intermediate feature matrices. |
| `output/*.tsv` | **Gitignored** | Generated submission outputs (`matching_results.tsv`, `candidate_pairs.tsv`). |
| `experiments/experiment_log.csv` | **Tracked** | Version-controlled experiment log with metrics and commit SHAs. |
| `data/samples/` | **Tracked** | Small synthetic fixtures for unit tests. |

---

## 5. Environment Setup

The codebase requires **Python 3.10+**.

Install core dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Core dependencies:
- `numpy`, `pandas`, `scipy`
- `scikit-learn`, `xgboost`
- `pyyaml`, `tqdm`
- `pytest`

---

## 6. Running Tests & Sanity Checks

Run the automated test suite:
```bash
python -m pytest -v tests/
```

Verify compilation and module imports:
```bash
python -m compileall src scripts tests challenge
python -c "import src; import src.io; import src.normalization; import src.blocking; import src.features; import src.model; import src.graph; import src.decision; import src.evaluation; print('OK')"
```

---

## 7. Submission Invariants & Validator

The pipeline outputs two required TSV files:
1. `output/matching_results.tsv`: Final entity matches per $S_1$ entity.
2. `output/candidate_pairs.tsv`: Final candidate pairs fed to the matching model.

### Key Invariant:
$$\text{predicted\_matches}(S_1) \subseteq \text{candidate\_pairs}(S_1) \quad \forall S_1 \in \text{Test}$$

Before any submission, run the organizer-provided validator:
```bash
python challenge/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir data/raw/test
```

---

## 8. Experiment Tracking

Every major change is logged in `experiments/experiment_log.csv` using the schema:
```
experiment_id,date,git_commit,hypothesis,change,configuration,candidate_recall,candidate_count,precision,recall,macro_f0_5,singleton_accuracy,runtime,status,notes
```
Metrics are recorded only when measured experimentally on held-out validation data.