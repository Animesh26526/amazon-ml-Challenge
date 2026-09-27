# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** KyuNahiHoRahiCoding  
**Team Members:** Animesh Bhavsar, Krish Modi, Yash Ganatra, Supreeti Pal  
**Submission Date:** September 2026  

---

## 1. Executive Summary
Our approach, **HYBRID-V7-99-FAST**, is an industrial-grade, memory-optimized entity resolution pipeline. We engineered a highly scalable 11-route lexical and GPU-accelerated semantic blocking engine (using FAISS), coupled with a 39-dimensional feature extraction layer and a two-stage LightGBM classifier with aggressive hard-negative mining. This architecture specifically optimizes for the S1-level Macro F0.5 metric by introducing strict target-hub penalization and missing-address relief mechanisms.

---

## 2. Methodology

### 2.1 Problem Analysis
During our EDA, we discovered several critical noise vectors driving false negatives and false positives:
- **Open-Set Country Distribution (The "France" Anomaly):** The train set contains only US and India data, while the test set introduces 259k entities from France. Standard models overfit to US/India address topologies. 
- **Missing Geometries:** Approximately 3.3% of S2 and S3 records lack address data completely.
- **Vocabulary Disjoint:** Pure string matching fails when conceptually identical businesses share zero tokens (e.g., "Alphabet" vs "Google").
- **The "Hub" Trap:** Generic entities (like "Starbucks" or "Bank") pull in thousands of false candidates, bloating the search space and destroying precision if not strictly bounded.

### 2.2 Solution Strategy
**Approach Type:** Hybrid Multi-Modal (Lexical Inverted Indexes + Semantic Vector Clustering + Pairwise GBDT)  
**Core Innovation:** **Unique-Name Semantic Clustering & Hard-Negative Mining.** We bypass the scale limitations of billion-row cross-joins by deduplicating 5 million raw targets into unique normalized strings, embedding them via `all-MiniLM-L6-v2`, and grouping them using a `FAISS IndexIVFFlat` index. The pipeline then surgically mines its own false positives during training (Hard-Negative Mining) to recalibrate the LightGBM thresholds, explicitly protecting the precision-heavy F0.5 metric.

---

## 3. Candidate Generation (Blocking)
*To adhere strictly to the challenge guidelines, our blocking algorithm is mathematically bounded to favor precision and minimize candidate pool bloat. We achieved O(1) lexical lookup speeds using pure Inverted Indices.*

- **Blocking keys used:** 11 active routes, including:
  1. *Lexical Routes:* Exact name, Sorted Name, Exact Address, Numeric Anchor (First token + Address Num), Char Prefix (4-char + Address Num).
  2. *Suffix-Stripped Routes:* Core Name (with explicit French/US/India legal suffix removal).
  3. *Frequency-Capped Rare Tokens:* TF-IDF style token indexing strictly bounded by Document Frequency (`1 <= DF <= 100`) to completely eliminate generic hubs.
  4. *Semantic GPU Route:* Sentence-Transformer embeddings via FAISS nearest neighbor (Top-5 bounded).
- **Candidate pairs generated:** Strictly capped at `MAX_CANDIDATES_PER_S1 = 80`. Average candidates generated is tightly controlled at ~13-35 per entity.
- **How you ensured true matches were not lost:** The combination of strict frequency caps on text, combined with the semantic FAISS pass, ensures we capture deep conceptual matches without flooding the pool with string-collision noise.

---

## 4. Matching Model

**Features used (39 Dimensions):**
- **Name features:** Exact match, Token Jaccard, Token Overlap, Fuzzy Edit Ratio, Token Sort Ratio, Partial Ratio, Length Difference, Prefix Match.
- **Address features:** Exact match, Token Jaccard, Fuzzy Edit, Numeric Extraction Overlap, and explicit Missing Target/Source flags.
- **Interaction features:** Cross-field similarities (e.g., Name High + Address Low), product similarities.
- **Retrieval & Target Context:** Route count provenance, Same Country flag, Legal Form Match, and crucially, `target_name_freq_log` (mathematically teaching the model to mistrust generic names).

**Model type:** Two-Stage `LightGBM` (GBDT). Stage 1 trains normally; Stage 2 resamples false positives with $p \ge 0.28$ as hard negatives to enforce strict precision boundaries.  
**Threshold selection method:** Grid Search optimization against S1-Level Macro F0.5 on a 20% validation split. The decision engine uses dual thresholds: `base_threshold = 0.50` for standard matching, and `singleton_threshold = 0.28` to rigorously protect zero-match entities.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9087+** (Validation holdout)
- **Precision:** 0.9571
- **Recall:** 0.8162
- **Common false positives (wrong merges):** Franchise disambiguation. When businesses have identical generic names (e.g., "McDonalds") and missing addresses, the model struggles to confidently distinguish between branches without geographic context.
- **Common false negatives (missed matches):** Extreme abbreviation combined with heavy misspellings that push the entity outside the FAISS semantic cluster and the 3-char prefix block.

---

## 6. Conclusion
We successfully designed an enterprise-grade entity resolution pipeline that processes millions of rows without succumbing to exponential scaling traps. By combining strict lexical frequency caps, FAISS-accelerated semantic clustering, and precision-optimized GBDT thresholding, we delivered a system that cleanly handles missing data, international unobserved variables (France), and singleton anomalies. 

---

## Appendix

### A. Code Artefacts
Our pipeline is completely unified within the submission zip.
- **`hybrid_v7_99_fast.py`**: The complete production python script handling data processing, blocking, FAISS indexing, LightGBM training, and streaming output generation. 
- **`hybrid_v7_99_fast.ipynb`**: Colab-ready notebook for direct execution in a GPU runtime.
- **Entry Points:** 
  - `!python hybrid_v7_99_fast.py --mode train` (Generates the model and recalibrates thresholds).
  - `!python hybrid_v7_99_fast.py --mode test` (Streams the test data and outputs `matching_results.tsv` and `candidate_pairs.tsv`).
- **Dependencies:** `lightgbm`, `rapidfuzz`, `sentence-transformers`, `faiss-gpu`, `scikit-learn`, `pandas`.
