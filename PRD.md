Product Requirements Document
Amazon ML Challenge 2026 — Business Entity Resolution Engine
Team: KyuNahiHoRahiCoding
Team Members: Animesh Bhavsar, Krish Modi, Yash Ganatra, Supreeti Pal
Competition: Amazon ML Challenge 2026
Problem: Business Entity Resolution
Primary Objective: Maximize macro-averaged F₀.₅ on the hidden test set while maintaining complete reproducibility and competition compliance.
---
1. Executive Summary
We will build a large-scale, offline Business Entity Resolution system that identifies all Source 2 (S2) and Source 3 (S3) records corresponding to every Source 1 (S1) reference entity.
The system will use a high-recall, multi-pass candidate-generation layer followed by precision-oriented pairwise matching and an entity-level decision layer.
The core architecture will be:
Data → Normalization → Multi-Pass Blocking → Candidate Set → Pairwise Features → Matching Model → Cross-Source Evidence → Entity-Level Decision → Final Matches
The system will be designed around the competition's most important characteristics:
- S1 entities can have zero, one, or many matches.
- Candidate generation determines the maximum achievable recall.
- F₀.₅ gives greater importance to precision than recall.
- False merges are especially costly.
- Correctly predicting an empty match set for a true singleton/no-match entity is valuable.
- Test data contains France even though training contains US and India.
- External entity lookup and data augmentation are prohibited.
The system will therefore favor high candidate recall followed by aggressive but evidence-based precision control, rather than forcing a match for every S1 entity.
---
2. Problem Definition
2.1 Input
Three independent sources are provided:
Source 1
S1 is the deduplicated reference source.
Each record contains:
- "entity_id"
- "business_name"
- "business_address"
- "country"
Source 2
S2 contains noisy business records.
Source 3
S3 contains noisy business records.
The sources do not share a common business identifier.
---
2.2 Required Output
For every S1 entity, determine the complete set of corresponding S2 and S3 entity IDs.
An S1 entity can have:
- zero matches
- one match
- multiple matches
The ground-truth representation contains one row per S1 entity with a potentially comma-separated list of matching S2/S3 IDs.
The final system must generate:
"matching_results.tsv"
The final predicted match set for every S1 entity.
"candidate_pairs.tsv"
The exact final set of candidate pairs that reaches the matching model.
This distinction is critical: "candidate_pairs.tsv" must represent the actual final candidate set, not an intermediate blocking stage.
---
3. Competition Constraints
The system must comply with all competition rules.
3.1 Data Usage
Only competition-provided data may be used.
The system must NOT use:
- external business databases
- business registries
- geocoding services
- external entity-resolution APIs
- external business lookup
- internet-based identity enrichment
- external datasets for entity augmentation
External lookup or augmentation can result in disqualification.
---
3.2 Country Handling
Training contains:
- US
- India
The test set additionally contains:
- France
Country must therefore be treated as an open-set attribute.
The system must not:
- hardcode "{US, India}"
- discard France
- assume every future country is known
- make country-specific assumptions that prevent unseen countries from being processed
Country may be used as a feature or partition only when empirically justified.
---
3.3 Model Constraints
Any final model must comply with the competition's model requirements:
- permitted license
- MIT or Apache 2.0 requirement
- maximum 8B parameters
The final submission must also contain sufficient code and methodology for reproduction and audit.
---
4. Evaluation Objective
4.1 Primary Metric
The primary optimization target is:
Macro-averaged F₀.₅ per S1 entity.
F₀.₅ is precision-heavy.
Therefore, a false merge is substantially more damaging than a missed match.
---
4.2 Singleton / No-Match Behavior
A particularly important population consists of S1 entities with no true S2/S3 matches.
For such an entity:
- predicting an empty set is correct
- predicting any incorrect match is catastrophic for that entity
The system must therefore explicitly support:
"No confident match"
as a legitimate final prediction.
The model must never force at least one candidate simply because candidates exist.
---
5. Known Dataset Profile
Current training data:
Dataset| Rows
S1| 2,206,821
S2| 5,034,616
S3| 5,285,603
Ground Truth| 2,206,821
5.1 Country Distribution
S1
Country| Rows
US| 1,323,633
India| 883,188
S2
Country| Rows
US| 3,016,817
India| 2,017,799
S3
Country| Rows
US| 3,170,056
India| 2,115,547
---
6. Data Quality Profile
S1 has no missing:
- business name
- business address
- country
S2 has approximately 3.36% missing addresses.
S3 has approximately 3.33% missing addresses.
Names and countries are complete.
Therefore, address-based matching must explicitly handle missing addresses rather than interpreting missingness as negative evidence.
---
7. Match Cardinality
Known ground-truth match-count distribution:
Matches per S1| Count
0| 123,247
1| 119,157
2| 375,212
3| 530,841
4| 484,115
5| 321,957
6| 164,868
7| 63,968
8| 18,680
9| 4,205
10| 534
11| 37
The zero-match population is approximately 5.58%.
This means the system must be capable of producing sets, not merely a single best match.
---
8. Source Relationship Profile
Known training statistics:
- S1 entities with S2 matches: 1,919,076
- S1 entities without S2 matches: 287,745
- S1 entities with S3 matches: 1,940,545
- S1 entities without S3 matches: 266,276
- S1 entities with both S2 and S3 matches: 1,776,047
Therefore, cross-source evidence is potentially valuable, but it must not be assumed to be universally reliable.
---
9. Product Goals
Goal 1 — High Candidate Recall
Recover as many true matches as possible during candidate generation.
Candidate recall defines the theoretical ceiling for final recall.
---
Goal 2 — High Precision
Remove false candidate pairs through learned similarity and entity-level reasoning.
Because F₀.₅ is precision-heavy, precision control is a first-class design objective.
---
Goal 3 — Correct Empty Predictions
Identify S1 entities for which there is insufficient evidence to make a match.
---
Goal 4 — Handle Multiple Matches
The system must correctly retain multiple valid matches for one S1 entity.
It must not assume:
«one S1 → one S2/S3»
---
Goal 5 — Generalize Across Countries
The architecture must work with France and potentially other unseen country labels without architectural changes.
---
Goal 6 — Reproducibility
The complete system must be executable from a clean environment and regenerate the required outputs.
---
Goal 7 — Fast Experimentation
Because the challenge permits multiple leaderboard submissions over the competition window, the architecture must allow rapid experimentation and comparison.
---
10. Non-Goals
The system will NOT attempt to:
- create a global universal business database
- perform external business research
- build a global one-to-one assignment
- force every S1 entity to have a match
- use an LLM as the primary matching mechanism
- rely solely on semantic embeddings
- perform exhaustive cross-product comparisons
- implement every proposed research idea without validation
- optimize ordinary accuracy at the expense of macro F₀.₅
---
11. High-Level Architecture
┌──────────────────┐
│    Raw Sources   │
│ S1 / S2 / S3     │
└────────┬─────────┘
│
▼
┌─────────────────────┐
│ Data Validation &   │
│ Forensics           │
└─────────┬───────────┘
│
▼
┌─────────────────────┐
│ Multi-View          │
│ Normalization       │
└─────────┬───────────┘
│
▼
┌──────────────────────────────┐
│ Multi-Pass Candidate         │
│ Generation / Blocking        │
└──────────────┬───────────────┘
│
▼
┌──────────────────────┐
│ candidate_pairs.tsv  │
└──────────┬───────────┘
│
▼
┌──────────────────────────────┐
│ Pairwise Feature Engineering│
└──────────────┬───────────────┘
│
▼
┌────────────────────┐
│ Pairwise Matcher   │
│ CatBoost/LightGBM  │
└─────────┬──────────┘
│
▼
┌──────────────────────────────┐
│ Reciprocal / Ranking         │
│ Evidence                     │
└──────────────┬───────────────┘
│
▼
┌──────────────────────────────┐
│ S1-Centered Cross-Source     │
│ Witness / Graph Evidence     │
└──────────────┬───────────────┘
│
▼
┌──────────────────────────────┐
│ Ambiguity + Singleton +      │
│ Entity-Level Decision        │
└──────────────┬───────────────┘
│
▼
┌──────────────────────┐
│ matching_results.tsv │
└──────────────────────┘
---
12. Design Principle: Separate Candidate Generation From Matching
Candidate generation and matching have different objectives.
Candidate Generation
Optimize:
Recall
It is acceptable for candidates to contain false positives.
Matching
Optimize:
Precision + F₀.₅
It must remove false candidates.
This separation is fundamental to the architecture.
---
13. Data Representation
The system must preserve multiple representations of each record.
For each relevant text field:
raw
normalized
tokenized
character representation
phonetic representation
numeric/address anchors
Raw values must never be destroyed.
Different representations should be used for different matching mechanisms.
---
14. Normalization Requirements
Normalization should investigate:
- Unicode normalization
- case normalization
- whitespace normalization
- punctuation normalization
- safe symbol normalization
- common legal suffix normalization
- abbreviation handling
- transliteration-aware representation
- tokenization
- numeric token extraction
- phonetic representation
Examples of potentially useful transformations include:
Corporation → corp
Limited → ltd
Road → rd
Street → st
However, normalization must be conservative.
Over-normalization can turn genuinely different businesses into identical representations.
Therefore, multiple representations should coexist.
---
15. Candidate Generation
Candidate generation will use a union of independent retrieval routes.
Potential blocking mechanisms:
15.1 Exact normalized name
Useful for strong name matches.
15.2 Exact normalized address
Useful when addresses survive normalization.
15.3 Exact name + address
Extremely strong but potentially low recall.
15.4 Character n-gram retrieval
Designed to tolerate:
- typos
- punctuation differences
- abbreviations
- word variations
15.5 TF-IDF retrieval
Useful for lexical similarity.
15.6 Token retrieval
Useful for reordered words.
15.7 Rare-token retrieval
Rare business-name/address tokens can provide strong identity signals.
15.8 Phonetic retrieval
Useful for certain spelling/transliteration variations.
15.9 Numeric/address-anchor retrieval
Numbers can act as highly discriminative address fingerprints.
15.10 Approximate retrieval
RapidFuzz or equivalent methods may be used where computationally viable.
---
16. Candidate Route Metadata
Every candidate should retain information about how it was discovered.
Example:
exact_name = 1
exact_address = 0
char_ngram = 1
tfidf = 1
phonetic = 0
rare_token = 1
numeric = 0
Additional retrieval information may include:
- retrieval rank
- retrieval score
- number of routes retrieving candidate
- strongest route
This information can become model features.
---
17. Candidate Recall Measurement
Candidate generation must be evaluated independently.
For training data:
candidate recall =
true matches present in candidate set
------------------------------------
total true matches
Candidate recall must also be measured by important subgroups:
- singleton
- one-match
- multi-match
- S2-only
- S3-only
- both-source
- duplicate-name
- duplicate-address
- high-cardinality
No final model improvement can compensate for a true match that never entered the candidate set.
---
18. Pairwise Feature Engineering
18.1 Name Features
Potential features:
- exact equality
- normalized equality
- edit distance
- normalized edit similarity
- Jaro/Jaro-Winkler
- token Jaccard
- token overlap
- token-sort similarity
- character n-gram similarity
- TF-IDF similarity
- rare-token overlap
- phonetic similarity
- length difference
- prefix/suffix compatibility
---
18.2 Address Features
Potential features:
- exact equality
- normalized equality
- edit similarity
- token overlap
- character similarity
- TF-IDF similarity
- token-sort similarity
- numeric-token overlap
- postal/PIN-like overlap
- component overlap
- length differences
Missing address must be represented explicitly.
Missing ≠ contradiction.
---
18.3 Interaction Features
Important combinations:
name_similarity × address_similarity
min(name_similarity, address_similarity)
mean(name_similarity, address_similarity)
name strong + address weak
address strong + name weak
The model should learn which combinations are reliable.
---
19. Baseline Matching Model
The first production-capable baseline should use a gradient-boosted model such as:
- CatBoost
- LightGBM
The model receives pairwise features and outputs a match score/probability.
A threshold of 0.5 must NOT be assumed.
Threshold selection must be performed against validation F₀.₅.
---
20. Hard-Negative Mining
Hard negatives are essential because many false matches will have high lexical similarity.
Training should deliberately include difficult non-matches such as:
- same business name, different address
- nearly identical names
- same address, different business
- generic business names
- similar transliterations
- same locality but different business
- high similarity but incorrect cross-source evidence
Hard-negative mining must be based on generalizable characteristics, not memorized IDs.
---
21. LCCM Research Concept
Locality-Agnostic Consensus & Contrastive Manifold Alignment
LCCM is a research direction, not a mandatory final architecture.
Core hypothesis:
«A match becomes more trustworthy when independent sources provide mutually consistent evidence, while genuine contradictions should reduce confidence.»
The system should test whether cross-source evidence improves held-out macro F₀.₅.
---
22. Cross-Source Evidence
For an S1 entity:
S1
/  \
/    \
S2 ─── S3
Possible relationships:
Direct evidence
S1 ↔ S2
S1 ↔ S3
Witness evidence
S2 ↔ S3
The S2-S3 relationship can act as supporting evidence for an S1 candidate.
---
23. Support / Neutral / Contradiction Model
A critical design rule:
Weak evidence is not automatically contradiction.
Cross-source relationships should be classified conceptually as:
SUPPORT
NEUTRAL / UNKNOWN
CONTRADICTION
Examples:
Support
S1 strongly resembles S2.
S2 strongly resembles S3.
S1 moderately/strongly resembles S3.
Neutral
S1 strongly resembles S2.
S2 and S3 have insufficient information to establish a relationship.
Contradiction
Strong evidence indicates incompatible identities.
The system must not manufacture contradiction merely from missing or weak evidence.
---
24. Graph Scope
The graph layer will be:
local and S1-centered.
We will NOT initially construct a massive global graph containing all S1/S2/S3 nodes and all possible edges.
Reasons:
- computational cost
- memory consumption
- unnecessary complexity
- S1 can have multiple matches
- no one-to-one assignment requirement
The graph exists primarily to generate evidence features, not to replace the matching model.
---
25. Graph Features
Potential features:
- direct S1-S2 score
- direct S1-S3 score
- S2-S3 witness score
- strongest witness
- average witness
- minimum witness
- number of strong witnesses
- triangle-support indicator
- reciprocal evidence
- cross-source consistency
- contradiction evidence
All graph features must be evaluated through ablation.
---
26. Graph-Assisted Candidate Recovery
Experimental mechanism:
S1 → S2 strong
S2 → S3 strong
↓
candidate S3 may be promoted for S1
Purpose:
Recover true S1-S3 matches that direct S1-S3 blocking missed.
Risk:
Candidate explosion.
Therefore this mechanism must be:
- optional
- configurable
- independently measurable
- bounded
It must only be adopted if it improves the complete pipeline.
---
27. Reciprocal Retrieval
A candidate may become more trustworthy if retrieval is mutual.
Potential signals:
S1 retrieves S2 highly
AND
S2 retrieves S1 highly
Features:
- reciprocal rank
- mutual top-k
- rank difference
- retrieval agreement
Again, these are features, not hard rules.
---
28. Entity-Level Decision Engine
Pairwise probability is not the final answer.
For every S1 entity, compute:
- top score
- second score
- third score
- top-second margin
- candidate count
- number above threshold
- number of high-confidence candidates
- name uniqueness
- address uniqueness
- retrieval route count
- reciprocal evidence
- graph support
- witness count
- contradiction evidence
- source-specific reliability
The final decision operates at the S1 entity level.
---
29. Ambiguity Handling
The system must identify ambiguous entities.
Example:
Candidate A = 0.91
Candidate B = 0.90
Candidate C = 0.89
versus:
Candidate A = 0.97
Candidate B = 0.51
Candidate C = 0.32
The raw top score alone is insufficient.
Margin and candidate distribution should influence the decision.
---
30. Singleton Decision
For every S1 entity, the decision engine must be capable of returning:
[]
when evidence is insufficient.
This is not an error state.
It is a legitimate prediction.
---
31. Source-Specific Calibration
S1-S2 and S1-S3 may exhibit different noise characteristics.
The system should measure:
- S1-S2 score distributions
- S1-S3 score distributions
- precision at thresholds
- recall at thresholds
- false merge behavior
Potentially use source-specific calibration/thresholds if validation supports it.
---
32. Optional Contrastive Model
A lightweight Siamese/contrastive component may eventually be tested.
Purpose:
Capture difficult lexical variations that traditional similarity features fail to represent.
It is explicitly not Phase 1.
It may only be introduced after strong classical baselines are established.
Success criterion:
It must demonstrate a measurable improvement on held-out macro F₀.₅ or another clearly justified metric.
If it does not improve the system, it will not be included merely because it is technically sophisticated.
---
33. Validation Strategy
Validation must be performed at the S1 entity level.
Do NOT randomly split individual candidate pairs.
All candidate pairs belonging to the same S1 entity must remain in the same validation partition.
This prevents leakage through the same reference entity.
---
34. Validation Cohorts
Every major experiment should report performance on:
1. all S1 entities
2. zero-match entities
3. one-match entities
4. multi-match entities
5. S2-only
6. S3-only
7. both-source
8. high-cardinality entities
9. duplicate-name entities
10. duplicate-address entities
11. ambiguous entities
---
35. Metrics
Primary
Macro F₀.₅ per S1
Secondary
- precision
- recall
- false merge rate
- singleton/no-match accuracy
- candidate recall
- candidate count
- reduction ratio
- average candidates per S1
- maximum candidates per S1
- runtime
- memory consumption
---
36. Experiment Framework
Every experiment must have:
experiment_id
date
git_commit
hypothesis
change
configuration
candidate_recall
candidate_count
precision
recall
macro_F0.5
singleton_accuracy
runtime
memory
decision
notes
Example:
EXP001
EXP002
EXP003
...
---
37. Planned Experiment Ladder
V0 — Data Forensics
Validate:
- schemas
- row counts
- missing values
- country distribution
- duplicate structure
- match cardinality
---
V1 — Basic Blocking
Implement:
- exact normalized name
- exact normalized address
- exact name + address
Measure candidate recall and volume.
---
V2 — Multi-Pass Blocking
Add:
- character n-grams
- TF-IDF
- token retrieval
- phonetic
- rare tokens
- numeric anchors
---
V3 — Pairwise Features
Implement name/address similarity features.
---
V4 — Gradient-Boosted Matcher
Train CatBoost/LightGBM.
---
V5 — Entity-Level Thresholding
Optimize:
- threshold
- top-vs-second margin
- empty prediction behavior
---
V6 — Hard Negatives
Introduce difficult false candidates into training.
---
V7 — Reciprocal Retrieval
Add retrieval agreement features.
---
V8 — Witness Evidence
Add S2-S3 cross-source evidence.
---
V9 — Triangle Support
Test triangle-style consistency.
---
V10 — Contradiction Features
Test explicit contradiction evidence.
---
V11 — Ambiguity/Singularity Safeguards
Improve:
- empty prediction
- ambiguous candidate suppression
- margin handling
---
V12 — Graph-Assisted Candidate Recovery
Test controlled cross-source candidate promotion.
---
V13 — Contrastive Model
Only if classical approaches plateau.
---
V14 — Best Combined System
Combine only experimentally validated components.
---
V15 — Final Calibration
Tune:
- thresholds
- source-specific thresholds
- entity-level rules
- ambiguity handling
using held-out validation.
---
38. Ablation Principle
No feature or architectural component becomes part of the final system simply because it sounds theoretically useful.
For every component:
Baseline
↓
Add component
↓
Measure
↓
Compare
↓
Keep / Reject
A component should be retained only when its benefits justify:
- complexity
- runtime
- memory
- candidate growth
- risk of overfitting
---
39. Data Leakage Prevention
The system must prevent:
- test-label leakage
- manual test-set labeling
- external lookup
- future information leakage
- cross-validation entity leakage
- accidental use of ground truth during inference
- hardcoded test identities
Ground truth may be used only for training/validation experiments where appropriate.
---
40. Scalability Requirements
The total dataset is large enough that naive all-pairs matching is infeasible.
The system must avoid:
S1 × S2
S1 × S3
S2 × S3
full Cartesian products.
Indexes should be reused across experiments where possible.
Memory-heavy representations should be controlled.
Sparse matrices should be preferred where appropriate.
Large intermediate artifacts should be written to disk when necessary.
---
41. Repository Architecture
amazon-ml-challenge-2026/
├── .github/
│   └── workflows/
│       ├── tests.yml
│       └── repo-check.yml
│
├── src/
│   ├── io.py
│   ├── normalization.py
│   │
│   ├── blocking/
│   │   ├── __init__.py
│   │   ├── exact.py
│   │   ├── char_ngram.py
│   │   ├── tfidf.py
│   │   ├── phonetic.py
│   │   ├── rare_tokens.py
│   │   └── numeric.py
│   │
│   ├── features/
│   │   ├── name.py
│   │   ├── address.py
│   │   ├── interactions.py
│   │   └── retrieval.py
│   │
│   ├── model/
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── calibration.py
│   │
│   ├── graph/
│   │   ├── witnesses.py
│   │   ├── triangle.py
│   │   └── contradiction.py
│   │
│   ├── decision/
│   │   ├── entity_decision.py
│   │   ├── ambiguity.py
│   │   └── threshold.py
│   │
│   └── evaluation/
│       ├── metrics.py
│       ├── diagnostics.py
│       └── candidate_recall.py
│
├── scripts/
│   ├── profile_data.py
│   ├── build_candidates.py
│   ├── build_features.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── configs/
├── experiments/
│   └── experiment_log.csv
│
├── docs/
├── tests/
│
├── .gitignore
├── README.md
├── requirements.txt
└── LICENSE
---
42. Output Contract
Final output must contain:
output/
├── matching_results.tsv
└── candidate_pairs.tsv
The final submission package must additionally contain the runnable code and methodology documentation required by the challenge.
---
43. Submission Validator
Before every leaderboard submission, automatically verify:
Matching output
- every S1 entity exists
- no duplicate S1 rows
- IDs are valid
- no S1 IDs appear as matches
- matched IDs belong to S2/S3
- no duplicate match IDs
- correct empty-list representation
- correct TSV format
Candidate output
- valid S1 IDs
- valid S2/S3 IDs
- no invalid pair
- no duplicate pair
- every predicted match is present
- candidate file corresponds to final model input
---
44. Experiment Artifacts
Artifacts must NOT be committed to Git.
Use:
data/
artifacts/
models/
indexes/
features/
predictions/
as local/generated directories.
Only source code, configuration, documentation and experiment metadata should normally be tracked.
---
45. Team Responsibilities
Animesh
Lead:
- system architecture
- ML strategy
- entity-level decision logic
- validation methodology
- experiment design
- LCCM formulation
- graph evidence formulation
- final integration
- experiment control
---
Krish
Lead:
- candidate generation
- retrieval
- character n-grams
- TF-IDF
- phonetic blocking
- rare-token retrieval
- numeric/address retrieval
- candidate-recall optimization
---
Yash
Lead:
- pipeline engineering
- data IO
- CatBoost/LightGBM
- training/inference
- performance optimization
- memory management
- production integration
---
Supreeti
Lead:
- cross-source evidence
- local graph
- witness features
- triangle evidence
- contradiction analysis
- hard-negative/error analysis
- graph experiments
---
46. Git Workflow
"main" remains stable.
Development branches:
animesh/*
krish/*
yash/*
supreeti/*
Commit prefixes:
feat:
fix:
exp:
model:
perf:
docs:
test:
Heavy ML artifacts must never be committed.
---
47. CI Requirements
CI should validate:
- Python syntax
- imports
- unit tests
- configuration validity
- basic pipeline interfaces
- output validation logic
CI must NOT run the complete competition dataset.
---
48. Antigravity Role
Antigravity is the engineering execution agent.
It may:
- inspect repository structure
- create files
- refactor modules
- implement algorithms
- run tests
- run experiments
- inspect logs
- analyze generated metrics
- propose improvements
- maintain documentation
However, Antigravity must NOT independently decide that a sophisticated method is better merely because it sounds theoretically superior.
It must operate experimentally.
---
49. Antigravity Decision Rules
For every major proposed improvement:
Step 1
State:
Hypothesis
Example:
«Reciprocal retrieval will reduce false merges among duplicate business names.»
Step 2
Implement the smallest testable version.
Step 3
Evaluate on held-out validation.
Step 4
Compare against the previous best.
Step 5
Record results.
Step 6
Either:
KEEP
or
REJECT
with justification.
---
50. What Antigravity Must Never Do
Antigravity must never:
- fabricate metrics
- claim improvement without measurement
- use external business information
- silently introduce external datasets
- hardcode test identities
- hardcode France-specific answers
- assume one-to-one matching
- force a match
- replace classical matching with an LLM without experiment
- delete existing experiments without preserving history
- overwrite a stronger validated approach without comparison
- commit competition data
- treat weak evidence as contradiction automatically
- create a global one-to-one graph assignment
- optimize only pairwise accuracy
---
51. Definition of Done
The project is considered competition-ready only when:
Data
- all required datasets load correctly
- validation passes
- country handling is open-set
- missing values are handled
Candidate Generation
- multiple blocking routes implemented
- candidate recall measured
- candidate set is reproducible
- candidate volume is computationally manageable
Matching
- pairwise features implemented
- trained matcher implemented
- threshold optimized for macro F₀.₅
- hard negatives evaluated
Entity Decision
- multiple matches supported
- empty predictions supported
- ambiguity handled
- singleton behavior evaluated
Cross-Source Reasoning
- witness/graph system experimentally evaluated
- support/neutral/contradiction distinction implemented if validated
- graph-assisted candidate recovery tested separately
Evaluation
- macro F₀.₅ reported
- subgroup diagnostics available
- candidate recall reported
- false merge behavior analyzed
Reproducibility
- deterministic configuration
- experiment log
- pinned dependencies
- runnable pipeline
- submission validator
Submission
- "matching_results.tsv" generated
- "candidate_pairs.tsv" generated
- every predicted match exists in candidate set
- complete S1 coverage
- final package structure validated
---
52. Research Philosophy
The project should follow this hierarchy:
Correctness
↓
Candidate Recall
↓
Precision
↓
F₀.₅
↓
Robustness
↓
Runtime
↓
Complexity
A sophisticated model with poor candidate recall is not useful.
A high-recall candidate generator with uncontrolled false positives is also insufficient.
The target is the complete system.
---
53. Strategic Principle
The team should not ask:
«"What is the most advanced model we can build?"»
Instead ask:
«"What evidence allows us to confidently say these two records represent the same real-world business?"»
The final decision should combine:
Lexical Evidence
+
Address Evidence
+
Structural Evidence
+
Retrieval Evidence
+
Reciprocal Evidence
+
Cross-Source Evidence
+
Ambiguity Analysis
while preserving the ability to say:
NO CONFIDENT MATCH
---
54. Final Product
The final product is an offline, reproducible, large-scale entity-resolution engine that transforms:
S1 + S2 + S3
into:
candidate_pairs.tsv
+
matching_results.tsv
using a modular architecture capable of rapid experimentation.
The system's strongest design principle is:
«Generate broadly, compare intelligently, reason across evidence, and merge conservatively.»
No individual technique—including blocking, CatBoost, graph reasoning, LCCM, hard negatives, or contrastive learning—is considered successful until empirical validation demonstrates that it improves the complete system on held-out data.
