# Recommender System Research Project

### Breaking the Filter Bubble with Unexpectedness-Aware Recommendations

## 1. Project Overview

This repository contains the implementation of a research prototype for an **unexpectedness-aware recommender system**. The goal is to design a system that balances **relevance and unexpectedness** in recommendations to mitigate the **filter bubble problem**.

Modern recommender systems often over-optimize for historical relevance. As a result, users are repeatedly shown items similar to what they have already consumed, which leads to boredom, lack of discovery, and reduced long-term engagement.

This project explores the concept of **unexpectedness-aware recommendation**, where recommendations are ranked not only by relevance but also by how meaningfully they deviate from a user's expected preferences.

The final goal is to implement a **Transformer-based sequential recommender architecture** that incorporates:

* relevance
* unexpectedness
* temporal behavior
* rating intensity
* user preference stability

The current repository represents the **initial experimental pipeline** required before implementing the final model.

---

# 2. Dataset Selection

## MovieLens 32M Dataset

Dataset chosen:

MovieLens 32M

Source:
https://grouplens.org/datasets/movielens/

Dataset characteristics:

* ~32 million interactions
* ~200k users
* ~87k movies
* timestamps for interactions
* explicit ratings (1–5)

### Why MovieLens 32M?

The dataset was chosen for several reasons:

1. It contains **explicit ratings**, which allows us to preserve rating intensity instead of converting interactions to binary feedback.
2. It includes **timestamps**, which are necessary for modeling **temporal behavior and stability**.
3. The dataset size is large enough to simulate **real-world recommender system environments**.
4. It is widely used in recommender system research, making it suitable for benchmarking.

---

# 3. Project Structure

Current repository structure:

```
recsys/
│
├── configs/
├── data/
│   └── train_test_split.py
│
├── evaluation/
│   ├── precision.py
│   ├── recall.py
│   ├── ndcg.py
│   ├── serendipity.py
│   └── unexpectedness.py
│
├── experiments/
│   ├── test_popularity.py
│   ├── experiment_runner.py
│   └── preprocess_ml1m.py
│
├── models/
│   ├── popularity.py
│   ├── purs.py
│   └── purs_train.py
│
└── requirements.txt
```

---

# 4. Environment Setup

A Python virtual environment was created using:

```
python -m venv .venv
```

Libraries were installed via pip.

## Core Libraries

The following libraries were installed:

```
numpy
pandas
scikit-learn
scipy
matplotlib
seaborn
tqdm
```

### Justification

| Library      | Purpose                                    |
| ------------ | ------------------------------------------ |
| NumPy        | numerical computation                      |
| Pandas       | dataset preprocessing                      |
| Scikit-Learn | machine learning utilities                 |
| SciPy        | sparse matrices and scientific computation |
| Matplotlib   | visualization                              |
| Seaborn      | statistical visualization                  |
| tqdm         | progress tracking                          |

Deep learning libraries such as PyTorch were intentionally **not installed yet**, since early experiments use classical recommender models.

---

# 5. Evaluation Metrics Implemented

Evaluation metrics were implemented before models to ensure proper comparison between algorithms.

Metrics implemented:

* Precision@K
* Recall@K
* NDCG@K
* Serendipity
* Unexpectedness

### Why These Metrics?

Traditional recommender systems optimize relevance only. However, our research focuses on **unexpectedness and serendipity**, so evaluation must reflect that.

Metric meanings:

| Metric         | Purpose                                         |
| -------------- | ----------------------------------------------- |
| Precision@K    | fraction of recommended items that are relevant |
| Recall@K       | fraction of relevant items retrieved            |
| NDCG@K         | ranking quality metric                          |
| Serendipity    | relevant items that were not expected           |
| Unexpectedness | distance from user's expected set               |

These metrics will allow us to evaluate how well the system balances **relevance vs novelty**.

---

# 6. Baseline Models Implemented

Before implementing the final architecture, we implemented **baseline models**.

This is essential for research validation.

## Baselines Implemented

### 1. Popularity Recommender

Recommends globally popular items.

Logic:

```
count interactions per item
sort descending
recommend top K items
```

Purpose:

* simplest recommender baseline
* measures if the system is better than naive recommendations

---

### 2. PURS (Personalized Unexpected Recommender System)

This model is designed to optimize recommendation utility by combining relevance and unexpectedness.

Training objective:

```
score(user, positive_item) > score(user, negative_item)
```

Training samples consist of triplets:

```
(user, positive item, negative item)
```

PURS is the current primary baseline in this repository.

---

# 7. Train/Test Splitting Strategy

We implemented a **temporal split**:

```
Train = all interactions except last one
Test = last interaction per user
```

This simulates real-world recommendation scenarios where the model predicts **the next item a user interacts with**.

---

# 8. Problems Discovered During Implementation

During early baseline implementation, several issues were identified.

## Problem 1: Negative Sampling Error

Current implementation samples negative items randomly.

```
j = random item
```

However, this can produce:

```
j ∈ user's positive history
```

This creates incorrect training signals.

Correct approach:

```
sample j such that j ∉ user history
```

We are currently fixing this.

---

## Problem 2: Training on All Interactions

Current training loops over **all user-item interactions**.

This leads to extremely slow training for MovieLens 32M.

Example complexity:

```
users × interactions
```

This results in millions of updates.
# Unexpectedness-Aware Sequential Recommender

This repository supports a research workflow for an unexpectedness-aware sequential recommender system, currently centered on MovieLens-1M and PURS as the active model path.

## Part 1: What We Have Done So Far (with Logic)

### 1. Defined a clear research objective
We fixed the core objective as balancing relevance and unexpectedness, because a pure relevance objective reinforces filter bubbles and hurts discovery.

### 2. Switched to MovieLens-1M for the current implementation phase
We moved practical development to MovieLens-1M, because Kaggle free-tier iteration requires shorter experiment cycles and predictable memory/runtime.

### 3. Standardized configuration around this objective
We updated defaults to MovieLens-1M and PURS-only configuration, because reducing branching in early development lowers integration bugs and accelerates validation.

### 4. Added dataset-aware loading infrastructure
We implemented loaders that handle both ML-1M and ML-32M formats, because we need a stable migration path while still allowing scale-up later.

### 5. Implemented one-time sequential preprocessing primitives
We added:
1. k-core filtering (min user/item interactions)
2. user/item contiguous reindexing
3. chronological ordering
4. temporal train/val/test splitting (last=test, second-last=val)
5. log-scale time-gap buckets

We did this because sequential recommendation quality depends on correct temporal ordering and consistent ID space.

### 6. Added preprocessing artifact persistence
We save train/val/test splits, encoder, histories, and metadata to disk, because one-time preprocessing prevents repeated expensive work and guarantees reproducibility across model runs.

### 7. Added a dedicated preprocessing entrypoint
We created a standalone preprocessing script for ML-1M, because data preparation must be runnable independently from training for clean experiment management.

### 8. Removed BPR from the active code path
We removed BPR models/scripts and made runner/model exports PURS-only, because your immediate research track is unexpectedness-aware sequential modeling and BPR branches were adding maintenance overhead.

### 9. Refactored generic training dataset naming
We renamed the training dataset abstraction from BPR-specific naming to pairwise naming, because the negative sampling pattern is reusable across methods and should not be tied to a removed baseline.

### 10. Updated notebook and documentation alignment
We updated Kaggle notebook framing and repository docs to PURS + ML-1M, because documentation must match executable code to avoid experiment drift.

### 11. Current status after Part 1
What works now:
1. ML-1M loading and environment-aware paths
2. Sequential preprocessing and artifact generation
3. PURS training/evaluation pipeline in runner
4. Kaggle notebook flow aligned to PURS

What is intentionally not completed yet:
1. SASRec baseline
2. NOVA-inspired non-invasive gated attention model
3. MeanShift-based unexpectedness lookup table for all user-item pairs
4. Unified three-model comparison report

## Part 2: Complete Future Implementation Plan (End-to-End)

This section is the full roadmap to finish implementation of the project, from current code state to final three-model comparison.

### Phase 1: Lock the experimental contract
1. Finalize immutable experiment rules: dataset split policy, sequence length, negative sampling, and target metrics.
Why: if these rules change mid-way, model comparisons become invalid.
Output: one frozen protocol document used by every run.

2. Freeze seeds and run settings for reproducibility.
Why: we need to separate model improvements from randomness.
Output: fixed seed policy and reproducible run template.

### Phase 2: Complete data foundation
1. Finalize one-time preprocessing artifacts for ML-1M.
Why: all models must train on exactly the same processed data.
Output: train, val, test splits, encoder, sequence metadata, and history artifacts.

2. Add artifact versioning and manifest metadata.
Why: we must always know which preprocessing settings produced each result.
Output: artifact manifest with configuration hash, timestamp, and parameters.

### Phase 3: Build unexpectedness infrastructure
1. Choose and lock the item embedding backbone (for distance computation).
Why: unexpectedness distance is only meaningful if embedding space is consistent.
Output: item embedding checkpoint and embedding export.

2. Run user-wise MeanShift on user history embeddings.
Why: this defines each user's expected preference regions.
Output: per-user cluster representation.

3. Precompute unexpectedness distances for user-item scoring.
Why: on-the-fly clustering/distance is too slow for repeated training and evaluation.
Output: lookup artifact optimized for dataloader access.

### Phase 4: Unify training/evaluation data interface
1. Build one shared sequential dataset contract for all models.
Why: model comparison is fair only when inputs and sampling policy are identical.
Output: common dataloader outputs: items, ratings, time gaps, unexpectedness, positives, negatives.

2. Enforce evaluation candidate protocol (1 positive + 99 negatives).
Why: this is the paper-aligned ranking setup for all reported metrics.
Output: single evaluation pipeline used by SASRec, PURS, and NOVA.

### Phase 5: Implement SASRec baseline fully
1. Implement SASRec model with the shared dataloader contract.
Why: SASRec is the relevance-first sequential baseline needed to quantify gains from unexpectedness modeling.
Output: trainable/evaluable SASRec module.

2. Validate SASRec metrics and stability.
Why: a weak baseline creates misleading conclusions.
Output: stable baseline runs with metric logs.

### Phase 6: Align and finalize PURS implementation
1. Match PURS scoring exactly to chosen utility definition.
Why: baseline must reflect the intended unexpectedness-aware objective, not an approximate variant.
Output: validated PURS implementation with clear equation-code mapping.

2. Verify PURS with the new shared pipeline.
Why: all models must use identical data/eval mechanics.
Output: PURS benchmark runs directly comparable to SASRec.

### Phase 7: Implement NOVA-inspired model
1. Build four-embedding gated fusion (item, rating, time gap, unexpectedness).
Why: this is the core mechanism for multi-signal contextualization.
Output: fused representation module.

2. Enforce non-invasive attention rule: Q/K from fused representation, V from pure item embedding.
Why: this preserves item semantic integrity while still leveraging side information.
Output: custom attention layer with architectural constraint guaranteed in code.

3. Integrate NOVA model into the shared trainer.
Why: training logic must remain identical across models for fair comparison.
Output: trainable/evaluable NOVA path in experiment runner.

### Phase 8: Build robust experiment system
1. Unify trainer, logger, checkpointing, and config handling for all three models.
Why: mixed training code paths increase bugs and reduce reproducibility.
Output: single experiment framework with model-specific hooks.

2. Add resume/restart and failure-safe checkpoints.
Why: Kaggle sessions are time-limited and can terminate unexpectedly.
Output: resilient run pipeline.

### Phase 9: Hyperparameter and ablation studies
1. Run constrained hyperparameter search per model.
Why: each architecture needs reasonable tuning before comparison.
Output: best config per model under equal budget.

2. Run ablations for NOVA components.
Why: we need to prove which design components actually contribute.
Output: ablation table for gate, rating, time-gap, unexpectedness, and non-invasive attention constraint.

### Phase 10: Statistical validation and error analysis
1. Repeat final runs with multiple seeds.
Why: single-run improvements are not scientifically reliable.
Output: mean and variance for each reported metric.

2. Perform targeted failure analysis (user/item segments).
Why: this explains where the model helps and where it fails.
Output: qualitative and quantitative error analysis section.

### Phase 11: Final reporting package
1. Generate final comparison tables for SASRec, PURS, NOVA.
Why: this is the core evidence for the research claim.
Output: publication-ready model comparison results.

2. Generate plots: training curves, metric trade-offs, and ablations.
Why: visual evidence improves interpretability and paper clarity.
Output: figure set aligned with final metrics.

3. Export reproducible artifacts and run cards.
Why: every reported number must be traceable to a config and checkpoint.
Output: experiment cards, artifact index, and final result bundle.

### Phase 12: Done criteria
Implementation is complete when all conditions below are true:
1. SASRec, PURS, and NOVA train and evaluate on the same pipeline.
2. NOVA respects the non-invasive Q/K/V constraint in verified code.
3. NDCG@10, Recall@10, and Unexpectedness@10 are reported for all models.
4. Ablation and multi-seed stability results are available.
5. Paper tables and figures are reproducible from saved artifacts.

## Immediate execution order

1. Freeze experiment contract and preprocessing manifest.
2. Finalize unexpectedness lookup generation.
3. Implement and validate SASRec baseline.
4. Revalidate PURS under shared pipeline.
5. Implement NOVA non-invasive model.
6. Run benchmark, ablations, and multi-seed validation.
7. Export final results for paper integration.

This order is intentional: data and protocol first, model implementation second, scientific validation and reporting last.

