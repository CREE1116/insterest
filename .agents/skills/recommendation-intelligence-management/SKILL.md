---
name: recommendation-intelligence-management
description: >-
  Recommendation intelligence management skill covering ML model retrieval-ranking pipelines,
  embedding and concept drift monitoring, feedback loops, counterfactual off-policy evaluations
  (IPS, DR), and Redis HNSW ANN index tuning.
---

# Recommendation Intelligence Management

This skill document defines procedures for monitoring recommendation quality, detecting drift, executing counterfactual offline evaluations, and optimizing Redis HNSW vector indices.

---

## 1. Concept Drift, Embedding Drift & Feedback Loops

Recommendation engines are vulnerable to feedback loops (filter bubbles) where models repeatedly recommend similar categories to users, amplifying observed interactions and biasing training datasets.

### A. Detecting Embedding & Concept Drift
*   **Concept Drift**: Occurs when user behavior patterns change over time (e.g. shift in category interest). Monitor via time-decay evaluation metrics.
*   **Embedding Drift**: Occurs when newly trained models map features to coords inconsistent with older, cached embeddings in Redis.
*   **Drift Analysis Protocol**:
    1.  Compare cosine distances of identical items before and after model retraining.
    2.  Check alignment similarities: Run `python recommendation_service/test_bias.py` to inspect if projection layers maintain separation boundaries.

### B. Feedback Loop Mitigation (Freshness vs Popularity)
Avoid popularity collapse (long-tail starvation) by adding exploration offsets:
*   Use time-decay pooling weights ($0.5^{N-i}$) to prioritize recent interests.
*   Implement $\epsilon$-greedy exploration: Mix a fraction (e.g., $10\%$) of randomized or high-freshness items (posts created within the last 24 hours) into recommendations.

---

## 2. Counterfactual Offline Evaluation

Since offline interaction logs are biased by exposure (users only like items that were shown to them), evaluate models using unbiased off-policy estimators.

### A. Inverse Propensity Scoring (IPS)
IPS adjusts metrics by the inverse probability of an item being shown (propensity score $P(O_i = 1)$) to simulate an unbiased randomized test:
$$\text{Recall}_{\text{IPS}} = \frac{1}{|U|} \sum_{u \in U} \sum_{i \in \text{likes}_u} \frac{\mathbb{I}(i \in \text{Recs}_u)}{P(O_i = 1)}$$
*   **Propensity Estimation**: Estimate $P(O_i = 1)$ using item exposure rates or historical positioning ranks.

### B. Doubly Robust (DR) Estimator
To reduce high variance in IPS, combine IPS with a reward model prediction (imputation):
$$\mathcal{M}_{\text{DR}} = \sum_{i} \left( \hat{R}_i + \frac{\mathbb{I}(O_i=1)(R_i - \hat{R}_i)}{P(O_i=1)} \right)$$
*   $\hat{R}_i$: Predicted probability of a user liking an item from a regression baseline.
*   $R_i$: Actual interaction label.

---

## 3. Redis HNSW (ANN) Sharding & Memory Tuning

The recommendation service relies on Redis Search HNSW vector indexes for Approximate Nearest Neighbors (ANN) searches.

### A. Shard Balancing & Retrieval Skew
*   **Retrieval Skew**: Heavy retrieval traffic hitting a small subset of hot items. Mitigate by caching popular post vector outputs.
*   **Shard Balancing**: Ensure Redis cluster partitions index keys uniformly.

### B. HNSW Index Parameters Tuning
Tune HNSW index creation inside `app/ml/vector_store.py`:
```bash
FT.CREATE idx_post
ON HASH PREFIX 1 post:
SCHEMA
  vector VECTOR HNSW 6
    TYPE FLOAT32
    DIM 128
    DISTANCE_METRIC COSINE
    M 16
    EF_CONSTRUCTION 200
    EF_RUNTIME 10
```
*   `M` (Max outgoing links per node): Increase (e.g. 16 to 32) for higher recall accuracy at the cost of memory.
*   `EF_RUNTIME` (Search candidate pool size): Increase (e.g. 10 to 50) dynamically during peak search demands to optimize precision-latency trade-offs.
*   **Memory Fragmentation**: Check index size periodically and run `FT.INFO` to verify docs indexing footprint.
