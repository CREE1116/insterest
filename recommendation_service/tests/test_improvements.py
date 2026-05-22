"""
Unit tests for recommendation performance improvements:
  A. Adaptive RRF (personalization boost)
  B. Freshness-decay trending (score ordering validation)
  C. Dynamic hashtag weighting (model.py)
  D. MMR diversity reranking
"""
import math


# ── A. Adaptive RRF ──────────────────────────────────────────────────────────

def _adaptive_rrf_merge(
    rank_lists: list[list[str]],
    weights: list[float] | None = None,
    k: int = 60,
    exclude: set[str] | None = None,
) -> list[tuple[str, float]]:
    """Adaptive RRF: each rank-list has an individual weight multiplier."""
    exclude = exclude or set()
    weights = weights or [1.0] * len(rank_lists)
    scores: dict[str, float] = {}
    for w, r_list in zip(weights, rank_lists):
        for rank, pid in enumerate(r_list):
            if pid in exclude:
                continue
            scores[pid] = scores.get(pid, 0.0) + w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def test_adaptive_rrf_personalization_boost_promotes_user_items():
    """Items appearing only in the user-personalization list should rank higher
    than items only in the CF list when personalization weight is doubled."""
    user_only = ["u1", "u2"]
    cf_only = ["c1", "c2"]
    merged = _adaptive_rrf_merge([user_only, cf_only], weights=[2.0, 1.0])
    ids = [i for i, _ in merged]
    assert ids.index("u1") < ids.index("c1"), "personalization item u1 should outrank cf item c1"


def test_adaptive_rrf_shared_item_still_tops():
    """An item in both lists should score highest regardless of weights."""
    shared = "shared"
    merged = _adaptive_rrf_merge([[shared, "a"], [shared, "b"]], weights=[2.0, 1.0])
    assert merged[0][0] == shared


def test_adaptive_rrf_equal_weights_matches_standard_rrf():
    """With equal weights of 1.0, adaptive RRF should produce the same ranking as standard RRF."""
    rank_lists = [["a", "b", "c"], ["c", "a"]]

    def _std_rrf(rank_lists, k=60):
        scores: dict[str, float] = {}
        for r_list in rank_lists:
            for rank, pid in enumerate(r_list):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)

    std = [i for i, _ in _std_rrf(rank_lists)]
    adaptive = [i for i, _ in _adaptive_rrf_merge(rank_lists, weights=[1.0, 1.0])]
    assert std == adaptive


# ── B. Freshness-decay trending score ordering ────────────────────────────────

def _trending_score(popularity_count: float, hours_since_upload: float, lam: float = 0.01) -> float:
    """popularity × exp(-λ × hours_since_upload)"""
    return popularity_count * math.exp(-lam * hours_since_upload)


def test_freshness_decay_newer_beats_older_with_same_popularity():
    old_score = _trending_score(100, hours_since_upload=100)
    new_score = _trending_score(100, hours_since_upload=1)
    assert new_score > old_score


def test_freshness_decay_very_popular_old_can_beat_unpopular_new():
    old_viral = _trending_score(10_000, hours_since_upload=72)
    new_niche = _trending_score(1, hours_since_upload=0)
    assert old_viral > new_niche


def test_freshness_decay_zero_hours_equals_popularity():
    """At upload time, score should equal raw popularity."""
    assert abs(_trending_score(50, 0) - 50.0) < 1e-9


def test_freshness_decay_is_monotonically_decreasing():
    """As time passes, score must decrease monotonically."""
    prev = _trending_score(100, 0)
    for h in [1, 6, 24, 48, 168]:
        curr = _trending_score(100, h)
        assert curr < prev
        prev = curr


# ── C. Dynamic hashtag weight ─────────────────────────────────────────────────

def _hashtag_weight(num_hashtags: int) -> float:
    """Mirrors model.py: 0.3 + 0.1 * min(num_hashtags, 5)"""
    return 0.3 + 0.1 * min(num_hashtags, 5)


def test_hashtag_weight_zero_hashtags():
    assert abs(_hashtag_weight(0) - 0.3) < 1e-9


def test_hashtag_weight_five_hashtags():
    assert abs(_hashtag_weight(5) - 0.8) < 1e-9


def test_hashtag_weight_caps_at_five():
    assert _hashtag_weight(5) == _hashtag_weight(10) == _hashtag_weight(100)


def test_hashtag_weight_is_monotonically_increasing_up_to_cap():
    weights = [_hashtag_weight(n) for n in range(6)]
    assert weights == sorted(weights)


# ── D. MMR diversity ──────────────────────────────────────────────────────────

def _mmr_rerank(
    candidates: list[tuple[str, float]],
    embeddings: dict[str, list[float]],
    lam: float = 0.7,
    top_k: int | None = None,
) -> list[str]:
    """
    Maximal Marginal Relevance (MMR) reranking.
    lam=1.0 → pure relevance, lam=0.0 → pure diversity.
    """
    import math

    def cosine(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x ** 2 for x in a))
        nb = math.sqrt(sum(x ** 2 for x in b))
        return dot / (na * nb + 1e-9)

    top_k = top_k or len(candidates)
    remaining = list(candidates)
    selected: list[str] = []

    while remaining and len(selected) < top_k:
        best_pid, best_score = None, float("-inf")
        for pid, rel_score in remaining:
            if not selected:
                mmr = lam * rel_score
            else:
                max_sim = max(cosine(embeddings[pid], embeddings[s]) for s in selected)
                mmr = lam * rel_score - (1.0 - lam) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_pid = pid
        selected.append(best_pid)
        remaining = [(p, s) for p, s in remaining if p != best_pid]

    return selected


def test_mmr_pure_relevance_preserves_order():
    """With lam=1.0 (pure relevance), MMR should follow the relevance ranking."""
    candidates = [("a", 1.0), ("b", 0.8), ("c", 0.5)]
    embeddings = {
        "a": [1.0, 0.0],
        "b": [0.9, 0.1],
        "c": [0.0, 1.0],
    }
    result = _mmr_rerank(candidates, embeddings, lam=1.0)
    assert result == ["a", "b", "c"]


def test_mmr_promotes_diverse_item():
    """With lam=0.7, a slightly lower relevance item that is very different from
    the top item should be promoted above a near-duplicate of the top item."""
    # "a" and "b" are nearly identical; "c" is orthogonal
    candidates = [("a", 1.0), ("b", 0.95), ("c", 0.8)]
    embeddings = {
        "a": [1.0, 0.0],
        "b": [0.99, 0.01],  # near-duplicate of a
        "c": [0.0, 1.0],    # diverse
    }
    result = _mmr_rerank(candidates, embeddings, lam=0.7, top_k=3)
    # "c" (diverse) should appear before "b" (duplicate)
    assert result.index("c") < result.index("b")


def test_mmr_returns_correct_length():
    candidates = [("a", 1.0), ("b", 0.8), ("c", 0.6), ("d", 0.4)]
    embeddings = {p: [1.0, 0.0] for p in ["a", "b", "c", "d"]}
    result = _mmr_rerank(candidates, embeddings, top_k=2)
    assert len(result) == 2
