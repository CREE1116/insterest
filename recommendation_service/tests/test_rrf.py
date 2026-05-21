"""
RRF (Reciprocal Rank Fusion) 로직 순수 Python 유닛 테스트.
torch / 모델 의존성 없음.
"""


def _rrf_merge(rank_lists: list[list[str]], k: int = 60, exclude: set[str] | None = None) -> list[tuple[str, float]]:
    exclude = exclude or set()
    scores: dict[str, float] = {}
    for r_list in rank_lists:
        for rank, pid in enumerate(r_list):
            if pid in exclude:
                continue
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def test_rrf_single_list_preserves_order():
    items = _rrf_merge([["a", "b", "c"]])
    ids = [i for i, _ in items]
    assert ids == ["a", "b", "c"]


def test_rrf_higher_rank_yields_higher_score():
    items = _rrf_merge([["first", "second"]])
    assert items[0][1] > items[1][1]


def test_rrf_fusion_promotes_common_item():
    # "shared" appears in both lists at rank 1; "only_a" only in first
    items = _rrf_merge([["only_a", "shared"], ["shared", "only_b"]])
    ids = [i for i, _ in items]
    assert ids[0] == "shared"


def test_rrf_exclude_filters_items():
    items = _rrf_merge([["a", "b", "c"]], exclude={"b"})
    ids = [i for i, _ in items]
    assert "b" not in ids
    assert "a" in ids


def test_rrf_empty_list_returns_empty():
    assert _rrf_merge([]) == []


def test_rrf_empty_sublists():
    assert _rrf_merge([[], []]) == []


def test_rrf_k_parameter_affects_scores():
    items_k60 = _rrf_merge([["x"]], k=60)
    items_k1 = _rrf_merge([["x"]], k=1)
    # smaller k → larger score
    assert items_k1[0][1] > items_k60[0][1]
