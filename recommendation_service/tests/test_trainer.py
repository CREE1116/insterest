"""
InfoNCE 손실 유닛 테스트.
torch가 없으면 전체 모듈 스킵.
"""
import math
import pytest

torch = pytest.importorskip("torch")


def _info_nce(query_vecs, item_vecs, temperature=0.07):
    import torch.nn as nn
    logits = torch.matmul(query_vecs, item_vecs.T) / temperature
    labels = torch.arange(query_vecs.size(0))
    return nn.CrossEntropyLoss()(logits, labels)


def test_info_nce_positive():
    B = 4
    q = torch.nn.functional.normalize(torch.randn(B, 512), p=2, dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, 512), p=2, dim=-1)
    loss = _info_nce(q, k)
    assert loss.item() > 0


def test_info_nce_perfect_alignment_is_low():
    B = 4
    vecs = torch.nn.functional.normalize(torch.randn(B, 512), p=2, dim=-1)
    loss = _info_nce(vecs, vecs.clone())
    random_baseline = math.log(B)
    # Perfect alignment should be well below random baseline
    assert loss.item() < random_baseline


def test_info_nce_random_baseline():
    """Random embeddings → loss ≈ ln(B)."""
    B = 8
    losses = []
    for _ in range(10):
        q = torch.nn.functional.normalize(torch.randn(B, 512), p=2, dim=-1)
        k = torch.nn.functional.normalize(torch.randn(B, 512), p=2, dim=-1)
        losses.append(_info_nce(q, k).item())
    avg = sum(losses) / len(losses)
    # Should be within 40% of ln(B)
    assert abs(avg - math.log(B)) < math.log(B) * 0.4


def test_gradient_flows():
    B = 4
    q_raw = torch.randn(B, 512, requires_grad=True)
    q = torch.nn.functional.normalize(q_raw, p=2, dim=-1)
    k = torch.nn.functional.normalize(torch.randn(B, 512), p=2, dim=-1)
    loss = _info_nce(q, k)
    loss.backward()
    assert q_raw.grad is not None
