from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from prepare_synthetic_ave import long_tail_targets
from multimodal.training.joint import _scope_alignment
from multimodal.contribution.fcd import InfoNCEFCD


class UniformFCD:
    def distribution(self, information: torch.Tensor) -> torch.Tensor:
        scores = torch.nn.functional.softplus(information.mean(dim=0))
        return scores / scores.sum()


def test_long_tail_targets_are_monotone_capped_and_hit_requested_ratio():
    natural = {i: 152 - 4 * i for i in range(28)}
    for factor in (10, 20):
        targets = long_tail_targets(natural, factor)
        assert targets[0] / targets[27] == factor
        assert all(0 < targets[i] <= natural[i] for i in natural)
        assert all(targets[i] >= targets[i + 1] for i in range(27))


def test_reliability_granularity_uses_stable_local_else_global_target():
    targets = {
        "global": {"target": [0.5, 0.5], "stable": True},
        "classes": {
            "0": {"target": [0.9, 0.1], "stable": True, "fallback": None},
            "1": {"target": [0.5, 0.5], "stable": False, "fallback": "global"},
        },
        "groups": {},
    }
    information = torch.zeros((2, 2), requires_grad=True)
    loss = _scope_alignment(information, torch.tensor([0, 1]), torch.tensor([0, 0]),
                            targets, "reliability", 1e-8, UniformFCD())
    assert torch.isfinite(loss)
    loss.backward()
    assert information.grad is not None


def test_vectorized_info_nce_matches_anchorwise_queue_and_source_exclusions():
    torch.manual_seed(19)
    estimator = InfoNCEFCD(queue_size=8, exclude_same_source_negatives=True)
    past = [torch.randn(4, 5) for _ in range(3)]
    estimator.per_sample_information(*past, update_queue=True,
        sample_ids=["old-a", "old-b", "old-c", "old-d"],
        source_ids=["source-a", "source-b", "source-c", "source-d"])
    current = [torch.randn(3, 5, requires_grad=True) for _ in range(3)]
    sample_ids = ["new-a", "new-b", "new-c"]
    source_ids = ["source-a", "source-x", "source-c"]
    actual = estimator.per_sample_information(*current, sample_ids=sample_ids,
                                               source_ids=source_ids)

    fused, x1, x2 = [torch.nn.functional.normalize(value, dim=-1) for value in current]
    expected = []
    for i in range(len(sample_ids)):
        identity, source = sample_ids[i], source_ids[i]
        batch_keep = [j for j in range(len(sample_ids))
                      if j == i or sample_ids[j] != identity]
        batch_keep = [j for j in batch_keep if j == i or source_ids[j] != source]
        historical = [value for value, queue_id, queue_source in estimator._queue
                      if queue_id != identity and queue_source != source]
        candidates = torch.cat((fused[batch_keep], *historical), dim=0)
        bounds = []
        for modality in (x1, x2):
            scores = modality[i] @ candidates.T
            positive = (modality[i] * fused[i]).sum()
            bounds.append(torch.log(scores.new_tensor(float(len(candidates)))) + positive
                           - torch.logsumexp(scores, dim=-1))
        expected.append(torch.stack(bounds))
    assert torch.allclose(actual, torch.stack(expected), atol=1e-6, rtol=1e-6)
    actual.sum().backward()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() for value in current)
