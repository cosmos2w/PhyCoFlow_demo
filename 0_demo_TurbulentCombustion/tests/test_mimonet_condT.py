"""Information-budget and physical-metric checks for the Cond_T adapter."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from train_mimonet_condT import physical_relative_l2, sample_batch  # noqa: E402


def test_non_temperature_fields_cannot_enter_mimonet_branches():
    cfg = {"cond_fields": [2], "n_obs_min_list": [2], "n_obs_max_list": [4],
           "n_query_points": 5}
    coords = torch.rand(12, 3, generator=torch.Generator().manual_seed(17))
    fields = torch.randn(2, 12, 5, generator=torch.Generator().manual_seed(18))
    changed = fields.clone()
    changed[..., [0, 1, 3, 4]] += 1000.0
    torch.manual_seed(19)
    branches_a, query_a, target_a, mask_a = sample_batch(fields, coords, cfg)
    torch.manual_seed(19)
    branches_b, query_b, target_b, mask_b = sample_batch(changed, coords, cfg)
    assert all(torch.equal(a, b) for a, b in zip(branches_a, branches_b))
    assert torch.equal(query_a, query_b)
    assert torch.equal(mask_a, mask_b)
    assert torch.equal(target_a[..., 2], target_b[..., 2])
    assert not torch.equal(target_a[..., 0], target_b[..., 0])


def test_physical_relative_l2_matches_a0_definition():
    truth = torch.tensor([[[1., 2., 3., 4., 5.], [2., 3., 4., 5., 6.]]])
    prediction = truth + 0.25
    mean = torch.tensor([10., 20., 30., 40., 50.])
    std = torch.tensor([1., 2., 3., 4., 5.])
    result = physical_relative_l2(truth, prediction, mean, std)[0].numpy()
    physical_truth = (truth.numpy() * std.numpy() + mean.numpy())[0]
    physical_pred = (prediction.numpy() * std.numpy() + mean.numpy())[0]
    expected = [np.linalg.norm(physical_pred[:, i] - physical_truth[:, i]) /
                (np.linalg.norm(physical_truth[:, i]) + 1e-12) for i in range(5)]
    np.testing.assert_allclose(result, expected, atol=1e-8, rtol=1e-8)
