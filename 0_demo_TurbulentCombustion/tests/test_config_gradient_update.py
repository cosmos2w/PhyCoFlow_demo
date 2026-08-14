from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from direct_coherence_loss import apply_two_objective_update  # noqa: E402


class _VectorModel(torch.nn.Module):
    def __init__(self, values: tuple[float, ...], device: torch.device) -> None:
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, device=device))


class _DisjointModel(torch.nn.Module):
    def __init__(self, device: torch.device) -> None:
        super().__init__()
        self.data_value = torch.nn.Parameter(torch.tensor(1.0, device=device))
        self.coherence_value = torch.nn.Parameter(torch.tensor(-1.0, device=device))


def _devices() -> list[torch.device]:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("device", _devices(), ids=str)
def test_config_aligned_step_updates_parameters_and_optimizer(device: torch.device) -> None:
    model = _VectorModel((1.0, -2.0), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    before = model.values.detach().clone()

    data_loss = model.values.square().sum()
    coherence_loss = (2.0 * model.values).square().sum()
    info = apply_two_objective_update(
        model=model,
        optimizer=optimizer,
        data_loss=data_loss,
        coherence_loss=coherence_loss,
        mode="config",
        data_weight=0.1,
        coherence_weight=1.0,
        grad_clip_norm=1.0,
    )

    assert info["config_update_mode"] == "weighted_sum_aligned"
    assert not torch.equal(model.values.detach(), before)
    assert model.values in optimizer.state
    assert float(optimizer.state[model.values]["step"]) == 1.0


@pytest.mark.parametrize("device", _devices(), ids=str)
def test_config_conflicting_step_updates_parameters_and_optimizer(device: torch.device) -> None:
    model = _VectorModel((0.0, 0.0), device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    before = model.values.detach().clone()

    data_loss = (model.values[0] - 1.0).square() + model.values[1].square()
    coherence_loss = (model.values[0] + 1.0).square() + (model.values[1] - 1.0).square()
    info = apply_two_objective_update(
        model=model,
        optimizer=optimizer,
        data_loss=data_loss,
        coherence_loss=coherence_loss,
        mode="config",
        data_weight=1.0,
        coherence_weight=1.0,
        grad_clip_norm=1.0,
    )

    assert info["gradient_conflict"]
    assert info["config_update_mode"] == "config"
    assert not torch.equal(model.values.detach(), before)
    assert model.values in optimizer.state
    assert float(optimizer.state[model.values]["step"]) == 1.0


@pytest.mark.parametrize("device", _devices(), ids=str)
def test_config_supports_objectives_with_disjoint_parameters(device: torch.device) -> None:
    model = _DisjointModel(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-2)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())

    info = apply_two_objective_update(
        model=model,
        optimizer=optimizer,
        data_loss=model.data_value.square(),
        coherence_loss=model.coherence_value.square(),
        mode="config",
        data_weight=1.0,
        coherence_weight=1.0,
        grad_clip_norm=1.0,
    )

    assert info["config_update_mode"] == "weighted_sum_aligned"
    for parameter, initial_value in zip(model.parameters(), before):
        assert not torch.equal(parameter.detach(), initial_value)
        assert parameter in optimizer.state
