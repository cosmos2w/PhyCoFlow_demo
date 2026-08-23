"""Checkpointable exponential moving average for PointCloudFFM training."""

from collections import OrderedDict
from contextlib import contextmanager
from typing import Dict, Iterator, Mapping

import torch
import torch.nn as nn


class ModelEMA:
    """Track every model parameter and buffer without changing the live module."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")
        self.decay = float(decay)
        self.num_updates = 0
        self.shadow = self._clone_model_state(model)

    @staticmethod
    def _clone_model_state(model: nn.Module) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict(
            (name, value.detach().clone()) for name, value in model.state_dict().items()
        )

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        live = model.state_dict()
        if live.keys() != self.shadow.keys():
            raise RuntimeError("EMA/model state keys differ; architecture changed after EMA creation.")
        for name, value in live.items():
            target = self.shadow[name]
            source = value.detach().to(device=target.device)
            if target.is_floating_point() or target.is_complex():
                target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                target.copy_(source)
        self.num_updates += 1

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": OrderedDict(
                (name, value.detach().clone()) for name, value in self.shadow.items()
            ),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise ValueError("EMA checkpoint is missing a shadow state mapping.")
        if shadow.keys() != self.shadow.keys():
            missing = sorted(set(self.shadow) - set(shadow))
            unexpected = sorted(set(shadow) - set(self.shadow))
            raise RuntimeError(
                f"EMA checkpoint keys differ: missing={missing}, unexpected={unexpected}."
            )
        self.decay = float(state.get("decay", self.decay))
        self.num_updates = int(state.get("num_updates", 0))
        for name, value in shadow.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"EMA value for {name!r} is not a tensor.")
            self.shadow[name].copy_(
                value.to(device=self.shadow[name].device, dtype=self.shadow[name].dtype)
            )

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Iterator[None]:
        """Temporarily evaluate with EMA weights, then restore the live weights."""
        live = self._clone_model_state(model)
        self.copy_to(model)
        try:
            yield
        finally:
            model.load_state_dict(live, strict=True)
