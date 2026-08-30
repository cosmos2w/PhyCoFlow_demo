"""LoRA multi-adapter helpers for RAM fine-tuning PointCloudFFM models."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiAdapterLinear(nn.Module):
    """
    Wrap an ``nn.Linear`` with several LoRA adapters.

    ``active_adapter=None`` is the frozen base/reference path.  The ``default``
    adapter is trainable; lagged RAM adapters are updated only by copy/EMA.
    """

    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: float,
        adapter_names: Sequence[str] = ("default", "old", "evaluation"),
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"LoRA rank must be positive, got {rank}")
        self.in_features = int(linear.in_features)
        self.out_features = int(linear.out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.adapter_names = tuple(str(name) for name in adapter_names)
        self.active_adapter: Optional[str] = None

        self.base_weight = nn.Parameter(linear.weight.detach().clone(), requires_grad=False)
        if linear.bias is None:
            self.register_parameter("base_bias", None)
        else:
            self.base_bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in self.adapter_names:
            a = nn.Parameter(
                torch.empty(
                    self.rank,
                    self.in_features,
                    dtype=linear.weight.dtype,
                    device=linear.weight.device,
                )
            )
            b = nn.Parameter(
                torch.zeros(
                    self.out_features,
                    self.rank,
                    dtype=linear.weight.dtype,
                    device=linear.weight.device,
                )
            )
            nn.init.kaiming_uniform_(a, a=5**0.5)
            if name != "default":
                a.requires_grad_(False)
                b.requires_grad_(False)
            self.lora_A[name] = a
            self.lora_B[name] = b

    @property
    def weight(self) -> torch.Tensor:
        if self.active_adapter is None:
            return self.base_weight
        return self.merge_weight(self.active_adapter)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.base_weight, self.base_bias)
        if self.active_adapter is None:
            return y
        if self.active_adapter not in self.lora_A:
            raise ValueError(f"Unknown LoRA adapter: {self.active_adapter!r}")
        a = self.lora_A[self.active_adapter]
        b = self.lora_B[self.active_adapter]
        delta = F.linear(F.linear(x, a, None), b, None) * self.scaling
        return y + delta

    def set_active_adapter(self, name: Optional[str]) -> None:
        if name is not None and name not in self.lora_A:
            raise ValueError(f"Unknown LoRA adapter {name!r}; choices={list(self.lora_A.keys())}")
        self.active_adapter = name

    def get_active_adapter(self) -> Optional[str]:
        return self.active_adapter

    def adapter_state_dict(self, adapter_name: str) -> dict:
        if adapter_name not in self.lora_A:
            raise ValueError(f"Unknown LoRA adapter: {adapter_name!r}")
        return {
            "A": self.lora_A[adapter_name].detach().cpu().clone(),
            "B": self.lora_B[adapter_name].detach().cpu().clone(),
        }

    def load_adapter_state_dict(self, adapter_name: str, state: dict) -> None:
        if adapter_name not in self.lora_A:
            raise ValueError(f"Unknown LoRA adapter: {adapter_name!r}")
        with torch.no_grad():
            self.lora_A[adapter_name].copy_(state["A"].to(self.lora_A[adapter_name].device))
            self.lora_B[adapter_name].copy_(state["B"].to(self.lora_B[adapter_name].device))

    def merge_weight(self, adapter_name: str) -> torch.Tensor:
        if adapter_name not in self.lora_A:
            raise ValueError(f"Unknown LoRA adapter: {adapter_name!r}")
        delta_w = self.lora_B[adapter_name] @ self.lora_A[adapter_name]
        return self.base_weight + self.scaling * delta_w

    def merge_bias(self, adapter_name: str) -> Optional[torch.Tensor]:
        if adapter_name not in self.lora_A:
            raise ValueError(f"Unknown LoRA adapter: {adapter_name!r}")
        return self.base_bias


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _get_submodule(root: nn.Module, path: str) -> Optional[nn.Module]:
    if not path:
        return root
    obj: nn.Module = root
    for name in path.split("."):
        if not hasattr(obj, name):
            return None
        obj = getattr(obj, name)
        if not isinstance(obj, nn.Module):
            return None
    return obj


def _linear_paths_under(root: nn.Module, prefix: str) -> list[str]:
    module = _get_submodule(root, prefix)
    if module is None:
        return []
    out = []
    for local_name, child in module.named_modules():
        if local_name == "":
            continue
        if isinstance(child, MultiAdapterLinear):
            continue
        if isinstance(child, nn.Linear):
            out.append(f"{prefix}.{local_name}")
    return out


def _replace_module(root: nn.Module, path: str, module: nn.Module) -> None:
    parent_path, name = path.rsplit(".", 1)
    parent = _get_submodule(root, parent_path)
    if parent is None:
        raise KeyError(f"Cannot find parent module for path {path!r}")
    setattr(parent, name, module)


def inject_lora_adapters(
    model: nn.Module,
    scope: str,
    rank: int,
    alpha: float,
    adapter_names: Sequence[str] = ("default", "old", "evaluation"),
) -> list[str]:
    """Inject multi-adapter LoRA layers into supported GL_rbf PointCloudFFM scopes."""
    host = _unwrap_model(model)
    if not hasattr(host, "model") or str(getattr(host.model, "gather_mode", "")) not in ("rbf", "topk_rbf_glres"):
        raise NotImplementedError("LoRA RAM injection is supported only for GL_rbf PointCloudFFM models.")

    scope = str(scope).strip().lower()
    if scope == "head_glres":
        target_roots = [
            "model.head",
            "model.coarse_head",
            "model.coarse_film",
            "model.query_readout_in",
            "model.query_readout_out",
            "model.sensor_importance",
            "model.query_latent_readout",
        ]
        candidate_paths: list[str] = []
        for root_path in target_roots:
            module = _get_submodule(host, root_path)
            if module is None:
                continue
            if isinstance(module, nn.Linear):
                candidate_paths.append(root_path)
            candidate_paths.extend(_linear_paths_under(host, root_path))
    elif scope == "all_linear_glrbf":
        candidate_paths = [
            f"model.{name}"
            for name, module in host.model.named_modules()
            if name and isinstance(module, nn.Linear) and not isinstance(module, MultiAdapterLinear)
        ]
    else:
        raise ValueError("LoRA scope must be 'head_glres' or 'all_linear_glrbf'.")

    wrapped: list[str] = []
    seen = set()
    for path in candidate_paths:
        if path in seen:
            continue
        seen.add(path)
        module = _get_submodule(host, path)
        if module is None or isinstance(module, MultiAdapterLinear):
            continue
        if not isinstance(module, nn.Linear):
            continue
        _replace_module(host, path, MultiAdapterLinear(module, rank=rank, alpha=alpha, adapter_names=adapter_names))
        wrapped.append(path)

    print(f"[*] LoRA adapters injected ({scope}): {wrapped}")
    if not wrapped:
        raise RuntimeError(f"No nn.Linear modules were wrapped for LoRA scope={scope!r}.")
    return wrapped


def _iter_lora_modules(model: nn.Module):
    for name, module in _unwrap_model(model).named_modules():
        if isinstance(module, MultiAdapterLinear):
            yield name, module


def set_active_adapter(model: nn.Module, adapter_name: Optional[str]) -> None:
    for _, module in _iter_lora_modules(model):
        module.set_active_adapter(adapter_name)


def get_active_adapter(model: nn.Module) -> dict[str, Optional[str]]:
    return {name: module.get_active_adapter() for name, module in _iter_lora_modules(model)}


@contextmanager
def use_adapter(model: nn.Module, adapter_name: Optional[str]):
    previous = get_active_adapter(model)
    set_active_adapter(model, adapter_name)
    try:
        yield model
    finally:
        for name, module in _iter_lora_modules(model):
            module.set_active_adapter(previous.get(name))


def lora_trainable_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for _, module in _iter_lora_modules(model):
        params.extend([module.lora_A["default"], module.lora_B["default"]])
    params = [p for p in params if p.requires_grad]
    trainable_count = sum(p.numel() for p in params)
    total_count = sum(p.numel() for p in _unwrap_model(model).parameters())
    print(f"[*] LoRA trainable parameters: {trainable_count:,} / {total_count:,} ({trainable_count / max(total_count, 1):.2%})")
    if not params:
        raise RuntimeError("No trainable LoRA default-adapter parameters found.")
    return params


@torch.no_grad()
def sync_lora_adapter(model: nn.Module, source: str = "default", target: str = "old") -> None:
    for _, module in _iter_lora_modules(model):
        module.lora_A[target].copy_(module.lora_A[source].detach())
        module.lora_B[target].copy_(module.lora_B[source].detach())


@torch.no_grad()
def ema_lora_adapter(model: nn.Module, source: str = "default", target: str = "old", decay: float = 0.9) -> None:
    decay = float(decay)
    for _, module in _iter_lora_modules(model):
        module.lora_A[target].mul_(decay).add_(module.lora_A[source].detach(), alpha=1.0 - decay)
        module.lora_B[target].mul_(decay).add_(module.lora_B[source].detach(), alpha=1.0 - decay)


def collect_lora_state(model: nn.Module) -> dict:
    modules = {}
    for name, module in _iter_lora_modules(model):
        modules[name] = {
            "rank": module.rank,
            "alpha": module.alpha,
            "active_adapter": module.get_active_adapter(),
            "base_weight": module.base_weight.detach().cpu().clone(),
            "base_bias": None if module.base_bias is None else module.base_bias.detach().cpu().clone(),
            "adapters": {
                adapter: module.adapter_state_dict(adapter)
                for adapter in module.adapter_names
            },
        }
    return {
        "format": "phycoflow_multi_adapter_lora_v1",
        "modules": modules,
    }


def load_lora_state(model: nn.Module, state: dict) -> None:
    modules = state.get("modules", {})
    for name, module in _iter_lora_modules(model):
        payload = modules.get(name)
        if payload is None:
            continue
        with torch.no_grad():
            module.base_weight.copy_(payload["base_weight"].to(module.base_weight.device))
            if module.base_bias is not None and payload.get("base_bias") is not None:
                module.base_bias.copy_(payload["base_bias"].to(module.base_bias.device))
        for adapter, adapter_state in payload.get("adapters", {}).items():
            if adapter in module.lora_A:
                module.load_adapter_state_dict(adapter, adapter_state)
        module.set_active_adapter(payload.get("active_adapter", None))


def export_merged_lora_state_dict(
    lora_model: nn.Module,
    source_cfg: dict,
    dataset,
    adapter: str = "evaluation",
    device: torch.device | str = "cpu",
) -> dict:
    """
    Export a normal non-LoRA state_dict with ``adapter`` merged into Linear weights.
    """
    from model_finetune import build_ffm_model_from_config

    host = _unwrap_model(lora_model)
    clean_model = build_ffm_model_from_config(source_cfg, dataset).to(device)
    clean_state = clean_model.state_dict()
    lora_state = host.state_dict()

    for key in list(clean_state.keys()):
        if key in lora_state:
            clean_state[key].copy_(lora_state[key].detach().to(clean_state[key].device))

    clean_model.load_state_dict(clean_state, strict=False)
    clean_modules = dict(clean_model.named_modules())
    for path, module in _iter_lora_modules(host):
        clean_linear = clean_modules.get(path)
        if not isinstance(clean_linear, nn.Linear):
            raise RuntimeError(f"Clean model path {path!r} is not nn.Linear; got {type(clean_linear)}")
        with torch.no_grad():
            clean_linear.weight.copy_(module.merge_weight(adapter).detach().to(clean_linear.weight.device))
            if clean_linear.bias is not None and module.merge_bias(adapter) is not None:
                clean_linear.bias.copy_(module.merge_bias(adapter).detach().to(clean_linear.bias.device))

    return {key: value.detach().cpu().clone() for key, value in clean_model.state_dict().items()}
