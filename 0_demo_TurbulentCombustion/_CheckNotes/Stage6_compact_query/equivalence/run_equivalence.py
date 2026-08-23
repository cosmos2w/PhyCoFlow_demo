#!/usr/bin/env python3
"""Emit quantitative CQ reconstruction/training equivalence evidence."""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from Model import ConditionalPointHybridLocalGlobalRBFCQ, PointCloudFFM


class RecordingRFFPrior(torch.nn.Module):
    def __init__(self, coord_dim: int = 3, n_features: int = 11):
        super().__init__()
        self.register_buffer(
            "omega",
            torch.linspace(-2.0, 2.0, coord_dim * n_features).reshape(
                coord_dim, n_features
            ),
        )
        self.register_buffer("phase", torch.linspace(0.0, 2.0 * math.pi, n_features))
        self.calls: list[int] = []

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        self.calls.append(int(coords.shape[1]))
        phi = math.sqrt(2.0 / self.omega.shape[1]) * torch.cos(
            coords @ self.omega + self.phase
        )
        weights = torch.randn(
            coords.shape[0], n_channels, self.omega.shape[1],
            device=coords.device, dtype=coords.dtype,
        )
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def model(mode: str) -> PointCloudFFM:
    backbone = ConditionalPointHybridLocalGlobalRBFCQ(
        n_fields=2,
        hidden_dim=16,
        cond_dim=8,
        field_embed_dim=4,
        latent_dim=16,
        num_latents=8,
        num_heads=4,
        num_latent_blocks=1,
        ff_mult=2,
        gather_mode="topk_rbf_glres",
        gather_topk=3,
        gather_query_chunk_size=4,
        learnable_rbf_sigma=True,
        neighbor_backend="torch",
        use_fourier_pe=True,
        fourier_pe_num_bands=2,
        fourier_pe_max_freq=4.0,
        sensor_coord_encoding="fourier",
        latent_sensor_reinject=True,
        glres_scale_init=1.0e-2,
        cq_query_dim=8,
        cq_readout_mode=mode,
        cq_readout_rank=4,
        cq_readout_heads=2,
    )
    return PointCloudFFM(backbone, RecordingRFFPrior())


def inputs(batch: int = 2, n_query: int = 31) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(313)
    coords = torch.rand(batch, n_query, 3, generator=generator)
    indices = torch.stack(
        [torch.tensor([0, 3, 7, 10, n_query - 2]) for _ in range(batch)]
    )
    return {
        "x1": torch.randn(batch, n_query, 2, generator=generator),
        "coords": coords,
        "obs_coords": torch.stack([coords[i, indices[i]] for i in range(batch)]),
        "obs_values": torch.randn(batch, 5, 1, generator=generator),
        "obs_mask": torch.ones(batch, 5),
        "obs_field_ids": torch.tensor([[0, 1, 0, 1, 0]]).expand(batch, -1).clone(),
        "obs_indices": indices,
    }


def max_tensor_difference(
    left: dict[str, torch.Tensor | None],
    right: dict[str, torch.Tensor | None],
) -> tuple[float, float, str]:
    max_abs = 0.0
    max_rel = 0.0
    max_name = ""
    for name, lhs in left.items():
        rhs = right[name]
        if lhs is None or rhs is None:
            if lhs is not rhs:
                raise AssertionError(f"None mismatch for {name}")
            continue
        difference = (lhs - rhs).abs()
        current_abs = float(difference.max())
        denominator = torch.maximum(lhs.abs(), rhs.abs()).clamp_min(1.0e-12)
        current_rel = float((difference / denominator).max())
        if current_abs > max_abs:
            max_abs = current_abs
            max_name = name
        max_rel = max(max_rel, current_rel)
    return max_abs, max_rel, max_name


def reconstruction_evidence(mode: str) -> list[dict]:
    result = []
    scenarios = [
        ("euler", 1, "none"),
        ("euler", 2, "default_hard"),
        ("euler", 4, "endpoint_smooth"),
        ("heun", 2, "endpoint_smooth"),
    ]
    for index, (solver, nfe, consistency) in enumerate(scenarios):
        torch.manual_seed(100 + index)
        current = model(mode).eval()
        value = inputs(batch=1, n_query=13)
        kwargs = {
            key: value[key]
            for key in (
                "coords", "obs_coords", "obs_values", "obs_mask",
                "obs_field_ids",
            )
        }
        kwargs["clamp_indices"] = value["obs_indices"]
        kwargs.update(
            n_steps=nfe,
            ode_solver=solver,
            obs_consistency_mode=consistency,
        )
        torch.manual_seed(900 + index)
        full = current.sample(
            **kwargs, reconstruction_execution_mode="legacy_full",
        )
        torch.manual_seed(900 + index)
        cached = current.sample(
            **kwargs,
            reconstruction_execution_mode="cached_streamed",
            reconstruction_query_chunk_size=5,
            reconstruction_cache_level="static_features",
        )
        difference = cached - full
        result.append({
            "solver": solver,
            "nfe": nfe,
            "obs_consistency_mode": consistency,
            "max_abs_difference": float(difference.abs().max()),
            "mean_abs_difference": float(difference.abs().mean()),
            "relative_l2_difference": float(
                torch.linalg.vector_norm(difference)
                / torch.linalg.vector_norm(full).clamp_min(1.0e-12)
            ),
        })
    return result


def training_evidence(mode: str) -> dict:
    torch.manual_seed(500)
    full = model(mode)
    micro = copy.deepcopy(full)
    value = inputs()
    optimizer_full = torch.optim.AdamW(full.parameters(), lr=3.0e-4, weight_decay=1.0e-6)
    optimizer_micro = torch.optim.AdamW(micro.parameters(), lr=3.0e-4, weight_decay=1.0e-6)

    optimizer_full.zero_grad(set_to_none=True)
    torch.manual_seed(777)
    loss_full, _ = full.training_loss(**value)
    loss_full.backward()
    gradients_full = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in full.named_parameters()
    }
    optimizer_full.step()

    optimizer_micro.zero_grad(set_to_none=True)
    torch.manual_seed(777)
    loss_micro, metrics = micro.training_loss_microbatched(
        **value,
        query_microbatch_size=7,
        backward=True,
        reuse_condition_context=True,
    )
    gradients_micro = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in micro.named_parameters()
    }
    optimizer_micro.step()

    grad_abs, grad_rel, grad_name = max_tensor_difference(
        gradients_full, gradients_micro
    )
    params_full = dict(full.named_parameters())
    params_micro = dict(micro.named_parameters())
    update_abs, update_rel, update_name = max_tensor_difference(
        {name: value.detach() for name, value in params_full.items()},
        {name: params_micro[name].detach() for name in params_full},
    )
    return {
        "n_query": int(value["coords"].shape[1]),
        "query_microbatch_size": 7,
        "query_microbatches": int(metrics["query_microbatches"]),
        "prior_calls_monolithic": full.prior.calls,
        "prior_calls_microbatched": micro.prior.calls,
        "loss_monolithic": float(loss_full.detach()),
        "loss_microbatched": float(loss_micro),
        "loss_abs_difference": float((loss_micro - loss_full.detach()).abs()),
        "max_gradient_abs_difference": grad_abs,
        "max_gradient_relative_difference": grad_rel,
        "max_gradient_abs_parameter": grad_name,
        "max_update_abs_difference": update_abs,
        "max_update_relative_difference": update_rel,
        "max_update_abs_parameter": update_name,
        "rbf_sigma_gradient_present": gradients_micro["model.log_rbf_sigma"] is not None,
    }


def condition_cache_evidence(mode: str) -> dict:
    torch.manual_seed(810)
    current = model(mode).eval()
    original = current.model.prepare_condition_context
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    current.model.prepare_condition_context = counted
    value = inputs(batch=1, n_query=13)
    current.sample(
        coords=value["coords"],
        obs_coords=value["obs_coords"],
        obs_values=value["obs_values"],
        obs_mask=value["obs_mask"],
        obs_field_ids=value["obs_field_ids"],
        n_steps=4,
        ode_solver="heun",
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level="static_features",
    )
    return {
        "solver": "heun",
        "nfe": 4,
        "query_chunk_size": 5,
        "condition_encoding_calls": calls,
        "expected_calls": 1,
    }


def kv_cache_evidence() -> dict:
    torch.manual_seed(800)
    current = model("lowrank").eval()
    original = current.model.cq_latent_readout.project_latents
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    current.model.cq_latent_readout.project_latents = counted
    value = inputs(batch=1, n_query=13)
    current.sample(
        coords=value["coords"],
        obs_coords=value["obs_coords"],
        obs_values=value["obs_values"],
        obs_mask=value["obs_mask"],
        obs_field_ids=value["obs_field_ids"],
        n_steps=4,
        ode_solver="heun",
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level="static_features",
    )
    return {
        "solver": "heun",
        "nfe": 4,
        "query_chunk_size": 5,
        "latent_kv_projection_calls": calls,
        "expected_calls": 1,
    }


def main() -> None:
    output = Path(__file__).resolve().parent / "gate_b_equivalence.json"
    result = {
        "dtype": "float32",
        "device": "cpu",
        "gather_mode": "topk_rbf_glres",
        "nondivisible_training_query_count": 31,
        "variants": {},
        "condition_context_reuse": {
            mode: condition_cache_evidence(mode)
            for mode in ("full", "lowrank")
        },
        "cq_lr_cache_reuse": kv_cache_evidence(),
    }
    for mode in ("full", "lowrank"):
        result["variants"][mode] = {
            "reconstruction": reconstruction_evidence(mode),
            "training": training_evidence(mode),
        }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
