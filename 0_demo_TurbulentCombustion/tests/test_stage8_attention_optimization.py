from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model import (
    ConditionalPointHybridLocalGlobalRBFCQ,
    CrossAttentionBlock,
    PointCloudFFM,
)
from model_ema import ModelEMA


class _Prior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        return torch.randn(
            coords.shape[0], coords.shape[1], n_channels,
            device=coords.device, dtype=coords.dtype,
        )


def _backbone(
    execution: str,
    padding: str,
    *,
    buckets: tuple[int, ...] = (8, 12, 16),
) -> ConditionalPointHybridLocalGlobalRBFCQ:
    return ConditionalPointHybridLocalGlobalRBFCQ(
        n_fields=2,
        coord_dim=3,
        hidden_dim=16,
        cond_dim=8,
        field_embed_dim=4,
        latent_dim=16,
        num_latents=8,
        num_heads=4,
        num_latent_blocks=4,
        ff_mult=2,
        attn_dropout=0.0,
        mlp_dropout=0.0,
        gather_mode="topk_rbf_glres",
        gather_topk=4,
        gather_query_chunk_size=7,
        learnable_rbf_sigma=True,
        neighbor_backend="torch",
        use_fourier_pe=True,
        fourier_pe_num_bands=2,
        fourier_pe_max_freq=4.0,
        sensor_coord_encoding="fourier",
        latent_sensor_reinject=True,
        latent_reinject_every=1,
        condition_attention_execution=execution,
        sensor_attention_padding_mode=padding,
        sensor_attention_buckets=buckets,
        glres_scale_init=1.0e-2,
        cq_query_dim=8,
        cq_readout_mode="lowrank",
        cq_readout_rank=4,
        cq_readout_heads=2,
        cq_time_conditioning="sinusoidal_film",
        cq_time_embed_dim=8,
        cq_measurement_support_mode="rbf_value_support",
    )


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(812)
    batch, n_query, max_sensors = 4, 19, 16
    counts = torch.tensor([7, 8, 9, 15])
    mask = torch.arange(max_sensors).unsqueeze(0) < counts.unsqueeze(1)
    return {
        "t": torch.tensor([0.17, 0.31, 0.59, 0.83]),
        "x_t": torch.randn(batch, n_query, 2, generator=generator),
        "x1": torch.randn(batch, n_query, 2, generator=generator),
        "coords": torch.rand(batch, n_query, 3, generator=generator),
        "obs_coords": torch.rand(batch, max_sensors, 3, generator=generator),
        "obs_values": torch.randn(batch, max_sensors, 1, generator=generator),
        "obs_mask": mask.float(),
        "obs_field_ids": torch.arange(max_sensors).remainder(2).expand(batch, -1),
    }


def _grads(model: torch.nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def test_cross_attention_prepared_kv_forward_and_all_gradients_match() -> None:
    torch.manual_seed(31)
    legacy = CrossAttentionBlock(32, 4, ff_mult=2).double()
    cached = copy.deepcopy(legacy)
    q_a = torch.randn(3, 5, 32, dtype=torch.double, requires_grad=True)
    kv_a = torch.randn(3, 11, 32, dtype=torch.double, requires_grad=True)
    q_b = q_a.detach().clone().requires_grad_(True)
    kv_b = kv_a.detach().clone().requires_grad_(True)
    mask = torch.arange(11).unsqueeze(0) >= torch.tensor([11, 8, 6]).unsqueeze(1)

    out_a = legacy(q_a, kv_a, mask)
    prepared = cached.prepare_kv(kv_b, mask)
    out_b = cached.forward_prepared(q_b, prepared)
    torch.testing.assert_close(out_b, out_a, rtol=1e-12, atol=1e-12)

    out_a.square().mean().backward()
    out_b.square().mean().backward()
    torch.testing.assert_close(q_b.grad, q_a.grad, rtol=1e-11, atol=1e-12)
    torch.testing.assert_close(kv_b.grad, kv_a.grad, rtol=1e-11, atol=1e-12)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        legacy.named_parameters(), cached.named_parameters(), strict=True
    ):
        assert name_a == name_b
        torch.testing.assert_close(
            parameter_b.grad, parameter_a.grad, rtol=1e-11, atol=1e-12, msg=name_a
        )


def test_four_reinjections_share_one_kv_graph_and_match_legacy() -> None:
    torch.manual_seed(32)
    legacy = CrossAttentionBlock(16, 4, ff_mult=2).double()
    cached = copy.deepcopy(legacy)
    latent_a = torch.randn(2, 7, 16, dtype=torch.double, requires_grad=True)
    sensor_a = torch.randn(2, 13, 16, dtype=torch.double, requires_grad=True)
    latent_b = latent_a.detach().clone().requires_grad_(True)
    sensor_b = sensor_a.detach().clone().requires_grad_(True)
    mask = torch.arange(13).unsqueeze(0) >= torch.tensor([9, 13]).unsqueeze(1)

    for _ in range(4):
        latent_a = legacy(latent_a, sensor_a, mask)
    prepared = cached.prepare_kv(sensor_b, mask)
    for _ in range(4):
        latent_b = cached.forward_prepared(latent_b, prepared)

    torch.testing.assert_close(latent_b, latent_a, rtol=1e-11, atol=1e-12)
    latent_a.sum().backward()
    latent_b.sum().backward()
    assert legacy.kv_projection_calls == 4
    assert cached.kv_projection_calls == 1
    torch.testing.assert_close(sensor_b.grad, sensor_a.grad, rtol=1e-10, atol=1e-11)
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        legacy.named_parameters(), cached.named_parameters(), strict=True
    ):
        assert name_a == name_b
        torch.testing.assert_close(
            parameter_b.grad, parameter_a.grad, rtol=1e-10, atol=1e-11, msg=name_a
        )


@pytest.mark.parametrize(
    ("execution", "padding", "expected_kv_calls"),
    [("cached_kv", "full", 1), ("cached_kv", "static_buckets", 3)],
)
def test_whole_cq_context_output_gradients_and_adamw_update_match(
    execution: str,
    padding: str,
    expected_kv_calls: int,
) -> None:
    torch.manual_seed(33)
    oracle = _backbone("legacy_mha", "full")
    candidate = _backbone(execution, padding)
    assert oracle.state_dict().keys() == candidate.state_dict().keys()
    candidate.load_state_dict(oracle.state_dict(), strict=True)
    values = _inputs()
    optimizer_a = torch.optim.AdamW(oracle.parameters(), lr=2.0e-4, weight_decay=1.0e-6)
    optimizer_b = torch.optim.AdamW(candidate.parameters(), lr=2.0e-4, weight_decay=1.0e-6)

    oracle.input_cross_attn.reset_execution_counters()
    candidate.input_cross_attn.reset_execution_counters()
    context_a = oracle.prepare_condition_context(
        values["obs_coords"], values["obs_values"], values["obs_mask"], values["obs_field_ids"]
    )
    context_b = candidate.prepare_condition_context(
        values["obs_coords"], values["obs_values"], values["obs_mask"], values["obs_field_ids"]
    )
    for key in ("latents", "global_feat", "refined_sensor_feat", "global_q"):
        torch.testing.assert_close(context_b[key], context_a[key], rtol=2e-5, atol=2e-6)
    out_a = oracle.forward_query_chunk(
        values["t"], values["x_t"], values["coords"], context_a
    )
    out_b = candidate.forward_query_chunk(
        values["t"], values["x_t"], values["coords"], context_b
    )
    torch.testing.assert_close(out_b, out_a, rtol=3e-5, atol=3e-6)

    loss_a = out_a.square().mean()
    loss_b = out_b.square().mean()
    loss_a.backward()
    loss_b.backward()
    gradients_a = _grads(oracle)
    gradients_b = _grads(candidate)
    for name in gradients_a:
        assert (gradients_a[name] is None) == (gradients_b[name] is None), name
        if gradients_a[name] is not None:
            torch.testing.assert_close(
                gradients_b[name], gradients_a[name], rtol=3e-4, atol=3e-7, msg=name
            )
    optimizer_a.step()
    optimizer_b.step()
    for (name_a, parameter_a), (name_b, parameter_b) in zip(
        oracle.named_parameters(), candidate.named_parameters(), strict=True
    ):
        assert name_a == name_b
        torch.testing.assert_close(
            parameter_b, parameter_a, rtol=3e-5, atol=7e-6, msg=name_a
        )
    assert oracle.input_cross_attn.kv_projection_calls == 4
    assert candidate.input_cross_attn.kv_projection_calls == expected_kv_calls


def test_non_prefix_sensor_mask_safely_falls_back_to_full_padding() -> None:
    torch.manual_seed(34)
    full = _backbone("cached_kv", "full")
    bucketed = _backbone("cached_kv", "static_buckets")
    bucketed.load_state_dict(full.state_dict(), strict=True)
    values = _inputs()
    values["obs_mask"][0, 2] = 0
    values["obs_mask"][0, 10] = 1
    full.input_cross_attn.reset_execution_counters()
    bucketed.input_cross_attn.reset_execution_counters()
    forward_values = {key: value for key, value in values.items() if key != "x1"}
    out_a = full(**forward_values)
    out_b = bucketed(**forward_values)
    torch.testing.assert_close(out_b, out_a, rtol=0, atol=0)
    assert full.input_cross_attn.kv_projection_calls == 1
    assert bucketed.input_cross_attn.kv_projection_calls == 1


@pytest.mark.parametrize("solver", ["euler", "heun"])
def test_cached_bucketed_preserves_microbatch_ema_reconstruction_and_persistent_knn(
    solver: str,
) -> None:
    torch.manual_seed(35)
    oracle = PointCloudFFM(_backbone("legacy_mha", "full"), _Prior())
    candidate = PointCloudFFM(_backbone("cached_kv", "static_buckets"), _Prior())
    candidate.load_state_dict(oracle.state_dict(), strict=True)
    values = _inputs()
    training_values = {
        key: values[key]
        for key in (
            "x1", "coords", "obs_coords", "obs_values", "obs_mask", "obs_field_ids"
        )
    }
    training_values["obs_indices"] = torch.arange(16).expand(4, -1)
    optimizer_a = torch.optim.AdamW(oracle.parameters(), lr=2.0e-4)
    optimizer_b = torch.optim.AdamW(candidate.parameters(), lr=2.0e-4)
    ema_a = ModelEMA(oracle, decay=0.9)
    ema_b = ModelEMA(candidate, decay=0.9)

    torch.manual_seed(901)
    loss_a, _ = oracle.training_loss_microbatched(
        **training_values,
        query_microbatch_size=7,
        backward=True,
        reuse_condition_context=True,
    )
    torch.manual_seed(901)
    loss_b, _ = candidate.training_loss_microbatched(
        **training_values,
        query_microbatch_size=7,
        backward=True,
        reuse_condition_context=True,
    )
    torch.testing.assert_close(loss_b, loss_a, rtol=3e-5, atol=3e-6)
    optimizer_a.step()
    optimizer_b.step()
    ema_a.update(oracle)
    ema_b.update(candidate)
    for name in ema_a.shadow:
        torch.testing.assert_close(
            ema_b.shadow[name], ema_a.shadow[name], rtol=3e-5, atol=8e-6, msg=name
        )

    oracle.eval()
    candidate.eval()
    reconstruction_values = {key: value[:1] for key, value in values.items()}
    geometry = candidate.prepare_reconstruction_geometry_cache(
        coords=reconstruction_values["coords"],
        obs_coords=reconstruction_values["obs_coords"],
        obs_mask=reconstruction_values["obs_mask"],
        chunk_size=7,
    )
    calls = 0
    original = candidate.model._get_topk_neighbors

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    candidate.model._get_topk_neighbors = counted
    common = {
        "coords": reconstruction_values["coords"],
        "obs_coords": reconstruction_values["obs_coords"],
        "obs_values": reconstruction_values["obs_values"],
        "obs_mask": reconstruction_values["obs_mask"],
        "obs_field_ids": reconstruction_values["obs_field_ids"],
        "n_steps": 2,
        "ode_solver": solver,
        "obs_consistency_mode": "none",
        "reconstruction_execution_mode": "cached_streamed",
        "reconstruction_query_chunk_size": 7,
        "reconstruction_cache_level": "static_features",
    }
    torch.manual_seed(902)
    reconstruction_a = oracle.sample(**common)
    torch.manual_seed(902)
    reconstruction_b = candidate.sample(
        **common, reconstruction_geometry_cache=geometry,
    )
    torch.testing.assert_close(
        reconstruction_b, reconstruction_a, rtol=4e-5, atol=4e-6
    )
    assert calls == 0
