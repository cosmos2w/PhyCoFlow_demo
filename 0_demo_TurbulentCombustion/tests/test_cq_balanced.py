from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model import ConditionalPointHybridLocalGlobalRBFCQ, PointCloudFFM


class _RecordingRFFPrior(torch.nn.Module):
    def __init__(self, coord_dim: int = 3, n_features: int = 11):
        super().__init__()
        self.register_buffer(
            "omega",
            torch.linspace(-2.0, 2.0, coord_dim * n_features).reshape(
                coord_dim, n_features
            ),
        )
        self.register_buffer("phase", torch.linspace(0.0, 2.0 * math.pi, n_features))

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = math.sqrt(2.0 / self.omega.shape[1]) * torch.cos(
            coords @ self.omega + self.phase
        )
        weights = torch.randn(
            coords.shape[0], n_channels, self.omega.shape[1],
            device=coords.device, dtype=coords.dtype,
        )
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def _backbone(fusion_mode: str = "structured_concat"):
    return ConditionalPointHybridLocalGlobalRBFCQ(
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
        cq_readout_mode="full",
        cq_fusion_mode=fusion_mode,
        cq_readout_rank=4,
        cq_readout_heads=2,
        cq_global_scale_init=1.0,
        cq_local_scale_init=1.0,
        cq_readout_scale_init=1.0e-2,
    )


def _model(fusion_mode: str = "structured_concat") -> PointCloudFFM:
    return PointCloudFFM(_backbone(fusion_mode), _RecordingRFFPrior())


def _inputs(batch: int = 1, n_query: int = 19):
    generator = torch.Generator().manual_seed(313)
    coords = torch.rand(batch, n_query, 3, generator=generator)
    obs_indices = torch.tensor([0, 3, 7, 11, n_query - 1]).repeat(batch, 1)
    return {
        "x1": torch.randn(batch, n_query, 2, generator=generator),
        "coords": coords,
        "obs_coords": torch.stack([coords[i, obs_indices[i]] for i in range(batch)]),
        "obs_values": torch.randn(batch, 5, 1, generator=generator),
        "obs_mask": torch.ones(batch, 5),
        "obs_field_ids": torch.tensor([[0, 1, 0, 1, 0]]).expand(batch, -1).clone(),
        "obs_indices": obs_indices,
    }


def _sample(model, values, *, geometry=None, solver="euler", n_steps=2, cache_level="geometry"):
    return model.sample(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_values=values["obs_values"],
        obs_mask=values["obs_mask"],
        obs_field_ids=values["obs_field_ids"],
        clamp_indices=values["obs_indices"],
        n_steps=n_steps,
        ode_solver=solver,
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level=cache_level,
        reconstruction_geometry_cache=geometry,
    )


def test_structured_concat_contract_and_forward():
    torch.manual_seed(1)
    model = _backbone()
    values = _inputs()
    output = model(
        torch.rand(1), values["x1"], values["coords"], values["obs_coords"],
        values["obs_values"], values["obs_mask"], values["obs_field_ids"],
    )
    summary = model.model_summary()
    assert output.shape == (1, 19, 2)
    assert model.cq_fusion_norm.normalized_shape == (24,)
    assert model.cq_head[0].in_features == 24
    assert model.cq_head[0].out_features == 8
    assert model.cq_head[3].in_features == 8
    assert model.cq_head[3].out_features == 8
    assert summary["fusion_mode"] == "structured_concat"
    assert summary["point_state_width"] == 8
    assert summary["global_width"] == 8
    assert summary["local_width"] == 8
    assert summary["cq_fused_width"] == 24
    assert not hasattr(model, "cq_local_proj")
    assert not hasattr(model, "cq_global_scale")
    assert not hasattr(model, "cq_local_scale")


def test_default_additive_state_is_unchanged_and_strict_loadable():
    torch.manual_seed(2)
    legacy_default = ConditionalPointHybridLocalGlobalRBFCQ(
        n_fields=2, hidden_dim=16, cond_dim=8, latent_dim=16,
        num_latents=8, num_heads=4, num_latent_blocks=1,
        cq_query_dim=8, cq_readout_mode="lowrank", cq_readout_rank=4,
        cq_readout_heads=2,
    )
    torch.manual_seed(2)
    explicit_additive = ConditionalPointHybridLocalGlobalRBFCQ(
        n_fields=2, hidden_dim=16, cond_dim=8, latent_dim=16,
        num_latents=8, num_heads=4, num_latent_blocks=1,
        cq_query_dim=8, cq_readout_mode="lowrank",
        cq_fusion_mode="additive", cq_readout_rank=4, cq_readout_heads=2,
    )
    assert legacy_default.cq_fusion_mode == "additive"
    assert legacy_default.state_dict().keys() == explicit_additive.state_dict().keys()
    for name, value in legacy_default.state_dict().items():
        torch.testing.assert_close(value, explicit_additive.state_dict()[name], msg=name)
    result = explicit_additive.load_state_dict(legacy_default.state_dict(), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


def test_unknown_fusion_mode_rejected():
    with pytest.raises(ValueError, match="cq_fusion_mode"):
        ConditionalPointHybridLocalGlobalRBFCQ(
            n_fields=2, cq_fusion_mode="unknown"
        )


def test_structured_concat_all_required_gradients_reachable():
    torch.manual_seed(3)
    model = _backbone()
    values = _inputs(batch=2)
    output = model(
        torch.rand(2), values["x1"], values["coords"], values["obs_coords"],
        values["obs_values"], values["obs_mask"], values["obs_field_ids"],
    )
    output.square().mean().backward()
    grads = {name: parameter.grad for name, parameter in model.named_parameters()}
    required = [
        "log_rbf_sigma",
        "sensor_importance.1.weight",
        "latents",
        "input_cross_attn.attn.in_proj_weight",
        "sensor_back_attn.attn.in_proj_weight",
        "cq_point_encoder.0.weight",
        "cq_global_proj.weight",
        "cq_fusion_norm.weight",
        "cq_head.0.weight",
        "cq_coarse_film.weight",
        "cq_coarse_head.1.weight",
        "cq_readout_in.weight",
        "cq_latent_readout.attn.in_proj_weight",
        "cq_readout_out.weight",
        "cq_readout_scale",
    ]
    for name in required:
        assert grads[name] is not None, name
        assert torch.isfinite(grads[name]).all(), name


@pytest.mark.parametrize(
    ("solver", "n_steps"),
    [("euler", 1), ("euler", 2), ("euler", 4), ("heun", 2)],
)
@pytest.mark.parametrize("cache_level", ["geometry", "static_features"])
def test_structured_concat_cached_and_persistent_equivalence(
    solver: str, n_steps: int, cache_level: str,
):
    torch.manual_seed(4)
    model = _model().eval()
    values = _inputs()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=5,
    )
    torch.manual_seed(991)
    fresh = _sample(
        model, values, solver=solver, n_steps=n_steps, cache_level=cache_level,
    )
    torch.manual_seed(991)
    persistent = _sample(
        model, values, geometry=geometry, solver=solver, n_steps=n_steps,
        cache_level=cache_level,
    )
    torch.testing.assert_close(persistent, fresh, rtol=1e-5, atol=2e-6)


def test_structured_concat_persistent_sample_performs_zero_new_knn_searches():
    torch.manual_seed(5)
    model = _model().eval()
    values = _inputs()
    original = model.model._get_topk_neighbors
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.model._get_topk_neighbors = counted
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=5,
    )
    build_calls = calls
    assert build_calls > 0
    torch.manual_seed(992)
    _sample(model, values, geometry=geometry, n_steps=4, cache_level="static_features")
    assert calls == build_calls


def test_structured_concat_microbatch_loss_gradients_and_update_match():
    torch.manual_seed(6)
    monolithic = _model()
    microbatched = copy.deepcopy(monolithic)
    values = _inputs(batch=2, n_query=31)
    optimizer_full = torch.optim.AdamW(monolithic.parameters(), lr=3e-4, weight_decay=1e-6)
    optimizer_micro = torch.optim.AdamW(microbatched.parameters(), lr=3e-4, weight_decay=1e-6)

    optimizer_full.zero_grad(set_to_none=True)
    torch.manual_seed(993)
    loss_full, _ = monolithic.training_loss(**values)
    loss_full.backward()
    gradients_full = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in monolithic.named_parameters()
    }
    optimizer_full.step()

    optimizer_micro.zero_grad(set_to_none=True)
    torch.manual_seed(993)
    loss_micro, _ = microbatched.training_loss_microbatched(
        **values, query_microbatch_size=7, backward=True,
        reuse_condition_context=True,
    )
    gradients_micro = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in microbatched.named_parameters()
    }
    optimizer_micro.step()

    torch.testing.assert_close(loss_micro, loss_full.detach(), rtol=3e-6, atol=3e-7)
    assert gradients_micro["model.log_rbf_sigma"] is not None
    for name, full in gradients_full.items():
        micro = gradients_micro[name]
        assert (full is None) == (micro is None), name
        if full is not None:
            torch.testing.assert_close(micro, full, rtol=1e-4, atol=3e-7, msg=name)
    for name, parameter in monolithic.named_parameters():
        torch.testing.assert_close(
            parameter, dict(microbatched.named_parameters())[name],
            rtol=3e-5, atol=3e-5, msg=name,
        )
