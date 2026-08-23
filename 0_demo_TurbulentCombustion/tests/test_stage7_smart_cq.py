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
from model_ema import ModelEMA
from train_pointcloud_ffm import checkpoint_model_state


class _Prior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        return torch.randn(
            coords.shape[0], coords.shape[1], n_channels,
            device=coords.device, dtype=coords.dtype,
        )


def _backbone(
    *, latent_dim: int = 16, n_fields: int = 2, all_on: bool = False,
    query_dim: int = 8, readout_rank: int = 4,
) -> ConditionalPointHybridLocalGlobalRBFCQ:
    return ConditionalPointHybridLocalGlobalRBFCQ(
        n_fields=n_fields,
        hidden_dim=16,
        cond_dim=8,
        field_embed_dim=4,
        latent_dim=latent_dim,
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
        cq_query_dim=query_dim,
        cq_readout_mode="lowrank",
        cq_fusion_mode="additive",
        cq_readout_rank=readout_rank,
        cq_readout_heads=4 if query_dim >= 128 else 2,
        cq_time_conditioning="sinusoidal_film" if all_on else "scalar_concat",
        cq_time_embed_dim=query_dim,
        cq_measurement_support_mode="rbf_value_support" if all_on else "none",
        cq_measurement_support_normalize=True,
    )


def _model(**kwargs) -> PointCloudFFM:
    return PointCloudFFM(_backbone(**kwargs), _Prior())


def _inputs(batch: int = 1, n_query: int = 19):
    generator = torch.Generator().manual_seed(713)
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


def _sample(model, values, *, geometry=None, cache_level="static_features"):
    return model.sample(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_values=values["obs_values"],
        obs_mask=values["obs_mask"],
        obs_field_ids=values["obs_field_ids"],
        clamp_indices=values["obs_indices"],
        n_steps=4,
        ode_solver="euler",
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level=cache_level,
        reconstruction_geometry_cache=geometry,
    )


def test_stage7_disabled_preserves_exact_historical_state_and_output():
    torch.manual_seed(10)
    historical = _backbone()
    torch.manual_seed(10)
    explicit = ConditionalPointHybridLocalGlobalRBFCQ(
        n_fields=2, hidden_dim=16, cond_dim=8, field_embed_dim=4,
        latent_dim=16, num_latents=8, num_heads=4, num_latent_blocks=1,
        ff_mult=2, gather_mode="topk_rbf_glres", gather_topk=3,
        gather_query_chunk_size=4, learnable_rbf_sigma=True,
        neighbor_backend="torch", use_fourier_pe=True,
        fourier_pe_num_bands=2, fourier_pe_max_freq=4.0,
        sensor_coord_encoding="fourier", latent_sensor_reinject=True,
        glres_scale_init=1.0e-2, cq_query_dim=8,
        cq_readout_mode="lowrank", cq_fusion_mode="additive",
        cq_readout_rank=4, cq_readout_heads=2,
        cq_time_conditioning="scalar_concat",
        cq_measurement_support_mode="none",
    )
    assert historical.state_dict().keys() == explicit.state_dict().keys()
    explicit.load_state_dict(historical.state_dict(), strict=True)
    values = _inputs()
    t = torch.tensor([0.37])
    out_a = historical(t, values["x1"], values["coords"], values["obs_coords"],
                       values["obs_values"], values["obs_mask"], values["obs_field_ids"])
    out_b = explicit(t, values["x1"], values["coords"], values["obs_coords"],
                     values["obs_values"], values["obs_mask"], values["obs_field_ids"])
    torch.testing.assert_close(out_a, out_b, rtol=0, atol=0)
    assert not hasattr(explicit, "cq_timestep_film")
    assert not hasattr(explicit, "cq_measurement_support_norm")


def test_ema_formula_apply_restore_and_round_trip(tmp_path: Path):
    model = torch.nn.Linear(2, 1)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ema = ModelEMA(model, decay=0.75)
    with torch.no_grad():
        model.weight.add_(4.0)
        model.bias.sub_(2.0)
    live = {name: value.detach().clone() for name, value in model.state_dict().items()}
    ema.update(model)
    torch.testing.assert_close(ema.shadow["weight"], initial["weight"] * 0.75 + live["weight"] * 0.25)
    torch.testing.assert_close(ema.shadow["bias"], initial["bias"] * 0.75 + live["bias"] * 0.25)
    with ema.average_parameters(model):
        torch.testing.assert_close(model.weight, ema.shadow["weight"])
        torch.testing.assert_close(model.bias, ema.shadow["bias"])
    torch.testing.assert_close(model.weight, live["weight"])
    torch.testing.assert_close(model.bias, live["bias"])

    path = tmp_path / "ema.pt"
    torch.save({
        "model": model.state_dict(), "model_ema": ema.state_dict(),
        "model_ema_enabled": True, "model_ema_eval": True,
    }, path)
    checkpoint = torch.load(path, weights_only=False)
    torch.testing.assert_close(checkpoint_model_state(checkpoint)["weight"], ema.shadow["weight"])
    resumed_model = torch.nn.Linear(2, 1)
    resumed_model.load_state_dict(checkpoint_model_state(checkpoint, prefer_ema=False))
    resumed_ema = ModelEMA(resumed_model, decay=0.1)
    resumed_ema.load_state_dict(checkpoint["model_ema"])
    assert resumed_ema.num_updates == 1
    assert resumed_ema.decay == pytest.approx(0.75)
    torch.testing.assert_close(resumed_ema.shadow["weight"], ema.shadow["weight"])


def test_zero_initialized_film_is_identity_then_all_film_parameters_receive_gradients():
    torch.manual_seed(11)
    backbone = _backbone(all_on=True)
    point_q = torch.randn(2, 7, 8)
    t = torch.tensor([0.2, 0.8])
    torch.testing.assert_close(backbone._cq_apply_timestep_film(point_q, t), point_q, rtol=0, atol=0)
    with torch.no_grad():
        backbone.cq_timestep_film.weight.fill_(0.01)
    output = backbone._cq_apply_timestep_film(point_q, t).square().mean()
    output.backward()
    for name in (
        "cq_timestep_mlp.0.weight", "cq_timestep_mlp.2.weight",
        "cq_timestep_film.weight",
    ):
        gradient = dict(backbone.named_parameters())[name].grad
        assert gradient is not None, name
        assert torch.isfinite(gradient).all(), name


def test_measurement_support_matches_hand_computation_and_missing_field_zero():
    backbone = _backbone(n_fields=3, all_on=True)
    with torch.no_grad():
        backbone.log_rbf_sigma.zero_()
    topk_d2 = torch.tensor([[[0.0, 2.0 * math.log(2.0), 0.0]]])
    topk_idx = torch.tensor([[[0, 1, 2]]])
    topk_valid = torch.ones_like(topk_idx, dtype=torch.bool)
    context = {
        "raw_obs_values": torch.tensor([[[2.0], [4.0], [10.0]]]),
        "raw_obs_field_ids": torch.tensor([[0, 0, 1]]),
    }
    actual = backbone._cq_measurement_support_from_geometry(
        topk_d2, topk_idx, topk_valid, context,
    )
    expected = torch.tensor([[[8.0 / 3.0, 10.0, 0.0, 0.6, 0.4, 0.0]]])
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_measurement_support_has_sigma_gradient_and_changes_with_values_and_ids():
    backbone = _backbone(all_on=True)
    topk_d2 = torch.tensor([[[0.0, 0.1, 0.4]]])
    topk_idx = torch.tensor([[[0, 1, 2]]])
    valid = torch.ones_like(topk_idx, dtype=torch.bool)
    context = {
        "raw_obs_values": torch.tensor([[[1.0], [3.0], [8.0]]]),
        "raw_obs_field_ids": torch.tensor([[0, 0, 1]]),
    }
    first = backbone._cq_measurement_support_from_geometry(topk_d2, topk_idx, valid, context)
    first.square().sum().backward()
    assert backbone.log_rbf_sigma.grad is not None
    assert torch.isfinite(backbone.log_rbf_sigma.grad)
    changed_values = dict(context, raw_obs_values=context["raw_obs_values"] + 2.0)
    changed_ids = dict(context, raw_obs_field_ids=torch.tensor([[1, 1, 0]]))
    assert not torch.equal(first.detach(), backbone._cq_measurement_support_from_geometry(
        topk_d2, topk_idx, valid, changed_values,
    ).detach())
    assert not torch.equal(first.detach(), backbone._cq_measurement_support_from_geometry(
        topk_d2, topk_idx, valid, changed_ids,
    ).detach())


@pytest.mark.parametrize("latent_dim", [128, 256])
def test_all_on_latent_widths_keep_compact_query_and_output_shape(latent_dim: int):
    model = _backbone(
        latent_dim=latent_dim, all_on=True, query_dim=128, readout_rank=64,
    )
    values = _inputs()
    output = model(
        torch.tensor([0.4]), values["x1"], values["coords"], values["obs_coords"],
        values["obs_values"], values["obs_mask"], values["obs_field_ids"],
    )
    assert output.shape == values["x1"].shape
    assert model.cq_query_dim == 128
    assert model.cq_latent_readout.k_proj.in_features == latent_dim
    assert model.cq_latent_readout.k_proj.out_features == 64
    assert model.cq_latent_readout.v_proj.out_features == 128
    assert model.cq_head[0].in_features == 132


def test_all_on_cached_persistent_equivalence_and_zero_post_build_knn():
    torch.manual_seed(12)
    model = _model(all_on=True).eval()
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
    torch.manual_seed(99)
    persistent = _sample(model, values, geometry=geometry)
    assert calls == build_calls
    model.model._get_topk_neighbors = original
    torch.manual_seed(99)
    fresh = _sample(model, values)
    torch.testing.assert_close(persistent, fresh, rtol=1e-5, atol=2e-6)


def test_all_on_uncached_forward_uses_one_knn_and_cached_nfe_reuses_one_condition():
    torch.manual_seed(121)
    model = _model(all_on=True).eval()
    values = _inputs()
    backbone = model.model
    original_knn = backbone._get_topk_neighbors
    knn_calls = 0

    def counted_knn(*args, **kwargs):
        nonlocal knn_calls
        knn_calls += 1
        return original_knn(*args, **kwargs)

    backbone._get_topk_neighbors = counted_knn
    backbone(
        torch.tensor([0.5]), values["x1"], values["coords"], values["obs_coords"],
        values["obs_values"], values["obs_mask"], values["obs_field_ids"],
    )
    assert knn_calls == 1

    condition_calls = 0
    original_condition = backbone.prepare_condition_context

    def counted_condition(*args, **kwargs):
        nonlocal condition_calls
        condition_calls += 1
        return original_condition(*args, **kwargs)

    backbone.prepare_condition_context = counted_condition
    torch.manual_seed(122)
    _sample(model, values, cache_level="static_features")
    assert condition_calls == 1


@pytest.mark.parametrize("all_on", [False, True])
def test_monolithic_microbatch_loss_gradients_and_update_match(all_on: bool):
    torch.manual_seed(13)
    monolithic = _model(all_on=all_on)
    microbatched = copy.deepcopy(monolithic)
    values = _inputs(batch=2, n_query=31)
    opt_full = torch.optim.AdamW(monolithic.parameters(), lr=3e-4, weight_decay=1e-6)
    opt_micro = torch.optim.AdamW(microbatched.parameters(), lr=3e-4, weight_decay=1e-6)

    opt_full.zero_grad(set_to_none=True)
    torch.manual_seed(101)
    loss_full, _ = monolithic.training_loss(**values)
    loss_full.backward()
    gradients_full = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in monolithic.named_parameters()
    }
    opt_full.step()

    opt_micro.zero_grad(set_to_none=True)
    torch.manual_seed(101)
    loss_micro, _ = microbatched.training_loss_microbatched(
        **values, query_microbatch_size=7, backward=True,
        reuse_condition_context=True,
    )
    gradients_micro = {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in microbatched.named_parameters()
    }
    opt_micro.step()

    torch.testing.assert_close(loss_micro, loss_full.detach(), rtol=3e-6, atol=3e-7)
    assert gradients_micro["model.log_rbf_sigma"] is not None
    for name, full in gradients_full.items():
        micro = gradients_micro[name]
        assert (full is None) == (micro is None), name
        if full is not None:
            torch.testing.assert_close(micro, full, rtol=1e-4, atol=5e-7, msg=name)
    for name, parameter in monolithic.named_parameters():
        torch.testing.assert_close(
            parameter, dict(microbatched.named_parameters())[name],
            rtol=3e-5, atol=3e-5, msg=name,
        )
