from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model import ConditionalPointHybridLocalGlobalRBFCQ, PointCloudFFM
from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm
from train_pointcloud_ffm import checkpoint_model_state


class _RecordingRFFPrior(torch.nn.Module):
    def __init__(self, coord_dim: int = 3, n_features: int = 11):
        super().__init__()
        omega = torch.linspace(-2.0, 2.0, coord_dim * n_features).reshape(
            coord_dim, n_features
        )
        phase = torch.linspace(0.0, 2.0 * math.pi, n_features)
        self.register_buffer("omega", omega)
        self.register_buffer("phase", phase)
        self.calls: list[int] = []

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        self.calls.append(int(coords.shape[1]))
        phi = math.sqrt(2.0 / self.omega.shape[1]) * torch.cos(
            coords @ self.omega + self.phase
        )
        weights = torch.randn(
            coords.shape[0],
            n_channels,
            self.omega.shape[1],
            device=coords.device,
            dtype=coords.dtype,
        )
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def _backbone(mode: str) -> ConditionalPointHybridLocalGlobalRBFCQ:
    return ConditionalPointHybridLocalGlobalRBFCQ(
        n_fields=2,
        coord_dim=3,
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
        cq_global_scale_init=1.0,
        cq_local_scale_init=1.0,
        cq_readout_scale_init=1.0e-2,
    )


def _model(mode: str) -> PointCloudFFM:
    return PointCloudFFM(_backbone(mode), _RecordingRFFPrior())


def _inputs(batch: int = 2, n_query: int = 31) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(313)
    coords = torch.rand(batch, n_query, 3, generator=generator)
    x1 = torch.randn(batch, n_query, 2, generator=generator)
    obs_indices = torch.stack(
        [torch.tensor([0, 3, 7, 10, n_query - 2]) for _ in range(batch)]
    )
    obs_coords = torch.stack([coords[i, obs_indices[i]] for i in range(batch)])
    return {
        "x1": x1,
        "coords": coords,
        "obs_coords": obs_coords,
        "obs_values": torch.randn(batch, 5, 1, generator=generator),
        "obs_mask": torch.ones(batch, 5),
        "obs_field_ids": torch.tensor([[0, 1, 0, 1, 0]]).expand(batch, -1).clone(),
        "obs_indices": obs_indices,
    }


@pytest.mark.parametrize("mode", ["full", "lowrank"])
def test_cq_output_shape_and_compact_invariants(mode: str):
    torch.manual_seed(1)
    model = _backbone(mode)
    values = _inputs(batch=1, n_query=13)
    output = model(
        torch.rand(1),
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert output.shape == (1, 13, 2)
    assert not hasattr(model, "point_encoder")
    assert not hasattr(model, "head")
    assert model.cq_head[0].in_features == model.cq_query_dim
    assert model.model_summary()["legacy_concat_width"] == 40
    assert model.model_summary()["cq_fused_width"] == 8
    assert model.cq_readout_heads == (4 if mode == "full" else 2)


def test_cq_variants_share_seeded_initialization_outside_readout():
    torch.manual_seed(19)
    full = _backbone("full")
    torch.manual_seed(19)
    lowrank = _backbone("lowrank")
    full_parameters = dict(full.named_parameters())
    lowrank_parameters = dict(lowrank.named_parameters())
    variant_prefixes = (
        "cq_query_decoder_token",
        "cq_readout_in",
        "cq_latent_readout",
        "cq_readout_out",
    )
    common_names = [
        name for name in full_parameters
        if name in lowrank_parameters and not name.startswith(variant_prefixes)
    ]
    assert common_names
    for name in common_names:
        torch.testing.assert_close(full_parameters[name], lowrank_parameters[name], msg=name)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cq_readout_mode": "unknown"},
        {"cq_query_dim": 7, "cq_readout_heads": 2},
        {"cq_readout_rank": 5, "cq_readout_heads": 2},
        {"cq_readout_heads": 0},
    ],
)
def test_cq_config_validation(kwargs):
    with pytest.raises(ValueError):
        ConditionalPointHybridLocalGlobalRBFCQ(n_fields=2, **kwargs)


@pytest.mark.parametrize("mode", ["full", "lowrank"])
def test_cq_glres_all_required_modules_receive_gradients(mode: str):
    torch.manual_seed(2)
    model = _backbone(mode)
    values = _inputs(batch=2, n_query=13)
    output = model(
        torch.rand(2),
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
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
    ]
    if mode == "full":
        required.extend([
            "cq_readout_in.weight",
            "cq_latent_readout.attn.in_proj_weight",
            "cq_readout_out.weight",
        ])
    else:
        required.extend([
            "cq_latent_readout.q_proj.weight",
            "cq_latent_readout.k_proj.weight",
            "cq_latent_readout.v_proj.weight",
        ])
    for name in required:
        assert grads[name] is not None, name
        assert torch.isfinite(grads[name]).all(), name


@pytest.mark.parametrize("mode", ["full", "lowrank"])
@pytest.mark.parametrize(
    ("solver", "n_steps", "consistency"),
    [
        ("euler", 1, "none"),
        ("euler", 2, "default_hard"),
        ("euler", 4, "endpoint_smooth"),
        ("heun", 2, "endpoint_smooth"),
    ],
)
def test_cq_cached_streamed_matches_full(
    mode: str, solver: str, n_steps: int, consistency: str
):
    torch.manual_seed(3)
    model = _model(mode).eval()
    values = _inputs(batch=1, n_query=13)
    kwargs = {
        key: values[key]
        for key in (
            "coords", "obs_coords", "obs_values", "obs_mask",
            "obs_field_ids", "obs_indices",
        )
    }
    kwargs["clamp_indices"] = kwargs.pop("obs_indices")
    kwargs.update(
        n_steps=n_steps,
        ode_solver=solver,
        obs_consistency_mode=consistency,
    )
    torch.manual_seed(991)
    full = model.sample(**kwargs, reconstruction_execution_mode="legacy_full")
    torch.manual_seed(991)
    cached = model.sample(
        **kwargs,
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level="static_features",
    )
    torch.testing.assert_close(cached, full, rtol=2e-5, atol=3e-6)


@pytest.mark.parametrize("mode", ["full", "lowrank"])
@pytest.mark.parametrize("cache_level", ["none", "geometry", "static_features"])
def test_cq_all_cache_levels(mode: str, cache_level: str):
    torch.manual_seed(4)
    model = _model(mode).eval()
    values = _inputs(batch=1, n_query=13)
    kwargs = {
        key: values[key]
        for key in (
            "coords", "obs_coords", "obs_values", "obs_mask",
            "obs_field_ids",
        )
    }
    torch.manual_seed(992)
    full = model.sample(
        **kwargs,
        n_steps=2,
        ode_solver="euler",
        obs_consistency_mode="none",
        reconstruction_execution_mode="legacy_full",
    )
    torch.manual_seed(992)
    cached = model.sample(
        **kwargs,
        n_steps=2,
        ode_solver="euler",
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level=cache_level,
    )
    torch.testing.assert_close(cached, full, rtol=2e-5, atol=3e-6)


def _gradients(model: PointCloudFFM):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


@pytest.mark.parametrize("mode", ["full", "lowrank"])
def test_cq_query_microbatch_matches_loss_gradients_and_adam_update(mode: str):
    torch.manual_seed(5)
    monolithic = _model(mode)
    microbatched = copy.deepcopy(monolithic)
    values = _inputs()
    optimizer_full = torch.optim.AdamW(monolithic.parameters(), lr=3e-4, weight_decay=1e-6)
    optimizer_micro = torch.optim.AdamW(microbatched.parameters(), lr=3e-4, weight_decay=1e-6)

    optimizer_full.zero_grad(set_to_none=True)
    torch.manual_seed(993)
    loss_full, _ = monolithic.training_loss(**values)
    loss_full.backward()
    gradients_full = _gradients(monolithic)
    optimizer_full.step()

    optimizer_micro.zero_grad(set_to_none=True)
    torch.manual_seed(993)
    loss_micro, metrics = microbatched.training_loss_microbatched(
        **values,
        query_microbatch_size=7,
        backward=True,
        reuse_condition_context=True,
    )
    gradients_micro = _gradients(microbatched)
    optimizer_micro.step()

    torch.testing.assert_close(loss_micro, loss_full.detach(), rtol=3e-6, atol=3e-7)
    assert metrics["query_microbatches"] == 5.0
    assert gradients_micro["model.log_rbf_sigma"] is not None
    for name, full in gradients_full.items():
        micro = gradients_micro[name]
        assert (full is None) == (micro is None), name
        if full is not None:
            torch.testing.assert_close(micro, full, rtol=1e-4, atol=3e-7, msg=name)
    for (name_full, parameter_full), (name_micro, parameter_micro) in zip(
        monolithic.named_parameters(), microbatched.named_parameters()
    ):
        assert name_full == name_micro
        torch.testing.assert_close(
            parameter_micro, parameter_full, rtol=3e-5, atol=3e-5, msg=name_full
        )


@pytest.mark.parametrize("mode", ["full", "lowrank"])
def test_cq_condition_context_encoded_once_per_cached_trajectory(mode: str):
    torch.manual_seed(61)
    model = _model(mode).eval()
    original = model.model.prepare_condition_context
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.model.prepare_condition_context = counted
    values = _inputs(batch=1, n_query=13)
    model.sample(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_values=values["obs_values"],
        obs_mask=values["obs_mask"],
        obs_field_ids=values["obs_field_ids"],
        n_steps=4,
        ode_solver="heun",
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level="static_features",
    )
    assert calls == 1


def test_cq_lr_projects_latent_kv_once_per_cached_heun_trajectory():
    torch.manual_seed(6)
    model = _model("lowrank").eval()
    original = model.model.cq_latent_readout.project_latents
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.model.cq_latent_readout.project_latents = counted
    values = _inputs(batch=1, n_query=13)
    model.sample(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_values=values["obs_values"],
        obs_mask=values["obs_mask"],
        obs_field_ids=values["obs_field_ids"],
        n_steps=4,
        ode_solver="heun",
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level="static_features",
    )
    assert calls == 1


@pytest.mark.parametrize("mode", ["full", "lowrank"])
def test_cq_large_query_chunk_smoke(mode: str):
    torch.manual_seed(7)
    model = _backbone(mode).eval()
    values = _inputs(batch=1, n_query=1003)
    with torch.no_grad():
        condition = model.prepare_condition_context(
            values["obs_coords"],
            values["obs_values"],
            values["obs_mask"],
            values["obs_field_ids"],
        )
        output = model.forward_query_chunk(
            t=torch.rand(1),
            x_t_chunk=values["x1"],
            coords_chunk=values["coords"],
            condition_context=condition,
        )
    assert output.shape == (1, 1003, 2)


def test_legacy_f0_checkpoint_still_loads_strictly():
    config_path = ROOT / "_CheckNotes/Stage6_formal_baseline/F0_frozen_current.yaml"
    checkpoint_path = (
        ROOT / "_CheckNotes/Stage6_formal_baseline/runs/"
        "F0_frozen_current_DemoN9300_20260821_075633/best.pt"
    )
    if not checkpoint_path.exists():
        pytest.skip("Immutable F0 checkpoint artifact is unavailable.")
    config = yaml.safe_load(config_path.read_text())
    model = build_gl_rbf_ffm(config, n_fields=5, device=torch.device("cpu"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    result = model.load_state_dict(checkpoint_model_state(checkpoint), strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys
