from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model import ConditionalPointHybridLocalGlobalRBF, PointCloudFFM


class _IIDPrior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        return torch.randn(
            coords.shape[0], coords.shape[1], n_channels,
            device=coords.device, dtype=coords.dtype,
        )


def _model(gather_mode: str) -> PointCloudFFM:
    backbone = ConditionalPointHybridLocalGlobalRBF(
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
        gather_mode=gather_mode,
        gather_topk=3,
        gather_query_chunk_size=4,
        learnable_rbf_sigma=True,
        neighbor_backend="torch",
        use_fourier_pe=True,
        fourier_pe_num_bands=2,
        fourier_pe_max_freq=4.0,
        enhanced_backbone=True,
        sensor_coord_encoding="fourier",
        latent_sensor_reinject=True,
        query_latent_readout=True,
        query_readout_type="coord",
        query_readout_scale_init=1e-2,
        enhanced_head_norm=True,
        glres_scale_init=1e-2,
    )
    return PointCloudFFM(backbone, _IIDPrior()).eval()


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(123)
    coords = torch.rand(1, 13, 3, generator=generator)
    obs_indices = torch.tensor([[0, 3, 7, 10, 12]])
    return {
        "coords": coords,
        "obs_coords": coords[:, obs_indices[0]],
        "obs_values": torch.randn(1, 5, 1, generator=generator),
        "obs_mask": torch.ones(1, 5),
        "obs_field_ids": torch.tensor([[0, 1, 0, 1, 0]]),
        "clamp_indices": obs_indices,
    }


def _compare(model: PointCloudFFM, *, solver: str, n_steps: int, mode: str = "none"):
    kwargs = {
        **_inputs(),
        "n_steps": n_steps,
        "ode_solver": solver,
        "obs_consistency_mode": mode,
    }
    torch.manual_seed(991)
    legacy = model.sample(**kwargs, reconstruction_execution_mode="legacy_full")
    torch.manual_seed(991)
    streamed = model.sample(
        **kwargs,
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level="static_features",
    )
    torch.testing.assert_close(streamed, legacy, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("gather_mode", ["topk_rbf", "topk_rbf_glres"])
@pytest.mark.parametrize("solver", ["euler", "heun"])
@pytest.mark.parametrize("n_steps", [1, 2, 4])
def test_cached_streamed_matches_legacy_across_gather_solver_nfe(
    gather_mode: str, solver: str, n_steps: int,
):
    torch.manual_seed(8)
    _compare(_model(gather_mode), solver=solver, n_steps=n_steps)


@pytest.mark.parametrize(
    "consistency_mode", ["none", "default_hard", "endpoint", "endpoint_smooth"],
)
def test_cached_streamed_preserves_observation_consistency(consistency_mode: str):
    torch.manual_seed(11)
    _compare(
        _model("topk_rbf_glres"),
        solver="heun",
        n_steps=2,
        mode=consistency_mode,
    )


@pytest.mark.parametrize("cache_level", ["none", "geometry", "static_features"])
def test_all_cache_levels_match_legacy(cache_level: str):
    torch.manual_seed(19)
    model = _model("topk_rbf_glres")
    kwargs = {**_inputs(), "n_steps": 2, "ode_solver": "heun", "obs_consistency_mode": "none"}
    torch.manual_seed(77)
    legacy = model.sample(**kwargs, reconstruction_execution_mode="legacy_full")
    torch.manual_seed(77)
    streamed = model.sample(
        **kwargs,
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level=cache_level,
    )
    torch.testing.assert_close(streamed, legacy, rtol=1e-5, atol=2e-6)


def test_static_context_has_measurable_bounded_storage():
    torch.manual_seed(31)
    model = _model("topk_rbf")
    values = _inputs()
    with torch.no_grad():
        condition = model.model.prepare_condition_context(
            values["obs_coords"], values["obs_values"],
            values["obs_mask"], values["obs_field_ids"],
        )
        query = model.model.prepare_query_context(
            values["coords"], condition, cache_level="static_features", chunk_size=5,
        )
    assert query["local_cond"].shape == (1, 13, 8)
    assert query["query_global"].shape == (1, 13, 16)
    assert model.model.context_nbytes(query) > 0


def test_cached_streamed_encodes_observations_once_for_heun_trajectory():
    torch.manual_seed(47)
    model = _model("topk_rbf_glres")
    original = model.model._encode_latents
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.model._encode_latents = counted
    torch.manual_seed(90)
    model.sample(
        **_inputs(),
        n_steps=4,
        ode_solver="heun",
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level="static_features",
    )
    assert calls == 1
    assert model._last_reconstruction_condition_bytes > 0
    assert model._last_reconstruction_cache_bytes > 0
