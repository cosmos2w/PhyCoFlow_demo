from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model import PointCloudFFM
from persistent_topk_geometry_cache import (
    build_persistent_topk_geometry_cache,
    validate_persistent_topk_geometry_cache,
)


class _IIDPrior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        return torch.randn(
            coords.shape[0],
            coords.shape[1],
            n_channels,
            device=coords.device,
            dtype=coords.dtype,
        )


def _resolve_cq_class():
    """Adapt this only if the local Stage-6 CQ class uses another name."""
    import Model

    candidates = (
        "ConditionalPointHybridLocalGlobalRBFCQ",
        "ConditionalPointHybridLocalGlobalRBF_CQ",
        "ConditionalPointHybridLocalGlobalRBFCompactQuery",
    )
    for name in candidates:
        if hasattr(Model, name):
            return getattr(Model, name)
    raise RuntimeError(
        "Could not find the CQ backbone class. Update _resolve_cq_class() to "
        "the actual local Stage-6 class name before running these tests."
    )


def _model(mode: str = "lowrank") -> PointCloudFFM:
    CQ = _resolve_cq_class()

    # These kwargs reflect the intended CQ-LR shape but remain deliberately
    # small for unit tests. Adapt keyword names only if the local CQ constructor
    # differs; do not change the tested semantics.
    backbone = CQ(
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
        latent_reinject_every=1,
        glres_scale_init=1e-2,
        cq_query_dim=8,
        cq_readout_mode=mode,
        cq_readout_rank=4,
        cq_readout_heads=2,
    )
    return PointCloudFFM(backbone, _IIDPrior()).eval()


def _inputs(seed: int = 123):
    g = torch.Generator().manual_seed(seed)
    coords = torch.rand(1, 19, 3, generator=g)
    obs_idx = torch.tensor([[0, 3, 7, 11, 15, 18]])
    obs_coords = coords[:, obs_idx[0]]
    return {
        "coords": coords,
        "obs_coords": obs_coords,
        "obs_values": torch.randn(1, 6, 1, generator=g),
        "obs_mask": torch.ones(1, 6),
        "obs_indices": obs_idx,
        "obs_field_ids": torch.tensor([[0, 1, 0, 1, 0, 1]]),
    }


def _sample(
    model, values, *, geometry=None, n_steps=2, solver="euler",
    cache_level="geometry",
):
    return model.sample(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_values=values["obs_values"],
        obs_mask=values["obs_mask"],
        obs_field_ids=values["obs_field_ids"],
        n_steps=n_steps,
        clamp_indices=values["obs_indices"],
        ode_solver=solver,
        obs_consistency_mode="none",
        reconstruction_execution_mode="cached_streamed",
        reconstruction_query_chunk_size=5,
        reconstruction_cache_level=cache_level,
        reconstruction_geometry_cache=geometry,
    )


@pytest.mark.parametrize("cache_level", ["geometry", "static_features"])
@pytest.mark.parametrize("n_steps", [1, 2, 4])
def test_cq_persistent_geometry_matches_fresh_euler(n_steps: int, cache_level: str):
    torch.manual_seed(7)
    model = _model()
    values = _inputs()

    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
        chunk_size=5,
    )

    torch.manual_seed(99)
    fresh = _sample(model, values, geometry=None, n_steps=n_steps, solver="euler", cache_level=cache_level)
    torch.manual_seed(99)
    persistent = _sample(
        model,
        values,
        geometry=geometry,
        n_steps=n_steps,
        solver="euler",
        cache_level=cache_level,
    )

    torch.testing.assert_close(persistent, fresh, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("cache_level", ["geometry", "static_features"])
def test_cq_persistent_geometry_matches_fresh_heun(cache_level: str):
    torch.manual_seed(8)
    model = _model()
    values = _inputs()

    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
        chunk_size=5,
    )

    torch.manual_seed(101)
    fresh = _sample(model, values, geometry=None, n_steps=2, solver="heun", cache_level=cache_level)
    torch.manual_seed(101)
    persistent = _sample(model, values, geometry=geometry, n_steps=2, solver="heun", cache_level=cache_level)

    torch.testing.assert_close(persistent, fresh, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("cache_level", ["geometry", "static_features"])
def test_geometry_reuse_remains_correct_when_sensor_values_change(cache_level: str):
    torch.manual_seed(9)
    model = _model()
    a = _inputs(seed=321)
    b = dict(a)
    b["obs_values"] = a["obs_values"] + 0.37

    geometry = model.prepare_reconstruction_geometry_cache(
        coords=a["coords"],
        obs_coords=a["obs_coords"],
        obs_mask=a["obs_mask"],
        chunk_size=5,
    )

    for seed, values in ((201, a), (202, b)):
        torch.manual_seed(seed)
        fresh = _sample(model, values, geometry=None, cache_level=cache_level)
        torch.manual_seed(seed)
        persistent = _sample(model, values, geometry=geometry, cache_level=cache_level)
        torch.testing.assert_close(persistent, fresh, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("cache_level", ["geometry", "static_features"])
def test_persistent_sample_performs_zero_new_knn_searches(cache_level: str):
    torch.manual_seed(10)
    model = _model()
    values = _inputs()

    original = model.model._get_topk_neighbors
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.model._get_topk_neighbors = counted

    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
        chunk_size=5,
    )
    build_calls = calls
    assert build_calls > 0

    torch.manual_seed(333)
    _sample(model, values, geometry=geometry, n_steps=4, cache_level=cache_level)

    assert calls == build_calls


def test_geometry_validation_rejects_new_query_tensor():
    torch.manual_seed(11)
    model = _model()
    values = _inputs()

    # The standalone helper is also validated directly.
    geometry = build_persistent_topk_geometry_cache(
        model.model,
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
        chunk_size=5,
    )

    validate_persistent_topk_geometry_cache(
        geometry,
        model.model,
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
    )

    new_coords = values["coords"].clone()
    with pytest.raises(ValueError, match="stale|incompatible"):
        validate_persistent_topk_geometry_cache(
            geometry,
            model.model,
            coords=new_coords,
            obs_coords=values["obs_coords"],
            obs_mask=values["obs_mask"],
        )


def test_geometry_cache_is_smaller_than_static_query_features():
    torch.manual_seed(12)
    model = _model()
    values = _inputs()

    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
        chunk_size=5,
    )

    condition = model.model.prepare_condition_context(
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    static = model.model.prepare_query_context(
        values["coords"],
        condition,
        cache_level="static_features",
        chunk_size=5,
        precomputed_geometry=geometry,
    )

    geometry_bytes = (
        geometry.nbytes()
        if hasattr(geometry, "nbytes")
        else model.model.context_nbytes(geometry)
    )
    static_bytes = model.model.context_nbytes(static)

    assert geometry_bytes > 0
    assert static_bytes > 0
    assert geometry_bytes < static_bytes


@pytest.mark.parametrize(
    ("cache_level", "solver", "n_steps"),
    [
        (level, solver, n_steps)
        for level in ("none", "geometry", "static_features")
        for solver, n_steps in (
            ("euler", 1), ("euler", 2), ("euler", 4),
            ("euler", 8), ("heun", 2),
        )
    ],
)
def test_intra_trajectory_topk_call_counts_are_nfe_independent_when_cached(
    cache_level: str, solver: str, n_steps: int,
):
    torch.manual_seed(13)
    model = _model()
    values = _inputs()
    original = model.model._get_topk_neighbors
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.model._get_topk_neighbors = counted
    torch.manual_seed(405)
    _sample(
        model, values, n_steps=n_steps, solver=solver,
        cache_level=cache_level,
    )
    if cache_level == "none":
        velocity_evaluations = n_steps * (2 if solver == "heun" else 1)
        assert calls == 7 * velocity_evaluations
    else:
        assert calls == 4


def test_cq_full_accepts_persistent_geometry():
    torch.manual_seed(14)
    model = _model("full")
    values = _inputs()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=5,
    )
    torch.manual_seed(406)
    fresh = _sample(model, values, n_steps=2, cache_level="static_features")
    torch.manual_seed(406)
    persistent = _sample(
        model, values, geometry=geometry, n_steps=2,
        cache_level="static_features",
    )
    torch.testing.assert_close(persistent, fresh, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("target", ["obs_coords", "obs_mask"])
def test_geometry_validation_rejects_new_sensor_geometry_tensor(target: str):
    torch.manual_seed(15)
    model = _model()
    values = _inputs()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=5,
    )
    changed = dict(values)
    changed[target] = values[target].clone()
    with pytest.raises(ValueError, match="stale|incompatible"):
        validate_persistent_topk_geometry_cache(
            geometry, model.model, coords=changed["coords"],
            obs_coords=changed["obs_coords"], obs_mask=changed["obs_mask"],
        )


def test_geometry_validation_rejects_in_place_geometry_change():
    torch.manual_seed(16)
    model = _model()
    values = _inputs()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=5,
    )
    values["coords"].add_(0.01)
    with pytest.raises(ValueError, match="stale|incompatible"):
        validate_persistent_topk_geometry_cache(
            geometry, model.model, coords=values["coords"],
            obs_coords=values["obs_coords"], obs_mask=values["obs_mask"],
        )


@pytest.mark.parametrize("attribute", ["gather_topk", "gather_mode"])
def test_geometry_validation_rejects_changed_knn_contract(attribute: str):
    torch.manual_seed(17)
    model = _model()
    values = _inputs()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=5,
    )
    if attribute == "gather_topk":
        model.model.gather_topk = 2
    else:
        model.model.gather_mode = "topk_rbf"
    with pytest.raises(ValueError, match="stale|incompatible"):
        validate_persistent_topk_geometry_cache(
            geometry, model.model, coords=values["coords"],
            obs_coords=values["obs_coords"], obs_mask=values["obs_mask"],
        )


def test_persistent_cache_payload_is_geometry_only():
    torch.manual_seed(18)
    model = _model()
    values = _inputs()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=5,
    )
    keys = set(geometry.as_mapping())
    assert {"topk_d2", "topk_idx", "topk_valid"} <= keys
    assert not keys & {
        "obs_values", "refined_sensor_feat", "sensor_importance_bias",
        "latents", "cq_latent_k", "cq_latent_v", "local_cond",
        "query_global", "x_t",
    }


def test_geometry_cache_rejected_for_legacy_full_execution():
    torch.manual_seed(19)
    model = _model()
    values = _inputs()
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"], obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"], chunk_size=5,
    )
    with pytest.raises(ValueError, match="cached_streamed"):
        model.sample(
            coords=values["coords"], obs_coords=values["obs_coords"],
            obs_values=values["obs_values"], obs_mask=values["obs_mask"],
            obs_field_ids=values["obs_field_ids"],
            reconstruction_execution_mode="legacy_full",
            reconstruction_geometry_cache=geometry,
        )


def test_cq_lr_latent_kv_remains_per_condition_with_persistent_geometry():
    torch.manual_seed(20)
    model = _model()
    values_a = _inputs()
    values_b = dict(values_a)
    values_b["obs_values"] = values_a["obs_values"] + 0.25
    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values_a["coords"], obs_coords=values_a["obs_coords"],
        obs_mask=values_a["obs_mask"], chunk_size=5,
    )
    original = model.model.cq_latent_readout.project_latents
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    model.model.cq_latent_readout.project_latents = counted
    _sample(model, values_a, geometry=geometry, cache_level="static_features")
    _sample(model, values_b, geometry=geometry, cache_level="static_features")
    assert calls == 2
