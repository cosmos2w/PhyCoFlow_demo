from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model import ConditionalPointHybridLocalGlobalRBFCQ
from phycoflow_pointcloud.cache import PersistentTopKGeometryCache
from phycoflow_pointcloud.checkpointing import (
    checkpoint_model_state,
    resolve_checkpoint_state,
)
from phycoflow_pointcloud.config import load_public_config, resolve_model_identity
from phycoflow_pointcloud.models import GL_rbf_CQ, GL_rbf_ENH_CQ, build_pointcloud_model
from phycoflow_pointcloud.priors import RFFGaussianPrior


def _tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _small_cq_config(latent_dim: int = 16) -> dict:
    return {
        "model_name": "GL_rbf_CQ",
        "backbone": "GL_rbf_ENH_CQ",
        "coord_dim": 3,
        "prior": "iid",
        "hidden_dim": 16,
        "cond_dim": 8,
        "field_embed_dim": 4,
        "latent_dim": latent_dim,
        "num_latents": 8,
        "num_heads": 4,
        "num_latent_blocks": 1,
        "ff_mult": 2,
        "summary_type": "mean",
        "gather_mode": "topk_rbf_glres",
        "gather_topk": 3,
        "gather_query_chunk_size": 4,
        "learnable_rbf_sigma": True,
        "neighbor_backend": "torch",
        "USE_FOURIER_PE": True,
        "fourier_pe_num_bands": 2,
        "fourier_pe_max_freq": 4.0,
        "sensor_coord_encoding": "fourier",
        "latent_sensor_reinject": True,
        "glres_scale_init": 1.0e-2,
        "cq_query_dim": 8,
        "cq_readout_mode": "lowrank",
        "cq_readout_rank": 4,
        "cq_readout_heads": 2,
        "cq_fusion_mode": "additive",
        "cq_time_conditioning": "sinusoidal_film",
        "cq_time_embed_dim": 8,
        "cq_measurement_support_mode": "rbf_value_support",
    }


def _varying_xyz_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(3207)
    coords = torch.rand(1, 19, 3, generator=generator)
    assert torch.unique(coords[0, :, 2]).numel() == 19
    obs_indices = torch.tensor([[0, 3, 7, 11, 15, 18]])
    return {
        "coords": coords,
        "x1": torch.randn(1, 19, 2, generator=generator),
        "obs_coords": coords[:, obs_indices[0]],
        "obs_values": torch.randn(1, 6, 1, generator=generator),
        "obs_mask": torch.ones(1, 6),
        "obs_field_ids": torch.tensor([[0, 1, 0, 1, 0, 1]]),
        "obs_indices": obs_indices,
    }


def test_public_aliases_preserve_the_historical_class():
    assert GL_rbf_CQ is ConditionalPointHybridLocalGlobalRBFCQ
    assert GL_rbf_ENH_CQ is ConditionalPointHybridLocalGlobalRBFCQ


def test_public_identity_rejects_unknown_and_conflicting_names():
    assert (
        resolve_model_identity({"backbone": "GL_rbf_ENH_CQ"}).public_name == "GL_rbf_CQ"
    )
    with pytest.raises(ValueError, match="Unknown public"):
        resolve_model_identity({"model_name": "CQ-new"})
    with pytest.raises(ValueError, match="Conflicting"):
        resolve_model_identity({"model_name": "GL_rbf_CQ", "backbone": "GL_rbf_ENH"})


@pytest.mark.parametrize(
    ("path", "public_name", "backbone", "latent_dim"),
    [
        ("configs/gl_rbf_cq.yaml", "GL_rbf_CQ", "GL_rbf_ENH_CQ", 256),
        ("configs/gl_rbf_cq_fast.yaml", "GL_rbf_CQ-fast", "GL_rbf_ENH_CQ", 128),
        ("configs/legacy_gl_rbf_enh.yaml", "GL_rbf_ENH", "GL_rbf_ENH", 128),
    ],
)
def test_public_configs_are_portable_and_unambiguous(
    path, public_name, backbone, latent_dim
):
    config = load_public_config(ROOT / path)
    assert config["model_name"] == public_name
    assert config["backbone"] == backbone
    assert config["latent_dim"] == latent_dim
    assert config["coord_dim"] == 3
    for key in ("data", "dataset_stats_path", "save_dir"):
        assert Path(config[key]).is_absolute()
        assert str(config[key]).startswith(str(ROOT))


def test_balanced_public_scientific_flags_match_the_rc1_run_config():
    public = yaml.safe_load((ROOT / "configs/gl_rbf_cq.yaml").read_text())
    rc1 = yaml.safe_load(
        (ROOT / "ReleaseArtifacts/GL_rbf_CQ_rc1/run_config_training.yaml").read_text()
    )
    ignored = {
        "model_name",
        "coord_dim",
        "Demo_Num",
        "device_ids",
        "data",
        "dataset_stats_path",
        "save_dir",
        "RELOAD",
    }
    for key, expected in rc1.items():
        if key not in ignored:
            assert public[key] == expected, key


def test_portable_checkpoint_is_strict_and_matches_the_rc1_resolved_state():
    manifest_path = ROOT / "artifacts/GL_rbf_CQ_v0.9.0-rc1_portable.json"
    if not manifest_path.exists():
        pytest.skip(
            "Portable binary is generated by scripts/export_gl_rbf_cq_release.py"
        )
    manifest = json.loads(manifest_path.read_text())
    path = ROOT / manifest["artifact"]
    if not path.exists():
        pytest.skip("Portable release checkpoint is external/ignored in this checkout")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["model_ema_enabled"] is False
    assert payload["field_names"] == ("CO", "T", "U_0", "U_1", "p")
    assert _tensor_digest(payload["model"]) == manifest["resolved_state_sha256"]
    torch.manual_seed(9917)
    oracle_config = dict(payload["config"])
    oracle_config["neighbor_backend"] = "torch"
    model = build_pointcloud_model(oracle_config, n_fields=5, device="cpu")
    model.load_state_dict(checkpoint_model_state(payload, model=model), strict=True)
    assert resolve_checkpoint_state(payload, model=model).selection == "live"
    generator = torch.Generator().manual_seed(20260823)
    coords = torch.rand(1, 13, 3, generator=generator)
    obs_indices = torch.tensor([[0, 2, 3, 5, 7, 8, 10, 11, 12]])
    inputs = {
        "t": torch.tensor([0.375]),
        "x_t": torch.randn(1, 13, 5, generator=generator),
        "coords": coords,
        "obs_coords": coords[:, obs_indices[0]],
        "obs_values": torch.randn(1, 9, 1, generator=generator),
        "obs_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32),
        "obs_field_ids": torch.tensor([[0, 1, 2, 3, 4, 1, 3, 0, 4]]),
    }
    model.eval()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            output = model.model(**inputs)
    finally:
        torch.set_num_threads(previous_threads)
    assert (
        _tensor_digest({"output": output})
        == "63e4e5189f54e659aa84f0fff0552080bee4b5e2dac15dc5e25f9f06761cb90a"
    )


def test_training_prior_override_preserves_seeded_factory_state():
    config = _small_cq_config()
    config.update(prior="rff", rff_features=11, rff_lengthscale=0.2)
    torch.manual_seed(197)
    normal = build_pointcloud_model(config, n_fields=2, device="cpu")
    torch.manual_seed(197)
    prior = RFFGaussianPrior(coord_dim=3, n_features=11, lengthscale=0.2)
    training_path = build_pointcloud_model(
        config, n_fields=2, device="cpu", prior_override=prior
    )
    assert normal.state_dict().keys() == training_path.state_dict().keys()
    for key in normal.state_dict():
        torch.testing.assert_close(
            normal.state_dict()[key], training_path.state_dict()[key], rtol=0, atol=0
        )


def test_live_checkpoint_resume_preserves_optimizer_scheduler_and_counters():
    torch.manual_seed(19)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss = model(torch.ones(4, 3)).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()
    checkpoint = {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "epoch": 7,
        "global_step": 311,
    }
    resumed = torch.nn.Linear(3, 2)
    resumed.load_state_dict(
        checkpoint_model_state(checkpoint, prefer_ema=False), strict=True
    )
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=3e-4)
    resumed_optimizer.load_state_dict(checkpoint["optimizer"])
    resumed_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        resumed_optimizer, T_max=10
    )
    resumed_scheduler.load_state_dict(checkpoint["scheduler"])
    assert checkpoint["epoch"] == 7 and checkpoint["global_step"] == 311
    assert resumed_scheduler.state_dict() == scheduler.state_dict()
    assert (
        resumed_optimizer.state_dict()["param_groups"]
        == optimizer.state_dict()["param_groups"]
    )


@pytest.mark.parametrize("latent_dim", [16, 32])
def test_public_cq_varying_xyz_forward_cache_and_microbatch(latent_dim):
    torch.manual_seed(701)
    model = build_pointcloud_model(
        _small_cq_config(latent_dim), n_fields=2, device="cpu"
    )
    values = _varying_xyz_inputs()
    output = model.model(
        torch.tensor([0.35]),
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert output.shape == values["x1"].shape
    assert torch.isfinite(output).all()

    geometry = model.prepare_reconstruction_geometry_cache(
        coords=values["coords"],
        obs_coords=values["obs_coords"],
        obs_mask=values["obs_mask"],
        chunk_size=5,
    )
    assert isinstance(geometry, PersistentTopKGeometryCache)
    full = copy.deepcopy(model)
    micro = copy.deepcopy(model)
    torch.manual_seed(991)
    loss_full, _ = full.training_loss(**values)
    torch.manual_seed(991)
    loss_micro, _ = micro.training_loss_microbatched(
        **values,
        query_microbatch_size=7,
        backward=False,
        reuse_condition_context=True,
    )
    torch.testing.assert_close(loss_micro, loss_full, rtol=2e-6, atol=2e-7)
