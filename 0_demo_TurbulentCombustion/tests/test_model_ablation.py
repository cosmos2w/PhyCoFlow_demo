from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_ablation_configs import CONFIGS, CONFIG_DIR, audit  # noqa: E402
from model_ablation import (  # noqa: E402
    DeterministicDMFRegressor,
    LocalSensorTokensOnlyBackbone,
    NoLocalQueryConditioningBackbone,
    NoSensorGlobalFeedbackBackbone,
)
from phycoflow_pointcloud.models.factory import build_pointcloud_model  # noqa: E402
from phycoflow_pointcloud.priors import IIDGaussianPrior, RFFGaussianPrior  # noqa: E402


def _configs() -> dict[str, dict]:
    return {
        ablation_id: yaml.safe_load((CONFIG_DIR / filename).read_text())
        for ablation_id, filename in CONFIGS.items()
    }


def _small(config: dict) -> dict:
    value = copy.deepcopy(config)
    value.update(
        {
            "hidden_dim": 16,
            "cond_dim": 8,
            "field_embed_dim": 4,
            "latent_dim": 16,
            "num_latents": 8,
            "num_heads": 4,
            "num_latent_blocks": 2,
            "ff_mult": 2,
            "gather_topk": 3,
            "gather_query_chunk_size": 5,
            "neighbor_backend": "torch",
            "fourier_pe_num_bands": 2,
            "fourier_pe_max_freq": 4.0,
        }
    )
    return value


def _inputs(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(4102)
    coords = torch.rand(2, 13, 3, generator=generator)
    obs_indices = torch.tensor([[0, 2, 5, 8, 11], [1, 3, 6, 9, 12]])
    values = {
        "x1": torch.randn(2, 13, 5, generator=generator),
        "coords": coords,
        "obs_coords": torch.stack([coords[i, obs_indices[i]] for i in range(2)]),
        "obs_values": torch.randn(2, 5, 1, generator=generator),
        "obs_mask": torch.ones(2, 5),
        "obs_field_ids": torch.full((2, 5), 2, dtype=torch.long),
        "obs_indices": obs_indices,
    }
    return {key: tensor.to(device) for key, tensor in values.items()}


def _state_schema(module: torch.nn.Module) -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    return {
        key: (tensor.dtype, tuple(tensor.shape))
        for key, tensor in module.state_dict().items()
    }


def test_all_configs_resolve_are_cond_t_and_match_a0():
    report, failures = audit()
    assert failures == [], report
    for ablation_id, config in _configs().items():
        assert config["ablation"]["id"] == ablation_id
        assert config["cond_fields"] == [2]
        assert config["vis_cond_fields"] == [2]
        assert config["vis_n_obs_list"] == [256]
        model = build_pointcloud_model(_small(config), n_fields=5)
        assert model.ablation_metadata["ablation_id"] == ablation_id


def test_a1_shape_backward_determinism_and_no_prior():
    config = _small(_configs()["A1"])
    model = build_pointcloud_model(config, n_fields=5)
    assert isinstance(model, DeterministicDMFRegressor)
    assert not hasattr(model, "prior")
    values = _inputs()
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    loss, _ = model.training_loss(**values)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    model.eval()
    sample_args = {key: values[key] for key in (
        "coords", "obs_coords", "obs_values", "obs_mask", "obs_field_ids"
    )}
    first = model.sample(**sample_args, obs_consistency_mode="none")
    second = model.sample(**sample_args, obs_consistency_mode="none")
    assert first.shape == (2, 13, 5)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="no generative source prior"):
        model.sample_source(values["coords"])


def test_a2_schema_and_execution_routes():
    configs = _configs()
    baseline = _small(configs["A2"])
    baseline.pop("ablation")
    torch.manual_seed(12)
    a0 = build_pointcloud_model(baseline, n_fields=5)
    torch.manual_seed(12)
    a2 = build_pointcloud_model(_small(configs["A2"]), n_fields=5)
    assert isinstance(a2.model, NoSensorGlobalFeedbackBackbone)
    assert _state_schema(a2) == _state_schema(a0)

    calls = {"sensor_back": 0, "sensor_to_latent": 0, "latent": 0, "readout": 0, "local": 0}

    def count_module(module, key):
        original = module.forward

        def counted(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        module.forward = counted

    count_module(a2.model.sensor_back_attn, "sensor_back")
    count_module(a2.model.input_cross_attn, "sensor_to_latent")
    count_module(a2.model.latent_blocks[0], "latent")
    count_module(a2.model.query_latent_readout, "readout")
    original_local = a2.model._get_topk_neighbors

    def counted_local(*args, **kwargs):
        calls["local"] += 1
        return original_local(*args, **kwargs)

    a2.model._get_topk_neighbors = counted_local
    values = _inputs()
    prediction = a2.model(
        torch.full((2,), 0.5),
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert torch.isfinite(prediction).all()
    assert calls["sensor_back"] == 0
    assert all(calls[key] > 0 for key in ("sensor_to_latent", "latent", "readout", "local"))


def test_a3_schema_zero_local_and_global_routes_active():
    configs = _configs()
    baseline = _small(configs["A3"])
    baseline.pop("ablation")
    a0 = build_pointcloud_model(baseline, n_fields=5)
    a3 = build_pointcloud_model(_small(configs["A3"]), n_fields=5)
    assert isinstance(a3.model, NoLocalQueryConditioningBackbone)
    assert _state_schema(a3) == _state_schema(a0)
    values = _inputs()
    calls = {"sensor_back": 0, "sensor_to_latent": 0, "readout": 0, "knn": 0}

    def count_module(module, key):
        original = module.forward

        def counted(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        module.forward = counted

    count_module(a3.model.sensor_back_attn, "sensor_back")
    count_module(a3.model.input_cross_attn, "sensor_to_latent")
    count_module(a3.model.query_latent_readout, "readout")
    original_knn = a3.model._get_topk_neighbors

    def counted_knn(*args, **kwargs):
        calls["knn"] += 1
        return original_knn(*args, **kwargs)

    a3.model._get_topk_neighbors = counted_knn
    local = a3.model.aggregate_sparse_obs(
        values["coords"],
        torch.randn(2, 13, 16),
        values["obs_coords"],
        torch.randn(2, 5, 8),
        values["obs_mask"],
    )
    assert local.shape == (2, 13, 8)
    assert torch.count_nonzero(local) == 0
    prediction = a3.model(
        torch.full((2,), 0.5),
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert torch.isfinite(prediction).all()
    assert calls["knn"] == 0
    assert all(calls[key] > 0 for key in ("sensor_back", "sensor_to_latent", "readout"))


def test_a4_backbone_schema_iid_prior_and_rf_backward():
    configs = _configs()
    baseline = _small(configs["A4"])
    baseline["prior"] = "rff"
    baseline.pop("ablation")
    a0 = build_pointcloud_model(baseline, n_fields=5)
    a4 = build_pointcloud_model(_small(configs["A4"]), n_fields=5)
    assert _state_schema(a4.model) == _state_schema(a0.model)
    assert isinstance(a4.prior, IIDGaussianPrior)
    assert not isinstance(a4.prior, RFFGaussianPrior)
    loss, _ = a4.training_loss(**_inputs())
    assert torch.isfinite(loss)
    loss.backward()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in a4.parameters()
    )


def test_a5_schema_and_only_local_sensor_conditioning_executes():
    configs = _configs()
    baseline = _small(configs["A5"])
    baseline.pop("ablation")
    a0 = build_pointcloud_model(baseline, n_fields=5)
    a5 = build_pointcloud_model(_small(configs["A5"]), n_fields=5)
    assert isinstance(a5.model, LocalSensorTokensOnlyBackbone)
    assert _state_schema(a5) == _state_schema(a0)
    assert a5.model.use_query_latent_readout is False

    values = _inputs()
    calls = {
        "sensor_to_latent": 0,
        "latent": 0,
        "sensor_back": 0,
        "readout": 0,
        "local": 0,
    }

    def count_module(module, key):
        original = module.forward

        def counted(*args, **kwargs):
            calls[key] += 1
            return original(*args, **kwargs)

        module.forward = counted

    count_module(a5.model.input_cross_attn, "sensor_to_latent")
    count_module(a5.model.latent_blocks[0], "latent")
    count_module(a5.model.sensor_back_attn, "sensor_back")
    count_module(a5.model.query_latent_readout, "readout")
    original_local = a5.model._get_topk_neighbors

    def counted_local(*args, **kwargs):
        calls["local"] += 1
        return original_local(*args, **kwargs)

    a5.model._get_topk_neighbors = counted_local
    t = torch.full((2,), 0.5)
    prediction = a5.model(
        t,
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert torch.isfinite(prediction).all()
    assert calls["local"] > 0
    assert all(calls[key] == 0 for key in (
        "sensor_to_latent", "latent", "sensor_back", "readout"
    ))

    context = a5.model.prepare_condition_context(
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert torch.count_nonzero(context["latents"]) == 0
    assert torch.count_nonzero(context["global_feat"]) == 0

    changed_values = values["obs_values"] + 2.0
    changed = a5.model(
        t,
        values["x1"],
        values["coords"],
        values["obs_coords"],
        changed_values,
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert not torch.allclose(prediction, changed)


@pytest.mark.skipif(
    os.environ.get("PHYCOFLOW_RUN_REAL_ABLATION_SMOKE") != "1",
    reason="set PHYCOFLOW_RUN_REAL_ABLATION_SMOKE=1 for two-batch GPU smoke",
)
def test_real_two_batch_gpu_smoke_checkpoint_roundtrip(tmp_path: Path):
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 2
    device = torch.device("cuda:1")
    configs = _configs()
    data_path = ROOT / configs["A1"]["data"]
    stats = torch.load(
        ROOT / configs["A1"]["dataset_stats_path"], map_location="cpu", weights_only=True
    )
    with h5py.File(data_path, "r") as handle:
        raw_coords = torch.from_numpy(
            handle["coordinates"][:, 0, 0, :].astype(np.float32)
        )
        coords = (raw_coords - raw_coords.amin(0)) / (
            raw_coords.amax(0) - raw_coords.amin(0)
        ).clamp_min(1.0e-8)
        fields = [
            torch.from_numpy(handle["fields"][0, index, :, 0, 0, :].astype(np.float32))
            for index in (0, 1)
        ]
    fields = [(value - stats["mean"]) / stats["std"] for value in fields]
    query_idx = torch.linspace(0, coords.shape[0] - 1, 64).long()
    obs_idx = torch.linspace(0, coords.shape[0] - 1, 192).long()

    for ablation_id, config in configs.items():
        smoke_config = copy.deepcopy(config)
        smoke_config["neighbor_backend"] = "torch"
        model = build_pointcloud_model(smoke_config, n_fields=5, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-6)
        for snapshot in fields:
            values = {
                "x1": snapshot[query_idx].unsqueeze(0).to(device),
                "coords": coords[query_idx].unsqueeze(0).to(device),
                "obs_coords": coords[obs_idx].unsqueeze(0).to(device),
                "obs_values": snapshot[obs_idx, 2:3].unsqueeze(0).to(device),
                "obs_mask": torch.ones(1, len(obs_idx), device=device),
                "obs_field_ids": torch.full(
                    (1, len(obs_idx)), 2, dtype=torch.long, device=device
                ),
                "obs_indices": None,
            }
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model.training_loss(**values)
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
        model.eval()
        output = model.sample(
            coords=values["coords"],
            obs_coords=values["obs_coords"],
            obs_values=values["obs_values"],
            obs_mask=values["obs_mask"],
            obs_field_ids=values["obs_field_ids"],
            n_steps=1,
            obs_consistency_mode="none",
        )
        assert torch.isfinite(output).all()
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "ablation": model.ablation_metadata,
        }
        checkpoint_path = tmp_path / f"{ablation_id}.pt"
        torch.save(checkpoint, checkpoint_path)
        loaded = torch.load(checkpoint_path, map_location=device, weights_only=False)
        restored = build_pointcloud_model(smoke_config, n_fields=5, device=device)
        restored.load_state_dict(loaded["model"])
        assert loaded["ablation"]["ablation_id"] == ablation_id
        del model, restored, optimizer, loaded
        torch.cuda.empty_cache()
