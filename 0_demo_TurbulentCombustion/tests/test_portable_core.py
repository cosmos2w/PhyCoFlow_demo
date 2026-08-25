from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phycoflow_pointcloud import (  # noqa: E402
    ReconstructionConfig,
    build_pointcloud_model,
    reconstruct_from_tensors,
)
from phycoflow_pointcloud.checkpointing import checkpoint_model_state  # noqa: E402
from phycoflow_pointcloud.evaluation import model_schema_digest  # noqa: E402


STAGE8_SHA = "7b5dd2df672a86d5b5cf2502b93208786d282e17"
EXPECTED_SCHEMA = "f221ee2b26268fe64e116f9f57165d34aac9ff524202e40d225ed17f76dc970e"
PORTABLE_PACKAGE = SRC / "phycoflow_pointcloud"


def _small_config(coord_dim: int, *, cached: bool = True) -> dict:
    return {
        "model_name": "GL_rbf_CQ",
        "backbone": "GL_rbf_ENH_CQ",
        "coord_dim": coord_dim,
        "prior": "iid",
        "hidden_dim": 16,
        "cond_dim": 8,
        "field_embed_dim": 4,
        "latent_dim": 16,
        "num_latents": 8,
        "num_heads": 4,
        "num_latent_blocks": 2,
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
        "latent_reinject_every": 1,
        "condition_attention_execution": "cached_kv" if cached else "legacy_mha",
        "sensor_attention_padding_mode": "full",
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


def _inputs(coord_dim: int, n_fields: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(8201 + coord_dim + n_fields)
    coords = torch.rand(2, 17, coord_dim, generator=generator)
    obs_indices = torch.tensor([[0, 2, 5, 8, 12, 16], [1, 3, 6, 9, 13, 15]])
    obs_mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0]], dtype=torch.float32)
    obs_field_ids = torch.arange(6).remainder(n_fields).repeat(2, 1)
    return {
        "x1": torch.randn(2, 17, n_fields, generator=generator),
        "coords": coords,
        "obs_coords": torch.stack([coords[i, obs_indices[i]] for i in range(2)]),
        "obs_values": torch.randn(2, 6, 1, generator=generator),
        "obs_mask": obs_mask,
        "obs_field_ids": obs_field_ids,
        "obs_indices": obs_indices,
    }


def test_portable_source_has_no_demo_or_monolith_imports():
    forbidden = {
        "Model",
        "helpers",
        "model_baseline",
        "coherence_dist",
        "persistent_topk_geometry_cache",
        "obs_consistency",
        "model_ema",
        "train_pointcloud_ffm",
        "evaluate_pointcloud_fixed_manifest",
    }
    violations = []
    for path in PORTABLE_PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots = [node.module.split(".", 1)[0]]
            else:
                continue
            for root in roots:
                if root in forbidden:
                    violations.append(f"{path.relative_to(ROOT)} imports {root}")
        text = path.read_text()
        for marker in ("TurbulentCombustionH5Dataset", "Dataset/", "matplotlib"):
            if marker in text:
                violations.append(f"{path.relative_to(ROOT)} contains {marker}")
    assert violations == []


def test_declared_package_imports_and_builds_in_isolation(tmp_path: Path):
    isolated_src = tmp_path / "src"
    manifest = yaml.safe_load((ROOT / "GL_rbf_CQ_RELEASE_MANIFEST.yaml").read_text())
    for relative in manifest["portable_core_files"]:
        source = ROOT / relative
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(ROOT / "src/Model.py", isolated_src / "Model.py")
    script = r'''
import json
import sys
import torch
sys.modules["pykeops"] = None
import Model
from phycoflow_pointcloud import build_pointcloud_model
from phycoflow_pointcloud.cache import PersistentTopKGeometryCache
from phycoflow_pointcloud.models import GL_rbf_CQ

config = {
    "model_name": "GL_rbf_CQ", "backbone": "GL_rbf_ENH_CQ",
    "coord_dim": 2, "prior": "iid", "hidden_dim": 16, "cond_dim": 8,
    "field_embed_dim": 4, "latent_dim": 16, "num_latents": 8,
    "num_heads": 4, "num_latent_blocks": 1, "ff_mult": 2,
    "summary_type": "mean", "gather_mode": "topk_rbf_glres",
    "gather_topk": 3, "neighbor_backend": "torch", "USE_FOURIER_PE": True,
    "fourier_pe_num_bands": 2, "fourier_pe_max_freq": 4.0,
    "sensor_coord_encoding": "fourier", "latent_sensor_reinject": True,
    "condition_attention_execution": "cached_kv",
    "sensor_attention_padding_mode": "full", "cq_query_dim": 8,
    "cq_readout_mode": "lowrank", "cq_readout_rank": 4,
    "cq_readout_heads": 2, "cq_fusion_mode": "additive",
}
model = build_pointcloud_model(config, n_fields=3, device="cpu").eval()
coords = torch.rand(1, 9, 2)
obs_coords = coords[:, [0, 3, 7]]
out = model.model(
    torch.tensor([0.25]), torch.randn(1, 9, 3), coords, obs_coords,
    torch.randn(1, 3, 1), torch.ones(1, 3), torch.tensor([[0, 1, 2]]),
)
forbidden = ["helpers", "model_baseline", "coherence_dist", "_legacy_model_full"]
assert not any(name in sys.modules for name in forbidden)
assert Model.ConditionalPointHybridLocalGlobalRBFCQ is GL_rbf_CQ
assert "neuralop" not in sys.modules
print(json.dumps({"shape": list(out.shape), "module": GL_rbf_CQ.__module__}))
'''
    env = dict(os.environ)
    env["PYTHONPATH"] = str(isolated_src)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout.strip().splitlines()[-1])
    assert payload["shape"] == [1, 9, 3]
    assert payload["module"] == "phycoflow_pointcloud.models.portable_core"


@pytest.mark.parametrize(("coord_dim", "n_fields"), [(2, 1), (2, 3), (3, 5)])
def test_synthetic_forward_backward_microbatch_and_cached_kv(coord_dim: int, n_fields: int):
    torch.manual_seed(301)
    model = build_pointcloud_model(_small_config(coord_dim), n_fields=n_fields)
    micro = copy.deepcopy(model)
    values = _inputs(coord_dim, n_fields)

    model.model.input_cross_attn.reset_execution_counters()
    prediction = model.model(
        torch.tensor([0.2, 0.8]),
        values["x1"],
        values["coords"],
        values["obs_coords"],
        values["obs_values"],
        values["obs_mask"],
        values["obs_field_ids"],
    )
    assert prediction.shape == values["x1"].shape
    prediction.square().mean().backward()
    assert model.model.input_cross_attn.kv_projection_calls == 1
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    model.zero_grad(set_to_none=True)
    torch.manual_seed(991)
    full_loss, _ = model.training_loss(**values)
    torch.manual_seed(991)
    micro_loss, metrics = micro.training_loss_microbatched(
        **values,
        query_microbatch_size=5,
        backward=False,
        reuse_condition_context=True,
    )
    torch.testing.assert_close(micro_loss, full_loss.detach(), rtol=3e-6, atol=3e-7)
    assert metrics["query_microbatches"] == 4.0


def test_persistent_topk_reconstruction_has_zero_post_build_knn_calls():
    torch.manual_seed(411)
    model = build_pointcloud_model(_small_config(3), n_fields=3).eval()
    values = _inputs(3, 3)
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
    cfg = ReconstructionConfig(
        n_steps=2,
        obs_consistency_mode="none",
        execution_mode="cached_streamed",
        query_chunk_size=5,
        cache_level="static_features",
    )
    tensor_values = {key: value for key, value in values.items() if key != "x1"}
    torch.manual_seed(717)
    fresh = reconstruct_from_tensors(model, **tensor_values, config=cfg)
    torch.manual_seed(717)
    persistent = reconstruct_from_tensors(
        model, **tensor_values, config=cfg, geometry_cache=geometry
    )
    torch.testing.assert_close(persistent, fresh, rtol=1e-5, atol=2e-6)
    assert build_calls > 0
    assert calls == build_calls + build_calls  # fresh rebuilds once; persistent adds zero.


def test_core_config_matches_validated_demo_scientific_model():
    core = yaml.safe_load((ROOT / "configs/gl_rbf_cq_core.yaml").read_text())
    demo = yaml.safe_load((ROOT / "configs/gl_rbf_cq.yaml").read_text())
    for key, expected in core.items():
        assert demo[key] == expected, key
    assert not {
        "data", "dataset_stats_path", "save_dir", "Demo_Num", "device_ids",
        "FIELD_NAMES", "field_names", "cond_fields", "n_obs_min_list",
        "n_obs_max_list", "train_ratio", "batch_size", "epochs",
    }.intersection(core)
    torch.manual_seed(520)
    core_model = build_pointcloud_model(core, n_fields=5)
    torch.manual_seed(520)
    demo_model = build_pointcloud_model(demo, n_fields=5)
    assert core_model.state_dict().keys() == demo_model.state_dict().keys()
    for key in core_model.state_dict():
        torch.testing.assert_close(
            core_model.state_dict()[key], demo_model.state_dict()[key], rtol=0, atol=0
        )


def test_release_manifest_is_complete_and_checkpoint_hash_matches():
    manifest = yaml.safe_load((ROOT / "GL_rbf_CQ_RELEASE_MANIFEST.yaml").read_text())
    assert manifest["source"]["stage8_tip_sha"] == STAGE8_SHA
    assert manifest["defaults"]["condition_attention_execution"] == "cached_kv"
    assert manifest["defaults"]["sensor_attention_padding_mode"] == "full"
    for relative in manifest["portable_core_files"]:
        assert (ROOT / relative).is_file(), relative
    for record in manifest["compatibility_files"]:
        assert (ROOT / record["path"]).is_file(), record["path"]
    checkpoint = ROOT / manifest["release_checkpoint"]["path"]
    if checkpoint.exists():
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == manifest["release_checkpoint"]["sha256"]


def _load_stage8_oracle(tmp_path: Path):
    def git_blob(relative: str) -> str:
        return subprocess.check_output(
            ["git", "show", f"{STAGE8_SHA}:0_demo_TurbulentCombustion/{relative}"],
            cwd=REPO_ROOT,
            text=True,
        )

    model_path = tmp_path / "stage8_model.py"
    factory_path = tmp_path / "stage8_factory.py"
    model_path.write_text(git_blob("src/Model.py"))
    factory_path.write_text(git_blob("src/phycoflow_pointcloud/models/factory.py"))

    model_spec = importlib.util.spec_from_file_location("stage8_oracle_model", model_path)
    oracle_model = importlib.util.module_from_spec(model_spec)
    assert model_spec.loader is not None
    model_spec.loader.exec_module(oracle_model)

    previous_model = sys.modules.get("Model")
    sys.modules["Model"] = oracle_model
    try:
        factory_spec = importlib.util.spec_from_file_location(
            "stage8_oracle_factory", factory_path
        )
        oracle_factory = importlib.util.module_from_spec(factory_spec)
        assert factory_spec.loader is not None
        factory_spec.loader.exec_module(oracle_factory)
    finally:
        if previous_model is None:
            sys.modules.pop("Model", None)
        else:
            sys.modules["Model"] = previous_model
    return oracle_factory


def test_release_checkpoint_matches_stage8_source_forward_gradients_and_reconstruction(
    tmp_path: Path,
):
    checkpoint_path = ROOT / (
        "ReleaseArtifacts/GL_rbf_CQ_rc1/"
        "GL_rbf_CQ_v0.9.0-rc1_e1000_ema_resolved_portable.pt"
    )
    if not checkpoint_path.exists():
        pytest.skip("External RC1 release checkpoint is unavailable.")
    config = yaml.safe_load((ROOT / "configs/gl_rbf_cq.yaml").read_text())
    config["neighbor_backend"] = "torch"
    oracle_factory = _load_stage8_oracle(tmp_path)

    torch.manual_seed(9917)
    oracle = oracle_factory.build_pointcloud_model(config, n_fields=5, device="cpu")
    torch.manual_seed(9917)
    portable = build_pointcloud_model(config, n_fields=5, device="cpu")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint_model_state(checkpoint, model=portable)
    oracle.load_state_dict(state, strict=True)
    portable.load_state_dict(state, strict=True)
    assert model_schema_digest(portable) == EXPECTED_SCHEMA
    assert oracle.state_dict().keys() == portable.state_dict().keys()

    generator = torch.Generator().manual_seed(20260825)
    coords = torch.rand(1, 13, 3, generator=generator)
    obs_indices = torch.tensor([[0, 2, 3, 5, 7, 8, 10, 11]])
    inputs = {
        "t": torch.tensor([0.375]),
        "x_t": torch.randn(1, 13, 5, generator=generator),
        "coords": coords,
        "obs_coords": coords[:, obs_indices[0]],
        "obs_values": torch.randn(1, 8, 1, generator=generator),
        "obs_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32),
        "obs_field_ids": torch.tensor([[0, 1, 2, 3, 4, 1, 3, 0]]),
    }
    oracle_out = oracle.model(**inputs)
    portable_out = portable.model(**inputs)
    torch.testing.assert_close(portable_out, oracle_out, rtol=0, atol=0)

    oracle_loss = F.mse_loss(oracle_out, torch.zeros_like(oracle_out))
    portable_loss = F.mse_loss(portable_out, torch.zeros_like(portable_out))
    oracle_loss.backward()
    portable_loss.backward()
    torch.testing.assert_close(portable_loss, oracle_loss, rtol=0, atol=0)
    for (oracle_name, oracle_parameter), (portable_name, portable_parameter) in zip(
        oracle.named_parameters(), portable.named_parameters()
    ):
        assert oracle_name == portable_name
        assert (oracle_parameter.grad is None) == (portable_parameter.grad is None)
        if oracle_parameter.grad is not None:
            torch.testing.assert_close(
                portable_parameter.grad, oracle_parameter.grad, rtol=0, atol=0
            )

    sample_kwargs = {
        "coords": coords,
        "obs_coords": inputs["obs_coords"],
        "obs_values": inputs["obs_values"],
        "obs_mask": inputs["obs_mask"],
        "obs_field_ids": inputs["obs_field_ids"],
        "clamp_indices": obs_indices,
        "n_steps": 2,
        "ode_solver": "euler",
        "obs_consistency_mode": "none",
        "reconstruction_execution_mode": "cached_streamed",
        "reconstruction_query_chunk_size": 5,
        "reconstruction_cache_level": "static_features",
    }
    oracle.eval()
    portable.eval()
    torch.manual_seed(1729)
    oracle_reconstruction = oracle.sample(**sample_kwargs)
    torch.manual_seed(1729)
    portable_reconstruction = portable.sample(**sample_kwargs)
    torch.testing.assert_close(portable_reconstruction, oracle_reconstruction, rtol=0, atol=0)


def test_historical_model_module_reexports_portable_classes():
    import Model
    from phycoflow_pointcloud.models.portable_core import (
        ConditionalPointHybridLocalGlobalRBF,
        ConditionalPointHybridLocalGlobalRBFCQ,
        PointCloudFFM,
    )

    assert Model.ConditionalPointHybridLocalGlobalRBF is ConditionalPointHybridLocalGlobalRBF
    assert Model.ConditionalPointHybridLocalGlobalRBFCQ is ConditionalPointHybridLocalGlobalRBFCQ
    assert Model.PointCloudFFM is PointCloudFFM
