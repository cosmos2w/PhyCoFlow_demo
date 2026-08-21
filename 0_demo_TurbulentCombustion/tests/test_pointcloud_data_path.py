from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from helpers import TurbulentCombustionH5Dataset
from pointcloud_data_path import (
    DataPathDiagnostics,
    PointCloudBatchCollator,
    materialize_queries_from_full,
    materialize_selected_batch,
    materialize_sparse_condition_from_layout,
    resolve_data_path_config,
    sample_query_indices,
    sample_sparse_observation_indices,
    sample_unique_indices_scalable,
)
from train_pointcloud_ffm import build_pointcloud_loader, run_epoch


@pytest.fixture()
def tiny_h5(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.h5"
    n_times, n_points, n_fields = 8, 101, 3
    coords = np.stack(
        [
            np.linspace(-1.0, 1.0, n_points),
            np.linspace(2.0, 4.0, n_points),
            np.zeros(n_points),
        ],
        axis=-1,
    ).astype(np.float32)
    fields = np.arange(n_times * n_points * n_fields, dtype=np.float32).reshape(
        n_times, n_points, n_fields
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coordinates", data=coords[:, None, None, :])
        handle.create_dataset("fields", data=fields[None, :, :, None, None, :])
        handle.create_dataset("time", data=np.arange(n_times, dtype=np.float32))
    return path


def _dataset(path: Path, *, coord_mode: str, defer: bool) -> TurbulentCombustionH5Dataset:
    return TurbulentCombustionH5Dataset(
        str(path),
        split="train",
        train_ratio=0.75,
        seed=7,
        field_names=["a", "b", "c"],
        stats_path=str(path.with_suffix(".stats.pt")),
        coord_batch_mode=coord_mode,
        defer_field_read=defer,
    )


def test_profile_resolution_and_incompatible_overrides():
    legacy = resolve_data_path_config({"data_path_mode": "legacy"})
    assert legacy.coord_batch_mode == "legacy_clone"
    assert legacy.sampling_device == "legacy_gpu"
    assert legacy.field_normalization_mode == "legacy_full_after_read"
    assert legacy.gpu_transfer_mode == "legacy_full"
    assert legacy.data_path_diag_storage_mode == "legacy_rewrite"
    assert legacy.training_log_every_n_steps == 1

    optimized = resolve_data_path_config({"data_path_mode": "optimized"})
    assert optimized.coord_batch_mode == "shared_mesh"
    assert optimized.index_sampling_mode == "scalable"
    assert optimized.field_read_mode == "legacy_full_snapshot"
    assert optimized.field_normalization_mode == "selected_after_full_read"
    assert optimized.gpu_transfer_mode == "selected_only"
    assert optimized.data_path_diag_storage_mode == "append"

    hybrid = resolve_data_path_config(
        {"data_path_mode": "legacy", "coord_batch_mode": "shared_mesh"}
    )
    assert hybrid.coord_batch_mode == "shared_mesh"
    assert hybrid.index_sampling_mode == "legacy_randperm"

    with pytest.raises(ValueError, match="indexed_union requires"):
        resolve_data_path_config(
            {"data_path_mode": "legacy", "field_read_mode": "indexed_union"}
        )
    with pytest.raises(ValueError, match="scalable.*requires sampling_device='cpu'"):
        resolve_data_path_config(
            {"data_path_mode": "legacy", "index_sampling_mode": "scalable"}
        )
    with pytest.raises(ValueError, match="indexed_union.*requires.*selected_after_full_read"):
        resolve_data_path_config({
            "data_path_mode": "optimized",
            "field_read_mode": "indexed_union",
            "field_normalization_mode": "legacy_full_after_read",
        })


def test_scalable_sampler_is_unique_sorted_reproducible_at_million_scale():
    first = sample_unique_indices_scalable(
        1_000_000, 100_000, generator=torch.Generator().manual_seed(123)
    )
    second = sample_unique_indices_scalable(
        1_000_000, 100_000, generator=torch.Generator().manual_seed(123)
    )
    assert torch.equal(first, second)
    assert first.shape == (100_000,)
    assert torch.all(first[1:] > first[:-1])
    assert int(first[0]) >= 0 and int(first[-1]) < 1_000_000


def test_full_query_indices_own_contiguous_storage_for_pin_memory():
    indices = sample_query_indices(
        batch_size=3,
        n_full=101,
        n_query=None,
        query_sampling="uniform",
        index_sampling_mode="scalable",
    )
    assert indices.shape == (3, 101)
    assert indices.is_contiguous()
    assert indices.stride() == (101, 1)


def test_shared_mesh_items_do_not_clone_coordinates(tiny_h5: Path):
    legacy = _dataset(tiny_h5, coord_mode="legacy_clone", defer=False)
    shared = _dataset(tiny_h5, coord_mode="shared_mesh", defer=True)
    legacy_item = legacy[0]
    shared_item = shared[0]
    assert "coords" in legacy_item and "coords_raw" in legacy_item
    assert "coords" not in shared_item and "coords_raw" not in shared_item
    assert "fields" not in shared_item
    assert shared.fixed_mesh is True
    full = shared.get_full_snapshot(0)
    assert full["coords"].data_ptr() == shared.coords.data_ptr()
    assert full["coords_raw"].data_ptr() == shared.coords_raw.data_ptr()
    assert full["fields"].shape == (shared.num_points, shared.num_fields)


@pytest.mark.parametrize(
    ("field_read_mode", "field_normalization_mode"),
    [
        ("legacy_full_snapshot", "legacy_full_after_read"),
        ("legacy_full_snapshot", "selected_after_full_read"),
        ("indexed_union", "selected_after_full_read"),
    ],
)
def test_selected_materialization_matches_full_path_for_identical_indices(
    tiny_h5: Path, field_read_mode: str, field_normalization_mode: str
):
    full_ds = _dataset(tiny_h5, coord_mode="legacy_clone", defer=False)
    selected_ds = _dataset(tiny_h5, coord_mode="shared_mesh", defer=True)
    items_full = [full_ds[0], full_ds[1]]
    items_selected = [selected_ds[0], selected_ds[1]]
    generator = torch.Generator().manual_seed(99)
    layout = sample_sparse_observation_indices(
        batch_size=2,
        n_full=full_ds.num_points,
        cond_fields=[0, 2],
        n_obs_min=[5, 7],
        n_obs_max=[8, 10],
        index_sampling_mode="scalable",
        generator=generator,
    )
    queries = sample_query_indices(
        batch_size=2,
        n_full=full_ds.num_points,
        n_query=31,
        query_sampling="uniform",
        index_sampling_mode="scalable",
        generator=generator,
    )

    fields_full = torch.stack([item["fields"] for item in items_full])
    coords_full = torch.stack([item["coords"] for item in items_full])
    expected_sparse = materialize_sparse_condition_from_layout(coords_full, fields_full, layout)
    expected_coords_q, expected_fields_q = materialize_queries_from_full(
        coords_full, fields_full, queries
    )
    actual = materialize_selected_batch(
        dataset=selected_ds,
        items=items_selected,
        query_indices=queries,
        obs_layout=layout,
        field_read_mode=field_read_mode,
        field_normalization_mode=field_normalization_mode,
    )

    assert torch.equal(actual["query_indices"], queries)
    assert torch.equal(actual["obs_indices"], layout["obs_indices"])
    assert torch.equal(actual["obs_field_ids"], layout["obs_field_ids"])
    assert torch.equal(actual["obs_mask"].bool(), layout["obs_mask"])
    assert torch.allclose(actual["coords_q"], expected_coords_q, rtol=0, atol=0)
    assert torch.allclose(actual["fields_q"], expected_fields_q, rtol=1e-6, atol=1e-6)
    assert torch.allclose(actual["obs_coords"], expected_sparse["obs_coords"], rtol=0, atol=0)
    assert torch.allclose(actual["obs_values"], expected_sparse["obs_values"], rtol=1e-6, atol=1e-6)


def test_optimized_collator_outputs_selected_tensors_only(tiny_h5: Path):
    dataset = _dataset(tiny_h5, coord_mode="shared_mesh", defer=True)
    config = resolve_data_path_config({"data_path_mode": "optimized"})
    collator = PointCloudBatchCollator(
        dataset=dataset,
        config=config,
        cond_fields=[1],
        n_obs_min=[4],
        n_obs_max=[6],
        n_query_points=17,
        query_sampling="uniform",
    )
    batch = collator([dataset[0], dataset[1]])
    assert batch["materialized_selected"] is True
    assert batch["coords_q"].shape == (2, 17, 3)
    assert batch["fields_q"].shape == (2, 17, 3)
    assert "coords" not in batch and "coords_shared" not in batch and "fields" not in batch
    assert batch["n_full"] == 101


class _SmokeFFM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.requires_full_grid = False

    def training_loss(self, *, x1, coords, obs_values, **kwargs):
        # Exercise every selected floating input while keeping the smoke model tiny.
        if self.training:
            assert self.scale.grad is None, "zero_grad must run before the standard forward"
        loss = (self.scale * x1).square().mean()
        loss = loss + 0.0 * coords.sum() + 0.0 * obs_values.sum()
        return loss, {}


_TEST_DEVICES = [torch.device("cpu")]
if torch.cuda.is_available():
    _TEST_DEVICES.append(torch.device("cuda"))


@pytest.mark.parametrize("profile", ["legacy", "optimized"])
@pytest.mark.parametrize("device", _TEST_DEVICES, ids=str)
def test_training_loop_smoke_and_diagnostics(
    tiny_h5: Path, tmp_path: Path, profile: str, device: torch.device
):
    cfg = resolve_data_path_config({
        "data_path_mode": profile,
        "data_path_diagnostics": True,
        "data_path_diag_every_n_steps": 1,
        "data_path_diag_warmup_steps": 0,
        "data_path_diag_max_steps_per_epoch": 2,
    })
    dataset = _dataset(
        tiny_h5,
        coord_mode=cfg.coord_batch_mode,
        defer=(cfg.sampling_device == "cpu"),
    )
    dataset.instrument_data_path = True
    args = SimpleNamespace(
        cond_fields=[1],
        n_obs_min_list=[4],
        n_obs_max_list=[6],
        n_query_points=17,
        query_sampling="uniform",
        query_sample_near_ratio=0.25,
        query_sample_far_ratio=0.25,
        query_sample_sigma_ratio=0.05,
        batch_size=2,
        num_workers=0,
        backbone="GL_rbf_ENH",
    )
    loader = build_pointcloud_loader(
        dataset, args, cfg, training=True, shuffle=False
    )
    model = _SmokeFFM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output_dir = tmp_path / f"{profile}-{device.type}"
    diagnostics = DataPathDiagnostics(output_dir, cfg)
    loss = run_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        device=device,
        cond_fields=args.cond_fields,
        n_obs_min_list=args.n_obs_min_list,
        n_obs_max_list=args.n_obs_max_list,
        n_query_points=args.n_query_points,
        query_sampling=args.query_sampling,
        epoch=1,
        data_path_config=cfg,
        diagnostics=diagnostics,
    )
    assert np.isfinite(loss)
    assert len(diagnostics.rows) == 2
    assert (output_dir / "data_path_diagnostics.csv").exists()
    if profile == "legacy":
        assert (output_dir / "data_path_diagnostics.json").exists()
    else:
        assert (output_dir / "data_path_diagnostics.jsonl").exists()
        assert (output_dir / "data_path_diagnostics_summary.json").exists()
    required = {
        "loader_wait_ms", "index_sampling_ms", "hdf5_read_ms",
        "cpu_normalization_ms", "cpu_materialization_ms", "h2d_ms",
        "sparse_condition_materialization_ms", "query_materialization_ms",
        "pre_model_total_ms", "model_forward_ms", "backward_ms",
        "optimizer_ms", "total_training_step_ms", "gpu_peak_allocated_mb",
        "gpu_peak_reserved_mb", "allocated_before_model_mb",
    }
    assert required.issubset(diagnostics.rows[0])
    if device.type == "cuda":
        assert diagnostics.rows[0]["gpu_peak_allocated_mb"] > 0


def test_append_diagnostics_are_bounded_and_do_not_rewrite_history(tmp_path: Path):
    cfg = resolve_data_path_config({
        "data_path_mode": "optimized",
        "data_path_diagnostics": True,
        "data_path_diag_every_n_steps": 1,
        "data_path_diag_warmup_steps": 0,
        "data_path_diag_max_steps_per_epoch": 2,
    })
    diagnostics = DataPathDiagnostics(tmp_path, cfg)
    diagnostics.record({"epoch": 1, "step": 0, "latency_ms": 2.0})
    diagnostics.record({"epoch": 1, "step": 1, "latency_ms": 4.0})
    diagnostics.flush()
    diagnostics.record({"epoch": 2, "step": 0, "latency_ms": 6.0})
    diagnostics.flush()

    assert len(diagnostics.rows) == 1
    assert diagnostics.rows[0]["epoch"] == 2
    with open(tmp_path / "data_path_diagnostics.csv") as handle:
        assert len(handle.readlines()) == 4  # one header plus three appended rows
    with open(tmp_path / "data_path_diagnostics.jsonl") as handle:
        assert len(handle.readlines()) == 3
    with open(tmp_path / "data_path_diagnostics_summary.json") as handle:
        summary = __import__("json").load(handle)
    assert summary["latest_epoch"]["samples"] == 1
    assert summary["cumulative"]["samples"] == 3
    assert summary["cumulative"]["mean"]["latency_ms"] == pytest.approx(4.0)
