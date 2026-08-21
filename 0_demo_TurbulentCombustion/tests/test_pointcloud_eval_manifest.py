from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_pointcloud_fixed_manifest import reset_rf_rng
from helpers import TurbulentCombustionH5Dataset
from Model import PointCloudFFM
from pointcloud_eval_manifest import (
    generate_validation_manifest,
    load_validation_manifest,
    save_validation_manifest,
    validate_manifest,
)
from train_pointcloud_ffm import RFFGaussianPrior


@pytest.fixture()
def manifest_dataset(tmp_path: Path) -> TurbulentCombustionH5Dataset:
    path = tmp_path / "manifest.h5"
    n_times, n_points, n_fields = 10, 47, 3
    coords = np.stack(
        [
            np.linspace(-1.0, 1.0, n_points),
            np.linspace(1.0, 2.0, n_points),
            np.linspace(-0.25, 0.25, n_points),
        ],
        axis=-1,
    ).astype(np.float32)
    fields = np.random.default_rng(5).normal(size=(n_times, n_points, n_fields)).astype(np.float32)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("coordinates", data=coords[:, None, None, :])
        handle.create_dataset("fields", data=fields[None, :, :, None, None, :])
        handle.create_dataset("time", data=np.arange(n_times, dtype=np.float32))
    return TurbulentCombustionH5Dataset(
        str(path),
        split="val",
        train_ratio=0.6,
        seed=11,
        field_names=["a", "b", "c"],
        stats_path=str(tmp_path / "stats.pt"),
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )


def _manifest(dataset: TurbulentCombustionH5Dataset) -> dict:
    return generate_validation_manifest(
        dataset,
        n_query_points=19,
        cond_fields=[0, 2],
        n_obs_min=[3, 4],
        n_obs_max=[5, 7],
        seed=12345,
        num_samples=4,
    )


def test_manifest_generation_is_reproducible_and_roundtrips(
    manifest_dataset: TurbulentCombustionH5Dataset, tmp_path: Path
):
    first = _manifest(manifest_dataset)
    second = _manifest(manifest_dataset)
    assert first["checksum_sha256"] == second["checksum_sha256"]
    for key in (
        "sample_indices",
        "time_indices",
        "query_indices",
        "obs_indices",
        "obs_field_ids",
        "obs_mask",
        "obs_counts_by_field",
    ):
        assert torch.equal(first[key], second[key])

    output, summary = save_validation_manifest(first, tmp_path / "fixed.pt")
    loaded = load_validation_manifest(output, dataset=manifest_dataset)
    assert loaded["checksum_sha256"] == first["checksum_sha256"]
    assert summary.exists()


def test_manifest_checksum_detects_tensor_mutation(manifest_dataset: TurbulentCombustionH5Dataset):
    manifest = _manifest(manifest_dataset)
    manifest["query_indices"][0, 0] = (manifest["query_indices"][0, 0] + 1) % manifest_dataset.num_points
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_manifest(manifest, dataset=manifest_dataset)


class _ToyVelocity(torch.nn.Module):
    n_fields = 2

    def forward(self, t, x_t, coords, obs_coords, obs_values, obs_mask, obs_field_ids):
        return 0.25 * x_t + t[:, None, None] + 0.0 * coords[..., :1]


def test_controlled_rf_rng_reproduces_loss_and_source_draws():
    torch.manual_seed(7)
    ffm = PointCloudFFM(
        _ToyVelocity(), RFFGaussianPrior(coord_dim=3, n_features=16, lengthscale=0.2)
    )
    coords = torch.randn(2, 31, 3)
    x1 = torch.randn(2, 31, 2)
    obs_coords = coords[:, :5]
    obs_values = x1[:, :5, :1]
    obs_mask = torch.ones(2, 5)
    obs_field_ids = torch.zeros(2, 5, dtype=torch.long)

    reset_rf_rng(991, torch.device("cpu"))
    first, first_metrics = ffm.training_loss(
        x1, coords, obs_coords, obs_values, obs_mask, obs_field_ids
    )
    reset_rf_rng(991, torch.device("cpu"))
    second, second_metrics = ffm.training_loss(
        x1, coords, obs_coords, obs_values, obs_mask, obs_field_ids
    )
    assert torch.equal(first, second)
    assert first_metrics == second_metrics
