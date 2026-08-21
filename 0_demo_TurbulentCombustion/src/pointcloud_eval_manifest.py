"""Deterministic validation manifests for matched PointCloud FFM evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from pointcloud_data_path import sample_query_indices, sample_sparse_observation_indices


MANIFEST_VERSION = 1


def _int_list(value: int | Sequence[int]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def dataset_split_fingerprint(dataset: Any) -> dict[str, Any]:
    """Return a compact fingerprint for the exact dataset split and mesh."""
    path = Path(dataset.h5_path).resolve()
    stat = path.stat()
    split_indices = np.asarray(dataset.indices, dtype=np.int64)
    split_hash = hashlib.sha256(split_indices.tobytes()).hexdigest()
    coords_hash = hashlib.sha256(dataset.coords.contiguous().numpy().tobytes()).hexdigest()
    return {
        "path": str(path),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "split": str(dataset.split),
        "split_size": int(len(dataset)),
        "split_indices_sha256": split_hash,
        "coordinates_sha256": coords_hash,
        "num_times": int(dataset.num_times),
        "num_points": int(dataset.num_points),
        "num_fields": int(dataset.num_fields),
        "time_stride": int(dataset.time_stride),
        "field_names": list(dataset.field_names),
    }


def _update_checksum(digest: Any, value: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value):
            if key == "checksum_sha256":
                continue
            digest.update(str(key).encode())
            _update_checksum(digest, value[key])
    elif isinstance(value, (list, tuple)):
        for item in value:
            _update_checksum(digest, item)
    else:
        digest.update(json.dumps(value, sort_keys=True, default=str).encode())


def manifest_checksum(manifest: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    _update_checksum(digest, manifest)
    return digest.hexdigest()


def generate_validation_manifest(
    dataset: Any,
    *,
    n_query_points: int,
    cond_fields: int | Sequence[int],
    n_obs_min: int | Sequence[int],
    n_obs_max: int | Sequence[int],
    seed: int,
    num_samples: int | None = None,
    query_sampling: str = "uniform",
    index_sampling_mode: str = "scalable",
) -> dict[str, Any]:
    """Generate fixed query/observation layouts for a deterministic split prefix."""
    if dataset.split not in {"val", "test"}:
        raise ValueError(f"Validation manifests require a val/test split, got {dataset.split!r}.")
    sample_count = len(dataset) if num_samples is None else min(int(num_samples), len(dataset))
    if sample_count < 1:
        raise ValueError("num_samples must select at least one validation sample.")
    if query_sampling != "uniform":
        raise ValueError("Stage-1 fixed manifests currently support query_sampling='uniform'.")

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    obs_layout = sample_sparse_observation_indices(
        batch_size=sample_count,
        n_full=dataset.num_points,
        cond_fields=cond_fields,
        n_obs_min=n_obs_min,
        n_obs_max=n_obs_max,
        index_sampling_mode=index_sampling_mode,
        generator=generator,
    )
    query_indices = sample_query_indices(
        batch_size=sample_count,
        n_full=dataset.num_points,
        n_query=n_query_points,
        query_sampling=query_sampling,
        index_sampling_mode=index_sampling_mode,
        generator=generator,
    )
    sample_indices = torch.arange(sample_count, dtype=torch.long)
    time_indices = torch.as_tensor(
        np.asarray(dataset.indices[:sample_count], dtype=np.int64), dtype=torch.long
    )
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "dataset": dataset_split_fingerprint(dataset),
        "sampling": {
            "seed": int(seed),
            "n_query_points": int(query_indices.shape[1]),
            "cond_fields": _int_list(cond_fields),
            "n_obs_min": _int_list(n_obs_min),
            "n_obs_max": _int_list(n_obs_max),
            "query_sampling": query_sampling,
            "index_sampling_mode": index_sampling_mode,
        },
        "sample_indices": sample_indices,
        "time_indices": time_indices,
        "query_indices": query_indices,
        "obs_indices": obs_layout["obs_indices"],
        "obs_field_ids": obs_layout["obs_field_ids"],
        "obs_mask": obs_layout["obs_mask"],
        "obs_counts_by_field": obs_layout["obs_counts_by_field"],
    }
    manifest["checksum_sha256"] = manifest_checksum(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any], dataset: Any | None = None) -> None:
    if int(manifest.get("manifest_version", -1)) != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported manifest version {manifest.get('manifest_version')!r}; "
            f"expected {MANIFEST_VERSION}."
        )
    expected = str(manifest.get("checksum_sha256", ""))
    actual = manifest_checksum(manifest)
    if expected != actual:
        raise ValueError(f"Manifest checksum mismatch: stored={expected}, computed={actual}.")
    sample_count = int(manifest["sample_indices"].numel())
    for key in (
        "time_indices",
        "query_indices",
        "obs_indices",
        "obs_field_ids",
        "obs_mask",
        "obs_counts_by_field",
    ):
        if int(manifest[key].shape[0]) != sample_count:
            raise ValueError(f"Manifest tensor {key!r} has an inconsistent sample dimension.")
    if dataset is not None:
        current = dataset_split_fingerprint(dataset)
        stored = manifest["dataset"]
        stable_keys = (
            "path",
            "size_bytes",
            "split",
            "split_size",
            "split_indices_sha256",
            "coordinates_sha256",
            "num_points",
            "num_fields",
            "time_stride",
            "field_names",
        )
        mismatches = {
            key: (stored.get(key), current.get(key))
            for key in stable_keys
            if stored.get(key) != current.get(key)
        }
        if mismatches:
            raise ValueError(f"Manifest dataset fingerprint mismatch: {mismatches}.")


def save_validation_manifest(manifest: Mapping[str, Any], path: str | Path) -> tuple[Path, Path]:
    validate_manifest(manifest)
    output = Path(path)
    if output.suffix != ".pt":
        output = output.with_suffix(".pt")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(manifest), output)
    summary_path = output.with_suffix(".json")
    summary = {
        "manifest_version": int(manifest["manifest_version"]),
        "checksum_sha256": manifest["checksum_sha256"],
        "dataset": manifest["dataset"],
        "sampling": manifest["sampling"],
        "num_samples": int(manifest["sample_indices"].numel()),
        "query_shape": list(manifest["query_indices"].shape),
        "observation_shape": list(manifest["obs_indices"].shape),
        "observation_count_min": int(manifest["obs_mask"].sum(dim=1).min()),
        "observation_count_max": int(manifest["obs_mask"].sum(dim=1).max()),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return output, summary_path


def load_validation_manifest(path: str | Path, dataset: Any | None = None) -> dict[str, Any]:
    manifest = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(manifest, dict):
        raise TypeError(f"Expected a dictionary manifest, got {type(manifest).__name__}.")
    validate_manifest(manifest, dataset=dataset)
    return manifest


def slice_manifest_layout(manifest: Mapping[str, Any], start: int, end: int) -> dict[str, torch.Tensor]:
    return {
        "obs_indices": manifest["obs_indices"][start:end],
        "obs_field_ids": manifest["obs_field_ids"][start:end],
        "obs_mask": manifest["obs_mask"][start:end],
        "obs_counts_by_field": manifest["obs_counts_by_field"][start:end],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> None:
    import yaml

    from helpers import TurbulentCombustionH5Dataset

    args = _parse_args()
    config = yaml.safe_load(args.config.read_text()) or {}
    dataset = TurbulentCombustionH5Dataset(
        str(config["data"]),
        split="val",
        train_ratio=float(config.get("train_ratio", 0.9)),
        seed=int(config.get("seed", 42)),
        time_stride=int(config.get("time_stride", 1)),
        field_names=config.get("FIELD_NAMES", config.get("field_names")),
        stats_path=config.get("dataset_stats_path"),
        coord_batch_mode="shared_mesh",
        defer_field_read=True,
    )
    manifest = generate_validation_manifest(
        dataset,
        n_query_points=int(config.get("n_query_points", dataset.num_points)),
        cond_fields=config.get("cond_fields", [config.get("cond_field", 0)]),
        n_obs_min=config.get("n_obs_min_list", [config.get("n_obs_min", 16)]),
        n_obs_max=config.get("n_obs_max_list", [config.get("n_obs_max", 16)]),
        seed=args.seed,
        num_samples=args.num_samples,
        query_sampling="uniform",
        index_sampling_mode="scalable",
    )
    pt_path, json_path = save_validation_manifest(manifest, args.output)
    print(f"Saved manifest: {pt_path}")
    print(f"Saved summary: {json_path}")
    print(f"Checksum: {manifest['checksum_sha256']}")


if __name__ == "__main__":
    main()
