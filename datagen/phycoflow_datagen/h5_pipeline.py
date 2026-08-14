"""Streaming conversion from raw trajectories to the unified PhyCoFlow HDF5 schema."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from .cases import CASES
from .diagnostics import derive_kolmogorov_fields
from .storage import SCHEMA_VERSION, list_raw_trajectories, load_raw_trajectory, sha256_file


def _split_indices(count: int, ratios: tuple[float, float, float], seed: int):
    if count < 1:
        raise ValueError("at least one trajectory is required")
    ratios_array = np.asarray(ratios, dtype=np.float64)
    if np.any(ratios_array < 0) or not np.isclose(ratios_array.sum(), 1.0):
        raise ValueError("split ratios must be nonnegative and sum to 1")
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(count)
    n_train = max(1, int(np.floor(ratios_array[0] * count)))
    remaining = count - n_train
    n_validation = min(remaining, int(np.floor(ratios_array[1] * count)))
    return (
        np.sort(permutation[:n_train]),
        np.sort(permutation[n_train : n_train + n_validation]),
        np.sort(permutation[n_train + n_validation :]),
    )


def _canonical_fields(case: str, state: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    if case == "kolmogorov":
        return derive_kolmogorov_fields(state, config)
    return np.asarray(state)


def _field_layout(fields: np.ndarray) -> np.ndarray:
    """Convert `[T,C,*space]` to `[T,N,1,1,C]`."""

    if fields.ndim not in {3, 4}:
        raise ValueError(f"expected fields [T,C,X] or [T,C,Y,X], got {fields.shape}")
    channel_last = np.moveaxis(fields, 1, -1)
    return channel_last.reshape(fields.shape[0], -1, 1, 1, fields.shape[1])


def _coordinates(arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, list[int]]:
    x = np.asarray(arrays["x"], dtype=np.float64)
    if "y" not in arrays:
        coordinates = np.stack((x, np.zeros_like(x), np.zeros_like(x)), axis=-1)
        return coordinates[:, None, None, :].astype(np.float32), [x.size]
    y = np.asarray(arrays["y"], dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    coordinates = np.stack((xx, yy, np.zeros_like(xx)), axis=-1).reshape(-1, 3)
    return coordinates[:, None, None, :].astype(np.float32), [y.size, x.size]


def _compression_kwargs(compression: str) -> dict[str, Any]:
    if compression == "none":
        return {}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": 4, "shuffle": True}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    raise ValueError(f"unsupported compression {compression!r}")


def _validate_consistency(
    arrays: dict[str, np.ndarray],
    metadata: dict[str, Any],
    reference_arrays: dict[str, np.ndarray],
    reference_metadata: dict[str, Any],
) -> None:
    for key in ("state", "time", "step", "x"):
        if arrays[key].shape != reference_arrays[key].shape:
            raise ValueError(f"inconsistent {key} shape: {arrays[key].shape} vs {reference_arrays[key].shape}")
    if ("y" in arrays) != ("y" in reference_arrays):
        raise ValueError("inconsistent spatial dimension among trajectories")
    if "y" in arrays and arrays["y"].shape != reference_arrays["y"].shape:
        raise ValueError("inconsistent y grid among trajectories")
    for key in ("time", "step", "x", "y"):
        if key in arrays and not np.allclose(arrays[key], reference_arrays[key], rtol=0.0, atol=1.0e-12):
            raise ValueError(f"inconsistent {key} values among trajectories")
    if metadata["case"] != reference_metadata["case"]:
        raise ValueError("raw directory mixes multiple physical cases")
    reference_config = reference_metadata["config"]
    for key in (
        "resolution",
        "domain_length",
        "dt",
        "burn_in_time",
        "record_time",
        "save_every",
    ):
        if metadata["config"][key] != reference_config[key]:
            raise ValueError(f"inconsistent configuration key {key!r} among trajectories")


def validate_h5(path: Path, expected_case: str | None = None) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        required = {"fields", "coordinates", "time", "conditions", "trajectory_id", "seed", "splits"}
        missing = required.difference(handle.keys())
        if missing:
            raise ValueError(f"HDF5 file is missing required entries: {sorted(missing)}")
        fields = handle["fields"]
        if fields.ndim != 6 or fields.shape[3:5] != (1, 1):
            raise ValueError(f"invalid fields layout {fields.shape}")
        if handle["coordinates"].shape != (fields.shape[2], 1, 1, 3):
            raise ValueError("coordinates do not match fields point count")
        if handle["time"].shape != (fields.shape[1],):
            raise ValueError("time axis does not match fields")
        if handle["conditions"].shape[0] != fields.shape[0]:
            raise ValueError("conditions do not match trajectory count")
        case = str(handle.attrs["case_name"])
        if expected_case is not None and case != expected_case:
            raise ValueError(f"expected case {expected_case}, found {case}")
        for index in range(fields.shape[0]):
            if not np.isfinite(fields[index]).all():
                raise ValueError(f"non-finite values in processed trajectory {index}")
        split_values = np.concatenate(
            [handle[f"splits/{name}"][:] for name in ("train", "validation", "test")]
        )
        if sorted(split_values.tolist()) != list(range(fields.shape[0])):
            raise ValueError("split indices are not a disjoint exhaustive trajectory partition")
        return {
            "case": case,
            "shape": list(fields.shape),
            "dtype": str(fields.dtype),
            "field_names": json.loads(handle.attrs["field_names"]),
            "file_bytes": path.stat().st_size,
        }


def _write_dataset_readme(
    output_path: Path,
    summary: dict[str, Any],
    split_sizes: dict[str, int],
    raw_dir: Path,
    command: str,
) -> Path:
    readme_path = output_path.with_name(f"{output_path.stem}_README.md")
    fields = ", ".join(summary["field_names"])
    robust_statistics = ""
    if "channel_offset" in summary:
        robust_statistics = (
            "\nThe `statistics/channel_offset` and `statistics/channel_scale_99` "
            "datasets provide the fixed training-only robust normalization used "
            "by the multiphysics notebooks.\n"
        )
    content = f"""# {summary['case']} processed dataset

- Source raw directory: `{raw_dir}`
- HDF5 file: `{output_path.name}`
- Unified `fields` shape: `{summary['shape']}`
- Field channels: `{fields}`
- Storage dtype: `{summary['dtype']}`
- Split trajectories: train={split_sizes['train']}, validation={split_sizes['validation']}, test={split_sizes['test']}
- Schema version: `{SCHEMA_VERSION}`
- SHA-256: `{summary['sha256']}`

Core datasets are `fields [B,T,N,1,1,C]`, `coordinates [N,1,1,3]`,
`time [T]`, and `conditions [B,P]`. Fields and coordinates are stored in
physical units. Normalization statistics use training trajectories only.
{robust_statistics}

Generation command recorded by the processor:

```bash
{command}
```
"""
    temporary_path = readme_path.with_suffix(readme_path.suffix + ".tmp")
    temporary_path.write_text(content, encoding="utf-8")
    os.replace(temporary_path, readme_path)
    return readme_path


def process_raw_to_h5(
    case: str,
    raw_dir: Path,
    output_path: Path,
    *,
    split_ratios: tuple[float, float, float],
    split_seed: int,
    compression: str,
    include_auxiliary: bool,
    overwrite: bool,
    progress_factory,
    command: str,
) -> dict[str, Any]:
    paths = list_raw_trajectories(raw_dir)
    first_arrays, first_metadata = load_raw_trajectory(paths[0])
    if first_metadata["case"] != case:
        raise ValueError(f"wrapper expects case {case}, raw data contain {first_metadata['case']}")
    config = first_metadata["config"]
    first_fields = _canonical_fields(case, first_arrays["state"], config)
    first_layout = _field_layout(first_fields)
    coordinates, grid_shape = _coordinates(first_arrays)
    count, n_time, n_points, _, _, n_channels = (
        len(paths),
        *first_layout.shape,
    )
    condition_names = CASES[case]["condition_names"]
    train_indices, validation_indices, test_indices = _split_indices(count, split_ratios, split_seed)
    train_set = set(train_indices.tolist())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite intentionally")
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    compression_kwargs = _compression_kwargs(compression)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    diagnostic_rows: list[dict[str, Any]] = []
    running_sum = np.zeros(n_channels, dtype=np.float64)
    running_sum_sq = np.zeros(n_channels, dtype=np.float64)
    running_count = 0

    try:
        with h5py.File(temporary_path, "w") as handle:
            fields_dataset = handle.create_dataset(
                "fields",
                shape=(count, n_time, n_points, 1, 1, n_channels),
                dtype=np.float32,
                chunks=(1, 1, n_points, 1, 1, n_channels),
                fletcher32=True,
                **compression_kwargs,
            )
            coordinates_dataset = handle.create_dataset(
                "coordinates", data=coordinates, dtype=np.float32
            )
            time_dataset = handle.create_dataset(
                "time", data=first_arrays["time"], dtype=np.float64
            )
            conditions_dataset = handle.create_dataset(
                "conditions", shape=(count, len(condition_names)), dtype=np.float32
            )
            fields_dataset.attrs["B"] = count
            fields_dataset.attrs["C"] = n_channels
            fields_dataset.attrs["Nt"] = n_time
            fields_dataset.attrs["Nx"] = n_points
            fields_dataset.attrs["Ny"] = 1
            fields_dataset.attrs["Nz"] = 1
            periodic_axes = CASES[case].get(
                "periodic_axes", ["x"] if len(grid_shape) == 1 else ["x", "y"]
            )
            fields_dataset.attrs["mesh"] = (
                "structured periodic grid flattened in C order"
                if periodic_axes
                else "structured boundary-inclusive nonperiodic grid flattened in C order"
            )
            fields_dataset.attrs["field_names"] = json.dumps(CASES[case]["field_names"])
            fields_dataset.attrs["field_units"] = json.dumps(
                CASES[case].get("field_units", ["dimensionless"] * n_channels)
            )
            fields_dataset.attrs["grid_shape"] = json.dumps(grid_shape)
            coordinates_dataset.attrs["axis_order"] = "x,y,z"
            coordinates_dataset.attrs["units"] = CASES[case].get(
                "coordinate_units", "dimensionless physical coordinates"
            )
            time_dataset.attrs["units"] = CASES[case].get(
                "time_units", "dimensionless physical time"
            )
            conditions_dataset.attrs["condition_names"] = json.dumps(condition_names)
            conditions_dataset.attrs["units"] = json.dumps(
                CASES[case].get(
                    "condition_units", ["dimensionless"] * len(condition_names)
                )
            )
            trajectory_ids = handle.create_dataset("trajectory_id", shape=(count,), dtype=string_dtype)
            seeds = handle.create_dataset("seed", shape=(count,), dtype=np.int64)
            if case == "kolmogorov" and include_auxiliary:
                auxiliary = handle.create_dataset(
                    "auxiliary/vorticity",
                    shape=(count, n_time, n_points, 1, 1, 1),
                    dtype=np.float32,
                    chunks=(1, 1, n_points, 1, 1, 1),
                    fletcher32=True,
                    **compression_kwargs,
                )
            elif case == "electro_thermal" and include_auxiliary:
                auxiliary = {
                    "ellipse_mask": handle.create_dataset(
                        "auxiliary/ellipse_mask",
                        shape=(count, n_points, 1, 1, 1),
                        dtype=np.uint8,
                        chunks=(1, n_points, 1, 1, 1),
                        fletcher32=True,
                        **compression_kwargs,
                    )
                }
                for name in ("conductivity", "joule_heating", "thermal_conductivity"):
                    auxiliary[name] = handle.create_dataset(
                        f"auxiliary/{name}",
                        shape=(count, n_points, 1, 1, 1),
                        dtype=np.float32,
                        chunks=(1, n_points, 1, 1, 1),
                        fletcher32=True,
                        **compression_kwargs,
                    )
            elif case == "mass_transport_fluid" and include_auxiliary:
                auxiliary = {
                    "source_field": handle.create_dataset(
                        "auxiliary/source_field",
                        shape=(count, n_points, 1, 1, 1),
                        dtype=np.float32,
                        chunks=(1, n_points, 1, 1, 1),
                        fletcher32=True,
                        **compression_kwargs,
                    ),
                    "pressure": handle.create_dataset(
                        "auxiliary/pressure",
                        shape=(count, n_time, n_points, 1, 1, 1),
                        dtype=np.float32,
                        chunks=(1, 1, n_points, 1, 1, 1),
                        fletcher32=True,
                        **compression_kwargs,
                    ),
                }
            else:
                auxiliary = None

            iterable = enumerate(paths)
            if progress_factory is not None:
                iterable = progress_factory(iterable, total=count, desc="Post-processing trajectories")
            for index, path in iterable:
                arrays, metadata = (first_arrays, first_metadata) if index == 0 else load_raw_trajectory(path)
                _validate_consistency(arrays, metadata, first_arrays, first_metadata)
                fields = _canonical_fields(case, arrays["state"], metadata["config"])
                layout = _field_layout(fields).astype(np.float32, copy=False)
                fields_dataset[index] = layout
                if case == "kolmogorov" and auxiliary is not None:
                    auxiliary[index] = _field_layout(arrays["state"]).astype(np.float32, copy=False)
                elif case == "electro_thermal" and auxiliary is not None:
                    for name, dataset in auxiliary.items():
                        values = np.asarray(arrays[name]).reshape(n_points, 1, 1, 1)
                        dataset[index] = values.astype(dataset.dtype, copy=False)
                elif case == "mass_transport_fluid" and auxiliary is not None:
                    auxiliary["source_field"][index] = np.asarray(
                        arrays["source_field"]
                    ).reshape(n_points, 1, 1, 1)
                    auxiliary["pressure"][index] = _field_layout(
                        np.asarray(arrays["pressure"])[:, None]
                    ).astype(np.float32, copy=False)
                condition_mapping = metadata.get("conditions") or metadata["config"]
                conditions_dataset[index] = np.asarray(
                    [condition_mapping[name] for name in condition_names], dtype=np.float32
                )
                trajectory_ids[index] = f"trajectory_{int(metadata['trajectory_id']):06d}"
                seeds[index] = int(metadata["seed"])
                diagnostic_rows.append(metadata.get("diagnostics", {}))
                if index in train_set:
                    flat = np.moveaxis(fields.astype(np.float64), 1, -1).reshape(-1, n_channels)
                    running_sum += flat.sum(axis=0)
                    running_sum_sq += np.square(flat).sum(axis=0)
                    running_count += flat.shape[0]

            splits_group = handle.require_group("splits")
            splits_group.create_dataset("train", data=train_indices, dtype=np.int64)
            splits_group.create_dataset("validation", data=validation_indices, dtype=np.int64)
            splits_group.create_dataset("test", data=test_indices, dtype=np.int64)
            mean = running_sum / running_count
            variance = np.maximum(running_sum_sq / running_count - mean**2, 0.0)
            standard_deviation = np.sqrt(variance)
            minimum_resolved_scale = np.maximum(np.abs(mean) * 1.0e-12, 1.0e-30)
            standard_deviation = np.where(
                standard_deviation > minimum_resolved_scale,
                standard_deviation,
                1.0,
            )
            statistics = handle.require_group("statistics")
            statistics.create_dataset("train_mean", data=mean, dtype=np.float64)
            statistics.create_dataset(
                "train_std", data=standard_deviation, dtype=np.float64
            )
            if case in {"electro_thermal", "mass_transport_fluid"}:
                training_values = np.asarray(
                    fields_dataset[train_indices], dtype=np.float64
                )
                if case == "electro_thermal":
                    channel_offset = mean.copy()
                    channel_offset[2] = float(config["ambient_temperature"])
                else:
                    channel_offset = mean.copy()
                    channel_offset[2] = 0.0
                centered = np.abs(
                    training_values
                    - channel_offset.reshape(1, 1, 1, 1, 1, n_channels)
                )
                channel_scale = np.quantile(
                    centered.reshape(-1, n_channels), 0.99, axis=0
                )
                channel_scale = np.maximum(channel_scale, 1.0e-30)
                if case == "mass_transport_fluid":
                    channel_scale[2] = float(config["surface_concentration"])
                statistics.create_dataset(
                    "channel_offset", data=channel_offset, dtype=np.float64
                )
                statistics.create_dataset(
                    "channel_scale_99", data=channel_scale, dtype=np.float64
                )
            else:
                channel_offset = None
                channel_scale = None

            metric_names = sorted({name for row in diagnostic_rows for name in row})
            diagnostics_group = handle.require_group("diagnostics")
            for metric_name in metric_names:
                values = []
                for row in diagnostic_rows:
                    value = row.get(metric_name)
                    values.append(np.nan if value is None else float(value))
                diagnostics_group.create_dataset(metric_name, data=np.asarray(values, dtype=np.float64))

            metadata_group = handle.require_group("metadata")
            metadata_group.create_dataset(
                "json",
                data=json.dumps(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "source_raw_directory": str(raw_dir.resolve()),
                        "resolved_config": config,
                        "split_ratios": split_ratios,
                        "split_seed": split_seed,
                        "processing_command": command,
                    },
                    sort_keys=True,
                ),
                dtype=string_dtype,
            )
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["case_name"] = case
            handle.attrs["display_name"] = CASES[case]["display_name"]
            handle.attrs["equation"] = CASES[case]["equation"]
            handle.attrs["field_names"] = json.dumps(CASES[case]["field_names"])
            handle.attrs["field_units"] = json.dumps(
                CASES[case].get(
                    "field_units", ["dimensionless"] * len(CASES[case]["field_names"])
                )
            )
            handle.attrs["state_names"] = json.dumps(CASES[case]["state_names"])
            handle.attrs["condition_names"] = json.dumps(condition_names)
            handle.attrs["grid_shape"] = json.dumps(grid_shape)
            handle.attrs["spatial_dimension"] = CASES[case]["spatial_dimension"]
            handle.attrs["domain_length"] = float(config["domain_length"])
            handle.attrs["periodic_axes"] = json.dumps(
                CASES[case].get(
                    "periodic_axes", ["x"] if len(grid_shape) == 1 else ["x", "y"]
                )
            )
            handle.attrs["coordinate_order"] = "C order; x varies fastest"
            handle.attrs["pressure_gauge"] = (
                "zero spatial mean"
                if case in {"kolmogorov", "mass_transport_fluid"}
                else "not applicable"
            )
            handle.attrs["source_dataset_id"] = str(config["dataset_id"])
            handle.attrs["code_commit"] = str(first_metadata.get("code_commit", "unknown"))
            handle.attrs["compression"] = compression
            handle.flush()

        os.replace(temporary_path, output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    summary = validate_h5(output_path, expected_case=case)
    summary["sha256"] = sha256_file(output_path)
    split_sizes = {
        "train": train_indices.size,
        "validation": validation_indices.size,
        "test": test_indices.size,
    }
    summary["splits"] = split_sizes
    summary["train_mean"] = mean.tolist()
    summary["train_std"] = standard_deviation.tolist()
    if channel_offset is not None:
        summary["channel_offset"] = channel_offset.tolist()
        summary["channel_scale_99"] = channel_scale.tolist()
    summary["readme"] = str(
        _write_dataset_readme(output_path, summary, split_sizes, raw_dir, command)
    )
    return summary
