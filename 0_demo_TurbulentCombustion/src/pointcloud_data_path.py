"""Explicit legacy/optimized data paths for point-cloud FFM training.

This module intentionally keeps the candidate implementation separate from the
historical trainer path so the latter can be deleted cleanly after A/B testing.
It changes data movement only; model and Rectified Flow mathematics live
elsewhere.
"""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_CHOICES = {
    "data_path_mode": {"legacy", "optimized"},
    "coord_batch_mode": {"legacy_clone", "shared_mesh"},
    "index_sampling_mode": {"legacy_randperm", "scalable"},
    "sampling_device": {"legacy_gpu", "cpu"},
    "field_read_mode": {"legacy_full_snapshot", "indexed_union"},
    "field_normalization_mode": {"legacy_full_after_read", "selected_after_full_read"},
    "gpu_transfer_mode": {"legacy_full", "selected_only"},
    "data_path_diag_storage_mode": {"legacy_rewrite", "append"},
}

_PROFILE_DEFAULTS = {
    "legacy": {
        "coord_batch_mode": "legacy_clone",
        "index_sampling_mode": "legacy_randperm",
        "sampling_device": "legacy_gpu",
        "field_read_mode": "legacy_full_snapshot",
        "field_normalization_mode": "legacy_full_after_read",
        "gpu_transfer_mode": "legacy_full",
        "data_path_diag_storage_mode": "legacy_rewrite",
        "dataloader_persistent_workers": False,
        "dataloader_prefetch_factor": None,
        "non_blocking_transfer": False,
        "training_log_every_n_steps": 1,
    },
    "optimized": {
        "coord_batch_mode": "shared_mesh",
        "index_sampling_mode": "scalable",
        "sampling_device": "cpu",
        # Contiguous HDF5 layouts often favor a full sequential read.  Keep the
        # indexed union as an explicit benchmark dimension, not an assumption.
        "field_read_mode": "legacy_full_snapshot",
        "field_normalization_mode": "selected_after_full_read",
        "gpu_transfer_mode": "selected_only",
        "data_path_diag_storage_mode": "append",
        "dataloader_persistent_workers": True,
        "dataloader_prefetch_factor": 2,
        "non_blocking_transfer": True,
        "training_log_every_n_steps": 20,
    },
}


@dataclass(frozen=True)
class ResolvedDataPathConfig:
    data_path_mode: str
    coord_batch_mode: str
    index_sampling_mode: str
    sampling_device: str
    field_read_mode: str
    field_normalization_mode: str
    gpu_transfer_mode: str
    data_path_diag_storage_mode: str
    dataloader_persistent_workers: bool
    dataloader_prefetch_factor: int | None
    non_blocking_transfer: bool
    data_path_diagnostics: bool
    data_path_diag_every_n_steps: int
    data_path_diag_warmup_steps: int
    data_path_diag_max_steps_per_epoch: int
    training_log_every_n_steps: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get(source: Any, key: str, default: Any = None) -> Any:
    if isinstance(source, Mapping):
        return source.get(key, default)
    return getattr(source, key, default)


def resolve_data_path_config(source: Any) -> ResolvedDataPathConfig:
    """Resolve profile defaults once; explicit non-None values always win."""
    profile = str(_get(source, "data_path_mode", "legacy"))
    if profile not in _CHOICES["data_path_mode"]:
        raise ValueError(f"data_path_mode must be one of {sorted(_CHOICES['data_path_mode'])}.")
    defaults = _PROFILE_DEFAULTS[profile]

    values: dict[str, Any] = {"data_path_mode": profile}
    for key in (
        "coord_batch_mode",
        "index_sampling_mode",
        "sampling_device",
        "field_read_mode",
        "field_normalization_mode",
        "gpu_transfer_mode",
        "data_path_diag_storage_mode",
        "dataloader_persistent_workers",
        "dataloader_prefetch_factor",
        "non_blocking_transfer",
        "training_log_every_n_steps",
    ):
        explicit = _get(source, key, None)
        values[key] = defaults[key] if explicit is None else explicit

    for key, allowed in _CHOICES.items():
        if key == "data_path_mode":
            continue
        if values[key] not in allowed:
            raise ValueError(f"{key} must be one of {sorted(allowed)}, got {values[key]!r}.")

    values["dataloader_persistent_workers"] = bool(values["dataloader_persistent_workers"])
    values["non_blocking_transfer"] = bool(values["non_blocking_transfer"])
    if values["dataloader_prefetch_factor"] is not None:
        values["dataloader_prefetch_factor"] = int(values["dataloader_prefetch_factor"])
        if values["dataloader_prefetch_factor"] <= 0:
            raise ValueError("dataloader_prefetch_factor must be positive or null.")

    values.update(
        data_path_diagnostics=bool(_get(source, "data_path_diagnostics", False)),
        data_path_diag_every_n_steps=max(1, int(_get(source, "data_path_diag_every_n_steps", 50))),
        data_path_diag_warmup_steps=max(0, int(_get(source, "data_path_diag_warmup_steps", 5))),
        data_path_diag_max_steps_per_epoch=max(
            0, int(_get(source, "data_path_diag_max_steps_per_epoch", 10))
        ),
        training_log_every_n_steps=max(1, int(values["training_log_every_n_steps"])),
    )

    if values["gpu_transfer_mode"] == "selected_only" and values["sampling_device"] != "cpu":
        raise ValueError("selected_only GPU transfer requires sampling_device='cpu'.")
    if values["index_sampling_mode"] == "scalable" and values["sampling_device"] != "cpu":
        raise ValueError(
            "index_sampling_mode='scalable' requires sampling_device='cpu'; the explicit "
            "legacy_gpu path intentionally retains historical randperm sampling."
        )
    if values["field_read_mode"] == "indexed_union" and (
        values["sampling_device"] != "cpu" or values["gpu_transfer_mode"] != "selected_only"
    ):
        raise ValueError(
            "indexed_union requires sampling_device='cpu' and gpu_transfer_mode='selected_only'."
        )
    if (
        values["field_read_mode"] == "indexed_union"
        and values["field_normalization_mode"] != "selected_after_full_read"
    ):
        raise ValueError(
            "indexed_union reads only selected rows and therefore requires "
            "field_normalization_mode='selected_after_full_read'."
        )

    return ResolvedDataPathConfig(**values)


def apply_resolved_data_path_config(args: Any, config: ResolvedDataPathConfig) -> Any:
    """Persist resolved values in argparse state so args.json is self-describing."""
    for key, value in config.to_dict().items():
        setattr(args, key, value)
    args.data_path_resolved = config.to_dict()
    return args


def print_resolved_data_path_config(config: ResolvedDataPathConfig) -> None:
    print("[*] PointCloudFFM data path:")
    labels = (
        ("profile", config.data_path_mode),
        ("coord_batch_mode", config.coord_batch_mode),
        ("index_sampling_mode", config.index_sampling_mode),
        ("sampling_device", config.sampling_device),
        ("field_read_mode", config.field_read_mode),
        ("normalization_mode", config.field_normalization_mode),
        ("gpu_transfer_mode", config.gpu_transfer_mode),
        ("diagnostic_storage", config.data_path_diag_storage_mode),
        ("persistent_workers", config.dataloader_persistent_workers),
        ("prefetch_factor", config.dataloader_prefetch_factor),
        ("non_blocking_transfer", config.non_blocking_transfer),
    )
    for key, value in labels:
        print(f"    {key:<22}= {value}")


# =====================================================================
# LEGACY DATA PATH — BEGIN
# Temporary A/B reference implementation.
# Remove after optimized path is validated.
# =====================================================================
def sample_unique_indices_legacy(
    n_full: int,
    n_select: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    sort: bool = True,
) -> torch.Tensor:
    n_select = min(max(0, int(n_select)), int(n_full))
    result = torch.randperm(int(n_full), device=device, generator=generator)[:n_select]
    return result.sort().values if sort else result
# =====================================================================
# LEGACY DATA PATH — END
# =====================================================================


# =====================================================================
# OPTIMIZED DATA PATH — BEGIN
# Candidate production implementation.
# =====================================================================
def sample_unique_indices_scalable(
    n_full: int,
    n_select: int,
    *,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    sort: bool = True,
) -> torch.Tensor:
    """Sample without replacement with O(K)-scale work in the sparse regime.

    Random integers are oversampled and deduplicated until K unique values have
    been collected.  For dense selections, where the output itself is O(N), a
    randperm fallback avoids rejection-sampling degeneration.
    """
    n_full = int(n_full)
    n_select = min(max(0, int(n_select)), n_full)
    device = torch.device(device)
    if n_select == 0:
        return torch.empty(0, dtype=torch.long, device=device)
    if n_select == n_full:
        return torch.arange(n_full, dtype=torch.long, device=device)
    if n_select > n_full // 3:
        result = torch.randperm(n_full, device=device, generator=generator)[:n_select]
        return result.sort().values if sort else result

    selected = torch.empty(0, dtype=torch.long, device=device)
    while selected.numel() < n_select:
        remaining = n_select - selected.numel()
        draw_count = max(remaining + 8, math.ceil(remaining * 1.35))
        draw = torch.randint(0, n_full, (draw_count,), device=device, generator=generator)
        selected = torch.unique(torch.cat((selected, draw)), sorted=True)
        if selected.numel() > n_select:
            # Randomly choose which of the final excess unique candidates remain.
            keep = torch.randperm(selected.numel(), device=device, generator=generator)[:n_select]
            selected = selected.index_select(0, keep)
    if sort:
        selected = selected.sort().values
    return selected


def sample_unique_indices(
    n_full: int,
    n_select: int,
    *,
    mode: str,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
    sort: bool = True,
) -> torch.Tensor:
    if mode == "legacy_randperm":
        return sample_unique_indices_legacy(
            n_full, n_select, device=device, generator=generator, sort=sort
        )
    if mode == "scalable":
        return sample_unique_indices_scalable(
            n_full, n_select, device=device, generator=generator, sort=sort
        )
    raise ValueError(f"Unknown index sampling mode: {mode!r}.")


def _to_int_list(value: int | Sequence[int]) -> list[int]:
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def _broadcast(values: int | Sequence[int], size: int, name: str) -> list[int]:
    result = _to_int_list(values)
    if len(result) == 1:
        result *= size
    if len(result) != size:
        raise ValueError(f"{name} must have length 1 or {size}, got {len(result)}.")
    return result


def sample_sparse_observation_indices(
    *,
    batch_size: int,
    n_full: int,
    cond_fields: int | Sequence[int],
    n_obs_min: int | Sequence[int],
    n_obs_max: int | Sequence[int],
    index_sampling_mode: str,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Sample observation counts/layout on CPU without any CUDA scalar sync."""
    cond_fields = _to_int_list(cond_fields)
    if not cond_fields:
        raise ValueError("cond_fields must contain at least one field index.")
    mins = _broadcast(n_obs_min, len(cond_fields), "n_obs_min")
    maxs = _broadcast(n_obs_max, len(cond_fields), "n_obs_max")
    if any(high < low for low, high in zip(mins, maxs)):
        raise ValueError("Each n_obs_max must be >= n_obs_min.")
    if any(high > int(n_full) for high in maxs):
        raise ValueError("Observation count cannot exceed the number of mesh points.")

    max_obs = sum(maxs)
    indices = torch.zeros((batch_size, max_obs), dtype=torch.long)
    field_ids = torch.full((batch_size, max_obs), -1, dtype=torch.long)
    mask = torch.zeros((batch_size, max_obs), dtype=torch.bool)
    counts_by_field = torch.stack(
        [
            torch.randint(low, high + 1, (batch_size,), generator=generator)
            if high > low
            else torch.full((batch_size,), low, dtype=torch.long)
            for low, high in zip(mins, maxs)
        ],
        dim=1,
    )

    for batch_idx, row_counts in enumerate(counts_by_field.tolist()):
        cursor = 0
        for field_id, count in zip(cond_fields, row_counts):
            selected = sample_unique_indices(
                n_full,
                count,
                mode=index_sampling_mode,
                device="cpu",
                generator=generator,
                sort=True,
            )
            end = cursor + count
            indices[batch_idx, cursor:end] = selected
            field_ids[batch_idx, cursor:end] = int(field_id)
            mask[batch_idx, cursor:end] = True
            cursor = end
    return {
        "obs_indices": indices,
        "obs_field_ids": field_ids,
        "obs_mask": mask,
        "obs_counts_by_field": counts_by_field,
    }


def _weighted_query_indices(
    coords: torch.Tensor,
    obs_indices: torch.Tensor,
    n_query: int,
    *,
    generator: torch.Generator | None,
    near_ratio: float,
    far_ratio: float,
    sigma_ratio: float,
) -> torch.Tensor:
    """CPU equivalent of historical obs_mix; intentionally O(N_full x M)."""
    obs_coords = coords.index_select(0, obs_indices)
    d_min = torch.cdist(coords.unsqueeze(0), obs_coords.unsqueeze(0), p=2.0).squeeze(0).amin(dim=-1)
    bbox_diag = (coords.amax(dim=0) - coords.amin(dim=0)).norm().clamp_min(1e-6)
    sigma = (float(sigma_ratio) * bbox_diag).clamp_min(1e-6)
    near_count = min(n_query, max(0, round(n_query * near_ratio)))
    far_count = min(n_query - near_count, max(0, round(n_query * far_ratio)))
    uniform_count = n_query - near_count - far_count
    selected = torch.zeros(coords.shape[0], dtype=torch.bool)
    pieces = []

    def take(weights: torch.Tensor, count: int) -> None:
        available_count = int((~selected).sum())
        count = min(int(count), available_count)
        if count <= 0:
            return
        weights = weights.clamp_min(0).masked_fill(selected, 0)
        positive_count = int((weights > 0).sum())
        if positive_count:
            weighted_count = min(count, positive_count)
            idx = torch.multinomial(weights, weighted_count, replacement=False, generator=generator)
            pieces.append(idx)
            selected[idx] = True
            count -= weighted_count
        if count:
            available = (~selected).nonzero(as_tuple=False).squeeze(-1)
            take_pos = sample_unique_indices_scalable(
                available.numel(), count, generator=generator, sort=False
            )
            idx = available.index_select(0, take_pos)
            pieces.append(idx)
            selected[idx] = True

    take(torch.exp(-(d_min.square()) / (2 * sigma.square() + 1e-12)), near_count)
    take(d_min, far_count)
    take(torch.ones_like(d_min), uniform_count)
    take(torch.ones_like(d_min), n_query - int(selected.sum()))
    return torch.cat(pieces).sort().values


def sample_query_indices(
    *,
    batch_size: int,
    n_full: int,
    n_query: int | None,
    query_sampling: str,
    index_sampling_mode: str,
    coords_shared: torch.Tensor | None = None,
    obs_layout: Mapping[str, torch.Tensor] | None = None,
    generator: torch.Generator | None = None,
    near_ratio: float = 0.25,
    far_ratio: float = 0.25,
    sigma_ratio: float = 0.05,
) -> torch.Tensor:
    count = int(n_full) if n_query is None else min(int(n_query), int(n_full))
    if count >= int(n_full):
        # Pin-memory workers cannot pin an expanded zero-stride view because
        # multiple logical elements alias the same storage.  Own contiguous
        # per-batch index storage for this full-query boundary case.
        return torch.arange(int(n_full), dtype=torch.long).repeat(batch_size, 1)
    rows = []
    for batch_idx in range(batch_size):
        if query_sampling == "uniform":
            row = sample_unique_indices(
                n_full,
                count,
                mode=index_sampling_mode,
                generator=generator,
                sort=True,
            )
        elif query_sampling == "obs_mix":
            if coords_shared is None or obs_layout is None:
                raise ValueError("obs_mix CPU sampling requires shared coordinates and an observation layout.")
            valid = obs_layout["obs_mask"][batch_idx]
            obs_idx = obs_layout["obs_indices"][batch_idx, valid]
            row = _weighted_query_indices(
                coords_shared,
                obs_idx,
                count,
                generator=generator,
                near_ratio=near_ratio,
                far_ratio=far_ratio,
                sigma_ratio=sigma_ratio,
            )
        else:
            raise ValueError(f"Unknown query_sampling: {query_sampling!r}.")
        rows.append(row)
    return torch.stack(rows, dim=0)


def materialize_sparse_condition_from_layout(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    obs_layout: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Gather sparse tensors from already materialized full tensors."""
    indices = obs_layout["obs_indices"].to(coords_full.device)
    field_ids = obs_layout["obs_field_ids"].to(coords_full.device)
    mask_bool = obs_layout["obs_mask"].to(coords_full.device).bool()
    coord_idx = indices.unsqueeze(-1).expand(-1, -1, coords_full.shape[-1])
    obs_coords = torch.gather(coords_full, 1, coord_idx)
    safe_fields = field_ids.clamp_min(0)
    values_at_points = torch.gather(
        fields_full,
        1,
        indices.unsqueeze(-1).expand(-1, -1, fields_full.shape[-1]),
    )
    obs_values = torch.gather(values_at_points, 2, safe_fields.unsqueeze(-1))
    obs_values = obs_values.masked_fill(~mask_bool.unsqueeze(-1), 0)
    obs_coords = obs_coords.masked_fill(~mask_bool.unsqueeze(-1), 0)
    return {
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": mask_bool.to(dtype=coords_full.dtype),
        "obs_indices": indices,
        "obs_field_ids": field_ids,
    }


def materialize_queries_from_full(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    query_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = query_indices.to(coords_full.device)
    coords_q = torch.gather(
        coords_full, 1, indices.unsqueeze(-1).expand(-1, -1, coords_full.shape[-1])
    )
    fields_q = torch.gather(
        fields_full, 1, indices.unsqueeze(-1).expand(-1, -1, fields_full.shape[-1])
    )
    return coords_q, fields_q


def materialize_selected_batch(
    *,
    dataset: Any,
    items: Sequence[Mapping[str, torch.Tensor]],
    query_indices: torch.Tensor,
    obs_layout: Mapping[str, torch.Tensor],
    field_read_mode: str,
    field_normalization_mode: str = "legacy_full_after_read",
) -> dict[str, Any]:
    """Read/normalize/materialize only tensors needed by the model call."""
    if (
        field_read_mode == "indexed_union"
        and field_normalization_mode != "selected_after_full_read"
    ):
        raise ValueError(
            "indexed_union requires field_normalization_mode="
            "'selected_after_full_read'."
        )
    batch_size = len(items)
    max_obs = obs_layout["obs_indices"].shape[1]
    n_query = query_indices.shape[1]
    coord_dim = dataset.coords.shape[-1]
    n_fields = dataset.num_fields
    coords_q = torch.empty((batch_size, n_query, coord_dim), dtype=dataset.coords.dtype)
    fields_q = torch.empty((batch_size, n_query, n_fields), dtype=torch.float32)
    obs_coords = torch.zeros((batch_size, max_obs, coord_dim), dtype=dataset.coords.dtype)
    obs_values = torch.zeros((batch_size, max_obs, 1), dtype=torch.float32)

    hdf5_s = 0.0
    normalize_s = 0.0
    materialize_start = time.perf_counter()
    for batch_idx, item in enumerate(items):
        q_idx = query_indices[batch_idx]
        valid = obs_layout["obs_mask"][batch_idx]
        o_idx = obs_layout["obs_indices"][batch_idx, valid]
        field_ids = obs_layout["obs_field_ids"][batch_idx, valid]
        time_index = int(item["time_index"])

        if field_read_mode == "indexed_union":
            union = torch.unique(torch.cat((q_idx, o_idx)), sorted=True)
            read_start = time.perf_counter()
            raw = dataset.read_fields(time_index, union)
            hdf5_s += time.perf_counter() - read_start
            norm_start = time.perf_counter()
            selected = (raw - dataset.mean) / dataset.std
            normalize_s += time.perf_counter() - norm_start
            q_pos = torch.searchsorted(union, q_idx)
            o_pos = torch.searchsorted(union, o_idx)
            fields_q[batch_idx] = selected.index_select(0, q_pos)
            obs_values[batch_idx, valid, 0] = selected[o_pos, field_ids]
        elif field_read_mode == "legacy_full_snapshot":
            read_start = time.perf_counter()
            raw = dataset.read_fields(time_index)
            hdf5_s += time.perf_counter() - read_start
            norm_start = time.perf_counter()
            if field_normalization_mode == "legacy_full_after_read":
                full = (raw - dataset.mean) / dataset.std
                fields_q[batch_idx] = full.index_select(0, q_idx)
                obs_values[batch_idx, valid, 0] = full[o_idx, field_ids]
            elif field_normalization_mode == "selected_after_full_read":
                union = torch.unique(torch.cat((q_idx, o_idx)), sorted=True)
                raw_selected = raw.index_select(0, union)
                selected = (raw_selected - dataset.mean) / dataset.std
                q_pos = torch.searchsorted(union, q_idx)
                o_pos = torch.searchsorted(union, o_idx)
                fields_q[batch_idx] = selected.index_select(0, q_pos)
                obs_values[batch_idx, valid, 0] = selected[o_pos, field_ids]
            else:
                raise ValueError(
                    f"Unknown field_normalization_mode: {field_normalization_mode!r}."
                )
            normalize_s += time.perf_counter() - norm_start
        else:
            raise ValueError(f"Unknown field_read_mode: {field_read_mode!r}.")

        coords_q[batch_idx] = dataset.coords.index_select(0, q_idx)
        obs_coords[batch_idx, valid] = dataset.coords.index_select(0, o_idx)

    cpu_materialization_s = time.perf_counter() - materialize_start - hdf5_s - normalize_s
    return {
        "materialized_selected": True,
        "coords_q": coords_q,
        "fields_q": fields_q,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_layout["obs_mask"].to(dtype=dataset.coords.dtype),
        "obs_indices": obs_layout["obs_indices"],
        "obs_field_ids": obs_layout["obs_field_ids"],
        "query_indices": query_indices,
        "time_index": torch.stack([item["time_index"] for item in items]),
        "physical_time": torch.stack([item["physical_time"] for item in items]),
        "n_full": int(dataset.num_points),
        "data_path_timings": {
            "hdf5_read_ms": hdf5_s * 1000.0,
            "cpu_normalization_ms": normalize_s * 1000.0,
            "cpu_materialization_ms": max(0.0, cpu_materialization_s * 1000.0),
        },
    }


class PointCloudBatchCollator:
    """CPU sampler/materializer used by optimized and hybrid ablation paths."""

    def __init__(
        self,
        *,
        dataset: Any,
        config: ResolvedDataPathConfig,
        cond_fields: Sequence[int],
        n_obs_min: Sequence[int],
        n_obs_max: Sequence[int],
        n_query_points: int | None,
        query_sampling: str,
        query_sample_near_ratio: float = 0.25,
        query_sample_far_ratio: float = 0.25,
        query_sample_sigma_ratio: float = 0.05,
    ) -> None:
        if config.coord_batch_mode == "shared_mesh" and not bool(dataset.fixed_mesh):
            raise ValueError("shared_mesh collator requires dataset.fixed_mesh=True.")
        self.dataset = dataset
        self.config = config
        self.cond_fields = tuple(int(v) for v in cond_fields)
        self.n_obs_min = tuple(int(v) for v in n_obs_min)
        self.n_obs_max = tuple(int(v) for v in n_obs_max)
        self.n_query_points = n_query_points
        self.query_sampling = str(query_sampling)
        self.query_sample_near_ratio = float(query_sample_near_ratio)
        self.query_sample_far_ratio = float(query_sample_far_ratio)
        self.query_sample_sigma_ratio = float(query_sample_sigma_ratio)

    def __call__(self, items: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, Any]:
        collate_start = time.perf_counter()
        batch_size = len(items)
        timings = {"index_sampling_ms": 0.0, "hdf5_read_ms": 0.0,
                   "cpu_normalization_ms": 0.0, "cpu_materialization_ms": 0.0}
        obs_layout = None
        query_indices = None

        if self.config.sampling_device == "cpu":
            sampling_start = time.perf_counter()
            obs_layout = sample_sparse_observation_indices(
                batch_size=batch_size,
                n_full=self.dataset.num_points,
                cond_fields=self.cond_fields,
                n_obs_min=self.n_obs_min,
                n_obs_max=self.n_obs_max,
                index_sampling_mode=self.config.index_sampling_mode,
            )
            query_indices = sample_query_indices(
                batch_size=batch_size,
                n_full=self.dataset.num_points,
                n_query=self.n_query_points,
                query_sampling=self.query_sampling,
                index_sampling_mode=self.config.index_sampling_mode,
                coords_shared=self.dataset.coords,
                obs_layout=obs_layout,
                near_ratio=self.query_sample_near_ratio,
                far_ratio=self.query_sample_far_ratio,
                sigma_ratio=self.query_sample_sigma_ratio,
            )
            timings["index_sampling_ms"] = (time.perf_counter() - sampling_start) * 1000.0

        if self.config.gpu_transfer_mode == "selected_only":
            batch = materialize_selected_batch(
                dataset=self.dataset,
                items=items,
                query_indices=query_indices,
                obs_layout=obs_layout,
                field_read_mode=self.config.field_read_mode,
                field_normalization_mode=self.config.field_normalization_mode,
            )
            timings.update(batch["data_path_timings"])
            timings["cpu_materialization_ms"] = max(
                0.0,
                (time.perf_counter() - collate_start) * 1000.0
                - timings["index_sampling_ms"]
                - timings["hdf5_read_ms"]
                - timings["cpu_normalization_ms"],
            )
            batch["data_path_timings"] = timings
            return batch

        # Hybrid ablations may sample indices on CPU but still transfer full
        # fields.  Deferred items are read/normalized here so timings are visible.
        fields = []
        read_s = 0.0
        norm_s = 0.0
        for item in items:
            if "fields" in item:
                fields.append(item["fields"])
                continue
            start = time.perf_counter()
            raw = self.dataset.read_fields(int(item["time_index"]))
            read_s += time.perf_counter() - start
            start = time.perf_counter()
            fields.append((raw - self.dataset.mean) / self.dataset.std)
            norm_s += time.perf_counter() - start
        batch = {
            "materialized_selected": False,
            "fields": torch.stack(fields),
            "time_index": torch.stack([item["time_index"] for item in items]),
            "physical_time": torch.stack([item["physical_time"] for item in items]),
            "n_full": int(self.dataset.num_points),
        }
        if self.config.coord_batch_mode == "legacy_clone":
            batch["coords"] = torch.stack([item["coords"] for item in items])
        else:
            batch["coords_shared"] = self.dataset.coords
        if obs_layout is not None:
            batch["obs_layout"] = obs_layout
            batch["query_indices"] = query_indices
        timings["hdf5_read_ms"] = read_s * 1000.0
        timings["cpu_normalization_ms"] = norm_s * 1000.0
        timings["cpu_materialization_ms"] = max(
            0.0,
            (time.perf_counter() - collate_start) * 1000.0
            - timings["index_sampling_ms"]
            - timings["hdf5_read_ms"]
            - timings["cpu_normalization_ms"],
        )
        batch["data_path_timings"] = timings
        return batch
# =====================================================================
# OPTIMIZED DATA PATH — END
# =====================================================================


class DataPathDiagnostics:
    """Sampled step diagnostics persisted as CSV and JSON under the run dir."""

    def __init__(self, save_dir: Path | str, config: ResolvedDataPathConfig) -> None:
        self.save_dir = Path(save_dir)
        self.config = config
        self.rows: list[dict[str, Any]] = []
        self._epoch_samples: dict[int, int] = {}
        self._active_epoch: int | None = None
        self._cumulative_count = 0
        self._cumulative_sums: dict[str, float] = {}
        self._csv_keys: list[str] | None = None

    def should_sample(self, epoch: int, step: int) -> bool:
        if not self.config.data_path_diagnostics:
            return False
        if step < self.config.data_path_diag_warmup_steps:
            return False
        if (step - self.config.data_path_diag_warmup_steps) % self.config.data_path_diag_every_n_steps:
            return False
        count = self._epoch_samples.get(int(epoch), 0)
        return count < self.config.data_path_diag_max_steps_per_epoch

    def record(self, row: Mapping[str, Any]) -> None:
        clean = dict(row)
        epoch = int(clean["epoch"])
        if self.config.data_path_diag_storage_mode == "append":
            if self._active_epoch != epoch:
                self.rows.clear()
                self._epoch_samples.clear()
                self._active_epoch = epoch
        self.rows.append(clean)
        self._epoch_samples[epoch] = self._epoch_samples.get(epoch, 0) + 1
        if self.config.data_path_diag_storage_mode == "append":
            self._append_row(clean)
            self._cumulative_count += 1
            for key, value in clean.items():
                if isinstance(value, (int, float)) and key not in {"epoch", "step"}:
                    self._cumulative_sums[key] = self._cumulative_sums.get(key, 0.0) + float(value)

    def _append_row(self, row: Mapping[str, Any]) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = self.save_dir / "data_path_diagnostics.csv"
        keys = list(row.keys())
        if self._csv_keys is None:
            self._csv_keys = keys
        elif keys != self._csv_keys:
            raise ValueError("Diagnostic append rows must use a stable schema.")
        write_header = not csv_path.exists() or csv_path.stat().st_size == 0
        with open(csv_path, "a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        with open(self.save_dir / "data_path_diagnostics.jsonl", "a") as handle:
            handle.write(json.dumps(dict(row), separators=(",", ":")) + "\n")

    def flush(self) -> None:
        if not self.rows:
            return
        self.save_dir.mkdir(parents=True, exist_ok=True)
        if self.config.data_path_diag_storage_mode == "append":
            numeric_keys = [
                key for key, value in self.rows[0].items()
                if isinstance(value, (int, float)) and key not in {"epoch", "step"}
            ]
            latest = {
                "epoch": int(self.rows[-1]["epoch"]),
                "samples": len(self.rows),
                "mean": {
                    key: sum(float(row[key]) for row in self.rows) / len(self.rows)
                    for key in numeric_keys
                },
            }
            cumulative = {
                "samples": self._cumulative_count,
                "mean": {
                    key: value / max(self._cumulative_count, 1)
                    for key, value in self._cumulative_sums.items()
                },
            }
            with open(self.save_dir / "data_path_diagnostics_summary.json", "w") as handle:
                json.dump({"latest_epoch": latest, "cumulative": cumulative}, handle, indent=2)
            return
        keys = list(self.rows[0].keys())
        with open(self.save_dir / "data_path_diagnostics.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.rows)
        with open(self.save_dir / "data_path_diagnostics.json", "w") as handle:
            json.dump(self.rows, handle, indent=2)

    def print_epoch_summary(self, epoch: int) -> None:
        rows = [row for row in self.rows if int(row["epoch"]) == int(epoch)]
        if not rows:
            return

        def mean(key: str) -> float:
            return sum(float(row.get(key, 0.0)) for row in rows) / len(rows)

        first = rows[0]
        print(
            "[data-path] "
            f"mode={self.config.data_path_mode} epoch={epoch} samples={len(rows)} "
            f"Nfull={int(first['N_full']):,} Nq={int(first['N_query']):,} "
            f"pre-model={mean('pre_model_total_ms'):.2f}ms "
            f"forward={mean('model_forward_ms'):.2f}ms "
            f"backward={mean('backward_ms'):.2f}ms "
            f"peak={mean('gpu_peak_allocated_mb'):.1f}MB"
        )
