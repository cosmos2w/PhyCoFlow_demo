"""Canonical PDEBench multi-resolution manifests, datasets, and test identity."""
from __future__ import annotations
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml
import numpy as np

from .config import DEMO_DIR, RESULTS_DIR, SRC_DIR
from .recipe_registry import flatten_run_config

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from helpers import PDEBenchMultiResDataset  # noqa: E402
from organize_train_MultiRes import build_manifest, default_manifest_path  # noqa: E402


def read_run_config(run_dir: Path) -> tuple[dict, dict]:
    config_path = resolve_run_config_path(run_dir)
    if config_path.name == "args.json":
        raw = json.loads(config_path.read_text(encoding="utf-8")) or {}
    else:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return raw, flatten_run_config(raw)


def resolve_run_config_path(run_dir: Path) -> Path:
    """Return the immutable run-local config, including legacy ``args.json``."""
    for name in ("run_config.yaml", "args.json"):
        path = run_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"No run_config.yaml or args.json in {run_dir}")


def _absolute_from_task(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (DEMO_DIR / path).resolve()


def locate_or_rebuild_manifest(run_dir: Path, recovered_dir: Path) -> tuple[Path, dict]:
    """Locate the exact configured manifest or deterministically rebuild it."""
    _, cfg = read_run_config(run_dir)
    explicit = cfg.get("multires_manifest_path")
    source = None
    if explicit:
        candidates = [_absolute_from_task(explicit), run_dir / explicit]
        source = next((p.resolve() for p in candidates if p.exists()), None)
    else:
        processed = _absolute_from_task(cfg["pdebench_processed_root"])
        source = default_manifest_path(
            processed, str(cfg["pdebench_dataset_name"]), int(cfg["selected_field_idx_raw"]),
            str(cfg["multires_ratio"]), float(cfg.get("Case_Truncate_Ratio", 0.0)),
            float(cfg.get("multires_train_case_fraction", 1.0)),
        )
    recovered_dir.mkdir(parents=True, exist_ok=True)
    target = recovered_dir / "manifest_resolved.json"
    if source is not None and source.exists():
        manifest = json.loads(source.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(
            processed_root=_absolute_from_task(cfg["pdebench_processed_root"]),
            dataset_name=str(cfg["pdebench_dataset_name"]),
            selected_field_idx=int(cfg["selected_field_idx_raw"]),
            multires_ratio=str(cfg["multires_ratio"]),
            train_fraction=float(cfg.get("train_ratio", .9)),
            case_truncate_ratio=float(cfg.get("Case_Truncate_Ratio", 0.0)),
            multires_train_case_fraction=float(cfg.get("multires_train_case_fraction", 1.0)),
        )
    manifest["paths"] = {tag: str(_absolute_from_task(path)) for tag, path in manifest["paths"].items()}
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return target, manifest


def build_run_dataset(run_dir: Path, model_key: str, recipe_key: str, *, split="test", eval_resolution="H"):
    recovered = RESULTS_DIR / "DatasetStats" / model_key / recipe_key
    manifest_path, manifest = locate_or_rebuild_manifest(run_dir, recovered)
    copied_stats = run_dir / "dataset_stats.pt"
    recovered_stats = recovered / "dataset_stats.pt"
    # The recovered copy may belong to an older checkpoint package.  It is
    # tiny, so refreshing it on every load is safer than relying on mtime.
    if copied_stats.exists():
        shutil.copy2(copied_stats, recovered_stats)
    dataset = PDEBenchMultiResDataset(
        manifest_path=str(manifest_path), split=split, eval_resolution=eval_resolution,
        force_resolution=eval_resolution, stats_path=str(recovered_stats),
    )
    return dataset, manifest_path, manifest, recovered_stats


def canonical_rows(dataset, maximum: int | None = 300, *, strategy: str = "sequential", seed: int = 42,
                   allowed_time_indices: set[int] | None = None) -> list[dict[str, Any]]:
    """Build canonical sample identities, optionally using one time per held-out case.

    ``stratified_unique_cases`` avoids filling the screen with adjacent times from
    one trajectory.  It selects distinct held-out cases and one deterministic
    usable time from each case, making the default n=300 a true 300-case screen.
    """
    if strategy == "stratified_unique_cases":
        by_case: dict[int, list[tuple[int, str, int]]] = {}
        for dataset_index, (res, case_id, time_index) in enumerate(dataset.entries):
            if allowed_time_indices is not None and int(time_index) not in allowed_time_indices:
                continue
            by_case.setdefault(int(case_id), []).append((dataset_index, res, int(time_index)))
        case_ids = np.array(sorted(by_case), dtype=int)
        rng = np.random.default_rng(int(seed))
        requested = len(case_ids) if maximum is None else min(len(case_ids), int(maximum))
        selected_cases = np.sort(rng.choice(case_ids, size=requested, replace=False))
        selected = []
        for case_id in selected_cases:
            entries = by_case[int(case_id)]
            pick = int(rng.integers(0, len(entries)))
            dataset_index, res, time_index = entries[pick]
            selected.append((dataset_index, res, int(case_id), time_index))
    elif strategy == "sequential":
        n = len(dataset) if maximum is None else min(len(dataset), int(maximum))
        selected = [(i, *dataset.entries[i]) for i in range(n)]
    else:
        raise ValueError(f"Unknown canonical selection strategy: {strategy}")
    rows = []
    for snapshot_index, (dataset_index, res, case_id, time_index) in enumerate(selected):
        rows.append({
            "snapshot_index": snapshot_index, "dataset_index": int(dataset_index), "case_id": int(case_id),
            "time_index": int(time_index), "physical_time": float(dataset.times[time_index]),
            "eval_resolution": dataset.output_resolution or res, "selection_strategy": strategy,
            "selection_seed": int(seed),
        })
    return rows


def find_snapshot(dataset, case_id: int, time_index: int) -> int:
    if not hasattr(dataset,"_postprocess_identity_to_index"):
        dataset._postprocess_identity_to_index={(int(case),int(time)):i for i,(_,case,time) in enumerate(dataset.entries)}
    key=(int(case_id),int(time_index))
    if key in dataset._postprocess_identity_to_index:
        return int(dataset._postprocess_identity_to_index[key])
    raise KeyError(f"No canonical sample case_id={case_id}, time_index={time_index}")


def physical_native_field(manifest: dict, case_id: int, time_index: int, resolution: str):
    import h5py
    import numpy as np
    tag = resolution.upper()
    with h5py.File(manifest["paths"][tag], "r") as h5:
        field = h5["fields"][case_id, time_index, :, 0, 0, int(manifest["selected_field_idx"])].astype(np.float32)
        coords = h5["coordinates"][:, 0, 0, :].astype(np.float32)
    return coords, field
