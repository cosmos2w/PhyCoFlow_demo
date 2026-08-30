#!/usr/bin/env python
"""Paired observation-consistency audit on reproducible canonical snapshots.

This script performs inference only; it never modifies formal caches.  Each
mode receives the same checkpoint, physical snapshots, nested sensor rows, and
generation seed so the mean-L2 comparison is paired and auditable.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from common.config import RESULTS_DIR, load_config, method_items, recipe_items, stable_seed
from common.dataset_loader import find_snapshot
from common.io_utils import read_csv, write_csv, write_json
from common.model_loader import load_model
from common.statistics import relative_l2


def _canonical_sensor_rows(path: Path, sensor_count: int) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped.setdefault(int(row["snapshot_index"]), []).append(row)
    for snapshot, rows in grouped.items():
        rows.sort(key=lambda row: int(row["sensor_order"]))
        if len(rows) < sensor_count:
            raise ValueError(
                f"Snapshot {snapshot} has {len(rows)} sensor rows; {sensor_count} required"
            )
        grouped[snapshot] = rows[:sensor_count]
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    parser.add_argument("--sensor-plan", type=Path, required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--model", default="DMFGen")
    parser.add_argument("--recipes", nargs="+")
    parser.add_argument("--legacy-mode")
    parser.add_argument("--candidate-mode", default="default_hard")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--selection-seed", type=int, default=20260713)
    parser.add_argument("--sensor-count", type=int, default=256)
    parser.add_argument("--checkpoint", choices=["last", "best"], default="last")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = load_config(args.config)
    manifest_rows = read_csv(args.cache_manifest)
    model_spec = next(iter(method_items(cfg, [args.model])), None)
    if model_spec is None:
        raise ValueError(f"Unknown model {args.model!r}")
    ok_rows = [
        row for row in manifest_rows
        if row.get("model") == model_spec["key"] and row.get("status") == "ok"
    ]
    observed_modes = Counter(row.get("consistency_mode", "") for row in ok_rows)
    legacy_mode = args.legacy_mode
    if legacy_mode is None:
        nonempty = [mode for mode, count in observed_modes.items() if mode and count]
        if len(nonempty) != 1:
            raise ValueError(f"Cannot infer one legacy mode from manifest: {observed_modes}")
        legacy_mode = nonempty[0]
    if legacy_mode == args.candidate_mode:
        raise ValueError(f"Legacy and candidate modes are both {legacy_mode!r}; no comparison needed")

    available_recipes = {row.get("recipe") for row in ok_rows}
    selected_recipes = [
        (key, spec) for key, spec in recipe_items(cfg, args.recipes or ["all"])
        if key in available_recipes
    ]
    if not selected_recipes:
        raise ValueError(f"No cached recipes are available for {model_spec['key']}")

    sensor_rows = _canonical_sensor_rows(args.sensor_plan, args.sensor_count)
    snapshots = np.asarray(sorted(sensor_rows), dtype=int)
    if args.samples <= 0 or args.samples > len(snapshots):
        raise ValueError(f"--samples must be in [1, {len(snapshots)}]")
    rng = np.random.default_rng(args.selection_seed)
    selected_snapshots = sorted(int(v) for v in rng.choice(snapshots, args.samples, replace=False))

    results: list[dict] = []
    for recipe_key, recipe_spec in selected_recipes:
        loaded = load_model(
            model_spec, recipe_key, recipe_spec, checkpoint=args.checkpoint,
            allow_fallback=False, split=cfg["dataset"]["split"],
            eval_resolution=cfg["dataset"]["eval_resolution"], device=args.device,
        )
        try:
            mean = loaded.dataset.mean.cpu().numpy()
            std = loaded.dataset.std.cpu().numpy()
            for snapshot in selected_snapshots:
                rows = sensor_rows[snapshot]
                first = rows[0]
                dataset_index = find_snapshot(
                    loaded.dataset, int(first["case_id"]), int(first["time_index"])
                )
                generation_seed = stable_seed(
                    cfg["cache"]["generation_seed"], "generation",
                    first["case_id"], first["time_index"],
                )
                for mode in (legacy_mode, args.candidate_mode):
                    output = loaded.reconstruct(
                        dataset_index, rows, n_steps=int(cfg["cache"]["n_steps"]),
                        ode_solver=cfg["cache"]["ode_solver"],
                        consistency_mode=mode, generation_seed=generation_seed,
                    )
                    truth = output["truth_norm"] * std + mean
                    recon = output["recon_norm"] * std + mean
                    results.append({
                        "model": model_spec["key"], "recipe": recipe_key,
                        "snapshot_index": snapshot, "case_id": int(first["case_id"]),
                        "time_index": int(first["time_index"]),
                        "sensor_count": args.sensor_count, "mode": mode,
                        "relative_l2": relative_l2(truth, recon),
                        "generation_seed": generation_seed,
                        "checkpoint_path": str(loaded.checkpoint_path),
                        "obs_consistency_applied": output["obs_consistency_applied"],
                    })
        finally:
            loaded.close()

    mode_means = {
        mode: float(np.mean([row["relative_l2"] for row in results if row["mode"] == mode]))
        for mode in (legacy_mode, args.candidate_mode)
    }
    recipe_means = {
        recipe_key: {
            mode: float(np.mean([
                row["relative_l2"] for row in results
                if row["recipe"] == recipe_key and row["mode"] == mode
            ]))
            for mode in (legacy_mode, args.candidate_mode)
        }
        for recipe_key, _ in selected_recipes
    }
    candidate_better = mode_means[args.candidate_mode] < mode_means[legacy_mode]
    run_id = args.run_id or datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_dir = RESULTS_DIR / "CacheAudit"
    csv_path = output_dir / f"ObsConsistencyAudit_rows_{run_id}.csv"
    json_path = output_dir / f"ObsConsistencyAudit_summary_{run_id}.json"
    write_csv(csv_path, results)
    write_json(json_path, {
        "run_id": run_id, "model": model_spec["key"],
        "legacy_mode": legacy_mode, "candidate_mode": args.candidate_mode,
        "selection_seed": args.selection_seed, "selected_snapshots": selected_snapshots,
        "sensor_count": args.sensor_count,
        "recipes": [key for key, _ in selected_recipes],
        "paired_comparisons": len(results) // 2,
        "mode_mean_relative_l2": mode_means,
        "recipe_mean_relative_l2": recipe_means,
        "candidate_better": candidate_better,
        "decision": "regenerate_DMFGen" if candidate_better else "retain_existing_DMFGen",
        "source_manifest": str(args.cache_manifest.resolve()),
        "sensor_plan": str(args.sensor_plan.resolve()),
        "rows_csv": str(csv_path.resolve()),
    })
    print(f"[AUDIT] selected snapshots: {selected_snapshots}")
    print(f"[AUDIT] mean {legacy_mode}: {mode_means[legacy_mode]:.8f}")
    print(f"[AUDIT] mean {args.candidate_mode}: {mode_means[args.candidate_mode]:.8f}")
    print(f"[DECISION] {'regenerate DMFGen' if candidate_better else 'retain existing DMFGen caches'}")
    print(f"[OK] {csv_path}")
    print(f"[OK] {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
