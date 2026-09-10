#!/usr/bin/env python
"""Resumable, paired Cond_T ablation reconstruction using the paper sensor plan.

Run with ``conda run -n phycoflow_env python src/evaluate_ablation_condT.py``.
Only inference is performed. Cached arrays follow the archived common.cache schema.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import yaml

DEMO = Path(__file__).resolve().parents[1]
ARCHIVE = DEMO / "Save_TrainedModel/_TrainedModels"
ABLATIONS = DEMO / "Save_TrainedModel/ablation_condT"
PLAN = ARCHIVE / "_Process_Results/SensorPlans/SensorPlan_paper_full_20260711.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(base: int, *parts: object) -> int:
    return int(hashlib.sha256("|".join(map(str, (base, *parts))).encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp.json")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def discover_runs(a0_run_dir: Path | None = None) -> dict[str, Path]:
    """Resolve the six training runs, allowing A0 to be a frozen snapshot.

    A0 is still the archived Cond_T baseline by default.  A dated evaluation
    can pass a copied run directory so a trainer writing ``last.pt`` cannot
    change the weights halfway through inference.
    """
    runs = {"A0": (a0_run_dir or ARCHIVE / "DMF_Gen/Cond_T").resolve()}
    if not (runs["A0"] / "run_config.yaml").is_file():
        raise FileNotFoundError(f"A0 run_config.yaml not found: {runs['A0']}")
    for i in range(1, 6):
        matches = sorted(path for path in ABLATIONS.glob(f"A{i}_*") if (path / "run_config.yaml").is_file())
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one A{i} run; found {matches}")
        runs[f"A{i}"] = matches[0].resolve()
    return runs


def read_plan(path: Path) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "Cond_T":
                if row["split"] != "test":
                    raise ValueError("Expected the archived test sensor plan")
                groups.setdefault(int(row["snapshot"]), []).append(row)
    for snapshot, rows in groups.items():
        rows.sort(key=lambda row: int(row["sensor_order"]))
        if len(rows) != 256 or any(int(row["field_index"]) != 2 for row in rows):
            raise ValueError(f"Snapshot {snapshot}: expected 256 temperature sensors")
        if [int(row["sensor_order"]) for row in rows] != list(range(256)):
            raise ValueError(f"Snapshot {snapshot}: invalid sensor ordering")
    return groups


def checkpoint_summary(path: Path, obj: dict | None = None) -> dict:
    import torch
    if obj is None:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    return {"path": str(path), "sha256": sha256(path), "size_bytes": path.stat().st_size,
            **{key: obj.get(key) for key in ("epoch", "global_step", "train_loss", "val_loss",
               "best_selection_weights", "model_ema_enabled", "model_ema_eval")}}


def sample_once(model, kwargs: dict, seed: int, mode: str, chunk_size: int):
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    with torch.inference_mode():
        return model.sample(**kwargs, reconstruction_execution_mode=mode,
                            reconstruction_query_chunk_size=chunk_size,
                            reconstruction_cache_level="static_features")


def equivalence_check(model, kwargs: dict, seed: int, mode: str, chunk_size: int) -> dict:
    """Compare both complete seeded reconstructions (including observed clamps)."""
    import torch
    ref = sample_once(model, kwargs, seed, "legacy_full", chunk_size)
    out = sample_once(model, kwargs, seed, mode, chunk_size)
    diff = (ref - out).abs()
    result = {"max_abs_normalized": float(diff.max()), "rms_normalized": float(diff.square().mean().sqrt()),
              "relative_l2": float(torch.linalg.vector_norm(ref - out) / torch.linalg.vector_norm(ref).clamp_min(1e-12)),
              "atol": 2e-4, "rtol": 2e-4, "passed": bool(torch.allclose(ref, out, atol=2e-4, rtol=2e-4))}
    if not result["passed"]:
        raise RuntimeError(f"Cached/legacy equivalence failed: {result}")
    return result


def reuse_code_hashes(reuse_root: Path, models: set[str]) -> dict[str, str]:
    """Read the evaluator identity recorded with explicitly reusable caches.

    The cache protocol and model source files are checked separately below.
    Allowing a cache from an earlier evaluator revision is therefore opt-in,
    and only the evaluator-script hash may differ.
    """
    hashes = {}
    for name in sorted(models):
        provenance_path = reuse_root / f"provenance_{name}.json"
        if not provenance_path.is_file():
            raise FileNotFoundError(f"Reusable cache provenance not found: {provenance_path}")
        provenance = json.loads(provenance_path.read_text())
        value = provenance.get("code_sha256")
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Reusable cache provenance has no valid code_sha256: {provenance_path}")
        hashes[name] = value
    return hashes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, default=ABLATIONS / "evaluation_20260909")
    parser.add_argument("--sensor-plan", type=Path, default=PLAN)
    parser.add_argument("--models", nargs="+", default=["A0", "A1", "A2", "A3", "A4", "A5"])
    parser.add_argument("--checkpoint", choices=["last", "best"], default="last")
    parser.add_argument("--a0-run-dir", type=Path, help="Frozen A0 run directory containing run_config.yaml and checkpoints")
    parser.add_argument("--device", default="cuda:0", help="Inference device; use CUDA_VISIBLE_DEVICES to mask physical GPU 0")
    parser.add_argument("--physical-gpu", type=int, default=0, help="Physical GPU authorized for this evaluation")
    parser.add_argument("--n-steps", type=int, default=2)
    parser.add_argument("--ode-solver", choices=["euler", "heun"], default="euler")
    parser.add_argument("--max-snapshots", type=int)
    parser.add_argument("--snapshots", nargs="+", type=int)
    parser.add_argument("--query-chunk-size", type=int, default=8192)
    parser.add_argument("--execution-mode", choices=["cached_streamed", "legacy_full"], default="cached_streamed")
    parser.add_argument("--verify-equivalence", action="store_true", help="Compare both execution modes on each model's first requested full state")
    parser.add_argument("--uncompressed", action="store_true", help="Faster writes at the cost of disk space")
    parser.add_argument("--reuse-cache-dir", type=Path,
                        help="Optional earlier policy directory whose compatible caches may be symlinked into this evaluation")
    parser.add_argument("--reuse-models", nargs="+", default=[], metavar="A1",
                        help="Models eligible for explicit cache reuse from --reuse-cache-dir")
    args = parser.parse_args()

    # Delay ML imports until after argument parsing, keeping --help/cache utilities lightweight.
    os.environ.setdefault("KEOPS_CACHE_FOLDER", "/tmp/phycoflow_ablation_keops")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/phycoflow_matplotlib")
    Path(os.environ["KEOPS_CACHE_FOLDER"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import torch
    from helpers import TurbulentCombustionH5Dataset
    from phycoflow_pointcloud.models.factory import build_pointcloud_model

    torch.set_num_threads(4)
    device = torch.device(args.device)
    if device.type == "cuda":
        # Reject accidental selection of a physical GPU outside the user allowance.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        logical_index = device.index or 0
        if visible:
            visible_devices = [item.strip() for item in visible.split(",") if item.strip()]
            if logical_index >= len(visible_devices):
                raise ValueError(f"Logical CUDA device {logical_index} is absent from CUDA_VISIBLE_DEVICES={visible!r}")
            physical = visible_devices[logical_index]
        else:
            physical = str(logical_index)
        if physical != str(args.physical_gpu):
            raise ValueError(
                f"This evaluation is authorized for physical GPU {args.physical_gpu} only, got {physical}"
            )
        torch.cuda.set_device(device)
    a0_run_dir = args.a0_run_dir
    if a0_run_dir is not None and not a0_run_dir.is_absolute():
        a0_run_dir = DEMO / a0_run_dir
    runs = discover_runs(a0_run_dir)
    models = list(runs) if args.models == ["all"] else args.models
    if any(model not in runs for model in models):
        raise ValueError(f"Unknown models {models}; use A0--A5")
    groups = read_plan(args.sensor_plan)
    snapshots = sorted(groups) if args.snapshots is None else sorted(set(args.snapshots))
    if not set(snapshots).issubset(groups):
        raise ValueError("Requested snapshots absent from the archived plan")
    if args.max_snapshots is not None:
        snapshots = snapshots[:args.max_snapshots]
    root = args.evaluation_dir.resolve() / args.checkpoint
    root.mkdir(parents=True, exist_ok=True)
    plan_hash = sha256(args.sensor_plan)
    code_hash = sha256(Path(__file__))
    reuse_models = set(args.reuse_models)
    if args.reuse_cache_dir is None and reuse_models:
        raise ValueError("--reuse-models requires --reuse-cache-dir")
    if args.reuse_cache_dir is not None:
        reuse_root = args.reuse_cache_dir.resolve()
        if not reuse_root.is_dir():
            raise FileNotFoundError(f"Reusable cache directory not found: {reuse_root}")
    else:
        reuse_root = None
    if any(name not in {"A0", "A1", "A2", "A3", "A4", "A5"} for name in reuse_models):
        raise ValueError(f"Unknown --reuse-models {sorted(reuse_models)}; use A0--A5")
    old_code_hashes = reuse_code_hashes(reuse_root, reuse_models) if reuse_root else {}

    for name in models:
        run = runs[name]
        cfg_path = run / "run_config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text())
        if (int(cfg.get("seed", 42)), float(cfg.get("train_ratio", .9)), int(cfg.get("time_stride", 1))) != (42, .9, 1):
            raise ValueError(f"{name}: split differs from the archived test population")
        if name != "A0" and cfg.get("ablation", {}).get("id") != name:
            raise ValueError(f"{name}: configuration ablation ID mismatch")
        data = Path(cfg["data"])
        if not data.is_absolute():
            data = DEMO / data
        stats = run / "dataset_stats.pt"
        if not stats.is_file() and cfg.get("dataset_stats_path"):
            stats = Path(cfg["dataset_stats_path"])
            if not stats.is_absolute():
                stats = DEMO / stats
        if not stats.is_file():
            raise FileNotFoundError(f"Stored training statistics required: {stats}")
        dataset = TurbulentCombustionH5Dataset(str(data), split="test", train_ratio=.9, seed=42,
                    time_stride=1, field_names=cfg.get("FIELD_NAMES", cfg.get("field_names")), stats_path=str(stats))
        canonical_fields = tuple(str(field).replace("_", "") for field in dataset.field_names)
        if len(dataset) != 1000 or canonical_fields != ("CH4", "CO", "T", "U1", "p"):
            raise ValueError(f"Unexpected dataset: {len(dataset)} states, {dataset.field_names}")
        cp_path = run / f"{args.checkpoint}.pt"
        cp = torch.load(cp_path, map_location="cpu", weights_only=False)
        checkpoints = {key: checkpoint_summary(run / f"{key}.pt", cp if key == args.checkpoint else None)
                       for key in ("best", "last")}
        for key in ("mean", "std"):
            if key in cp and not torch.equal(cp[key].cpu().float(), getattr(dataset, key).cpu().float()):
                raise ValueError(f"{name}: checkpoint and dataset {key} mismatch")
        model = build_pointcloud_model(cfg, n_fields=dataset.num_fields, device="cpu")
        model.load_state_dict(cp["model"], strict=True)
        model = model.to(device).eval()
        del cp
        mode = "legacy_full" if name == "A1" else args.execution_mode
        common = {"method": name, "variant": name, "condition": "Cond_T", "split": "test",
                  "checkpoint_name": cp_path.name, "checkpoint_path": str(cp_path),
                  "checkpoint_sha256": checkpoints[args.checkpoint]["sha256"],
                  "checkpoint_epoch": checkpoints[args.checkpoint]["epoch"], "config_sha256": sha256(cfg_path),
                  "stats_sha256": sha256(stats), "sensor_plan_hash": plan_hash,
                  "sensor_plan": str(args.sensor_plan.resolve()), "n_steps": args.n_steps, "ode_solver": args.ode_solver,
                  "obs_consistency": "default_hard", "obs_consistency_applied": "default_hard",
                  "reconstruction_execution_mode": mode, "query_chunk_size": args.query_chunk_size,
                  "generation_seed_policy": "stable_seed(20260711,'generation','DMF-Gen','Cond_T',snapshot); shared across variants",
                  "num_x": cfg.get("Num_x", 403), "num_y": cfg.get("Num_y", 100), "code_sha256": code_hash}
        provenance = {**common, "checkpoints": checkpoints, "run_directory": str(run), "config_path": str(cfg_path),
                      "dataset_path": str(data.resolve()), "dataset_stats_path": str(stats.resolve()),
                      "dataset_size_bytes": data.stat().st_size,
                      "dataset_mtime_ns": data.stat().st_mtime_ns, "dataset_stats_mean": dataset.mean.tolist(),
                      "dataset_stats_std": dataset.std.tolist(), "field_names": list(dataset.field_names),
                      "test_time_indices": dataset.indices.tolist(), "requested_snapshots": snapshots,
                      "device": str(device), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "torch_version": torch.__version__, "strict_load": True,
                      "weights_evaluated": "live model weights, no EMA substitution",
                      "parameter_count": sum(p.numel() for p in model.parameters())}
        snapshot_manifest_path = run / "snapshot_manifest.json"
        if snapshot_manifest_path.is_file():
            provenance["snapshot_manifest_path"] = str(snapshot_manifest_path.resolve())
            provenance["snapshot_manifest"] = json.loads(snapshot_manifest_path.read_text())
        if name in old_code_hashes:
            provenance["cache_reuse"] = {
                "source_directory": str(reuse_root),
                "source_code_sha256": old_code_hashes[name],
                "compatibility_rule": "all cache metadata fields must match except evaluator code_sha256",
            }
        provenance_path = root / f"provenance_{name}.json"
        write_json(provenance_path, provenance)
        model_dir = root / name
        model_dir.mkdir(exist_ok=True)
        rows_out = []
        reused_snapshots = []
        generated_snapshots = []
        start = time.perf_counter()
        for offset, snapshot in enumerate(snapshots):
            path = model_dir / f"RecCache_s{snapshot:04d}.npz"
            reuse_source = None
            if not path.exists() and name in old_code_hashes:
                candidate = reuse_root / name / path.name
                if candidate.is_file():
                    path.symlink_to(candidate.resolve())
                    reuse_source = candidate.resolve()
            seed = stable_seed(20260711, "generation", "DMF-Gen", "Cond_T", snapshot)
            meta = {**common, "snapshot": snapshot, "time_index": int(dataset.indices[snapshot]),
                    "generation_seed": seed, "sensor_seed": int(groups[snapshot][0]["sensor_seed"]), "status": "ok"}
            reused = False
            if path.is_file():
                with np.load(path, allow_pickle=False) as saved:
                    old = json.loads(str(saved["metadata_json"].item()))
                    mismatches = [key for key, value in meta.items() if old.get(key) != value]
                    compatible_old_code = (
                        name in old_code_hashes
                        and mismatches == ["code_sha256"]
                        and old.get("code_sha256") == old_code_hashes[name]
                    )
                    if not mismatches or compatible_old_code:
                        # Loading the reconstruction also checks for a truncated/corrupt zip member.
                        if not np.isfinite(saved["recon_norm"]).all():
                            raise ValueError(f"Nonfinite cached reconstruction: {path}")
                        meta = old
                        reused = True
                        reused_snapshots.append(snapshot)
                    else:
                        raise ValueError(
                            f"Incompatible existing cache {path}; mismatched metadata={mismatches}. "
                            "Choose a new evaluation directory or an explicitly compatible reuse source."
                        )
            if not reused or (offset == 0 and args.verify_equivalence):
                sample = dataset[snapshot]
                coords = sample["coords"].unsqueeze(0).to(device)
                truth = sample["fields"].unsqueeze(0).to(device)
                sensors = groups[snapshot]
                idx = torch.tensor([[int(row["point_index"]) for row in sensors]], device=device, dtype=torch.long)
                fld = torch.tensor([[int(row["field_index"]) for row in sensors]], device=device, dtype=torch.long)
                values = truth[0, idx[0], fld[0]].view(1, -1, 1)
                recorded = np.asarray([float(row["normalized_value"]) for row in sensors])
                if not np.allclose(values.detach().cpu().numpy().ravel(), recorded, atol=1e-6, rtol=1e-6):
                    raise ValueError(f"{name}/{snapshot}: sensor values differ from archived plan")
                kwargs = dict(coords=coords, obs_coords=coords[:, idx[0]], obs_values=values,
                              obs_mask=torch.ones((1, idx.shape[1]), device=device), obs_field_ids=fld,
                              clamp_indices=idx, n_steps=args.n_steps, ode_solver=args.ode_solver,
                              obs_consistency_mode="default_hard")
                if offset == 0 and args.verify_equivalence:
                    provenance["execution_equivalence"] = equivalence_check(model, kwargs, seed, mode, args.query_chunk_size)
                    write_json(provenance_path, provenance)
                    print(f"[EQUIVALENCE] {name}: {provenance['execution_equivalence']}", flush=True)
                if not reused:
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    tick = time.perf_counter()
                    recon = sample_once(model, kwargs, seed, mode, args.query_chunk_size)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    meta["inference_seconds"] = time.perf_counter() - tick
                    gt = truth[0].cpu().numpy()
                    pred = recon[0].cpu().numpy()
                    if not np.isfinite(pred).all():
                        raise ValueError(f"{name}/{snapshot}: nonfinite reconstruction")
                    np.testing.assert_array_equal(pred[idx[0].cpu().numpy(), fld[0].cpu().numpy()], recorded.astype(np.float32))
                    mean, std = dataset.mean.numpy(), dataset.std.numpy()
                    arrays = dict(truth_norm=gt, recon_norm=pred, truth_phys=gt * std + mean, recon_phys=pred * std + mean,
                                  coords_norm=sample["coords"].numpy(), coords_phys=dataset.coords_raw.numpy(),
                                  obs_indices=idx[0].cpu().numpy(), obs_field_ids=fld[0].cpu().numpy(),
                                  obs_values_norm=values[0, :, 0].cpu().numpy())
                    tmp = path.with_suffix(".tmp.npz")
                    saver = np.savez if args.uncompressed else np.savez_compressed
                    saver(tmp, **arrays, metadata_json=np.array(json.dumps(meta, sort_keys=True)))
                    os.replace(tmp, path)
                    generated_snapshots.append(snapshot)
                    del recon, arrays, gt, pred
                del coords, truth, idx, fld, values, kwargs, sample
            rows_out.append({**meta, "cache_path": str(path),
                             "cache_reused_from": "" if reuse_source is None else str(reuse_source)})
            if (offset + 1) % 25 == 0 or offset + 1 == len(snapshots):
                manifest = root / f"manifest_{name}.csv"
                temp = manifest.with_suffix(".tmp.csv")
                with temp.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(row.keys() for row in rows_out))))
                    writer.writeheader()
                    writer.writerows(rows_out)
                os.replace(temp, manifest)
                print(f"[PROGRESS] {name} {offset + 1}/{len(snapshots)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        provenance["completed_snapshots"] = len(rows_out)
        if name in old_code_hashes:
            provenance["cache_reuse"].update({
                "reused_snapshots": len(reused_snapshots),
                "generated_snapshots": len(generated_snapshots),
            })
        provenance["elapsed_seconds"] = time.perf_counter() - start
        write_json(provenance_path, provenance)
        dataset.close()
        del model, dataset
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
