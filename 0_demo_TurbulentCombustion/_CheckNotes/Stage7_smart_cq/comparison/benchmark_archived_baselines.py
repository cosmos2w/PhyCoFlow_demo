#!/usr/bin/env python3
"""Benchmark the frozen Senseiver and latent-FM reference checkpoints.

This is deliberately a reference-only comparison.  The archived baselines use
the historical CH4/CO/T/U1/p dataset, whereas Stage 7 uses CO/T/U0/U1/p.
The benchmark keeps the common Cond-T protocol (256 sensors, B128, Q4096) and
does not modify either baseline implementation or checkpoint.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-project-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_call(fn, device: torch.device) -> tuple[float, float]:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    sync(device)
    start = time.perf_counter()
    out = fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    peak_mib = (
        torch.cuda.max_memory_allocated(device) / (1024.0**2)
        if device.type == "cuda"
        else float("nan")
    )
    del out
    return elapsed_ms, peak_mib


def load_sensor_rows(path: Path, condition: str = "Cond_T", snapshot: int = 0) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["condition"] == condition and int(row["snapshot"]) == snapshot
        ]
    if len(rows) != 256:
        raise RuntimeError(f"expected 256 Cond-T sensors, found {len(rows)}")
    return rows


def archived_quality(path: Path, method: str) -> dict:
    keep = {"CH4", "CO", "T", "U1", "p", "Unobserved_mean"}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["method"] == method
            and row["condition"] == "Cond_T"
            and row["field"] in keep
            and row["status"] == "ok"
        ]
    values = {row["field"]: float(row["mean"]) for row in rows}
    values["common_unobserved_CO_U1_p_mean"] = statistics.fmean(
        values[name] for name in ("CO", "U1", "p")
    )
    values["snapshots"] = int(rows[0]["valid_n"])
    return values


def count_parameters(bundle) -> dict:
    params = list(bundle.model.parameters())
    return {
        "executable_total": sum(p.numel() for p in params),
        "trainable": sum(p.numel() for p in params if p.requires_grad),
    }


def main() -> int:
    args = parse_args()
    root = args.archive_project_root.resolve()
    scripts = root / "Save_TrainedModel" / "_TrainedModels" / "_Scripts"
    sys.path.insert(0, str(scripts))

    from common.model_loader import load_model  # noqa: PLC0415
    import model_baseline as baseline_lib  # noqa: PLC0415

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("this benchmark requires CUDA")

    archive = root / "Save_TrainedModel" / "_TrainedModels"
    results_root = archive / "_Process_Results"
    sensor_rows = load_sensor_rows(
        results_root / "SensorPlans" / "SensorPlan_paper_full_20260711.csv"
    )
    field_l2 = results_root / "FieldL2" / "FieldL2_summary_paper_full_20260711.csv"

    methods = [
        {"name": "Senseiver", "directory": "Senseiver", "nfe": [1]},
        {"name": "Latent FM", "directory": "Latent_FM", "nfe": [2, 4]},
    ]
    output = {
        "protocol": {
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "batch_size": args.batch_size,
            "query_points": 4096,
            "condition": "Cond_T",
            "sensors": 256,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "checkpoint": "last.pt",
            "inference_grid_points": 40300,
            "training_timing_scope": "native compute step with one pre-collated batch; includes H2D, condition construction, forward, backward, clipping, optimizer",
            "inference_timing_scope": "full-grid reconstruction after model load; includes sparse-condition construction and decode, excludes checkpoint/dataset load and CPU result copy",
            "dataset_caveat": "archived CH4/CO/T/U1/p reference; Stage 7 uses CO/T/U0/U1/p",
        },
        "models": {},
    }

    condition_cfg = {"cond_fields": [2], "n_obs": [256]}
    for method in methods:
        torch.cuda.empty_cache()
        loaded = load_model(
            method,
            "Cond_T",
            checkpoint="last",
            split="test",
            device=str(device),
            n_steps=max(method["nfe"]),
            ode_solver="euler",
        )
        bundle = loaded.model
        # Hold the sparse-condition cardinality fixed for the training timing.
        bundle.config["shared"]["conditioning"]["n_obs_min_list"] = [256]
        bundle.config["shared"]["conditioning"]["n_obs_max_list"] = [256]

        row = {
            "checkpoint": str(loaded.checkpoint_path),
            "parameters": count_parameters(bundle),
            "checkpoint_epoch": int(
                torch.load(loaded.checkpoint_path, map_location="cpu", weights_only=False)["epoch"]
            ),
            "quality_reference": archived_quality(field_l2, method["name"]),
            "inference": {},
        }

        for nfe in method["nfe"]:
            def infer():
                return loaded.reconstruct(
                    0,
                    condition_cfg,
                    sensor_rows,
                    n_steps=nfe,
                    ode_solver="euler",
                    obs_consistency="endpoint_smooth",
                    generation_seed=20260711,
                )

            for _ in range(args.warmup):
                infer()
            samples = [timed_call(infer, device) for _ in range(args.repeats)]
            row["inference"][f"nfe{nfe}"] = {
                "median_ms": statistics.median(v[0] for v in samples),
                "samples_ms": [v[0] for v in samples],
                "peak_allocated_mib": max(v[1] for v in samples),
            }

        stats_path = loaded.checkpoint_path.parent / "dataset_stats.pt"
        train_set = baseline_lib.build_dataset(bundle.config, split="train", stats_path=stats_path)
        sample = train_set[0]
        batch = baseline_lib.collate_snapshots([sample] * args.batch_size)

        def train_step():
            return bundle.adapter.run_epoch(bundle, [batch], training=True, epoch=0)

        for _ in range(args.warmup):
            train_step()
        train_samples = [timed_call(train_step, device) for _ in range(args.repeats)]
        step_ms = statistics.median(v[0] for v in train_samples)
        steps_per_epoch = math.ceil(len(train_set) / args.batch_size)
        row["training"] = {
            "median_step_ms": step_ms,
            "samples_ms": [v[0] for v in train_samples],
            "peak_allocated_mib": max(v[1] for v in train_samples),
            "train_snapshots": len(train_set),
            "steps_per_epoch": steps_per_epoch,
            "projected_compute_epoch_s": step_ms * steps_per_epoch / 1000.0,
        }
        output["models"][method["name"]] = row

        del batch, sample, train_set, bundle
        loaded.close()
        gc.collect()
        torch.cuda.empty_cache()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
