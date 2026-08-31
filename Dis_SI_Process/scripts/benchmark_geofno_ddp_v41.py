#!/usr/bin/env python
"""Two-GPU canonical Geo-FNO DDP training-update replay for Figure 5 V4.1.

Launch with ``CUDA_VISIBLE_DEVICES=1,2 torchrun --standalone --nproc-per-node=2``
and pass ``--physical-devices 1,2`` when those are the clean physical GPUs.
The global adopted batch of 192 is split evenly across two devices. The timed
boundary matches V4's preloaded-batch update core and adds native DDP gradient
communication. No checkpoint, dataset, optimizer archive, or raw cache is
written.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
# The three Ada devices are connected through PCIe host bridges without
# NVLink. Direct NCCL P2P hangs on this host for the non-adjacent 0/2 pair;
# shared-memory transport is the stable local path and remains GPU-resident at
# the model boundary.
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
DEMO_ROOT = REPO_ROOT / "0_demo_TurbulentCombustion"
for path in (
    DEMO_ROOT / "Save_TrainedModel" / "_TrainedModels" / "_Scripts",
    DEMO_ROOT / "src",
    DEMO_ROOT / "tools",
    PACKAGE_ROOT / "scripts",
):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark_training_replay_v4 import (  # noqa: E402
    _baseline_batch,
    _move_optimizer_state,
    _native_errors,
    _quiet_epoch,
    _resolve,
    build_replay_plan,
    sha256_file,
    write_csv,
)
from common.config import load_config  # noqa: E402
from common.model_loader import baseline_lib, load_model  # noqa: E402

TIMING_SCHEMA_VERSION = "figure5-validation-v4.2-geofno-ddp-timing-1"
MEMORY_SCHEMA_VERSION = "figure5-validation-v4.1-geofno-ddp-memory-1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "training_cost_audit_v4.yaml")
    parser.add_argument("--audit-root", type=Path, default=PACKAGE_ROOT / "results" / "ValidationV4" / "TrainingCost")
    parser.add_argument("--audit-run-id", default="training_cost_formal_v4")
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT / "results" / "ValidationV41" / "GeoFNOMultiGPU")
    parser.add_argument("--run-id", default="geofno_ddp_formal_v41")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-in-memory-replay", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--allow-contaminated-pilot", action="store_true")
    parser.add_argument("--warmup-updates", type=int)
    parser.add_argument("--measured-blocks", type=int)
    parser.add_argument("--updates-per-block", type=int)
    parser.add_argument("--transport-test-only", action="store_true")
    parser.add_argument("--memory-only", action="store_true", help="Promote process-local allocated memory only; wall timing remains inadmissible.")
    parser.add_argument(
        "--physical-devices",
        default="0,2",
        help="Expected physical CUDA_VISIBLE_DEVICES pair (for example 1,2).",
    )
    return parser.parse_args()


def _physical_devices(expected: str) -> list[int]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    try:
        devices = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must contain physical integer indices") from exc
    try:
        expected_devices = [int(value.strip()) for value in expected.split(",") if value.strip()]
    except ValueError as exc:
        raise RuntimeError("--physical-devices must contain physical integer indices") from exc
    if len(expected_devices) != 2 or len(set(expected_devices)) != 2:
        raise RuntimeError(f"Expected exactly two distinct physical devices, found {expected!r}")
    if devices != expected_devices:
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={raw!r} does not match --physical-devices={expected!r}"
        )
    return devices


def _gpu_inventory(devices: list[int]) -> list[dict[str, Any]]:
    rows = []
    for index in devices:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(index),
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        physical, uuid, name, memory, driver = (value.strip() for value in output.split(",", 4))
        rows.append({"physical_index": int(physical), "uuid": uuid, "name": name, "memory_total_mib": float(memory), "driver": driver})
    return rows


def _compute_processes() -> list[dict[str, str]]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader"],
        text=True,
    ).strip()
    rows = []
    for line in output.splitlines():
        if line.strip():
            uuid, pid, name, memory = (value.strip() for value in line.split(",", 3))
            rows.append({"gpu_uuid": uuid, "pid": pid, "process_name": name, "used_memory": memory})
    return rows


def _foreign_processes(inventory: list[dict[str, Any]], allowed_pids: set[int]) -> list[dict[str, str]]:
    uuids = {row["uuid"] for row in inventory}
    return [row for row in _compute_processes() if row["gpu_uuid"] in uuids and int(row["pid"]) not in allowed_pids]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def _quantiles(values: list[float]) -> tuple[float, float, float]:
    return tuple(float(value) for value in np.quantile(np.asarray(values, dtype=float), [0.25, 0.5, 0.75]))


def main() -> int:
    args = _args()
    if not args.execute or not args.confirm_in_memory_replay:
        print("Execution requires --execute --confirm-in-memory-replay", file=sys.stderr)
        return 2
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"Geo-FNO DDP replay requires exactly two ranks, found {world}")
    devices = _physical_devices(args.physical_devices)
    inventory = _gpu_inventory(devices) if rank == 0 else None
    inventory_holder = [inventory]
    dist.broadcast_object_list(inventory_holder, src=0)
    inventory = inventory_holder[0]
    pids: list[int] = [0 for _ in range(world)]
    dist.all_gather_object(pids, os.getpid())
    foreign_before = _foreign_processes(inventory, set(pids)) if rank == 0 else None
    before_holder = [foreign_before]
    dist.broadcast_object_list(before_holder, src=0)
    foreign_before = before_holder[0]
    if foreign_before and not (args.memory_only or (args.pilot and args.allow_contaminated_pilot)):
        raise RuntimeError(f"Formal two-GPU replay requires clean GPUs {devices}: {foreign_before}")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    audit_path = args.audit_root / args.audit_run_id / "audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    plan = build_replay_plan(config, audit)
    stage = next(row for row in plan["rows"] if row["method"] == "Geo-FNO")
    expected = next(row["sha256"] for row in config["checkpoints"] if row["method"] == "Geo-FNO")
    checkpoint_path = _resolve(stage["checkpoint_path"])
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise RuntimeError("Geo-FNO checkpoint identity mismatch")
    adopted_batch = int(stage["training_config"]["batch_size"])
    if adopted_batch != 192 or adopted_batch % world:
        raise RuntimeError(f"Expected adopted Geo-FNO batch 192 divisible by two, found {adopted_batch}")
    local_batch = adopted_batch // world

    protocol = plan["protocol"]
    memory_protocol = (1, 3, 1)
    default_protocol = memory_protocol if args.memory_only else (2, 3, 2) if args.pilot else (
        int(protocol["warmup_updates"]), int(protocol["measured_blocks"]), int(protocol["updates_per_block"])
    )
    warmups = int(args.warmup_updates if args.warmup_updates is not None else default_protocol[0])
    blocks = int(args.measured_blocks if args.measured_blocks is not None else default_protocol[1])
    per_block = int(args.updates_per_block if args.updates_per_block is not None else default_protocol[2])
    if not args.pilot and args.memory_only and (warmups, blocks, per_block) != memory_protocol:
        raise RuntimeError("Formal memory replay requires 1 warmup and 3×1 measured updates")
    if not args.pilot and not args.memory_only and (warmups, blocks, per_block) != (20, 10, 10):
        raise RuntimeError("Formal timing replay requires 20 warmups and 10×10 measured updates")

    run_dir = args.output_root / args.run_id
    exists = run_dir.exists() if rank == 0 else None
    exists_holder = [exists]
    dist.broadcast_object_list(exists_holder, src=0)
    if exists_holder[0]:
        raise RuntimeError(f"Refusing to overwrite {run_dir}")
    if rank == 0:
        run_dir.mkdir(parents=True)
        (run_dir / "gpu_state_before.json").write_text(json.dumps({"inventory": inventory, "foreign": foreign_before}, indent=2), encoding="utf-8")
    dist.barrier()

    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    nccl_group = dist.new_group(backend="nccl")
    probe = torch.tensor([float(rank + 1)], device=device)
    dist.all_reduce(probe, group=nccl_group)
    if not torch.isclose(probe, torch.tensor([3.0], device=device)).item():
        raise RuntimeError(f"NCCL transport probe failed on rank {rank}: {probe.item()}")
    if args.transport_test_only:
        if rank == 0:
            print(json.dumps({"transport": "pass", "NCCL_P2P_DISABLE": os.environ["NCCL_P2P_DISABLE"], "devices": devices}))
        dist.barrier()
        dist.destroy_process_group(nccl_group)
        dist.destroy_process_group()
        return 0
    torch.manual_seed(20260831 + rank)
    np.random.seed((20260831 + rank) & 0xFFFFFFFF)
    torch.cuda.manual_seed_all(20260831 + rank)
    post_config = load_config()
    method_config = next(row for row in post_config["methods"] if row["name"] == "Geo-FNO")
    loaded = load_model(method_config, "Cond_T", checkpoint="last", split="train", device=str(device), n_steps=2, ode_solver="euler")
    bundle = loaded.model
    if bundle.optimizer is None:
        raise RuntimeError("Geo-FNO bundle has no optimizer")
    _move_optimizer_state(bundle.optimizer, device)
    batch = _baseline_batch(loaded, local_batch)
    bundle.model = DistributedDataParallel(
        bundle.model,
        device_ids=[rank],
        output_device=rank,
        broadcast_buffers=False,
        process_group=nccl_group,
    )

    def update() -> float:
        return _quiet_epoch(baseline_lib, lambda: bundle.adapter.run_epoch(bundle, [batch], training=True, epoch=0))

    def measured_update() -> tuple[float, float, float, list[float], list[float]]:
        dist.barrier(group=nccl_group)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        loss = float(update())
        torch.cuda.synchronize(device)
        local_elapsed = (time.perf_counter() - start) * 1000.0
        local_peak = torch.cuda.max_memory_allocated(device) / 2**20
        elapsed_tensor = torch.tensor([local_elapsed], dtype=torch.float64, device=device)
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX, group=nccl_group)
        elapsed_values = [torch.zeros_like(elapsed_tensor) for _ in range(world)]
        peak_tensor = torch.tensor([local_peak], dtype=torch.float64, device=device)
        peak_values = [torch.zeros_like(peak_tensor) for _ in range(world)]
        dist.all_gather(elapsed_values, torch.tensor([local_elapsed], dtype=torch.float64, device=device), group=nccl_group)
        dist.all_gather(peak_values, peak_tensor, group=nccl_group)
        return loss, float(elapsed_tensor.item()), local_peak, [float(value.item()) for value in elapsed_values], [float(value.item()) for value in peak_values]

    for _ in range(warmups):
        measured_update()
    update_rows: list[dict[str, Any]] = []
    global_values: list[float] = []
    local_values: list[float] = []
    peak_by_rank = [0.0 for _ in range(world)]
    for block in range(blocks):
        for offset in range(per_block):
            loss, global_elapsed, local_peak, elapsed_ranks, peak_ranks = measured_update()
            global_values.append(global_elapsed)
            local_values.append(elapsed_ranks[rank])
            peak_by_rank = [max(peak_by_rank[index], peak_ranks[index]) for index in range(world)]
            if rank == 0:
                update_rows.append(
                    {
                        "block": block,
                        "update_in_block": offset,
                        "update_index": block * per_block + offset,
                        "global_wall_ms": global_elapsed,
                        "rank0_wall_ms": elapsed_ranks[0],
                        "rank1_wall_ms": elapsed_ranks[1],
                        "rank0_peak_allocated_mib": peak_ranks[0],
                        "rank1_peak_allocated_mib": peak_ranks[1],
                        "rank0_loss": loss,
                    }
                )
    local_all: list[list[float]] = [[] for _ in range(world)]
    dist.all_gather_object(local_all, local_values)

    block_medians = [statistics.median(global_values[index : index + per_block]) for index in range(0, len(global_values), per_block)]
    half = len(block_medians) // 2
    early = statistics.median(block_medians[:half])
    late = statistics.median(block_medians[half:])
    global_stability = abs(early - late) / max(statistics.median(global_values), 1e-12)
    rank_stabilities = []
    for values in local_all:
        medians = [statistics.median(values[index : index + per_block]) for index in range(0, len(values), per_block)]
        split = len(medians) // 2
        rank_stabilities.append(abs(statistics.median(medians[:split]) - statistics.median(medians[split:])) / max(statistics.median(values), 1e-12))

    del batch, bundle, loaded
    gc.collect()
    torch.cuda.empty_cache()
    dist.barrier(group=nccl_group)
    foreign_after = _foreign_processes(inventory, set(pids)) if rank == 0 else None
    after_holder = [foreign_after]
    dist.broadcast_object_list(after_holder, src=0)
    foreign_after = after_holder[0]

    if rank == 0:
        q25, median, q75 = _quantiles(global_values)
        rank0_peaks = [float(row["rank0_peak_allocated_mib"]) for row in update_rows]
        rank1_peaks = [float(row["rank1_peak_allocated_mib"]) for row in update_rows]
        total_peaks = [left + right for left, right in zip(rank0_peaks, rank1_peaks)]
        memory_span = max(total_peaks) - min(total_peaks)
        errors = _native_errors()["Geo-FNO"]
        summary = {
            "method": "Geo-FNO",
            "status": "ok",
            "device_count": world,
            "global_batch_size": adopted_batch,
            "local_batch_size": local_batch,
            "warmup_updates": warmups,
            "measured_updates": len(global_values),
            "measured_blocks": blocks,
            "updates_per_block": per_block,
            "wall_time_q25_ms": q25,
            "wall_time_median_ms": median,
            "wall_time_q75_ms": q75,
            "gpu_ms_q25_per_update": q25 * world,
            "gpu_ms_per_update": median * world,
            "gpu_ms_q75_per_update": q75 * world,
            "peak_allocated_mib_rank0": peak_by_rank[0],
            "peak_allocated_mib_rank1": peak_by_rank[1],
            "peak_allocated_mib_per_device_max": max(peak_by_rank),
            "peak_allocated_mib_total": sum(peak_by_rank),
            "peak_allocated_mib_total_min_repeat": min(total_peaks),
            "peak_allocated_mib_total_max_repeat": max(total_peaks),
            "peak_allocated_mib_total_span": memory_span,
            "global_stability_delta_fraction": global_stability,
            "rank0_stability_delta_fraction": rank_stabilities[0],
            "rank1_stability_delta_fraction": rank_stabilities[1],
            "error": float(errors["error"]),
            "error_ci_low": float(errors["error_ci_low"]),
            "error_ci_high": float(errors["error_ci_high"]),
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)),
            "checkpoint_sha256": actual,
        }
        formal = not args.pilot
        tolerance = float(config["promotion"]["replay_validation_tolerance_fraction"])
        memory_repeatable = memory_span <= 1.0
        qa = {
            "status": "pass" if (
                (not formal or (len(global_values) == 3 if args.memory_only else len(global_values) == 100))
                and adopted_batch == 192
                and world == 2
                and actual == expected
                and (memory_repeatable if args.memory_only else (global_stability <= tolerance and max(rank_stabilities) <= tolerance))
                and (args.memory_only or not foreign_before)
                and (args.memory_only or not foreign_after)
            ) else "fail",
            "global_batch_exact": adopted_batch == 192,
            "two_distinct_gpus": len({row["uuid"] for row in inventory}) == 2,
            "checkpoint_identity_pass": actual == expected,
            "all_rank_stability_pass": global_stability <= tolerance and max(rank_stabilities) <= tolerance,
            "memory_repeatability_pass": memory_repeatable,
            "process_local_allocated_metric": args.memory_only,
            "gpu_clean_before": not foreign_before,
            "gpu_clean_after": not foreign_after,
            "formal_sample_count_exact": (not formal) or (len(global_values) == 3 if args.memory_only else len(global_values) == 100),
            "no_archive_write": True,
            "tolerance_fraction": tolerance,
        }
        manifest = {
            "schema_version": MEMORY_SCHEMA_VERSION if args.memory_only else TIMING_SCHEMA_VERSION,
            "status": "complete" if qa["status"] == "pass" else "failed_qa",
            "formal": formal and qa["status"] == "pass",
            "run_id": args.run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "visible_physical_devices": devices,
                "inventory": inventory,
                "dtype": "float32",
            },
            "protocol": {
                "parallelism": "DistributedDataParallel",
                "backend": "nccl",
                "global_batch_size": adopted_batch,
                "local_batch_size": local_batch,
                "warmup_updates": warmups,
                "measured_blocks": blocks,
                "updates_per_block": per_block,
                "timing_boundary": "synchronized_update_core_preloaded_batch_with_ddp_gradient_communication",
                "metric": "wall_time_median_ms = synchronized max-rank wall ms/global optimizer update",
                "secondary_metric": "gpu_ms_per_update = max_rank_wall_ms * 2",
                "memory_metric": "maximum torch.cuda.max_memory_allocated across the two devices",
                "promoted_metric": "summed simultaneous per-rank peak allocated MiB" if args.memory_only else "synchronized wall ms/global optimizer update",
                "wall_timing_admissible": not args.memory_only and not foreign_before and not foreign_after,
                "foreign_processes_ignored_for_process_local_memory": bool(args.memory_only and (foreign_before or foreign_after)),
                "included": ["device_side_batch_preparation", "forward", "loss", "backward", "DDP_gradient_allreduce", "gradient_clipping", "optimizer_step"],
                "excluded": ["dataset_IO", "host_transfer", "validation", "logging", "checkpointing"],
            },
            "checkpoint": {"path": str(checkpoint_path.relative_to(REPO_ROOT)), "sha256": actual},
            "safety": {"checkpoint_write": False, "archive_mutation": False, "raw_cache_write": False},
        }
        write_csv(run_dir / "geofno_ddp_updates.csv", update_rows)
        write_csv(run_dir / "geofno_ddp_summary.csv", [summary])
        _atomic_json(run_dir / "qa.json", qa)
        _atomic_json(run_dir / "manifest.json", manifest)
        (run_dir / "gpu_state_after.json").write_text(json.dumps({"inventory": inventory, "foreign": foreign_after}, indent=2), encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "qa": qa["status"], "summary": summary}, indent=2))
    dist.barrier()
    dist.destroy_process_group(nccl_group)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
