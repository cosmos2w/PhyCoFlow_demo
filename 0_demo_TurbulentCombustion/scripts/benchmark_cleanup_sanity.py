#!/usr/bin/env python3
"""Run the focused RC1-vs-cleanup performance sanity checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmark_pointcloud_cq import make_inputs
from benchmark_pointcloud_stage7 import (
    benchmark_training_step,
    persistent_inference,
)
from phycoflow_pointcloud.checkpointing import resolve_checkpoint_state
from phycoflow_pointcloud.config import load_public_config
from phycoflow_pointcloud.models.factory import build_pointcloud_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_CheckNotes/Stage7_cleanup/phase5/performance_sanity.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    config = load_public_config(ROOT / "configs/gl_rbf_cq.yaml")
    checkpoint_path = (
        ROOT
        / json.loads(
            (ROOT / "artifacts/GL_rbf_CQ_v0.9.0-rc1_portable.json").read_text()
        )["artifact"]
    )

    torch.manual_seed(42)
    training_model = build_pointcloud_model(config, n_fields=5, device=device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    training_model.load_state_dict(
        resolve_checkpoint_state(checkpoint, model=training_model).state_dict
    )
    values = make_inputs(4096, 256, 5, device, seed=42, batch_size=128)
    training = benchmark_training_step(
        training_model,
        values,
        ema_enabled=True,
        iterations=args.iterations,
        warmup=args.warmup,
        device=device,
        query_microbatch_size=2048,
    )
    del training_model, values
    torch.cuda.empty_cache()

    torch.manual_seed(42)
    inference_model = build_pointcloud_model(config, n_fields=5, device=device)
    inference_model.load_state_dict(
        resolve_checkpoint_state(checkpoint, model=inference_model).state_dict
    )
    inference = persistent_inference(
        "GL_rbf_CQ-cleanup",
        inference_model,
        n_query=1_000_000,
        n_obs=256,
        chunk_size=32768,
        device=device,
    )
    baseline_path = (
        ROOT / "_CheckNotes/Stage7_smart_cq/benchmarks/pretraining_cost.json"
    )
    baseline = json.loads(baseline_path.read_text())
    rc_train = next(
        row
        for row in baseline["formal_training_step"]
        if row["label"] == "Stage7-All256"
    )
    rc_inference = next(
        row
        for row in baseline["persistent_inference"]
        if row["label"] == "Stage7-All256"
    )
    comparison = {
        "train_step_ratio_cleanup_over_rc1": training["full_step_ms"]
        / rc_train["full_step_ms"],
        "train_memory_ratio_cleanup_over_rc1": training["peak_allocated_mb"]
        / rc_train["peak_allocated_mb"],
        "persistent_nfe4_ratio_cleanup_over_rc1": inference["steady_nfe4_s"]
        / rc_inference["steady_nfe4_s"],
        "persistent_memory_ratio_cleanup_over_rc1": (
            inference["steady_nfe4_peak_allocated_mb"]
            / rc_inference["steady_nfe4_peak_allocated_mb"]
        ),
    }
    result = {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "protocol": {
            "training": "B128/Q4096, query microbatch 2048, EMA, AdamW",
            "inference": "1M queries, 256 observations, persistent static_features, Euler NFE4",
        },
        "rc1_source": str(baseline_path.relative_to(ROOT)),
        "rc1_training": rc_train,
        "cleanup_training": training,
        "rc1_persistent": rc_inference,
        "cleanup_persistent": inference,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
