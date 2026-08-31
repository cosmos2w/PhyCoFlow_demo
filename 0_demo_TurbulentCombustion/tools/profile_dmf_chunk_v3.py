#!/usr/bin/env python
"""Profile DMF query chunking to reconcile the historical exact-shape probe."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

import benchmark_validation_v3 as bench


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = bench.gpu_index(args.device)
    bench.assert_clean_gpu(index, allow_current=False)
    torch.cuda.set_device(index)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    states = list(map(int, plan["cohorts"]["cost_native_20"]["evaluation_indices"]))
    sensors = bench.load_sensor_rows(plan, states)
    post_cfg = bench.load_config()
    method_cfg = next(row for row in post_cfg["methods"] if row["name"] == "DMF-Gen")
    settings = bench.method_settings(post_cfg, "DMF-Gen")
    loaded = bench.load_model(method_cfg, "Cond_T", checkpoint="last", split="test", device=args.device, n_steps=settings["n_steps"], ode_solver="euler")
    rows = []
    try:
        with bench.evaluation_context(loaded):
            prepared = bench.prepare_state(loaded, states[0], sensors[states[0]], 40300, "DMF-Gen")
            reference = None
            for chunk_size in (2048, 4096, 8192, 16384, 40300):
                current = {**settings, "query_chunk_size": chunk_size}
                seed = bench.stable_seed(20260830, "dmf_chunk_profile", "equivalence")
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                result = bench.core_call(loaded, prepared, current, "DMF-Gen")
                if reference is None:
                    reference = result.detach().clone()
                max_abs_error = float(torch.max(torch.abs(result - reference)).item())
                for repeat in range(10):
                    torch.manual_seed(bench.stable_seed(20260830, "dmf_chunk_profile", chunk_size, "warmup", repeat))
                    torch.cuda.manual_seed_all(bench.stable_seed(20260830, "dmf_chunk_profile", chunk_size, "warmup", repeat))
                    bench.core_call(loaded, prepared, current, "DMF-Gen")
                values = []
                for repeat in range(30):
                    seed = bench.stable_seed(20260830, "dmf_chunk_profile", chunk_size, repeat)
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    _, elapsed = bench.time_cuda(lambda: bench.core_call(loaded, prepared, current, "DMF-Gen"), args.device)
                    values.append(elapsed)
                allocated, reserved = bench.measure_memory(lambda: bench.core_call(loaded, prepared, current, "DMF-Gen"), args.device)
                rows.append({
                    "query_chunk_size": chunk_size,
                    **bench.latency_summary(values),
                    "peak_allocated_mib": allocated,
                    "peak_reserved_mib": reserved,
                    "max_abs_error_vs_2048": max_abs_error,
                    "equivalent_atol_2e-5": max_abs_error <= 2.0e-5,
                })
    finally:
        loaded.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bench.write_csv(args.output, rows)
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
