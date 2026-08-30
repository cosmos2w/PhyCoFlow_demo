#!/usr/bin/env python
"""Compute one missing joint-PDF/JSD pair with CUDA.

This helper is intentionally read-only with respect to reconstruction caches.
The complete transient payload is emitted as JSON on stdout; no metric files
are written.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _pair_arrays(path: Path, x_index: int, y_index: int, *, truth: bool) -> tuple[np.ndarray | None, np.ndarray]:
    """Read only the pair columns needed from one compressed cache."""
    with np.load(path, allow_pickle=False) as payload:
        reconstruction = np.asarray(
            payload["recon_phys"][:, [x_index, y_index]], dtype=np.float32,
        )
        ground_truth = (
            np.asarray(payload["truth_phys"][:, [x_index, y_index]], dtype=np.float32)
            if truth else None
        )
    return ground_truth, reconstruction


def _select_device(requested: int | None) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA-enabled PyTorch is required; CPU fallback is disabled.")
    if requested is not None:
        if not 0 <= requested < torch.cuda.device_count():
            raise ValueError(f"CUDA device {requested} is unavailable.")
        return torch.device(f"cuda:{requested}")
    free_by_device = []
    for index in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(index)
        free_by_device.append((int(free_bytes), index))
    return torch.device(f"cuda:{max(free_by_device)[1]}")


def _quantile_edges(values: np.ndarray, bins: int, quantiles: tuple[float, float],
                    device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    tensor = tensor[torch.isfinite(tensor)]
    if tensor.numel() == 0:
        raise ValueError("Cannot derive histogram edges from an empty finite tensor.")
    # ``torch.quantile`` has a CUDA element-count ceiling below the full
    # 40.3-million-point truth vector. Two GPU kth-value selections plus
    # linear interpolation reproduce NumPy's default quantile definition
    # without subsampling or falling back to CPU.
    selected = []
    for quantile in quantiles:
        rank = float(quantile) * float(tensor.numel() - 1)
        lower = int(np.floor(rank))
        upper = int(np.ceil(rank))
        lower_value = torch.kthvalue(tensor, lower + 1).values
        if upper == lower:
            selected.append(lower_value)
        else:
            upper_value = torch.kthvalue(tensor, upper + 1).values
            selected.append(lower_value + (upper_value - lower_value) * (rank - lower))
    limits = torch.stack(selected)
    if not bool(limits[1] > limits[0]):
        limits = torch.stack((limits[0] - 0.5, limits[1] + 0.5))
    return torch.linspace(limits[0], limits[1], bins + 1, device=device)


def _histogram_probability(pair: np.ndarray, x_edges: torch.Tensor,
                           y_edges: torch.Tensor) -> torch.Tensor:
    values = torch.as_tensor(pair, dtype=torch.float32, device=x_edges.device)
    finite = torch.isfinite(values).all(dim=1)
    values = values[finite]
    bins = int(x_edges.numel() - 1)
    x_values = values[:, 0].contiguous()
    y_values = values[:, 1].contiguous()
    x_bin = torch.bucketize(x_values, x_edges, right=True) - 1
    y_bin = torch.bucketize(y_values, y_edges, right=True) - 1
    # NumPy histogram semantics include the rightmost edge in the final bin.
    # ``bucketize(..., right=True)`` otherwise places exact maxima one index
    # beyond the valid range, which matters for bounded mass-fraction fields.
    x_bin = torch.where(x_values == x_edges[-1], bins - 1, x_bin)
    y_bin = torch.where(y_values == y_edges[-1], bins - 1, y_bin)
    valid = (x_bin >= 0) & (x_bin < bins) & (y_bin >= 0) & (y_bin < bins)
    flat = x_bin[valid] * bins + y_bin[valid]
    counts = torch.bincount(flat, minlength=bins * bins).to(torch.float64)
    return counts / torch.clamp(counts.sum(), min=1.0)


def _jsd_base2(p: torch.Tensor, q: torch.Tensor, pseudocount: float) -> float:
    p = p + float(pseudocount)
    q = q + float(pseudocount)
    p = p / p.sum()
    q = q / q.sum()
    midpoint = 0.5 * (p + q)
    value = 0.5 * torch.sum(p * torch.log2(p / midpoint))
    value += 0.5 * torch.sum(q * torch.log2(q / midpoint))
    return float(value.item())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--x-index", type=int, required=True)
    parser.add_argument("--y-index", type=int, required=True)
    parser.add_argument("--proposed-model", required=True)
    parser.add_argument("--methods", nargs="+", required=True)
    parser.add_argument("--bins", type=int, default=64)
    parser.add_argument("--quantiles", type=float, nargs=2, default=(0.005, 0.995))
    parser.add_argument("--pseudocount", type=float, default=1.0e-12)
    parser.add_argument("--pdf-frame-count", type=int, default=25)
    parser.add_argument("--device", type=int)
    parser.add_argument("--max-snapshots", type=int, help="QA-only snapshot limit.")
    args = parser.parse_args()

    device = _select_device(args.device)
    torch.cuda.set_device(device)
    rows = [
        row for row in _read_manifest(args.manifest)
        if row.get("condition") == args.condition and row.get("method") in set(args.methods)
    ]
    lookup = {(row["method"], int(row["snapshot"])): row for row in rows}
    snapshots = sorted({int(row["snapshot"]) for row in rows})
    if args.max_snapshots is not None:
        snapshots = snapshots[:max(int(args.max_snapshots), 0)]
    if not snapshots:
        raise RuntimeError("No matching snapshots were found in the cache manifest.")

    truth_pairs: dict[int, np.ndarray] = {}
    proposed_recon_pairs: dict[int, np.ndarray] = {}
    valid_snapshots = []
    for number, snapshot in enumerate(snapshots, start=1):
        entry = lookup.get((args.proposed_model, snapshot), {})
        cache_path = Path(entry.get("cache_path", ""))
        if entry.get("status") != "ok" or not cache_path.is_file():
            continue
        truth, reconstruction = _pair_arrays(
            cache_path, args.x_index, args.y_index, truth=True,
        )
        if truth is None:
            continue
        truth_pairs[snapshot] = truth
        proposed_recon_pairs[snapshot] = reconstruction
        valid_snapshots.append(snapshot)
        if number % 250 == 0:
            print(f"[GPU] loaded truth bounds {number}/{len(snapshots)}", file=sys.stderr, flush=True)
    if not valid_snapshots:
        raise RuntimeError("No valid proposed-model truth caches were available.")

    pooled_truth = np.concatenate([truth_pairs[snapshot] for snapshot in valid_snapshots], axis=0)
    quantiles = (float(args.quantiles[0]), float(args.quantiles[1]))
    x_edges = _quantile_edges(pooled_truth[:, 0], args.bins, quantiles, device)
    y_edges = _quantile_edges(pooled_truth[:, 1], args.bins, quantiles, device)
    del pooled_truth

    pdf_indices = np.linspace(0, len(valid_snapshots) - 1, args.pdf_frame_count, dtype=int)
    pdf_snapshots = [valid_snapshots[int(index)] for index in pdf_indices]
    pooled_pdf = np.concatenate([truth_pairs[snapshot] for snapshot in pdf_snapshots], axis=0)
    pdf_x_edges = _quantile_edges(pooled_pdf[:, 0], args.bins, quantiles, device)
    pdf_y_edges = _quantile_edges(pooled_pdf[:, 1], args.bins, quantiles, device)
    pdf_probability = _histogram_probability(pooled_pdf, pdf_x_edges, pdf_y_edges)
    pdf_matrix = pdf_probability.reshape(args.bins, args.bins).cpu().numpy()
    pdf_extent = [
        float(pdf_x_edges[0].item()), float(pdf_x_edges[-1].item()),
        float(pdf_y_edges[0].item()), float(pdf_y_edges[-1].item()),
    ]
    del pooled_pdf

    output_rows = []
    values_by_method: dict[str, list[float]] = {method: [] for method in args.methods}
    expected_n = len(snapshots)
    for number, snapshot in enumerate(snapshots, start=1):
        truth_pair = truth_pairs.get(snapshot)
        truth_probability = (
            _histogram_probability(truth_pair, x_edges, y_edges)
            if truth_pair is not None else None
        )
        for method in args.methods:
            entry = lookup.get((method, snapshot), {})
            cache_path = Path(entry.get("cache_path", ""))
            status, value = entry.get("status", "missing cache"), float("nan")
            if truth_probability is not None and status == "ok" and cache_path.is_file():
                try:
                    if method == args.proposed_model:
                        reconstruction = proposed_recon_pairs[snapshot]
                    else:
                        _, reconstruction = _pair_arrays(
                            cache_path, args.x_index, args.y_index, truth=False,
                        )
                    reconstruction_probability = _histogram_probability(
                        reconstruction, x_edges, y_edges,
                    )
                    value = _jsd_base2(
                        truth_probability, reconstruction_probability, args.pseudocount,
                    )
                except Exception as exc:
                    status = f"gpu postprocess error:{type(exc).__name__}"
                else:
                    status = "ok"
                    values_by_method[method].append(value)
            output_rows.append({
                "method": method, "condition": args.condition, "snapshot": snapshot,
                "pair": args.pair, "jsd_base2": value, "status": status,
                "value_source": "on_the_fly_torch_cuda_read_only_cache",
            })
        if number % 100 == 0 or number == len(snapshots):
            print(f"[GPU] JSD snapshots {number}/{len(snapshots)}", file=sys.stderr, flush=True)

    summary = []
    for method in args.methods:
        values = np.asarray(values_by_method[method], dtype=float)
        summary.append({
            "method": method, "condition": args.condition, "pair": args.pair,
            "n_expected_snapshots": expected_n, "valid_n": int(values.size),
            "mean": float(values.mean()) if values.size else float("nan"),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "median": float(np.median(values)) if values.size else float("nan"),
            "q25": float(np.quantile(values, 0.25)) if values.size else float("nan"),
            "q75": float(np.quantile(values, 0.75)) if values.size else float("nan"),
            "status": "ok" if values.size == expected_n else "incomplete",
        })

    torch.cuda.synchronize(device)
    payload = {
        "rows": output_rows,
        "summary": summary,
        "pdf": {
            "pair": args.pair, "condition": args.condition,
            "matrix": pdf_matrix.tolist(), "extent": pdf_extent,
            "frame_count": len(pdf_snapshots), "snapshots": pdf_snapshots,
        },
        "metadata": {
            "backend": "torch.cuda", "torch_version": torch.__version__,
            "device_index": int(device.index or 0),
            "device_name": torch.cuda.get_device_name(device),
            "pair": args.pair, "condition": args.condition,
            "histogram_bins": int(args.bins), "quantiles": list(quantiles),
            "quantile_backend": "torch.cuda.kthvalue_linear_interpolation",
            "metric_extent": [
                float(x_edges[0].item()), float(x_edges[-1].item()),
                float(y_edges[0].item()), float(y_edges[-1].item()),
            ],
            "expected_snapshots": expected_n,
            "valid_snapshots_by_method": {
                method: len(values_by_method[method]) for method in args.methods
            },
            "cache_access": "read_only",
            "persistent_metric_artifact_written": False,
            "pdf_snapshots": pdf_snapshots,
        },
    }
    print(json.dumps(payload, allow_nan=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
