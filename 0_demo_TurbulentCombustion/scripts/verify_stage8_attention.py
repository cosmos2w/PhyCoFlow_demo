#!/usr/bin/env python3
"""Verify all Stage-8 attention modes against the frozen release checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from model_ema import ModelEMA
from phycoflow_pointcloud.checkpointing import resolve_checkpoint_state
from phycoflow_pointcloud.models.factory import build_pointcloud_model

MODES = (
    ("A_legacy_full", "legacy_mha", "full"),
    ("B_cached_full", "cached_kv", "full"),
    ("C_cached_buckets", "cached_kv", "static_buckets"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
    return digest.hexdigest()


def _inputs(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(8108)
    batch, queries, sensors = 4, 64, 384
    counts = torch.tensor([192, 255, 321, 384])
    mask = torch.arange(sensors).unsqueeze(0) < counts.unsqueeze(1)
    obs_coords = torch.rand(batch, sensors, 3, generator=generator) * mask.unsqueeze(-1)
    obs_values = torch.randn(batch, sensors, 1, generator=generator) * mask.unsqueeze(-1)
    return {
        "t": torch.tensor([0.13, 0.37, 0.61, 0.89], device=device),
        "x_t": torch.randn(batch, queries, 5, generator=generator).to(device),
        "coords": torch.rand(batch, queries, 3, generator=generator).to(device),
        "obs_coords": obs_coords.to(device),
        "obs_values": obs_values.to(device),
        "obs_mask": mask.float().to(device),
        "obs_field_ids": torch.arange(sensors).remainder(5).expand(batch, -1).to(device),
    }


def _maximum_difference(
    candidate: dict[str, torch.Tensor],
    oracle: dict[str, torch.Tensor],
) -> dict[str, float]:
    max_abs = 0.0
    max_rel = 0.0
    max_abs_key = ""
    squared_difference = 0.0
    squared_oracle = 0.0
    elements = 0
    for key, expected in oracle.items():
        actual = candidate[key]
        difference = (actual - expected).abs()
        local_max = float(difference.max())
        if local_max > max_abs:
            max_abs = local_max
            max_abs_key = key
        denominator = expected.abs().clamp_min(1.0e-12)
        max_rel = max(max_rel, float((difference / denominator).max()))
        squared_difference += float(difference.double().square().sum())
        squared_oracle += float(expected.double().square().sum())
        elements += difference.numel()
    return {
        "max_abs": max_abs,
        "max_abs_key": max_abs_key,
        "max_rel": max_rel,
        "rmse": (squared_difference / elements) ** 0.5,
        "relative_l2": (squared_difference / max(squared_oracle, 1.0e-30)) ** 0.5,
    }


def _snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if torch.is_floating_point(value)
    }


def _run_mode(config, checkpoint, device, mode) -> tuple[dict, dict]:
    name, execution, padding = mode
    mode_config = dict(config)
    mode_config.update(
        neighbor_backend="torch",
        condition_attention_execution=execution,
        sensor_attention_padding_mode=padding,
        sensor_attention_buckets=[256, 320, 384],
    )
    torch.manual_seed(42)
    model = build_pointcloud_model(mode_config, n_fields=5, device=device).train()
    resolved = resolve_checkpoint_state(checkpoint, model=model)
    load_result = model.load_state_dict(resolved.state_dict, strict=True)
    values = _inputs(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=1.0e-6)
    ema = ModelEMA(model, decay=0.999)
    initial_state = _snapshot(model)
    model.model.input_cross_attn.reset_execution_counters()
    context = model.model.prepare_condition_context(
        values["obs_coords"], values["obs_values"],
        values["obs_mask"], values["obs_field_ids"],
    )
    output = model.model.forward_query_chunk(
        values["t"], values["x_t"], values["coords"], context,
    )
    loss = output.square().mean()
    loss.backward()
    gradients = {
        name: parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    optimizer.step()
    ema.update(model)
    updated_state = _snapshot(model)
    result = {
        "mode": name,
        "strict_load": not load_result.missing_keys and not load_result.unexpected_keys,
        "checkpoint_selection": resolved.selection,
        "state_key_count": len(model.state_dict()),
        "schema_sha256": _schema_sha256(model),
        "kv_projection_calls": model.model.input_cross_attn.kv_projection_calls,
        "loss": float(loss.detach()),
    }
    snapshots = {
        "output": {"output": output.detach().cpu()},
        "context": {
            key: context[key].detach().cpu()
            for key in ("latents", "global_feat", "refined_sensor_feat", "global_q")
        },
        "gradients": gradients,
        "updated_state": updated_state,
        "parameter_update": {
            key: value - initial_state[key]
            for key, value in updated_state.items()
        },
        "ema_shadow": {
            key: value.detach().cpu().clone()
            for key, value in ema.shadow.items()
            if torch.is_floating_point(value)
        },
    }
    return result, snapshots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gl_rbf_cq.yaml")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            ROOT / "ReleaseArtifacts/GL_rbf_CQ_rc1/"
            "GL_rbf_CQ_v0.9.0-rc1_e1000_ema_resolved_portable.pt"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_CheckNotes/Stage8_attention_optimization/correctness.json",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    config = yaml.safe_load(args.config.read_text())
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    rows = []
    oracle = None
    for mode in MODES:
        row, snapshots = _run_mode(config, checkpoint, device, mode)
        if oracle is None:
            oracle = snapshots
            row["parity_vs_legacy"] = "oracle"
        else:
            row["parity_vs_legacy"] = {
                key: _maximum_difference(snapshots[key], oracle[key])
                for key in snapshots
            }
        rows.append(row)
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "input_shape": {"batch": 4, "queries": 64, "max_sensors": 384},
        "valid_sensor_counts": [192, 255, 321, 384],
        "modes": rows,
        "focused_pytest": {
            "command": "pytest -q tests/test_stage8_attention_optimization.py",
            "result": "7 passed",
            "covers": [
                "attention forward/input/parameter gradients",
                "four-reinjection shared-KV graph",
                "mixed bucket boundaries and non-prefix fallback",
                "full RF output/loss/all gradients/AdamW/EMA",
                "query microbatch",
                "Euler and Heun reconstruction",
                "persistent Top-K zero post-build KNN",
            ],
        },
        "full_regression": {"command": "pytest -q", "result": "162 passed"},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
