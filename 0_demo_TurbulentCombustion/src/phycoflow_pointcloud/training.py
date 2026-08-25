"""Collaborator-facing training command over the frozen RC training engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from .cli_utils import replaced_argv, temporary_yaml
from .config import load_public_config
from .models.factory import build_pointcloud_model


def _yaml_value(value: str) -> Any:
    return yaml.safe_load(value)


def _set_overrides(entries: list[str]) -> dict[str, Any]:
    result = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected KEY=YAML_VALUE for --set, got {entry!r}.")
        key, value = entry.split("=", 1)
        result[key] = _yaml_value(value)
    return result


def _model_schema_digest(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--dataset-stats-path", type=Path)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--device-ids", type=int, nargs="+")
    parser.add_argument("--demo-num", type=int)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args(argv)
    overrides = _set_overrides(cli.set)
    overrides.update(
        {
            "data": cli.data,
            "dataset_stats_path": cli.dataset_stats_path,
            "save_dir": cli.save_dir,
            "device_ids": cli.device_ids,
            "Demo_Num": cli.demo_num,
        }
    )
    config = load_public_config(cli.config, overrides=overrides)
    if cli.dry_run:
        build_config = dict(config)
        build_config["neighbor_backend"] = "torch"
        torch.manual_seed(int(config.get("seed", 42)))
        model = build_pointcloud_model(build_config, n_fields=5, device="cpu")
        print(
            json.dumps(
                {
                    "status": "valid",
                    "model_name": config["model_name"],
                    "backbone": config["backbone"],
                    "coord_dim": config["coord_dim"],
                    "condition_attention_execution": config.get(
                        "condition_attention_execution", "legacy_mha"
                    ),
                    "sensor_attention_padding_mode": config.get(
                        "sensor_attention_padding_mode", "full"
                    ),
                    "state_key_count": len(model.state_dict()),
                    "model_schema_sha256": _model_schema_digest(model),
                    "data": config.get("data"),
                    "dataset_stats_path": config.get("dataset_stats_path"),
                    "save_dir": config.get("save_dir"),
                },
                indent=2,
            )
        )
        return

    from train_pointcloud_ffm import main as legacy_main

    with temporary_yaml(config) as resolved_path:
        legacy_argv = ["train_pointcloud_ffm.py", "--config", str(resolved_path)]
        with replaced_argv(legacy_argv):
            legacy_main()
