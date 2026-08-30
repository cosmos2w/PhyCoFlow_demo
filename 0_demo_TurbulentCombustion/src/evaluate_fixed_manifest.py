"""Fixed-manifest evaluation boundary using the public config/checkpoint APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phycoflow_pointcloud.cli_utils import replaced_argv, temporary_yaml
from phycoflow_pointcloud.config import load_public_config, project_root


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=project_root() / "_CheckNotes/Stage1_fixed_val_manifest.pt",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--rf-seed", type=int, default=1729)
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root() / "runs/evaluation/fixed_manifest.json",
    )
    cli = parser.parse_args(argv)
    if not cli.manifest.is_file():
        raise SystemExit(
            f"Validation manifest not found: {cli.manifest}. Supply --manifest explicitly."
        )
    config = load_public_config(cli.config)
    from evaluate_pointcloud_fixed_manifest import main as legacy_main

    with temporary_yaml(config) as resolved_path:
        legacy_argv = [
            "evaluate_pointcloud_fixed_manifest.py",
            "--config",
            str(resolved_path),
            "--manifest",
            str(cli.manifest.resolve()),
            "--checkpoint",
            *[str(path.resolve()) for path in cli.checkpoint],
            "--device",
            cli.device,
            "--batch-size",
            str(cli.batch_size),
            "--repeats",
            str(cli.repeats),
            "--rf-seed",
            str(cli.rf_seed),
            "--output",
            str(cli.output.resolve()),
        ]
        with replaced_argv(legacy_argv):
            legacy_main()
    result = json.loads(cli.output.read_text())
    result.update(
        {
            "config": str(cli.config.resolve()),
            "config_sha256": hashlib.sha256(cli.config.read_bytes()).hexdigest(),
            "public_model_name": config["model_name"],
            "internal_backbone": config["backbone"],
        }
    )
    cli.output.write_text(json.dumps(result, indent=2) + "\n")
