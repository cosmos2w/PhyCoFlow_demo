#!/usr/bin/env python3
"""Export and verify a portable EMA-resolved GL_rbf_CQ checkpoint."""

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

from phycoflow_pointcloud.checkpointing import resolve_checkpoint_state
from phycoflow_pointcloud.models.factory import build_pointcloud_model

EXPECTED_STATE_SHA256 = (
    "f1c92d4bcf1b9e0ac90ad20b3b3468764f6cca200dde1575b914d9ab68d7b99f"
)


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT
        / "ReleaseArtifacts/GL_rbf_CQ_rc1/GL_rbf_CQ_v0.9.0-rc1_e1000_research.pt",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gl_rbf_cq.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT
        / "ReleaseArtifacts/GL_rbf_CQ_rc1/GL_rbf_CQ_v0.9.0-rc1_e1000_ema_resolved_portable.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "artifacts/GL_rbf_CQ_v0.9.0-rc1_portable.json",
    )
    args = parser.parse_args()

    public_config = yaml.safe_load(args.config.read_text()) or {}
    source = torch.load(args.source, map_location="cpu", weights_only=False)
    torch.manual_seed(9917)
    model = build_pointcloud_model(public_config, n_fields=5, device="cpu")
    resolved = resolve_checkpoint_state(source, model=model)
    state = {key: value.detach().cpu() for key, value in resolved.state_dict.items()}
    state_sha256 = tensor_digest(state)
    if state_sha256 != EXPECTED_STATE_SHA256:
        raise RuntimeError(
            f"Resolved state does not match RC1: {state_sha256} != {EXPECTED_STATE_SHA256}"
        )
    model.load_state_dict(state, strict=True)

    portable_config = dict(public_config)
    portable_config["data"] = "Dataset/Merged_COTU0U1P.h5"
    portable_config["dataset_stats_path"] = (
        "ReleaseArtifacts/GL_rbf_CQ_rc1/dataset_stats.pt"
    )
    portable_config["save_dir"] = "runs/GL_rbf_CQ"
    payload = {
        "format_version": 1,
        "artifact_kind": "phycoflow_pointcloud_inference",
        "public_model_name": "GL_rbf_CQ",
        "internal_backbone": "GL_rbf_ENH_CQ",
        "source_tag": "gl-rbf-cq-v0.9.0-rc1",
        "source_commit": "1b9a6d47f6c248364df6ba54155b5eac3d5e6e67",
        "source_checkpoint_sha256": file_digest(args.source),
        "resolved_weights": resolved.selection,
        "resolved_state_sha256": state_sha256,
        "model": state,
        "model_ema_enabled": False,
        "model_ema_eval": False,
        "epoch": int(source["epoch"]),
        "global_step": int(source["global_step"]),
        "mean": source["mean"].detach().cpu(),
        "std": source["std"].detach().cpu(),
        "field_names": tuple(source["field_names"]),
        "config": portable_config,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    reloaded = torch.load(args.output, map_location="cpu", weights_only=False)
    if tensor_digest(reloaded["model"]) != EXPECTED_STATE_SHA256:
        raise RuntimeError("Portable checkpoint state changed during serialization.")
    model.load_state_dict(reloaded["model"], strict=True)
    checkpoint_sha256 = file_digest(args.output)
    manifest = {
        "format_version": 1,
        "artifact": str(args.output.resolve().relative_to(ROOT)),
        "sha256": checkpoint_sha256,
        "size_bytes": args.output.stat().st_size,
        "resolved_state_sha256": state_sha256,
        "source_tag": payload["source_tag"],
        "source_commit": payload["source_commit"],
        "source_checkpoint_sha256": payload["source_checkpoint_sha256"],
        "resolved_weights": resolved.selection,
        "epoch": payload["epoch"],
        "global_step": payload["global_step"],
        "field_names": list(payload["field_names"]),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
