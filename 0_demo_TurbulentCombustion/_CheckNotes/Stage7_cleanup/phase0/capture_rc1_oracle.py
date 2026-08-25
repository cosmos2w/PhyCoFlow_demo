#!/usr/bin/env python3
"""Capture deterministic RC1 model fingerprints before cleanup refactoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm  # noqa: E402
from train_pointcloud_ffm import checkpoint_model_state  # noqa: E402


CANDIDATES = {
    "GL_rbf_CQ": (
        ROOT / "ReleaseArtifacts/GL_rbf_CQ_rc1/run_config_training.yaml",
        ROOT / "ReleaseArtifacts/GL_rbf_CQ_rc1/GL_rbf_CQ_v0.9.0-rc1_e1000_research.pt",
    ),
    "GL_rbf_CQ-fast": (
        ROOT / "_CheckNotes/Stage6_clean_ab/runs/CQ_LR_1K_B128_DemoN9511_20260821_235104/run_config.yaml",
        ROOT / "_CheckNotes/Stage6_clean_ab/runs/CQ_LR_1K_B128_DemoN9511_20260821_235104/best.pt",
    ),
    "GL_rbf_ENH": (
        ROOT / "_CheckNotes/Stage6_clean_ab/runs/F0_ENH_1K_B128_DemoN9510_20260821_235104/run_config.yaml",
        ROOT / "_CheckNotes/Stage6_clean_ab/runs/F0_ENH_1K_B128_DemoN9510_20260821_235104/best.pt",
    ),
}


def tensor_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def fixed_inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260823)
    coords = torch.rand(1, 13, 3, generator=generator)
    obs_indices = torch.tensor([[0, 2, 3, 5, 7, 8, 10, 11, 12]])
    return {
        "t": torch.tensor([0.375]),
        "x_t": torch.randn(1, 13, 5, generator=generator),
        "coords": coords,
        "obs_coords": coords[:, obs_indices[0]],
        "obs_values": torch.randn(1, 9, 1, generator=generator),
        "obs_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.float32),
        "obs_field_ids": torch.tensor([[0, 1, 2, 3, 4, 1, 3, 0, 4]]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.set_num_threads(1)
    values = fixed_inputs()
    result = {
        "oracle_tag": "gl-rbf-cq-v0.9.0-rc1",
        "oracle_commit": "1b9a6d47f6c248364df6ba54155b5eac3d5e6e67",
        "capture_seed": 20260823,
        "device": "cpu",
        "neighbor_backend_override": "torch",
        "candidates": {},
    }
    for label, (config_path, checkpoint_path) in CANDIDATES.items():
        config = yaml.safe_load(config_path.read_text()) or {}
        config["neighbor_backend"] = "torch"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        torch.manual_seed(9917)
        model = build_gl_rbf_ffm(config, n_fields=5, device=torch.device("cpu"))
        state = checkpoint_model_state(checkpoint, model=model)
        model.load_state_dict(state, strict=True)
        model.eval()
        with torch.no_grad():
            output = model.model(**values)
        result["candidates"][label] = {
            "config": str(config_path.relative_to(ROOT)),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "selected_weights": (
                "ema_trainable_plus_live_frozen"
                if label == "GL_rbf_CQ" else "live"
            ),
            "state_key_count": len(state),
            "state_sha256": tensor_digest(dict(state)),
            "state_keys": sorted(state),
            "output_shape": list(output.shape),
            "output_sha256": tensor_digest({"output": output}),
            "output": output.detach().cpu().tolist(),
            "output_sum": float(output.sum()),
            "output_l2": float(output.square().sum().sqrt()),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
