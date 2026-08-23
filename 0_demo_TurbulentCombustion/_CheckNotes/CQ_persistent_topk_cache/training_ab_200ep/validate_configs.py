#!/usr/bin/env python3
"""Validate the pinned pre-cache/current CQ-LR 200-epoch A/B protocol."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
GIT_ROOT = REPO.parent
OLD = ROOT / "CQ_LR_no_persistent_200ep.yaml"
NEW = ROOT / "CQ_LR_persistent_topk_200ep.yaml"
OLD_COMMIT = "01d284767af9cbbf6b2e185b2ea52c50545ca607"
NEW_COMMIT = "3f3eefbe5ddeb2d530318bf7686d03b61c051ff4"
PROJECT_PREFIX = "0_demo_TurbulentCombustion"
MILESTONES = [1, 20, 40, 60, 100, 150, 200]


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def blob(commit: str, relative: str) -> bytes:
    return subprocess.check_output(
        ["git", "show", f"{commit}:{PROJECT_PREFIX}/{relative}"],
        cwd=GIT_ROOT,
    )


def main() -> None:
    old = load(OLD)
    new = load(NEW)
    differences = {
        key for key in set(old) | set(new) if old.get(key) != new.get(key)
    }
    assert differences == {"Demo_Num", "save_dir"}, differences

    for label, config in (("without persistent Top-K", old), ("with persistent Top-K", new)):
        assert config["backbone"] == "GL_rbf_ENH_CQ"
        assert config["seed"] == 42
        assert config["epochs"] == config["scheduler_t_max"] == 200
        assert config["batch_size"] == 128
        assert config["n_query_points"] == 4096
        assert config["train_query_microbatch_size"] is None
        assert config["gather_mode"] == "topk_rbf_glres"
        assert config["gather_topk"] == 32
        assert config["cq_query_dim"] == 128
        assert config["cq_readout_mode"] == "lowrank"
        assert config["cq_readout_rank"] == 64
        assert config["cq_readout_heads"] == 4
        assert config["checkpoint_epochs"] == MILESTONES
        assert config["training_history_plot_every_n_epochs"] == 5
        print(f"{label}: config valid")

    # The optimization is inference-only. Guard the paired training path against
    # accidental source drift between the pinned revisions.
    identical_training_files = [
        "src/train_pointcloud_ffm.py",
        "src/helpers.py",
        "src/helpers_baseline.py",
        "src/pointcloud_data_path.py",
    ]
    digests = {}
    for relative in identical_training_files:
        old_data = blob(OLD_COMMIT, relative)
        new_data = blob(NEW_COMMIT, relative)
        assert old_data == new_data, f"Training-path drift: {relative}"
        digests[relative] = hashlib.sha256(old_data).hexdigest()
    print("Pinned revisions have byte-identical training/data paths:")
    for relative, digest in digests.items():
        print(f"  {relative}: {digest}")


if __name__ == "__main__":
    main()
