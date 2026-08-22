#!/usr/bin/env python3
"""Validate the clean F0-ENH versus CQ-LR training comparison."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
F0_REFERENCE = ROOT.parent / "Stage6_formal_baseline/F0_frozen_current.yaml"
BASELINE = ROOT / "F0_ENH_1000ep_b128.yaml"
NEW = ROOT / "CQ_LR_1000ep_b128.yaml"
MILESTONES = [1, 20, 40, 60, 100, 200, 400, 600, 800, 1000]


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def main() -> None:
    reference = load(F0_REFERENCE)
    baseline = load(BASELINE)
    new = load(NEW)

    pair_differences = {
        key for key in set(baseline) | set(new)
        if baseline.get(key) != new.get(key)
    }
    expected_pair_differences = {"Demo_Num", "save_dir", "backbone"}
    assert pair_differences == expected_pair_differences, pair_differences

    for label, config, backbone in (
        ("F0-ENH", baseline, "GL_rbf_ENH"),
        ("CQ-LR", new, "GL_rbf_ENH_CQ"),
    ):
        assert config["backbone"] == backbone
        assert config["epochs"] == 1000
        assert config["scheduler_t_max"] == 1000
        assert config["seed"] == reference["seed"] == 42
        assert config["batch_size"] == 128
        assert config["n_query_points"] == reference["n_query_points"] == 4096
        assert config["train_query_microbatch_size"] is None
        assert config["save_every"] == 1000
        assert config["checkpoint_epochs"] == MILESTONES
        assert config["gather_mode"] == "topk_rbf_glres"
        assert config["gather_topk"] == 32
        assert config["cq_query_dim"] == 128
        assert config["cq_readout_mode"] == "lowrank"
        print(f"{label}: valid")

    allowed_reference_differences = {
        "Demo_Num", "save_dir", "backbone", "epochs", "scheduler_t_max",
        "batch_size", "save_every",
        "checkpoint_epochs", "cq_query_dim", "cq_readout_mode",
        "cq_readout_rank", "cq_readout_heads", "cq_global_scale_init",
        "cq_local_scale_init", "cq_readout_scale_init",
    }
    for path, config in ((BASELINE, baseline), (NEW, new)):
        differences = {
            key for key in set(reference) | set(config)
            if reference.get(key) != config.get(key)
        }
        unexpected = differences - allowed_reference_differences
        assert not unexpected, f"{path.name} changes F0 protocol: {sorted(unexpected)}"

    print(
        "Pair differs only in run identity and backbone; both use the same "
        "extended 1,000-epoch, batch-128 protocol."
    )


if __name__ == "__main__":
    main()
