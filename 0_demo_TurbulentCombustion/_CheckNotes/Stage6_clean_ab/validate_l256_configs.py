#!/usr/bin/env python3
"""Validate the latent-256 clean A/B configs and optional old-schedule control."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
F0 = ROOT / "F0_ENH_L256_1000ep_b128.yaml"
CQ = ROOT / "CQ_LR_L256_1000ep_b128.yaml"
OLD_SCHEDULE = ROOT / "F0_ENH_L256_OLD_SCHED_1000ep_b128.yaml"
MILESTONES = [1, 20, 40, 60, 100, 200, 400, 600, 800, 1000]


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML mapping: {path}")
    return value


def main() -> None:
    f0 = load(F0)
    cq = load(CQ)
    old_schedule = load(OLD_SCHEDULE)

    pair_differences = {
        key for key in set(f0) | set(cq)
        if f0.get(key) != cq.get(key)
    }
    expected_pair_differences = {
        "Demo_Num", "save_dir", "backbone", "device_ids"
    }
    assert pair_differences == expected_pair_differences, pair_differences

    for label, config, backbone, demo_num, device_ids in (
        ("F0-L256", f0, "GL_rbf_ENH", 9560, [1]),
        ("CQ-LR-L256", cq, "GL_rbf_ENH_CQ", 9561, [0]),
        ("F0-L256-old-schedule", old_schedule, "GL_rbf_ENH", 9562, [2]),
    ):
        assert config["Demo_Num"] == demo_num
        assert config["device_ids"] == device_ids
        assert config["backbone"] == backbone
        assert config["latent_dim"] == 256
        assert config["seed"] == 42
        assert config["epochs"] == 1000
        assert config["batch_size"] == 128
        assert config["n_query_points"] == 4096
        assert config["train_query_microbatch_size"] is None
        assert config["gather_mode"] == "topk_rbf_glres"
        assert config["gather_topk"] == 32
        assert config["neighbor_backend"] == "keops"
        assert config["checkpoint_epochs"] == MILESTONES
        assert config["cq_query_dim"] == 128
        assert config["cq_readout_mode"] == "lowrank"
        assert config["cq_readout_rank"] == 64
        print(f"{label}: valid")

    assert f0["scheduler_t_max"] == cq["scheduler_t_max"] == 1000
    assert old_schedule["scheduler_t_max"] == 10000

    schedule_differences = {
        key for key in set(f0) | set(old_schedule)
        if f0.get(key) != old_schedule.get(key)
    }
    assert schedule_differences == {
        "Demo_Num", "save_dir", "scheduler_t_max", "device_ids"
    }, schedule_differences

    print(
        "Clean pair differs only in run identity, GPU placement, and backbone; "
        "optional control differs from F0 only in run identity, GPU placement, "
        "and scheduler horizon."
    )


if __name__ == "__main__":
    main()
