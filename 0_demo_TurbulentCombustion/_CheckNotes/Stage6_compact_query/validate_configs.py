#!/usr/bin/env python3
"""Validate Stage-6 CQ screens against the immutable F0 protocol."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT.parent / "Stage6_formal_baseline" / "F0_frozen_current.yaml"
CANDIDATES = [
    (ROOT / "CQ_full_60ep.yaml", 128, "full"),
    (ROOT / "CQ_lr_60ep.yaml", 128, "lowrank"),
    (ROOT / "CQ_rescue160_60ep.yaml", 160, "full"),
]
ALLOWED_DIFFERENCES = {
    "backbone", "epochs", "scheduler_t_max", "Demo_Num", "save_dir", "checkpoint_epochs",
    "cq_query_dim", "cq_readout_mode", "cq_readout_rank", "cq_readout_heads",
    "cq_global_scale_init", "cq_local_scale_init", "cq_readout_scale_init",
}


def load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} does not contain a YAML mapping.")
    return value


def main() -> None:
    reference = load(REFERENCE)
    assert reference["backbone"] == "GL_rbf_ENH"
    assert reference["n_query_points"] == 4096
    assert reference["train_query_microbatch_size"] is None
    assert reference["batch_size"] == 64
    assert reference["seed"] == 42

    for path, query_dim, readout_mode in CANDIDATES:
        config = load(path)
        assert config["backbone"] == "GL_rbf_ENH_CQ"
        assert config["epochs"] == 60
        assert config["scheduler_t_max"] == reference["epochs"] == 200
        assert config["n_query_points"] == 4096
        assert config["train_query_microbatch_size"] is None
        assert config["batch_size"] == 64
        assert config["seed"] == 42
        assert config["cq_query_dim"] == query_dim
        assert config["cq_readout_mode"] == readout_mode
        assert config["cq_readout_rank"] == 64
        assert config["cq_readout_heads"] == 4
        assert config["gather_mode"] == "topk_rbf_glres"
        assert config["gather_topk"] == 32
        differing = {
            key for key in set(reference) | set(config)
            if reference.get(key) != config.get(key)
        }
        unexpected = differing - ALLOWED_DIFFERENCES
        assert not unexpected, f"{path.name} changes F0 invariants: {sorted(unexpected)}"
        print(f"{path.name}: valid ({config['cq_readout_mode']})")


if __name__ == "__main__":
    main()
