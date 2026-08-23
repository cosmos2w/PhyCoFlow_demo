#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[1]
PRIMARY = PACKAGE / "CQ_Balanced_192_Full_200ep.yaml"
FALLBACK = PACKAGE / "CQ_Balanced_224_Full_200ep.yaml"
REFERENCE = ROOT / "_CheckNotes/Stage6_clean_ab/CQ_LR_1000ep_b128.yaml"


def load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def main() -> None:
    primary = load(PRIMARY)
    fallback = load(FALLBACK)
    reference = load(REFERENCE)
    allowed_primary_changes = {
        "Demo_Num", "save_dir", "epochs", "save_every", "checkpoint_epochs",
        "cq_query_dim", "cq_readout_mode", "cq_fusion_mode",
    }
    for key, value in reference.items():
        if key not in allowed_primary_changes:
            assert primary.get(key) == value, (key, value, primary.get(key))
    assert primary["backbone"] == "GL_rbf_ENH_CQ"
    assert primary["cq_query_dim"] == 192
    assert primary["cq_readout_mode"] == "full"
    assert primary["cq_fusion_mode"] == "structured_concat"
    assert primary["epochs"] == 200
    assert primary["scheduler_t_max"] == 1000
    assert primary["batch_size"] == 128
    assert primary["n_query_points"] == 4096
    assert primary["train_query_microbatch_size"] is None
    assert primary["gather_topk"] == 32
    assert primary["neighbor_backend"] == "keops"
    assert primary["reconstruction_cache_level"] == "static_features"
    assert primary["checkpoint_epochs"] == [1, 20, 40, 60, 100, 150, 200]

    allowed_fallback_changes = {"Demo_Num", "save_dir", "cq_query_dim"}
    for key, value in primary.items():
        if key not in allowed_fallback_changes:
            assert fallback.get(key) == value, (key, value, fallback.get(key))
    assert fallback["cq_query_dim"] == 224
    print("PASS: clean CQ-Balanced-192 and sole 224 fallback configs validated.")


if __name__ == "__main__":
    main()
