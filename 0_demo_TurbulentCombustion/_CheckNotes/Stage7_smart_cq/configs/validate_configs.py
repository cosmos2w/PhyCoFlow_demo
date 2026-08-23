#!/usr/bin/env python3
"""Validate the two Stage-7 screens against the clean CQ-LR protocol."""

from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "_CheckNotes/Stage6_clean_ab/CQ_LR_1000ep_b128.yaml"
CONFIGS = {
    "S7-A": Path(__file__).with_name("S7_A_Cond128_200ep_b128.yaml"),
    "S7-B": Path(__file__).with_name("S7_B_All256_200ep_b128.yaml"),
}
ALLOWED_COMMON = {
    "Demo_Num", "data", "dataset_stats_path", "save_dir", "epochs", "save_every",
    "checkpoint_epochs", "cq_fusion_mode", "model_ema_enabled", "model_ema_decay",
    "model_ema_eval", "cq_time_conditioning", "cq_time_embed_dim",
    "cq_time_max_period", "cq_time_film_zero_init", "cq_measurement_support_mode",
    "cq_measurement_support_normalize", "train_query_microbatch_size",
}


def load(path: Path):
    return yaml.safe_load(path.read_text())


def main() -> int:
    base = load(BASE)
    loaded = {name: load(path) for name, path in CONFIGS.items()}
    errors = []
    for name, config in loaded.items():
        allowed = set(ALLOWED_COMMON)
        if name == "S7-B":
            allowed.add("latent_dim")
        differing = {
            key for key in set(base) | set(config)
            if base.get(key, "<missing>") != config.get(key, "<missing>")
        }
        unexpected = sorted(differing - allowed)
        if unexpected:
            errors.append(f"{name}: unexpected protocol differences: {unexpected}")
        expected = {
            "backbone": "GL_rbf_ENH_CQ", "cq_query_dim": 128,
            "cq_readout_mode": "lowrank", "cq_readout_rank": 64,
            "cq_readout_heads": 4, "cq_fusion_mode": "additive",
            "gather_topk": 32, "n_query_points": 4096,
            "batch_size": 128, "seed": 42, "scheduler_t_max": 1000,
            "num_latents": 128, "num_latent_blocks": 4,
            "model_ema_enabled": True, "model_ema_decay": 0.999,
            "model_ema_eval": True, "cq_time_conditioning": "sinusoidal_film",
            "cq_measurement_support_mode": "rbf_value_support",
            "train_query_microbatch_size": 2048,
        }
        expected["latent_dim"] = 128 if name == "S7-A" else 256
        for key, value in expected.items():
            if config.get(key) != value:
                errors.append(f"{name}: {key}={config.get(key)!r}, expected {value!r}")
    pair_differences = {
        key for key in set(loaded["S7-A"]) | set(loaded["S7-B"])
        if loaded["S7-A"].get(key) != loaded["S7-B"].get(key)
    }
    expected_pair = {"Demo_Num", "save_dir", "latent_dim"}
    if pair_differences != expected_pair:
        errors.append(f"S7-A/B differ by {sorted(pair_differences)}, expected {sorted(expected_pair)}")
    if errors:
        print("\n".join(errors))
        return 1
    print("S7-A: valid")
    print("S7-B: valid")
    print("Pair differs only in run identity and latent_dim; all query/protocol keys are matched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
