#!/usr/bin/env python3
"""Validate the frozen F0/F1 protocol without starting training."""

from __future__ import annotations

from pathlib import Path

import yaml


PACKAGE = Path(__file__).resolve().parent
REPO = PACKAGE.parents[1]
F0_PATH = PACKAGE / "F0_frozen_current.yaml"
F1_PATH = PACKAGE / "F1_more_supervision.yaml"
ACTIVE_PATH = REPO / "Save_config/config_pointcloud_ffm.yaml"

ALLOWED_PAIR_DIFFERENCES = {
    "Demo_Num",
    "save_dir",
    "n_query_points",
    "train_query_microbatch_size",
}

ACTIVE_MODEL_KEYS = {
    "backbone",
    "hidden_dim",
    "cond_dim",
    "field_embed_dim",
    "sigma_min",
    "rbf_sigma",
    "USE_FOURIER_PE",
    "fourier_pe_num_bands",
    "fourier_pe_max_freq",
    "latent_dim",
    "num_latents",
    "num_heads",
    "num_latent_blocks",
    "ff_mult",
    "attn_dropout",
    "mlp_dropout",
    "decode_chunk_size",
    "share_query_proj",
    "summary_type",
    "gather_mode",
    "gather_topk",
    "sensor_local_topk",
    "gather_query_chunk_size",
    "learnable_rbf_sigma",
    "neighbor_backend",
    "sensor_local_dropout",
    "sensor_coord_encoding",
    "latent_sensor_reinject",
    "latent_reinject_every",
    "query_latent_readout",
    "query_readout_type",
    "query_readout_scale_init",
    "enhanced_head_norm",
    "glres_scale_init",
    "query_sampling",
    "query_sample_near_ratio",
    "query_sample_far_ratio",
    "query_sample_sigma_ratio",
    "prior",
    "rff_features",
    "rff_lengthscale",
    "cond_fields",
    "n_obs_min_list",
    "n_obs_max_list",
    "reconstruction_execution_mode",
    "reconstruction_query_chunk_size",
    "reconstruction_cache_level",
    "ode_solver",
    "benchmark_n_steps",
}


def load(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise AssertionError(f"{path} did not contain a YAML mapping")
    return value


def main() -> None:
    f0 = load(F0_PATH)
    f1 = load(F1_PATH)
    active = load(ACTIVE_PATH)

    assert set(f0) == set(f1), "F0/F1 key sets differ"
    actual_differences = {key for key in f0 if f0[key] != f1[key]}
    assert actual_differences == ALLOWED_PAIR_DIFFERENCES, (
        f"unexpected F0/F1 differences: {sorted(actual_differences)}"
    )

    expected_common = {
        "seed": 42,
        "epochs": 200,
        "batch_size": 96,
        "backbone": "GL_rbf_ENH",
        "gather_mode": "topk_rbf_glres",
        "gather_topk": 32,
        "reuse_condition_context_across_query_microbatches": True,
        "data_path_mode": "optimized",
        "field_read_mode": "legacy_full_snapshot",
        "field_normalization_mode": "selected_after_full_read",
        "gpu_transfer_mode": "selected_only",
        "reconstruction_execution_mode": "cached_streamed",
        "reconstruction_query_chunk_size": 8192,
        "reconstruction_cache_level": "static_features",
    }
    for key, expected in expected_common.items():
        assert f0[key] == expected and f1[key] == expected, key

    assert f0["n_query_points"] == 4096
    assert f0["train_query_microbatch_size"] is None
    assert f1["n_query_points"] == 16384
    assert f1["train_query_microbatch_size"] == 8192

    for key in ACTIVE_MODEL_KEYS:
        assert key in active, f"active config lacks {key}"
        assert f0[key] == active[key], f"F0 diverges from active {key}"
        assert f1[key] == active[key], f"F1 diverges from active {key}"

    stats_path = REPO / f0["dataset_stats_path"]
    data_path = REPO / f0["data"]
    assert stats_path.is_file(), f"missing normalization stats: {stats_path}"
    assert data_path.is_file(), f"missing dataset: {data_path}"

    print("F0/F1 formal configs validated")
    print("pair differences:", ", ".join(sorted(actual_differences)))
    print("architecture/RF/reconstruction settings match the active config")


if __name__ == "__main__":
    main()
