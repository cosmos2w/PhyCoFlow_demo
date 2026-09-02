"""Audit Cond_T ablation YAMLs against the archived adopted A0 config."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "Save_TrainedModel/_TrainedModels/DMF_Gen/Cond_T/run_config.yaml"
CONFIG_DIR = ROOT / "Save_config/ablation_condT"
REFERENCE_SHA256 = "e24de2d9b8909daa520b7b390a4a62396832ceb6e6810cd67f90673d9cf2f6c1"

CONFIGS = {
    "A1": "config_A1_deterministic_condT.yaml",
    "A2": "config_A2_no_sensor_global_feedback_condT.yaml",
    "A3": "config_A3_no_local_query_conditional_condT.yaml",
    "A4": "config_A4_iid_prior_condT.yaml",
    "A5": "config_A5_local_sensor_tokens_only_condT.yaml",
}

# User-selected reduced-budget group.  The 10,000-step cosine horizon remains
# fixed to A0; only the stopping epoch is shortened and explicitly reported.
APPROVED_PROTECTED_OVERRIDES = {
    "A2": {"epochs": 6000},
    "A3": {"epochs": 6000},
    "A5": {"epochs": 6000},
}

PROTECTED_KEYS = (
    "data",
    "seed",
    "epochs",
    "batch_size",
    "lr",
    "weight_decay",
    "train_ratio",
    "time_stride",
    "backbone",
    "hidden_dim",
    "cond_dim",
    "field_embed_dim",
    "latent_dim",
    "num_latents",
    "num_heads",
    "num_latent_blocks",
    "ff_mult",
    "summary_type",
    "gather_mode",
    "gather_topk",
    "gather_query_chunk_size",
    "learnable_rbf_sigma",
    "rbf_sigma",
    "neighbor_backend",
    "USE_FOURIER_PE",
    "fourier_pe_num_bands",
    "fourier_pe_max_freq",
    "sensor_coord_encoding",
    "latent_sensor_reinject",
    "latent_reinject_every",
    "query_latent_readout",
    "query_readout_type",
    "query_readout_scale_init",
    "enhanced_head_norm",
    "glres_scale_init",
    "n_query_points",
    "query_sampling",
    "rff_features",
    "rff_lengthscale",
    "cond_fields",
    "n_obs_min_list",
    "n_obs_max_list",
    "vis_cond_fields",
    "vis_n_obs_list",
)

ALLOWED_OPERATIONAL_KEYS = {
    "Demo_Num",
    "device_ids",
    "save_dir",
    "dataset_stats_path",
    "scheduler_t_max",
    "training_mode",
    "initialization",
    "data_path_mode",
    "model_name",
    "coord_dim",
    "condition_attention_execution",
    "sensor_attention_padding_mode",
    "sensor_attention_buckets",
    "ablation",
    "epochs",
}


def _load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}.")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit() -> tuple[str, list[str]]:
    reference = _load(REFERENCE)
    failures: list[str] = []
    if _sha256(REFERENCE) != REFERENCE_SHA256:
        failures.append("archived A0 run_config.yaml hash changed")
    lines = [
        "# Cond_T ablation config-diff audit",
        "",
        f"A0: `{REFERENCE.relative_to(ROOT)}`",
        f"A0 SHA-256: `{_sha256(REFERENCE)}`",
        "",
        "Operational additions (not scientific changes): unique Demo/output paths, device 1, "
        "the archived dataset-stat path, explicit 10,000-epoch scheduler horizon, explicit "
        "legacy execution defaults, model identity, and ablation provenance.",
        "A2/A3/A5 additionally use the documented user-selected 6,000-epoch stop while "
        "retaining the 10,000-epoch scheduler horizon.",
        "",
    ]
    for ablation_id, filename in CONFIGS.items():
        config = _load(CONFIG_DIR / filename)
        meta = config.get("ablation", {})
        if meta.get("id") != ablation_id or not meta.get("enabled"):
            failures.append(f"{ablation_id}: invalid ablation metadata")
        for key in PROTECTED_KEYS:
            expected = APPROVED_PROTECTED_OVERRIDES.get(ablation_id, {}).get(
                key, reference.get(key)
            )
            if config.get(key) != expected:
                failures.append(
                    f"{ablation_id}: protected {key}: expected={expected!r}, "
                    f"config={config.get(key)!r}, A0={reference.get(key)!r}"
                )
        expected_prior = "iid" if ablation_id == "A4" else "rff"
        if config.get("prior") != expected_prior:
            failures.append(
                f"{ablation_id}: prior must be {expected_prior!r}, got {config.get('prior')!r}"
            )
        differing = []
        for key in sorted(set(reference) | set(config)):
            if reference.get(key) != config.get(key):
                if key == "prior":
                    classification = "scientific"
                elif key in APPROVED_PROTECTED_OVERRIDES.get(ablation_id, {}):
                    classification = "approved matched-budget override"
                else:
                    classification = "operational/provenance"
                differing.append(
                    f"- `{key}` ({classification}): `{reference.get(key)!r}` → `{config.get(key)!r}`"
                )
                if key != "prior" and key not in ALLOWED_OPERATIONAL_KEYS:
                    failures.append(f"{ablation_id}: unexpected differing key {key!r}")
        lines.extend(
            [
                f"## {ablation_id}",
                "",
                f"Expected intervention: `{meta.get('variant')}`.",
                "",
                *differing,
                "",
            ]
        )
    lines.extend(
        [
            "## Result",
            "",
            "PASS: protected fields match A0 except documented approved budget overrides."
            if not failures
            else "FAIL:\n\n" + "\n".join(f"- {item}" for item in failures),
            "",
        ]
    )
    return "\n".join(lines), failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG_DIR / "config_diff_audit.md",
    )
    args = parser.parse_args()
    report, failures = audit()
    args.output.write_text(report)
    print(f"Wrote {args.output}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
