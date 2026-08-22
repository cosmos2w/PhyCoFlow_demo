#!/usr/bin/env python3
"""Apply the Stage-6 CQ-Full versus CQ-LR promotion and rescue rules."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
F0_ROOT = ROOT.parent / "Stage6_formal_baseline"


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required Stage-6 evidence is missing: {path}")
    return json.loads(path.read_text())


def epoch60_loss(result: dict) -> float:
    rows = [
        value for value in result["summary"].values()
        if int(value.get("epoch", -1)) == 60
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one epoch-60 fixed-manifest row, got {len(rows)}.")
    return float(rows[0]["mean_rf_loss"])


def reconstruction_means(result: list[dict]) -> dict[int, float]:
    return {
        int(row["nfe"]): float(row["mean_field_relative_l2"])
        for row in result
    }


def main() -> None:
    full_rf = epoch60_loss(load(
        ROOT / "screen_cq_full/evaluation/fixed_manifest/milestones.json"
    ))
    lr_rf = epoch60_loss(load(
        ROOT / "screen_cq_lr/evaluation/fixed_manifest/milestones.json"
    ))
    full_recon = reconstruction_means(load(
        ROOT / "screen_cq_full/evaluation/matched_reconstruction/epoch_0060/summary.json"
    ))
    lr_recon = reconstruction_means(load(
        ROOT / "screen_cq_lr/evaluation/matched_reconstruction/epoch_0060/summary.json"
    ))
    f0_fixed = load(F0_ROOT / "evaluation/fixed_manifest_best.json")
    f0_rf = next(
        float(value["mean_rf_loss"])
        for label, value in f0_fixed["summary"].items()
        if label.startswith("F0")
    )
    f0_recon = reconstruction_means(load(
        F0_ROOT / "evaluation/matched_reconstruction/F0_best/summary.json"
    ))

    rescue_rf_path = ROOT / "screen_cq_rescue160/evaluation/fixed_manifest/milestones.json"
    rescue_recon_path = (
        ROOT
        / "screen_cq_rescue160/evaluation/matched_reconstruction/epoch_0060/summary.json"
    )
    rescue_completed = rescue_rf_path.exists() and rescue_recon_path.exists()
    rescue_rf = epoch60_loss(load(rescue_rf_path)) if rescue_completed else None
    rescue_recon = (
        reconstruction_means(load(rescue_recon_path)) if rescue_completed else None
    )

    benchmark = load(ROOT / "benchmarks/cost_benchmark.json")
    scaling = {
        (row["label"], int(row["N_query"])): row
        for row in benchmark["scaling"]
        if row.get("status") == "ok"
    }
    full_ms = float(scaling[("CQ-Full", 65536)]["forward_ms"])
    lr_ms = float(scaling[("CQ-LR", 65536)]["forward_ms"])

    rf_relative = lr_rf / full_rf - 1.0
    recon_relative = {
        nfe: lr_recon[nfe] / full_recon[nfe] - 1.0 for nfe in sorted(full_recon)
    }
    recon_average_relative = (
        sum(lr_recon.values()) / sum(full_recon.values()) - 1.0
    )
    additional_speedup = full_ms / lr_ms - 1.0
    lr_passes = (
        rf_relative <= 0.03
        and recon_average_relative <= 0.03
        and additional_speedup >= 0.15
    )
    primary_selected = "CQ-LR" if lr_passes else "CQ-Full"
    relative_to_f0 = {
        "CQ-Full": {
            "fixed_rf": full_rf / f0_rf - 1.0,
            "reconstruction_average": (
                sum(full_recon.values()) / sum(f0_recon.values()) - 1.0
            ),
        },
        "CQ-LR": {
            "fixed_rf": lr_rf / f0_rf - 1.0,
            "reconstruction_average": (
                sum(lr_recon.values()) / sum(f0_recon.values()) - 1.0
            ),
        },
    }
    rescue_required = all(
        values["fixed_rf"] > 0.05 and values["reconstruction_average"] > 0.05
        for values in relative_to_f0.values()
    )
    rescue_relative_to_f0 = None
    if rescue_completed:
        rescue_relative_to_f0 = {
            "fixed_rf": rescue_rf / f0_rf - 1.0,
            "reconstruction_average": (
                sum(rescue_recon.values()) / sum(f0_recon.values()) - 1.0
            ),
        }
    rescue_failed = bool(
        rescue_completed
        and rescue_relative_to_f0["fixed_rf"] > 0.05
        and rescue_relative_to_f0["reconstruction_average"] > 0.05
    )

    if rescue_required and rescue_failed:
        status = "prepared_not_recommended_rescue_failed"
    elif rescue_required:
        status = "rescue_required"
    else:
        status = "primary_ready"

    result = {
        "selected_candidate": primary_selected,
        "formal_candidate_status": status,
        "rule": {
            "lr_rf_within_fraction": 0.03,
            "lr_reconstruction_within_fraction": 0.03,
            "lr_additional_speedup_at_least_fraction": 0.15,
        },
        "measurements": {
            "cq_full_epoch60_rf_loss": full_rf,
            "cq_lr_epoch60_rf_loss": lr_rf,
            "cq_lr_rf_relative_to_full": rf_relative,
            "cq_full_reconstruction_mean_by_nfe": full_recon,
            "cq_lr_reconstruction_mean_by_nfe": lr_recon,
            "cq_lr_reconstruction_average_relative_to_full": recon_average_relative,
            "cq_lr_reconstruction_relative_by_nfe": recon_relative,
            "cq_full_forward_ms_65536": full_ms,
            "cq_lr_forward_ms_65536": lr_ms,
            "cq_lr_additional_speedup_fraction": additional_speedup,
            "f0_best_fixed_manifest_rf_loss": f0_rf,
            "f0_best_reconstruction_mean_by_nfe": f0_recon,
            "primary_variants_relative_to_f0": relative_to_f0,
            "cq_rescue160_epoch60_rf_loss": rescue_rf,
            "cq_rescue160_reconstruction_mean_by_nfe": rescue_recon,
            "cq_rescue160_relative_to_f0": rescue_relative_to_f0,
        },
        "cq_lr_passes_all_criteria": lr_passes,
        "rescue_rule": {
            "threshold_fraction": 0.05,
            "requires_both_variants_worse_on_fixed_rf_and_reconstruction": True,
            "rescue_required": rescue_required,
            "rescue_completed": rescue_completed,
            "rescue_failed_materially": rescue_failed,
            "allowed_configuration": {
                "cq_query_dim": 160,
                "cq_readout_mode": "full",
            },
        },
    }

    selected = result["selected_candidate"]
    if not rescue_required or rescue_completed:
        source_name = "CQ_lr_60ep.yaml" if selected == "CQ-LR" else "CQ_full_60ep.yaml"
        formal = yaml.safe_load((ROOT / source_name).read_text())
        formal.update({
            "epochs": 200,
            "Demo_Num": 9411 if selected == "CQ-LR" else 9410,
            "save_dir": (
                "_CheckNotes/Stage6_compact_query/formal_candidate/runs/"
                + selected.replace("-", "_")
                + "_formal"
            ),
            "checkpoint_epochs": [200],
        })
        formal_path = ROOT / "formal_candidate/selected_200ep.yaml"
        formal_path.write_text(
            "# Prepared from the selected 60-epoch CQ screen; do not auto-launch.\n"
            + yaml.safe_dump(formal, sort_keys=False)
        )
        result["formal_config"] = str(formal_path)
        result["formal_launch_permitted"] = False
        result["replacement_recommendation"] = (
            "do_not_replace_f0" if rescue_failed else "await_formal_run"
        )

    output = ROOT / "formal_candidate/selection.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
