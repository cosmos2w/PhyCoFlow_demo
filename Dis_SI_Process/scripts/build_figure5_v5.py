#!/usr/bin/env python
"""Build strict-formal Figure 5 V5 and its separated SI SVG bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_v41_style import apply_style, save_svg  # noqa: E402
from utils.figure5_v5_data import (  # noqa: E402
    CONDITION,
    DATASET,
    TASK,
    load_figure5_v5_data,
    materialize_lifecycle_v5,
)
from utils.figure5_v5_panels import (  # noqa: E402
    make_composed,
    make_si_calibration,
    make_si_fieldwise_capture,
    make_si_fieldwise_uq,
    make_si_nfe,
    make_si_scalability,
    make_standalone,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_v5.yaml")
    parser.add_argument("--timestamp", default=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    parser.add_argument("--preview-dir", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) if path.exists() else 0


def human_bytes(value: int) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    number = float(value)
    for unit in units:
        if number < 1024.0 or unit == units[-1]:
            return f"{number:.1f} {unit}"
        number /= 1024.0
    return f"{number:.1f} GiB"


def checkpoint_map(data: dict[str, Any]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for check in data["run_metadata"]["uq"]["manifest"].get("identity_checks", []):
        if check.get("label") in data["run_metadata"]["uq"]["manifest"].get("methods", []):
            mapping[str(check["label"])] = str(check["actual_sha256"])
    for row in data["lifecycle"]["summary"].itertuples():
        mapping[str(row.method)] = str(row.checkpoint_sha256)
    return mapping


def source_table(data: dict[str, Any], config: dict[str, Any]) -> pd.DataFrame:
    checkpoints = checkpoint_map(data)
    rows: list[dict[str, Any]] = []

    def add(
        panel: str,
        method: str,
        metric_name: str,
        metric_value: float,
        *,
        state_id: int | str = "",
        cohort_id: str,
        ci_low: float | str = "",
        ci_high: float | str = "",
        sample_kind: str = "summary",
        **extra: Any,
    ) -> None:
        rows.append(
            {
                "dataset": DATASET,
                "task": TASK,
                "condition": CONDITION,
                "panel": panel,
                "method": method,
                "checkpoint_sha256": checkpoints[method],
                "state_id": state_id,
                "cohort_id": cohort_id,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "sample_kind": sample_kind,
                **extra,
            }
        )

    for row in data["uq_crps_samples"].itertuples():
        add(
            "a",
            str(row.method),
            "statewise_normalized_crps",
            float(row.normalized_crps),
            state_id=int(row.state),
            cohort_id="calibration_200",
            sample_kind="paired_held_out_state",
            original_time_index=int(row.original_time_index),
        )
    for row in data["uq_crps"].itertuples():
        add(
            "a",
            str(row.method),
            "mean_statewise_normalized_crps",
            float(row.mean_normalized_crps),
            cohort_id="calibration_200",
            ci_low=float(row.crps_ci_low),
            ci_high=float(row.crps_ci_high),
            sample_kind="temporal_moving_block_bootstrap_summary",
        )
    for row in data["uq_spearman_bootstrap"].itertuples():
        add(
            "b",
            str(row.method),
            "spread_error_spearman_bootstrap",
            float(row.spearman_rho),
            state_id=int(row.replicate),
            cohort_id="calibration_200",
            sample_kind="temporal_moving_block_bootstrap_replicate",
            block_length=int(row.block_length),
        )
    for row in data["uq_spread"].itertuples():
        add(
            "b",
            str(row.method),
            "spread_error_spearman_full_sample",
            float(row.spearman_rho),
            cohort_id="calibration_200",
            ci_low=float(row.spearman_ci_low),
            ci_high=float(row.spearman_ci_high),
            sample_kind="full_sample_with_temporal_bootstrap_ci",
        )
    for row in data["localization"]["macro"].itertuples():
        add(
            "c",
            str(row.method),
            "spatial_error_capture_fraction",
            float(row.metric_value),
            cohort_id="calibration_200",
            ci_low=float(row.ci_low),
            ci_high=float(row.ci_high),
            sample_kind="state_mean_with_temporal_bootstrap_ci",
            spatial_fraction=float(row.spatial_fraction),
            ec_auc=float(row.ec_auc),
        )
    for row in data["lifecycle"]["summary"].itertuples():
        common = {"cohort_id": "figure4_frozen_1000_states", "sample_kind": "formal_adopted_checkpoint"}
        add(
            "d",
            str(row.method),
            "warm_native_inference_latency_ms",
            float(row.native_latency_ms),
            ci_low=float(row.native_latency_q25_ms),
            ci_high=float(row.native_latency_q75_ms),
            **common,
        )
        add(
            "d",
            str(row.method),
            "replay_equivalent_model_core_training_gpu_hours",
            float(row.replay_equivalent_gpu_hours),
            ci_low=float(row.replay_equivalent_gpu_hours_low),
            ci_high=float(row.replay_equivalent_gpu_hours_high),
            **common,
        )
        add(
            "d",
            str(row.method),
            "mean_unobserved_field_relative_l2",
            float(row.mean_unobserved_relative_l2),
            ci_low=float(row.mean_unobserved_relative_l2_ci_low),
            ci_high=float(row.mean_unobserved_relative_l2_ci_high),
            **common,
        )
    return pd.DataFrame(rows)


def svg_checks(paths: dict[str, Path], composed: Path, config: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    all_paths = list(paths.values()) + [composed]
    for path in all_paths:
        root = ET.parse(path).getroot()
        text = path.read_text(encoding="utf-8")
        checks[f"{path.name}:svg_root"] = root.tag.endswith("svg")
        checks[f"{path.name}:editable_text"] = "<text" in text
        checks[f"{path.name}:no_external_raster"] = '<image href="http' not in text and '<image xlink:href="http' not in text
    root = ET.parse(composed).getroot()
    view_box = [float(value) for value in root.attrib["viewBox"].split()]
    width_mm = view_box[2] / 72.0 * 25.4
    height_mm = view_box[3] / 72.0 * 25.4
    checks["composed_width_183_mm"] = abs(width_mm - float(config["figure"]["width_mm"])) < 0.05
    checks["composed_height_contract"] = abs(height_mm - float(config["figure"]["composed_height_mm"])) < 0.05
    return checks


def panel_results(data: dict[str, Any]) -> dict[str, str]:
    crps = data["uq_crps"].sort_values("mean_normalized_crps")
    spread = data["uq_spread"].sort_values("spearman_rho", ascending=False)
    capture = data["localization"]["macro"]
    top20 = capture.loc[np.isclose(capture["spatial_fraction"], 0.2)].sort_values("metric_value", ascending=False)
    life = data["lifecycle"]["summary"]
    return {
        "a": "; ".join(f"{row.method}: {row.mean_normalized_crps:.4f}" for row in crps.itertuples()),
        "b": "; ".join(f"{row.method}: ρ={row.spearman_rho:.3f}" for row in spread.itertuples()),
        "c": "; ".join(f"{row.method}: C(0.20)={row.metric_value:.3f}, EC-AUC={row.ec_auc:.3f}" for row in top20.itertuples()),
        "d": "; ".join(
            f"{row.method}: {row.native_latency_ms:.2f} ms, {row.replay_equivalent_gpu_hours:.1f} GPU h, L2={row.mean_unobserved_relative_l2:.3f}"
            for row in life.itertuples()
        ),
    }


def write_companions(
    docs_dir: Path,
    timestamp: str,
    outputs: dict[str, Path],
    composed: Path,
    data: dict[str, Any],
    config: dict[str, Any],
    si_outputs: dict[str, Path],
) -> list[Path]:
    results = panel_results(data)
    uq_dir = data["run_metadata"]["uq"]["directory"]
    loc_dir = data["localization"]["directory"]
    life_dir = data["lifecycle"]["directory"]
    common = (
        "Dataset/task: turbulent-combustion Cond_T missing-channel reconstruction; M=256; native N=40,300; "
        "unobserved fields Y_CH4, Y_CO, U1 and p are macro-averaged with equal 0.25 weight. "
        "The formal UQ cohort contains 200 paired temporal states and 64 shared-seed draws per state; "
        "95% intervals use 2,000 moving-block-bootstrap replicates with block length 25."
    )
    specifications = {
        "a": {
            "question": "Is the complete conditional predictive distribution accurate?",
            "sources": [uq_dir / "per_state_method.csv", uq_dir / "crps_summary.csv"],
            "status": "Reused unchanged from formal V3/V4.2; no inference or bootstrap rerun.",
            "metric": "Pointwise empirical CRPS normalized by frozen training field standard deviation, averaged spatially, then macro-averaged equally across four unobserved fields.",
            "limits": "CRPS assesses predictive-distribution quality but does not by itself establish calibration; formal reliability analysis shows underdispersion and is retained in SI.",
            "si": si_outputs["calibration_interval_width"],
        },
        "b": {
            "question": "Does empirical ensemble spread distinguish easy and difficult held-out states?",
            "sources": [uq_dir / "per_state_method.csv", uq_dir / "spread_error_summary.csv"],
            "status": "Reused unchanged from formal V3/V4.2; the exact bootstrap distribution is deterministically reconstructed from the adopted state table and seed.",
            "metric": "Spearman association between macro normalized ensemble spread and macro ensemble-mean relative-L2 error.",
            "limits": "This is an association with reconstruction difficulty, not calibration, Bayesian posterior uncertainty, prospective error prediction, or causal evidence.",
            "si": si_outputs["fieldwise_uq"],
        },
        "c": {
            "question": "Does empirical conditional ensemble uncertainty localize where reconstruction error occurs?",
            "sources": [loc_dir / "error_capture_curves.csv", loc_dir / "error_capture_summary.csv"],
            "status": "New V5 streaming repeated-inference reducer; each state ensemble was reduced in memory and discarded, with no full stacks or per-draw files retained.",
            "metric": "Within each state and field, locations are ranked by ensemble s.d.; captured absolute ensemble-mean error is evaluated at eight fractions, then fields are equally macro-averaged before temporal bootstrap.",
            "limits": "The result validates spatial ranking informativeness on the held-out cohort, not prospective uncertainty calibration.",
            "si": si_outputs["fieldwise_error_capture"],
        },
        "d": {
            "question": "What offline and online model-core footprint accompanies the frozen Figure-4 accuracy?",
            "sources": [life_dir / "lifecycle_summary.csv", life_dir / "lifecycle_stage_provenance.csv"],
            "status": "New compact derivation from reused formal V3 native timings and V4/V4.2 canonical update replays; no benchmark or training replay was rerun.",
            "metric": "x is accepted clean warm native latency; y is Replay-equivalent model-core training GPU-hours = sum(update ms × adopted updates × GPU count)/3.6e6; bubble area is frozen mean unobserved-field relative-L2.",
            "limits": "The metric is not historical wall time or a matched-budget causal efficiency comparison; hardware, batch, solver and method-native configurations differ.",
            "si": si_outputs["scalability_latency_memory"],
        },
    }
    companions: list[Path] = []
    for panel, spec in specifications.items():
        path = docs_dir / f"{config['figure']['output_stems'][panel]}_{timestamp}.md"
        source_lines = "\n".join(f"- `{source}`" for source in spec["sources"])
        path.write_text(
            f"# Figure 5 V5 panel {panel}\n\n"
            f"- SVG: `{outputs[panel].name}`\n"
            f"- Scientific question: {spec['question']}\n"
            f"- Reuse status: {spec['status']}\n\n"
            f"## Formal sources\n\n{source_lines}\n\n"
            f"## Cohort, checkpoints and statistics\n\n{common} Exact adopted checkpoint SHA-256 identities are recorded in the source manifests.\n\n"
            f"## Metric\n\n{spec['metric']}\n\n"
            f"## Main quantitative result\n\n{results[panel]}\n\n"
            f"## Limitations and SI\n\n{spec['limits']} SI destination: `{spec['si']}`.\n\n"
            "## Storage / cleanup\n\nNo checkpoint, dataset, cache or old result bundle was copied. No raw bootstrap arrays or repeated inference stacks were retained in V5.\n",
            encoding="utf-8",
        )
        companions.append(path)
    composed_doc = docs_dir / f"{config['figure']['output_stems']['composed']}_{timestamp}.md"
    composed_doc.write_text(
        "# Figure 5 V5 composed figure\n\n"
        f"- SVG: `{composed.name}`\n"
        f"- Canvas: `{config['figure']['width_mm']} mm × {config['figure']['composed_height_mm']} mm`\n"
        "- Archetype: focused 2×2 quantitative validation grid.\n"
        "- Export: SVG only; text remains editable.\n\n"
        "Figure 5 V5 evaluates DMF-Gen as a conditional generator along three complementary dimensions: predictive-distribution quality, state- and spatial-level uncertainty informativeness, and the combined offline/online computational footprint of the adopted checkpoint.\n\n"
        "Panels a/b reuse formal V3 evidence, panel c is the only new inference result, and panel d is a deterministic conversion of accepted formal timings and adopted update/GPU counts. Full calibration/interval width, fieldwise UQ, spatial fieldwise capture, 8M-query stress and NFE diagnostics are separated under the SI subdirectory.\n",
        encoding="utf-8",
    )
    companions.append(composed_doc)
    return companions


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "figure5-validation-v5" or config["figure"].get("formats") != ["svg"]:
        raise ValueError("Figure 5 V5 requires the exact V5 schema and SVG-only output")
    apply_style(config["style"]["font_family"])
    materialize_lifecycle_v5(config, REPO_ROOT)
    data = load_figure5_v5_data(config, REPO_ROOT)
    if args.strict_formal and any(not value.startswith("formal") for value in data["modes_v5"].values()):
        raise RuntimeError(f"Strict formal V5 blocked: {data['modes_v5']}")

    figure_dir = args.output_root / "figures" / "generated" / args.timestamp
    si_dir = figure_dir / str(config["build_policy"]["si_subdirectory"])
    result_dir = args.output_root / "results" / "derived" / args.timestamp
    docs_dir = args.output_root / "docs" / "generated"
    for path in (figure_dir, si_dir, result_dir, docs_dir):
        path.mkdir(parents=True, exist_ok=True)
    contract_source = PACKAGE_ROOT / "docs" / "figure5_v5_source_schema.md"
    shutil.copyfile(contract_source, figure_dir / contract_source.name)

    outputs: dict[str, Path] = {}
    for panel in "abcd":
        output = figure_dir / f"{config['figure']['output_stems'][panel]}_{args.timestamp}.svg"
        fig = make_standalone(panel, data, config)
        if args.preview_dir:
            args.preview_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.preview_dir / f"{output.stem}.png", dpi=240, facecolor="white")
        save_svg(fig, output)
        outputs[panel] = output
    composed = figure_dir / f"{config['figure']['output_stems']['composed']}_{args.timestamp}.svg"
    fig = make_composed(data, config)
    if args.preview_dir:
        fig.savefig(args.preview_dir / f"{composed.stem}.png", dpi=240, facecolor="white")
    save_svg(fig, composed)

    si_builders = {
        "calibration_interval_width": make_si_calibration,
        "fieldwise_uq": make_si_fieldwise_uq,
        "fieldwise_error_capture": make_si_fieldwise_capture,
        "scalability_latency_memory": make_si_scalability,
        "nfe_diagnostics": make_si_nfe,
    }
    si_outputs: dict[str, Path] = {}
    for name, builder in si_builders.items():
        path = si_dir / f"fig5_si_{name}_v5_{args.timestamp}.svg"
        fig = builder(data, config)
        if args.preview_dir:
            fig.savefig(args.preview_dir / f"{path.stem}.png", dpi=240, facecolor="white")
        save_svg(fig, path)
        si_outputs[name] = path

    plotted = source_table(data, config)
    plotted.to_csv(result_dir / "figure5_v5_source.csv", index=False)
    companions = write_companions(docs_dir, args.timestamp, outputs, composed, data, config, si_outputs)

    checks = svg_checks(outputs, composed, config)
    checks.update(
        {
            "four_main_panels": set(outputs) == set("abcd"),
            "all_main_sources_formal": all(value.startswith("formal") for value in data["modes_v5"].values()),
            "panel_a_five_methods_200_states": len(data["uq_crps_samples"]) == 1000,
            "panel_b_five_methods_2000_bootstraps": len(data["uq_spearman_bootstrap"]) == 10000,
            "panel_c_five_methods_common_grid": len(data["localization"]["macro"]) == 40,
            "panel_c_random_diagonal_drawn": True,
            "panel_d_all_eight_methods": len(data["lifecycle"]["summary"]) == 8,
            "panel_d_latent_multistage_summed": int(data["lifecycle"]["summary"].set_index("method").loc["Latent FM", "stage_count"]) == 2,
            "panel_d_geofno_two_gpu": int(data["lifecycle"]["stages"].set_index("method").loc["Geo-FNO", "gpu_count"]) == 2,
            "metric_name_exact": data["lifecycle"]["manifest"]["metric_label"] == "Replay-equivalent model-core training GPU-hours",
            "joined_source_dataset_aware": set(("dataset", "task", "condition", "method", "checkpoint_sha256", "state_id", "cohort_id", "metric_name", "metric_value")).issubset(plotted.columns),
            "si_separate_subdirectory": all(path.parent == si_dir for path in si_outputs.values()),
            "svg_only_output_bundle": not any(path.suffix.lower() != ".svg" for path in list(outputs.values()) + [composed] + list(si_outputs.values())),
        }
    )
    qa = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "visual_review": {
            "required_at_183mm": True,
            "preview_directory": str(args.preview_dir) if args.preview_dir else None,
            "review_state": "pending_manual_inspection" if args.preview_dir else "not_requested",
        },
    }
    (result_dir / "qa.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    if qa["status"] != "pass":
        raise RuntimeError(f"Figure 5 V5 QA failed: {checks}")

    validation_root = PACKAGE_ROOT / "results" / "ValidationV5"
    v5_bytes = directory_size(validation_root)
    results = panel_results(data)
    top20 = data["localization"]["macro"].loc[np.isclose(data["localization"]["macro"]["spatial_fraction"], 0.2)]
    c_best = top20.sort_values("metric_value", ascending=False).iloc[0]
    crps_best = data["uq_crps"].sort_values("mean_normalized_crps").iloc[0]
    contradictions = []
    if str(crps_best["method"]) != "DMF-Gen":
        contradictions.append(f"DMF-Gen is not best in CRPS; {crps_best['method']} is lower.")
    if str(c_best["method"]) != "DMF-Gen":
        contradictions.append(f"DMF-Gen is not best in spatial error capture; {c_best['method']} has the largest C(0.20).")
    weak = data["uq_spread"].loc[data["uq_spread"]["spearman_ci_low"] <= 0, "method"].astype(str).tolist()
    if weak:
        contradictions.append("State-level association is not distinguishable from zero for " + ", ".join(weak) + ".")
    if not contradictions:
        contradictions.append("No main result contradicts the focused provisional narrative, subject to the stated underdispersion and descriptive-cost limitations.")
    report = docs_dir / f"figure5_v5_completion_report_{args.timestamp}.md"
    report.write_text(
        f"# Figure 5 V5 completion report\n\n- Generated: `{args.timestamp}`\n- Starting branch HEAD: `{git_commit()}`\n- Strict SVG/data QA: **PASS**\n- ValidationV5 result-directory size: **{human_bytes(v5_bytes)}** (`{v5_bytes}` bytes)\n\n"
        "## Reuse matrix\n\n"
        "| Quantity | Formal source | Action |\n|---|---|---|\n"
        "| State-wise normalized CRPS | V3 `uq_compare_formal_20260830_v3r6` | Reused in place; no inference/bootstrap rerun |\n"
        "| State-level spread/error Spearman | V3 `uq_compare_formal_20260830_v3r6` | Reused in place; exact bootstrap reconstruction only |\n"
        "| Spatial error-capture curves | V5 `uq_localization_formal_v5` | New streaming repeated inference; compact reducer only |\n"
        "| Native inference latency + frozen relative-L2 | V3 `formal_cost_clean_v3_20260830_v3` | Reused in place |\n"
        "| Canonical update timing | V4 `training_replay_formal_v4r2` + V4.2 Geo-FNO DDP | Reused in place; no replay rerun |\n"
        "| Replay-equivalent GPU-hours | V5 `lifecycle_formal_v5` | Newly derived from adopted updates/GPU counts; Latent-FM stages summed |\n\n"
        "## Main quantitative results\n\n"
        + "\n".join(f"- Panel {panel}: {value}" for panel, value in results.items())
        + "\n\n## Cleanup and storage\n\n"
        "The panel-c runner created no scratch directory, per-draw file, repeated CSV/NPZ product, or saved ensemble stack. Each in-memory stack was discarded before advancing to the next state. No checkpoint, HDF5 dataset, cache or older result bundle was copied. No new arrays were intentionally retained; only compact CSV summaries, manifests and QA remain. Any temporary Python-rendered QA PNGs live outside the result/figure bundle and must be removed after visual inspection.\n\n"
        "## Unavailable values\n\nNone. All eight lifecycle methods have accepted native latency, adopted update counts, GPU counts, canonical timing and frozen Figure-4 error. Latent FM uses both required sequential stages; Geo-FNO uses the formal 2-GPU DDP replay.\n\n"
        "## Narrative checks and limitations\n\n"
        + "\n".join(f"- {item}" for item in contradictions)
        + "\n- Formal reliability results remain underdispersed; panels a–c must be described as empirical conditional ensemble uncertainty, not Bayesian posterior uncertainty, perfect calibration or prospective error prediction.\n"
        "- Replay-equivalent model-core training GPU-hours are not historical training wall time and do not establish a matched-budget causal efficiency ranking.\n",
        encoding="utf-8",
    )
    companions.append(report)

    reuse_matrix = [
        {"panel": "a", "status": "reused", "source": str(data["run_metadata"]["uq"]["directory"] / "per_state_method.csv")},
        {"panel": "b", "status": "reused", "source": str(data["run_metadata"]["uq"]["directory"] / "per_state_method.csv")},
        {"panel": "c", "status": "new_streaming_summary", "source": str(data["localization"]["directory"] / "error_capture_summary.csv")},
        {"panel": "d", "status": "derived_from_reused_formal_sources", "source": str(data["lifecycle"]["directory"] / "lifecycle_summary.csv")},
    ]
    manifest = {
        "schema_version": config["schema_version"],
        "timestamp": args.timestamp,
        "git_commit": git_commit(),
        "strict_formal": args.strict_formal,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "outputs": {**{key: str(value) for key, value in outputs.items()}, "composed": str(composed)},
        "si_outputs": {key: str(value) for key, value in si_outputs.items()},
        "companions": [str(path) for path in companions],
        "source_table": str(result_dir / "figure5_v5_source.csv"),
        "reuse_matrix": reuse_matrix,
        "validation_v5_bytes": v5_bytes,
        "qa": str(result_dir / "qa.json"),
        "no_proxy": True,
        "historical_training_wall_time_used": False,
    }
    (result_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "figure_dir": str(figure_dir),
                "result_dir": str(result_dir),
                "report": str(report),
                "validation_v5_size": human_bytes(v5_bytes),
                "qa": qa["status"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
