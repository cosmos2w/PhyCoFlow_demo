#!/usr/bin/env python
"""Audit the additive Figure 5 V5.1 panel-c/d exploration bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
EXPECTED_METHODS = {
    "DMF-Gen", "FFM-FNO", "FFM-Perceiver", "Latent FM", "SiT"
}
EXPECTED_LIFECYCLE_METHODS = EXPECTED_METHODS | {"MLP-RBF", "Geo-FNO", "Senseiver"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestamp", default="20260902_1129")
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "figure5_v51_exploration.yaml",
    )
    parser.add_argument(
        "--benchmark-config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "training_footprint_common_b32_v51.yaml",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def add(checks: dict[str, bool], name: str, value: object) -> None:
    checks[name] = bool(value)


def audit_svg(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        return [f"invalid XML: {exc}"]
    if not root.attrib.get("width") or not root.attrib.get("height"):
        errors.append("missing fixed width/height")
    text_nodes = [node for node in root.iter() if node.tag.endswith("text")]
    if not text_nodes:
        errors.append("no editable text nodes")
    if any(node.tag.endswith("image") for node in root.iter()):
        errors.append("contains raster image node")
    return errors


def main() -> int:
    args = parse_args()
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    benchmark_config = yaml.safe_load(args.benchmark_config.read_text(encoding="utf-8"))
    add(checks, "panel_c_contract_schema", config.get("schema_version") == "figure5-v51-panel-c-exploration-1")
    add(checks, "contract_timestamp", str(config.get("timestamp")) == args.timestamp)
    add(checks, "benchmark_contract_schema", benchmark_config.get("schema_version") == "figure5-training-footprint-common-b32-v51")
    add(checks, "benchmark_common_b32", int(benchmark_config.get("batch_size", -1)) == 32)
    add(checks, "benchmark_float32", benchmark_config.get("dtype") == "float32")
    add(checks, "benchmark_m256", int(benchmark_config.get("sensor_count", -1)) == 256)

    v3_root = REPO_ROOT / config["v3_uq_root"]
    v3_manifest = read_json(v3_root / "manifest.json")
    v3_qa = read_json(v3_root / "qa.json")
    v3_states = pd.read_csv(v3_root / "per_state_method.csv")
    add(checks, "v3_formal_complete", v3_manifest.get("formal") is True and v3_manifest.get("status") == "complete")
    add(checks, "v3_qa_pass", v3_qa.get("status") == "pass")
    add(checks, "v3_paired_200x5", len(v3_states) == 1000 and v3_states["state"].nunique() == 200)
    add(checks, "v3_expected_methods", set(v3_states["method"]) == EXPECTED_METHODS)
    add(
        checks,
        "exact_fig4_state_mapping",
        int(config["fig4_state_test_index"]) == 0
        and int(config["fig4_state_original_time_index"]) == 5
        and bool(((v3_states["state"] == 0) & (v3_states["original_time_index"] == 5)).any()),
    )

    c_root = PACKAGE_ROOT / "results" / "ValidationV51" / "PanelC" / config["outputs"]["result_run_id"]
    c_manifest = read_json(c_root / "manifest.json")
    c_qa = read_json(c_root / "qa.json")
    allowed_panel_c_files = {"manifest.json", "qa.json", "selective_risk.csv", "interface_profile.csv", "pointwise_posterior.csv", "derived_functionals.csv"}
    add(checks, "panel_c_exact_allowed_files", {path.name for path in c_root.iterdir()} == allowed_panel_c_files)
    add(checks, "panel_c_manifest_complete", c_manifest.get("status") == "complete")
    add(checks, "panel_c_qa_pass", c_qa.get("status") == "pass")
    add(checks, "panel_c_no_saved_draw_stacks", not list(c_root.glob("*.npz")) and not list(c_root.glob("*draw*")))
    profile = pd.read_csv(c_root / "interface_profile.csv")
    posterior = pd.read_csv(c_root / "pointwise_posterior.csv")
    functionals = pd.read_csv(c_root / "derived_functionals.csv")
    add(checks, "panel_c_pointwise_rows_compact", len(profile) == 200 and len(posterior) == 960 and len(functionals) == 650)
    add(checks, "panel_c_pointwise_methods_complete", set(profile["method"]) == EXPECTED_METHODS and set(posterior["method"]) == EXPECTED_METHODS)
    add(checks, "panel_c_draw_count_64", (profile["draw_count"] == 64).all() and (posterior["draw_count"] == 64).all())
    validation_plan = yaml.safe_load((REPO_ROOT / config["validation_plan"]).read_text(encoding="utf-8"))
    frozen_ch4_std = float(validation_plan["dataset_statistics"]["std"][0])
    add(checks, "panel_c_profile_crps_scale_fixed", profile["training_std"].nunique() == 1 and abs(float(profile["training_std"].iloc[0]) - frozen_ch4_std) < 1.0e-12)
    add(checks, "panel_c_inference_provenance_explicit", c_manifest.get("pointwise_execution", {}).get("inference_executed") is True and c_manifest.get("pointwise_execution", {}).get("confirm_gpu2_free") is True)
    details["panel_c_files"] = sorted(path.name for path in c_root.iterdir())

    b_root = PACKAGE_ROOT / "results" / "ValidationV51" / "TrainingFootprint" / benchmark_config["output"]["run_id"]
    required_benchmark = {
        "manifest.json", "qa.json", "training_footprint_summary.csv",
        "training_stage_summary.csv", "benchmark_repeats.csv",
        "gpu_state_before.txt", "gpu_state_after.txt",
    }
    add(checks, "benchmark_exact_required_files", {path.name for path in b_root.iterdir()} == required_benchmark)
    b_manifest = read_json(b_root / "manifest.json")
    b_qa = read_json(b_root / "qa.json")
    stages = pd.read_csv(b_root / "training_stage_summary.csv")
    repeats = pd.read_csv(b_root / "benchmark_repeats.csv")
    add(checks, "benchmark_formal_complete", b_manifest.get("formal") is True and b_manifest.get("status") == "complete")
    add(checks, "benchmark_qa_pass", b_qa.get("status") == "pass")
    add(checks, "benchmark_all_b32", (stages["batch_size"] == 32).all())
    add(checks, "benchmark_methods_complete", set(stages["method"]) == EXPECTED_LIFECYCLE_METHODS)
    add(checks, "latent_two_stages", len(stages[stages["method"] == "Latent FM"]) == 2)
    add(checks, "repeat_table_exact", len(repeats) == 900)
    add(checks, "optimizer_updates_executed", (stages["optimizer_step_successes_warmup"] == 20).all() and (stages["optimizer_step_successes_measured"] == 100).all() and (stages["optimizer_step_skips_warmup"] == 0).all() and (stages["optimizer_step_skips_measured"] == 0).all())
    ema_expected = stages["ema_expected"].astype(str).str.lower().eq("true")
    add(checks, "ema_updates_executed_where_required", (stages.loc[ema_expected, "ema_update_successes_warmup"] == 20).all() and (stages.loc[ema_expected, "ema_update_successes_measured"] == 100).all() and (stages.loc[ema_expected, "ema_update_skips_measured"] == 0).all() and (stages.loc[~ema_expected, "ema_update_attempts_measured"] == 0).all())
    sit = stages.loc[stages["method"] == "SiT"].iloc[0]
    add(checks, "sit_spike_guard_control_audited", bool(sit["benchmark_control_sit_spike_state_reset"]) and b_qa.get("sit_spike_state_reset_before_warmups") is True)
    add(checks, "benchmark_no_per_draw_files", not list(b_root.glob("*.npz")))
    details["benchmark_statuses"] = stages[["method", "stage_id", "status"]].to_dict(orient="records")

    figure_root = PACKAGE_ROOT / "figures" / "exploration" / "figure5_v51" / args.timestamp
    docs_root = PACKAGE_ROOT / "docs" / "exploration" / "figure5_v51" / args.timestamp
    c_svgs = sorted((figure_root / "panel_c_candidates").glob("*.svg"))
    d_svgs = sorted((figure_root / "panel_d_candidates").glob("*.svg"))
    add(checks, "at_least_four_panel_c_svgs", len(c_svgs) >= 4)
    add(checks, "at_least_four_panel_d_svgs", len(d_svgs) >= 4)
    add(checks, "no_composed_v51_svg", not list(figure_root.rglob("*composed*.svg")))
    add(checks, "svg_only", not list(figure_root.rglob("*.pdf")) and not list(figure_root.rglob("*.png")))
    svg_errors = {str(path.relative_to(figure_root)): audit_svg(path) for path in c_svgs + d_svgs}
    svg_errors = {path: errors for path, errors in svg_errors.items() if errors}
    add(checks, "all_svgs_structurally_valid", not svg_errors)
    details["svg_errors"] = svg_errors
    details["panel_c_svgs"] = [path.name for path in c_svgs]
    details["panel_d_svgs"] = [path.name for path in d_svgs]

    missing_companions = []
    for category, paths in (("panel_c_candidates", c_svgs), ("panel_d_candidates", d_svgs)):
        for path in paths:
            companion = docs_root / category / f"{path.stem}.md"
            if not companion.is_file():
                missing_companions.append(str(companion.relative_to(docs_root)))
    add(checks, "one_companion_per_svg", not missing_companions)
    add(checks, "candidate_comparison_present", (docs_root / "candidate_comparison.md").is_file())
    add(checks, "completion_report_present", (docs_root / "completion_report.md").is_file())
    add(checks, "panel_c_assessment_present", (docs_root / "panel_c_candidate_assessment.md").is_file())
    add(checks, "panel_d_assessment_present", (docs_root / "panel_d_candidate_assessment.md").is_file())
    derived = PACKAGE_ROOT / "results" / "ValidationV51" / "Derived" / args.timestamp
    for mode in ("existing_formal", "common_b32"):
        add(checks, f"panel_d_{mode}_manifest_complete", read_json(derived / f"panel_d_manifest_{mode}.json").get("status") == "complete")
        add(checks, f"panel_d_{mode}_qa_pass", read_json(derived / f"panel_d_qa_{mode}.json").get("status") == "pass")
    add(checks, "no_common_b32_wait_artifact", not (derived / "panel_d_common_b32_wait.json").exists())
    details["missing_companions"] = missing_companions

    status = "pass" if checks and all(checks.values()) else "fail"
    report = {
        "schema_version": "figure5-v51-exploration-audit-1",
        "timestamp": args.timestamp,
        "status": status,
        "checks": checks,
        "details": details,
    }
    payload = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
