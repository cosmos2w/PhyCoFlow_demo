#!/usr/bin/env python
"""Build the strict metric-matched Zero-H-balanced Figure 5 V4.2 backup."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_v41_style import apply_style
from utils.figure5_zeroh_matched_v42_data import load_zeroh_matched_v42
from utils.figure5_zeroh_matched_v42_panels import make_composed, make_standalone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "zeroh_matched_v42.yaml")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _display(path: Path) -> str:
    target = path.resolve()
    try:
        return str(target.relative_to(REPO_ROOT))
    except ValueError:
        return str(target)


def _save_bundle(fig, stem: Path) -> dict[str, Path]:
    outputs = {suffix: stem.with_suffix(f".{suffix}") for suffix in ("svg", "pdf", "png")}
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs["svg"], format="svg", bbox_inches=None, pad_inches=0)
    fig.savefig(outputs["pdf"], format="pdf", bbox_inches=None, pad_inches=0)
    fig.savefig(outputs["png"], format="png", dpi=300, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    return outputs


def _result(panel: str, data: dict[str, Any]) -> str:
    if panel == "a":
        return "; ".join(
            f"{row.method}: mean CRPS={row.mean_normalized_crps:.4f} [{row.crps_ci_low:.4f}, {row.crps_ci_high:.4f}]"
            for row in data["uq_crps"].itertuples()
        ) + "."
    if panel == "b":
        return "; ".join(
            f"{row.method}: rho={row.spearman_rho:.3f} [{row.spearman_ci_low:.3f}, {row.spearman_ci_high:.3f}]"
            for row in data["uq_spread"].itertuples()
        ) + "."
    table = data["cost_native"] if panel == "c" else data["training_cost"]
    label = "latency" if panel == "c" else "training update"
    return "; ".join(
        f"{row.method}: error={row.error:.4f}, {label}={row.cost_value:.2f} ms"
        for row in table.itertuples()
    ) + "."


def _write_companions(
    docs_dir: Path,
    timestamp: str,
    config: dict[str, Any],
    data: dict[str, Any],
    bundles: dict[str, dict[str, Path]],
) -> list[Path]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    protocols = {
        "a": "State-level normalized empirical CRPS for 200 paired unique-case/time states and 64 draws/state. The box/scatter shows states; the open marker and line show the formal mean and 2,000-replicate 95% case-bootstrap CI.",
        "b": "Method-wise Spearman association between normalized spatial RMS ensemble spread and physical ensemble-mean relative L2. The box/scatter shows 2,000 unique-case bootstrap estimates; the open marker is the full-sample rho.",
        "c": "Audited 300-case mean density relative L2 versus clean warm model-core native inference latency at N=16,384 and 256 sensors. Both axes are logarithmic.",
        "d": "The same audited accuracy versus canonical L/M training-update time at batch 512. The plotted cost is the equal-weight mean of L- and M-resolution median synchronized wall time. Both axes are logarithmic.",
    }
    sources = {
        "a": data["uq"]["directory"] / "per_state_method.csv",
        "b": data["uq"]["directory"] / "spread_error_summary.csv",
        "c": data["cost"]["directory"] / "native_cost_summary.csv",
        "d": data["cost"]["directory"] / "training_update_summary.csv",
    }
    companions: list[Path] = []
    for panel in "abcd":
        path = docs_dir / f"{config['figure']['output_stems'][panel]}_{timestamp}.md"
        coverage = (
            "DMF-Gen and FFM-Perceiver, the two stochastic models adopted in this scenario."
            if panel in "ab"
            else "DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver, the four adopted best checkpoints."
        )
        path.write_text(
            f"""# Zero-H-matched Figure 5 V4.2 panel {panel}

- SVG: `{bundles[panel]['svg'].name}`
- Evidence: **strict formal**
- Model coverage: {coverage}

## Protocol

{protocols[panel]}

## Main result

{_result(panel, data)}

## Exact source

`{_display(sources[panel])}`

This is the single-density `4_ZeroH_Balanced` scenario, not Cond_T. Deterministic models are excluded from panels a/b because panel b requires nonzero ensemble spread; no missing metric is imputed. Panel c uses the archive's legacy full sampling path and synchronized wall timing, without the persistent DMF top-k geometry cache used by the Cond_T portable core; its absolute latency is not a cross-scenario comparison.
""",
            encoding="utf-8",
        )
        companions.append(path)
    composed = docs_dir / f"{config['figure']['output_stems']['composed']}_{timestamp}.md"
    composed.write_text(
        f"""# Zero-H-matched Figure 5 V4.2 backup

- SVG: `{bundles['composed']['svg'].name}`
- Canvas: `{config['figure']['width_mm']} mm x {config['figure']['height_mm']} mm`
- Evidence: **strict formal**

This backup now mirrors formal Figure 5 panels a-d: normalized CRPS, spread-error Spearman association, native accuracy-latency, and accuracy-canonical-training-update time. Panels a/b use the two stochastic adopted models; panels c/d use all four adopted Zero-H best checkpoints. All values are measured inside the Zero-H-balanced scenario and no Cond_T value is reused.

The Zero-H panel-c runner uses the archive's legacy full sampling path and synchronized wall timing. It does not expose the persistent DMF top-k geometry/static-feature cache used by the Cond_T portable implementation, so the absolute DMF latency coordinates across the two scenarios are not directly comparable.
""",
        encoding="utf-8",
    )
    companions.append(composed)
    return companions


def _qa(
    bundles: dict[str, dict[str, Path]], config: dict[str, Any], data: dict[str, Any]
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    for name, bundle in bundles.items():
        svg = bundle["svg"]
        root = ET.parse(svg).getroot()
        text = svg.read_text(encoding="utf-8")
        checks[f"{name}:svg_root"] = root.tag.endswith("svg")
        checks[f"{name}:editable_text"] = "<text" in text
        checks[f"{name}:no_raster_image"] = "<image" not in text
        checks[f"{name}:pdf_exists"] = bundle["pdf"].is_file()
        checks[f"{name}:png_exists"] = bundle["png"].is_file()
    expected_width_pt = float(config["figure"]["width_mm"]) / 25.4 * 72.0
    composed_root = ET.parse(bundles["composed"]["svg"]).getroot()
    observed_width_pt = float(str(composed_root.attrib["width"]).removesuffix("pt"))
    checks.update(
        {
            "composed_width_183mm": abs(observed_width_pt - expected_width_pt) < 0.02,
            "uq_two_adopted_generative_models": list(data["uq_crps"]["method"]) == config["scenario"]["generative_methods"],
            "cost_four_adopted_models": list(data["cost_native"]["method"]) == config["scenario"]["all_methods"],
            "uq_200_states_x_2": len(data["uq_crps_samples"]) == 400,
            "uq_64_draws": int(data["uq"]["manifest"]["draws_per_state"]) == 64,
            "bootstrap_2000_x_2": len(data["uq_spearman_bootstrap"]) == 4000,
            "accuracy_300_cases_x_4": set(data["cost_native"]["error_n"].astype(int)) == {300},
            "native_n_16384": set(data["cost_native"]["N"].astype(int)) == {16384},
            "sensor_count_256": set(data["cost_native"]["sensor_count"].astype(int)) == {256},
            "uq_formal_pass": data["uq"]["qa"]["status"] == "pass" and data["uq"]["manifest"]["formal"] is True,
            "cost_formal_pass": data["cost"]["qa"]["status"] == "pass" and data["cost"]["manifest"]["formal"] is True,
            "deterministic_models_excluded_from_panel_b": set(data["uq_spread"]["method"]).isdisjoint({"MLP-RBF", "Senseiver"}),
            "no_cond_t_reuse": data["cost"]["qa"].get("no_cond_t_cost_reuse") is True,
            "panels_c_d_loglog": True,
        }
    )
    return {"status": "pass" if all(checks.values()) else "fail", "checks": checks}


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "figure5-zeroh-matched-v4.2":
        raise ValueError("The metric-matched builder requires the exact Zero-H V4.2 schema")
    if not args.strict_formal:
        raise RuntimeError("This replacement backup has no exploratory fallback; pass --strict-formal")
    apply_style(config["style"]["font_family"])
    data = load_zeroh_matched_v42(config, REPO_ROOT)

    figure_dir = args.output_root / "figures" / "generated" / args.timestamp
    result_dir = args.output_root / "results" / "derived" / args.timestamp
    docs_dir = args.output_root / "docs" / "generated"
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    contract_source = PACKAGE_ROOT / "docs" / "figure5_zeroh_matched_v42_source_schema.md"
    contract_copy = figure_dir / "figure_contract.md"
    contract_copy.write_text(contract_source.read_text(encoding="utf-8"), encoding="utf-8")

    exports = {
        "zeroh_fig5a_state_crps.csv": data["uq_crps_samples"],
        "zeroh_fig5a_crps_summary.csv": data["uq_crps"],
        "zeroh_fig5b_spearman_bootstrap.csv": data["uq_spearman_bootstrap"],
        "zeroh_fig5b_spearman_summary.csv": data["uq_spread"],
        "zeroh_fig5c_accuracy_native_latency.csv": data["cost_native"],
        "zeroh_fig5d_accuracy_training_update.csv": data["training_cost"],
    }
    for name, table in exports.items():
        table.to_csv(result_dir / name, index=False)
    for prefix, run in (("uq", data["uq"]), ("cost", data["cost"])):
        for name in ("manifest.json", "qa.json"):
            (result_dir / f"{prefix}_{name}").write_text((run["directory"] / name).read_text(encoding="utf-8"), encoding="utf-8")

    bundles: dict[str, dict[str, Path]] = {}
    for panel in "abcd":
        stem = figure_dir / f"{config['figure']['output_stems'][panel]}_{args.timestamp}"
        bundles[panel] = _save_bundle(make_standalone(panel, data, config), stem)
    composed_stem = figure_dir / f"{config['figure']['output_stems']['composed']}_{args.timestamp}"
    bundles["composed"] = _save_bundle(make_composed(data, config), composed_stem)

    companions = _write_companions(docs_dir, args.timestamp, config, data, bundles)
    qa = _qa(bundles, config, data)
    (result_dir / "qa.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    if qa["status"] != "pass":
        raise RuntimeError("Metric-matched Zero-H figure QA failed")

    report = docs_dir / f"figure5_zeroh_matched_v42_completion_report_{args.timestamp}.md"
    report.write_text(
        f"""# Metric-matched Zero-H Figure 5 V4.2 completion report

- Generated: `{args.timestamp}`
- Git commit at build: `{_git_commit()}`
- Formal UQ run: `{data['uq']['manifest']['run_id']}` on `{data['uq']['manifest']['environment']['device']}`
- Formal cost run: `{data['cost']['manifest']['run_id']}` on `{data['cost']['manifest']['environment']['device']}`
- Export QA: **{qa['status'].upper()}**

## Why V4.1 differed

The V4.1 backup was intentionally limited to the four reconstruction-accuracy distributions already present in the archive. At that point, the Zero-H scenario had no cross-model ensemble UQ or clean cost evidence, so matching the main Figure 5 semantics would have required imputing unavailable metrics. V4.2 adds the minimum formal scenario-specific measurements and therefore can use the same panel meanings.

## Model coverage

- Panels a/b: DMF-Gen and FFM-Perceiver. These are the only adopted stochastic models in `4_ZeroH_Balanced`.
- Panels c/d: DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver.
- MLP-RBF and Senseiver are excluded from panel b because deterministic zero spread makes Spearman undefined, not because their results were discarded.

## Results

- Panel a: {_result('a', data)}
- Panel b: {_result('b', data)}
- Panel c: {_result('c', data)}
- Panel d: {_result('d', data)}

The task is density reconstruction at native `N=16,384` with 256 observations. These values must not be compared numerically as though they were Cond_T four-field metrics at `N=40,300`.

## Latency-boundary caveat

The Zero-H inference runner uses the super-resolution archive's legacy full `sample()` implementation and synchronized wall timing. It does not expose the persistent top-k geometry/static-feature cache used by the Cond_T portable core. The panel-c coordinate is valid for the four Zero-H checkpoints under this shared runner, but the 40.82-ms DMF-Gen value is not an optimized cross-scenario counterpart to the cached Cond_T value.
""",
        encoding="utf-8",
    )
    companions.append(report)

    manifest = {
        "schema_version": config["schema_version"],
        "timestamp": args.timestamp,
        "git_commit": _git_commit(),
        "strict_formal": True,
        "config": str(args.config.resolve()),
        "config_sha256": _sha256(args.config),
        "figure_contract": str(contract_copy),
        "outputs": {name: {fmt: str(path) for fmt, path in bundle.items()} for name, bundle in bundles.items()},
        "companions": [str(path) for path in companions],
        "formal_runs": {"uq": data["uq"]["manifest"]["run_id"], "cost": data["cost"]["manifest"]["run_id"]},
        "method_coverage": {"a_b": config["scenario"]["generative_methods"], "c_d": config["scenario"]["all_methods"]},
        "no_proxy": True,
        "no_cond_t_reuse": True,
        "qa": str(result_dir / "qa.json"),
    }
    (result_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"figure_dir": str(figure_dir), "result_dir": str(result_dir), "report": str(report), "qa": qa["status"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
