#!/usr/bin/env python
"""Build the strict SVG-only Mixed-HML matched Figure 5 backup."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from utils.figure5_v41_style import apply_style
from utils.figure5_zeroh_matched_v42_data import load_superres_matched
from utils.figure5_zeroh_matched_v42_panels import make_composed, make_standalone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "mixed_hml_matched_v43.yaml")
    parser.add_argument("--timestamp", default=datetime.now().strftime("%Y%m%d_%H%M"))
    parser.add_argument("--output-root", type=Path, default=PACKAGE_ROOT)
    parser.add_argument("--strict-formal", action="store_true")
    return parser.parse_args()


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _save_svg(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches=None, pad_inches=0)
    plt.close(fig)


def _results(panel: str, data: dict[str, Any]) -> str:
    if panel == "a":
        return "; ".join(
            f"{row.method}: mean={row.mean_normalized_crps:.4f}, 95% CI [{row.crps_ci_low:.4f}, {row.crps_ci_high:.4f}]"
            for row in data["uq_crps"].itertuples()
        )
    if panel == "b":
        return "; ".join(
            f"{row.method}: rho={row.spearman_rho:.3f}, 95% CI [{row.spearman_ci_low:.3f}, {row.spearman_ci_high:.3f}]"
            for row in data["uq_spread"].itertuples()
        )
    table = data["cost_native"] if panel == "c" else data["training_cost"]
    cost = "latency" if panel == "c" else "update"
    return "; ".join(
        f"{row.method}: error={row.error:.4f}, {cost}={row.cost_value:.2f} ms"
        for row in table.itertuples()
    )


def _qa(paths: dict[str, Path], config: dict[str, Any], data: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key, path in paths.items():
        root = ET.parse(path).getroot()
        text = path.read_text(encoding="utf-8")
        checks[f"{key}_svg_root"] = root.tag.endswith("svg")
        checks[f"{key}_editable_text"] = "<text" in text
        checks[f"{key}_no_raster"] = "<image" not in text
    expected_width_pt = float(config["figure"]["width_mm"]) / 25.4 * 72.0
    root = ET.parse(paths["composed"]).getroot()
    observed_width_pt = float(str(root.attrib["width"]).removesuffix("pt"))
    checks.update(
        {
            "composed_width_183mm": abs(observed_width_pt - expected_width_pt) < 0.02,
            "uq_formal_pass": data["uq"]["manifest"]["formal"] is True and data["uq"]["qa"]["status"] == "pass",
            "cost_formal_pass": data["cost"]["manifest"]["formal"] is True and data["cost"]["qa"]["status"] == "pass",
            "two_generative_methods_ab": list(data["uq_crps"]["method"]) == config["scenario"]["generative_methods"],
            "four_adopted_methods_cd": list(data["cost_native"]["method"]) == config["scenario"]["all_methods"],
            "uq_200_states_64_draws": len(data["uq_crps_samples"]) == 400 and int(data["uq"]["manifest"]["draws_per_state"]) == 64,
            "accuracy_300_cases": set(data["cost_native"]["error_n"].astype(int)) == {300},
            "native_domain_exact": set(data["cost_native"]["N"].astype(int)) == {16384} and set(data["cost_native"]["sensor_count"].astype(int)) == {256},
            "canonical_hml_weights": set(data["training_cost"]["resolution_weights"]) == {"L=0.333333333333;M=0.333333333333;H=0.333333333333"},
            "deterministic_excluded_from_spread": set(data["uq_spread"]["method"]).isdisjoint({"MLP-RBF", "Senseiver"}),
            "svg_only": all(path.suffix == ".svg" for path in paths.values()),
        }
    )
    return checks


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") != "figure5-mixed-hml-matched-v4.3":
        raise ValueError("The Mixed-HML builder requires its exact V4.3 schema")
    if not args.strict_formal:
        raise RuntimeError("Mixed-HML output has no fallback; pass --strict-formal")
    apply_style(config["style"]["font_family"])
    data = load_superres_matched(config, REPO_ROOT)

    output_dir = args.output_root / "figures" / "generated" / args.timestamp
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True)
    paths: dict[str, Path] = {}
    for panel in "abcd":
        path = output_dir / f"{config['figure']['output_stems'][panel]}_{args.timestamp}.svg"
        _save_svg(make_standalone(panel, data, config), path)
        paths[panel] = path
    composed = output_dir / f"{config['figure']['output_stems']['composed']}_{args.timestamp}.svg"
    _save_svg(make_composed(data, config), composed)
    paths["composed"] = composed

    checks = _qa(paths, config, data)
    if not all(checks.values()):
        raise RuntimeError(f"Mixed-HML SVG QA failed: {[key for key, value in checks.items() if not value]}")
    uq_dir = data["uq"]["directory"]
    cost_dir = data["cost"]["directory"]
    markdown = output_dir / f"{config['figure']['output_stems']['composed']}_{args.timestamp}.md"
    markdown.write_text(
        f"""# Mixed-HML matched Figure 5 V4.3

- Generated: `{args.timestamp}`
- Git commit: `{_git_commit()}`
- Canvas: `{config['figure']['width_mm']} mm x {config['figure']['height_mm']} mm`
- Outputs: four standalone SVG panels plus one composed SVG; no PDF or raster export
- Formal UQ run: `{data['uq']['manifest']['run_id']}` on `{data['uq']['manifest']['environment']['device']}`
- Formal cost run: `{data['cost']['manifest']['run_id']}` on `{data['cost']['manifest']['environment']['device']}`

## Figure contract

Core conclusion: compare uncertainty quality and computational trade-offs for the adopted `3_Mixed_HML` density-reconstruction checkpoints without borrowing Zero-H or Cond_T values. This is a 2x2 quantitative grid at 183-mm publication width.

- **a — normalized empirical CRPS.** Box/scatter contains 200 paired unique-case/time states for DMF-Gen and FFM-Perceiver, with 64 draws/state. The open marker and line show the mean and 2,000-replicate unique-case-bootstrap 95% CI.
- **b — spread-error association.** Box/scatter contains 2,000 bootstrap Spearman estimates relating normalized spatial RMS ensemble spread to physical ensemble-mean relative L2. The open marker is full-sample rho.
- **c — native accuracy-latency.** Audited 300-case mean density relative L2 versus clean warm model-core latency at `N=16,384` and 256 sensors; both axes are logarithmic.
- **d — accuracy-training cost.** The same accuracy versus the configured equal-weight mean of L-, M-, and H-resolution median update time under the adopted 1:1:1 recipe; both axes are logarithmic.

Panels a/b include the two stochastic adopted models. Panels c/d include DMF-Gen, FFM-Perceiver, MLP-RBF, and Senseiver. Deterministic models are excluded from b because zero spread makes Spearman undefined.

## Results

- Panel a: {_results('a', data)}.
- Panel b: {_results('b', data)}.
- Panel c: {_results('c', data)}.
- Panel d: {_results('d', data)}.

## Exact sources

- UQ states: `{uq_dir / 'per_state_method.csv'}`
- UQ summaries: `{uq_dir / 'crps_summary.csv'}` and `{uq_dir / 'spread_error_summary.csv'}`
- Native costs: `{cost_dir / 'native_cost_summary.csv'}`
- Training-update costs: `{cost_dir / 'training_update_summary.csv'}`
- Frozen accuracy: `{config['cohort']['accuracy_per_snapshot']}`
- Sensor plan: `{config['cohort']['sensor_plan']}`

## Interpretation limits

The super-resolution archive uses its legacy full sampling interface and synchronized wall timing. It does not expose the persistent DMF top-k geometry/static-feature cache used by the optimized Cond_T portable core. Panel c is valid as a common-runner Mixed-HML comparison, but its absolute DMF latency is not directly comparable to the cached Cond_T coordinate. No training was performed beyond ephemeral update replay, and no checkpoint or reconstruction cache was duplicated.

## QA

All {len(checks)} checks passed: formal source gates, exact model coverage, 200x64 UQ design, 300-case accuracy cohort, native domain, 1:1:1 training weights, editable SVG text, absence of raster images, and exact 183-mm composed width.
""",
        encoding="utf-8",
    )
    if len(list(output_dir.iterdir())) != 6:
        raise RuntimeError("SVG-only bundle must contain exactly five SVGs and one Markdown file")
    print(json.dumps({"output_dir": str(output_dir), "svg_outputs": {key: str(value) for key, value in paths.items()}, "markdown": str(markdown), "qa": "pass"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
