#!/usr/bin/env python
"""Structural and provenance QA for one Figure 5 V3 SVG bundle."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_STEMS = {
    "fig5a_normalized_crps", "fig5b_spread_error_methods", "fig5c_accuracy_latency_clean",
    "fig5d_query_latency", "fig5e_query_memory", "fig5_composed_v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--strict-formal", action="store_true")
    return parser.parse_args()


def _mm(value: str) -> float:
    match = re.fullmatch(r"([0-9.]+)(pt|mm|in)", value)
    if not match:
        raise ValueError(value)
    number, unit = float(match.group(1)), match.group(2)
    return number if unit == "mm" else number * 25.4 if unit == "in" else number * 25.4 / 72.0


def main() -> int:
    args = parse_args()
    timestamp = args.bundle.name
    svgs = sorted(args.bundle.glob("*.svg"))
    errors: list[str] = []
    if len(svgs) != 6:
        errors.append(f"expected 6 SVGs, found {len(svgs)}")
    present = {next((stem for stem in EXPECTED_STEMS if path.name == f"{stem}_{timestamp}.svg"), "") for path in svgs}
    if present != EXPECTED_STEMS:
        errors.append(f"V3 filename mismatch: expected {sorted(EXPECTED_STEMS)}, found {sorted(present)}")
    if list(args.bundle.glob("*.pdf")):
        errors.append("PDF output found in SVG-only testing bundle")
    composed_text = ""
    for path in svgs:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            errors.append(f"{path.name}: invalid XML ({exc})")
            continue
        text_nodes = [node for node in root.iter() if node.tag.endswith("text")]
        if not text_nodes:
            errors.append(f"{path.name}: no editable SVG text nodes")
        all_text = " ".join("".join(node.itertext()) for node in text_nodes)
        upper = all_text.upper()
        if any(token in upper for token in ("DRAFT PROXY", "U_0", "CQ-LR", "S7-B", "NFE", "CALIBRATION", "INTERVAL WIDTH", "127 MS")):
            errors.append(f"{path.name}: forbidden V2/proxy main-figure content")
        if args.strict_formal and "FORMAL V3 EVIDENCE PENDING" in upper:
            errors.append(f"{path.name}: pending content in strict-formal bundle")
        if not root.attrib.get("width") or not root.attrib.get("height"):
            errors.append(f"{path.name}: missing fixed canvas dimensions")
        if path.name.startswith("fig5_composed_v3_"):
            composed_text = upper
            try:
                width, height = _mm(root.attrib["width"]), _mm(root.attrib["height"])
                if abs(width - 183.0) > 0.2 or abs(height - 118.0) > 0.2:
                    errors.append(f"{path.name}: composed canvas is {width:.2f} x {height:.2f} mm, expected 183 x 118 mm")
            except (KeyError, ValueError):
                errors.append(f"{path.name}: unparseable composed dimensions")
    for phrase in ("NORMALIZED CRPS", "SPEARMAN", "WARM MODEL-CORE LATENCY", "REQUESTED QUERY POINTS", "PEAK ALLOCATED MEMORY"):
        if phrase not in composed_text:
            errors.append(f"composed SVG missing required V3 phrase: {phrase}")

    docs = PACKAGE_ROOT / "docs" / "generated"
    expected_docs = [docs / f"{stem}_{timestamp}.md" for stem in EXPECTED_STEMS]
    expected_docs.append(docs / f"figure5_v3_completion_report_{timestamp}.md")
    for path in expected_docs:
        if not path.is_file():
            errors.append(f"missing companion: {path.name}")
    derived = PACKAGE_ROOT / "results" / "derived" / timestamp
    manifest_path = derived / "build_manifest.json"
    for filename in ("fig5a_normalized_crps_source.csv", "fig5b_spread_error_source.csv", "fig5c_accuracy_latency_source.csv", "fig5d_query_latency_source.csv", "fig5e_query_memory_source.csv", "variable_query_support.csv", "timing_boundary_audit.csv"):
        if args.strict_formal and not (derived / filename).is_file():
            errors.append(f"missing strict source table: {filename}")
    if args.strict_formal:
        if not manifest_path.is_file():
            errors.append("missing strict build manifest")
        else:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("schema_version") != "figure5-validation-v3" or not manifest.get("strict_formal"):
                errors.append("build manifest is not strict Figure 5 V3")
            if any(source.get("mode") != "formal" for source in manifest.get("sources", [])):
                errors.append("build manifest contains non-formal panel source")
            if any("ValidationV2/Cost" in source.get("source", "") for source in manifest.get("sources", [])):
                errors.append("build manifest contains superseded V2 cost source")
    report = {"bundle": str(args.bundle), "svg_count": len(svgs), "errors": errors, "status": "pass" if not errors else "fail"}
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
