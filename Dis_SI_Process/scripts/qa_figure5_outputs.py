#!/usr/bin/env python
"""Structural QA for one generated six-panel Figure 5 V2 SVG bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--strict-formal", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    svgs = sorted(args.bundle.glob("*.svg"))
    errors: list[str] = []
    expected_stems = {
        "fig5a_calibration", "fig5b_sharpness", "fig5c_spread_error",
        "fig5d_accuracy_latency", "fig5e_query_memory", "fig5f_nfe_tradeoff",
        "fig5_composed_v2",
    }
    if len(svgs) != 7:
        errors.append(f"expected 7 SVGs, found {len(svgs)}")
    present = {next((stem for stem in expected_stems if path.name.startswith(stem + "_")), "") for path in svgs}
    if present != expected_stems:
        errors.append(f"V2 filename mismatch: expected {sorted(expected_stems)}, found {sorted(present)}")
    if list(args.bundle.glob("*.pdf")):
        errors.append("PDF output found in SVG-only draft bundle")
    for path in svgs:
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as exc:
            errors.append(f"{path.name}: invalid XML ({exc})")
            continue
        text_nodes = [node for node in root.iter() if node.tag.endswith("text")]
        if not text_nodes:
            errors.append(f"{path.name}: no editable SVG text nodes")
        all_text = " ".join("".join(node.itertext()) for node in text_nodes).upper()
        if "FORMAL EVIDENCE PENDING" in all_text and "AWAITING FORMAL RUN" not in all_text:
            errors.append(f"{path.name}: pending content has no status badge")
        if "DRAFT PROXY" in all_text or any(token in all_text for token in ("U_0", "CQ-LR", "S7-B", "F0")):
            errors.append(f"{path.name}: forbidden legacy/proxy label")
        if args.strict_formal and "AWAITING FORMAL RUN" in all_text:
            errors.append(f"{path.name}: pending content in strict-formal bundle")
        if not root.attrib.get("width") or not root.attrib.get("height"):
            errors.append(f"{path.name}: missing fixed canvas dimensions")
    report = {"bundle": str(args.bundle), "svg_count": len(svgs), "errors": errors, "status": "pass" if not errors else "fail"}
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
