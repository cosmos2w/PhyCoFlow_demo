#!/usr/bin/env python
"""Structural QA for one generated Figure 5 SVG bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree as ET


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    svgs = sorted(args.bundle.glob("*.svg"))
    errors: list[str] = []
    if len(svgs) != 9:
        errors.append(f"expected 9 SVGs, found {len(svgs)}")
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
        if not any(token in all_text for token in ("FORMAL VALIDATION", "DRAFT PROXY", "AWAITING FORMAL RUN")):
            errors.append(f"{path.name}: no evidence-status badge")
        if not root.attrib.get("width") or not root.attrib.get("height"):
            errors.append(f"{path.name}: missing fixed canvas dimensions")
    report = {"bundle": str(args.bundle), "svg_count": len(svgs), "errors": errors, "status": "pass" if not errors else "fail"}
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
