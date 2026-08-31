#!/usr/bin/env python
"""Structural, print-size, and provenance QA for one Figure 5 V4 bundle."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _mm(value: str) -> float:
    match = re.fullmatch(r"([0-9.]+)(pt|mm|in)", value)
    if not match:
        raise ValueError(value)
    number, unit = float(match.group(1)), match.group(2)
    return number if unit == "mm" else number * 25.4 if unit == "in" else number * 25.4 / 72.0


def _output_root(bundle: Path) -> Path:
    # bundle = <output-root>/figures/generated/<timestamp>
    return bundle.resolve().parents[2]


def audit_bundle(bundle: Path, *, config_path: Path = PACKAGE_ROOT / "configs" / "figure5_v4.yaml", strict_formal: bool = False) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    timestamp = bundle.name
    stems = set(config["figure"]["output_stems"].values())
    svgs = sorted(bundle.glob("*.svg"))
    errors: list[str] = []
    if len(svgs) != 6:
        errors.append(f"expected six SVGs, found {len(svgs)}")
    expected_names = {f"{stem}_{timestamp}.svg" for stem in stems}
    if {path.name for path in svgs} != expected_names:
        errors.append(f"filename mismatch: expected {sorted(expected_names)}, found {sorted(path.name for path in svgs)}")
    if list(bundle.glob("*.pdf")) or list(bundle.glob("*.tif")) or list(bundle.glob("*.png")):
        errors.append("V4 development bundle must contain editable SVG only")
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
        if any(node.tag.endswith("image") for node in root.iter()):
            errors.append(f"{path.name}: raster image element found; V4 text/lines must remain vector")
        text = " ".join("".join(node.itertext()) for node in text_nodes)
        upper = text.upper()
        if "CONDITIONAL ENSEMBLE QUALITY" in upper or "COMPUTATIONAL CHARACTERISTICS" in upper:
            errors.append(f"{path.name}: obsolete V3 row header found")
        if any(token in upper for token in ("DRAFT PROXY", "127 MS", "U_0", "CQ-LR", "S7-B", "CALIBRATION", "INTERVAL WIDTH")) or re.search(r"\bNFE\b", upper):
            errors.append(f"{path.name}: forbidden V2/SI main-figure content")
        if strict_formal and "V4 FORMAL EVIDENCE PENDING" in upper:
            errors.append(f"{path.name}: pending content in strict-formal bundle")
        if not root.attrib.get("width") or not root.attrib.get("height"):
            errors.append(f"{path.name}: missing fixed canvas dimensions")
        if path.name.startswith(f"{config['figure']['output_stems']['composed']}_"):
            composed_text = upper
            try:
                width, height = _mm(root.attrib["width"]), _mm(root.attrib["height"])
                target_w, target_h = float(config["figure"]["width_mm"]), float(config["figure"]["composed_height_mm"])
                if abs(width - target_w) > 0.2 or abs(height - target_h) > 0.2:
                    errors.append(f"{path.name}: canvas is {width:.2f} × {height:.2f} mm, expected {target_w:g} × {target_h:g} mm")
            except (KeyError, ValueError) as exc:
                errors.append(f"{path.name}: unparseable composed dimensions ({exc})")
    required_phrases = [
        "NORMALIZED CRPS",
        "UNCERTAINTY INFORMATIVENESS",
        "WARM MODEL-CORE LATENCY",
        "TRAINING",
    ]
    if strict_formal:
        required_phrases.append("THROUGHPUT-ONLY STRESS TEST")
    for phrase in required_phrases:
        if phrase not in composed_text:
            errors.append(f"composed SVG missing required V4 phrase: {phrase}")

    output_root = _output_root(bundle)
    derived = output_root / "results" / "derived" / timestamp
    manifest_path = derived / "build_manifest.json"
    if not manifest_path.is_file():
        errors.append("missing V4 build manifest")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid V4 build manifest: {exc}")
            manifest = {}
        if manifest.get("schema_version") != "figure5-validation-v4":
            errors.append("build manifest is not schema figure5-validation-v4")
        if strict_formal:
            if manifest.get("strict_formal") is not True:
                errors.append("strict-formal audit requires strict_formal=true in build manifest")
            if any(source.get("mode") != "formal" for source in manifest.get("sources", [])):
                errors.append("strict V4 build manifest contains non-formal panel source")
            modes = {source.get("panel"): source.get("mode") for source in manifest.get("sources", [])}
            if modes.get("d") != "formal":
                errors.append("strict V4 panel d training-cost evidence is absent/unsupported")
            if modes.get("e") != "formal":
                errors.append("strict V4 panel e scale-stress evidence is absent/unsupported")
            if manifest.get("no_v4_fallback") is not True:
                errors.append("strict V4 manifest does not assert no fallback")

    report = {"bundle": str(bundle), "svg_count": len(svgs), "errors": errors, "status": "pass" if not errors else "fail"}
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--config", type=Path, default=PACKAGE_ROOT / "configs" / "figure5_v4.yaml")
    parser.add_argument("--strict-formal", action="store_true")
    args = parser.parse_args()
    report = audit_bundle(args.bundle, config_path=args.config, strict_formal=args.strict_formal)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
