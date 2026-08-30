#!/usr/bin/env python3
"""Verify tracked release manifests without modifying or deleting artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "manifest",
        type=Path,
        nargs="?",
        default=ROOT / "artifacts/GL_rbf_CQ_v0.9.0-rc1_portable.json",
    )
    args = parser.parse_args()
    record = json.loads(args.manifest.read_text())
    artifact = ROOT / record["artifact"]
    if not artifact.is_file():
        raise SystemExit(f"MISSING: {artifact}")
    actual = sha256(artifact)
    if actual != record["sha256"]:
        raise SystemExit(f"MISMATCH: {artifact}: {actual} != {record['sha256']}")
    print(f"OK {record['sha256']}  {artifact}")


if __name__ == "__main__":
    main()
