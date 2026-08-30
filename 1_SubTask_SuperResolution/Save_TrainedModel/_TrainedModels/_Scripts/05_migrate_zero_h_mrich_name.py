#!/usr/bin/env python
"""Migrate the misnamed zero-H L-rich derived namespace to M-rich.

The trained recipe is L:M:H = 1:2:0 and its archive directory is already
``5_ZeroH_MRich``.  This one-time migration renames only derived directories
and metadata.  Reconstruction and observation arrays are verified byte-for-
byte before and after the metadata rewrite; model inference is never invoked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

from common.cache import cache_identity, update_cache_metadata
from common.config import ARCHIVE_DIR, FIGURES_DIR, RESULTS_DIR, load_config, method_items
from common.io_utils import write_json
from common.recipe_registry import flatten_run_config, resolve_recipe_dir


OLD_KEY = "5_ZeroH_LRich"
NEW_KEY = "5_ZeroH_MRich"
OLD_LABEL = "Zero-H-L-rich"
NEW_LABEL = "Zero-H-M-rich"
TEXT_SUFFIXES = {".csv", ".json", ".yaml", ".yml", ".md", ".txt"}


def validate_trained_recipes() -> list[dict]:
    cfg = load_config()
    spec = cfg["recipes"][NEW_KEY]
    checks = []
    for model in method_items(cfg):
        run_dir = resolve_recipe_dir(ARCHIVE_DIR / model["directory"], NEW_KEY, spec)
        path = run_dir / "run_config.yaml"
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8") as handle:
            actual = str(flatten_run_config(yaml.safe_load(handle) or {}).get("multires_ratio", ""))
        if actual != "1:2:0":
            raise ValueError(f"{model['key']} is not the expected M-rich recipe: ratio={actual}")
        checks.append({"model": model["key"], "run_config": str(path), "actual_ratio": actual})
    return checks


def array_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with np.load(path, allow_pickle=False) as data:
        for key in sorted(k for k in data.files if k != "metadata_json"):
            arr = np.ascontiguousarray(data[key])
            digest.update(key.encode("utf-8"))
            digest.update(str(arr.dtype).encode("ascii"))
            digest.update(str(arr.shape).encode("ascii"))
            digest.update(arr.tobytes())
    return digest.hexdigest()


def rename_recipe_directories(root: Path) -> list[tuple[Path, Path]]:
    renamed = []
    for old_dir in sorted(root.glob(f"*/{OLD_KEY}")):
        new_dir = old_dir.with_name(NEW_KEY)
        if new_dir.exists():
            raise FileExistsError(f"Both legacy and canonical directories exist: {old_dir} / {new_dir}")
        old_dir.rename(new_dir)
        renamed.append((old_dir, new_dir))
    return renamed


def migrate_cache_files() -> tuple[dict[str, str], dict[str, str], list[dict]]:
    path_map: dict[str, str] = {}
    identity_map: dict[str, str] = {}
    checks = []
    cache_root = RESULTS_DIR / "ReconstructionCache"
    for recipe_dir in sorted(cache_root.glob(f"*/*/{NEW_KEY}")):
        for source in sorted(recipe_dir.glob("RecCache_*.npz")):
            before = array_digest(source)
            legacy_source = Path(str(source).replace(f"/{NEW_KEY}/", f"/{OLD_KEY}/"))
            with np.load(source, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata_json"].item()))
            old_identity = source.stem.rsplit("_", 1)[-1]
            manifest_path = str(meta.get("manifest_path", "")).replace(OLD_KEY, NEW_KEY)
            meta = update_cache_metadata(source, {
                "recipe": NEW_KEY,
                "recipe_label": NEW_LABEL,
                "manifest_path": manifest_path,
            })
            new_identity = cache_identity(meta)
            target = source.with_name(
                f"RecCache_s{int(meta['snapshot_index']):04d}_n{int(meta['sensor_count'])}_{new_identity}.npz"
            )
            if target != source:
                if target.exists():
                    raise FileExistsError(target)
                os.replace(source, target)
            after = array_digest(target)
            if before != after:
                raise RuntimeError(f"Numerical cache arrays changed during naming migration: {target}")
            path_map[str(legacy_source)] = str(target)
            path_map[str(legacy_source.resolve())] = str(target.resolve())
            identity_map[old_identity] = new_identity
            checks.append({
                "model": meta.get("model"), "snapshot_index": int(meta["snapshot_index"]),
                "sensor_count": int(meta["sensor_count"]), "array_digest": after,
                "old_identity": old_identity, "new_identity": new_identity,
                "path": str(target),
            })
    return path_map, identity_map, checks


def rewrite_text_artifacts(path_map: dict[str, str], identity_map: dict[str, str]) -> list[str]:
    changed = []
    replacements = sorted(path_map.items(), key=lambda item: len(item[0]), reverse=True)
    replacements += sorted(identity_map.items(), key=lambda item: len(item[0]), reverse=True)
    replacements += [(OLD_KEY, NEW_KEY), (OLD_LABEL, NEW_LABEL)]
    for root in (RESULTS_DIR, FIGURES_DIR):
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES):
            if path.name.startswith("RecipeNamingMigration_"):
                continue
            original = path.read_text(encoding="utf-8")
            updated = original
            for old, new in replacements:
                updated = updated.replace(old, new)
            if updated == original:
                continue
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_text(updated, encoding="utf-8")
            os.replace(tmp, path)
            changed.append(str(path))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-id", default=datetime.now().strftime("%Y%m%d_%H%M"))
    args = parser.parse_args()

    trained_recipe_checks = validate_trained_recipes()
    renamed = []
    renamed += rename_recipe_directories(RESULTS_DIR / "DatasetStats")
    cache_root = RESULTS_DIR / "ReconstructionCache"
    for run_dir in sorted(p for p in cache_root.iterdir() if p.is_dir() and p.name != "Shared"):
        renamed += rename_recipe_directories(run_dir)

    path_map, identity_map, checks = migrate_cache_files()
    changed_text = rewrite_text_artifacts(path_map, identity_map)
    remaining = []
    for root in (RESULTS_DIR, FIGURES_DIR):
        for path in root.rglob("*"):
            if path.is_dir() and path.name == OLD_KEY:
                remaining.append(str(path))
            elif path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
                if path.name.startswith("RecipeNamingMigration_"):
                    continue
                if OLD_KEY in path.read_text(encoding="utf-8") or OLD_LABEL in path.read_text(encoding="utf-8"):
                    remaining.append(str(path))
    if remaining:
        raise RuntimeError(f"Legacy naming remains in derived artifacts: {remaining[:10]}")

    audit = {
        "audit_id": args.audit_id,
        "old_recipe": OLD_KEY,
        "new_recipe": NEW_KEY,
        "verified_training_ratio": "1:2:0",
        "trained_recipe_checks": trained_recipe_checks,
        "model_inference_rerun": False,
        "renamed_directories": [{"from": str(a), "to": str(b)} for a, b in renamed],
        "cache_files_migrated": len(checks),
        "cache_array_digests_preserved": all(bool(row["array_digest"]) for row in checks),
        "text_artifacts_updated": changed_text,
        "legacy_occurrences_remaining": remaining,
    }
    out = RESULTS_DIR / "ModelInventory" / f"RecipeNamingMigration_{args.audit_id}.json"
    write_json(out, audit)
    print(f"[OK] migrated {len(checks)} cache files without changing numerical arrays")
    print(f"[OK] naming migration audit: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
