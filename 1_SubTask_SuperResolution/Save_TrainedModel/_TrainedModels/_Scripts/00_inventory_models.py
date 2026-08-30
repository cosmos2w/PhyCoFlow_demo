#!/usr/bin/env python
"""Inventory every configured model/recipe and validate its actual manifest."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, recipe_items, run_id
from common.dataset_loader import locate_or_rebuild_manifest, read_run_config, resolve_run_config_path
from common.io_utils import artifact_name, write_csv
from common.model_loader import inspect_artifacts, load_model, status_from_exception
from common.recipe_registry import resolve_recipe_dir, validate_recipe


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    add_common_args(p); p.add_argument("--recipes", nargs="+", default=["all"])
    p.add_argument("--checkpoint", choices=["last", "best"]); p.add_argument("--allow-checkpoint-fallback", action="store_true")
    p.add_argument("--probe-load", action="store_true"); p.add_argument("--device")
    args = p.parse_args(); cfg = load_config(args.config); rid = run_id(args.run_id); ensure_output_dirs()
    kind = args.checkpoint or cfg["cache"]["checkpoint"]; rows = []
    for model in method_items(cfg, args.models):
        for recipe_key, recipe_spec in recipe_items(cfg, args.recipes):
            row = inspect_artifacts(model, recipe_key, recipe_spec, kind, args.allow_checkpoint_fallback)
            run_dir = Path(row["run_dir"])
            if run_dir.is_dir():
                try:
                    resolve_run_config_path(run_dir)
                    _, flat = read_run_config(run_dir)
                    _, manifest = locate_or_rebuild_manifest(run_dir, RESULTS_DIR / "DatasetStats" / model["key"] / recipe_key)
                    validation = validate_recipe(recipe_key, recipe_spec, flat, manifest)
                    row.update(validation)
                    if validation["status"] != "ok" and row["status"] == "ok":
                        row["status"] = validation["status"]
                except Exception as exc:
                    row["manifest_status"] = f"{type(exc).__name__}: {exc}"
            if args.probe_load and row["status"] in {"ok", "config_recipe_mismatch"}:
                loaded = None
                try:
                    loaded = load_model(model, recipe_key, recipe_spec, checkpoint=kind,
                                        allow_fallback=args.allow_checkpoint_fallback, device=args.device)
                    row["probe_load"] = "ok"
                except Exception as exc:
                    row["probe_load"] = status_from_exception(exc); row["detail"] = f"{type(exc).__name__}: {exc}"
                finally:
                    if loaded is not None: loaded.close()
            if row["status"] not in {"ok", "config_recipe_mismatch"}:
                print(f"[SKIP] {model['label']} / {recipe_key} | {row['status']}")
            rows.append(row)
    path = RESULTS_DIR / "ModelInventory" / artifact_name("ModelInventory", rid, "csv")
    write_csv(path, rows); print(f"[OK] inventory: {path}"); return 0


if __name__ == "__main__": raise SystemExit(main())
