#!/usr/bin/env python
"""Inventory every expected method-condition pair without stopping on gaps."""
from __future__ import annotations
import argparse
from common.config import RESULTS_DIR, add_common_args, ensure_output_dirs, load_config, method_items, run_id
from common.io_utils import artifact_name, write_csv
from common.model_loader import inspect_artifacts, load_model, status_from_exception


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_common_args(parser)
    parser.add_argument("--checkpoint", choices=["last", "best"], default=None)
    parser.add_argument("--allow-checkpoint-fallback", action="store_true")
    parser.add_argument("--probe-load", action="store_true", help="Actually rebuild/load each available model.")
    parser.add_argument("--device", default=None)
    args = parser.parse_args(); cfg = load_config(args.config); rid = run_id(args.run_id); ensure_output_dirs()
    checkpoint = args.checkpoint or cfg["defaults"]["checkpoint"]
    rows = []
    for method in method_items(cfg, args.models):
        for condition in cfg["conditions"]:
            row = inspect_artifacts(method, condition, checkpoint, args.allow_checkpoint_fallback)
            if args.probe_load and row["status"] == "ok":
                try:
                    loaded = load_model(method, condition, checkpoint=checkpoint,
                        allow_fallback=args.allow_checkpoint_fallback, split=cfg["defaults"]["split"], device=args.device)
                    loaded.close()
                except Exception as exc:
                    row["status"] = status_from_exception(exc); row["detail"] = f"{type(exc).__name__}: {exc}"
                    print(f"[ERROR] {method['name']} / {condition} | {row['detail']}")
            elif row["status"] != "ok":
                print(f"[SKIP] {method['name']} / {condition} | {row['status']} {row['detail']}")
            rows.append(row)
    path = RESULTS_DIR / "ModelInventory" / artifact_name("ModelInventory", rid, "csv")
    write_csv(path, rows)
    print("\nMethod         Cond_T             Cond_TU1           Cond_COTU1P")
    by_key = {(r["method"], r["condition"]): r["status"] for r in rows}
    for method in method_items(cfg, args.models):
        print(f"{method['name']:<14}" + "".join(f"{by_key[(method['name'], c)]:<19}" for c in cfg["conditions"]))
    print(f"\n[OK] inventory: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

