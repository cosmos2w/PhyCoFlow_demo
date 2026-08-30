"""Cache-only analysis helpers shared by all paper exporters."""
from __future__ import annotations
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable
import numpy as np

from .cache import load_cache
from .config import RESULTS_DIR, stable_seed
from .io_utils import latest, read_csv
from .statistics import relative_l2, summarize


_GRID_ORDER_CACHE = {}


def cache_manifest_path(value: str | Path | None = None) -> Path:
    return Path(value) if value else latest(RESULTS_DIR / "ReconstructionCache", "ReconstructionCache_manifest", "csv")


def cache_entries(path=None, *, models=None, recipes=None, sensor_count=None, snapshots=None):
    rows = read_csv(cache_manifest_path(path)); models = None if not models or "all" in models else set(models)
    recipes = None if not recipes or "all" in recipes else set(recipes); snapshots = None if snapshots is None else set(map(int, snapshots))
    out = []
    for row in rows:
        if row.get("status") != "ok" or not row.get("cache_path"): continue
        if models is not None and row.get("model") not in models: continue
        if recipes is not None and row.get("recipe") not in recipes: continue
        if sensor_count is not None and int(row.get("sensor_count", -1)) != int(sensor_count): continue
        if snapshots is not None and int(row.get("snapshot_index", -1)) not in snapshots: continue
        out.append(row)
    return out


def grid_order(coords, nx=None, ny=None):
    c = np.asarray(coords)
    # Formal comparisons share one canonical evaluation grid.  Keying the
    # verified ordering by shape, declared dimensions, and boundary points
    # avoids repeating the same lexsort/unique work for every cache row while
    # still separating genuinely different native grids.
    key = (
        tuple(c.shape), int(nx or 0), int(ny or 0),
        tuple(np.asarray(c[0], dtype=float)), tuple(np.asarray(c[-1], dtype=float)),
    )
    cached = _GRID_ORDER_CACHE.get(key)
    if cached is not None:
        return cached
    order = np.lexsort((c[:, 0], c[:, 1]))
    ux, uy = np.unique(np.round(c[:, 0], 8)), np.unique(np.round(c[:, 1], 8))
    nx = int(nx or len(ux)); ny = int(ny or len(uy))
    if nx * ny != len(c): raise ValueError(f"Not a complete structured grid: {ny}x{nx} != {len(c)}")
    result = (order, ny, nx)
    _GRID_ORDER_CACHE[key] = result
    return result


def _ssim(truth, pred, shape):
    try:
        from skimage.metrics import structural_similarity
        a, b = np.asarray(truth).reshape(shape), np.asarray(pred).reshape(shape)
        return float(structural_similarity(a, b, data_range=max(float(a.max()-a.min()), 1e-12)))
    except Exception:
        return float("nan")


def metric_row(entry):
    arrays, meta = load_cache(Path(entry["cache_path"]))
    truth_p = arrays["truth_phys"].reshape(-1); pred_p = arrays["recon_phys"].reshape(-1)
    truth_n = arrays["truth_norm"].reshape(-1); pred_n = arrays["recon_norm"].reshape(-1)
    excluded = np.ones(truth_p.size, dtype=bool); excluded[arrays["obs_indices"].astype(int)] = False
    order, ny, nx = grid_order(arrays["coords_phys"], meta.get("num_x"), meta.get("num_y"))
    tg = truth_p[order].reshape(ny, nx); pg = pred_p[order].reshape(ny, nx)
    gy_t, gx_t = np.gradient(tg); gy_p, gx_p = np.gradient(pg)
    grad_truth = np.hypot(gx_t, gy_t); grad_pred = np.hypot(gx_p, gy_p)
    return {
        **{k: meta.get(k, entry.get(k, "")) for k in ("model", "model_label", "recipe", "recipe_label", "case_id", "time_index", "physical_time", "snapshot_index", "sensor_count", "sensor_plan_id", "checkpoint_kind", "checkpoint_hash", "nfe", "ode_solver", "consistency_mode", "generation_seed")},
        "physical_rel_l2": relative_l2(truth_p, pred_p),
        "physical_rel_l2_sensor_excluded": relative_l2(truth_p, pred_p, excluded),
        "normalized_rel_l2": relative_l2(truth_n, pred_n),
        "SSIM": _ssim(tg, pg, (ny, nx)), "gradient_rel_l2": relative_l2(grad_truth, grad_pred),
        "status": "ok", "metadata": entry["cache_path"],
    }


def grouped_summary(rows, group_keys: Iterable[str], value_key: str, seed=42, n_boot=2000):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in group_keys)].append(float(row[value_key]))
    out = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        stats = summarize(values, seed=stable_seed(seed, *key), n_boot=n_boot)
        out.append({**dict(zip(group_keys, key)), "metric": value_key, **stats})
    return out


def missing_rows(cfg, models, recipes, snapshots, sensor_count, reason="missing_cache"):
    rows = []
    for model in models:
        for recipe in recipes:
            for snapshot in snapshots:
                rows.append({"model": model, "recipe": recipe, "snapshot_index": snapshot, "sensor_count": sensor_count,
                             "physical_rel_l2": float("nan"), "status": reason})
    return rows
