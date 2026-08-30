"""Shared global joint-PDF binning utilities."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .cache import load_cache

# The ordered pairs determine the physical x/y axes of a joint PDF.  Keep the
# legacy ``T-CO`` key for prior artefacts, and expose the publication-facing
# CO--T orientation explicitly for the coupling-validation panels.
PAIR_FIELDS = {
    "CH4-T": (0, 2),
    "CH4-U1": (0, 3),
    "T-CO": (2, 1),
    "T-U1": (2, 3),
    "CO-U1": (1, 3),
    "CO-T": (1, 2),
    "U1-p": (3, 4),
    "p-U1": (4, 3),
}


def global_edges(manifest_rows, pairs, bins=64, quantiles=(.005, .995)):
    pools = {pair: [[], []] for pair in pairs}
    # Truth is identical across methods and conditions. Count every test
    # snapshot exactly once so missing methods cannot reweight global limits.
    seen = set()
    for row in manifest_rows:
        path = row.get("cache_path", "")
        truth_key = (row.get("split", "test"), row.get("snapshot"))
        if row.get("status") != "ok" or not path or truth_key in seen: continue
        seen.add(truth_key)
        try: arrays, _ = load_cache(Path(path))
        except Exception: continue
        for pair in pairs:
            a, b = PAIR_FIELDS[pair]; pools[pair][0].append(arrays["truth_phys"][:, a]); pools[pair][1].append(arrays["truth_phys"][:, b])
    edges = {}
    for pair, (xs, ys) in pools.items():
        if not xs: edges[pair] = (np.linspace(0, 1, bins + 1), np.linspace(0, 1, bins + 1)); continue
        x, y = np.concatenate(xs), np.concatenate(ys)
        xlim = np.quantile(x[np.isfinite(x)], quantiles); ylim = np.quantile(y[np.isfinite(y)], quantiles)
        edges[pair] = (np.linspace(*xlim, bins + 1), np.linspace(*ylim, bins + 1))
    return edges


def histogram(x, y, edges):
    h, _, _ = np.histogram2d(x, y, bins=edges)
    return h / max(h.sum(), 1)
