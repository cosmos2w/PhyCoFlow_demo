"""Physical/normalized metrics and robust summary statistics."""
from __future__ import annotations
import numpy as np


def relative_l2(truth, recon, mask=None) -> float:
    truth = np.asarray(truth, float); recon = np.asarray(recon, float)
    valid = np.isfinite(truth) & np.isfinite(recon)
    if mask is not None:
        valid &= np.asarray(mask, bool)
    if not valid.any():
        return float("nan")
    # Avoid dispatching two small vectors through a multithreaded BLAS.  This
    # exporter is called thousands of times, where BLAS thread-launch overhead
    # dominates the actual reduction.  The expression is algebraically the
    # same relative Euclidean norm and keeps the calculation in NumPy's
    # elementwise/reduction path.
    diff = recon[valid] - truth[valid]
    numerator = np.sqrt(np.sum(diff * diff, dtype=np.float64))
    denominator = np.sqrt(np.sum(truth[valid] * truth[valid], dtype=np.float64))
    return float(numerator / (denominator + 1e-12))


def summarize(values, seed=42, n_boot=2000) -> dict:
    x = np.asarray(values, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return {k: float("nan") for k in ("mean", "std", "median", "q25", "q75", "ci95_low", "ci95_high")} | {"valid_n": 0}
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(x, size=(n_boot, x.size), replace=True), axis=1)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "median": float(np.median(x)), "q25": float(np.quantile(x, .25)), "q75": float(np.quantile(x, .75)),
            "ci95_low": float(np.quantile(means, .025)), "ci95_high": float(np.quantile(means, .975)), "valid_n": int(x.size)}


def jsd_base2(p, q, pseudocount=1e-12) -> float:
    p = np.asarray(p, float).ravel() + pseudocount; q = np.asarray(q, float).ravel() + pseudocount
    p /= p.sum(); q /= q.sum(); m = .5 * (p + q)
    return float(.5 * np.sum(p * np.log2(p / m)) + .5 * np.sum(q * np.log2(q / m)))
