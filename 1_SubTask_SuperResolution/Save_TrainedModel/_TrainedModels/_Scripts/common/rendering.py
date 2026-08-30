"""Shared native-cell, scatter, smooth, and hybrid field rendering."""
from __future__ import annotations
import numpy as np
import matplotlib.tri as mtri
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from .workflow import grid_order


def automatic_gradient_roi(coords, truth, fraction=.25):
    """Backward-compatible alias for the integrated-gradient ROI selector."""
    return automatic_integrated_gradient_roi(coords, truth, fraction=fraction)


def automatic_integrated_gradient_roi(coords, truth, fraction=.25):
    """Choose the fixed-size window with maximum integrated truth gradient.

    Selection depends only on the ground-truth field.  A summed-area table
    evaluates every valid window without reference to any model prediction.
    """
    order, ny, nx = grid_order(coords)
    c = np.asarray(coords)[order]
    z = np.asarray(truth).reshape(-1)[order].reshape(ny, nx)
    gy, gx = np.gradient(z)
    score = np.hypot(gx, gy).astype(np.float64, copy=False)
    win_x = min(nx, max(4, int(round(nx * float(fraction)))))
    win_y = min(ny, max(4, int(round(ny * float(fraction)))))
    integral = np.pad(score, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (
        integral[win_y:, win_x:] - integral[:-win_y, win_x:]
        - integral[win_y:, :-win_x] + integral[:-win_y, :-win_x]
    )
    iy, ix = np.unravel_index(int(np.nanargmax(sums)), sums.shape)
    x = c[:, 0].reshape(ny, nx)
    y = c[:, 1].reshape(ny, nx)
    ys = slice(iy, iy + win_y)
    xs = slice(ix, ix + win_x)
    return [
        float(x[ys, xs].min()), float(x[ys, xs].max()),
        float(y[ys, xs].min()), float(y[ys, xs].max()),
    ]


def automatic_model_contrast_roi(
    coords, truth, prediction_a, prediction_b, fraction=.25,
    min_truth_gradient_quantile=.5,
):
    """Choose a structured window with strong, direction-neutral model contrast.

    The score is the integrated absolute difference between two predictions.
    To avoid selecting a visually flat patch, eligible windows must also meet a
    configurable quantile of integrated ground-truth gradient magnitude.  The
    rule is symmetric in the two predictions: it does not favor either model.
    """
    order, ny, nx = grid_order(coords)
    c = np.asarray(coords)[order]
    truth_grid = np.asarray(truth).reshape(-1)[order].reshape(ny, nx)
    pred_a = np.asarray(prediction_a).reshape(-1)[order].reshape(ny, nx)
    pred_b = np.asarray(prediction_b).reshape(-1)[order].reshape(ny, nx)
    win_x = min(nx, max(4, int(round(nx * float(fraction)))))
    win_y = min(ny, max(4, int(round(ny * float(fraction)))))

    def window_sums(values):
        integral = np.pad(values, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        return (
            integral[win_y:, win_x:] - integral[:-win_y, win_x:]
            - integral[win_y:, :-win_x] + integral[:-win_y, :-win_x]
        )

    contrast_sums = window_sums(np.abs(pred_a - pred_b).astype(np.float64, copy=False))
    gy, gx = np.gradient(truth_grid)
    gradient_sums = window_sums(np.hypot(gx, gy).astype(np.float64, copy=False))
    quantile = float(np.clip(min_truth_gradient_quantile, 0.0, 1.0))
    threshold = float(np.nanquantile(gradient_sums, quantile))
    eligible = np.isfinite(contrast_sums) & (gradient_sums >= threshold)
    if not np.any(eligible):
        eligible = np.isfinite(contrast_sums)
    scored = np.where(eligible, contrast_sums, -np.inf)
    iy, ix = np.unravel_index(int(np.argmax(scored)), scored.shape)

    x = c[:, 0].reshape(ny, nx)
    y = c[:, 1].reshape(ny, nx)
    ys = slice(iy, iy + win_y)
    xs = slice(ix, ix + win_x)
    return [
        float(x[ys, xs].min()), float(x[ys, xs].max()),
        float(y[ys, xs].min()), float(y[ys, xs].max()),
    ]


def render_field(ax, coords, values, *, mode="smooth", cmap="viridis", vmin=None, vmax=None,
                 marker_size=3.0, roi=None, inset=False):
    c = np.asarray(coords); z = np.asarray(values).reshape(-1)
    if mode in {"scatter", "native_scatter"}:
        artist = ax.scatter(c[:, 0], c[:, 1], c=z, s=marker_size, marker="s", linewidths=0,
                            cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    else:
        order, ny, nx = grid_order(c); x = c[order, 0].reshape(ny, nx); y = c[order, 1].reshape(ny, nx); grid = z[order].reshape(ny, nx)
        if mode == "native_cells":
            artist = ax.pcolormesh(x, y, grid, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        else:
            artist = ax.contourf(x, y, grid, levels=64, cmap=cmap, vmin=vmin, vmax=vmax)
        if mode == "hybrid" or inset:
            roi = roi or [float(np.quantile(x, .35)), float(np.quantile(x, .65)), float(np.quantile(y, .35)), float(np.quantile(y, .65))]
            iax = inset_axes(ax, width="38%", height="38%", loc="lower right", borderpad=.4)
            iax.pcolormesh(x, y, grid, shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
            iax.set_xlim(roi[0], roi[1]); iax.set_ylim(roi[2], roi[3]); iax.set_xticks([]); iax.set_yticks([])
            mark_inset(ax, iax, loc1=1, loc2=3, fc="none", ec="#606060", lw=.6)
    ax.set_aspect("equal"); ax.set_xlabel("x"); ax.set_ylabel("y")
    return artist
