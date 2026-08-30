"""Cache-only channel-wise spectral-energy utilities.

The implementation intentionally follows the canonical FFT-shell approach in
``src/evaluate_ffm.py`` while adding explicit grid-validation and coordinate
mode decisions required for reproducible paper exports. It never interpolates
an arbitrary point cloud onto a grid.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class SpectralUnsupportedError(ValueError):
    """Raised when cached points cannot safely be treated as a 2D grid."""


def resolve_spectral_device(requested: str | None = None) -> str:
    """Return the requested execution backend without making CUDA mandatory.

    ``auto`` uses CUDA when the installed PyTorch build exposes it; otherwise
    the numerically equivalent NumPy implementation remains available.  This
    utility intentionally imports PyTorch lazily so CPU-only plotting and CSV
    readers never gain a PyTorch dependency.
    """
    choice = (requested or "cpu").strip().lower()
    if choice == "cpu":
        return "cpu"
    if choice == "auto":
        try:
            import torch
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if choice.startswith("cuda"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("CUDA spectral execution requires PyTorch.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested spectral device {requested!r}, but CUDA is unavailable.")
        return requested
    raise ValueError("Spectral device must be 'auto', 'cpu', or a CUDA device such as 'cuda:1'.")


def recover_structured_grid(
    coords_xy: np.ndarray,
    *,
    num_x: int | None = None,
    num_y: int | None = None,
    decimals: int = 8,
    coordinate_mode: str = "auto",
    spacing_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Recover a complete 2D ordering and choose physical/topological spacing.

    Explicit ``num_x``/``num_y`` metadata has priority when its product equals
    the number of cached points. Otherwise the function requires a complete
    rectilinear coordinate product; it does not interpolate scattered data.
    """
    coords = np.asarray(coords_xy, dtype=float)
    if coords.ndim != 2 or coords.shape[1] < 2 or not np.isfinite(coords[:, :2]).all():
        raise SpectralUnsupportedError("Coordinates must be a finite [N, >=2] array.")
    if coordinate_mode not in {"auto", "physical", "topological"}:
        raise ValueError("coordinate_mode must be auto, physical, or topological.")

    x = np.round(coords[:, 0], decimals=decimals)
    y = np.round(coords[:, 1], decimals=decimals)
    n_points = len(x)
    unique_x, unique_y = np.unique(x), np.unique(y)
    inferred_nx, inferred_ny = len(unique_x), len(unique_y)
    explicit_valid = bool(num_x and num_y and int(num_x) * int(num_y) == n_points)

    if explicit_valid:
        nx, ny = int(num_x), int(num_y)
        grid_source = "explicit_metadata"
    elif inferred_nx * inferred_ny == n_points:
        nx, ny = inferred_nx, inferred_ny
        grid_source = "coordinate_inference"
    else:
        raise SpectralUnsupportedError(
            "No complete structured grid: explicit Num_x/Num_y are unavailable "
            f"or invalid, while coordinate inference gives {inferred_ny}x{inferred_nx} for N={n_points}."
        )

    # Same stable y-then-x ordering used by evaluate_ffm.py. For coordinate
    # inference, also verify that every rectilinear coordinate pair is unique.
    sort_idx = np.lexsort((x, y))
    if grid_source == "coordinate_inference":
        pairs = np.stack([x, y], axis=1)
        if len(np.unique(pairs, axis=0)) != n_points:
            raise SpectralUnsupportedError("Coordinate grid has duplicate point locations.")

    x_grid = x[sort_idx].reshape(ny, nx)
    y_grid = y[sort_idx].reshape(ny, nx)
    dx_values = np.abs(np.diff(x_grid, axis=1)).ravel()
    dy_values = np.abs(np.diff(y_grid, axis=0)).ravel()
    dx_values = dx_values[dx_values > 0]
    dy_values = dy_values[dy_values > 0]
    if dx_values.size == 0 or dy_values.size == 0:
        raise SpectralUnsupportedError("Grid requires at least two non-degenerate points along both axes.")
    # Use the same mean unique-coordinate spacings as evaluate_ffm.py for
    # physical-mode compatibility. Uniformity itself is assessed from local
    # row/column spacings, not from this legacy-compatible summary value.
    physical_dx = float(np.mean(np.diff(unique_x))) if unique_x.size > 1 else float(np.mean(dx_values))
    physical_dy = float(np.mean(np.diff(unique_y))) if unique_y.size > 1 else float(np.mean(dy_values))
    dx_reference, dy_reference = float(np.median(dx_values)), float(np.median(dy_values))
    dx_spread = float(np.max(np.abs(dx_values - dx_reference)) / max(dx_reference, 1e-30))
    dy_spread = float(np.max(np.abs(dy_values - dy_reference)) / max(dy_reference, 1e-30))
    physical_uniform = dx_spread <= spacing_tolerance and dy_spread <= spacing_tolerance

    if coordinate_mode == "auto":
        chosen_mode = "physical" if physical_uniform else "topological"
    else:
        chosen_mode = coordinate_mode
    # Explicit physical mode is useful for reproducing the legacy evaluator:
    # it uses robust median spacings even on a mildly non-uniform grid. Auto
    # remains conservative and switches such grids to topological spacing.
    dx, dy = (physical_dx, physical_dy) if chosen_mode == "physical" else (1.0, 1.0)
    return {
        "nx": nx,
        "ny": ny,
        "sort_idx": sort_idx,
        "grid_source": grid_source,
        "coordinate_mode_requested": coordinate_mode,
        "coordinate_mode_used": chosen_mode,
        "physical_spacing_uniform": physical_uniform,
        "physical_dx": physical_dx,
        "physical_dy": physical_dy,
        "used_dx": dx,
        "used_dy": dy,
        "spacing_tolerance": float(spacing_tolerance),
        "dx_relative_spread": dx_spread,
        "dy_relative_spread": dy_spread,
    }


def reshape_flat_field(values: np.ndarray, grid: dict[str, Any]) -> np.ndarray:
    """Apply the recovered no-interpolation ordering to one flat channel."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size != grid["nx"] * grid["ny"]:
        raise SpectralUnsupportedError("Flat field length is incompatible with the recovered grid.")
    if not np.isfinite(values).all():
        raise SpectralUnsupportedError("Field contains non-finite values.")
    return values[grid["sort_idx"]].reshape(grid["ny"], grid["nx"])


def _preprocess(grid: np.ndarray, *, remove_mean: bool, window: str) -> tuple[np.ndarray, float]:
    if window not in {"none", "hann"}:
        raise ValueError("window must be none or hann.")
    values = np.asarray(grid, dtype=float).copy()
    if remove_mean:
        values -= np.mean(values)
    if window == "none":
        return values, 1.0
    win = np.outer(np.hanning(values.shape[0]), np.hanning(values.shape[1]))
    correction = float(np.mean(win ** 2))
    if correction <= 0:
        raise SpectralUnsupportedError("Hann window energy correction is degenerate.")
    return values * win, correction


def radial_spectrum(
    grid: np.ndarray,
    *,
    dx: float,
    dy: float,
    remove_mean: bool = True,
    window: str = "none",
    use_isotropic_cutoff: bool = True,
    min_shell_count: int = 4,
) -> dict[str, np.ndarray | float]:
    """Compute a native-spacing shell-averaged 2D channel power spectrum."""
    ny, nx = grid.shape
    if nx < 2 or ny < 2:
        raise SpectralUnsupportedError("FFT spectrum needs at least a 2x2 grid.")
    prepared, window_correction = _preprocess(grid, remove_mean=remove_mean, window=window)
    fft = np.fft.fftshift(np.fft.fft2(prepared))
    psd2 = (np.abs(fft) ** 2) / (nx * ny * window_correction)
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mag = np.hypot(kx_grid, ky_grid)
    dkx = float(np.min(np.abs(np.diff(np.unique(kx)))))
    dky = float(np.min(np.abs(np.diff(np.unique(ky)))))
    dk = max(min(dkx, dky), 1e-30)
    shell_id = np.rint(k_mag / dk).astype(np.int64)
    n_shells = int(shell_id.max()) + 1
    shell_sum = np.bincount(shell_id.ravel(), weights=psd2.ravel(), minlength=n_shells)
    shell_count = np.bincount(shell_id.ravel(), minlength=n_shells)
    shell_k_sum = np.bincount(shell_id.ravel(), weights=k_mag.ravel(), minlength=n_shells)
    k = shell_k_sum / np.maximum(shell_count, 1)
    energy = shell_sum / np.maximum(shell_count, 1)
    shell_index = np.arange(n_shells)
    valid = (shell_index > 0) & (shell_count >= int(min_shell_count))
    if use_isotropic_cutoff:
        valid &= k <= min(np.pi / dx, np.pi / dy) + 1e-12
    k, energy, shell_count, shell_index = k[valid], energy[valid], shell_count[valid], shell_index[valid]
    if k.size == 0:
        raise SpectralUnsupportedError("No non-zero FFT shells survived the configured validity filters.")
    total_energy = float(np.sum(energy * shell_count))
    normalized = energy / max(float(np.sum(energy)), 1e-30)
    return {
        "wavenumber": k,
        "spectral_energy": energy,
        "normalized_spectral_energy": normalized,
        "shell_count": shell_count,
        "wavenumber_index": shell_index,
        "total_energy": total_energy,
        "native_shell_spacing": dk,
        "window_energy_correction": window_correction,
    }


def _radial_spectra_cuda(
    grids: np.ndarray,
    *,
    dx: float,
    dy: float,
    remove_mean: bool,
    window: str,
    use_isotropic_cutoff: bool,
    min_shell_count: int,
    device: str,
) -> list[dict[str, np.ndarray | float]]:
    """Compute several same-grid spectra together with CUDA FFTs.

    Geometry and all reductions use the same native-shell definitions as
    :func:`radial_spectrum`.  Input and FFT arithmetic are float64/complex128
    so GPU results agree with the CPU reference to numerical tolerance.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - guarded by resolver
        raise RuntimeError("CUDA spectral execution requires PyTorch.") from exc

    values = np.asarray(grids, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 2 or values.shape[2] < 2:
        raise SpectralUnsupportedError("CUDA spectra require [channels, ny, nx] fields of at least 2x2.")
    if not np.isfinite(values).all():
        raise SpectralUnsupportedError("Field contains non-finite values.")
    n_channels, ny, nx = values.shape
    if window not in {"none", "hann"}:
        raise ValueError("window must be none or hann.")

    # All shell geometry is deterministic and small; constructing it in NumPy
    # keeps it bit-for-bit aligned with the CPU shell assignment.
    kx = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    kx_grid, ky_grid = np.meshgrid(kx, ky)
    k_mag = np.hypot(kx_grid, ky_grid)
    dkx = float(np.min(np.abs(np.diff(np.unique(kx)))))
    dky = float(np.min(np.abs(np.diff(np.unique(ky)))))
    dk = max(min(dkx, dky), 1e-30)
    shell_id = np.rint(k_mag / dk).astype(np.int64)
    n_shells = int(shell_id.max()) + 1
    shell_count = np.bincount(shell_id.ravel(), minlength=n_shells).astype(np.int64)
    shell_k_sum = np.bincount(shell_id.ravel(), weights=k_mag.ravel(), minlength=n_shells)
    k = shell_k_sum / np.maximum(shell_count, 1)
    shell_index = np.arange(n_shells)
    valid = (shell_index > 0) & (shell_count >= int(min_shell_count))
    if use_isotropic_cutoff:
        valid &= k <= min(np.pi / dx, np.pi / dy) + 1e-12
    if not np.any(valid):
        raise SpectralUnsupportedError("No non-zero FFT shells survived the configured validity filters.")

    tensor = torch.as_tensor(values, device=device, dtype=torch.float64)
    if remove_mean:
        tensor = tensor - tensor.mean(dim=(1, 2), keepdim=True)
    correction = 1.0
    if window == "hann":
        win = np.outer(np.hanning(ny), np.hanning(nx)).astype(np.float64)
        correction = float(np.mean(win ** 2))
        if correction <= 0:
            raise SpectralUnsupportedError("Hann window energy correction is degenerate.")
        tensor = tensor * torch.as_tensor(win, device=device, dtype=torch.float64)
    fft = torch.fft.fftshift(torch.fft.fft2(tensor), dim=(-2, -1))
    psd2 = fft.abs().square() / float(nx * ny * correction)
    flat_shell = torch.as_tensor(shell_id.ravel(), device=device, dtype=torch.long)
    expanded_shell = flat_shell.unsqueeze(0).expand(n_channels, -1)
    shell_sum = torch.zeros((n_channels, n_shells), dtype=torch.float64, device=device)
    shell_sum.scatter_add_(1, expanded_shell, psd2.reshape(n_channels, -1))
    energy = (shell_sum / torch.as_tensor(np.maximum(shell_count, 1), device=device, dtype=torch.float64)).cpu().numpy()

    output = []
    for channel_energy in energy:
        selected_energy = channel_energy[valid]
        selected_count = shell_count[valid]
        output.append({
            "wavenumber": k[valid],
            "spectral_energy": selected_energy,
            "normalized_spectral_energy": selected_energy / max(float(np.sum(selected_energy)), 1e-30),
            "shell_count": selected_count,
            "wavenumber_index": shell_index[valid],
            "total_energy": float(np.sum(selected_energy * selected_count)),
            "native_shell_spacing": dk,
            "window_energy_correction": correction,
        })
    return output


def band_energy_breakdown(wavenumber: np.ndarray, energy: np.ndarray) -> dict[str, np.ndarray | list[str]]:
    """Canonical low/mid/high thirds with trapezoidal shell-energy integration."""
    k, e = np.asarray(wavenumber, float), np.asarray(energy, float)
    if k.size == 0:
        return {"names": ["low", "mid", "high"], "edges": np.zeros(4), "energy": np.zeros(3), "ratio": np.zeros(3)}
    kmax = float(np.max(k)); edges = np.array([0.0, kmax / 3.0, 2.0 * kmax / 3.0, kmax])
    masks = [k <= edges[1], (k > edges[1]) & (k <= edges[2]), k > edges[2]]
    values = []
    for mask in masks:
        if np.count_nonzero(mask) >= 2:
            values.append(float(np.trapezoid(e[mask], k[mask])))
        elif np.count_nonzero(mask) == 1:
            values.append(float(e[mask][0]))
        else:
            values.append(0.0)
    values = np.asarray(values)
    return {"names": ["low", "mid", "high"], "edges": edges, "energy": values, "ratio": values / max(float(values.sum()), 1e-30)}


def compare_channel_spectra(
    truth_flat: np.ndarray,
    recon_flat: np.ndarray,
    coords_xy: np.ndarray,
    *,
    num_x: int | None = None,
    num_y: int | None = None,
    coordinate_mode: str = "auto",
    spacing_tolerance: float = 0.02,
    remove_mean: bool = True,
    window: str = "none",
    use_isotropic_cutoff: bool = True,
    min_shell_count: int = 4,
    relative_epsilon: float = 1e-12,
    device: str | None = None,
) -> dict[str, Any]:
    """Return common-shell spectra, dB/natural-log LSD, bands, and grid metadata."""
    return compare_channel_spectra_batch(
        np.asarray(truth_flat)[:, None], np.asarray(recon_flat)[:, None], coords_xy,
        num_x=num_x, num_y=num_y, coordinate_mode=coordinate_mode, spacing_tolerance=spacing_tolerance,
        remove_mean=remove_mean, window=window, use_isotropic_cutoff=use_isotropic_cutoff,
        min_shell_count=min_shell_count, relative_epsilon=relative_epsilon, device=device,
    )[0]


def compare_channel_spectra_batch(
    truth_fields: np.ndarray,
    recon_fields: np.ndarray,
    coords_xy: np.ndarray,
    *,
    num_x: int | None = None,
    num_y: int | None = None,
    coordinate_mode: str = "auto",
    spacing_tolerance: float = 0.02,
    remove_mean: bool = True,
    window: str = "none",
    use_isotropic_cutoff: bool = True,
    min_shell_count: int = 4,
    relative_epsilon: float = 1e-12,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Compare multiple physical channels sharing one recovered structured grid.

    Batching eliminates repeated coordinate sorting.  With a CUDA device, all
    channel FFTs and shell reductions execute together on the GPU; result
    arrays are returned on CPU for the existing CSV writers.
    """
    grid = recover_structured_grid(coords_xy, num_x=num_x, num_y=num_y, coordinate_mode=coordinate_mode, spacing_tolerance=spacing_tolerance)
    truth_values, recon_values = np.asarray(truth_fields, dtype=float), np.asarray(recon_fields, dtype=float)
    if truth_values.ndim != 2 or recon_values.ndim != 2 or truth_values.shape != recon_values.shape:
        raise SpectralUnsupportedError("Batched fields must be matching [N, channels] arrays.")
    if truth_values.shape[0] != grid["nx"] * grid["ny"]:
        raise SpectralUnsupportedError("Batched field length is incompatible with the recovered grid.")
    if not np.isfinite(truth_values).all() or not np.isfinite(recon_values).all():
        raise SpectralUnsupportedError("Field contains non-finite values.")
    truth_grid = truth_values[grid["sort_idx"]].T.reshape(-1, grid["ny"], grid["nx"])
    recon_grid = recon_values[grid["sort_idx"]].T.reshape(-1, grid["ny"], grid["nx"])
    options = dict(dx=grid["used_dx"], dy=grid["used_dy"], remove_mean=remove_mean, window=window,
                   use_isotropic_cutoff=use_isotropic_cutoff, min_shell_count=min_shell_count)
    backend = resolve_spectral_device(device)
    if backend == "cpu":
        truth_spectra = [radial_spectrum(channel, **options) for channel in truth_grid]
        recon_spectra = [radial_spectrum(channel, **options) for channel in recon_grid]
    else:
        truth_spectra = _radial_spectra_cuda(truth_grid, **options, device=backend)
        recon_spectra = _radial_spectra_cuda(recon_grid, **options, device=backend)

    results = []
    for truth, recon in zip(truth_spectra, recon_spectra):
        # Geometry is shared, but align defensively by native shell index.
        common, truth_pos, recon_pos = np.intersect1d(truth["wavenumber_index"], recon["wavenumber_index"], return_indices=True)
        if common.size == 0:
            raise SpectralUnsupportedError("Truth and reconstruction have no common valid FFT shells.")
        for payload, positions in ((truth, truth_pos), (recon, recon_pos)):
            for key in ("wavenumber", "spectral_energy", "normalized_spectral_energy", "shell_count", "wavenumber_index"):
                payload[key] = np.asarray(payload[key])[positions]
            payload["total_energy"] = float(np.sum(payload["spectral_energy"] * payload["shell_count"]))
            payload["normalized_spectral_energy"] = payload["spectral_energy"] / max(float(np.sum(payload["spectral_energy"])), 1e-30)
        eps = max(1e-30, float(relative_epsilon) * float(np.max(truth["spectral_energy"])))
        log_ratio = np.log(recon["spectral_energy"] + eps) - np.log(truth["spectral_energy"] + eps)
        lsd_loge = float(np.sqrt(np.mean(log_ratio ** 2)))
        lsd_db = float(np.sqrt(np.mean((10.0 * np.log10((recon["spectral_energy"] + eps) / (truth["spectral_energy"] + eps))) ** 2)))
        bands_true = band_energy_breakdown(truth["wavenumber"], truth["spectral_energy"])
        bands_pred = band_energy_breakdown(recon["wavenumber"], recon["spectral_energy"])
        band_ratio = bands_pred["energy"] / np.maximum(bands_true["energy"], eps)
        band_rel_error = np.abs(bands_pred["energy"] - bands_true["energy"]) / np.maximum(bands_true["energy"], eps)
        results.append({
            "truth": truth, "reconstruction": recon, "grid": {**grid, "execution_backend": backend},
            "lsd_db": lsd_db, "lsd_loge": lsd_loge, "epsilon": eps,
            "total_energy_ratio": float(recon["total_energy"] / max(float(truth["total_energy"]), eps)),
            "bands_true": bands_true, "bands_reconstruction": bands_pred,
            "band_energy_ratio": band_ratio, "band_rel_error": band_rel_error,
        })
    return results
