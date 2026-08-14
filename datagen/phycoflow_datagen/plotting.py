"""Headless Python QA figures for raw NPZ and processed HDF5 data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .cases import CASES
from .diagnostics import (
    compute_diagnostics,
    derive_kolmogorov_fields,
    vorticity_from_velocity,
)
from .storage import load_raw_trajectory


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def _decode_json(value) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return json.loads(str(value))


def _unpack_h5_fields(dataset, trajectory: int, grid_shape: list[int]) -> np.ndarray:
    layout = dataset[trajectory, :, :, 0, 0, :]
    channel_last = layout.reshape(layout.shape[0], *grid_shape, layout.shape[-1])
    return np.moveaxis(channel_last, -1, 1)


def load_visualization_data(case: str, input_path: Path, trajectory: int):
    if input_path.suffix == ".npz":
        arrays, metadata = load_raw_trajectory(input_path)
        if metadata["case"] != case:
            raise ValueError(f"expected {case} data, found {metadata['case']}")
        config = metadata["config"]
        config = dict(config)
        config.update(metadata.get("conditions") or {})
        state = arrays["state"]
        fields = derive_kolmogorov_fields(state, config) if case == "kolmogorov" else state
        return {
            "fields": fields,
            "state": state,
            "time": arrays["time"],
            "x": arrays["x"],
            "y": arrays.get("y"),
            "config": config,
            "source": str(input_path),
            "auxiliary": arrays,
            "stored_diagnostics": metadata.get("diagnostics", {}),
        }
    if input_path.suffix not in {".h5", ".hdf5"}:
        raise ValueError("--input must be a raw .npz trajectory or processed .h5 file")

    with h5py.File(input_path, "r") as handle:
        stored_case = str(handle.attrs["case_name"])
        if stored_case != case:
            raise ValueError(f"expected {case} data, found {stored_case}")
        if not 0 <= trajectory < handle["fields"].shape[0]:
            raise IndexError(f"trajectory must be in [0, {handle['fields'].shape[0] - 1}]")
        grid_shape = _decode_json(handle.attrs["grid_shape"])
        fields = _unpack_h5_fields(handle["fields"], trajectory, grid_shape)
        metadata = _decode_json(handle["metadata/json"][()])
        config = dict(metadata["resolved_config"])
        condition_names = _decode_json(handle["conditions"].attrs["condition_names"])
        condition_values = handle["conditions"][trajectory]
        config.update(
            {name: float(value) for name, value in zip(condition_names, condition_values)}
        )
        coordinates = handle["coordinates"][:, 0, 0, :]
        if len(grid_shape) == 1:
            x = coordinates[:, 0]
            y = None
        else:
            coordinate_grid = coordinates.reshape(*grid_shape, 3)
            x = coordinate_grid[0, :, 0]
            y = coordinate_grid[:, 0, 1]
        if case == "kolmogorov" and "auxiliary/vorticity" in handle:
            state = _unpack_h5_fields(handle["auxiliary/vorticity"], trajectory, grid_shape)
        elif case == "kolmogorov":
            state = vorticity_from_velocity(fields, config["domain_length"])[:, None]
        else:
            state = fields
        auxiliary: dict[str, np.ndarray] = {}
        if "auxiliary" in handle:
            for name, dataset in handle["auxiliary"].items():
                values = dataset[trajectory]
                if values.ndim == 4:  # [N,1,1,C]
                    channel_last = values[:, 0, 0, :].reshape(*grid_shape, values.shape[-1])
                    auxiliary[name] = np.moveaxis(channel_last, -1, 0)
                elif values.ndim == 5:  # [T,N,1,1,C]
                    channel_last = values[:, :, 0, 0, :].reshape(
                        values.shape[0], *grid_shape, values.shape[-1]
                    )
                    auxiliary[name] = np.moveaxis(channel_last, -1, 1)
        stored_diagnostics = {
            name: (
                bool(dataset[trajectory])
                if name == "finite"
                else float(dataset[trajectory])
            )
            for name, dataset in handle.get("diagnostics", {}).items()
        }
        return {
            "fields": fields,
            "state": state,
            "time": handle["time"][:],
            "x": x,
            "y": y,
            "config": config,
            "source": str(input_path),
            "auxiliary": auxiliary,
            "stored_diagnostics": stored_diagnostics,
        }


def _radial_spectrum(field: np.ndarray):
    ny, nx = field.shape
    mx, my = np.meshgrid(np.fft.fftfreq(nx) * nx, np.fft.fftfreq(ny) * ny, indexing="xy")
    radius_index = np.sqrt(mx**2 + my**2).astype(int)
    power = np.abs(np.fft.fft2(field - field.mean())) ** 2 / field.size**2
    total = np.bincount(radius_index.ravel(), weights=power.ravel())
    count = np.maximum(np.bincount(radius_index.ravel()), 1)
    return np.arange(total.size), total / count


def _style_axis_grid(axes) -> None:
    for axis in axes.ravel():
        axis.tick_params(direction="out", length=2.5, width=0.7)


def _plot_burgers_or_ks(case: str, data: dict[str, Any], time_index: int):
    time, x = data["time"], data["x"]
    u_all = data["fields"][:, 0]
    u = u_all[time_index]
    length = data["config"]["domain_length"]
    wave = 2.0 * np.pi * np.fft.fftfreq(x.size, d=length / x.size)
    gradient = np.fft.ifft(1j * wave * np.fft.fft(u)).real
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    image = axes[0, 0].imshow(
        u_all,
        origin="lower",
        aspect="auto",
        extent=(x[0], x[0] + length, time[0], time[-1]),
        cmap="RdBu_r",
    )
    axes[0, 0].axhline(time[time_index], color="black", lw=0.8)
    axes[0, 0].set(xlabel="x", ylabel="t", title="Space–time field")
    figure.colorbar(image, ax=axes[0, 0], shrink=0.75, label="u")
    axes[0, 1].plot(x, u, color="#3377AA", lw=1.2)
    axes[0, 1].set(xlabel="x", ylabel="u", xlim=(x[0], x[0] + length), title=f"Field at t={time[time_index]:.3g}")
    axes[0, 2].hist(u, bins=30, density=True, color="#66CCEE", alpha=0.85)
    axes[0, 2].set(xlabel="u", ylabel="density", title="Value distribution")
    power = np.abs(np.fft.rfft(u - u.mean())) ** 2 / u.size**2
    axes[1, 0].semilogy(np.arange(power.size)[1:], power[1:] + 1.0e-20, color="#EE7733")
    axes[1, 0].axvline(u.size / 3, color="0.4", ls="--", lw=0.8, label="2/3 cutoff")
    axes[1, 0].set(xlabel="mode", ylabel="power", title="Power spectrum")
    axes[1, 0].legend()
    threshold = np.median(u)
    axes[1, 1].fill_between(x, 0, u > threshold, step="mid", color="#AA4499", alpha=0.7)
    axes[1, 1].set(xlabel="x", ylim=(0, 1.05), title="Median superlevel set")
    if case == "burgers":
        axes[1, 2].plot(x, gradient, color="#CC3311", lw=1.0)
        axes[1, 2].set(xlabel="x", ylabel=r"$u_x$", title="Shock gradient")
    else:
        axes[1, 2].plot(u, gradient, color="#009988", lw=0.9)
        axes[1, 2].set(xlabel="u", ylabel=r"$u_x$", title="Spatial phase portrait")
    _style_axis_grid(axes)
    return figure


def _plot_brusselator(data: dict[str, Any], time_index: int):
    fields = data["fields"]
    u, v = fields[time_index]
    time = data["time"]
    length = data["config"]["domain_length"]
    extent = (0.0, length, 0.0, length)
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    image_u = axes[0, 0].imshow(u, origin="lower", extent=extent, cmap="magma")
    axes[0, 0].set(title=f"u at t={time[time_index]:.3g}", xlabel="x", ylabel="y")
    figure.colorbar(image_u, ax=axes[0, 0], shrink=0.75)
    image_v = axes[0, 1].imshow(v, origin="lower", extent=extent, cmap="viridis")
    axes[0, 1].set(title=f"v at t={time[time_index]:.3g}", xlabel="x", ylabel="y")
    figure.colorbar(image_v, ax=axes[0, 1], shrink=0.75)
    stride = max(1, u.size // 2500)
    axes[0, 2].scatter(u.ravel()[::stride], v.ravel()[::stride], s=5, alpha=0.35, color="#4477AA")
    axes[0, 2].set(xlabel="u", ylabel="v", title="Joint field-value cloud")
    axes[1, 0].hist(u.ravel(), bins=30, density=True, alpha=0.6, label="u", color="#EE7733")
    axes[1, 0].hist(v.ravel(), bins=30, density=True, alpha=0.6, label="v", color="#0077BB")
    axes[1, 0].set(xlabel="value", ylabel="density", title="Marginal distributions")
    axes[1, 0].legend()
    ku, su = _radial_spectrum(u)
    kv, sv = _radial_spectrum(v)
    axes[1, 1].semilogy(ku[1:], su[1:] + 1.0e-20, label="u", color="#EE7733")
    axes[1, 1].semilogy(kv[1:], sv[1:] + 1.0e-20, label="v", color="#0077BB")
    axes[1, 1].axvline(u.shape[-1] / 3, color="0.4", ls="--", lw=0.8)
    axes[1, 1].set(xlabel="radial mode", ylabel="power", title="Per-field radial spectra")
    axes[1, 1].legend()
    axes[1, 2].contour(u, levels=[np.median(u)], colors="#CC3311", extent=extent)
    axes[1, 2].contour(v, levels=[np.median(v)], colors="#0077BB", extent=extent)
    axes[1, 2].set(xlabel="x", ylabel="y", title="Median level-set geometry")
    _style_axis_grid(axes)
    return figure


def _plot_kolmogorov(data: dict[str, Any], time_index: int):
    fields = data["fields"]
    u, v, pressure = fields[time_index]
    omega = data["state"][time_index, 0]
    time = data["time"]
    length = data["config"]["domain_length"]
    extent = (0.0, length, 0.0, length)
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    for axis, field, name in zip(axes[0], (u, v, pressure), ("u", "v", "p")):
        image = axis.imshow(field, origin="lower", extent=extent, cmap="RdBu_r")
        axis.set(title=f"{name} at t={time[time_index]:.3g}", xlabel="x", ylabel="y")
        figure.colorbar(image, ax=axis, shrink=0.75)
    stride = max(1, u.size // 2500)
    cloud = axes[1, 0].scatter(
        u.ravel()[::stride],
        v.ravel()[::stride],
        c=pressure.ravel()[::stride],
        s=5,
        alpha=0.45,
        cmap="coolwarm",
    )
    axes[1, 0].set(xlabel="u", ylabel="v", title="Joint velocity cloud; color=p")
    figure.colorbar(cloud, ax=axes[1, 0], shrink=0.75, label="p")
    radius_u, spectrum_u = _radial_spectrum(u)
    radius_v, spectrum_v = _radial_spectrum(v)
    energy = 0.5 * (spectrum_u + spectrum_v)
    axes[1, 1].loglog(radius_u[1:], energy[1:] + 1.0e-20, color="black")
    axes[1, 1].axvline(u.shape[-1] / 3, color="0.4", ls="--", lw=0.8)
    axes[1, 1].set(xlabel="radial mode", ylabel="kinetic energy", title="Energy spectrum")
    levels = np.unique(np.quantile(omega, [0.2, 0.4, 0.6, 0.8]))
    if levels.size:
        axes[1, 2].contour(omega, levels=levels, colors="black", linewidths=0.6, extent=extent)
    image_omega = axes[1, 2].imshow(omega, origin="lower", extent=extent, cmap="Spectral_r", alpha=0.85)
    axes[1, 2].set(xlabel="x", ylabel="y", title="Vorticity level-set geometry")
    figure.colorbar(image_omega, ax=axes[1, 2], shrink=0.75, label=r"$\omega$")
    _style_axis_grid(axes)
    return figure


def _windowed_radial_spectrum(field: np.ndarray):
    """Shell-average a Hann-windowed nonperiodic field without claiming periodic modes."""

    window = np.outer(np.hanning(field.shape[0]), np.hanning(field.shape[1]))
    return _radial_spectrum((field - field.mean()) * window)


def _plot_electro_thermal(data: dict[str, Any], time_index: int):
    electric_real, electric_imag, temperature = data["fields"][time_index]
    electric = electric_real + 1j * electric_imag
    magnitude = np.abs(electric)
    x, y = data["x"], data["y"]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    extent = (x[0], x[-1], y[0], y[-1])
    config = data["config"]
    auxiliary = data.get("auxiliary", {})
    ellipse_mask = np.squeeze(auxiliary.get("ellipse_mask", np.zeros_like(temperature))).astype(bool)
    if not ellipse_mask.any():
        cosine, sine = np.cos(config["phi"]), np.sin(config["phi"])
        rotated_x = cosine * xx + sine * yy
        rotated_y = -sine * xx + cosine * yy
        ellipse_mask = (
            (rotated_x / config["a"]) ** 2 + (rotated_y / config["b"]) ** 2 <= 1.0
        )
    joule_heating = auxiliary.get("joule_heating")
    if joule_heating is None:
        sigma = np.where(
            ellipse_mask,
            1.602
            * config["Sigma_Si"]
            * np.exp(-1.12 / (8.6173e-5 * temperature)),
            config["conductivity_alumina"],
        )
        joule_heating = 0.5 * sigma * magnitude**2
    joule_heating = np.squeeze(joule_heating)
    temperature_rise = temperature - config["ambient_temperature"]

    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)
    electric_limit = max(np.max(np.abs(electric_real)), np.max(np.abs(electric_imag)))
    for axis, field, title in zip(
        axes[0, :2],
        (electric_real, electric_imag),
        (r"Real electric field $\mathrm{Re}(E_z)$", r"Imaginary electric field $\mathrm{Im}(E_z)$"),
    ):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-electric_limit,
            vmax=electric_limit,
        )
        axis.contour(xx, yy, ellipse_mask.astype(float), levels=[0.5], colors="black", linewidths=0.8)
        axis.set(xlabel="x [m]", ylabel="y [m]", title=title)
        figure.colorbar(image, ax=axis, shrink=0.76, label="V/m")

    temperature_image = axes[0, 2].imshow(
        temperature_rise, origin="lower", extent=extent, cmap="inferno"
    )
    axes[0, 2].contour(
        xx, yy, ellipse_mask.astype(float), levels=[0.5], colors="white", linewidths=0.8
    )
    axes[0, 2].set(xlabel="x [m]", ylabel="y [m]", title="Temperature rise")
    figure.colorbar(temperature_image, ax=axes[0, 2], shrink=0.76, label="K above ambient")

    magnitude_image = axes[1, 0].imshow(
        magnitude, origin="lower", extent=extent, cmap="cividis"
    )
    axes[1, 0].contour(
        xx, yy, ellipse_mask.astype(float), levels=[0.5], colors="white", linewidths=1.0
    )
    axes[1, 0].contour(
        xx, yy, temperature_rise, levels=5, colors="#CC6677", linewidths=0.65
    )
    axes[1, 0].set(
        xlabel="x [m]",
        ylabel="y [m]",
        title=r"Electric magnitude, ellipse, and $\Delta T$ contours",
    )
    figure.colorbar(magnitude_image, ax=axes[1, 0], shrink=0.76, label="|Ez| [V/m]")

    heating_image = axes[1, 1].imshow(
        joule_heating, origin="lower", extent=extent, cmap="magma"
    )
    axes[1, 1].contour(
        xx, yy, ellipse_mask.astype(float), levels=[0.5], colors="white", linewidths=0.8
    )
    axes[1, 1].set(xlabel="x [m]", ylabel="y [m]", title="Joule-heating source")
    figure.colorbar(heating_image, ax=axes[1, 1], shrink=0.76, label=r"$q_J$ [W/m³]")

    radius_e, spectrum_e = _windowed_radial_spectrum(magnitude)
    radius_t, spectrum_t = _windowed_radial_spectrum(temperature_rise)
    axes[1, 2].semilogy(
        radius_e[1:], spectrum_e[1:] + 1.0e-30, color="#4477AA", label=r"$|E_z|$"
    )
    axes[1, 2].semilogy(
        radius_t[1:], spectrum_t[1:] + 1.0e-30, color="#CC6677", label=r"$\Delta T$"
    )
    axes[1, 2].set(
        xlabel="radial grid-frequency shell",
        ylabel="Hann-windowed shell mean power",
        title="Coupled spatial scales",
    )
    axes[1, 2].legend()
    _style_axis_grid(axes)
    return figure


def _plot_mass_transport_fluid(data: dict[str, Any], time_index: int):
    fields = data["fields"]
    ux, uy, concentration = fields[time_index]
    x, y = data["x"], data["y"]
    extent = (x[0], x[-1], y[0], y[-1])
    time = data["time"]
    config = data["config"]
    auxiliary = data.get("auxiliary", {})
    source = auxiliary.get("source_field")
    if source is None:
        xx, yy = np.meshgrid(x, y, indexing="xy")
        seconds_per_year = 365.0 * 24.0 * 60.0 * 60.0
        source = config["A"] / seconds_per_year * np.exp(
            -(
                (xx - config["x0"]) ** 2 + (yy - config["y0"]) ** 2
            )
            / (2.0 * config["s"] ** 2)
        )
    source = np.squeeze(source)
    speed = np.sqrt(ux**2 + uy**2)
    figure, axes = plt.subplots(2, 3, figsize=(12.0, 7.0), constrained_layout=True)

    concentration_image = axes[0, 0].imshow(
        concentration,
        origin="lower",
        extent=extent,
        cmap="viridis",
        vmin=0.0,
        vmax=max(config["surface_concentration"], concentration.max()),
    )
    axes[0, 0].scatter(
        [config["x0"]], [config["y0"]], marker="*", s=55, c="#CC3311", edgecolors="white"
    )
    axes[0, 0].set(
        xlabel="x [m]", ylabel="y [m]", title=f"Concentration at t={time[time_index]:.3g} years"
    )
    figure.colorbar(concentration_image, ax=axes[0, 0], shrink=0.76, label="mol/m³")

    velocity_limit = max(np.max(np.abs(ux)), np.max(np.abs(uy)), 1.0e-30)
    for axis, field, title in zip(
        axes[0, 1:], (ux, uy), (r"Horizontal Darcy velocity $u_x$", r"Vertical Darcy velocity $u_y$")
    ):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-velocity_limit,
            vmax=velocity_limit,
        )
        axis.set(xlabel="x [m]", ylabel="y [m]", title=title)
        figure.colorbar(image, ax=axis, shrink=0.76, label="m/s")

    levels = np.unique(np.quantile(concentration, [0.1, 0.3, 0.5, 0.7, 0.9]))
    if levels.size:
        axes[1, 0].contour(x, y, concentration, levels=levels, cmap="viridis", linewidths=0.9)
    if speed.max() > 0.0:
        axes[1, 0].streamplot(
            x, y, ux, uy, color="0.25", density=0.85, linewidth=0.55, arrowsize=0.7
        )
    axes[1, 0].scatter(
        [config["x0"]], [config["y0"]], marker="*", s=65, c="#CC3311", label="source centre"
    )
    axes[1, 0].set(
        xlabel="x [m]",
        ylabel="y [m]",
        title="Density-driven circulation and transport fronts",
    )
    axes[1, 0].legend(loc="lower left", fontsize=7)

    cell_area = (x[1] - x[0]) * (y[1] - y[0])
    total_concentration = fields[:, 2].sum(axis=(-2, -1)) * cell_area
    maximum_speed = np.sqrt(fields[:, 0] ** 2 + fields[:, 1] ** 2).max(axis=(-2, -1))
    axes[1, 1].plot(
        time, total_concentration, marker="o", ms=3, color="#4477AA", label=r"$\int c\,dA$"
    )
    axes[1, 1].set(
        xlabel="time [years]", ylabel=r"integrated concentration [mol/m]", title="Transport accumulation and flow response"
    )
    speed_axis = axes[1, 1].twinx()
    speed_axis.plot(
        time, maximum_speed, marker="s", ms=2.5, color="#CC6677", label="maximum speed"
    )
    speed_axis.set_ylabel("maximum speed [m/s]", color="#CC6677")
    speed_axis.tick_params(axis="y", colors="#CC6677")
    handles_left, labels_left = axes[1, 1].get_legend_handles_labels()
    handles_right, labels_right = speed_axis.get_legend_handles_labels()
    axes[1, 1].legend(handles_left + handles_right, labels_left + labels_right, loc="upper left")

    kinetic_energy = 0.5 * speed**2
    radius_ke, spectrum_ke = _windowed_radial_spectrum(kinetic_energy)
    radius_c, spectrum_c = _windowed_radial_spectrum(concentration)
    axes[1, 2].semilogy(
        radius_ke[1:], spectrum_ke[1:] + 1.0e-30, color="#4477AA", label="kinetic energy"
    )
    axes[1, 2].semilogy(
        radius_c[1:], spectrum_c[1:] + 1.0e-30, color="#EE7733", label="concentration"
    )
    axes[1, 2].set(
        xlabel="radial grid-frequency shell",
        ylabel="Hann-windowed shell mean power",
        title="Flow and transport scales",
    )
    axes[1, 2].legend()
    _style_axis_grid(axes)
    return figure


def create_qa_figure(
    case: str,
    input_path: Path,
    output_path: Path,
    *,
    trajectory: int,
    time_index: int,
    dpi: int,
) -> dict[str, Any]:
    data = load_visualization_data(case, input_path, trajectory)
    n_time = data["time"].size
    if time_index < 0:
        time_index += n_time
    if not 0 <= time_index < n_time:
        raise IndexError(f"time index must be in [{-n_time}, {n_time - 1}]")
    diagnostics = compute_diagnostics(
        case,
        data["state"],
        data["time"],
        data["config"],
        result=data.get("auxiliary"),
    )
    diagnostics.update(data.get("stored_diagnostics", {}))
    if case in {"burgers", "ks"}:
        figure = _plot_burgers_or_ks(case, data, time_index)
    elif case == "brusselator":
        figure = _plot_brusselator(data, time_index)
    elif case == "kolmogorov":
        figure = _plot_kolmogorov(data, time_index)
    elif case == "electro_thermal":
        figure = _plot_electro_thermal(data, time_index)
    else:
        figure = _plot_mass_transport_fluid(data, time_index)
    figure.suptitle(
        f"{CASES[case]['display_name']} physical-coherence QA",
        fontsize=11,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() not in {".png", ".pdf", ".svg"}:
        raise ValueError("visualization output must end in .png, .pdf, or .svg")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {
        "case": case,
        "source": data["source"],
        "trajectory": trajectory,
        "time_index": time_index,
        "time": float(data["time"][time_index]),
        "output": str(output_path),
        "diagnostics": diagnostics,
    }
