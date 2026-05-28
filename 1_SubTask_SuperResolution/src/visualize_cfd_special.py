"""
Publication-quality CFD dataset visualization suite.

This script is intentionally post-processing only. It reads the processed
PDEBench CFD HDF5 files and writes all figures into a fresh
Save_reconstruction_files/ForViewDataset/CFD/SpecialVis_<timestamp> directory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.cm import ScalarMappable
import numpy as np


# =============================================================================
# Figure contract
# =============================================================================
# Core conclusion: one CFD snapshot can be read as four aligned physical fields
# over the same structured domain, with topography and color jointly exposing
# channel-specific spatial structure.
# Archetype: schematic-led composite with supporting individual image plates.


# =============================================================================
# Paths and dataset selection
# =============================================================================
# Keep these paths relative to 1_SubTask_SuperResolution/ unless you pass an
# absolute --input-file. The default case selector follows the repository's
# common 90/10 train/test split: --test-case-index 0 means the first holdout
# case, not necessarily case 0. Use --case-index for an explicit raw index.
DEFAULT_PROCESSED_ROOT = "Dataset/PDE_Bench/Processed"
DEFAULT_OUTPUT_ROOT = "Save_reconstruction_files/ForViewDataset/CFD"
DEFAULT_DATASET = "CFD"
DEFAULT_RESOLUTION = "H"  # H, M, or L
DEFAULT_CASE_SOURCE = "test"  # test uses the final holdout cases by train_fraction
DEFAULT_TRAIN_FRACTION = 0.90
DEFAULT_TEST_CASE_INDEX = 0
DEFAULT_CASE_INDEX = None
DEFAULT_FRAME_INDEX = 0


# =============================================================================
# Export controls and runtime feedback
# =============================================================================
DEFAULT_FORMATS = ("png",)  # png, svg, pdf
PNG_DPI = 600
FIGURE_FACE_COLOR = "white"
SVG_EDITABLE_TEXT = True
PDF_EDITABLE_TEXT = True
VERBOSE = True
HDF5_CHUNK_CACHE_MB = 128
RASTERIZE_DENSE_2D = True


# =============================================================================
# 3D camera, geometry, and stacking
# =============================================================================
# Camera tuning:
# - VIEW_ELEVATION controls how overhead the view is. Larger values flatten the
#   surfaces visually and make the parallelogram footprint clearer.
# - VIEW_AZIMUTH rotates the whole composition in the image plane.
# - DOMAIN_SHEAR_X and DOMAIN_Y_SCALE create the flattened parallelogram look
#   without modifying the physical data values.
# - FIELD_LAYER_GAP is the master spacing knob for the stacked composite:
#   mesh -> domain, domain -> first field, and field -> field all use this gap.
# - MESH_Z / BASE_Z / FIELD_BASE_Z remain fallback defaults for standalone
#   geometry and scalar views; the stacked composite derives its Z positions
#   from BASE_Z and FIELD_LAYER_GAP.
# - FIELD_HEIGHT_SCALE is used in the stacked composite; INDIVIDUAL_HEIGHT_SCALE
#   is deliberately larger to reveal topography in single-channel exports.
VIEW_ELEVATION = 45.0
VIEW_AZIMUTH = -58.0
VIEW_ROLL = 0.0
DOMAIN_SHEAR_X = 0.36
DOMAIN_Y_SCALE = 0.74
MESH_Z = -0.34
BASE_Z = 0.0
FIELD_BASE_Z = 0.45
FIELD_LAYER_GAP = 2.5
FIELD_HEIGHT_SCALE = 0.85
INDIVIDUAL_HEIGHT_SCALE = 1.05
FLOW_HEIGHT_SCALE = 0.80
SURFACE_STRIDE = 2
FIELD_SURFACE_STRIDE = 1
MESH_STRIDE = 4
Z_AXIS_COMPRESSION = 0.72
INDIVIDUAL_Z_AXIS_COMPRESSION = 1.65
FIELD_SURFACE_UPSAMPLE = 6
FIELD_SURFACE_MAX_AXIS = 256


# =============================================================================
# Styling, colors, line weights, contours, and lighting
# =============================================================================
METALLIC_BASE_LOW = "#b8bcc2"
METALLIC_BASE_HIGH = "#f0f1f3"
METALLIC_EDGE = "#5e636b"
MESH_FILL = "#d7d9dd"
MESH_LINE = "#32373f"
MESH_FILL_ALPHA = 0.38
BASE_ALPHA = 0.88
FIELD_ALPHA = 0.94
MESH_LINEWIDTH_3D = 0.34
MESH_LINEWIDTH_2D = 0.28
DOMAIN_OUTLINE_LINEWIDTH = 0.90
LIGHT_AZDEG = 315.0
LIGHT_ALTDEG = 42.0
LIGHT_AMBIENT = 0.56
LIGHT_DIFFUSE = 0.44
CONTOUR_LEVELS = 20
CONTOUR_COLOR = "white"
CONTOUR_LINEWIDTH_2D = 0.15
CONTOUR_LINEWIDTH_3D = 0.0
CONTOUR_ALPHA_2D = 0.78
CONTOUR_ALPHA_3D = 0.96
CONTOUR_Z_OFFSET = 0.08
ENABLE_3D_CONTOURS = False
CONTOUR_ON_SMOOTH_3D_SURFACE = False
CONTOUR_SHADOW_COLOR_3D = "#24303a"
CONTOUR_SHADOW_LINEWIDTH_3D = 1.55
CONTOUR_SHADOW_ALPHA_3D = 0.38

FIELD_CMAP_COLORS = {
    "Vx": ("#25345f", "#1f9bb4", "#f1e9b7", "#c84a5f"),
    "Vy": ("#24313f", "#5b4b8a", "#b875a6", "#f0c36d"),
    "density": ("#17351f", "#2f7f4f", "#9bcf7a", "#f4efb0"),
    "pressure": ("#20123a", "#6f3b8f", "#d45167", "#f8c45d"),
}
FALLBACK_CMAP_COLORS = ("#22223b", "#4ea8de", "#c7f9cc", "#ffb703")


# =============================================================================
# Sensor overlay for stacked_3d_composite_Sen
# =============================================================================
# Sensors are sampled from actual mesh nodes, so every marker in the mesh,
# domain, and physical-field layers shares the exact same projected (x, y).
# SENSOR_FIELD_MEASURED_MAP is indexed by physical field order. The default
# marks fields 1 and 3 as measured, and fields 2 and 4 as hidden/unmeasured.
SENSOR_COUNT = 8
SENSOR_RANDOM_SEED = 20260528
SENSOR_FIELD_MEASURED_MAP = (True, False, True, False)
SENSOR_MARKER = "o"
SENSOR_SIZE_MESH_DOMAIN = 12.0
SENSOR_SIZE_FIELD = 12.0
SENSOR_LINEWIDTH = 0.75
SENSOR_MESH_EDGE_COLOR = "black"
SENSOR_MESH_FACE_COLOR = "white"
SENSOR_DOMAIN_EDGE_COLOR = "black"
SENSOR_DOMAIN_FACE_COLOR = "white"
SENSOR_MEASURED_EDGE_COLOR = "#d62728"
SENSOR_MEASURED_FACE_COLOR = "white"
SENSOR_UNMEASURED_EDGE_COLOR = "#c9cdd3"
SENSOR_UNMEASURED_FACE_COLOR = "white"
SENSOR_MARKER_ALPHA = 0.98
SENSOR_CONNECTION_COLOR = "#7a7f87"
SENSOR_CONNECTION_LINESTYLE = "--"
SENSOR_CONNECTION_LINEWIDTH = 0.78
SENSOR_CONNECTION_ALPHA = 0.58
SENSOR_Z_LIFT = 0.16
SENSOR_CONNECTION_Z_LIFT = 0.045
SENSOR_DEPTHSHADE = False


# =============================================================================
# Artistic noise and flow-matching visualizations
# =============================================================================
# These panels are visual metaphors for the flow-matching path:
# random Gaussian noise -> intermediate state -> first physical channel. The
# noise and field are normalized to [0, 1] before interpolation, so FLOW_T=0.5
# means an equal visual blend rather than an equal physical-unit blend.
NOISE_RANDOM_SEED = 20260527
NOISE_STD = 1.0
NOISE_SCATTER_STRIDE = 1
NOISE_SCATTER_SIZE_2D = 5.0
NOISE_SCATTER_SIZE_3D = 5.0
NOISE_ALPHA = 0.86
NOISE_CMAP = "gray"
FLOW_T = 0.50
FLOW_CMAP_COLORS = ("#151515", "#52616b", "#b8d8d8", "#f6d365")


# =============================================================================
# Typography and layout
# =============================================================================
FONT_FAMILY = ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
BASE_FONT_SIZE = 7
TITLE_FONT_SIZE = 8
LABEL_FONT_SIZE = 7
COLORBAR_FONT_SIZE = 6
COMPOSITE_FIGSIZE = (7.4, 6.0)
CHANNEL_SUBPLOTS_FIGSIZE = (7.2, 5.8)
INDIVIDUAL_3D_FIGSIZE = (3.6, 3.2)
FIELD_2D_FIGSIZE = (3.4, 3.0)
DOMAIN_2D_FIGSIZE = (3.4, 3.0)
GRID_2D_FIGSIZE = (3.4, 3.0)
FLOW_TRIPTYCH_FIGSIZE = (7.2, 2.5)


# =============================================================================
# Output organization
# =============================================================================
# Every run still creates one SpecialVis_<timestamp>/ root. Figures are then
# grouped by role so that the directory stays readable even when all SVG/PDF/PNG
# variants are enabled.
OUTPUT_SUBDIRS = {
    "composites": "01_composites",
    "geometry_3d": "02_geometry/3d",
    "geometry_2d": "02_geometry/2d",
    "fields_3d": "03_physical_fields/3d",
    "fields_2d": "03_physical_fields/2d",
    "flow_matching": "04_flow_matching",
    "metadata": "00_metadata",
}


@dataclass
class Snapshot:
    h5_path: Path
    case_index: int
    frame_index: int
    time_value: float | None
    field_names: List[str]
    x: np.ndarray
    y: np.ndarray
    xp: np.ndarray
    yp: np.ndarray
    fields: Dict[str, np.ndarray]


def log(message: str) -> None:
    if VERBOSE:
        print(f"[visualize_cfd_special] {message}", flush=True)


def elapsed_label(seconds: float) -> str:
    return f"{seconds:.2f}s" if seconds >= 0.01 else f"{seconds * 1000.0:.1f}ms"


def configure_matplotlib() -> None:
    rc = {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FAMILY,
        "font.size": BASE_FONT_SIZE,
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": FIGURE_FACE_COLOR,
        "savefig.facecolor": FIGURE_FACE_COLOR,
        "savefig.dpi": PNG_DPI,
    }
    if SVG_EDITABLE_TEXT:
        rc["svg.fonttype"] = "none"
    if PDF_EDITABLE_TEXT:
        rc["pdf.fonttype"] = 42
    mpl.rcParams.update(rc)


def project_domain(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    xp = x + DOMAIN_SHEAR_X * y
    yp = DOMAIN_Y_SCALE * y
    return xp, yp


def unique_axis(values: np.ndarray, decimals: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    rounded = np.round(values.astype(np.float64), decimals=decimals)
    unique, inverse = np.unique(rounded, return_inverse=True)
    return unique.astype(np.float64), inverse.astype(np.int64)


def grid_from_flat(
    coords_xy: np.ndarray,
    values_by_channel: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    x_unique, x_rank = unique_axis(coords_xy[:, 0])
    y_unique, y_rank = unique_axis(coords_xy[:, 1])
    nx, ny = len(x_unique), len(y_unique)
    n_pts = coords_xy.shape[0]
    if nx * ny != n_pts:
        raise ValueError(
            "The visualization suite expects a structured 2D grid. "
            f"Found {n_pts} points but {nx} unique x times {ny} unique y = {nx * ny}."
        )

    x_grid, y_grid = np.meshgrid(x_unique, y_unique, indexing="xy")
    grids: Dict[int, np.ndarray] = {}
    for ch in range(values_by_channel.shape[1]):
        arr = np.full((ny, nx), np.nan, dtype=np.float32)
        arr[y_rank, x_rank] = values_by_channel[:, ch]
        if np.isnan(arr).any():
            raise ValueError(f"Channel {ch} could not be mapped onto the structured grid.")
        grids[ch] = arr
    return x_grid, y_grid, grids


def parse_field_names(fields_ds: h5py.Dataset, n_fields: int) -> List[str]:
    raw = fields_ds.attrs.get("selected_fields", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    names = [item.strip() for item in str(raw).split(",") if item.strip()]
    if len(names) == n_fields:
        return names
    fallback = ["Vx", "Vy", "density", "pressure"]
    return fallback[:n_fields]


def processed_file(project_root: Path, processed_root: str, resolution: str) -> Path:
    return project_root / processed_root / f"{DEFAULT_DATASET}_{resolution}_res.h5"


def choose_case_index(
    n_cases: int,
    case_source: str,
    train_fraction: float,
    test_case_index: int,
    explicit_case_index: int | None,
) -> int:
    if explicit_case_index is not None:
        case_index = explicit_case_index
    elif case_source == "all":
        case_index = test_case_index
    else:
        n_train = int(n_cases * train_fraction)
        if n_train >= n_cases:
            n_train = max(0, n_cases - 1)
        test_cases = list(range(n_train, n_cases))
        if not test_cases:
            test_cases = [0]
        if not (0 <= test_case_index < len(test_cases)):
            raise ValueError(
                f"test_case_index={test_case_index} is out of range for "
                f"{len(test_cases)} inferred test cases."
            )
        case_index = test_cases[test_case_index]

    if not (0 <= case_index < n_cases):
        raise ValueError(f"case_index={case_index} is out of range for {n_cases} cases.")
    return int(case_index)


def load_snapshot(args: argparse.Namespace) -> Snapshot:
    project_root = Path(__file__).resolve().parent.parent
    h5_path = Path(args.input_file) if args.input_file else processed_file(
        project_root, args.processed_root, args.resolution
    )
    if not h5_path.exists():
        raise FileNotFoundError(f"Processed CFD HDF5 file not found: {h5_path}")

    log(f"Opening HDF5 file: {h5_path}")
    log(
        "I/O mode: h5py hyperslab read. The processed files are HDF5 datasets "
        "chunked by case/time snapshot, so true numpy memmap is not applicable "
        "for compressed/chunked HDF5 storage; reading one aligned chunk is the "
        "lowest-latency safe path here."
    )
    t0 = time.perf_counter()
    cache_bytes = int(HDF5_CHUNK_CACHE_MB * 1024 * 1024)
    with h5py.File(h5_path, "r", rdcc_nbytes=cache_bytes) as f:
        if "coordinates" not in f or "fields" not in f:
            raise KeyError("Expected HDF5 datasets 'coordinates' and 'fields'.")
        fields_ds = f["fields"]
        n_cases, n_time, n_pts, _, _, n_fields = fields_ds.shape
        log(
            f"Detected fields shape: cases={n_cases}, frames={n_time}, "
            f"points={n_pts}, channels={n_fields}"
        )
        coords_t0 = time.perf_counter()
        coords = f["coordinates"][:, 0, 0, :2].astype(np.float32)
        log(f"Loaded coordinates {coords.shape} in {elapsed_label(time.perf_counter() - coords_t0)}")
        if coords.shape[0] != n_pts:
            raise ValueError(
                f"Coordinate count {coords.shape[0]} does not match field point count {n_pts}."
            )
        case_index = choose_case_index(
            n_cases=n_cases,
            case_source=args.case_source,
            train_fraction=args.train_fraction,
            test_case_index=args.test_case_index,
            explicit_case_index=args.case_index,
        )
        if not (0 <= args.frame_index < n_time):
            raise ValueError(f"frame_index={args.frame_index} is out of range for {n_time} frames.")

        read_t0 = time.perf_counter()
        values = np.empty((n_pts, n_fields), dtype=np.float32)
        fields_ds.read_direct(
            values,
            source_sel=np.s_[case_index, args.frame_index, :, 0, 0, :],
        )
        log(
            f"Loaded case={case_index}, frame={args.frame_index} snapshot "
            f"{values.shape} in {elapsed_label(time.perf_counter() - read_t0)}"
        )
        field_names = parse_field_names(fields_ds, n_fields)
        time_value = None
        if "time" in f:
            time_values = np.asarray(f["time"][:], dtype=np.float64)
            if args.frame_index < len(time_values):
                time_value = float(time_values[args.frame_index])

    grid_t0 = time.perf_counter()
    x_grid, y_grid, channel_grids = grid_from_flat(coords, values)
    xp, yp = project_domain(x_grid, y_grid)
    fields = {field_names[ch]: channel_grids[ch] for ch in range(len(field_names))}
    log(f"Mapped flat fields to grid {x_grid.shape} in {elapsed_label(time.perf_counter() - grid_t0)}")
    log(f"Finished data loading in {elapsed_label(time.perf_counter() - t0)}")
    return Snapshot(
        h5_path=h5_path,
        case_index=case_index,
        frame_index=int(args.frame_index),
        time_value=time_value,
        field_names=field_names,
        x=x_grid,
        y=y_grid,
        xp=xp,
        yp=yp,
        fields=fields,
    )


def make_output_dir(output_root: str) -> Path:
    project_root = Path(__file__).resolve().parent.parent
    root = project_root / output_root
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / f"SpecialVis_{stamp}"
    suffix = 1
    while out_dir.exists():
        out_dir = root / f"SpecialVis_{stamp}_{suffix:02d}"
        suffix += 1
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def make_output_subdirs(out_dir: Path) -> Dict[str, Path]:
    subdirs = {key: out_dir / rel_path for key, rel_path in OUTPUT_SUBDIRS.items()}
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def clean_field_name(name: str) -> str:
    return name.lower().replace(" ", "_").replace("/", "_")


def display_field_name(name: str) -> str:
    mapping = {
        "Vx": r"$V_x$",
        "Vy": r"$V_y$",
        "density": "density",
        "pressure": "pressure",
    }
    return mapping.get(name, name)


def normalize_field(values: np.ndarray) -> Tuple[np.ndarray, mcolors.Normalize]:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    normalized = norm(values).astype(np.float64)
    return normalized, norm


def make_cmap(name: str) -> mcolors.Colormap:
    colors = FIELD_CMAP_COLORS.get(name, FALLBACK_CMAP_COLORS)
    return mcolors.LinearSegmentedColormap.from_list(f"special_{clean_field_name(name)}", colors)


def stride_indices(length: int, stride: int) -> np.ndarray:
    stride = max(1, int(stride))
    idx = np.arange(0, length, stride, dtype=np.int64)
    if idx[-1] != length - 1:
        idx = np.r_[idx, length - 1]
    return idx


def decimate(*arrays: np.ndarray, stride: int) -> Tuple[np.ndarray, ...]:
    iy = stride_indices(arrays[0].shape[0], stride)
    ix = stride_indices(arrays[0].shape[1], stride)
    return tuple(arr[np.ix_(iy, ix)] for arr in arrays)


def interpolate_axis_length(length: int, upsample: int, max_axis: int) -> int:
    if upsample <= 1:
        return int(length)
    target = (int(length) - 1) * int(upsample) + 1
    return int(min(target, max_axis))


def bilinear_resize(values: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    src_y = np.linspace(0.0, 1.0, values.shape[0])
    src_x = np.linspace(0.0, 1.0, values.shape[1])
    dst_y = np.linspace(0.0, 1.0, target_shape[0])
    dst_x = np.linspace(0.0, 1.0, target_shape[1])

    x_interp = np.empty((values.shape[0], target_shape[1]), dtype=np.float64)
    for row_idx in range(values.shape[0]):
        x_interp[row_idx] = np.interp(dst_x, src_x, values[row_idx])

    out = np.empty(target_shape, dtype=np.float64)
    for col_idx in range(target_shape[1]):
        out[:, col_idx] = np.interp(dst_y, src_y, x_interp[:, col_idx])
    return out


def smooth_field_render_grid(
    snapshot: Snapshot,
    values: np.ndarray,
    z: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target_shape = (
        interpolate_axis_length(values.shape[0], FIELD_SURFACE_UPSAMPLE, FIELD_SURFACE_MAX_AXIS),
        interpolate_axis_length(values.shape[1], FIELD_SURFACE_UPSAMPLE, FIELD_SURFACE_MAX_AXIS),
    )
    if target_shape == values.shape:
        return snapshot.xp, snapshot.yp, values.astype(np.float64), z.astype(np.float64)

    x_smooth = bilinear_resize(snapshot.x, target_shape)
    y_smooth = bilinear_resize(snapshot.y, target_shape)
    xp_smooth, yp_smooth = project_domain(x_smooth, y_smooth)
    values_smooth = bilinear_resize(values, target_shape)
    z_smooth = bilinear_resize(z, target_shape)
    return xp_smooth, yp_smooth, values_smooth, z_smooth


def metallic_rgba(xp: np.ndarray, yp: np.ndarray, alpha: float) -> np.ndarray:
    xnorm = (xp - np.nanmin(xp)) / max(np.nanmax(xp) - np.nanmin(xp), 1e-8)
    ynorm = (yp - np.nanmin(yp)) / max(np.nanmax(yp) - np.nanmin(yp), 1e-8)
    gradient = 0.25 + 0.75 * (0.62 * xnorm + 0.38 * (1.0 - ynorm))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "soft_metal", [METALLIC_BASE_LOW, METALLIC_BASE_HIGH]
    )
    rgba = cmap(np.clip(gradient, 0.0, 1.0))
    rgba[..., 3] = alpha
    return rgba


def field_facecolors(
    values: np.ndarray,
    z_values: np.ndarray,
    cmap: mcolors.Colormap,
    norm: mcolors.Normalize,
) -> np.ndarray:
    base_rgb = cmap(norm(values))[..., :3]
    dzdy, dzdx = np.gradient(z_values.astype(np.float64))
    normal = np.dstack((-dzdx, -dzdy, np.ones_like(z_values, dtype=np.float64)))
    normal /= np.linalg.norm(normal, axis=2, keepdims=True).clip(min=1e-8)

    az = np.deg2rad(LIGHT_AZDEG)
    alt = np.deg2rad(LIGHT_ALTDEG)
    light_vec = np.array(
        [np.cos(alt) * np.cos(az), np.cos(alt) * np.sin(az), np.sin(alt)],
        dtype=np.float64,
    )
    intensity = np.clip(np.tensordot(normal, light_vec, axes=([2], [0])), 0.0, 1.0)
    shade = LIGHT_AMBIENT + LIGHT_DIFFUSE * intensity
    shaded = base_rgb * shade[..., None]
    rgba = np.empty((*values.shape, 4), dtype=np.float64)
    rgba[..., :3] = np.clip(shaded, 0.0, 1.0)
    rgba[..., 3] = FIELD_ALPHA
    return rgba


def style_3d_axis(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH, roll=VIEW_ROLL)
    try:
        ax.set_proj_type("ortho")
    except Exception:
        pass


def set_3d_limits(
    ax: plt.Axes,
    xp: np.ndarray,
    yp: np.ndarray,
    z_top: float,
    z_bottom: float = MESH_Z - 0.05,
    z_compression: float = Z_AXIS_COMPRESSION,
) -> None:
    xpad = 0.08 * (np.nanmax(xp) - np.nanmin(xp))
    ypad = 0.08 * (np.nanmax(yp) - np.nanmin(yp))
    ax.set_xlim(float(np.nanmin(xp) - xpad), float(np.nanmax(xp) + xpad))
    ax.set_ylim(float(np.nanmin(yp) - ypad), float(np.nanmax(yp) + ypad))
    ax.set_zlim(z_bottom, z_top + 0.20)
    ax.set_box_aspect(
        (
            float(np.nanmax(xp) - np.nanmin(xp)),
            float(np.nanmax(yp) - np.nanmin(yp)),
            max((z_top - z_bottom) * z_compression, 1e-6),
        )
    )


def stacked_mesh_z() -> float:
    return BASE_Z - FIELD_LAYER_GAP


def stacked_field_z(layer_index: int) -> float:
    return BASE_Z + (int(layer_index) + 1) * FIELD_LAYER_GAP


def add_domain_base(ax: plt.Axes, snapshot: Snapshot, z_level: float = BASE_Z) -> None:
    xp_base, yp_base = decimate(snapshot.xp, snapshot.yp, stride=SURFACE_STRIDE)
    z_base = np.full_like(xp_base, z_level, dtype=np.float64)
    ax.plot_surface(
        xp_base,
        yp_base,
        z_base,
        facecolors=metallic_rgba(xp_base, yp_base, BASE_ALPHA),
        linewidth=0,
        antialiased=True,
        shade=False,
    )


def add_mesh_layer(ax: plt.Axes, snapshot: Snapshot, z_level: float = MESH_Z) -> None:
    xp_mesh, yp_mesh = decimate(snapshot.xp, snapshot.yp, stride=MESH_STRIDE)
    z_mesh = np.full_like(xp_mesh, z_level, dtype=np.float64)
    mesh_rgba = mcolors.to_rgba(MESH_FILL, alpha=MESH_FILL_ALPHA)
    facecolors = np.empty((*xp_mesh.shape, 4), dtype=np.float64)
    facecolors[...] = mesh_rgba
    ax.plot_surface(
        xp_mesh,
        yp_mesh,
        z_mesh,
        facecolors=facecolors,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    ax.plot_wireframe(
        xp_mesh,
        yp_mesh,
        z_mesh,
        color=MESH_LINE,
        linewidth=MESH_LINEWIDTH_3D,
        alpha=0.82,
    )


def add_base_and_mesh(
    ax: plt.Axes,
    snapshot: Snapshot,
    domain_z: float = BASE_Z,
    mesh_z: float = MESH_Z,
) -> None:
    add_domain_base(ax, snapshot, z_level=domain_z)
    add_mesh_layer(ax, snapshot, z_level=mesh_z)


def generate_sensor_indices(snapshot: Snapshot) -> Tuple[np.ndarray, np.ndarray]:
    ny, nx = snapshot.x.shape
    n_total = ny * nx
    n_sensors = min(int(SENSOR_COUNT), n_total)
    rng = np.random.default_rng(SENSOR_RANDOM_SEED)
    flat_idx = rng.choice(n_total, size=n_sensors, replace=False)
    rows, cols = np.unravel_index(flat_idx, (ny, nx))
    return rows.astype(np.int64), cols.astype(np.int64)


def field_sensor_z(values: np.ndarray, rows: np.ndarray, cols: np.ndarray, z_offset: float) -> np.ndarray:
    _, norm = normalize_field(values)
    return z_offset + FIELD_HEIGHT_SCALE * norm(values[rows, cols])


def stacked_sensor_layers(
    snapshot: Snapshot,
    rows: np.ndarray,
    cols: np.ndarray,
    mesh_z: float,
    domain_z: float,
) -> List[np.ndarray]:
    layer_zs = [
        np.full(rows.shape, mesh_z, dtype=np.float64),
        np.full(rows.shape, domain_z, dtype=np.float64),
    ]
    for layer_index, field_name in enumerate(snapshot.field_names):
        layer_zs.append(
            field_sensor_z(
                snapshot.fields[field_name],
                rows,
                cols,
                z_offset=stacked_field_z(layer_index),
            )
        )
    return layer_zs


def field_is_measured(layer_index: int) -> bool:
    if layer_index < len(SENSOR_FIELD_MEASURED_MAP):
        return bool(SENSOR_FIELD_MEASURED_MAP[layer_index])
    return False


def scatter_sensor_layer(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    edge_color: str,
    face_color: str,
    size: float,
) -> None:
    collection = ax.scatter(
        x,
        y,
        z + SENSOR_Z_LIFT,
        marker=SENSOR_MARKER,
        s=size,
        facecolors=face_color,
        edgecolors=edge_color,
        linewidths=SENSOR_LINEWIDTH,
        alpha=SENSOR_MARKER_ALPHA,
        depthshade=SENSOR_DEPTHSHADE,
    )
    try:
        collection.set_sort_zpos(float(np.nanmax(z + SENSOR_Z_LIFT)))
    except Exception:
        pass


def draw_sensor_connectors(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    layer_zs: Sequence[np.ndarray],
) -> None:
    for sensor_i in range(x.shape[0]):
        z_path = np.array([layer[sensor_i] + SENSOR_CONNECTION_Z_LIFT for layer in layer_zs])
        ax.plot(
            np.full_like(z_path, x[sensor_i], dtype=np.float64),
            np.full_like(z_path, y[sensor_i], dtype=np.float64),
            z_path,
            color=SENSOR_CONNECTION_COLOR,
            linestyle=SENSOR_CONNECTION_LINESTYLE,
            linewidth=SENSOR_CONNECTION_LINEWIDTH,
            alpha=SENSOR_CONNECTION_ALPHA,
        )


def add_sensor_overlay_to_stacked_composite(
    ax: plt.Axes,
    snapshot: Snapshot,
    mesh_z: float,
    domain_z: float,
) -> float:
    rows, cols = generate_sensor_indices(snapshot)
    x = snapshot.xp[rows, cols]
    y = snapshot.yp[rows, cols]
    layer_zs = stacked_sensor_layers(snapshot, rows, cols, mesh_z, domain_z)

    draw_sensor_connectors(ax, x, y, layer_zs)
    scatter_sensor_layer(
        ax,
        x,
        y,
        layer_zs[0],
        edge_color=SENSOR_MESH_EDGE_COLOR,
        face_color=SENSOR_MESH_FACE_COLOR,
        size=SENSOR_SIZE_MESH_DOMAIN,
    )
    scatter_sensor_layer(
        ax,
        x,
        y,
        layer_zs[1],
        edge_color=SENSOR_DOMAIN_EDGE_COLOR,
        face_color=SENSOR_DOMAIN_FACE_COLOR,
        size=SENSOR_SIZE_MESH_DOMAIN,
    )
    for layer_index, field_name in enumerate(snapshot.field_names):
        measured = field_is_measured(layer_index)
        scatter_sensor_layer(
            ax,
            x,
            y,
            layer_zs[layer_index + 2],
            edge_color=SENSOR_MEASURED_EDGE_COLOR if measured else SENSOR_UNMEASURED_EDGE_COLOR,
            face_color=SENSOR_MEASURED_FACE_COLOR if measured else SENSOR_UNMEASURED_FACE_COLOR,
            size=SENSOR_SIZE_FIELD,
        )

    return float(max(np.max(layer_z) for layer_z in layer_zs))


def contour_levels(values: np.ndarray, n_levels: int = CONTOUR_LEVELS) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        return np.array([], dtype=np.float64)
    return np.linspace(vmin, vmax, int(n_levels) + 2, dtype=np.float64)[1:-1]


def add_3d_contours(
    ax: plt.Axes,
    xp: np.ndarray,
    yp: np.ndarray,
    values: np.ndarray,
    norm: mcolors.Normalize,
    z_offset: float,
    height_scale: float,
) -> None:
    if not ENABLE_3D_CONTOURS:
        return
    levels = contour_levels(values)
    if levels.size == 0:
        return
    tmp_fig, tmp_ax = plt.subplots(figsize=(1.0, 1.0))
    contour_set = tmp_ax.contour(xp, yp, values, levels=levels)
    all_segments = [list(level_segments) for level_segments in contour_set.allsegs]
    plt.close(tmp_fig)
    for level, level_segments in zip(levels, all_segments):
        z_value = float(z_offset + height_scale * norm(level) + CONTOUR_Z_OFFSET)
        for segment in level_segments:
            if segment.shape[0] < 2:
                continue
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                np.full(segment.shape[0], z_value + 0.001),
                color=CONTOUR_SHADOW_COLOR_3D,
                linewidth=CONTOUR_SHADOW_LINEWIDTH_3D,
                alpha=CONTOUR_SHADOW_ALPHA_3D,
                zorder=9,
            )
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                np.full(segment.shape[0], z_value + 0.002),
                color=CONTOUR_COLOR,
                linewidth=CONTOUR_LINEWIDTH_3D,
                alpha=CONTOUR_ALPHA_3D,
                zorder=10,
            )


def add_2d_contours(ax: plt.Axes, xp: np.ndarray, yp: np.ndarray, values: np.ndarray) -> None:
    levels = contour_levels(values)
    if levels.size == 0:
        return
    ax.contour(
        xp,
        yp,
        values,
        levels=levels,
        colors=CONTOUR_COLOR,
        linewidths=CONTOUR_LINEWIDTH_2D,
        alpha=CONTOUR_ALPHA_2D,
    )


def plot_field_surface(
    ax: plt.Axes,
    snapshot: Snapshot,
    field_name: str,
    layer_index: int,
    stride: int,
    height_scale: float = FIELD_HEIGHT_SCALE,
    z_offset: float | None = None,
) -> Tuple[mcolors.Colormap, mcolors.Normalize, float]:
    values = snapshot.fields[field_name]
    normalized, norm = normalize_field(values)
    if z_offset is None:
        z_offset = FIELD_BASE_Z + layer_index * FIELD_LAYER_GAP
    z = z_offset + height_scale * normalized
    xp_s, yp_s, values_s, z_s = smooth_field_render_grid(snapshot, values, z)
    xp_d, yp_d, values_d, z_d = decimate(xp_s, yp_s, values_s, z_s, stride=stride)
    cmap = make_cmap(field_name)
    facecolors = field_facecolors(values_d, z_d, cmap, norm)
    ax.plot_surface(
        xp_d,
        yp_d,
        z_d,
        facecolors=facecolors,
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=False,
    )
    if CONTOUR_ON_SMOOTH_3D_SURFACE:
        add_3d_contours(ax, xp_s, yp_s, values_s, norm, float(z_offset), height_scale)
    else:
        add_3d_contours(ax, snapshot.xp, snapshot.yp, values, norm, float(z_offset), height_scale)
    return cmap, norm, float(np.nanmax(z))


def add_colorbars(
    fig: plt.Figure,
    colorbar_specs: Sequence[Tuple[str, mcolors.Colormap, mcolors.Normalize]],
) -> None:
    left = 0.87
    width = 0.018
    height = 0.145
    top = 0.80
    gap = 0.045
    for i, (field_name, cmap, norm) in enumerate(colorbar_specs):
        bottom = top - i * (height + gap)
        cax = fig.add_axes([left, bottom, width, height])
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, cax=cax)
        cb.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2, width=0.5)
        cb.outline.set_linewidth(0.5)
        cb.set_label(display_field_name(field_name), fontsize=COLORBAR_FONT_SIZE)


def save_figure(fig: plt.Figure, out_base: Path, formats: Iterable[str], dpi: int = PNG_DPI) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fmt = fmt.lower().lstrip(".")
        out_path = out_base.with_suffix(f".{fmt}")
        save_kwargs = {"bbox_inches": "tight", "facecolor": FIGURE_FACE_COLOR}
        if fmt == "png":
            save_kwargs["dpi"] = dpi
        t0 = time.perf_counter()
        fig.savefig(out_path, **save_kwargs)
        log(f"Saved {out_path} in {elapsed_label(time.perf_counter() - t0)}")


def make_domain_3d(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    fig = plt.figure(figsize=INDIVIDUAL_3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    add_domain_base(ax, snapshot)
    style_3d_axis(ax)
    set_3d_limits(ax, snapshot.xp, snapshot.yp, BASE_Z + 0.02, z_bottom=-0.08)
    save_figure(fig, out_dir / "domain_3d", formats)
    plt.close(fig)


def make_mesh_3d(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    fig = plt.figure(figsize=INDIVIDUAL_3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    add_mesh_layer(ax, snapshot)
    style_3d_axis(ax)
    set_3d_limits(ax, snapshot.xp, snapshot.yp, MESH_Z + 0.08, z_bottom=MESH_Z - 0.08)
    save_figure(fig, out_dir / "mesh_3d", formats)
    plt.close(fig)


def make_composite_3d(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    fig = plt.figure(figsize=COMPOSITE_FIGSIZE)
    ax = fig.add_axes([0.02, 0.02, 0.80, 0.94], projection="3d")
    mesh_z = stacked_mesh_z()
    add_base_and_mesh(ax, snapshot, domain_z=BASE_Z, mesh_z=mesh_z)

    colorbar_specs = []
    z_top = BASE_Z
    for layer_index, field_name in enumerate(snapshot.field_names):
        cmap, norm, z_max = plot_field_surface(
            ax,
            snapshot,
            field_name,
            layer_index,
            stride=FIELD_SURFACE_STRIDE,
            z_offset=stacked_field_z(layer_index),
        )
        colorbar_specs.append((field_name, cmap, norm))
        z_top = max(z_top, z_max)

    style_3d_axis(ax)
    set_3d_limits(ax, snapshot.xp, snapshot.yp, z_top, z_bottom=mesh_z - 0.05)
    title = f"CFD stacked fields: case {snapshot.case_index}, frame {snapshot.frame_index}"
    if snapshot.time_value is not None:
        title += f", t={snapshot.time_value:.4g}"
    fig.text(0.04, 0.965, title, ha="left", va="top", fontsize=TITLE_FONT_SIZE)
    add_colorbars(fig, colorbar_specs)
    save_figure(fig, out_dir / "stacked_3d_composite", formats)
    plt.close(fig)


def make_composite_3d_sensor(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    fig = plt.figure(figsize=COMPOSITE_FIGSIZE)
    ax = fig.add_axes([0.02, 0.02, 0.80, 0.94], projection="3d")
    mesh_z = stacked_mesh_z()
    rows, cols = generate_sensor_indices(snapshot)
    x = snapshot.xp[rows, cols]
    y = snapshot.yp[rows, cols]
    layer_zs = stacked_sensor_layers(
        snapshot,
        rows,
        cols,
        mesh_z=mesh_z,
        domain_z=BASE_Z,
    )

    # Build the sensor composite as actual layers. Each layer is rendered with
    # its own markers immediately above its surface instead of adding one late
    # global overlay after the stack is complete.
    add_mesh_layer(ax, snapshot, z_level=mesh_z)
    scatter_sensor_layer(
        ax,
        x,
        y,
        layer_zs[0],
        edge_color=SENSOR_MESH_EDGE_COLOR,
        face_color=SENSOR_MESH_FACE_COLOR,
        size=SENSOR_SIZE_MESH_DOMAIN,
    )
    add_domain_base(ax, snapshot, z_level=BASE_Z)
    scatter_sensor_layer(
        ax,
        x,
        y,
        layer_zs[1],
        edge_color=SENSOR_DOMAIN_EDGE_COLOR,
        face_color=SENSOR_DOMAIN_FACE_COLOR,
        size=SENSOR_SIZE_MESH_DOMAIN,
    )

    colorbar_specs = []
    z_top = BASE_Z
    for layer_index, field_name in enumerate(snapshot.field_names):
        cmap, norm, z_max = plot_field_surface(
            ax,
            snapshot,
            field_name,
            layer_index,
            stride=FIELD_SURFACE_STRIDE,
            z_offset=stacked_field_z(layer_index),
        )
        colorbar_specs.append((field_name, cmap, norm))
        z_top = max(z_top, z_max)
        measured = field_is_measured(layer_index)
        scatter_sensor_layer(
            ax,
            x,
            y,
            layer_zs[layer_index + 2],
            edge_color=SENSOR_MEASURED_EDGE_COLOR if measured else SENSOR_UNMEASURED_EDGE_COLOR,
            face_color=SENSOR_MEASURED_FACE_COLOR if measured else SENSOR_UNMEASURED_FACE_COLOR,
            size=SENSOR_SIZE_FIELD,
        )

    draw_sensor_connectors(ax, x, y, layer_zs)
    # Re-emit markers after connectors so hollow circles remain crisp on their
    # own surfaces while the 3D Z coordinates still preserve the layer geometry.
    scatter_sensor_layer(
        ax,
        x,
        y,
        layer_zs[0],
        edge_color=SENSOR_MESH_EDGE_COLOR,
        face_color=SENSOR_MESH_FACE_COLOR,
        size=SENSOR_SIZE_MESH_DOMAIN,
    )
    scatter_sensor_layer(
        ax,
        x,
        y,
        layer_zs[1],
        edge_color=SENSOR_DOMAIN_EDGE_COLOR,
        face_color=SENSOR_DOMAIN_FACE_COLOR,
        size=SENSOR_SIZE_MESH_DOMAIN,
    )
    for layer_index in range(len(snapshot.field_names)):
        measured = field_is_measured(layer_index)
        scatter_sensor_layer(
            ax,
            x,
            y,
            layer_zs[layer_index + 2],
            edge_color=SENSOR_MEASURED_EDGE_COLOR if measured else SENSOR_UNMEASURED_EDGE_COLOR,
            face_color=SENSOR_MEASURED_FACE_COLOR if measured else SENSOR_UNMEASURED_FACE_COLOR,
            size=SENSOR_SIZE_FIELD,
        )

    sensor_z_top = float(max(np.max(layer_z) for layer_z in layer_zs))
    z_top = max(z_top, sensor_z_top + SENSOR_Z_LIFT)

    style_3d_axis(ax)
    set_3d_limits(ax, snapshot.xp, snapshot.yp, z_top, z_bottom=mesh_z - 0.05)
    title = (
        f"CFD stacked fields with sensors: case {snapshot.case_index}, "
        f"frame {snapshot.frame_index}"
    )
    if snapshot.time_value is not None:
        title += f", t={snapshot.time_value:.4g}"
    fig.text(0.04, 0.965, title, ha="left", va="top", fontsize=TITLE_FONT_SIZE)
    add_colorbars(fig, colorbar_specs)
    save_figure(fig, out_dir / "stacked_3d_composite_Sen", formats)
    plt.close(fig)


def make_individual_3d(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    for field_name in snapshot.field_names:
        fig = plt.figure(figsize=INDIVIDUAL_3D_FIGSIZE)
        ax = fig.add_subplot(111, projection="3d")
        cmap, norm, z_top = plot_field_surface(
            ax,
            snapshot,
            field_name,
            0,
            stride=FIELD_SURFACE_STRIDE,
            height_scale=INDIVIDUAL_HEIGHT_SCALE,
            z_offset=0.0,
        )
        style_3d_axis(ax)
        set_3d_limits(
            ax,
            snapshot.xp,
            snapshot.yp,
            z_top,
            z_bottom=-0.06,
            z_compression=INDIVIDUAL_Z_AXIS_COMPRESSION,
        )
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.02, shrink=0.70)
        cb.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2, width=0.5)
        cb.set_label(display_field_name(field_name), fontsize=COLORBAR_FONT_SIZE)
        save_figure(fig, out_dir / f"individual_3d_{clean_field_name(field_name)}", formats)
        plt.close(fig)


def make_channel_3d_subplots(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    n_fields = len(snapshot.field_names)
    n_cols = 2
    n_rows = int(np.ceil(n_fields / n_cols))
    fig = plt.figure(figsize=CHANNEL_SUBPLOTS_FIGSIZE)
    for i, field_name in enumerate(snapshot.field_names, start=1):
        ax = fig.add_subplot(n_rows, n_cols, i, projection="3d")
        cmap, norm, z_top = plot_field_surface(
            ax,
            snapshot,
            field_name,
            0,
            stride=FIELD_SURFACE_STRIDE,
            height_scale=INDIVIDUAL_HEIGHT_SCALE,
            z_offset=0.0,
        )
        style_3d_axis(ax)
        set_3d_limits(
            ax,
            snapshot.xp,
            snapshot.yp,
            z_top,
            z_bottom=-0.06,
            z_compression=INDIVIDUAL_Z_AXIS_COMPRESSION,
        )
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.01, shrink=0.54)
        cb.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2, width=0.5)
        cb.set_label(display_field_name(field_name), fontsize=COLORBAR_FONT_SIZE)
    fig.suptitle(
        f"CFD physical channels: case {snapshot.case_index}, frame {snapshot.frame_index}",
        x=0.03,
        y=0.98,
        ha="left",
        fontsize=TITLE_FONT_SIZE,
    )
    fig.subplots_adjust(left=0.01, right=0.96, bottom=0.02, top=0.92, wspace=0.02, hspace=0.02)
    save_figure(fig, out_dir / "channel_3d_subplots", formats)
    plt.close(fig)


def axis_equal_2d(ax: plt.Axes, xp: np.ndarray, yp: np.ndarray) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    xpad = 0.04 * (np.nanmax(xp) - np.nanmin(xp))
    ypad = 0.04 * (np.nanmax(yp) - np.nanmin(yp))
    ax.set_xlim(float(np.nanmin(xp) - xpad), float(np.nanmax(xp) + xpad))
    ax.set_ylim(float(np.nanmin(yp) - ypad), float(np.nanmax(yp) + ypad))


def grid_segments(xp: np.ndarray, yp: np.ndarray, stride: int) -> List[np.ndarray]:
    iy = stride_indices(xp.shape[0], stride)
    ix = stride_indices(xp.shape[1], stride)
    segments: List[np.ndarray] = []
    for row in iy:
        segments.append(np.column_stack([xp[row, :], yp[row, :]]))
    for col in ix:
        segments.append(np.column_stack([xp[:, col], yp[:, col]]))
    return segments


def make_2d_domain(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=DOMAIN_2D_FIGSIZE)
    ax.pcolormesh(
        snapshot.xp,
        snapshot.yp,
        np.linspace(0.0, 1.0, snapshot.xp.size).reshape(snapshot.xp.shape),
        cmap=mcolors.LinearSegmentedColormap.from_list("domain_metal_2d", [METALLIC_BASE_LOW, METALLIC_BASE_HIGH]),
        shading="gouraud",
    )
    boundary = np.array(
        [
            [snapshot.xp[0, 0], snapshot.yp[0, 0]],
            [snapshot.xp[0, -1], snapshot.yp[0, -1]],
            [snapshot.xp[-1, -1], snapshot.yp[-1, -1]],
            [snapshot.xp[-1, 0], snapshot.yp[-1, 0]],
            [snapshot.xp[0, 0], snapshot.yp[0, 0]],
        ]
    )
    ax.plot(boundary[:, 0], boundary[:, 1], color=METALLIC_EDGE, linewidth=DOMAIN_OUTLINE_LINEWIDTH)
    axis_equal_2d(ax, snapshot.xp, snapshot.yp)
    fig.text(0.06, 0.94, "domain", ha="left", va="top", fontsize=TITLE_FONT_SIZE)
    save_figure(fig, out_dir / "domain_2d", formats)
    plt.close(fig)


def make_2d_grid(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    fig, ax = plt.subplots(figsize=GRID_2D_FIGSIZE)
    ax.pcolormesh(
        snapshot.xp,
        snapshot.yp,
        np.zeros_like(snapshot.xp),
        color=MESH_FILL,
        alpha=MESH_FILL_ALPHA,
        shading="gouraud",
    )
    segments = grid_segments(snapshot.xp, snapshot.yp, stride=MESH_STRIDE)
    lc = LineCollection(
        segments,
        colors=MESH_LINE,
        linewidths=MESH_LINEWIDTH_2D,
        alpha=0.82,
    )
    ax.add_collection(lc)
    axis_equal_2d(ax, snapshot.xp, snapshot.yp)
    fig.text(0.06, 0.94, "mesh", ha="left", va="top", fontsize=TITLE_FONT_SIZE)
    save_figure(fig, out_dir / "mesh_2d", formats)
    plt.close(fig)


def make_2d_fields(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    for field_name in snapshot.field_names:
        values = snapshot.fields[field_name]
        _, norm = normalize_field(values)
        cmap = make_cmap(field_name)
        fig, ax = plt.subplots(figsize=FIELD_2D_FIGSIZE)
        im = ax.pcolormesh(
            snapshot.xp,
            snapshot.yp,
            values,
            cmap=cmap,
            norm=norm,
            shading="gouraud",
            rasterized=RASTERIZE_DENSE_2D,
        )
        add_2d_contours(ax, snapshot.xp, snapshot.yp, values)
        axis_equal_2d(ax, snapshot.xp, snapshot.yp)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2, width=0.5)
        cb.set_label(display_field_name(field_name), fontsize=COLORBAR_FONT_SIZE)
        save_figure(fig, out_dir / f"field_2d_{clean_field_name(field_name)}", formats)
        plt.close(fig)


def make_channel_2d_subplots(snapshot: Snapshot, out_dir: Path, formats: Sequence[str]) -> None:
    n_fields = len(snapshot.field_names)
    n_cols = 2
    n_rows = int(np.ceil(n_fields / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=CHANNEL_SUBPLOTS_FIGSIZE, squeeze=False)
    for ax in axes.ravel()[n_fields:]:
        ax.set_visible(False)
    for ax, field_name in zip(axes.ravel(), snapshot.field_names):
        values = snapshot.fields[field_name]
        _, norm = normalize_field(values)
        cmap = make_cmap(field_name)
        im = ax.pcolormesh(
            snapshot.xp,
            snapshot.yp,
            values,
            cmap=cmap,
            norm=norm,
            shading="gouraud",
            rasterized=RASTERIZE_DENSE_2D,
        )
        add_2d_contours(ax, snapshot.xp, snapshot.yp, values)
        axis_equal_2d(ax, snapshot.xp, snapshot.yp)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2, width=0.5)
        cb.set_label(display_field_name(field_name), fontsize=COLORBAR_FONT_SIZE)
    fig.subplots_adjust(left=0.02, right=0.96, bottom=0.03, top=0.97, wspace=0.14, hspace=0.10)
    save_figure(fig, out_dir / "channel_2d_subplots", formats)
    save_figure(fig, out_dir / "stacked_2d_composite", formats)
    plt.close(fig)


def normalize_to_01(values: np.ndarray) -> np.ndarray:
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if np.isclose(vmin, vmax):
        return np.zeros_like(values, dtype=np.float64)
    return ((values - vmin) / (vmax - vmin)).astype(np.float64)


def domain_boundary(snapshot: Snapshot) -> np.ndarray:
    return np.array(
        [
            [snapshot.xp[0, 0], snapshot.yp[0, 0]],
            [snapshot.xp[0, -1], snapshot.yp[0, -1]],
            [snapshot.xp[-1, -1], snapshot.yp[-1, -1]],
            [snapshot.xp[-1, 0], snapshot.yp[-1, 0]],
            [snapshot.xp[0, 0], snapshot.yp[0, 0]],
        ],
        dtype=np.float64,
    )


def gaussian_noise_field(shape: Tuple[int, int]) -> np.ndarray:
    rng = np.random.default_rng(NOISE_RANDOM_SEED)
    noise = rng.normal(loc=0.0, scale=NOISE_STD, size=shape)
    return normalize_to_01(noise)


def flow_cmap() -> mcolors.Colormap:
    return mcolors.LinearSegmentedColormap.from_list("flow_intermediate", FLOW_CMAP_COLORS)


def make_noise_visualizations(
    snapshot: Snapshot,
    out_dir: Path,
    formats: Sequence[str],
    noise: np.ndarray,
) -> None:
    iy = stride_indices(noise.shape[0], NOISE_SCATTER_STRIDE)
    ix = stride_indices(noise.shape[1], NOISE_SCATTER_STRIDE)
    xp = snapshot.xp[np.ix_(iy, ix)]
    yp = snapshot.yp[np.ix_(iy, ix)]
    values = noise[np.ix_(iy, ix)]
    boundary = domain_boundary(snapshot)

    fig, ax = plt.subplots(figsize=DOMAIN_2D_FIGSIZE)
    ax.scatter(
        xp.ravel(),
        yp.ravel(),
        c=values.ravel(),
        cmap=NOISE_CMAP,
        s=NOISE_SCATTER_SIZE_2D,
        linewidths=0,
        alpha=NOISE_ALPHA,
        rasterized=RASTERIZE_DENSE_2D,
    )
    ax.plot(boundary[:, 0], boundary[:, 1], color=METALLIC_EDGE, linewidth=DOMAIN_OUTLINE_LINEWIDTH)
    axis_equal_2d(ax, snapshot.xp, snapshot.yp)
    save_figure(fig, out_dir / "noise_layer_2d", formats)
    plt.close(fig)

    fig = plt.figure(figsize=INDIVIDUAL_3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    z = np.full_like(xp, BASE_Z, dtype=np.float64)
    ax.scatter(
        xp.ravel(),
        yp.ravel(),
        z.ravel(),
        c=values.ravel(),
        cmap=NOISE_CMAP,
        s=NOISE_SCATTER_SIZE_3D,
        linewidths=0,
        alpha=NOISE_ALPHA,
        depthshade=False,
    )
    ax.plot(boundary[:, 0], boundary[:, 1], np.full(boundary.shape[0], BASE_Z), color=METALLIC_EDGE, linewidth=DOMAIN_OUTLINE_LINEWIDTH)
    style_3d_axis(ax)
    set_3d_limits(ax, snapshot.xp, snapshot.yp, BASE_Z + 0.02, z_bottom=-0.08)
    save_figure(fig, out_dir / "noise_layer_3d", formats)
    plt.close(fig)


def make_scalar_2d(
    snapshot: Snapshot,
    values: np.ndarray,
    cmap: mcolors.Colormap | str,
    colorbar_label: str,
    out_base: Path,
    formats: Sequence[str],
    add_contours: bool = True,
) -> None:
    _, norm = normalize_field(values)
    fig, ax = plt.subplots(figsize=FIELD_2D_FIGSIZE)
    im = ax.pcolormesh(
        snapshot.xp,
        snapshot.yp,
        values,
        cmap=cmap,
        norm=norm,
        shading="gouraud",
        rasterized=RASTERIZE_DENSE_2D,
    )
    if add_contours:
        add_2d_contours(ax, snapshot.xp, snapshot.yp, values)
    axis_equal_2d(ax, snapshot.xp, snapshot.yp)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2, width=0.5)
    cb.set_label(colorbar_label, fontsize=COLORBAR_FONT_SIZE)
    save_figure(fig, out_base, formats)
    plt.close(fig)


def make_scalar_3d(
    snapshot: Snapshot,
    values: np.ndarray,
    cmap: mcolors.Colormap | str,
    colorbar_label: str,
    out_base: Path,
    formats: Sequence[str],
    height_scale: float = FLOW_HEIGHT_SCALE,
) -> None:
    normalized, norm = normalize_field(values)
    z = height_scale * normalized
    xp_s, yp_s, values_s, z_s = smooth_field_render_grid(snapshot, values, z)
    xp_d, yp_d, values_d, z_d = decimate(xp_s, yp_s, values_s, z_s, stride=FIELD_SURFACE_STRIDE)
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    fig = plt.figure(figsize=INDIVIDUAL_3D_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        xp_d,
        yp_d,
        z_d,
        facecolors=field_facecolors(values_d, z_d, cmap_obj, norm),
        rstride=1,
        cstride=1,
        linewidth=0,
        edgecolor="none",
        antialiased=False,
        shade=False,
    )
    if CONTOUR_ON_SMOOTH_3D_SURFACE:
        add_3d_contours(ax, xp_s, yp_s, values_s, norm, 0.0, height_scale)
    else:
        add_3d_contours(ax, snapshot.xp, snapshot.yp, values, norm, 0.0, height_scale)
    style_3d_axis(ax)
    set_3d_limits(
        ax,
        snapshot.xp,
        snapshot.yp,
        float(np.nanmax(z)),
        z_bottom=-0.06,
        z_compression=INDIVIDUAL_Z_AXIS_COMPRESSION,
    )
    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.036, pad=0.02, shrink=0.70)
    cb.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2, width=0.5)
    cb.set_label(colorbar_label, fontsize=COLORBAR_FONT_SIZE)
    save_figure(fig, out_base, formats)
    plt.close(fig)


def make_flow_matching_visualizations(
    snapshot: Snapshot,
    out_dir: Path,
    formats: Sequence[str],
) -> None:
    first_name = snapshot.field_names[0]
    first_field = snapshot.fields[first_name]
    first_norm = normalize_to_01(first_field)
    noise = gaussian_noise_field(first_field.shape)
    intermediate = (1.0 - FLOW_T) * noise + FLOW_T * first_norm
    flow_label = f"intermediate t={FLOW_T:.2f}"

    log(
        f"Building flow-matching panels with first field '{first_name}', "
        f"noise seed={NOISE_RANDOM_SEED}, interpolation t={FLOW_T:.2f}"
    )
    make_noise_visualizations(snapshot, out_dir, formats, noise)
    make_scalar_2d(
        snapshot,
        first_field,
        make_cmap(first_name),
        display_field_name(first_name),
        out_dir / f"first_field_mapping_2d_{clean_field_name(first_name)}",
        formats,
    )
    make_scalar_3d(
        snapshot,
        first_field,
        make_cmap(first_name),
        display_field_name(first_name),
        out_dir / f"first_field_mapping_3d_{clean_field_name(first_name)}",
        formats,
        height_scale=INDIVIDUAL_HEIGHT_SCALE,
    )
    make_scalar_2d(
        snapshot,
        intermediate,
        flow_cmap(),
        flow_label,
        out_dir / "flow_intermediate_2d",
        formats,
    )
    make_scalar_3d(
        snapshot,
        intermediate,
        flow_cmap(),
        flow_label,
        out_dir / "flow_intermediate_3d",
        formats,
        height_scale=FLOW_HEIGHT_SCALE,
    )

    fig, axes = plt.subplots(1, 3, figsize=FLOW_TRIPTYCH_FIGSIZE)
    panels = [
        (noise, plt.get_cmap(NOISE_CMAP), "noise"),
        (intermediate, flow_cmap(), flow_label),
        (first_field, make_cmap(first_name), display_field_name(first_name)),
    ]
    for ax, (values, cmap, label) in zip(axes, panels):
        _, norm = normalize_field(values)
        im = ax.pcolormesh(
            snapshot.xp,
            snapshot.yp,
            values,
            cmap=cmap,
            norm=norm,
            shading="gouraud",
            rasterized=RASTERIZE_DENSE_2D,
        )
        if label != "noise":
            add_2d_contours(ax, snapshot.xp, snapshot.yp, values)
        axis_equal_2d(ax, snapshot.xp, snapshot.yp)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.ax.tick_params(labelsize=COLORBAR_FONT_SIZE, length=2, width=0.5)
        cb.set_label(label, fontsize=COLORBAR_FONT_SIZE)
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.96, wspace=0.22)
    save_figure(fig, out_dir / "flow_matching_path_2d", formats)
    plt.close(fig)


def write_metadata(snapshot: Snapshot, out_dir: Path, args: argparse.Namespace, formats: Sequence[str]) -> None:
    metadata = {
        "input_file": str(snapshot.h5_path),
        "case_index": snapshot.case_index,
        "frame_index": snapshot.frame_index,
        "time_value": snapshot.time_value,
        "field_names": snapshot.field_names,
        "grid_shape": list(snapshot.x.shape),
        "formats": list(formats),
        "view": {
            "elevation": VIEW_ELEVATION,
            "azimuth": VIEW_AZIMUTH,
            "roll": VIEW_ROLL,
            "domain_shear_x": DOMAIN_SHEAR_X,
            "domain_y_scale": DOMAIN_Y_SCALE,
            "surface_stride": SURFACE_STRIDE,
            "field_surface_stride": FIELD_SURFACE_STRIDE,
            "mesh_stride": MESH_STRIDE,
            "field_height_scale": FIELD_HEIGHT_SCALE,
            "individual_height_scale": INDIVIDUAL_HEIGHT_SCALE,
            "field_surface_upsample": FIELD_SURFACE_UPSAMPLE,
            "field_surface_max_axis": FIELD_SURFACE_MAX_AXIS,
            "stacked_domain_z": BASE_Z,
            "stacked_mesh_z": stacked_mesh_z(),
            "stacked_first_field_z": stacked_field_z(0),
            "stacked_gap_controls": "mesh-to-domain, domain-to-first-field, and field-to-field spacing",
        },
        "contours": {
            "levels": CONTOUR_LEVELS,
            "color": CONTOUR_COLOR,
            "linewidth_2d": CONTOUR_LINEWIDTH_2D,
            "linewidth_3d": CONTOUR_LINEWIDTH_3D,
            "enable_3d_contours": ENABLE_3D_CONTOURS,
            "on_smooth_3d_surface": CONTOUR_ON_SMOOTH_3D_SURFACE,
            "shadow_color_3d": CONTOUR_SHADOW_COLOR_3D,
            "shadow_linewidth_3d": CONTOUR_SHADOW_LINEWIDTH_3D,
        },
        "flow_matching": {
            "noise_seed": NOISE_RANDOM_SEED,
            "noise_std": NOISE_STD,
            "interpolation_t": FLOW_T,
        },
        "sensors": {
            "count": SENSOR_COUNT,
            "random_seed": SENSOR_RANDOM_SEED,
            "field_measured_map": list(SENSOR_FIELD_MEASURED_MAP),
            "measured_fields": [
                name for i, name in enumerate(snapshot.field_names) if field_is_measured(i)
            ],
            "unmeasured_fields": [
                name for i, name in enumerate(snapshot.field_names) if not field_is_measured(i)
            ],
            "mesh_marker": {
                "edge_color": SENSOR_MESH_EDGE_COLOR,
                "face_color": SENSOR_MESH_FACE_COLOR,
                "size": SENSOR_SIZE_MESH_DOMAIN,
                "linewidth": SENSOR_LINEWIDTH,
            },
            "field_marker": {
                "measured_edge_color": SENSOR_MEASURED_EDGE_COLOR,
                "unmeasured_edge_color": SENSOR_UNMEASURED_EDGE_COLOR,
                "size": SENSOR_SIZE_FIELD,
            },
            "connection": {
                "color": SENSOR_CONNECTION_COLOR,
                "linestyle": SENSOR_CONNECTION_LINESTYLE,
                "linewidth": SENSOR_CONNECTION_LINEWIDTH,
                "alpha": SENSOR_CONNECTION_ALPHA,
                "z_lift": SENSOR_CONNECTION_Z_LIFT,
            },
        },
        "io_notes": (
            "HDF5 chunked/compressed datasets cannot be memory-mapped like raw "
            "contiguous numpy arrays. This script reads one case/time hyperslab, "
            "which matches the preprocessing chunk layout and avoids loading the "
            "full CFD file."
        ),
        "output_subdirs": OUTPUT_SUBDIRS,
        "args": vars(args),
    }
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create publication-quality stacked and 2D CFD dataset visualizations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-file", type=str, default=None, help="Optional direct HDF5 path.")
    parser.add_argument("--processed-root", type=str, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--resolution", choices=("H", "M", "L"), default=DEFAULT_RESOLUTION)
    parser.add_argument("--case-source", choices=("test", "all"), default=DEFAULT_CASE_SOURCE)
    parser.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    parser.add_argument("--test-case-index", type=int, default=DEFAULT_TEST_CASE_INDEX)
    parser.add_argument("--case-index", type=int, default=DEFAULT_CASE_INDEX, help="Explicit case index override.")
    parser.add_argument("--frame-index", type=int, default=DEFAULT_FRAME_INDEX, help="Time/frame index.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_FORMATS),
        choices=("svg", "pdf", "png"),
        help="One or more export formats.",
    )
    return parser.parse_args()


def main() -> None:
    configure_matplotlib()
    args = parse_args()
    try:
        run_t0 = time.perf_counter()
        snapshot = load_snapshot(args)
        out_dir = make_output_dir(args.output_root)
        out_paths = make_output_subdirs(out_dir)
        log(f"Created output directory: {out_dir}")
        log("Figure groups: " + ", ".join(f"{key}={path.relative_to(out_dir)}" for key, path in out_paths.items()))
        formats = tuple(dict.fromkeys(fmt.lower().lstrip(".") for fmt in args.formats))
        log(f"Export formats: {', '.join(formats)}")
        log("Rendering stacked 3D composite")
        make_composite_3d(snapshot, out_paths["composites"], formats)
        log("Rendering sensor-aware stacked 3D composite")
        make_composite_3d_sensor(snapshot, out_paths["composites"], formats)
        log("Rendering 3D and 2D channel subplot panels")
        make_channel_3d_subplots(snapshot, out_paths["composites"], formats)
        make_channel_2d_subplots(snapshot, out_paths["composites"], formats)
        log("Rendering standalone 3D domain and mesh")
        make_domain_3d(snapshot, out_paths["geometry_3d"], formats)
        make_mesh_3d(snapshot, out_paths["geometry_3d"], formats)
        log("Rendering isolated 3D physical fields")
        make_individual_3d(snapshot, out_paths["fields_3d"], formats)
        log("Rendering 2D mesh/domain/field maps")
        make_2d_grid(snapshot, out_paths["geometry_2d"], formats)
        make_2d_domain(snapshot, out_paths["geometry_2d"], formats)
        make_2d_fields(snapshot, out_paths["fields_2d"], formats)
        log("Rendering flow-matching noise/intermediate/final panels")
        make_flow_matching_visualizations(snapshot, out_paths["flow_matching"], formats)
        write_metadata(snapshot, out_paths["metadata"], args, formats)
        log(f"Wrote metadata: {out_paths['metadata'] / 'metadata.json'}")
        log(f"Completed full visualization suite in {elapsed_label(time.perf_counter() - run_t0)}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"Saved CFD visualization suite to: {out_dir}")


if __name__ == "__main__":
    main()
