
"""
Visualise a *single* variable contained in the CFD-like HDF5 files that have the
layout

    ├── coordinates   - (N_pts, 1, 1, 3)   - x, y, z of every grid point
    ├── fields        - (1, 1001, N_pts, 1, 1, N_c)
    └── time          - (1001,)             - physical time stamps

Typical spatial resolution: 512*512 = 262 144 points.

The script can

1. Export one PNG for a chosen time index (`N_T = 1`), *or*
2. Export one PNG **per** time index **and** stitch those PNGs into a GIF
   (`N_T > 1` *and* `--create-gif`).

The output file names follow

    Case_<case>_ch<channel>_t<T_ini>_NT<N_T>.png
    Case_<case>_ch<channel>_t<T_ini>_NT<N_T>.gif      (only if GIF requested)
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # head-less
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import imageio.v2 as imageio                # single dependency for gifs


# -----------------------------------------------------------------------------#
#                              helper functions                                #
# -----------------------------------------------------------------------------#
def load_case_file(case_number: int, root: Path) -> Path:
    """
    Convert a case number to the on-disk path and check that the file exists.
    """
    h5_path = root / "Dataset" / f"Merged_CH4COTU1P.h5"
    if not h5_path.exists():
        sys.exit(f"ERROR: file not found: {h5_path}")
    return h5_path


def fetch_data(
    h5_path: Path, case_number: int, channel: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return
        X      - (Npts, 2)
        times  - (Nt,)
        field  - (Nt, Npts)
    The first axis of the HDF5 dataset *fields* is the case selector.
    """
    BASE_CASE = 0          # first case stored in the file

    with h5py.File(h5_path, "r") as f:
        # ---------------- coordinates (same for all cases) ------------------
        xyz = f["coordinates"][...]              # (Npts, 1, 1, 3)
        xyz = xyz.reshape(-1, 3)
        X   = xyz[:, :2]                         # keep x, y only
        print(f'All coordinate.shape is {X.shape}')
        print(f'All coordinate is {X}')

        # ---------------- time (same for all cases) -------------------------
        times = f["time"][...]                  # (Nt,)

        # ---------------- fields -------------------------------------------
        fld_ds = f["fields"]                    # (Ncase, Nt, Npts, 1, 1, N_c)
        case_idx = case_number - BASE_CASE
        if not (0 <= case_idx < fld_ds.shape[0]):
            raise ValueError(
                f"Case {case_number} not in file "
                f"(valid: {BASE_CASE} … {BASE_CASE + fld_ds.shape[0] - 1})"
            )

        field = fld_ds[case_idx, :, :, 0, 0, channel]   # (Nt, Npts)
        print(f'field.shape is {field.shape}')

    return X, times, field

def triangulation(X: np.ndarray) -> mtri.Triangulation:
    """
    One triangulation for all frames (unstructured meshes work too).
    """
    x, y = X[:, 0], X[:, 1]
    return mtri.Triangulation(x, y)

def _create_png(
    field: np.ndarray, times: np.ndarray, triang: mtri.Triangulation,
    t_idx: int, out: Path, vmin: float, vmax: float, cmap: str = "viridis"
) -> None:
    """
    Render *one* frame (filled contour) and write it to *out*.
    """
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    im = ax.tricontourf(
        triang, field[t_idx], levels=200, cmap=cmap, vmin=vmin, vmax=vmax
    )
    cb = plt.colorbar(im)
    cb.set_label("u")
    ax.set_aspect("equal")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"t = {times[t_idx]:.4f}")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  ✓ {out.name}")

def __create_png(
    field: np.ndarray, times: np.ndarray, triang: mtri.Triangulation,
    t_idx: int, out: Path, vmin: float, vmax: float, cmap: str = "viridis",
    show_mesh: bool = False, shading: str = "gouraud"
) -> None:
    """
    Render *one* frame and write it to *out*.
    Uses tripcolor to accurately reflect grid resolution instead of over-smoothing.
    """
    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    # tripcolor respects the underlying mesh resolution much better than tricontourf
    im = ax.tripcolor(
        triang, field[t_idx], cmap=cmap, vmin=vmin, vmax=vmax, shading=shading
    )
    
    # Overlaying the mesh lines makes the resolution differences immediately obvious
    if show_mesh:
        ax.triplot(triang, 'w-', linewidth=0.2, alpha=0.3)

    cb = plt.colorbar(im)
    cb.set_label("u")
    
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t = {times[t_idx]:.4f}")
    
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  ✓ {out.name}")

# Scatter version
def create_png(
    field: np.ndarray, times: np.ndarray, triang: mtri.Triangulation,
    t_idx: int, out: Path, vmin: float, vmax: float, cmap: str = "viridis",
    marker_size: float = 5.0  # Adjustable marker size for the scatter points
) -> None:
    """
    Render *one* frame (scatter plot) and write it to *out*.
    Scatter plots map individual data nodes, perfectly highlighting resolution density.
    """
    plt.figure(figsize=(5, 4))
    ax = plt.gca()

    # The triangulation object holds the raw x and y coordinates of your nodes
    x = triang.x
    y = triang.y

    # Scatter plots place a dot exactly at each node location
    im = ax.scatter(
        x, y,
        c=field[t_idx],       # Color maps to your physical field (e.g., density, velocity)
        s=marker_size,        # Size of the dots
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        edgecolors='none'     # Removes point borders so dense areas don't turn solid black
    )
    
    cb = plt.colorbar(im)
    cb.set_label("u")
    
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"t = {times[t_idx]:.4f}")
    
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  ✓ {out.name}")

def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--case", type=int, default = 900,
                        help="case number (e.g. 0)")
    parser.add_argument("--channel", type=int, default = 0,
                        help="0 ≤ channel ≤ 4")
    parser.add_argument("--T_ini", type=int, default = 95,
                        help="first time index to plot (0-based)")
    parser.add_argument("--N_T", type=int, default = 1,
                        help="number of *consecutive* time steps to plot")
    parser.add_argument("--create_gif", type=bool, default = False,
                        help="also export a GIF (requires N_T > 1)")
    parser.add_argument("--out_dir", type=Path, default=Path("Save_reconstruction_files/ForViewDataset"),
                        help="directory to save the figures")
    parser.add_argument("--cmap", default="coolwarm",
                        help="matplotlib colormap")
    args = parser.parse_args()

    root    = Path(__file__).resolve().parent.parent   # project/
    # h5_path = root / "Dataset" / "Merged_CH4COTU1P.h5"
    data_TYPE = 'RD'
    Res = 'L' # H, M, L
    h5_path = root / "Dataset" / "PDE_Bench" / "Processed" / f"{data_TYPE}_{Res}_Res.h5"

    # Dataset/PDE_Bench/Processed/RD_H_Res.h5
    # Dataset/PDE_Bench/Processed/RD_M_Res.h5
    # Dataset/PDE_Bench/Processed/RD_L_Res.h5
    # Dataset/PDE_Bench/Processed/CFD_H_Res.h5
    # Dataset/PDE_Bench/Processed/CFD_M_Res.h5
    # Dataset/PDE_Bench/Processed/CFD_L_Res.h5

    if not h5_path.exists():
        sys.exit(f"ERROR: file not found: {h5_path}")

    print(f"Reading '{h5_path.name}' …")
    X, times, field = fetch_data(h5_path, args.case, args.channel)

    Nt, _ = field.shape
    if not (0 <= args.T_ini < Nt):
        sys.exit(f"T_ini must be within [0, {Nt-1}]")
    if args.T_ini + args.N_T > Nt:
        sys.exit("T_ini + N_T exceeds available frames")

    # ------------------------------------------------------------------ plots
    tri = triangulation(X)

    # fmin, fmax = field.min(), field.max()
    print(f'\nUsing hand-set min-max range!!!\n')
    fmin, fmax = -1, 1

    t_indices = range(args.T_ini, args.T_ini + args.N_T)

    out_subdir = (
        args.out_dir
        / f"Case_{args.case}"
        / f"ch{args.channel}"
    )
    out_subdir.mkdir(parents=True, exist_ok=True)

    png_paths: list[Path] = []
    for t_idx in t_indices:
        png_name = (
            f"Case_{args.case}_Res{Res}_ch{args.channel}_t{t_idx:04d}_NT{args.N_T}.png"
            if args.N_T > 1
            else f"Case_{args.case}_Res{Res}_ch{args.channel}_t{t_idx:04d}.png"
        )
        png_path = out_subdir / png_name
        create_png(
            field, times, tri, t_idx, png_path,
            vmin=fmin, vmax=fmax, cmap=args.cmap
        )
        png_paths.append(png_path)

    # ------------------------------------------------------------------ GIF
    if args.create_gif and args.N_T > 1:
        gif_name = (
            f"Case_{args.case}_Res{Res}_ch{args.channel}_t{args.T_ini}_NT{args.N_T}.gif"
        )
        gif_path = out_subdir / gif_name
        print(f"Writing GIF {gif_path.name} …")
        with imageio.get_writer(gif_path, mode="I", duration=0.2) as writer:
            for png in png_paths:
                writer.append_data(imageio.imread(png))
        print(f"  ✓ {gif_path.name}")

    print("Done.")


if __name__ == "__main__":
    main()