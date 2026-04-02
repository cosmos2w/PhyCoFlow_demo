import argparse
import os
from pathlib import Path
from typing import Tuple, List

import h5py
import numpy as np


# ==========================================================
# Utilities
# ==========================================================

def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def normalize_to_minus1_1(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a 1D coordinate array to [-1, 1].
    If all values are identical, return zeros.
    """
    arr = np.asarray(arr, dtype=np.float64)
    amin = arr.min()
    amax = arr.max()
    if np.isclose(amax, amin):
        return np.zeros_like(arr, dtype=np.float32)
    out = 2.0 * (arr - amin) / (amax - amin) - 1.0
    return out.astype(np.float32)


def make_default_normalized_axis(n: int) -> np.ndarray:
    """
    Default axis when the raw dataset does not provide coordinates.
    """
    return np.linspace(-1.0, 1.0, n, dtype=np.float32)


def downsample_axis_by_mean(axis: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample a 1D coordinate array by block averaging.
    This makes the coarse coordinates equal to the centers of pooled cells/nodes.
    """
    axis = np.asarray(axis, dtype=np.float32)
    if axis.shape[0] % factor != 0:
        raise ValueError(f"Axis length {axis.shape[0]} is not divisible by factor {factor}.")
    return axis.reshape(axis.shape[0] // factor, factor).mean(axis=1).astype(np.float32)


def build_flattened_coordinates(x_axis: np.ndarray, y_axis: np.ndarray) -> np.ndarray:
    """
    Build flattened 2D coordinates with z=0.
    Output shape: [N_pts, 1, 1, 3]
    """
    xx, yy = np.meshgrid(x_axis, y_axis, indexing="xy")
    zz = np.zeros_like(xx, dtype=np.float32)
    coords = np.stack([xx, yy, zz], axis=-1).reshape(-1, 1, 1, 3).astype(np.float32)
    return coords


def average_pool_frames(frames: np.ndarray, factor: int) -> np.ndarray:
    """
    Average-pool frames in a physically sensible way.

    Accepts either:
      [T, H, W, C]
      [B, T, H, W, C]

    Returns the same rank with H and W downsampled by 'factor'.
    """
    frames = np.asarray(frames, dtype=np.float32)

    if frames.ndim == 4:
        t, h, w, c = frames.shape
        if h % factor != 0 or w % factor != 0:
            raise ValueError(f"Shape {(t, h, w, c)} is not divisible by factor {factor}.")
        out = frames.reshape(t, h // factor, factor, w // factor, factor, c).mean(axis=(2, 4))
        return out.astype(np.float32)

    if frames.ndim == 5:
        b, t, h, w, c = frames.shape
        if h % factor != 0 or w % factor != 0:
            raise ValueError(f"Shape {(b, t, h, w, c)} is not divisible by factor {factor}.")
        out = frames.reshape(b, t, h // factor, factor, w // factor, factor, c).mean(axis=(3, 5))
        return out.astype(np.float32)

    raise ValueError(f"Unsupported frame rank {frames.ndim}. Expected 4D or 5D array.")


def fields_to_standard_layout(frames: np.ndarray) -> np.ndarray:
    """
    Convert [T, H, W, C] -> [T, N_pts, 1, 1, C]
    """
    t, h, w, c = frames.shape
    return frames.reshape(t, h * w, 1, 1, c).astype(np.float32)


def create_standard_h5(
    out_path: str,
    n_cases: int,
    n_timesteps: int,
    n_pts: int,
    n_channels: int,
    coordinates: np.ndarray,
    time_values: np.ndarray,
    selected_fields: str,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> h5py.File:
    """
    Create an output HDF5 file following the unified structure requested by the user.
    """
    ensure_parent(out_path)
    f = h5py.File(out_path, "w")

    # Dummy conditions tensor required by the existing codebase.
    f.create_dataset("conditions", data=np.empty((1, 0), dtype=np.float32))

    # Coordinates are shared by all cases/time steps.
    f.create_dataset("coordinates", data=coordinates, dtype=np.float32)

    # Time axis
    f.create_dataset("time", data=time_values.astype(np.float64), dtype=np.float64)

    # Main fields tensor with chunking optimized for reading one case / one time snapshot.
    fields = f.create_dataset(
        "fields",
        shape=(n_cases, n_timesteps, n_pts, 1, 1, n_channels),
        dtype=np.float32,
        chunks=(1, 1, n_pts, 1, 1, n_channels),
        compression=compression,
        compression_opts=compression_opts,
    )

    fields.attrs["B"] = int(n_cases)
    fields.attrs["C"] = int(n_channels)
    fields.attrs["Nt"] = int(n_timesteps)
    fields.attrs["Nx"] = int(n_pts)
    fields.attrs["Ny"] = 1
    fields.attrs["Nz"] = 1
    fields.attrs["mesh"] = "structured"
    fields.attrs["selected_fields"] = selected_fields

    return f


# ==========================================================
# Reaction-diffusion processing
# ==========================================================

def get_reaction_diffusion_sample_array(obj) -> np.ndarray:
    """
    Resolve one reaction-diffusion sample into a numpy array of shape [T, H, W, C].

    The raw PDEBench file may store each sample as:
      1) a dataset directly, or
      2) a group containing exactly one dataset, or
      3) a group containing a dataset named 'data' or similar.

    This helper makes the loader robust to those variants.
    """
    # Case 1: already a dataset
    if isinstance(obj, h5py.Dataset):
        return obj[...].astype(np.float32)

    # Case 2/3: group containing the actual dataset
    if isinstance(obj, h5py.Group):
        # First try common names
        preferred_names = ["data", "field", "fields", "u", "tensor", "solution"]
        for name in preferred_names:
            if name in obj and isinstance(obj[name], h5py.Dataset):
                return obj[name][...].astype(np.float32)

        # Otherwise, search all child datasets recursively and require exactly one
        found = []

        def visitor(name, item):
            if isinstance(item, h5py.Dataset):
                found.append(item)

        obj.visititems(visitor)

        if len(found) == 1:
            return found[0][...].astype(np.float32)

        if len(found) == 0:
            raise ValueError(
                "Reaction-diffusion sample group does not contain any dataset."
            )

        raise ValueError(
            "Reaction-diffusion sample group contains multiple datasets and the "
            "target array is ambiguous. Please inspect the group structure."
        )

    raise TypeError(f"Unsupported HDF5 object type: {type(obj)}")

def process_reaction_diffusion(
    raw_path: str,
    out_h: str,
    out_m: str,
    out_l: str,
) -> None:
    """
    Raw format:
      keys: '0000', '0001', ...
      each key -> [T=101, H=128, W=128, C=2]
    """
    print("\n=== Processing PDEBench reaction-diffusion dataset ===")
    print(f"Raw file: {raw_path}")

    with h5py.File(raw_path, "r") as src:

        sample_keys = sorted(list(src.keys()))
        print(f"First 5 top-level keys: {sample_keys[:5]}")
        print(f"Type of first sample object: {type(src[sample_keys[0]])}")

        n_cases = len(sample_keys)
        if n_cases == 0:
            raise ValueError("No samples found in reaction-diffusion HDF5 file.")

        example = get_reaction_diffusion_sample_array(src[sample_keys[0]])
        nt, h, w, c = example.shape
        print(f"Detected shape per sample: (Nt={nt}, H={h}, W={w}, C={c})")

        # No coordinates provided in the raw file -> default normalized grid.
        x_h = make_default_normalized_axis(w)
        y_h = make_default_normalized_axis(h)
        x_m = downsample_axis_by_mean(x_h, factor=2)
        y_m = downsample_axis_by_mean(y_h, factor=2)
        x_l = downsample_axis_by_mean(x_h, factor=4)
        y_l = downsample_axis_by_mean(y_h, factor=4)

        coords_h = build_flattened_coordinates(x_h, y_h)
        coords_m = build_flattened_coordinates(x_m, y_m)
        coords_l = build_flattened_coordinates(x_l, y_l)

        # No time axis provided -> default evenly spaced integer indices.
        time_values = np.arange(nt, dtype=np.float64)

        fh = create_standard_h5(
            out_path=out_h,
            n_cases=n_cases,
            n_timesteps=nt,
            n_pts=h * w,
            n_channels=c,
            coordinates=coords_h,
            time_values=time_values,
            selected_fields="species_0,species_1",
        )
        fm = create_standard_h5(
            out_path=out_m,
            n_cases=n_cases,
            n_timesteps=nt,
            n_pts=(h // 2) * (w // 2),
            n_channels=c,
            coordinates=coords_m,
            time_values=time_values,
            selected_fields="species_0,species_1",
        )
        fl = create_standard_h5(
            out_path=out_l,
            n_cases=n_cases,
            n_timesteps=nt,
            n_pts=(h // 4) * (w // 4),
            n_channels=c,
            coordinates=coords_l,
            time_values=time_values,
            selected_fields="species_0,species_1",
        )

        try:
            for i, key in enumerate(sample_keys):
                frames_h = get_reaction_diffusion_sample_array(src[key])   # [T, H, W, C]
                frames_m = average_pool_frames(frames_h, factor=2)      # [T, 64, 64, C]
                frames_l = average_pool_frames(frames_h, factor=4)      # [T, 32, 32, C]

                fh["fields"][i] = fields_to_standard_layout(frames_h)
                fm["fields"][i] = fields_to_standard_layout(frames_m)
                fl["fields"][i] = fields_to_standard_layout(frames_l)

                if (i + 1) % 50 == 0 or i == 0 or (i + 1) == n_cases:
                    print(f"Processed RD samples: {i + 1}/{n_cases}")
        finally:
            fh.close()
            fm.close()
            fl.close()

    print("Finished processing reaction-diffusion dataset.")
    print(f"Saved: {out_h}")
    print(f"Saved: {out_m}")
    print(f"Saved: {out_l}")


# ==========================================================
# Compressible Navier-Stokes / CFD processing
# ==========================================================

def process_cfd(
    raw_path: str,
    out_h: str,
    out_m: str,
    out_l: str,
    chunk_cases: int = 8,
) -> None:
    """
    Raw format:
      Vx       : [B=10000, T=21, H=128, W=128]
      Vy       : [B=10000, T=21, H=128, W=128]
      density  : [B=10000, T=21, H=128, W=128]
      pressure : [B=10000, T=21, H=128, W=128]
      x-coordinate: [128]
      y-coordinate: [128]
      t-coordinate: [22]
    """
    print("\n=== Processing PDEBench 2-D CFD dataset ===")
    print(f"Raw file: {raw_path}")
    print(f"Chunk size (cases per write): {chunk_cases}")

    with h5py.File(raw_path, "r") as src:
        vx = src["Vx"]
        vy = src["Vy"]
        rho = src["density"]
        p = src["pressure"]

        n_cases, nt, h, w = vx.shape
        print(f"Detected field tensor shape: (B={n_cases}, Nt={nt}, H={h}, W={w})")

        # Normalize raw coordinates to [-1, 1].
        x_h = normalize_to_minus1_1(src["x-coordinate"][:])
        y_h = normalize_to_minus1_1(src["y-coordinate"][:])

        # Coarse-grid coordinates are defined as the centers of pooled cells/nodes.
        x_m = downsample_axis_by_mean(x_h, factor=2)
        y_m = downsample_axis_by_mean(y_h, factor=2)
        x_l = downsample_axis_by_mean(x_h, factor=4)
        y_l = downsample_axis_by_mean(y_h, factor=4)

        coords_h = build_flattened_coordinates(x_h, y_h)
        coords_m = build_flattened_coordinates(x_m, y_m)
        coords_l = build_flattened_coordinates(x_l, y_l)

        # The provided time coordinate length is 22 while Nt=21 in the data.
        # We interpret this as edge coordinates and convert them to cell centers.
        t_raw = np.asarray(src["t-coordinate"][:], dtype=np.float64)
        if t_raw.shape[0] == nt + 1:
            time_values = 0.5 * (t_raw[:-1] + t_raw[1:])
            print(
                "Detected t-coordinate length Nt+1. Using midpoint time centers "
                "to align with the 21 stored frames."
            )
        elif t_raw.shape[0] == nt:
            time_values = t_raw
        else:
            raise ValueError(
                f"Unexpected CFD time axis length {t_raw.shape[0]} for field Nt={nt}."
            )

        fh = create_standard_h5(
            out_path=out_h,
            n_cases=n_cases,
            n_timesteps=nt,
            n_pts=h * w,
            n_channels=4,
            coordinates=coords_h,
            time_values=time_values,
            selected_fields="Vx,Vy,density,pressure",
        )
        fm = create_standard_h5(
            out_path=out_m,
            n_cases=n_cases,
            n_timesteps=nt,
            n_pts=(h // 2) * (w // 2),
            n_channels=4,
            coordinates=coords_m,
            time_values=time_values,
            selected_fields="Vx,Vy,density,pressure",
        )
        fl = create_standard_h5(
            out_path=out_l,
            n_cases=n_cases,
            n_timesteps=nt,
            n_pts=(h // 4) * (w // 4),
            n_channels=4,
            coordinates=coords_l,
            time_values=time_values,
            selected_fields="Vx,Vy,density,pressure",
        )

        try:
            for start in range(0, n_cases, chunk_cases):
                end = min(start + chunk_cases, n_cases)
                b = end - start

                # Build [B, T, H, W, C] chunk without loading the whole file into memory.
                chunk_h = np.stack(
                    [
                        vx[start:end],
                        vy[start:end],
                        rho[start:end],
                        p[start:end],
                    ],
                    axis=-1,
                ).astype(np.float32)  # [B, T, H, W, 4]

                chunk_m = average_pool_frames(chunk_h, factor=2)   # [B, T, 64, 64, 4]
                chunk_l = average_pool_frames(chunk_h, factor=4)   # [B, T, 32, 32, 4]

                fh["fields"][start:end] = chunk_h.reshape(b, nt, h * w, 1, 1, 4)
                fm["fields"][start:end] = chunk_m.reshape(b, nt, (h // 2) * (w // 2), 1, 1, 4)
                fl["fields"][start:end] = chunk_l.reshape(b, nt, (h // 4) * (w // 4), 1, 1, 4)

                print(f"Processed CFD samples: {end}/{n_cases}")
        finally:
            fh.close()
            fm.close()
            fl.close()

    print("Finished processing CFD dataset.")
    print(f"Saved: {out_h}")
    print(f"Saved: {out_m}")
    print(f"Saved: {out_l}")


# ==========================================================
# Main
# ==========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess PDEBench datasets into the unified HDF5 format used by the project."
    )

    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root containing Dataset/PDE_Bench and src/",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["rd", "cfd", "all"],
        help="Which dataset to preprocess.",
    )
    parser.add_argument(
        "--cfd-chunk-cases",
        type=int,
        default=8,
        help="Number of CFD cases loaded/written per chunk.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.project_root).resolve()

    raw_rd = root / "Dataset" / "PDE_Bench" / "2D" / "diffusion-reaction" / "2D_diff-react_NA_NA.h5"
    raw_cfd = root / "Dataset" / "PDE_Bench" / "2D" / "CFD" / "2d_cfd.hdf5"
    out_dir = root / "Dataset" / "PDE_Bench" / "Processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep the output names exactly as requested by the user.
    rd_h = out_dir / "RD_H_Res.h5"
    rd_m = out_dir / "RD_M_Res.h5"
    rd_l = out_dir / "RD_L_Res.h5"

    cfd_h = out_dir / "CFD_H_res.h5"
    cfd_m = out_dir / "CFD_M_res.h5"
    cfd_l = out_dir / "CFD_L_res.h5"

    if args.dataset in ["rd", "all"]:
        if not raw_rd.exists():
            raise FileNotFoundError(f"Reaction-diffusion file not found: {raw_rd}")
        process_reaction_diffusion(
            raw_path=str(raw_rd),
            out_h=str(rd_h),
            out_m=str(rd_m),
            out_l=str(rd_l),
        )

    if args.dataset in ["cfd", "all"]:
        if not raw_cfd.exists():
            raise FileNotFoundError(f"CFD file not found: {raw_cfd}")
        process_cfd(
            raw_path=str(raw_cfd),
            out_h=str(cfd_h),
            out_m=str(cfd_m),
            out_l=str(cfd_l),
            chunk_cases=args.cfd_chunk_cases,
        )


if __name__ == "__main__":
    main()
    print("Done.")
