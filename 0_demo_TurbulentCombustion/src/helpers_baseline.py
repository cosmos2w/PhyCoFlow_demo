'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training
'''

# ═════════ Imports ═════════
import os
import csv
import math
import shutil
import torch
import torch.nn.functional as F
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon

import h5py
from datetime import datetime
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Union
from scipy.ndimage import binary_dilation

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

def normalize_field_names(value: Optional[Union[str, Sequence[object]]]) -> Optional[Tuple[str, ...]]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        names = [part.strip() for part in value.split(",")]
    else:
        names = []
        for item in value:
            if isinstance(item, bytes):
                item = item.decode("utf-8")
            names.extend(part.strip() for part in str(item).split(","))
    names = [name for name in names if name]
    return tuple(names) if names else None


def resolve_field_names(
    h5_file: h5py.File,
    config_field_names: Optional[Union[str, Sequence[object]]] = None,
    default_field_names: Sequence[str] = FIELD_NAMES,
) -> Tuple[str, ...]:
    num_fields = int(h5_file["fields"].shape[-1])
    candidates = (
        h5_file["fields"].attrs.get("selected_fields"),
        h5_file.attrs.get("selected_fields"),
        config_field_names,
        default_field_names,
    )
    for candidate in candidates:
        names = normalize_field_names(candidate)
        if names is not None and len(names) == num_fields:
            return names
    return tuple(f"field_{i}" for i in range(num_fields))

__all__ = [
    "FIELD_NAMES",
    "normalize_coords",
    "create_recon_dir",
    "TurbulentCombustionH5Dataset",
    "NonlinearPoissonDataset",
    "ElasticityDataset",
    "AirfoilCGridDataset",
    "CarCFDDataset",
    "collate_snapshots",
    "validate_regular_grid_compatibility",
    "pointcloud_to_grid",
    "splat_to_grid",
    "gather_from_grid",
    "splat_obs_to_grid",
    "build_sparse_condition",
    "nearest_fill_grid",
    "MetricsLogger",
    "visualize_reconstruction",
    "find_latest_run_dir",
    "extract_run_timestamp",
    "backup_path",
    "backup_existing_artifact",
    # ── Shared utilities used across methods (SiT, S3GM, ...) ────────────
    "set_seed",
    "compute_pad_size",
    "pointcloud_to_grid_padded",
    "grid_to_pointcloud",
    "build_obs_grid_mask",
    "scatter_sensors_to_nodes",
]

# ═════════ §1. Utility helpers ═════════

def _to_int_list(x: Union[int, Sequence[int], None]) -> list[int]:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [int(v) for v in x]
    return [int(x)]

def _broadcast_per_field(
    values: Union[int, Sequence[int]],
    cond_fields: Sequence[int],
    name: str,
) -> list[int]:
    values = _to_int_list(values)
    if len(values) == 1:
        values = values * len(cond_fields)
    if len(values) != len(cond_fields):
        raise ValueError(
            f"{name} must have length 1 or match len(cond_fields). "
            f"Got {len(values)} vs {len(cond_fields)}."
        )
    return values

def _normalized_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    scale = (cmax - cmin).clamp_min(1e-8)
    return (coords - cmin) / scale

def create_recon_dir(base_dir: str, Demo_Num: int, timestamp: str,
                     method_name: str = "ffm_tc_pointcloud") -> str:
    """Creates a timestamped directory for saving evaluation plots."""
    path = os.path.join(base_dir, method_name, f"demo_N{Demo_Num}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


def _torch_load_compat(path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

# ═════════ §2. Datasets ═════════

class TurbulentCombustionH5Dataset(Dataset):
    """Treat each time snapshot as one point-cloud sample."""

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        field_names: Optional[Union[str, Sequence[object]]] = None,
        stats_path: Optional[str] = None,
        stats_chunk: int = 32,
        time_stride: int = 1,
    ) -> None:
        super().__init__()
        self.h5_path     = str(h5_path)
        self.split       = split
        self.field_names = tuple(FIELD_NAMES)
        self.stats_chunk = stats_chunk
        self.time_stride = time_stride
        self._h5         = None

        with h5py.File(self.h5_path, "r") as f:
            self.num_times  = int(f["fields"].shape[1])
            raw_coords      = torch.from_numpy(f["coordinates"][:, 0, 0, :].astype(np.float32))

            self.coords_raw = raw_coords.clone()
            self.coords     = normalize_coords(raw_coords)
            self.num_points = int(raw_coords.shape[0])
            self.num_fields = int(f["fields"].shape[-1])
            self.times      = torch.from_numpy(f["time"][:].astype(np.float32))
            self.field_names = resolve_field_names(f, config_field_names=field_names)

        all_indices = np.arange(0, self.num_times, self.time_stride, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)
        n_train = int(len(all_indices) * train_ratio)
        if split == "train":
            self.indices = all_indices[:n_train]
        elif split in ["val", "test"]:
            self.indices = all_indices[n_train:]
        else:
            raise ValueError(f"Unknown split: {split}")

        self.indices = np.sort(self.indices)
        self.stats_path = stats_path or str(Path(self.h5_path).with_suffix(".stats.pt"))
        self.mean, self.std = self._load_or_compute_stats(train_indices=np.sort(all_indices[:n_train]))

    def _require_h5(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def _load_or_compute_stats(self, train_indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        stats_path = Path(self.stats_path)
        if stats_path.exists():
            obj = _torch_load_compat(stats_path, map_location="cpu")
            return obj["mean"].float(), obj["std"].float()

        h5 = self._require_h5()
        total_sum = torch.zeros(self.num_fields, dtype=torch.float64)
        total_sq = torch.zeros(self.num_fields, dtype=torch.float64)
        total_count = 0

        for start in range(0, len(train_indices), self.stats_chunk):
            idx = train_indices[start : start + self.stats_chunk]
            arr = h5["fields"][0, idx, :, 0, 0, :]  # [Tchunk, N, C]
            x = torch.from_numpy(arr.astype(np.float32))
            total_sum += x.sum(dim=(0, 1), dtype=torch.float64)
            total_sq += (x.double() ** 2).sum(dim=(0, 1))
            total_count += x.shape[0] * x.shape[1]

        mean = (total_sum / total_count).float()
        var = (total_sq / total_count - mean.double() ** 2).clamp_min(1e-12).float()
        std = torch.sqrt(var)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, stats_path)
        return mean, std

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        t_idx = int(self.indices[i])
        h5 = self._require_h5()
        x = h5["fields"][0, t_idx, :, 0, 0, :].astype(np.float32)
        x = torch.from_numpy(x)
        x = (x - self.mean) / self.std
        return {
            "coords": self.coords.clone(),          # normalized coordinates for model
            "coords_raw": self.coords_raw.clone(),  # original physical coordinates for plotting
            "fields": x,                    
            "time_index": torch.tensor(t_idx, dtype=torch.long),
            "physical_time": self.times[t_idx].clone(),
        }

# -------------------------------------------

class NonlinearPoissonDataset(Dataset):
    """
    Wraps the neuraloperator nonlinear Poisson .obj dataset into the same
    interface used by TurbulentCombustionH5Dataset so that the training
    scripts can switch datasets via a single flag.

    Each instance solves:  div((1 + 0.1 u^2) grad u) = f(x)
    on a 2-D domain with an irregular boundary parameterised by (c1, c2).

    The returned dict has the same keys as TurbulentCombustionH5Dataset:
        coords      [N, 3]   normalised coordinates (z=0 padding for 3-D compat)
        coords_raw  [N, 2]   original 2-D coordinates
        fields      [N, 1]   normalised solution u(x)
        time_index           instance index (no notion of time here)
        physical_time        dummy 0.0
    """
    FIELD_NAMES = ('u',)

    def __init__(self, obj_path, split, train_ratio, seed,
                 n_points, n_bound, field_names=None, stats_path=None):
        super().__init__()
        import pickle
        import random as _random

        self.field_names = field_names if field_names else self.FIELD_NAMES
        self.n_points = n_points
        self.n_bound = n_bound

        with open(obj_path, 'rb') as f:
            raw_data = pickle.load(f)

        rng = _random.Random(seed)
        rng.shuffle(raw_data)
        n_train = int(len(raw_data) * train_ratio)
        if split == 'train':
            instances = raw_data[:n_train]
        elif split in ('val', 'test'):
            instances = raw_data[n_train:]
        else:
            raise ValueError(f'Unknown split: {split}')

        self._samples = []
        for inst in instances:
            bp = torch.tensor(inst['train_points_boundary'][:n_bound], dtype=torch.float32)
            dp = torch.tensor(inst['train_points_domain'][:n_points], dtype=torch.float32)
            coords_2d = torch.cat([bp, dp], dim=0)
            bv = torch.tensor(inst['val_values_boundary'][:n_bound], dtype=torch.float32)
            dv = torch.tensor(inst['val_values_domain'][:n_points], dtype=torch.float32)
            u_vals = torch.cat([bv, dv], dim=0).unsqueeze(-1)
            self._samples.append((coords_2d, u_vals))

        stats_path = Path(stats_path) if stats_path else Path(obj_path).with_suffix('.stats.pt')
        if stats_path.exists():
            obj = torch.load(stats_path, map_location='cpu')
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
        elif split == 'train':
            all_u = torch.cat([s[1] for s in self._samples], dim=0)
            self.mean = all_u.mean(dim=0)
            self.std = all_u.std(dim=0).clamp_min(1e-08)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': self.mean, 'std': self.std}, stats_path)
        else:
            raise FileNotFoundError(
                f'Stats file {stats_path} not found. Run the train split first.'
            )

        self.num_fields = 1
        ref_coords = self._samples[0][0]
        self.num_points_total = ref_coords.shape[0]
        self.num_points = self.num_points_total
        self.coords_raw = ref_coords.clone()
        self.coords = normalize_coords(torch.cat(
            [ref_coords, torch.zeros(ref_coords.shape[0], 1)], dim=-1
        ))

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, i):
        coords_2d, u_vals = self._samples[i]
        u_norm = (u_vals - self.mean) / self.std
        coords_3d = torch.cat(
            [coords_2d, torch.zeros(coords_2d.shape[0], 1)], dim=-1
        )
        coords_norm = normalize_coords(coords_3d)
        return {
            'coords': coords_norm,
            'coords_raw': coords_2d,
            'fields': u_norm,
            'time_index': torch.tensor(i, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }

class ElasticityDataset(Dataset):
    """
    Loads the elasticity unit-cell dataset (irregular hole geometry).

    Each sample is a 41x41 regular-grid snapshot with two fields:
        field 0 — sigma (von-Mises stress)
        field 1 — mask  (1 = material, 0 = hole interior)

    The task is sparse inversion: given sparse sigma sensors on the
    material region, reconstruct the full stress field *and* the
    geometry (mask).
    """
    FIELD_NAMES = ('sigma', 'mask')

    def __init__(self, data_dir, split, train_ratio, seed,
                 field_names=None, stats_path=None):
        super().__init__()
        self.field_names = field_names if field_names else self.FIELD_NAMES
        self.data_dir = data_dir

        sigma_all = np.load(os.path.join(data_dir, 'Interp', 'Random_UnitCell_sigma_10_interp.npy'))
        mask_all = np.load(os.path.join(data_dir, 'Interp', 'Random_UnitCell_mask_10_interp.npy'))
        Ny, Nx, N_samples = sigma_all.shape
        self.grid_Ny = Ny
        self.grid_Nx = Nx
        # grid_shape=(Ny,Nx) consumed by helpers._build_structured_triangulation
        # (see train_latentfm_baseline.py:258, train_sit_baseline.py:645,
        # evaluate_ffm.py:714 for the same (Ny,Nx) convention).
        self.grid_shape = (Ny, Nx)

        yy, xx = np.meshgrid(
            np.linspace(0, 1, Ny), np.linspace(0, 1, Nx), indexing='ij'
        )
        coords_2d = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)
        coords_3d = np.concatenate(
            [coords_2d, np.zeros((coords_2d.shape[0], 1), dtype=np.float32)], axis=-1
        )
        self.coords = torch.from_numpy(coords_3d)
        self.coords_raw = torch.from_numpy(coords_2d)
        self.num_points = Ny * Nx

        sigma_flat = sigma_all.reshape(-1, N_samples).T.astype(np.float32)
        mask_flat = mask_all.reshape(-1, N_samples).T.astype(np.float32)
        self._fields = np.stack([sigma_flat, mask_flat], axis=-1)
        self.num_fields = 2

        all_indices = np.arange(N_samples, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)
        n_train = int(N_samples * train_ratio)
        if split == 'train':
            self.indices = np.sort(all_indices[:n_train])
        elif split in ('val', 'test'):
            self.indices = np.sort(all_indices[n_train:])
        else:
            raise ValueError(f'Unknown split: {split}')

        stats_file = Path(stats_path) if stats_path else Path(data_dir) / 'elasticity_stats.pt'
        if stats_file.exists():
            obj = torch.load(stats_file, map_location='cpu')
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
            return

        train_fields = self._fields[all_indices[:n_train]]
        self.mean = torch.from_numpy(train_fields.mean(axis=(0, 1)).astype(np.float32))
        self.std = torch.from_numpy(
            train_fields.std(axis=(0, 1)).astype(np.float32)
        ).clamp_min(1e-08)
        stats_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'mean': self.mean, 'std': self.std}, stats_file)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        fields = torch.from_numpy(self._fields[idx])
        fields_norm = (fields - self.mean) / self.std

        mask_2d = self._fields[idx][:, 1].reshape(self.grid_Ny, self.grid_Nx)
        material = mask_2d > 0.5
        hole = ~material
        hole_dilated = binary_dilation(hole, iterations=3)
        near_boundary = material & hole_dilated
        sensor_mask = torch.from_numpy(near_boundary.ravel().astype(np.float32)).bool()

        return {
            'coords': self.coords.clone(),
            'coords_raw': self.coords_raw.clone(),
            'fields': fields_norm,
            'valid_sensor_mask': sensor_mask,
            'time_index': torch.tensor(idx, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }

class AirfoilCGridDataset(Dataset):
    """
    Loads the NACA airfoil C-grid dataset.

    Each sample is a body-fitted C-mesh (221x51 = 11271 points) with
    per-sample deformed coordinates (Y varies per airfoil shape) and
    5 compressible-flow fields:
        field 0 — rho     (density)
        field 1 — rho*u   (x-momentum)
        field 2 — rho*v   (y-momentum)
        field 3 — rho*E   (energy)
        field 4 — p       (pressure)
    """
    FIELD_NAMES = ('rho', 'rho_u', 'rho_v', 'rho_E', 'p')

    def __init__(self, data_dir, split, train_ratio, seed,
                 field_names=None, stats_path=None, select_fields=None,
                 sensor_surface_offset_min=0, sensor_surface_offset_max=0):
        super().__init__()
        self.data_dir = data_dir
        naca_dir = os.path.join(data_dir, 'naca')
        CX = np.load(os.path.join(naca_dir, 'NACA_Cylinder_X.npy'))
        CY = np.load(os.path.join(naca_dir, 'NACA_Cylinder_Y.npy'))
        Q = np.load(os.path.join(naca_dir, 'NACA_Cylinder_Q.npy'))

        if select_fields is not None:
            Q = Q[:, list(select_fields)]
            all_names = self.FIELD_NAMES
            # Propagate names consistent with Q channel selection; verified via
            # AirfoilCGridDataset(..., select_fields=(0,4)) yielding ('rho','p').
            self.field_names = field_names if field_names else tuple(
                all_names[i] for i in select_fields
            )
        else:
            self.field_names = field_names if field_names else self.FIELD_NAMES

        N_samples = CX.shape[0]
        Ny = CX.shape[1]
        Nx = CX.shape[2]
        self.num_points = Ny * Nx
        self.num_fields = Q.shape[1]
        self.grid_shape = (Ny, Nx)

        self._CX = CX.reshape(N_samples, -1).astype(np.float32)
        self._CY = CY.reshape(N_samples, -1).astype(np.float32)

        # Identify which radial ring (i=0 or i=Nx-1) corresponds to the
        # airfoil surface by comparing spatial extents.
        idx_ring0 = np.arange(Ny) * Nx
        idx_ringN = np.arange(Ny) * Nx + (Nx - 1)
        extent_0 = np.ptp(self._CX[0, idx_ring0]) + np.ptp(self._CY[0, idx_ring0])
        extent_N = np.ptp(self._CX[0, idx_ringN]) + np.ptp(self._CY[0, idx_ringN])
        surface_is_i0 = extent_0 < extent_N
        surface_ring = idx_ring0 if surface_is_i0 else idx_ringN
        self._surface_is_i0 = surface_is_i0

        # Identify surface j-indices that lie on the airfoil body
        # (non-trivial gap between upper and lower surfaces).
        sx = self._CX[0, surface_ring]
        sy = self._CY[0, surface_ring]
        j_LE = int(np.argmin(sx))
        n_half = min(j_LE, len(sx) - 1 - j_LE)
        upper_idx = j_LE - np.arange(n_half + 1)
        lower_idx = j_LE + np.arange(n_half + 1)
        gaps = np.abs(sy[upper_idx] - sy[lower_idx])
        max_gap = gaps.max()
        in_body = gaps > 0.01 * max_gap
        # Union of upper- and lower-surface j-indices with non-trivial gap.
        # Empirically produces 118 body points forming a closed NACA outline
        # for the provided NACA_Cylinder dataset (221x51 C-grid).
        body_js = np.concatenate([upper_idx[in_body], lower_idx[in_body]])
        body_js = np.unique(body_js)
        if surface_is_i0:
            self.airfoil_body_indices = body_js * Nx
        else:
            self.airfoil_body_indices = body_js * Nx + (Nx - 1)

        self._fields = Q.reshape(N_samples, self.num_fields, -1).transpose(0, 2, 1).astype(np.float32)

        # Radial offsets from surface define "near-surface" sensor band.
        off_min = max(0, int(sensor_surface_offset_min))
        off_max = max(off_min, int(sensor_surface_offset_max))
        off_max = min(off_max, Nx - 1)
        # Radial axis is i (the Nx=51 dimension); surface ring sits at i=0 or
        # i=Nx-1 depending on surface_is_i0 (established above by ptp test).
        if surface_is_i0:
            radial_offsets = np.arange(off_min, off_max + 1)
        else:
            radial_offsets = (Nx - 1) - np.arange(off_min, off_max + 1)

        j_body = self.airfoil_body_indices // Nx if surface_is_i0 else \
            (self.airfoil_body_indices - (Nx - 1)) // Nx
        surface_mask = np.zeros(Ny * Nx, dtype=bool)
        for i_off in radial_offsets:
            surface_mask[j_body * Nx + i_off] = True
        self._valid_sensor_mask = surface_mask
        self._sensor_surface_offsets = (off_min, off_max)

        all_indices = np.arange(N_samples, dtype=np.int64)
        rng = np.random.default_rng(seed)
        rng.shuffle(all_indices)
        n_train = int(N_samples * train_ratio)
        if split == 'train':
            self.indices = np.sort(all_indices[:n_train])
        elif split in ('val', 'test'):
            self.indices = np.sort(all_indices[n_train:])
        else:
            raise ValueError(f'Unknown split: {split}')

        stats_file = Path(stats_path) if stats_path else Path(data_dir) / 'airfoil_cgrid_stats.pt'

        self._coord_min = np.array(
            [self._CX.min(), self._CY.min()], dtype=np.float32
        )
        self._coord_range = np.array(
            [self._CX.max() - self._CX.min(), self._CY.max() - self._CY.min()],
            dtype=np.float32,
        )
        self._coord_range = np.maximum(self._coord_range, 1e-08)

        ref_xy = np.stack([self._CX[0], self._CY[0]], axis=-1)
        self.coords_raw = torch.from_numpy(ref_xy)
        ref_norm = (ref_xy - self._coord_min) / self._coord_range
        self.coords = torch.from_numpy(np.concatenate(
            [ref_norm, np.zeros((self.num_points, 1), dtype=np.float32)], axis=-1
        ))

        # Compute or load stats.
        if stats_file.exists():
            obj = torch.load(stats_file, map_location='cpu')
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
        else:
            # Per-field mean/std over the training split, matching the
            # ElasticityDataset stats-computation convention (mean over
            # axis=(0,1) of a (N_samples, num_points, num_fields) array).
            train_fields = self._fields[all_indices[:n_train]]
            self.mean = torch.from_numpy(
                train_fields.mean(axis=(0, 1)).astype(np.float32)
            )
            self.std = torch.from_numpy(
                train_fields.std(axis=(0, 1)).astype(np.float32)
            ).clamp_min(1e-08)
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': self.mean, 'std': self.std}, stats_file)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = int(self.indices[i])
        xy = np.stack([self._CX[idx], self._CY[idx]], axis=-1)
        xy_norm = (xy - self._coord_min) / self._coord_range
        coords_3d = np.concatenate(
            [xy_norm, np.zeros((self.num_points, 1), dtype=np.float32)], axis=-1
        )
        fields = torch.from_numpy(self._fields[idx])
        fields_norm = (fields - self.mean) / self.std
        return {
            'coords': torch.from_numpy(coords_3d),
            'coords_raw': torch.from_numpy(xy),
            'fields': fields_norm,
            'valid_sensor_mask': torch.from_numpy(self._valid_sensor_mask),
            'time_index': torch.tensor(idx, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }

# -------------------------------------------

def _read_ahmed_ply(ply_path: str):
    """Parse one Ahmed body PLY (binary little-endian, float verts with normals,
    list-uchar-int triangle faces). Returns (vertices [N_v, 3], normals [N_v, 3],
    faces [N_f, 3]).
    """
    import struct
    with open(ply_path, 'rb') as f:
        header_lines = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF in PLY header: {ply_path}")
            header_lines.append(line.decode('latin-1').rstrip())
            if header_lines[-1] == 'end_header':
                break

        n_vert = n_face = None
        vprops = []
        in_v = in_f = False
        for ln in header_lines:
            if ln.startswith('element vertex'):
                n_vert = int(ln.split()[-1]); in_v, in_f = True, False
            elif ln.startswith('element face'):
                n_face = int(ln.split()[-1]); in_v, in_f = False, True
            elif in_v and ln.startswith('property'):
                vprops.append(ln.split()[-1])

        dtype = np.dtype([(p, '<f4') for p in vprops])
        verts_raw = np.frombuffer(f.read(n_vert * dtype.itemsize),
                                  dtype=dtype, count=n_vert)
        xyz = np.stack([verts_raw['x'], verts_raw['y'], verts_raw['z']],
                       axis=-1).astype(np.float32)
        if 'nx' in vprops:
            nrm = np.stack([verts_raw['nx'], verts_raw['ny'], verts_raw['nz']],
                           axis=-1).astype(np.float32)
        else:
            nrm = None

        # Face list: uchar count (expected 3) followed by int32 indices.
        face_blob = f.read()
    tri = np.empty((n_face, 3), dtype=np.int32)
    off = 0
    for i in range(n_face):
        cnt = face_blob[off]; off += 1
        tri[i] = struct.unpack_from('<3i', face_blob, off)[:3]
        off += 4 * cnt
    return xyz, nrm, tri


class CarCFDDataset(Dataset):
    """Surface-pressure CFD dataset for 3D point-cloud FFM.

    Supports two on-disk variants, auto-detected from the directory layout:

    * ``ahmed`` — half-Ahmed-body meshes with *per-face* pressure. Samples
      live in ``{data_dir}/train`` and ``{data_dir}/test``. Face centroids
      are used as the surface point cloud.

    * ``shapenet`` — ShapeNet-Car meshes with *per-vertex* pressure. Samples
      live in ``{data_dir}/data`` (no native split; one is carved from
      ``train_ratio``). Vertex positions are used directly as the surface
      point cloud.

    In both variants the dataset returns a fixed-size subsampled point cloud
    per sample (so that snapshots collate into a tensor of shape
    [B, n_points, 3]) along with the scalar pressure field.

    Args:
        data_dir: root of the extracted dataset.
        split: 'train', 'val', or 'test'. For the ``ahmed`` variant the
            native 'test' directory is used as-is and a validation slice is
            carved out of 'train' via ``train_ratio``. For the ``shapenet``
            variant there is no native test split, so 'test' returns the
            held-out validation slice.
        n_points: fixed number of surface points returned per sample.
        seed: deterministic per-sample subsample seed.
        stats_path: where pressure mean/std is cached.
    """
    FIELD_NAMES = ('p',)

    def __init__(self, data_dir, split, train_ratio=0.9, seed=42,
                 n_points=8192, field_names=None, stats_path=None,
                 cache_dir=None):
        super().__init__()
        self.data_dir = str(data_dir)
        self.split = split
        self.n_points = int(n_points)
        self.seed = int(seed)
        self.field_names = field_names if field_names else self.FIELD_NAMES
        self.num_fields = 1

        # Auto-detect the on-disk variant. ShapeNet-Car ships a single
        # ``data/`` directory with per-vertex pressure; Ahmed ships
        # ``train/``+``test/`` with per-face pressure.
        has_train = os.path.isdir(os.path.join(self.data_dir, 'train'))
        has_data = os.path.isdir(os.path.join(self.data_dir, 'data'))
        if has_train:
            self._data_format = 'ahmed'
        elif has_data:
            self._data_format = 'shapenet'
        else:
            raise FileNotFoundError(
                f"CarCFDDataset: expected either '{self.data_dir}/train' "
                f"(ahmed) or '{self.data_dir}/data' (shapenet) to exist."
            )

        # Coordinate normalization uses the dataset-wide bounds so every car
        # lives in the same reference box (sensors at the same physical place
        # have the same normalized coord regardless of which car is loaded).
        bounds_path = os.path.join(self.data_dir, 'global_bounds.txt')
        if os.path.exists(bounds_path):
            with open(bounds_path) as fh:
                lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
            gmin = np.asarray(lines[0].split(), dtype=np.float32)
            gmax = np.asarray(lines[1].split(), dtype=np.float32)
        else:
            # Fallback hardcoded Ahmed bounds from the published dataset.
            gmin = np.asarray([-1.344, 0.0, 0.0], dtype=np.float32)
            gmax = np.asarray([0.0005, 0.2545, 0.4305], dtype=np.float32)
        self._coord_min = gmin
        self._coord_range = np.maximum(gmax - gmin, 1e-8)

        if self._data_format == 'ahmed':
            # Pool 'train' + 'test' directories and carve a val slice out of
            # 'train' via train_ratio; the native 'test' dir is held out.
            train_ids = self._scan_ids(os.path.join(self.data_dir, 'train'))
            test_ids = self._scan_ids(os.path.join(self.data_dir, 'test'))
            rng = np.random.default_rng(seed)
            order = np.arange(len(train_ids))
            rng.shuffle(order)
            n_train = max(1, int(len(train_ids) * train_ratio))
            tr_idx = np.sort(order[:n_train])
            va_idx = np.sort(order[n_train:])
            if split == 'train':
                self._sample_keys = [('train', train_ids[i]) for i in tr_idx]
            elif split == 'val':
                keys = [('train', train_ids[i]) for i in va_idx]
                if len(keys) == 0:
                    keys = [('test', sid) for sid in test_ids]
                self._sample_keys = keys
            elif split == 'test':
                self._sample_keys = [('test', sid) for sid in test_ids]
            else:
                raise ValueError(f'Unknown split: {split}')
        else:  # 'shapenet'
            # Single 'data/' pool; carve train / val via train_ratio. No
            # native test split exists, so 'test' aliases the val slice.
            all_ids = self._scan_ids(os.path.join(self.data_dir, 'data'))
            rng = np.random.default_rng(seed)
            order = np.arange(len(all_ids))
            rng.shuffle(order)
            n_train = max(1, int(len(all_ids) * train_ratio))
            tr_idx = np.sort(order[:n_train])
            va_idx = np.sort(order[n_train:])
            if split == 'train':
                self._sample_keys = [('data', all_ids[i]) for i in tr_idx]
            elif split in ('val', 'test'):
                self._sample_keys = [('data', all_ids[i]) for i in va_idx]
            else:
                raise ValueError(f'Unknown split: {split}')

        # Cache parsed meshes to disk so re-runs skip PLY parsing.
        self.cache_dir = cache_dir or os.path.join(self.data_dir, '_cache_pt')
        os.makedirs(self.cache_dir, exist_ok=True)

        # Expose a reference point cloud for visualization helpers / model
        # construction; coords_raw holds physical mm-scale coordinates.
        ref = self._load_sample(0)
        self.coords = ref['coords'].clone()
        self.coords_raw = ref['coords_raw'].clone()
        self.num_points = self.n_points

        # Pressure stats on the training split.
        stats_file = Path(stats_path) if stats_path else \
            Path(self.data_dir) / f'car_cfd_stats_N{self.n_points}.pt'
        if stats_file.exists():
            obj = _torch_load_compat(stats_file, map_location="cpu")
            self.mean = obj['mean'].float()
            self.std = obj['std'].float()
        else:
            # Stream through the training split once to accumulate stats.
            if split != 'train':
                raise FileNotFoundError(
                    f"Stats file {stats_file} not found. Instantiate the "
                    f"'train' split first (or pass an existing stats_path)."
                )
            total = 0.0; total_sq = 0.0; count = 0
            for k in range(len(self._sample_keys)):
                fields = self._load_sample(k)['fields']  # [N, 1] physical
                total += float(fields.sum())
                total_sq += float((fields.double() ** 2).sum())
                count += fields.numel()
            mean = total / max(count, 1)
            var = max(total_sq / max(count, 1) - mean ** 2, 1e-12)
            self.mean = torch.tensor([mean], dtype=torch.float32)
            self.std = torch.tensor([var ** 0.5], dtype=torch.float32).clamp_min(1e-8)
            stats_file.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'mean': self.mean, 'std': self.std}, stats_file)

    @staticmethod
    def _scan_ids(dirpath: str):
        if not os.path.isdir(dirpath):
            return []
        ids = []
        for fn in sorted(os.listdir(dirpath)):
            if fn.startswith('press_') and fn.endswith('.npy'):
                sid = fn[len('press_'):-len('.npy')]
                mesh_path = os.path.join(dirpath, f'mesh_{sid}.ply')
                if os.path.exists(mesh_path):
                    ids.append(sid)
        return ids

    def _sample_cache_path(self, split_dir: str, sid: str) -> Path:
        return Path(self.cache_dir) / f'{split_dir}_sample_{sid}_N{self.n_points}_seed{self.seed}.pt'

    def _load_sample(self, i: int):
        split_dir, sid = self._sample_keys[i]
        cache_path = self._sample_cache_path(split_dir, sid)
        if cache_path.exists():
            obj = _torch_load_compat(cache_path, map_location="cpu")
            return obj
        base = os.path.join(self.data_dir, split_dir)
        xyz, _, tri = _read_ahmed_ply(os.path.join(base, f'mesh_{sid}.ply'))
        press = np.load(os.path.join(base, f'press_{sid}.npy')).astype(np.float32)
        if self._data_format == 'ahmed':
            # Pressure is per-face; use face centroids as the point cloud.
            pts = xyz[tri].mean(axis=1).astype(np.float32)
            n_total = pts.shape[0]
            if press.shape[0] != n_total:
                raise ValueError(f"press/face mismatch for {sid}: "
                                 f"press={press.shape[0]} faces={n_total}")
        else:
            # ShapeNet-Car: pressure is per-vertex; use vertex positions.
            pts = xyz.astype(np.float32)
            n_total = pts.shape[0]
            if press.shape[0] != n_total:
                raise ValueError(f"press/vertex mismatch for {sid}: "
                                 f"press={press.shape[0]} verts={n_total}")
        # Deterministic subsample using a per-sample seed derived from sid so
        # the training point cloud stays consistent across DataLoader workers,
        # epochs, and re-runs. Python's builtin hash() is salted per-process,
        # so we use a stable hashlib digest instead.
        import hashlib
        digest = hashlib.blake2b(f'car_cfd::{sid}::{self.seed}'.encode(),
                                 digest_size=4).digest()
        sub_rng = np.random.default_rng(int.from_bytes(digest, 'little'))
        if n_total >= self.n_points:
            idx = sub_rng.choice(n_total, size=self.n_points, replace=False)
        else:
            idx = sub_rng.choice(n_total, size=self.n_points, replace=True)
        idx.sort()
        coords_raw = torch.from_numpy(pts[idx])                 # [N, 3] physical
        fields = torch.from_numpy(press[idx][:, None])          # [N, 1] physical
        coords_norm = (coords_raw.numpy() - self._coord_min) / self._coord_range
        coords = torch.from_numpy(coords_norm.astype(np.float32))
        obj = {
            'coords': coords,
            'coords_raw': coords_raw,
            'fields': fields,
            'sample_id': sid,
            'split_dir': split_dir,
        }
        try:
            torch.save({k: v for k, v in obj.items() if k in ('coords', 'coords_raw', 'fields')},
                       cache_path)
        except OSError:
            pass
        return obj

    def __len__(self):
        return len(self._sample_keys)

    def __getitem__(self, i):
        s = self._load_sample(i)
        fields_norm = (s['fields'] - self.mean) / self.std
        return {
            'coords': s['coords'].clone(),
            'coords_raw': s['coords_raw'].clone(),
            'fields': fields_norm,
            'time_index': torch.tensor(i, dtype=torch.long),
            'physical_time': torch.tensor(0),
        }

    def load_full_mesh_sample(self, i: int):
        """Load every face centroid + pressure for sample ``i`` without the
        training-time subsample. Used only for visualization so the field
        renders as a dense continuous color map instead of a sparse scatter.

        Returns a dict with:
            coords       [N_face, 3]  normalized face-centroid coords (model input)
            coords_raw   [N_face, 3]  physical (m) face-centroid coords
            fields       [N_face, 1]  normalized pressure
            fields_phys  [N_face, 1]  physical pressure (Pa)
            vertices     [N_vert, 3]  physical vertex coords for mesh plotting
            triangles    [N_face, 3]  int triangle vertex indices
        """
        split_dir, sid = self._sample_keys[i]
        base = os.path.join(self.data_dir, split_dir)
        xyz, _, tri = _read_ahmed_ply(os.path.join(base, f'mesh_{sid}.ply'))
        press = np.load(os.path.join(base, f'press_{sid}.npy')).astype(np.float32)
        if self._data_format == 'ahmed':
            pts = xyz[tri].mean(axis=1).astype(np.float32)
        else:
            pts = xyz.astype(np.float32)
        coords_raw = torch.from_numpy(pts)
        fields_phys = torch.from_numpy(press[:, None])
        coords_norm = (pts - self._coord_min) / self._coord_range
        coords = torch.from_numpy(coords_norm.astype(np.float32))
        fields_norm = (fields_phys - self.mean) / self.std
        return {
            'coords': coords,
            'coords_raw': coords_raw,
            'fields': fields_norm,
            'fields_phys': fields_phys,
            'vertices': torch.from_numpy(xyz),
            'triangles': torch.from_numpy(tri.astype(np.int64)),
        }


# ═════════ §3. Collation & grid validation ═════════

def collate_snapshots(batch):
    """Shared collate that handles optional keys like valid_sensor_mask, coords_raw."""
    out = {
        'coords':        torch.stack([b['coords']        for b in batch], dim=0),
        'fields':        torch.stack([b['fields']        for b in batch], dim=0),
        'time_index':    torch.stack([b['time_index']    for b in batch], dim=0),
        'physical_time': torch.stack([b['physical_time'] for b in batch], dim=0),
    }
    if 'coords_raw' in batch[0]:
        out['coords_raw'] = torch.stack([b['coords_raw'] for b in batch], dim=0)
    if 'valid_sensor_mask' in batch[0]:
        out['valid_sensor_mask'] = torch.stack(
            [b['valid_sensor_mask'] for b in batch], dim=0
        )
    return out

def validate_regular_grid_compatibility(
    dataset: Dataset,
    Num_x: Optional[int],
    Num_y: Optional[int],
    decimals: int = 6,
    atol: float = 1e-5,
) -> Dict[str, object]:
    """
    Validate that a point-cloud dataset can be interpreted as a Num_x by Num_y regular grid.

    Required behavior for grid baselines:
      - Num_x and Num_y must be provided in YAML / args
      - Num_x * Num_y must match the number of points
      - the coordinate set must contain exactly Num_x unique x-values
        and Num_y unique y-values (up to rounding)
      - every (x, y) tensor-product grid cell must exist exactly once

    Like the FFM-FNO path, this function does not require row-major HDF5 point
    order or uniform physical spacing. It reports those facts and returns a
    coordinate-derived row-major permutation so direct-reshape grid baselines
    can reorder fields/observations safely before reshaping.

    Raises ValueError if the dataset is not compatible with the requested grid.
    Returns a small diagnostics dict for launch-time logging.
    """
    def _grid_baseline_error(reason: str) -> ValueError:
        return ValueError(
            "Regular-grid validation failed for a grid baseline "
            f"(SiT/FNO/S3GM-style): {reason} The dataset may be a valid point "
            "cloud, but it cannot be safely interpreted as a complete structured "
            "grid for this baseline. Use a point-cloud model or add an explicit "
            "dataset-specific gridding/interpolation step."
        )

    if Num_x is None or Num_y is None:
        raise ValueError(
            "Direct-reshape grid baselines require Num_x and Num_y to be explicitly "
            "provided in YAML / args."
        )

    Num_x = int(Num_x)
    Num_y = int(Num_y)

    if Num_x <= 0 or Num_y <= 0:
        raise ValueError(f"Num_x and Num_y must be positive, got Num_x={Num_x}, Num_y={Num_y}.")

    expected_points = Num_x * Num_y
    if int(dataset.num_points) != expected_points:
        raise _grid_baseline_error(
            f"Grid mismatch: dataset has {dataset.num_points} points, but "
            f"Num_x * Num_y = {Num_x} * {Num_y} = {expected_points}."
        )

    coords = dataset.coords.cpu()
    x = torch.round(coords[:, 0] * (10 ** decimals)) / (10 ** decimals)
    y = torch.round(coords[:, 1] * (10 ** decimals)) / (10 ** decimals)

    unique_x = int(torch.unique(x).numel())
    unique_y = int(torch.unique(y).numel())

    if unique_x != Num_x or unique_y != Num_y:
        raise _grid_baseline_error(
            f"Dataset unique counts are ({unique_x}, {unique_y}) in (x, y), "
            f"but requested (Num_x, Num_y)=({Num_x}, {Num_y})."
        )

    unique_x_vals = torch.unique(x, sorted=True)
    unique_y_vals = torch.unique(y, sorted=True)

    def _spacing_stats(values: torch.Tensor) -> tuple[bool, torch.Tensor]:
        if values.numel() <= 1:
            diffs = torch.ones(1, dtype=values.dtype, device=values.device)
            return True, diffs
        diffs = values[1:] - values[:-1]
        if diffs.numel() <= 1:
            return True, diffs
        regular = bool(torch.allclose(diffs, diffs.median().expand_as(diffs), atol=atol, rtol=1e-4))
        return regular, diffs

    x_spacing_regular, x_diffs = _spacing_stats(unique_x_vals)
    y_spacing_regular, y_diffs = _spacing_stats(unique_y_vals)
    coord_dim = int(coords.shape[-1])
    coords_grid = coords.reshape(Num_y, Num_x, coord_dim)
    x_grid = torch.round(coords_grid[..., 0] * (10 ** decimals)) / (10 ** decimals)
    y_grid = torch.round(coords_grid[..., 1] * (10 ** decimals)) / (10 ** decimals)

    x_first_row = x_grid[0]
    y_first_col = y_grid[:, 0]
    x_expected_asc = unique_x_vals
    x_expected_desc = torch.flip(unique_x_vals, dims=[0])
    y_expected_asc = unique_y_vals
    y_expected_desc = torch.flip(unique_y_vals, dims=[0])

    x_constant_down_columns = bool(torch.allclose(
        x_grid,
        x_first_row.view(1, Num_x).expand(Num_y, Num_x),
        atol=atol,
        rtol=1e-4,
    ))
    y_constant_across_rows = bool(torch.allclose(
        y_grid,
        y_first_col.view(Num_y, 1).expand(Num_y, Num_x),
        atol=atol,
        rtol=1e-4,
    ))
    x_varies_along_axis1 = bool(
        torch.allclose(x_first_row, x_expected_asc, atol=atol, rtol=1e-4)
        or torch.allclose(x_first_row, x_expected_desc, atol=atol, rtol=1e-4)
    )
    y_varies_along_axis0 = bool(
        torch.allclose(y_first_col, y_expected_asc, atol=atol, rtol=1e-4)
        or torch.allclose(y_first_col, y_expected_desc, atol=atol, rtol=1e-4)
    )

    row_major = (
        x_constant_down_columns
        and y_constant_across_rows
        and x_varies_along_axis1
        and y_varies_along_axis0
    )

    _, x_rank = torch.unique(x, sorted=True, return_inverse=True)
    _, y_rank = torch.unique(y, sorted=True, return_inverse=True)
    point_to_grid = (y_rank.long() * Num_x + x_rank.long()).contiguous()
    unique_cells = int(torch.unique(point_to_grid).numel())
    if unique_cells != expected_points:
        raise _grid_baseline_error(
            "The coordinate set is not a complete tensor-product grid. "
            f"Detected unique counts are (x={unique_x}, y={unique_y}), but "
            f"unique (x, y) cells are {unique_cells} out of {expected_points}."
        )

    diagnostics = []
    if not x_spacing_regular:
        diagnostics.append("unique x coordinates are not approximately regularly spaced")
    if not y_spacing_regular:
        diagnostics.append("unique y coordinates are not approximately regularly spaced")
    if not row_major:
        diagnostics.append(
            "flattened point order is not row-major; grid baselines will apply a coordinate-derived permutation"
        )

    grid_order = torch.argsort(point_to_grid).contiguous()
    return {
        "Num_x": Num_x,
        "Num_y": Num_y,
        "num_points": expected_points,
        "unique_x": unique_x,
        "unique_y": unique_y,
        "complete_tensor_product": True,
        "row_major": row_major,
        "requires_permutation": not row_major,
        "spacing_regular": x_spacing_regular and y_spacing_regular,
        "x_spacing_min": float(x_diffs.min().item()),
        "x_spacing_median": float(x_diffs.median().item()),
        "x_spacing_max": float(x_diffs.max().item()),
        "y_spacing_min": float(y_diffs.min().item()),
        "y_spacing_median": float(y_diffs.median().item()),
        "y_spacing_max": float(y_diffs.max().item()),
        "diagnostics": diagnostics,
        "grid_order": grid_order,
        "point_to_grid": point_to_grid,
        "first_row_original_indices": [int(v) for v in grid_order[: min(8, grid_order.numel())].tolist()],
        "first_original_to_grid_indices": [int(v) for v in point_to_grid[: min(8, point_to_grid.numel())].tolist()],
    }

# ═════════ §4. Grid ↔ point-cloud interchange ═════════

# ----- Irregular <-> regular grid transfer helpers ---------------------------

def pointcloud_to_grid(x: torch.Tensor, Num_y: int, Num_x: int) -> torch.Tensor:
    """[B, N, C] -> [B, C, Num_y, Num_x]. Assumes N == Num_y * Num_x, row-major."""
    B = x.shape[0]
    return x.reshape(B, Num_y, Num_x, -1).permute(0, 3, 1, 2).contiguous()


def splat_to_grid(coords_2d: torch.Tensor, values: torch.Tensor, Ny: int, Nx: int):
    """
    Bilinearly splat per-point values onto a regular (Ny, Nx) grid.

    Args:
        coords_2d: [B, K, 2] normalized coords in [0, 1]
        values:    [B, K, D]
        Ny, Nx:    target grid size
    Returns:
        grid:   [B, D, Ny, Nx] values averaged by accumulated weight
        weight: [B, 1, Ny, Nx] accumulated weights
    """
    B, K, D = values.shape
    device = values.device
    dtype = values.dtype

    px = coords_2d[..., 0] * (Nx - 1)
    py = coords_2d[..., 1] * (Ny - 1)

    ix0 = px.long().clamp(0, Nx - 2)
    iy0 = py.long().clamp(0, Ny - 2)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    wx = (px - ix0.float()).unsqueeze(-1)
    wy = (py - iy0.float()).unsqueeze(-1)

    grid = torch.zeros(B, D, Ny * Nx, device=device, dtype=dtype)
    weight = torch.zeros(B, 1, Ny * Nx, device=device, dtype=dtype)

    for dy, dwy in [(iy0, 1 - wy), (iy1, wy)]:
        for dx, dwx in [(ix0, 1 - wx), (ix1, wx)]:
            w = dwy * dwx
            flat = (dy * Nx + dx).unsqueeze(1)        # [B, 1, K]

            wf = (values * w).permute(0, 2, 1)        # [B, D, K]
            grid.scatter_add_(2, flat.expand_as(wf), wf)

            ww = w.permute(0, 2, 1)                    # [B, 1, K]
            weight.scatter_add_(2, flat, ww)

    weight_c = weight.clamp(min=1e-6)
    grid = grid / weight_c
    return grid.reshape(B, D, Ny, Nx), weight.reshape(B, 1, Ny, Nx)

def gather_from_grid(grid: torch.Tensor, coords_2d: torch.Tensor) -> torch.Tensor:
    """
    Bilinear interpolation from regular grid to arbitrary query points.

    Args:
        grid:      [B, C, Ny, Nx]
        coords_2d: [B, M, 2] normalized coords in [0, 1]
    Returns:
        values: [B, M, C]
    """
    sample_pts = coords_2d * 2.0 - 1.0
    sample_pts = sample_pts.unsqueeze(2)                 # [B, M, 1, 2]
    gathered = F.grid_sample(
        grid, sample_pts, mode="bilinear",
        padding_mode="border", align_corners=True,
    )                                                     # [B, C, M, 1]
    return gathered.squeeze(-1).permute(0, 2, 1)

def splat_obs_to_grid(obs_coords_2d: torch.Tensor,
                      obs_values: torch.Tensor,
                      obs_mask: torch.Tensor,
                      obs_field_ids: torch.Tensor,
                      n_fields: int,
                      Ny: int,
                      Nx: int):
    """
    Scatter sparse sensor observations onto per-field (Ny, Nx) grids.

    Args:
        obs_coords_2d: [B, M, 2] normalized coords in [0, 1]
        obs_values:    [B, M, 1]
        obs_mask:      [B, M]
        obs_field_ids: [B, M] integer field id for each sensor (-1 if invalid)
        n_fields:      number of target channels
        Ny, Nx:        grid size
    Returns:
        val_grid: [B, n_fields, Ny, Nx]
        msk_grid: [B, n_fields, Ny, Nx]
    """
    B, M, _ = obs_coords_2d.shape
    device = obs_coords_2d.device
    dtype = obs_coords_2d.dtype

    val_grid = torch.zeros(B, n_fields, Ny * Nx, device=device, dtype=dtype)
    count_grid = torch.zeros(B, n_fields, Ny * Nx, device=device, dtype=dtype)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        xy = obs_coords_2d[b, valid]
        ix = (xy[:, 0] * (Nx - 1)).long().clamp(0, Nx - 1)
        iy = (xy[:, 1] * (Ny - 1)).long().clamp(0, Ny - 1)
        flat = iy * Nx + ix
        # Accumulate; multiple sensors colliding in the same cell are averaged
        # after the loop. Assignment would silently drop all but the last.
        val_grid[b].index_put_((fld, flat), val, accumulate=True)
        count_grid[b].index_put_(
            (fld, flat), torch.ones_like(val), accumulate=True,
        )

    val_grid = val_grid / count_grid.clamp_min(1.0)
    msk_grid = (count_grid > 0).to(dtype)

    return val_grid.reshape(B, n_fields, Ny, Nx), msk_grid.reshape(B, n_fields, Ny, Nx)

# ----- Plotting helpers for unstructured / body-fitted geometries ------------

def _build_structured_triangulation(coords_xy: np.ndarray, grid_shape):
    """
    Build a matplotlib Triangulation that follows the (Ny, Nx) logical
    connectivity of a structured body-fitted mesh. Each quad cell is
    split into two triangles.

    Args:
        coords_xy:  [N, 2] flattened (row-major, j varies fastest along Nx)
        grid_shape: (Ny, Nx)
    Returns:
        mtri.Triangulation
    """
    # Row-major (C-order) flattening: index = j*Nx + i. Verified against
    # ElasticityDataset (meshgrid indexing='ij' + ravel) and
    # AirfoilCGridDataset (CX shape (N,Ny,Nx) reshaped via reshape(N,-1)).
    # Empirical check: 2*(Ny-1)*(Nx-1) non-degenerate unit-area triangles.
    Ny, Nx = int(grid_shape[0]), int(grid_shape[1])
    x = coords_xy[:, 0]
    y = coords_xy[:, 1]

    idx = np.arange(Ny * Nx).reshape(Ny, Nx)
    v00 = idx[:-1, :-1].ravel()
    v10 = idx[1:, :-1].ravel()
    v01 = idx[:-1, 1:].ravel()
    v11 = idx[1:, 1:].ravel()

    tris = np.concatenate(
        [np.stack([v00, v10, v11], axis=1),
         np.stack([v00, v11, v01], axis=1)],
        axis=0,
    )
    return mtri.Triangulation(x, y, triangles=tris)

# ═════════ §5. Sparse-condition construction ═════════

def build_sparse_condition(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    cond_fields: Union[int, Sequence[int]],
    n_obs_min: Union[int, Sequence[int]],
    n_obs_max: Union[int, Sequence[int]],
    valid_mask: Optional[torch.Tensor] = None,
    coords_2d: Optional[torch.Tensor] = None,
    Ny: Optional[int] = None,
    Nx: Optional[int] = None,
):
    """
    Generalized sparse conditioning.

    Args:
        coords_full: [B, N, D]
        fields_full: [B, N, C]
        cond_fields: int or list[int], e.g. 2 or [0, 2]
        n_obs_min: int or list[int], per conditioned field
        n_obs_max: int or list[int], per conditioned field
        valid_mask: optional [B, N] or [N] bool/float mask restricting sensor
            placement to a subset of points (e.g. near the stress
            concentration around a hole for elasticity, or near the airfoil
            surface). Points where the mask is 0/False are never sampled.
            If a batch item has fewer valid points than n_obs_max for a
            field, sampling is capped at the number of valid points (and
            the unused slots stay zero-masked).
        coords_2d, Ny, Nx: optional grid-aware sampling. If coords_2d [B, N, 2]
            in [0, 1] and (Ny, Nx) are provided, the sampler guarantees that
            each sampled sensor lands in a distinct latent grid cell for its
            field. Use for irregular meshes where the Geo-FNO-style deformer
            can collapse multiple mesh points into the same cell. If the pool
            (after valid_mask) contains fewer than n_obs distinct cells for a
            field, m is capped at the number of reachable cells.

    Returns:
        obs_coords:    [B, M, D]
        obs_values:    [B, M, 1]
        obs_mask:      [B, M]
        obs_indices:   [B, M]
        obs_field_ids: [B, M]   # which field each sensor belongs to
    """
    cond_fields = _to_int_list(cond_fields)
    if len(cond_fields) == 0:
        raise ValueError("cond_fields must contain at least one field index.")

    n_obs_min = _broadcast_per_field(n_obs_min, cond_fields, "n_obs_min")
    n_obs_max = _broadcast_per_field(n_obs_max, cond_fields, "n_obs_max")

    for a, b in zip(n_obs_min, n_obs_max):
        if b < a:
            raise ValueError(f"Each n_obs_max must be >= n_obs_min, got {a} and {b}.")

    bsz, n_pts, coord_dim = coords_full.shape
    device = coords_full.device

    grid_aware = coords_2d is not None
    if grid_aware:
        if Ny is None or Nx is None:
            raise ValueError("Grid-aware sampling requires both Ny and Nx.")
        if coords_2d.shape[:2] != (bsz, n_pts):
            raise ValueError(
                f"coords_2d shape {tuple(coords_2d.shape)} incompatible with "
                f"coords_full shape {tuple(coords_full.shape)}."
            )
        ix_all = (coords_2d[..., 0] * (Nx - 1)).long().clamp(0, Nx - 1)
        iy_all = (coords_2d[..., 1] * (Ny - 1)).long().clamp(0, Ny - 1)
        cells_all = iy_all * Nx + ix_all   # [B, N]
        n_cells_total = Ny * Nx

    max_obs = sum(n_obs_max)

    obs_coords = torch.zeros(
        bsz, max_obs, coord_dim, device=device, dtype=coords_full.dtype
    )
    obs_values = torch.zeros(
        bsz, max_obs, 1, device=device, dtype=fields_full.dtype
    )
    obs_mask = torch.zeros(
        bsz, max_obs, device=device, dtype=coords_full.dtype
    )
    obs_indices = torch.zeros(
        bsz, max_obs, device=device, dtype=torch.long
    )
    obs_field_ids = torch.full(
        (bsz, max_obs), -1, device=device, dtype=torch.long
    )

    if valid_mask is not None:
        if valid_mask.dim() == 1:
            valid_mask = valid_mask.unsqueeze(0).expand(bsz, -1)
        valid_mask = valid_mask.to(device=device, dtype=torch.bool)

    for b in range(bsz):
        if valid_mask is not None:
            valid_idx_b = torch.nonzero(valid_mask[b], as_tuple=False).squeeze(-1)
        else:
            valid_idx_b = None

        cursor = 0
        for fld, nmin, nmax in zip(cond_fields, n_obs_min, n_obs_max):
            if grid_aware:
                if valid_idx_b is not None:
                    pool = valid_idx_b
                else:
                    pool = torch.arange(n_pts, device=device)
                V = int(pool.numel())
                if V == 0:
                    continue
                # Random permutation, then keep only the first-seen point per
                # unique grid cell so each sensor lands in a distinct cell.
                perm = pool[torch.randperm(V, device=device)]
                perm_cells = cells_all[b, perm]                       # [V]
                positions = torch.arange(V, device=device)
                min_pos = torch.full(
                    (n_cells_total,), V, device=device, dtype=torch.long,
                )
                min_pos.scatter_reduce_(0, perm_cells, positions, reduce="amin")
                is_first = positions == min_pos[perm_cells]           # [V]
                first_points = perm[is_first]                         # [U]
                pool_size = int(first_points.numel())
                eff_max = min(nmax, pool_size)
                eff_min = min(nmin, eff_max)
                if eff_max <= 0:
                    continue
                m = int(torch.randint(
                    low=eff_min, high=eff_max + 1, size=(1,), device=device,
                ).item())
                idx = first_points[:m].sort().values
            elif valid_idx_b is not None:
                pool_size = int(valid_idx_b.numel())
                eff_max = min(nmax, pool_size)
                eff_min = min(nmin, eff_max)
                if eff_max <= 0:
                    continue
                m = int(torch.randint(low=eff_min, high=eff_max + 1, size=(1,), device=device).item())
                perm = torch.randperm(pool_size, device=device)[:m]
                idx = valid_idx_b[perm].sort().values
            else:
                m = int(torch.randint(low=nmin, high=nmax + 1, size=(1,), device=device).item())
                idx = torch.randperm(n_pts, device=device)[:m].sort().values

            obs_coords[b, cursor:cursor + m] = coords_full[b, idx]
            obs_values[b, cursor:cursor + m, 0] = fields_full[b, idx, fld]
            obs_mask[b, cursor:cursor + m] = 1.0
            obs_indices[b, cursor:cursor + m] = idx
            obs_field_ids[b, cursor:cursor + m] = fld

            cursor += m

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def nearest_fill_grid(value_grid: torch.Tensor,
                      mask_grid: torch.Tensor) -> torch.Tensor:
    """Per-(batch, field) nearest-neighbor Voronoi fill of a sparse grid.

    Every unobserved cell is filled with the value of the closest observed
    cell under L2 distance on grid indices. Used by the 'interp' sparse
    conditioning mode: instead of feeding the denoiser a mostly-zero sparse
    field + mask, feed a dense Voronoi-interpolated field + mask so the
    network has an immediate inductive bias for smooth infill (Shu et al.,
    arXiv:2409.00230).

    Channels/samples with zero observations are left as-is (all zeros). The
    mask is NOT modified — callers should still pass the original binary
    mask alongside so the network can distinguish interpolated guesses
    from actual measurements.

    Implementation: scipy's exact Euclidean distance transform on CPU.
    Cheap at typical grid sizes (~1 ms per field per sample at 100x400);
    move to a GPU variant if it becomes a training bottleneck.
    """
    from scipy.ndimage import distance_transform_edt
    dev = value_grid.device
    dtype = value_grid.dtype
    vg = value_grid.detach().cpu().numpy()
    mg = mask_grid.detach().cpu().numpy() > 0.5
    filled = vg.copy()
    B, C = vg.shape[:2]
    for b in range(B):
        for c in range(C):
            m = mg[b, c]
            if not m.any() or m.all():
                continue
            inds = distance_transform_edt(
                ~m, return_distances=False, return_indices=True)
            filled[b, c] = vg[b, c][tuple(inds)]
    return torch.from_numpy(filled).to(device=dev, dtype=dtype)


# ═════════ §6. Metrics & logging ═════════

class MetricsLogger:
    def __init__(self, base_dir: str, Demo_Num: int, timestamp: str,
                 method_name: Optional[str] = None):
        """
        Initializes the logger, creates the timestamped directory,
        and sets up the CSV file with headers.
        """
        self.method_name = method_name or "Model"
        # Create timestamped directory: Loss_YYYYMMDD_HHMMSS
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(base_dir, f"Loss_DemoN{Demo_Num}_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.csv_path = os.path.join(self.save_dir, "losses.csv")
        self.plot_path = os.path.join(self.save_dir, "loss_curve.png")
        
        # Initialize CSV with headers
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss"])
            
        # Store history for dynamic plotting
        self.epochs = []
        self.train_losses = []
        self.val_losses = []

    def log_and_plot(self, epoch: int, train_loss: float, val_loss: float = None):
        """
        Saves the current epoch's losses to the CSV and updates the loss curve plot.
        Pass val_loss=None if validation wasn't run this epoch.
        """
        # 1. Update history
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # 2. Append to CSV
        with open(self.csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            # If val_loss is None, it writes an empty string for that cell
            writer.writerow([epoch, train_loss, val_loss if val_loss is not None else ""])
            
        # 3. Update the Plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.epochs, self.train_losses, label='Train Loss', marker='o', color='blue', markersize=4)
        
        # Filter out 'None' values for validation plotting
        v_epochs = [e for e, v in zip(self.epochs, self.val_losses) if v is not None]
        v_losses = [v for v in self.val_losses if v is not None]
        
        if v_losses:
            plt.plot(v_epochs, v_losses, label='Validation Loss', marker='s', color='orange', markersize=5)
            
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title(f'{self.method_name} Training Progress')
        plt.yscale('log')  # Log scale is usually best for flow matching MSE
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        # Overwrite the previous image
        plt.savefig(self.plot_path)
        plt.close() # Close figure to free memory

# ═════════ §7. Visualization ═════════

def _save_single_field_plot(
    true_f=None, pred_f=None, coords_xy=None, sensor_coords=None,
    field_name="field", epoch=0, save_dir=".", file_prefix=None,
    triang=None, body_polygon=None,
    dpi=300, cmap_field="coolwarm", cmap_err="inferno",
    contour_levels=20, contour_linewidth=0.5, contour_alpha=0.5,
    **kwargs,
):
    """
    Module-level 3-panel plot (Ground truth / Reconstruction / |Error|).

    Compatible with both the original keyword-only call from
    visualize_reconstruction and the positional calls in the baselines,
    which pass extra ``triang`` / ``body_polygon`` kwargs for body-fitted
    plotting. If ``triang`` is not supplied, a Delaunay triangulation is
    built from ``coords_xy`` as before.
    """
    # Superset of the earlier keyword-only signature; adds optional
    # triang / body_polygon kwargs used by baseline scripts
    # (train_latentfm_baseline.py, train_sit_baseline.py,
    # train_senseiver_baseline.py, evaluate_ffm.py, evaluate_s3gm.py).
    x_plot = coords_xy[:, 0]
    y_plot = coords_xy[:, 1]

    if triang is None:
        triang = mtri.Triangulation(x_plot, y_plot)

    err = np.abs(true_f - pred_f)
    l2_error = _normalized_l2(true_f, pred_f)

    field_min = float(np.nanmin([true_f.min(), pred_f.min()]))
    field_max = float(np.nanmax([true_f.max(), pred_f.max()]))

    positive_err = err[err > 0]
    err_min = float(positive_err.min()) if positive_err.size > 0 else 0.0
    err_max = float(err.max()) if err.size > 0 else 1.0

    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 1, wspace=0.0, hspace=0.20)
    ax_true = fig.add_subplot(gs[0, 0])
    ax_pred = fig.add_subplot(gs[1, 0])
    ax_err = fig.add_subplot(gs[2, 0])

    im_true = ax_true.tricontourf(
        triang, true_f, levels=100, cmap=cmap_field,
        vmin=field_min, vmax=field_max,
    )
    if contour_levels is not None:
        ax_true.tricontour(
            triang, true_f, levels=contour_levels, colors="white",
            linewidths=contour_linewidth, alpha=contour_alpha,
        )

    im_pred = ax_pred.tricontourf(
        triang, pred_f, levels=100, cmap=cmap_field,
        vmin=field_min, vmax=field_max,
    )
    if contour_levels is not None:
        ax_pred.tricontour(
            triang, pred_f, levels=contour_levels, colors="white",
            linewidths=contour_linewidth, alpha=contour_alpha,
        )

    im_err = ax_err.tricontourf(
        triang, err, levels=100, cmap=cmap_err,
        vmin=err_min, vmax=err_max, extend="both",
    )

    # Optional body polygon overlay (airfoil etc.)
    if body_polygon is not None and len(body_polygon) > 0:
        for ax in (ax_true, ax_pred, ax_err):
            ax.add_patch(Polygon(body_polygon, closed=True,
                                 facecolor="white", edgecolor="black",
                                 linewidth=0.8, zorder=3))

    if sensor_coords is not None and len(sensor_coords) > 0:
        ax_true.scatter(
            sensor_coords[:, 0], sensor_coords[:, 1],
            s=12.5, c="none", edgecolors="tab:green", linewidths=2.0,
            marker="o", zorder=4,
        )

    ax_true.set_title("Ground truth", fontsize=13)
    ax_pred.set_title("Reconstruction", fontsize=13)
    ax_err.set_title("|Error|", fontsize=13)

    for ax in (ax_true, ax_pred, ax_err):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    # Auto-zoom around the airfoil: the C-grid far field is ~10× the chord,
    # which makes the airfoil a single pixel at full extent. Frame the view
    # to a neighbourhood of the body so the field structure near the surface
    # is actually visible.
    if body_polygon is not None and len(body_polygon) > 0:
        bp = np.asarray(body_polygon)
        bx_min, bx_max = float(bp[:, 0].min()), float(bp[:, 0].max())
        by_min, by_max = float(bp[:, 1].min()), float(bp[:, 1].max())
        chord = max(bx_max - bx_min, by_max - by_min, 1e-6)
        pad_x = 1.0 * chord
        pad_y = 0.75 * chord
        cx = 0.5 * (bx_min + bx_max)
        cy = 0.5 * (by_min + by_max)
        half_w = 0.5 * (bx_max - bx_min) + pad_x
        half_h = 0.5 * (by_max - by_min) + pad_y
        for ax in (ax_true, ax_pred, ax_err):
            ax.set_xlim(cx - half_w, cx + half_w)
            ax.set_ylim(cy - half_h, cy + half_h)

    cbar_field = fig.colorbar(im_true, ax=[ax_true, ax_pred], shrink=0.6, pad=0.02)
    cbar_field.set_label(field_name)
    cbar_err = fig.colorbar(im_err, ax=ax_err, shrink=0.6, pad=0.02)
    cbar_err.set_label(f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}    |    Normalized L2 = {l2_error:.3e}",
        y=0.96, fontsize=14,
    )

    prefix = file_prefix if file_prefix is not None else f"epoch_{epoch:04d}"
    filename = os.path.join(save_dir, f"{prefix}_field_{field_name}.png")
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return l2_error

def _save_car_surface_field_plot(
    true_f, pred_f, coords_xyz, sensor_coords=None,
    vertices=None, triangles=None,
    field_name='p', epoch=0, save_dir='.', file_prefix=None,
    dpi=180, cmap_field='coolwarm', cmap_err='inferno', point_size=None,
):
    """3-row x 3-col projection plot of a 3D car surface scalar field.

    Rows: (Ground truth | Reconstruction | |Error|).
    Cols: (side view x-z | top view x-y | rear view y-z).
    Sensor locations are overlaid as green rings on the ground-truth row.

    coords_xyz: [N, 3] physical coordinates of the surface points.
    Field arrays are 1-D [N]. sensor_coords is [M, 3] or None.

    When ``vertices`` and ``triangles`` are provided (and N matches the number
    of triangles), each view renders the mesh as filled triangles coloured by
    the per-face field value — giving a continuous-looking surface instead of
    a scatter. Triangles are painter-sorted per view so near-camera faces
    draw on top of far ones.
    """
    import matplotlib.colors as _mcolors
    from matplotlib.collections import PolyCollection

    err = np.abs(true_f - pred_f)
    l2_error = _normalized_l2(true_f, pred_f)

    fmin = float(np.nanmin([true_f.min(), pred_f.min()]))
    fmax = float(np.nanmax([true_f.max(), pred_f.max()]))
    emin = float(err.min()) if err.size else 0.0
    emax = float(err.max()) if err.size else 1.0

    # Decide whether to render the mesh as filled triangles (continuous field)
    # or fall back to auto-sized scatter (sparse points / no mesh available).
    mesh_mode = (
        vertices is not None and triangles is not None
        and len(true_f) == len(triangles)
    )

    if point_size is None:
        # 8k points => s~6, 100k points => s~2; only used in scatter fallback.
        n = max(1, coords_xyz.shape[0])
        point_size = float(np.clip(3000.0 / np.sqrt(n), 1.5, 14.0))

    if mesh_mode:
        verts_np = vertices.cpu().numpy() if hasattr(vertices, 'cpu') else np.asarray(vertices)
        tris_np = triangles.cpu().numpy() if hasattr(triangles, 'cpu') else np.asarray(triangles)
        tris_np = tris_np.astype(np.int64, copy=False)
        vx, vy, vz = verts_np[:, 0], verts_np[:, 1], verts_np[:, 2]
        x_bounds = (float(vx.min()), float(vx.max()))
        y_bounds = (float(vy.min()), float(vy.max()))
        z_bounds = (float(vz.min()), float(vz.max()))
    else:
        x, y, z = coords_xyz[:, 0], coords_xyz[:, 1], coords_xyz[:, 2]
        x_bounds = (float(x.min()), float(x.max()))
        y_bounds = (float(y.min()), float(y.max()))
        z_bounds = (float(z.min()), float(z.max()))

    # Each view drops one axis (the "depth" axis). Painter's algorithm: sort
    # faces/points so far-camera ones are drawn first and near-camera ones
    # overlay them. Camera convention:
    #   Side view — look at the car from +y (outside the half-body).
    #   Top view  — look down from +z.
    #   Rear view — look at the back of the Ahmed body from +x (rear is x=0).
    views = [
        # (title, u_idx, v_idx, depth_idx, ulim, vlim)
        ('Side (x,z)', 0, 2, 1, x_bounds, z_bounds),
        ('Top (x,y)',  0, 1, 2, x_bounds, y_bounds),
        ('Rear (y,z)', 1, 2, 0, y_bounds, z_bounds),
    ]

    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(3, 3, wspace=0.08, hspace=0.15)
    row_titles = ['Ground truth', 'Reconstruction', '|Error|']
    row_data = [true_f, pred_f, err]
    row_cmap = [cmap_field, cmap_field, cmap_err]
    row_vmin = [fmin, fmin, emin]
    row_vmax = [fmax, fmax, emax]

    last_im = [None, None, None]  # colorbars per-row (field / field / error)
    for ri in range(3):
        for ci, (title, ui, vi, di, ulim, vlim) in enumerate(views):
            ax = fig.add_subplot(gs[ri, ci])
            norm = _mcolors.Normalize(vmin=row_vmin[ri], vmax=row_vmax[ri])

            if mesh_mode:
                # Painter's ordering by mean depth per triangle.
                tri_depth = verts_np[:, di][tris_np].mean(axis=1)
                order = np.argsort(tri_depth)
                polys = np.stack(
                    [verts_np[:, ui][tris_np[order]], verts_np[:, vi][tris_np[order]]],
                    axis=-1,
                )  # [N_face, 3, 2]
                coll = PolyCollection(
                    polys,
                    array=row_data[ri][order],
                    cmap=row_cmap[ri], norm=norm,
                    edgecolors='face', linewidths=0,
                )
                ax.add_collection(coll)
                im = coll
            else:
                # Scatter fallback: same painter idea on the point cloud.
                depth_vals = coords_xyz[:, di]
                order = np.argsort(depth_vals)
                uu = coords_xyz[:, ui][order]
                vv = coords_xyz[:, vi][order]
                im = ax.scatter(
                    uu, vv, c=row_data[ri][order],
                    s=point_size, marker='.',
                    cmap=row_cmap[ri], vmin=row_vmin[ri], vmax=row_vmax[ri],
                    linewidths=0,
                )
            last_im[ri] = im

            if sensor_coords is not None and len(sensor_coords) > 0 and ri == 0:
                # Sensors only on the ground-truth row so the recon / error
                # panels stay uncluttered.
                su = sensor_coords[:, ui]
                sv = sensor_coords[:, vi]
                ax.scatter(su, sv, s=40, facecolors='none',
                           edgecolors='lime', linewidths=1.5, marker='o', zorder=5)
            if ri == 0:
                ax.set_title(title, fontsize=11)
            if ci == 0:
                ax.set_ylabel(row_titles[ri], fontsize=11)
            ax.set_xlim(*ulim); ax.set_ylim(*vlim)
            ax.set_aspect('equal', adjustable='box')
            ax.set_xticks([]); ax.set_yticks([])

    cbar_field = fig.colorbar(last_im[0], ax=[fig.axes[i] for i in range(6)],
                              shrink=0.7, pad=0.02)
    cbar_field.set_label(field_name)
    cbar_err = fig.colorbar(last_im[2], ax=[fig.axes[i] for i in range(6, 9)],
                            shrink=0.7, pad=0.02)
    cbar_err.set_label(f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}   |   Normalized L2 = {l2_error:.3e}",
        y=0.995, fontsize=13,
    )

    prefix = file_prefix if file_prefix is not None else f"epoch_{epoch:04d}"
    path = os.path.join(save_dir, f"{prefix}_field_{field_name}.png")
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return l2_error


@torch.no_grad()
def visualize_reconstruction(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    epoch: int,
    device: torch.device,
    save_dir: str,
    cond_fields: Union[int, Sequence[int]] = (2,),
    n_obs: Union[int, Sequence[int]] = 256,
    n_steps: int = 32,
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    save_metrics_json: bool = True,

    return_payload: bool = False,
):
    """
    Reconstruct full fields from arbitrary sparse sensors and save improved plots.

    Example:
        cond_fields=[0, 2], n_obs=[128, 256]

    Returns
    -------
    metrics : dict
        Per-field normalized L2 errors.
    """
    model.eval()

    cond_fields = _to_int_list(cond_fields)
    n_obs = _broadcast_per_field(n_obs, cond_fields, "n_obs")

    sample = dataset[snapshot_index]

    # Normalized coordinates go into the model.
    coords = sample["coords"].unsqueeze(0).to(device)   # [1, N, D]
    # Original coordinates are used only for plotting.
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)

    truth = sample["fields"].unsqueeze(0).to(device)    # [1, N, C]

    # Datasets that restrict sensor placement to a subset of nodes (e.g. the
    # airfoil near-surface band) expose a `valid_sensor_mask`. The training
    # loop and the standalone evaluator both forward it to build_sparse_condition,
    # so the periodic-viz path must match — otherwise we plot reconstructions
    # conditioned on free-stream sensors the model was never trained on.
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs,
        n_obs_max=n_obs,   # exact sensor counts for visualization
        valid_mask=valid_mask,
    )

    # Optional full-mesh viz: datasets that expose `load_full_mesh_sample`
    # (e.g. CarCFDDataset) can provide every surface point for plotting, while
    # the model only saw `n_points` training points. We run the ODE sampler
    # on the dense coordinate set to paint a continuous color field; sensors
    # are still drawn from the training subsample since those are the points
    # the model conditioned on.
    use_full_mesh = hasattr(dataset, 'load_full_mesh_sample')
    mesh_vertices = None
    mesh_triangles = None
    if use_full_mesh:
        full = dataset.load_full_mesh_sample(snapshot_index)
        plot_coords = full['coords'].unsqueeze(0).to(device)
        plot_coords_raw = full['coords_raw'].unsqueeze(0).to(device)
        plot_truth = full['fields'].unsqueeze(0).to(device)
        mesh_vertices = full.get('vertices')
        mesh_triangles = full.get('triangles')
    else:
        plot_coords = coords
        plot_coords_raw = coords_raw
        plot_truth = truth

    recon = model.sample(
        coords=plot_coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=n_steps,
        # clamp_indices assumes coords == training subsample. When sampling on
        # the full mesh, these indices don't correspond to the same points,
        # so we drop the clamp there and let the model produce values at all
        # surface points from the sensor inputs.
        clamp_indices=obs_indices if not use_full_mesh else None,
    )

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = plot_truth * std.view(1, 1, -1) + mean.view(1, 1, -1)

    recon_phys = recon_phys[0].cpu().numpy()   # [N, C]
    truth_phys = truth_phys[0].cpu().numpy()   # [N, C]

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    # Sensor coords come from the training subsample; carry the raw version
    # so we can overlay them on the full-mesh plot in physical units.
    obs_coords_raw = coords_raw[0, obs_indices[0]].cpu().numpy()[valid.cpu().numpy()]

    # Use the plot-time coord set (full mesh for CarCFDDataset, training
    # subsample otherwise) so field values align spatially with what the
    # plotting helpers render.
    coords_np = plot_coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]
    # Detect 3-D surface datasets (e.g. CarCFDDataset). The car plot uses
    # coords_xyz for three orthographic projections instead of a single 2-D
    # triangulation, since there is no consistent unrolling of a car surface.
    is_3d_surface = (
        coords_np.shape[-1] >= 3 and getattr(dataset, 'grid_shape', None) is None
        and bool(np.ptp(coords_np[:, 2]) > 1e-6)
    )

    # Structured triangulation / body polygon when the dataset provides them
    # (e.g. AirfoilCGridDataset). Mirrors the standalone evaluator visuals.
    triang = None
    body_polygon = None
    if getattr(dataset, "grid_shape", None) is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if getattr(dataset, "airfoil_body_indices", None) is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    field_names = tuple(getattr(dataset, "field_names", FIELD_NAMES))
    metrics = {}

    for c, name in enumerate(field_names):
        true_f = truth_phys[:, c]
        pred_f = recon_phys[:, c]

        # Only overlay sensors belonging to this field.
        sensor_coords = None
        field_sensor_mask = (obs_field_ids_cpu == c)
        if np.any(field_sensor_mask):
            if use_full_mesh:
                # obs_coords_raw was gathered from the training subsample so
                # it is always valid regardless of which coord set is plotted.
                sensor_coords = obs_coords_raw[field_sensor_mask]
                if not is_3d_surface:
                    sensor_coords = sensor_coords[:, :2]
            elif is_3d_surface:
                sensor_coords = coords_np[obs_indices_cpu[field_sensor_mask]]
            else:
                sensor_coords = coords_xy[obs_indices_cpu[field_sensor_mask]]

        if is_3d_surface:
            l2_error = _save_car_surface_field_plot(
                true_f=true_f,
                pred_f=pred_f,
                coords_xyz=coords_np,
                sensor_coords=sensor_coords,
                vertices=mesh_vertices,
                triangles=mesh_triangles,
                field_name=name,
                epoch=epoch,
                save_dir=save_dir,
                file_prefix=file_tag,
            )
        else:
            l2_error = _save_single_field_plot(
                true_f=true_f,
                pred_f=pred_f,
                coords_xy=coords_xy,
                sensor_coords=sensor_coords,
                field_name=name,
                epoch=epoch,
                save_dir=save_dir,
                file_prefix=file_tag,
                triang=triang,
                body_polygon=body_polygon,
            )
        metrics[name] = l2_error

    if save_metrics_json:
        prefix = file_tag if file_tag is not None else f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "n_steps": int(n_steps),
            "metrics": metrics,
        }
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)

    if return_payload:
        payload = {
            "coords_xy": coords_xy,
            "truth_phys": truth_phys,
            "recon_phys": recon_phys,
            "obs_indices": obs_indices_cpu,
            "obs_field_ids": obs_field_ids_cpu,
            "field_names": list(field_names),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "n_steps": int(n_steps),
        }
        return metrics, payload

    return metrics


# ═════════ §8. Training-resume utilities ═════════

def find_latest_run_dir(save_root: Path, run_prefix: str) -> Optional[Path]:
    """Return the most recently-named run directory matching run_prefix
    directly under save_root, or None. Sort is lexicographic on the trailing
    YYYYMMDD_HHMMSS timestamp baked into the directory name.
    """
    save_root = Path(save_root)
    if not save_root.exists():
        return None
    candidates = [p for p in save_root.glob(f"{run_prefix}*") if p.is_dir()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.name)[-1]


def extract_run_timestamp(run_dir: Path, run_prefix: str) -> str:
    """Recover the YYYYMMDD_HHMMSS suffix from a run directory created with
    `{run_prefix}{timestamp}`. Falls back to the current time if the name
    does not match the expected pattern.
    """
    name = Path(run_dir).name
    if name.startswith(run_prefix):
        return name[len(run_prefix):]
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def backup_path(path: Path, suffix: str = "_bk") -> Path:
    """Return a sibling path with `suffix` inserted before the extension that
    does not already exist on disk. Adds a numeric counter if needed.
    """
    path = Path(path)
    candidate = path.with_name(f"{path.stem}{suffix}{path.suffix}")
    if not candidate.exists():
        return candidate
    idx = 1
    while True:
        candidate = path.with_name(f"{path.stem}{suffix}{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def backup_existing_artifact(path: Path) -> None:
    """Copy `path` (file or directory) to a `_bk` sibling if it exists. Used
    before overwriting artifacts from a prior run during RELOAD.
    """
    path = Path(path)
    if not path.exists():
        return
    target = backup_path(path)
    if path.is_dir():
        shutil.copytree(path, target)
    else:
        shutil.copy2(path, target)


# ═════════════════════════════════════════════════════════════════════════════
# §9. Shared grid ↔ point-cloud utilities used by multiple baselines
#     (SiT, S3GM, and any future grid-based method). These live here rather
#     than in a method-specific script because they are pure data-layout
#     helpers — no loss, no sampler, no training loop.
# ═════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int) -> None:
    """Seed numpy + torch (CPU & CUDA) for reproducible training runs."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_pad_size(Num_y: int, Num_x: int, factor: int) -> Tuple[int, int]:
    """Pad (Num_y, Num_x) up to the nearest multiple of `factor`.

    Callers pass the factor their architecture requires:
      - SiT patch tokenizer:  factor = patch_size
      - S3GM UNet:            factor = 2 ** (len(ch_mult) - 1)
    """
    H_pad = math.ceil(Num_y / factor) * factor
    W_pad = math.ceil(Num_x / factor) * factor
    return H_pad, W_pad


def pointcloud_to_grid_padded(
    x: torch.Tensor,
    Num_y: int,
    Num_x: int,
    H_pad: int,
    W_pad: int,
    grid_order: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[B, N, C] -> [B, C, H_pad, W_pad] with zero padding if needed.

    Distinct from `pointcloud_to_grid` above (which returns unpadded
    [B, C, Num_y, Num_x]); this variant produces the padded tensor that
    fully-convolutional / patch-tokenized architectures require. If
    grid_order is supplied, it maps row-major grid cells to original point
    indices and is applied before reshape, matching the FFM-FNO path.
    """
    B = x.shape[0]
    if grid_order is not None:
        x = x[:, grid_order.to(device=x.device), :]
    g = x.reshape(B, Num_y, Num_x, -1).permute(0, 3, 1, 2).contiguous()
    if H_pad > Num_y or W_pad > Num_x:
        g = F.pad(g, (0, W_pad - Num_x, 0, H_pad - Num_y), mode="constant", value=0)
    return g


def grid_to_pointcloud(
    x_grid: torch.Tensor,
    Num_y: int,
    Num_x: int,
    point_to_grid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """[B, C, H_pad, W_pad] -> [B, N, C], cropping padding back off.

    If point_to_grid is supplied, it maps original point indices to row-major
    grid cells and is used to return the point cloud in original dataset order.
    """
    x_crop = x_grid[:, :, :Num_y, :Num_x]
    B, C, H, W = x_crop.shape
    x = x_crop.permute(0, 2, 3, 1).reshape(B, H * W, C).contiguous()
    if point_to_grid is not None:
        x = x[:, point_to_grid.to(device=x.device), :]
    return x


def build_obs_grid_mask(
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: torch.Tensor,
    n_fields: int,
    n_pts: int,
    Num_y: int,
    Num_x: int,
    H_pad: int,
    W_pad: int,
    point_to_grid: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert PhyCoFlow-style sparse obs tensors into dense grid maps
    (observation values + binary mask), padded to (H_pad, W_pad).
    """
    B = obs_values.shape[0]
    device = obs_values.device
    dtype = obs_values.dtype

    value_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)
    mask_flat = torch.zeros(B, n_fields, n_pts, device=device, dtype=dtype)

    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        idx = obs_indices[b, valid].long()
        if point_to_grid is not None:
            idx = point_to_grid.to(device=device)[idx]
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        value_flat[b, fld, idx] = val
        mask_flat[b, fld, idx] = 1.0

    obs_value_grid = value_flat.reshape(B, n_fields, Num_y, Num_x)
    obs_mask_grid = mask_flat.reshape(B, n_fields, Num_y, Num_x)

    if H_pad > Num_y or W_pad > Num_x:
        obs_value_grid = F.pad(obs_value_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)
        obs_mask_grid = F.pad(obs_mask_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)

    return obs_value_grid, obs_mask_grid


def scatter_sensors_to_nodes(obs_values, obs_mask, obs_field_ids, obs_indices,
                             B, N, n_fields, device, dtype):
    """Build per-node sparse conditioning tensors for point-token based models."""
    value = torch.zeros(B, N, n_fields, device=device, dtype=dtype)
    mask = torch.zeros(B, N, n_fields, device=device, dtype=dtype)
    for b in range(B):
        valid = obs_mask[b].bool()
        if not valid.any():
            continue
        idx = obs_indices[b, valid].long()
        fld = obs_field_ids[b, valid].long()
        val = obs_values[b, valid, 0]
        value[b, idx, fld] = val
        mask[b, idx, fld] = 1.0
    return value, mask
