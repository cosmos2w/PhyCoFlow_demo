
'''
With this patch:

- training can use any configured field combination like [0], [2], [0, 2], [0, 2, 4]

- each conditioned field can have its own n_obs_min / n_obs_max

- visualization can use its own cond_fields and exact n_obs list, independent of training
'''

import os
import csv
import torch
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.gridspec as gridspec
import torch.nn.functional as F

import h5py
from torch.utils.data import Dataset, DataLoader, Sampler
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence, Union

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

def validate_regular_grid_compatibility(
    dataset: Dataset,
    Num_x: Optional[int],
    Num_y: Optional[int],
    decimals: int = 6,
) -> None:
    """
    Validate that a point-cloud dataset can be interpreted as a Num_x by Num_y regular grid.

    Required behavior for the FNO branch:
      - Num_x and Num_y must be provided in YAML / args
      - Num_x * Num_y must match the number of points
      - the coordinate set must contain exactly Num_x unique x-values
        and Num_y unique y-values (up to rounding)

    Raises ValueError if the dataset is not compatible with the requested grid.
    """
    if Num_x is None or Num_y is None:
        raise ValueError(
            "FNO backbone requires Num_x and Num_y to be explicitly provided in YAML / args."
        )

    Num_x = int(Num_x)
    Num_y = int(Num_y)

    if Num_x <= 0 or Num_y <= 0:
        raise ValueError(f"Num_x and Num_y must be positive, got Num_x={Num_x}, Num_y={Num_y}.")

    expected_points = Num_x * Num_y
    if int(dataset.num_points) != expected_points:
        raise ValueError(
            f"Grid mismatch: dataset has {dataset.num_points} points, but "
            f"Num_x * Num_y = {Num_x} * {Num_y} = {expected_points}."
        )

    coords = dataset.coords.cpu()
    x = torch.round(coords[:, 0] * (10 ** decimals)) / (10 ** decimals)
    y = torch.round(coords[:, 1] * (10 ** decimals)) / (10 ** decimals)

    unique_x = int(torch.unique(x).numel())
    unique_y = int(torch.unique(y).numel())

    if unique_x != Num_x or unique_y != Num_y:
        raise ValueError(
            "[x] Grid compatibility check failed. "
            f"Dataset unique counts are ({unique_x}, {unique_y}) in (x, y), "
            f"but requested (Num_x, Num_y)=({Num_x}, {Num_y})."
        )

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    scale = (cmax - cmin).clamp_min(1e-8)
    return (coords - cmin) / scale

# The following helpers solve the sensor-comparison question cleanly:
# define sensors once in physical coordinates on the H domain,
# project them to nearest nodes in L/M/H,
# compare reconstructions under the same physical sensor layout, not the same discrete indices.

def load_multires_manifest(manifest_path: str) -> dict:
    with open(manifest_path, "r") as f:
        return json.load(f)

def _resolution_sort_key(tag: str) -> int:
    order = {"L": 0, "M": 1, "H": 2}
    return order[tag]

def project_physical_sensor_coords_to_indices(
    coords_raw_xy: torch.Tensor,
    sensor_coords_xy: torch.Tensor,
) -> torch.Tensor:
    """
    Map canonical physical sensor coordinates to the nearest available points
    on a target resolution.

    This is the correct way to compare 'the same sensors' across resolutions:
    define the sensors in physical space once, then snap them to each grid.
    """
    # coords_raw_xy: [N, 2]
    # sensor_coords_xy: [K, 2]
    d2 = torch.cdist(sensor_coords_xy.unsqueeze(0), coords_raw_xy.unsqueeze(0), p=2.0).squeeze(0)
    return torch.argmin(d2, dim=1)

def build_sparse_condition_from_fixed_sensor_coords(
    coords_full: torch.Tensor,
    coords_raw_full: torch.Tensor,
    fields_full: torch.Tensor,
    sensor_coords_xy: torch.Tensor,
    cond_fields: Union[int, Sequence[int]],
):
    """
    Optional helper for future cross-resolution evaluation:
    define sensor positions once in physical coordinates, then snap them to
    the current resolution. Not used by the default training loop because the
    validation set is high resolution only, but this is the correct mechanism
    for fair L/M/H comparison later.
    """
    cond_fields = _to_int_list(cond_fields)
    device = coords_full.device
    bsz = coords_full.shape[0]
    n_obs = sensor_coords_xy.shape[0]

    obs_coords = torch.zeros(bsz, n_obs * len(cond_fields), coords_full.shape[-1], device=device, dtype=coords_full.dtype)
    obs_values = torch.zeros(bsz, n_obs * len(cond_fields), 1, device=device, dtype=fields_full.dtype)
    obs_mask = torch.zeros(bsz, n_obs * len(cond_fields), device=device, dtype=coords_full.dtype)
    obs_indices = torch.zeros(bsz, n_obs * len(cond_fields), device=device, dtype=torch.long)
    obs_field_ids = torch.full((bsz, n_obs * len(cond_fields)), -1, device=device, dtype=torch.long)

    sensor_coords_xy = sensor_coords_xy.to(device)

    for b in range(bsz):
        idx = project_physical_sensor_coords_to_indices(coords_raw_full[b, :, :2], sensor_coords_xy)
        cursor = 0
        for fld in cond_fields:
            m = idx.numel()
            obs_coords[b, cursor:cursor + m] = coords_full[b, idx]
            obs_values[b, cursor:cursor + m, 0] = fields_full[b, idx, fld]
            obs_mask[b, cursor:cursor + m] = 1.0
            obs_indices[b, cursor:cursor + m] = idx
            obs_field_ids[b, cursor:cursor + m] = fld
            cursor += m

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


class TurbulentCombustionH5Dataset(Dataset):
    """Treat each time snapshot as one point-cloud sample."""

    def __init__(
        self,
        h5_path: str,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        field_names: Tuple[str, ...] = ("CH4", "CO", "T", "U_1", "p"),
        stats_path: Optional[str] = None,
        stats_chunk: int = 32,
        time_stride: int = 1,
    ) -> None:
        super().__init__()
        self.h5_path     = str(h5_path)
        self.split       = split
        self.field_names = field_names
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
            obj = torch.load(stats_path, map_location="cpu")
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


class PDEBenchMultiResDataset(Dataset):
    """
    Lightweight multi-resolution dataset wrapper for PDEBench processed files.

    Key behaviors:
    - split is case-based, not time-based
    - train set is reorganized by resolution ratio from a JSON manifest
    - val/test set uses the held-out cases and defaults to high resolution
    - only one selected raw field is exposed for this task
    - normalization is computed once from HIGH-resolution train cases only
      so that L/M/H live in the same value space
    """

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        eval_resolution: str = "H",
        force_resolution: Optional[str] = None,
        stats_path: Optional[str] = None,
        stats_chunk_cases: int = 8,
    ) -> None:
        super().__init__()

        self.manifest_path = str(manifest_path)
        self.manifest = load_multires_manifest(self.manifest_path)
        self.split = split
        self.eval_resolution = eval_resolution.upper()
        self.force_resolution = None if force_resolution is None else force_resolution.upper()
        self.stats_chunk_cases = stats_chunk_cases
        self._h5 = {}

        self.dataset_name = self.manifest["dataset_name"]
        self.selected_field_idx_raw = int(self.manifest["selected_field_idx"])
        self.field_names = (self.manifest["selected_field_name"],)
        self.num_fields = 1

        self.paths = {k: str(v) for k, v in self.manifest["paths"].items()}
        self.resolutions = self.manifest["resolutions"]

        # Preload coordinates for each resolution
        self.coords_raw_by_res = {}
        self.coords_by_res = {}
        for res_tag, path in self.paths.items():
            with h5py.File(path, "r") as f:
                raw_coords = torch.from_numpy(f["coordinates"][:, 0, 0, :].astype(np.float32))
                self.coords_raw_by_res[res_tag] = raw_coords
                self.coords_by_res[res_tag] = normalize_coords(raw_coords)

        # Build sample entries: (native_resolution, case_id, time_id)
        self.entries = []
        self.indices_by_res = {"L": [], "M": [], "H": []}

        split_info = self.manifest["split"]
        n_time = int(self.manifest["n_time"])

        if split == "train":
            for res_tag in ["L", "M", "H"]:
                case_ids = split_info["train_cases_by_res"][res_tag]
                for case_id in case_ids:
                    for t_idx in range(n_time):
                        ds_idx = len(self.entries)
                        self.entries.append((res_tag, int(case_id), int(t_idx)))
                        self.indices_by_res[res_tag].append(ds_idx)
        elif split in ["val", "test"]:
            case_ids = split_info["val_cases"]
            for case_id in case_ids:
                for t_idx in range(n_time):
                    ds_idx = len(self.entries)
                    self.entries.append((self.eval_resolution, int(case_id), int(t_idx)))
                    self.indices_by_res[self.eval_resolution].append(ds_idx)
        else:
            raise ValueError(f"Unknown split: {split}")

        # Output resolution used by __getitem__
        if self.force_resolution is not None:
            self.output_resolution = self.force_resolution
        elif split == "train":
            # Mixed train dataset: output resolution varies by entry.
            self.output_resolution = None
        else:
            self.output_resolution = self.eval_resolution

        # num_points is fixed only if the dataset outputs one resolution
        if self.output_resolution is not None:
            self.num_points = self.coords_by_res[self.output_resolution].shape[0]
            self.coords = self.coords_by_res[self.output_resolution]
            self.coords_raw = self.coords_raw_by_res[self.output_resolution]
        else:
            self.num_points = None
            self.coords = None
            self.coords_raw = None

        # Use time vector from H file as canonical time
        with h5py.File(self.paths["H"], "r") as f:
            self.times = torch.from_numpy(f["time"][:].astype(np.float32))

        self.stats_path = stats_path or str(Path(self.manifest_path).with_suffix(".stats.pt"))
        self.mean, self.std = self._load_or_compute_stats()

        # Mixed-resolution train batches must be grouped by resolution.
        self.requires_grouped_batches = (split == "train" and self.force_resolution is None)

    def _require_h5(self, res_tag: str):
        if res_tag not in self._h5:
            self._h5[res_tag] = h5py.File(self.paths[res_tag], "r")
        return self._h5[res_tag]

    def _base_train_cases(self):
        split_info = self.manifest["split"]["train_cases_by_res"]
        all_cases = split_info["L"] + split_info["M"] + split_info["H"]
        return sorted([int(c) for c in all_cases])

    def _load_or_compute_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        stats_path = Path(self.stats_path)
        if stats_path.exists():
            obj = torch.load(stats_path, map_location="cpu")
            return obj["mean"].float(), obj["std"].float()

        # Important:
        # compute common normalization from HIGH-resolution train cases only
        # so that L/M/H are normalized in one shared value space.
        train_cases = self._base_train_cases()
        h5 = self._require_h5("H")

        total_sum = torch.zeros(1, dtype=torch.float64)
        total_sq = torch.zeros(1, dtype=torch.float64)
        total_count = 0

        for start in range(0, len(train_cases), self.stats_chunk_cases):
            case_chunk = train_cases[start:start + self.stats_chunk_cases]
            arr = h5["fields"][case_chunk, :, :, 0, 0, self.selected_field_idx_raw]   # [B_case, T, N]
            x = torch.from_numpy(arr.astype(np.float32)).reshape(-1, 1)
            total_sum += x.sum(dim=0, dtype=torch.float64)
            total_sq += (x.double() ** 2).sum(dim=0)
            total_count += x.shape[0]

        mean = (total_sum / total_count).float()
        var = (total_sq / total_count - mean.double() ** 2).clamp_min(1e-12).float()
        std = torch.sqrt(var)

        stats_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mean": mean, "std": std}, stats_path)
        return mean, std

    def _resample_field(self, x_native: torch.Tensor, native_res: str, target_res: str) -> torch.Tensor:
        """
        Resample a single-channel field from native resolution to target resolution.

        x_native: [N_native, 1]
        returns : [N_target, 1]
        """
        if native_res == target_res:
            return x_native

        ny_src = int(self.resolutions[native_res]["Num_y"])
        nx_src = int(self.resolutions[native_res]["Num_x"])
        ny_tgt = int(self.resolutions[target_res]["Num_y"])
        nx_tgt = int(self.resolutions[target_res]["Num_x"])

        x_grid = x_native.view(ny_src, nx_src, 1).permute(2, 0, 1).unsqueeze(0)   # [1,1,Ny,Nx]
        x_grid = F.interpolate(
            x_grid,
            size=(ny_tgt, nx_tgt),
            mode="bilinear",
            align_corners=False,
        )
        x_out = x_grid.squeeze(0).permute(1, 2, 0).reshape(-1, 1)
        return x_out

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        native_res, case_id, t_idx = self.entries[i]

        if self.force_resolution is not None:
            out_res = self.force_resolution
        elif self.split == "train":
            out_res = native_res
        else:
            out_res = self.eval_resolution

        h5 = self._require_h5(native_res)
        x = h5["fields"][case_id, t_idx, :, 0, 0, self.selected_field_idx_raw].astype(np.float32)
        x = torch.from_numpy(x).reshape(-1, 1)

        if out_res != native_res:
            x = self._resample_field(x, native_res=native_res, target_res=out_res)

        x = (x - self.mean) / self.std

        return {
            "coords": self.coords_by_res[out_res].clone(),
            "coords_raw": self.coords_raw_by_res[out_res].clone(),
            "fields": x,
            "time_index": torch.tensor(t_idx, dtype=torch.long),
            "physical_time": self.times[t_idx].clone(),
            "resolution_tag": out_res,
            "native_resolution_tag": native_res,
            "case_id": torch.tensor(case_id, dtype=torch.long),
        }


class ResolutionGroupedBatchSampler(Sampler):
    """
    Ensure that each training batch contains samples from only one resolution,
    so tensors can still be stacked without padding.

    This is essential for mixed-resolution point-cloud training.
    """
    def __init__(self, dataset: PDEBenchMultiResDataset, batch_size: int, shuffle: bool = True, drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.drop_last = drop_last

        if not getattr(dataset, "requires_grouped_batches", False):
            raise ValueError("ResolutionGroupedBatchSampler should only be used for mixed-resolution train datasets.")

    def __iter__(self):
        rng = np.random.default_rng()

        all_batches = []
        for res_tag in ["L", "M", "H"]:
            indices = list(self.dataset.indices_by_res[res_tag])
            if len(indices) == 0:
                continue

            if self.shuffle:
                rng.shuffle(indices)

            for start in range(0, len(indices), self.batch_size):
                batch = indices[start:start + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        if self.shuffle:
            rng.shuffle(all_batches)

        for batch in all_batches:
            yield batch

    def __len__(self):
        total = 0
        for res_tag in ["L", "M", "H"]:
            n = len(self.dataset.indices_by_res[res_tag])
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


class MetricsLogger:
    def __init__(self, base_dir: str, Demo_Num: int, timestamp: str):
        """
        Initializes the logger, creates the timestamped directory, 
        and sets up the CSV file with headers.
        """
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
        plt.title('Conditional Point-Cloud FFM Training Progress')
        plt.yscale('log')  # Log scale is usually best for flow matching MSE
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        
        # Overwrite the previous image
        plt.savefig(self.plot_path)
        plt.close() # Close figure to free memory


def create_recon_dir(base_dir: str, Demo_Num: int, timestamp: str) -> str:
    """Creates a timestamped directory for saving evaluation plots."""
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, "ffm_tc_pointcloud", f"demo_N{Demo_Num}_{timestamp}")
    os.makedirs(path, exist_ok=True)
    return path


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


def build_sparse_condition(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    cond_fields: Union[int, Sequence[int]],
    n_obs_min: Union[int, Sequence[int]],
    n_obs_max: Union[int, Sequence[int]],
):
    """
    Generalized sparse conditioning.

    Args:
        coords_full: [B, N, D]
        fields_full: [B, N, C]
        cond_fields: int or list[int], e.g. 2 or [0, 2]
        n_obs_min: int or list[int], per conditioned field
        n_obs_max: int or list[int], per conditioned field

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

    for b in range(bsz):
        cursor = 0
        for fld, nmin, nmax in zip(cond_fields, n_obs_min, n_obs_max):
            m = int(torch.randint(low=nmin, high=nmax + 1, size=(1,), device=device).item())
            idx = torch.randperm(n_pts, device=device)[:m].sort().values

            obs_coords[b, cursor:cursor + m] = coords_full[b, idx]
            obs_values[b, cursor:cursor + m, 0] = fields_full[b, idx, fld]
            obs_mask[b, cursor:cursor + m] = 1.0
            obs_indices[b, cursor:cursor + m] = idx
            obs_field_ids[b, cursor:cursor + m] = fld

            cursor += m

    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def _normalized_l2(u_true: np.ndarray, u_pred: np.ndarray) -> float:
    return float(np.linalg.norm(u_true - u_pred) / (np.linalg.norm(u_true) + 1e-8))


def _save_single_field_plot(
    *,
    true_f: np.ndarray,
    pred_f: np.ndarray,
    coords_xy: np.ndarray,
    sensor_coords: Optional[np.ndarray],
    field_name: str,
    epoch: int,
    save_dir: str,
    file_prefix: Optional[str] = None,

    dpi: int = 300,
    cmap_field: str = "coolwarm", # "viridis",
    cmap_err: str = "inferno",

    contour_levels: Optional[int] = 20,
    contour_linewidth: float = 0.5,
    contour_alpha: float = 0.5,
):
    """
    Save one high-quality 3-panel plot:
        Ground truth | Reconstruction | |Error|
    while preserving the same per-field saving logic and L2 evaluation logic.
    """
    x_plot = coords_xy[:, 0]
    y_plot = coords_xy[:, 1]
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
        vmin=field_min, vmax=field_max
    )
    if contour_levels is not None:
        ax_true.tricontour(
            triang, true_f, levels=contour_levels, colors="white",
            linewidths=contour_linewidth, alpha=contour_alpha
        )

    im_pred = ax_pred.tricontourf(
        triang, pred_f, levels=100, cmap=cmap_field,
        vmin=field_min, vmax=field_max
    )
    if contour_levels is not None:
        ax_pred.tricontour(
            triang, pred_f, levels=contour_levels, colors="white",
            linewidths=contour_linewidth, alpha=contour_alpha
        )

    im_err = ax_err.tricontourf(
        triang, err, levels=100, cmap=cmap_err,
        vmin=err_min, vmax=err_max, extend="both"
    )

    if sensor_coords is not None and len(sensor_coords) > 0:
        ax_true.scatter(
            sensor_coords[:, 0], sensor_coords[:, 1],
            s=12.5, c="none", edgecolors="tab:green", linewidths=2.0,
            marker="o", zorder=4
        )

    ax_true.set_title("Ground truth", fontsize=13)
    ax_pred.set_title("Reconstruction", fontsize=13)
    ax_err.set_title("|Error|", fontsize=13)

    for ax in (ax_true, ax_pred, ax_err):
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    cbar_field = fig.colorbar(
        im_true, ax=[ax_true, ax_pred], shrink=0.6, pad=0.02
    )
    cbar_field.set_label(field_name)

    cbar_err = fig.colorbar(im_err, ax=ax_err, shrink=0.6, pad=0.02)
    cbar_err.set_label(f"|{field_name} - û|")

    fig.suptitle(
        f"{field_name}    |    Normalized L2 = {l2_error:.3e}",
        y=0.96,
        fontsize=14,
    )

    prefix = file_prefix if file_prefix is not None else f"epoch_{epoch:04d}"
    filename = os.path.join(save_dir, f"{prefix}_field_{field_name}.png")
    fig.savefig(filename, dpi=dpi, bbox_inches="tight")
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

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs,
        n_obs_max=n_obs,   # exact sensor counts for visualization
    )

    recon = model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=n_steps,
        clamp_indices=obs_indices,
    )

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)

    recon_phys = recon_phys[0].cpu().numpy()   # [N, C]
    truth_phys = truth_phys[0].cpu().numpy()   # [N, C]

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()

    # coords_np = coords[0].cpu().numpy()
    coords_np = coords_raw[0].cpu().numpy()
    coords_xy = coords_np[:, :2]

    metrics = {}
    field_names = tuple(getattr(dataset, "field_names", FIELD_NAMES))

    for c, name in enumerate(field_names):
        true_f = truth_phys[:, c]
        pred_f = recon_phys[:, c]

        # Only overlay sensors belonging to this field.
        sensor_coords = None
        field_sensor_mask = (obs_field_ids_cpu == c)
        if np.any(field_sensor_mask):
            sensor_coords = coords_xy[obs_indices_cpu[field_sensor_mask]]

        l2_error = _save_single_field_plot(
            true_f=true_f,
            pred_f=pred_f,
            coords_xy=coords_xy,
            sensor_coords=sensor_coords,
            field_name=name,
            epoch=epoch,
            save_dir=save_dir,
            file_prefix=file_tag,
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

    return metrics

# -------------------------------------------
