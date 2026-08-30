import os
import csv
import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

import h5py
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Optional, Tuple

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

def normalize_coords(coords: torch.Tensor) -> torch.Tensor:
    cmin = coords.min(dim=0).values
    cmax = coords.max(dim=0).values
    scale = (cmax - cmin).clamp_min(1e-8)
    return (coords - cmin) / scale

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
        self.h5_path = str(h5_path)
        self.split = split
        self.field_names = field_names
        self.stats_chunk = stats_chunk
        self.time_stride = time_stride
        self._h5 = None

        with h5py.File(self.h5_path, "r") as f:
            self.num_times = int(f["fields"].shape[1])
            raw_coords = torch.from_numpy(f["coordinates"][:, 0, 0, :].astype(np.float32))
            self.coords = normalize_coords(raw_coords)
            self.num_points = int(raw_coords.shape[0])
            self.num_fields = int(f["fields"].shape[-1])
            self.times = torch.from_numpy(f["time"][:].astype(np.float32))

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
            "coords": self.coords.clone(),
            "fields": x,
            "time_index": torch.tensor(t_idx, dtype=torch.long),
            "physical_time": self.times[t_idx].clone(),
        }

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

def build_sparse_condition(
    coords_full: torch.Tensor,
    fields_full: torch.Tensor,
    cond_field_idx: torch.Tensor,
    n_obs_min: int,
    n_obs_max: int,
):
    
    bsz, n_pts, _ = coords_full.shape
    device = coords_full.device
    max_obs = n_obs_max

    obs_coords  = torch.zeros(bsz, max_obs, coords_full.shape[-1], device=device, dtype=coords_full.dtype)
    obs_values  = torch.zeros(bsz, max_obs, 1, device=device, dtype=fields_full.dtype)
    obs_mask    = torch.zeros(bsz, max_obs, device=device, dtype=coords_full.dtype)
    obs_indices = torch.zeros(bsz, max_obs, device=device, dtype=torch.long)

    for b in range(bsz):
        # Sample m random observation points
        m   = int(torch.randint(low=n_obs_min, high=n_obs_max + 1, size=(1,), device=device).item())
        idx = torch.randperm(n_pts, device=device)[:m].sort().values
        fld = int(cond_field_idx[b].item())

        obs_coords[b, :m]    = coords_full[b, idx]
        obs_values[b, :m, 0] = fields_full[b, idx, fld]
        obs_mask[b, :m]      = 1.0
        obs_indices[b, :m]   = idx

    return obs_coords, obs_values, obs_mask, obs_indices

@torch.no_grad()
def visualize_reconstruction(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    epoch: int,
    device: torch.device,
    save_dir: str,
    cond_field: int = 2,
    n_obs: int = 256,
    n_steps: int = 100,
    snapshot_index: int = 0
):
    """
    Randomly samples sensors, reconstructs the fields, calculates Normalized L2 error, 
    and saves the plots to the timestamped directory using unstructured triangulation.
    """
    model.eval()
    
    # Grab a sample from the validation dataset
    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    
    n_pts = coords.shape[1]
    
    # Randomly select sensor coordinates
    idx = torch.randperm(n_pts, device=device)[:n_obs].sort().values
    
    obs_coords = coords[:, idx]
    obs_values = truth[:, idx, cond_field : cond_field + 1]
    obs_mask = torch.ones(1, n_obs, device=device, dtype=coords.dtype)
    obs_indices = idx.unsqueeze(0)
    cond_field_idx = torch.tensor([cond_field], device=device, dtype=torch.long)
    
    # Reconstruct the field
    recon = model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        cond_field_idx=cond_field_idx,
        n_steps=n_steps,
        clamp_indices=obs_indices,
    )
    
    # Denormalize
    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)
    
    # Move to CPU and numpy for plotting
    recon_phys = recon_phys[0].cpu().numpy()
    truth_phys = truth_phys[0].cpu().numpy()
    idx_cpu = idx.cpu().numpy()
    
    # Extract X and Y coordinates to build the unstructured mesh triangulation
    coords_np = coords[0].cpu().numpy()
    x_coords = coords_np[:, 0]
    y_coords = coords_np[:, 1]
    tri = mtri.Triangulation(x_coords, y_coords)
    
    # Plotting
    for c, name in enumerate(FIELD_NAMES):
        fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        
        # Calculate Normalized L2 Error
        true_f = truth_phys[:, c]
        pred_f = recon_phys[:, c]
        l2_error = np.linalg.norm(true_f - pred_f) / (np.linalg.norm(true_f) + 1e-8)
        
        # Get unified min/max so both plots share the exact same color scale
        vmin = min(true_f.min(), pred_f.min())
        vmax = max(true_f.max(), pred_f.max())
        
        # Truth Plot
        im0 = axs[0].tricontourf(tri, true_f, levels=200, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axs[0].set_title(f"Truth: {name}")
        axs[0].set_aspect("equal")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        plt.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        
        # Reconstruction Plot (with Normalized L2 Error in title)
        im1 = axs[1].tricontourf(tri, pred_f, levels=200, cmap="coolwarm", vmin=vmin, vmax=vmax)
        axs[1].set_title(f"Recon: {name} | L2 Err: {l2_error:.4f}")
        axs[1].set_aspect("equal")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("y")
        plt.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
        
        # Overlay white dots for sensors using their exact (x, y) spatial coordinates
        if c == cond_field:
            sensor_x = x_coords[idx_cpu]
            sensor_y = y_coords[idx_cpu]
            axs[1].scatter(sensor_x, sensor_y, s=15, c="white", edgecolors='black', linewidth=0.5)
            
        # Save image
        save_path = os.path.join(save_dir, f"epoch_{epoch:04d}_field_{c}_{name}.png")
        fig.savefig(save_path, dpi=120)
        plt.close(fig)