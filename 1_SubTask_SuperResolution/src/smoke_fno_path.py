"""
Lightweight smoke checks for the mixed-resolution FNO path.

Run from the task root:
    python src/smoke_fno_path.py
"""

import tempfile
from pathlib import Path

import torch

from helpers import validate_regular_grid_compatibility, visualize_reconstruction
from Model import FNO, FNOFFM


class ZeroPrior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.zeros(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


def make_grid(num_x: int, num_y: int, shuffled: bool = False, broken: bool = False):
    x = torch.linspace(0.0, 1.0, num_x)
    y = torch.linspace(0.0, 1.0, num_y)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    coords = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(num_x * num_y)], dim=-1)
    if broken:
        coords[-1] = coords[0]
    if shuffled:
        coords = coords[torch.randperm(coords.shape[0])]
    fields = torch.sin(2.0 * torch.pi * coords[:, :1]) + torch.cos(2.0 * torch.pi * coords[:, 1:2])
    return coords, fields


class SyntheticMixedDataset(torch.utils.data.Dataset):
    def __init__(self, broken_h: bool = False):
        self.coords_by_res = {}
        self.coords_raw_by_res = {}
        self.resolutions = {}
        for tag, size in (("L", 32), ("M", 64), ("H", 128)):
            coords, _ = make_grid(size, size, shuffled=(tag != "H"), broken=(broken_h and tag == "H"))
            self.coords_by_res[tag] = coords
            self.coords_raw_by_res[tag] = coords.clone()
            self.resolutions[tag] = {"Num_x": size, "Num_y": size, "num_points": size * size}

        self.coords = self.coords_by_res["L"]
        self.coords_raw = self.coords_raw_by_res["L"]
        _, self.fields = make_grid(32, 32, shuffled=False)
        self.num_points = int(self.coords.shape[0])
        self.num_fields = 1
        self.field_names = ("u",)
        self.mean = torch.zeros(1)
        self.std = torch.ones(1)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return {
            "coords": self.coords,
            "coords_raw": self.coords_raw,
            "fields": self.fields,
            "resolution_tag": "L",
        }


class DummySamplerNoOde(torch.nn.Module):
    def sample(
        self,
        coords,
        obs_coords,
        obs_values,
        obs_mask,
        obs_field_ids,
        n_steps,
        clamp_indices,
        obs_consistency_mode="default_hard",
        obs_consistency_strength=1.0,
        obs_consistency_sigma=0.05,
        obs_consistency_schedule_power=2.0,
        obs_consistency_final_clamp=True,
        obs_consistency_chunk_size=8192,
    ):
        return torch.zeros(coords.shape[0], coords.shape[1], 1, device=coords.device, dtype=coords.dtype)


def make_observations(coords: torch.Tensor, fields: torch.Tensor):
    obs_indices = torch.tensor([[0, coords.shape[1] // 3, 2 * coords.shape[1] // 3]], device=coords.device)
    obs_field_ids = torch.zeros_like(obs_indices)
    obs_mask = torch.ones_like(obs_indices, dtype=coords.dtype)
    obs_coords = coords[:, obs_indices[0]]
    obs_values = fields[:, obs_indices[0], :1]
    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")

    grid_info = validate_regular_grid_compatibility(SyntheticMixedDataset(), 128, 128)
    assert set(grid_info["resolutions"]) == {"L", "M", "H"}
    try:
        validate_regular_grid_compatibility(SyntheticMixedDataset(broken_h=True), 128, 128)
    except ValueError:
        pass
    else:
        raise AssertionError("validate_regular_grid_compatibility should fail for incomplete grids")

    backbone = FNO(
        n_fields=1,
        Num_x=128,
        Num_y=128,
        n_modes_x=8,
        n_modes_y=8,
        hidden_channels=8,
        n_layers=2,
        condition_blur=True,
        condition_blur_kernel=3,
        condition_blur_sigma=1.0,
    ).to(device)
    model = FNOFFM(backbone, ZeroPrior()).to(device)

    for size in (32, 64, 128):
        coords, fields = make_grid(size, size, shuffled=(size != 64))
        coords = coords.unsqueeze(0).to(device)
        fields = fields.unsqueeze(0).to(device)
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = make_observations(coords, fields)
        y = backbone(
            torch.tensor([0.25], device=device),
            fields,
            coords,
            obs_coords,
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices=obs_indices,
        )
        assert y.shape == fields.shape

        grid_order, point_to_grid, num_x, num_y = backbone._get_grid_permutation(coords)
        roundtrip = backbone._grid_to_pointcloud(
            backbone._pointcloud_to_grid(fields, grid_order, num_x, num_y),
            point_to_grid,
        )
        assert torch.allclose(roundtrip, fields)

    coords, fields = make_grid(32, 32, shuffled=True)
    coords = coords.unsqueeze(0).to(device)
    fields = fields.unsqueeze(0).to(device)
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = make_observations(coords, fields)
    for solver in ("euler", "heun"):
        sample = model.sample(
            coords=coords,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            n_steps=2,
            clamp_indices=obs_indices,
            ode_solver=solver,
        )
        assert sample.shape == fields.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = visualize_reconstruction(
            model=DummySamplerNoOde(),
            dataset=SyntheticMixedDataset(),
            epoch=0,
            device=device,
            save_dir=tmpdir,
            cond_fields=[0],
            n_obs=[4],
            n_steps=1,
            ode_solver="heun",
            file_tag="dummy_no_ode",
            save_metrics_json=False,
        )
        assert "u" in metrics
        assert Path(tmpdir).exists()

    print("Mixed-resolution FNO path smoke checks passed.")


if __name__ == "__main__":
    main()
