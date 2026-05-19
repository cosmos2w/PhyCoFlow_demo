"""
Lightweight smoke checks for the FNO baseline path.

Run from the demo root:
    python src/smoke_fno_path.py
"""

import tempfile
from pathlib import Path

import torch

from helpers import (
    reconstruct_snapshot,
    validate_regular_grid_compatibility,
    visualize_reconstruction,
)
from Model import FNO, FNOFFM


class ZeroPrior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.zeros(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class SyntheticGridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_x: int = 8,
        num_y: int = 6,
        shuffled: bool = False,
        broken: bool = False,
    ):
        x = torch.linspace(0.0, 1.0, num_x)
        y = torch.linspace(0.0, 1.0, num_y)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1), torch.zeros(num_x * num_y)], dim=-1)
        if broken:
            coords[-1] = coords[0]
        if shuffled:
            coords = coords[torch.randperm(coords.shape[0])]

        self.coords = coords
        self.coords_raw = coords.clone()
        self.num_points = int(coords.shape[0])
        self.num_fields = 2
        self.field_names = ("A", "B")
        self.mean = torch.zeros(self.num_fields)
        self.std = torch.ones(self.num_fields)

        self.fields = torch.stack(
            [
                torch.sin(2.0 * torch.pi * coords[:, 0]),
                torch.cos(2.0 * torch.pi * coords[:, 1]),
            ],
            dim=-1,
        )

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int):
        return {
            "coords": self.coords,
            "coords_raw": self.coords_raw,
            "fields": self.fields,
        }


class DummySamplerNoOde(torch.nn.Module):
    """Exercises helper signature inspection for samplers without ode_solver."""

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
        return torch.zeros(coords.shape[0], coords.shape[1], 2, device=coords.device, dtype=coords.dtype)


def _make_observations(coords: torch.Tensor, fields: torch.Tensor):
    obs_indices = torch.tensor([[0, coords.shape[1] // 2]], device=coords.device)
    obs_field_ids = torch.tensor([[0, 1]], device=coords.device)
    obs_mask = torch.ones(1, 2, device=coords.device)
    obs_coords = coords[:, obs_indices[0]]
    obs_values = torch.stack(
        [
            fields[:, obs_indices[0, 0], 0],
            fields[:, obs_indices[0, 1], 1],
        ],
        dim=1,
    ).unsqueeze(-1)
    return obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids


def main() -> None:
    torch.manual_seed(7)
    device = torch.device("cpu")
    num_x, num_y, n_fields = 8, 6, 2
    dataset = SyntheticGridDataset(num_x=num_x, num_y=num_y, shuffled=True)

    grid_info = validate_regular_grid_compatibility(dataset, num_x, num_y)
    assert grid_info["requires_permutation"]
    try:
        validate_regular_grid_compatibility(SyntheticGridDataset(num_x, num_y, broken=True), num_x, num_y)
    except ValueError:
        pass
    else:
        raise AssertionError("validate_regular_grid_compatibility should fail for incomplete tensor grids")

    backbone = FNO(
        n_fields=n_fields,
        Num_x=num_x,
        Num_y=num_y,
        n_modes_x=4,
        n_modes_y=3,
        hidden_channels=8,
        n_layers=2,
        condition_blur=True,
        condition_blur_kernel=3,
        condition_blur_sigma=1.0,
    ).to(device)
    model = FNOFFM(backbone, ZeroPrior()).to(device)

    sample = dataset[0]
    coords = sample["coords"].unsqueeze(0).to(device)
    fields = sample["fields"].unsqueeze(0).to(device)
    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = _make_observations(coords, fields)

    t = torch.tensor([0.25], device=device)
    y = backbone(t, fields, coords, obs_coords, obs_values, obs_mask, obs_field_ids, obs_indices=obs_indices)
    assert y.shape == fields.shape

    euler = model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=2,
        clamp_indices=obs_indices,
        ode_solver="euler",
    )
    heun = model.sample(
        coords=coords,
        obs_coords=obs_coords,
        obs_values=obs_values,
        obs_mask=obs_mask,
        obs_field_ids=obs_field_ids,
        n_steps=2,
        clamp_indices=obs_indices,
        ode_solver="heun",
    )
    assert euler.shape == fields.shape
    assert heun.shape == fields.shape

    recon = reconstruct_snapshot(
        model=model,
        dataset=dataset,
        device=device,
        snapshot_index=0,
        cond_fields=[0, 1],
        n_obs_list=[2, 2],
        n_steps=1,
        ode_solver="euler",
    )
    assert recon["recon"].shape == fields.shape

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics = visualize_reconstruction(
            model=DummySamplerNoOde(),
            dataset=dataset,
            epoch=0,
            device=device,
            save_dir=tmpdir,
            cond_fields=[0],
            n_obs=[2],
            n_steps=1,
            ode_solver="heun",
            file_tag="dummy_no_ode",
            save_metrics_json=False,
        )
        assert "A" in metrics and "B" in metrics
        assert Path(tmpdir).exists()

    print("FNO path smoke checks passed.")


if __name__ == "__main__":
    main()
