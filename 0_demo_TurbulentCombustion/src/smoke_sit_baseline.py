"""
Lightweight smoke checks for the unified SiT baseline path.

Run from the demo root:
    python src/smoke_sit_baseline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

from helpers_baseline import validate_regular_grid_compatibility
from model_baseline import (
    SiTAdapter,
    SiTPhysics,
    load_yaml,
    validate_and_normalize_config,
)
import evaluate_Gen_Baseline


class SyntheticGridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_x: int,
        num_y: int,
        num_fields: int = 5,
        shuffled: bool = False,
    ) -> None:
        x = torch.linspace(0.0, 1.0, num_x)
        y = torch.linspace(0.0, 1.0, num_y)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        coords = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)
        if shuffled:
            coords = coords[torch.randperm(coords.shape[0])]

        self.coords = coords
        self.coords_raw = coords.clone()
        self.num_points = int(coords.shape[0])
        self.num_fields = int(num_fields)
        self.field_names = tuple(f"field_{idx}" for idx in range(num_fields))
        self.mean = torch.zeros(num_fields)
        self.std = torch.ones(num_fields)
        self.fields = torch.zeros(self.num_points, num_fields)

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "coords": self.coords,
            "coords_raw": self.coords_raw,
            "fields": self.fields,
            "time_index": torch.tensor(idx),
            "physical_time": torch.tensor(float(idx)),
        }


def _check_grid_validation() -> None:
    num_x, num_y = 8, 6
    row_major = SyntheticGridDataset(num_x, num_y)
    info = validate_regular_grid_compatibility(row_major, num_x, num_y)
    assert info["row_major"] is True
    assert info["requires_permutation"] is False

    shuffled = SyntheticGridDataset(num_x, num_y, shuffled=True)
    info = validate_regular_grid_compatibility(shuffled, num_x, num_y)
    assert info["row_major"] is False
    assert info["requires_permutation"] is True
    assert info["grid_order"].numel() == num_x * num_y

    broken = SyntheticGridDataset(num_x, num_y)
    broken.coords[-1] = broken.coords[0]
    try:
        validate_regular_grid_compatibility(broken, num_x, num_y)
    except ValueError as exc:
        assert "complete tensor-product grid" in str(exc)
    else:
        raise AssertionError("validate_regular_grid_compatibility should fail for duplicate grid cells")


def _check_sit_forward() -> None:
    torch.manual_seed(7)
    model = SiTPhysics(
        input_size_h=10,
        input_size_w=15,
        patch_size=5,
        in_channels=5,
        cond_channels=11,
        hidden_size=64,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
        tokenizer="patch",
    )
    x = torch.randn(2, 5, 10, 15)
    t = torch.rand(2)
    obs_value_grid = torch.zeros(2, 5, 10, 15)
    obs_mask_grid = torch.zeros(2, 5, 10, 15)
    obs_value_grid[:, 2, 0, 0] = 1.0
    obs_mask_grid[:, 2, 0, 0] = 1.0
    out = model(x, t, obs_value_grid=obs_value_grid, obs_mask_grid=obs_mask_grid)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def _check_adapter_build_from_launch_config() -> None:
    demo_root = Path(__file__).resolve().parents[1]
    cfg_path = demo_root / "Save_config" / "config_baseline_Gen.yaml"
    cfg = validate_and_normalize_config(load_yaml(cfg_path))
    num_x = int(cfg["shared"]["data"]["num_x"])
    num_y = int(cfg["shared"]["data"]["num_y"])
    dataset = SyntheticGridDataset(num_x, num_y, num_fields=5)

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = SiTAdapter().build_for_training(
            cfg=cfg,
            device=torch.device("cpu"),
            run_dir=Path(tmpdir),
            train_set=dataset,
            val_set=dataset,
        )
    assert bundle.baseline_model == "sit"
    assert bundle.components["tokenizer"] == "patch"
    assert bundle.components["cond_mode"] == "interp"


def _check_evaluator_cli_overrides() -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "evaluate_Gen_Baseline.py",
            "--n-steps",
            "8",
            "--ode-solver",
            "heun",
            "--vis-cond-fields",
            "2",
            "3",
            "--vis-n-obs-list",
            "256",
            "256",
        ]
        args = evaluate_Gen_Baseline.parse_args()
    finally:
        sys.argv = old_argv
    assert args.n_steps == 8
    assert args.ode_solver == "heun"
    assert args.vis_cond_fields == [2, 3]
    assert args.vis_n_obs_list == [256, 256]


def main() -> None:
    _check_grid_validation()
    _check_sit_forward()
    _check_adapter_build_from_launch_config()
    _check_evaluator_cli_overrides()
    print("SiT baseline smoke checks passed.")


if __name__ == "__main__":
    main()
