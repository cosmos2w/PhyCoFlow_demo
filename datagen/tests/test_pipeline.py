"""Lightweight regression coverage for all six canonical data pipelines.

CPU suite:
  conda run -n phycoflow_env pytest -q datagen/tests

Optional assigned-GPU suite:
  CUDA_VISIBLE_DEVICES=1 PHYCOFLOW_TEST_CUDA=1 conda run -n phycoflow_env pytest -q datagen/tests -k cuda
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import h5py
import numpy as np
import pytest

from phycoflow_datagen.cases import CASES
from phycoflow_datagen.diagnostics import compute_diagnostics
from phycoflow_datagen.h5_pipeline import process_raw_to_h5, validate_h5
from phycoflow_datagen.plotting import create_qa_figure
from phycoflow_datagen.solvers import run_solver
from phycoflow_datagen.storage import (
    SCHEMA_VERSION,
    load_raw_trajectory,
    package_versions,
    write_raw_trajectory,
)


CASES_UNDER_TEST = (
    "burgers",
    "ks",
    "brusselator",
    "kolmogorov",
    "electro_thermal",
    "mass_transport_fluid",
)
GPU_CAPABLE_CASES = ("burgers", "ks", "brusselator", "kolmogorov")


def smoke_config(case: str, backend: str = "numpy", device: str = "cpu") -> dict:
    base = {
        "case": case,
        "dataset_id": "pytest_smoke",
        "num_trajectories": 1,
        "seed_start": 3,
        "resolution": 32 if CASES[case]["spatial_dimension"] == 1 else 16,
        "domain_length": CASES[case]["domain_length"],
        "dt": 0.01 if case != "kolmogorov" else 0.005,
        "burn_in_time": 0.01,
        "record_time": 0.03 if case != "kolmogorov" else 0.015,
        "save_every": 1,
        "backend": backend,
        "device": device,
        "solver_dtype": "float64",
        "storage_dtype": "float32",
        "contour_points": 16,
        "schema_version": SCHEMA_VERSION,
        "output_dir": "pytest-managed",
        "resume": False,
        "overwrite": False,
        "dry_run": False,
        "no_progress": True,
    }
    if case == "burgers":
        base.update(viscosity=0.01, ic_amplitude_jitter=0.08, ic_phase_jitter=0.15)
    elif case == "ks":
        base.update(
            advection_coefficient=1.0,
            second_order_coefficient=1.0,
            fourth_order_coefficient=1.0,
            noise_amplitude=0.05,
            noise_filter=0.02,
        )
    elif case == "brusselator":
        base.update(
            A=1.0,
            B=3.0,
            diffusivity_u=1.0,
            diffusivity_v=0.1,
            noise_amplitude=0.06,
            noise_filter=0.12,
        )
    elif case == "kolmogorov":
        base.update(
            reynolds_number=40.0,
            forcing_amplitude=1.0,
            forcing_wavenumber=4,
            perturbation_amplitude=0.5,
            perturbation_filter=0.025,
        )
    elif case == "electro_thermal":
        base.update(
            resolution=16,
            dt=1.0,
            burn_in_time=0.0,
            record_time=1.0,
            save_every=1,
            parameter_seed=23,
            ellipse_a_min=0.020,
            ellipse_a_max=0.030,
            ellipse_b_min=0.010,
            ellipse_b_max=0.020,
            ellipse_angle_min=0.0,
            ellipse_angle_max=2.0 * np.pi,
            sigma_silicon_min=1.0e11,
            sigma_silicon_max=3.0e11,
            kappa_alumina_min=10.0,
            kappa_alumina_max=20.0,
            absorbing_thickness=0.010,
            frequency=4.0e9,
            incident_amplitude=3.0e5,
            incident_angle=np.pi / 3.0,
            ambient_temperature=293.15,
            convective_coefficient=15.0,
            thermal_conductivity_silicon=70.0,
            permittivity_silicon=11.7,
            permittivity_alumina=1.0,
            conductivity_alumina=1.0e-7,
            coupling_tolerance=1.0e-6,
            maximum_coupling_iterations=30,
            under_relaxation=0.65,
            pml_strength=4.0,
            pml_power=3,
        )
    else:
        base.update(
            resolution=16,
            domain_length=300.0,
            domain_height=150.0,
            dt=0.5,
            burn_in_time=0.0,
            record_time=1.0,
            save_every=1,
            parameter_seed=29,
            source_amplitude_min=1.0e-3,
            source_amplitude_max=8.0e-3,
            source_x_min=-70.0,
            source_x_max=70.0,
            source_y_min=-30.0,
            source_y_max=30.0,
            source_width_min=10.0,
            source_width_max=70.0,
            rho0=1000.0,
            density_coefficient=200.0,
            dynamic_viscosity=1.0e-3,
            permeability=4.9346165e-13,
            porosity=0.1,
            diffusivity=3.56e-6,
            surface_concentration=1.0,
            gravity=9.81,
            picard_tolerance=2.0e-5,
            maximum_picard_iterations=18,
            picard_relaxation=0.65,
            advective_cfl=0.45,
            minimum_step_years=2.0e-4,
        )
    return base


@pytest.mark.parametrize("case", CASES_UNDER_TEST)
def test_solver_raw_h5_and_visualization_roundtrip(case: str, tmp_path: Path) -> None:
    config = smoke_config(case)
    result, backend = run_solver(case, config, seed=3)
    assert backend.device_description == "CPU (NumPy)"
    assert result["state"].shape[0] == result["time"].size
    assert np.isfinite(result["state"]).all()
    condition_values = result.get("condition_values")
    diagnostic_config = dict(config)
    if condition_values is not None:
        diagnostic_config.update(
            dict(zip(CASES[case]["condition_names"], condition_values))
        )
    diagnostics = compute_diagnostics(
        case, result["state"], result["time"], diagnostic_config, result=result
    )
    assert diagnostics["finite"] is True

    raw_dir = tmp_path / "raw"
    metadata = {
        "case": case,
        "dataset_id": config["dataset_id"],
        "seed": 3,
        "config": config,
        "conditions": (
            dict(zip(CASES[case]["condition_names"], condition_values))
            if condition_values is not None
            else None
        ),
        "diagnostics": diagnostics,
        "state_names": CASES[case]["state_names"],
        "field_names": CASES[case]["field_names"],
        "code_commit": "pytest",
        "package_versions": package_versions(),
    }
    npz_path, _ = write_raw_trajectory(raw_dir, 0, result, metadata)
    loaded, loaded_metadata = load_raw_trajectory(npz_path)
    np.testing.assert_array_equal(loaded["state"], result["state"])
    assert loaded_metadata["seed"] == 3

    raw_figure_path = tmp_path / f"{case}_raw.png"
    raw_figure_summary = create_qa_figure(
        case,
        npz_path,
        raw_figure_path,
        trajectory=0,
        time_index=-1,
        dpi=72,
    )
    assert raw_figure_summary["diagnostics"]["finite"] is True
    assert raw_figure_path.stat().st_size > 10_000

    h5_path = tmp_path / f"{case}.h5"
    summary = process_raw_to_h5(
        case,
        raw_dir,
        h5_path,
        split_ratios=(1.0, 0.0, 0.0),
        split_seed=1,
        compression="gzip",
        include_auxiliary=True,
        overwrite=False,
        progress_factory=None,
        command="pytest",
    )
    assert summary["case"] == case
    assert Path(summary["readme"]).is_file()
    validated = validate_h5(h5_path, expected_case=case)
    assert validated["field_names"] == CASES[case]["field_names"]
    with h5py.File(h5_path, "r") as handle:
        assert handle["fields"].shape[0] == 1
        assert json.loads(handle.attrs["grid_shape"])[-1] == config["resolution"]
        assert handle["fields"].attrs["C"] == len(CASES[case]["field_names"])
        assert json.loads(handle["conditions"].attrs["condition_names"]) == CASES[case]["condition_names"]
        assert np.isfinite(handle["statistics/train_mean"][:]).all()
        if case == "kolmogorov":
            assert "auxiliary/vorticity" in handle
        elif case == "electro_thermal":
            assert "auxiliary/ellipse_mask" in handle
            assert "auxiliary/joule_heating" in handle
        elif case == "mass_transport_fluid":
            assert "auxiliary/source_field" in handle
            assert "auxiliary/pressure" in handle

    figure_path = tmp_path / f"{case}.png"
    figure_summary = create_qa_figure(
        case,
        h5_path,
        figure_path,
        trajectory=0,
        time_index=-1,
        dpi=72,
    )
    assert figure_summary["diagnostics"]["finite"] is True
    assert figure_path.stat().st_size > 10_000


@pytest.mark.parametrize("case", GPU_CAPABLE_CASES)
def test_numpy_and_torch_cpu_agree(case: str) -> None:
    numpy_config = smoke_config(case, backend="numpy", device="cpu")
    torch_config = smoke_config(case, backend="torch", device="cpu")
    numpy_result, _ = run_solver(case, numpy_config, seed=13)
    torch_result, _ = run_solver(case, torch_config, seed=13)
    np.testing.assert_allclose(
        torch_result["state"],
        numpy_result["state"],
        rtol=3.0e-5,
        atol=3.0e-6,
    )


@pytest.mark.parametrize("case", GPU_CAPABLE_CASES)
def test_cuda_solver_smoke(case: str) -> None:
    if os.environ.get("PHYCOFLOW_TEST_CUDA") != "1":
        pytest.skip("set PHYCOFLOW_TEST_CUDA=1 and CUDA_VISIBLE_DEVICES to run CUDA tests")
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    config = smoke_config(case, backend="torch", device="cuda:0")
    result, backend = run_solver(case, config, seed=23)
    assert "cuda:0" in backend.device_description
    assert np.isfinite(result["state"]).all()
