"""Command-line entry points used by the per-case wrapper scripts."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .cases import CASES, canonical_case_name
from .diagnostics import compute_diagnostics
from .h5_pipeline import process_raw_to_h5
from .plotting import create_qa_figure
from .solvers import run_solver
from .storage import (
    SCHEMA_VERSION,
    atomic_write_json,
    git_commit,
    package_versions,
    raw_trajectory_is_complete,
    trajectory_paths,
    write_raw_trajectory,
)


DATAGEN_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = DATAGEN_ROOT.parent


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _add_physical_arguments(parser: argparse.ArgumentParser, case: str) -> None:
    if case == "burgers":
        parser.add_argument("--viscosity", type=_positive_float, default=1.0e-2, help="Kinematic viscosity nu (notebook default: 0.01).")
        parser.add_argument("--ic-amplitude-jitter", type=_nonnegative_float, default=0.08, help="Seeded relative standard deviation applied to the three Fourier-mode amplitudes; set 0 for notebook amplitudes.")
        parser.add_argument("--ic-phase-jitter", type=_nonnegative_float, default=0.15, help="Seeded phase perturbation standard deviation in radians; set 0 for notebook phases.")
    elif case == "ks":
        parser.add_argument("--advection-coefficient", type=float, default=1.0, help="Coefficient multiplying u*u_x in the KS equation.")
        parser.add_argument("--second-order-coefficient", type=_positive_float, default=1.0, help="Destabilizing coefficient multiplying u_xx in the canonical left-hand-side equation.")
        parser.add_argument("--fourth-order-coefficient", type=_positive_float, default=1.0, help="Stabilizing coefficient multiplying u_xxxx in the canonical left-hand-side equation.")
        parser.add_argument("--noise-amplitude", type=_positive_float, default=0.05, help="Amplitude of the smooth zero-mean initial perturbation.")
        parser.add_argument("--noise-filter", type=_positive_float, default=0.02, help="Coefficient in the initial spectral filter exp(-coefficient*k^4).")
    elif case == "brusselator":
        parser.add_argument("--A", type=_positive_float, default=1.0, help="Brusselator feed parameter A.")
        parser.add_argument("--B", type=_positive_float, default=3.0, help="Brusselator feed parameter B.")
        parser.add_argument("--diffusivity-u", dest="diffusivity_u", type=_positive_float, default=1.0, help="Diffusion coefficient D_u.")
        parser.add_argument("--diffusivity-v", dest="diffusivity_v", type=_positive_float, default=0.1, help="Diffusion coefficient D_v.")
        parser.add_argument("--noise-amplitude", type=_positive_float, default=0.06, help="Standard deviation of each smooth initial concentration perturbation.")
        parser.add_argument("--noise-filter", type=_positive_float, default=0.12, help="Coefficient in the initial spectral filter exp(-coefficient*|k|^4).")
    elif case == "kolmogorov":
        parser.add_argument("--reynolds-number", type=_positive_float, default=40.0, help="Reynolds number Re; viscosity is exactly 1/Re.")
        parser.add_argument("--forcing-amplitude", type=float, default=1.0, help="Amplitude of sin(n*y) e_x body forcing.")
        parser.add_argument("--forcing-wavenumber", type=_positive_int, default=4, help="Integer periodic forcing wavenumber n.")
        parser.add_argument("--perturbation-amplitude", type=_positive_float, default=0.5, help="Standard deviation of the smooth perturbation added to laminar vorticity.")
        parser.add_argument("--perturbation-filter", type=_positive_float, default=0.025, help="Coefficient in the initial spectral filter exp(-coefficient*|k|^4).")
    elif case == "electro_thermal":
        parser.add_argument("--parameter-seed", type=int, default=23, help="Scrambled Sobol seed used for the deterministic five-parameter design.")
        parser.add_argument("--ellipse-a-min", type=_positive_float, default=0.020, help="Minimum silicon ellipse semi-axis a in metres.")
        parser.add_argument("--ellipse-a-max", type=_positive_float, default=0.030, help="Maximum silicon ellipse semi-axis a in metres.")
        parser.add_argument("--ellipse-b-min", type=_positive_float, default=0.010, help="Minimum silicon ellipse semi-axis b in metres.")
        parser.add_argument("--ellipse-b-max", type=_positive_float, default=0.020, help="Maximum silicon ellipse semi-axis b in metres.")
        parser.add_argument("--ellipse-angle-min", type=float, default=0.0, help="Minimum ellipse rotation angle in radians.")
        parser.add_argument("--ellipse-angle-max", type=float, default=2.0 * np.pi, help="Maximum ellipse rotation angle in radians.")
        parser.add_argument("--sigma-silicon-min", type=_positive_float, default=1.0e11, help="Minimum silicon conductivity prefactor Sigma_Si in S/m.")
        parser.add_argument("--sigma-silicon-max", type=_positive_float, default=3.0e11, help="Maximum silicon conductivity prefactor Sigma_Si in S/m.")
        parser.add_argument("--kappa-alumina-min", type=_positive_float, default=10.0, help="Minimum alumina thermal conductivity in W/(m K).")
        parser.add_argument("--kappa-alumina-max", type=_positive_float, default=20.0, help="Maximum alumina thermal conductivity in W/(m K).")
        parser.add_argument("--absorbing-thickness", type=_positive_float, default=0.010, help="Complex-stretch absorbing-layer thickness in metres.")
        parser.add_argument("--frequency", type=_positive_float, default=4.0e9, help="Incident electric-field frequency in Hz.")
        parser.add_argument("--incident-amplitude", type=_positive_float, default=3.0e5, help="Incident E_z amplitude in V/m.")
        parser.add_argument("--incident-angle", type=float, default=float(np.pi / 3.0), help="Incident-wave angle in radians.")
        parser.add_argument("--ambient-temperature", type=_positive_float, default=293.15, help="External temperature in K.")
        parser.add_argument("--convective-coefficient", type=_positive_float, default=15.0, help="Thermal Robin coefficient in W/(m^2 K).")
        parser.add_argument("--thermal-conductivity-silicon", type=_positive_float, default=70.0, help="Silicon thermal conductivity in W/(m K).")
        parser.add_argument("--permittivity-silicon", type=_positive_float, default=11.7, help="Silicon relative permittivity.")
        parser.add_argument("--permittivity-alumina", type=_positive_float, default=1.0, help="Alumina relative permittivity.")
        parser.add_argument("--conductivity-alumina", type=_positive_float, default=1.0e-7, help="Alumina electrical conductivity in S/m.")
        parser.add_argument("--coupling-tolerance", type=_positive_float, default=1.0e-6, help="Relative infinity-norm tolerance for both T and sigma Picard updates.")
        parser.add_argument("--maximum-coupling-iterations", type=_positive_int, default=30, help="Hard limit for bidirectional electro-thermal Picard iterations.")
        parser.add_argument("--under-relaxation", type=_positive_float, default=0.65, help="Picard temperature relaxation in (0,1].")
        parser.add_argument("--pml-strength", type=_positive_float, default=4.0, help="Polynomial complex-coordinate stretch strength.")
        parser.add_argument("--pml-power", type=_positive_int, default=3, help="Polynomial power of the complex-coordinate stretch.")
    else:
        parser.add_argument("--parameter-seed", type=int, default=29, help="Scrambled Sobol seed used for the deterministic source design.")
        parser.add_argument("--source-amplitude-min", type=_positive_float, default=1.0e-3, help="Minimum Gaussian source amplitude A.")
        parser.add_argument("--source-amplitude-max", type=_positive_float, default=8.0e-3, help="Maximum Gaussian source amplitude A.")
        parser.add_argument("--source-x-min", type=float, default=-70.0, help="Minimum Gaussian source centre x0 in metres.")
        parser.add_argument("--source-x-max", type=float, default=70.0, help="Maximum Gaussian source centre x0 in metres.")
        parser.add_argument("--source-y-min", type=float, default=-30.0, help="Minimum Gaussian source centre y0 in metres.")
        parser.add_argument("--source-y-max", type=float, default=30.0, help="Maximum Gaussian source centre y0 in metres.")
        parser.add_argument("--source-width-min", type=_positive_float, default=10.0, help="Minimum Gaussian source width s in metres.")
        parser.add_argument("--source-width-max", type=_positive_float, default=70.0, help="Maximum Gaussian source width s in metres.")
        parser.add_argument("--domain-height", type=_positive_float, default=150.0, help="Physical y-extent of the centered rectangle in metres.")
        parser.add_argument("--rho0", type=_positive_float, default=1000.0, help="Pure-water density in kg/m^3.")
        parser.add_argument("--density-coefficient", type=_positive_float, default=200.0, help="Linear concentration-density coefficient beta.")
        parser.add_argument("--dynamic-viscosity", type=_positive_float, default=1.0e-3, help="Dynamic viscosity in Pa s.")
        parser.add_argument("--permeability", type=_positive_float, default=4.9346165e-13, help="Porous-medium permeability in m^2 (500 mD canonical).")
        parser.add_argument("--porosity", type=_positive_float, default=0.1, help="Fluid fraction epsilon.")
        parser.add_argument("--diffusivity", type=_positive_float, default=3.56e-6, help="Effective molecular diffusion D_L in m^2/s.")
        parser.add_argument("--surface-concentration", type=_positive_float, default=1.0, help="Top-right Dirichlet concentration in mol/m^3.")
        parser.add_argument("--gravity", type=_positive_float, default=9.81, help="Magnitude of downward gravitational acceleration in m/s^2.")
        parser.add_argument("--picard-tolerance", type=_positive_float, default=2.0e-5, help="Relative nonlinear concentration/density coupling tolerance.")
        parser.add_argument("--maximum-picard-iterations", type=_positive_int, default=18, help="Maximum pressure/transport Picard iterations per internal step.")
        parser.add_argument("--picard-relaxation", type=_positive_float, default=0.65, help="Nonlinear concentration relaxation in (0,1].")
        parser.add_argument("--advective-cfl", type=_positive_float, default=0.45, help="Maximum accepted pore-velocity Courant number.")
        parser.add_argument("--minimum-step-years", type=_positive_float, default=2.0e-4, help="Failure threshold for adaptive internal time-step reduction in years.")


def _generation_parser(case: str) -> argparse.ArgumentParser:
    defaults = CASES[case]
    parser = argparse.ArgumentParser(
        description=f"Generate raw trajectories for {defaults['display_name']}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Raw dataset directory. Default: datagen/data/raw/<case>/<dataset-id>.")
    parser.add_argument("--dataset-id", default=f"{case}_canonical", help="Stable dataset variant identifier stored in all metadata.")
    parser.add_argument("--num-trajectories", type=_positive_int, default=10, help="Number of independently seeded trajectories.")
    parser.add_argument("--seed-start", type=int, default=0, help="Seed for trajectory 0; subsequent trajectories use consecutive seeds.")
    parser.add_argument("--resolution", type=_positive_int, default=defaults["resolution"], help="Grid points per spatial axis; nonperiodic multiphysics grids include their physical boundaries.")
    parser.add_argument("--domain-length", type=_positive_float, default=defaults["domain_length"], help="Physical x-extent (and y-extent for square periodic cases).")
    parser.add_argument("--dt", type=_positive_float, default=defaults["dt"], help="Numerical time step; for mass transport this is the maximum adaptive internal step in years and is unused by the steady electro-thermal solve.")
    parser.add_argument("--burn-in-time", type=_nonnegative_float, default=defaults["burn_in_time"], help="Transient time integrated before the first saved frame.")
    parser.add_argument("--record-time", type=_positive_float, default=defaults["record_time"], help="Physical duration retained after burn-in.")
    parser.add_argument("--save-every", type=_positive_int, default=defaults["save_every"], help="Save one frame every this many nominal solver steps; the final frame is always saved.")
    parser.add_argument("--backend", choices=("numpy", "torch"), default=defaults.get("backend", "torch"), help="Array/FFT implementation. The sparse SciPy multiphysics solvers require NumPy/CPU.")
    parser.add_argument("--device", default=defaults.get("device", "cuda:0"), help="Logical compute device; use cpu for the sparse SciPy multiphysics solvers.")
    parser.add_argument("--solver-dtype", choices=("float32", "float64"), default="float64", help="Numerical precision used during time integration.")
    parser.add_argument("--storage-dtype", choices=("float32", "float64"), default="float32", help="Precision of saved raw snapshots.")
    parser.add_argument("--contour-points", type=_positive_int, default=16, help="Complex-contour quadrature points for ETDRK4 coefficients (unused by Brusselator).")
    parser.add_argument("--resume", action="store_true", help="Skip existing trajectories only when checksum and required arrays validate.")
    parser.add_argument("--overwrite", action="store_true", help="Intentionally replace trajectories with matching IDs. Mutually exclusive with --resume.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration, storage estimate, and device request without creating files.")
    parser.add_argument("--no-progress", action="store_true", help="Disable both trajectory and numerical-iteration progress bars (useful for logs/tests).")
    _add_physical_arguments(parser, case)
    return parser


def _estimate_raw_bytes(case: str, config: dict[str, Any]) -> int:
    n = config["resolution"]
    if case == "electro_thermal":
        frames = 1
    else:
        steps = int(round(config["record_time"] / config["dt"]))
        frames = 1 + int(np.ceil(steps / config["save_every"]))
    channels = len(CASES[case]["state_names"])
    spatial_points = n if CASES[case]["spatial_dimension"] == 1 else n * n
    itemsize = np.dtype(config["storage_dtype"]).itemsize
    auxiliary_factor = 1.15
    if case == "electro_thermal":
        auxiliary_factor = 2.25  # ellipse, sigma, heating, and material fields
    elif case == "mass_transport_fluid":
        auxiliary_factor = 2.10  # pressure, source, and diagnostic face velocities
    return int(
        config["num_trajectories"]
        * frames
        * channels
        * spatial_points
        * itemsize
        * auxiliary_factor
    )


def _validate_case_arguments(parser: argparse.ArgumentParser, args: argparse.Namespace, case: str) -> None:
    """Reject inconsistent physical ranges before any output directory is created."""

    if case in {"electro_thermal", "mass_transport_fluid"}:
        if args.backend != "numpy" or args.device != "cpu":
            parser.error(
                f"{case} uses SciPy sparse direct solves and currently requires "
                "--backend numpy --device cpu; CUDA is not used by this solver"
            )
        if args.burn_in_time != 0.0:
            parser.error(f"{case} requires --burn-in-time 0")
        if args.seed_start < 0:
            parser.error(f"{case} requires a nonnegative --seed-start for Sobol indexing")
    if case == "electro_thermal":
        ranges = (
            ("ellipse a", args.ellipse_a_min, args.ellipse_a_max),
            ("ellipse b", args.ellipse_b_min, args.ellipse_b_max),
            ("ellipse angle", args.ellipse_angle_min, args.ellipse_angle_max),
            ("Sigma_Si", args.sigma_silicon_min, args.sigma_silicon_max),
            ("kappa_alumina", args.kappa_alumina_min, args.kappa_alumina_max),
        )
        if not 0.0 < args.under_relaxation <= 1.0:
            parser.error("--under-relaxation must be in (0, 1]")
    elif case == "mass_transport_fluid":
        ranges = (
            ("source amplitude", args.source_amplitude_min, args.source_amplitude_max),
            ("source x", args.source_x_min, args.source_x_max),
            ("source y", args.source_y_min, args.source_y_max),
            ("source width", args.source_width_min, args.source_width_max),
        )
        if not 0.0 < args.picard_relaxation <= 1.0:
            parser.error("--picard-relaxation must be in (0, 1]")
        output_interval = args.dt * args.save_every
        output_count = args.record_time / output_interval
        if not np.isclose(output_count, round(output_count), rtol=0.0, atol=1.0e-10):
            parser.error(
                "mass-transport --record-time must be an integer multiple of "
                "--dt * --save-every so output times remain exact"
            )
    else:
        ranges = ()
    for label, lower, upper in ranges:
        if not lower < upper:
            parser.error(f"{label} minimum must be strictly smaller than its maximum")


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path.resolve(strict=False)
    while not candidate.exists():
        candidate = candidate.parent
    return candidate


def _simulation_config(args: argparse.Namespace, case: str) -> dict[str, Any]:
    config = vars(args).copy()
    config["case"] = case
    config["output_dir"] = str(args.output_dir)
    config["schema_version"] = SCHEMA_VERSION
    return config


def _comparable_simulation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove only launch controls that may safely change on resume."""

    ignored = {
        "resume",
        "overwrite",
        "dry_run",
        "no_progress",
        "num_trajectories",
        "output_dir",
    }
    return {key: value for key, value in config.items() if key not in ignored}


def generate_main(case_name: str, argv: list[str] | None = None) -> None:
    case = canonical_case_name(case_name)
    parser = _generation_parser(case)
    args = parser.parse_args(argv)
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.backend == "numpy" and args.device != "cpu":
        parser.error("use --device cpu with --backend numpy")
    _validate_case_arguments(parser, args, case)
    if args.output_dir is None:
        args.output_dir = DATAGEN_ROOT / "data" / "raw" / case / args.dataset_id
    args.output_dir = args.output_dir.resolve(strict=False)
    config = _simulation_config(args, case)
    estimated_bytes = _estimate_raw_bytes(case, config)
    disk = shutil.disk_usage(_nearest_existing_parent(args.output_dir))
    print(json.dumps(config, indent=2, sort_keys=True))
    print(
        f"Estimated raw snapshot storage: {estimated_bytes / 2**30:.3f} GiB; "
        f"available on target filesystem: {disk.free / 2**30:.1f} GiB"
    )
    if estimated_bytes > 0.85 * disk.free:
        parser.error("estimated output exceeds the 85% free-space safety threshold")
    if args.dry_run:
        print("Dry run complete; no directory or data file was created.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = args.output_dir / "resolved_config.json"
    if resolved_path.exists() and not (args.resume or args.overwrite):
        parser.error(f"{resolved_path} already exists; choose --resume or --overwrite intentionally")
    if resolved_path.exists():
        try:
            existing_config = json.loads(resolved_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            parser.error(f"cannot read existing resolved configuration: {error}")
        existing_comparable = _comparable_simulation_config(existing_config)
        requested_comparable = _comparable_simulation_config(config)
        if existing_comparable != requested_comparable:
            changed = sorted(
                key
                for key in set(existing_comparable) | set(requested_comparable)
                if existing_comparable.get(key) != requested_comparable.get(key)
            )
            parser.error(
                "refusing to mix incompatible simulations in one raw directory; "
                f"changed keys: {changed}. Use a new --output-dir and --dataset-id."
            )
    atomic_write_json(resolved_path, config)
    command = shlex.join([sys.executable, *sys.argv])
    commit = git_commit(REPOSITORY_ROOT)
    versions = package_versions()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case": case,
        "dataset_id": args.dataset_id,
        "resolved_config": "resolved_config.json",
        "trajectory_count_requested": args.num_trajectories,
        "completed_trajectories": [],
        "generation_command": command,
        "code_commit": commit,
        "package_versions": versions,
    }
    manifest_path = args.output_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completed = set(int(value) for value in manifest.get("completed_trajectories", []))

    outer_values = range(args.num_trajectories)
    outer = outer_values if args.no_progress else tqdm(
        outer_values, total=args.num_trajectories, desc="Overall trajectories", unit="trajectory", position=0
    )

    def inner_progress(iterable, **kwargs):
        return tqdm(iterable, leave=False, position=1, unit="step", dynamic_ncols=True, **kwargs)

    generated = 0
    skipped = 0
    for trajectory_id in outer:
        if args.resume and raw_trajectory_is_complete(args.output_dir, trajectory_id):
            completed.add(trajectory_id)
            skipped += 1
            if not args.no_progress:
                outer.set_postfix_str(f"skipped validated {trajectory_id:06d}")
            continue
        npz_path, json_path = trajectory_paths(args.output_dir, trajectory_id)
        if args.resume and (npz_path.exists() or json_path.exists()):
            raise RuntimeError(
                f"trajectory {trajectory_id} exists but failed validation; inspect it or rerun with --overwrite"
            )
        seed = args.seed_start + trajectory_id
        result, backend = run_solver(
            case,
            config,
            seed,
            None if args.no_progress else inner_progress,
        )
        condition_values = result.pop("condition_values", None)
        diagnostic_config = dict(config)
        conditions = None
        if condition_values is not None:
            condition_values = np.asarray(condition_values, dtype=np.float64)
            expected_conditions = CASES[case]["condition_names"]
            if condition_values.shape != (len(expected_conditions),):
                raise ValueError(
                    f"solver returned condition_values {condition_values.shape}; "
                    f"expected {(len(expected_conditions),)}"
                )
            conditions = {
                name: float(value)
                for name, value in zip(expected_conditions, condition_values)
            }
            diagnostic_config.update(conditions)
        diagnostics = compute_diagnostics(
            case,
            result["state"],
            result["time"],
            diagnostic_config,
            result=result,
        )
        metadata = {
            "case": case,
            "display_name": CASES[case]["display_name"],
            "equation": CASES[case]["equation"],
            "dataset_id": args.dataset_id,
            "seed": seed,
            "state_names": CASES[case]["state_names"],
            "field_names": CASES[case]["field_names"],
            "config": config,
            "conditions": conditions,
            "backend_description": backend.device_description,
            "code_commit": commit,
            "package_versions": versions,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "generation_command": command,
            "diagnostics": diagnostics,
        }
        write_raw_trajectory(
            args.output_dir,
            trajectory_id,
            result,
            metadata,
            overwrite=args.overwrite,
        )
        completed.add(trajectory_id)
        generated += 1
        manifest["completed_trajectories"] = sorted(completed)
        manifest["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(manifest_path, manifest)
        if not args.no_progress:
            outer.set_postfix_str(f"saved {trajectory_id:06d}")
    if hasattr(outer, "close"):
        outer.close()
    print(
        f"Generation complete: generated={generated}, skipped={skipped}, "
        f"validated total={len(completed)}, raw directory={args.output_dir}"
    )


def postprocess_main(case_name: str, argv: list[str] | None = None) -> None:
    case = canonical_case_name(case_name)
    parser = argparse.ArgumentParser(
        description=f"Convert raw {CASES[case]['display_name']} trajectories to unified HDF5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, required=True, help="Raw dataset directory containing trajectories/ and manifest.json.")
    parser.add_argument("--output", type=Path, required=True, help="Destination .h5 file; a dataset-specific README is written beside it.")
    parser.add_argument("--split-ratios", nargs=3, type=float, metavar=("TRAIN", "VAL", "TEST"), default=(0.8, 0.1, 0.1), help="Trajectory-level train/validation/test ratios.")
    parser.add_argument("--split-seed", type=int, default=2026, help="Seed controlling the deterministic trajectory split.")
    parser.add_argument("--compression", choices=("gzip", "lzf", "none"), default="gzip", help="HDF5 field compression.")
    parser.add_argument("--no-auxiliary", action="store_true", help="Do not retain case-specific fields such as vorticity, material masks, Joule heating, source fields, or pressure.")
    parser.add_argument("--overwrite", action="store_true", help="Intentionally replace an existing HDF5 file.")
    parser.add_argument("--no-progress", action="store_true", help="Disable the post-processing trajectory progress bar.")
    args = parser.parse_args(argv)
    if args.output.suffix not in {".h5", ".hdf5"}:
        parser.error("--output must end in .h5 or .hdf5")
    progress_factory = None if args.no_progress else lambda iterable, **kwargs: tqdm(
        iterable, unit="trajectory", dynamic_ncols=True, **kwargs
    )
    command = shlex.join([sys.executable, *sys.argv])
    summary = process_raw_to_h5(
        case,
        args.raw_dir.resolve(),
        args.output.resolve(strict=False),
        split_ratios=tuple(args.split_ratios),
        split_seed=args.split_seed,
        compression=args.compression,
        include_auxiliary=not args.no_auxiliary,
        overwrite=args.overwrite,
        progress_factory=progress_factory,
        command=command,
    )
    print("Processed dataset summary:")
    print(json.dumps(summary, indent=2, sort_keys=True))


def visualize_main(case_name: str, argv: list[str] | None = None) -> None:
    case = canonical_case_name(case_name)
    parser = argparse.ArgumentParser(
        description=f"Create a physical-coherence QA figure for {CASES[case]['display_name']}.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw trajectory .npz or processed dataset .h5.")
    parser.add_argument("--output", type=Path, required=True, help="Output figure ending in .png, .pdf, or .svg.")
    parser.add_argument("--trajectory", type=int, default=0, help="Processed-HDF5 trajectory index (ignored for a raw NPZ).")
    parser.add_argument("--time-index", type=int, default=-1, help="Saved time index; negative values count backward from the final frame.")
    parser.add_argument("--dpi", type=_positive_int, default=200, help="Raster resolution for PNG output.")
    args = parser.parse_args(argv)
    summary = create_qa_figure(
        case,
        args.input.resolve(),
        args.output.resolve(strict=False),
        trajectory=args.trajectory,
        time_index=args.time_index,
        dpi=args.dpi,
    )
    print("Visualization summary:")
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=True))
