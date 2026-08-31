"""V4 high-resolution, throughput-only scale-stress benchmark.

This runner is deliberately separate from ``benchmark_validation_v3.py``.  V3
measures the native 40,300-point problem; V4 measures only the query-evaluable
architectures on a shared, deterministic dummy-coordinate sequence above the
native grid.  No accuracy value is produced for the stress region and no
fixed-grid model is converted into a query model.

The formal runner is intentionally conservative:

* the query counts through 4M and the global cap are declared before loading a
  model;
* every method at a given N receives exactly the same float32 coordinates;
* sensor coordinates are a prefix of the sequence so hard clamping remains
  well-defined for point-cloud FFM models;
* geometry preparation is measured and reported separately from warm model-core
  inference;
* a failed count terminates that method's curve and is retained as a row;
* the runner requires an explicit clean CUDA device and never falls back to
  CPU for a formal run.

Use ``--audit-only`` or ``--dry-run`` for CPU-side planning.  A long GPU run is
never started by either mode.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = REPO_ROOT / "0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels/_Scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

# Runtime model imports are deliberately lazy.  Importing the canonical
# PointCloudFFM stack can JIT-compile KeOps; ``--audit-only`` and ``--dry-run``
# must remain CPU-side planning operations.  V3 files/results are never
# modified by this module; its support audit and stress protocol are V4-only.
from common.config import load_config, stable_seed

METHODS = (
    "DMF-Gen",
    "FFM-FNO",
    "FFM-Perceiver",
    "Latent FM",
    "SiT",
    "MLP-RBF",
    "Geo-FNO",
    "Senseiver",
)

_RUNTIME_IMPORTS_READY = False


def _ensure_runtime_imports() -> None:
    """Load model/timing helpers only for a formal CUDA run."""

    global _RUNTIME_IMPORTS_READY
    global _plan_tensors, load_model, core_call, evaluation_context
    global gpu_index, gpu_state, assert_clean_gpu, v3_environment, method_settings
    global time_cuda, verify_identities, load_sensor_rows
    if _RUNTIME_IMPORTS_READY:
        return
    from benchmark_validation_v3 import (
        assert_clean_gpu as assert_clean_gpu_fn,
    )
    from benchmark_validation_v3 import (
        core_call as core_call_fn,
    )
    from benchmark_validation_v3 import (
        environment as environment_fn,
    )
    from benchmark_validation_v3 import (
        evaluation_context as evaluation_context_fn,
    )
    from benchmark_validation_v3 import (
        gpu_index as gpu_index_fn,
    )
    from benchmark_validation_v3 import (
        gpu_state as gpu_state_fn,
    )
    from benchmark_validation_v3 import (
        load_sensor_rows as load_sensor_rows_fn,
    )
    from benchmark_validation_v3 import (
        method_settings as method_settings_fn,
    )
    from benchmark_validation_v3 import (
        time_cuda as time_cuda_fn,
    )
    from benchmark_validation_v3 import (
        verify_identities as verify_identities_fn,
    )
    from common.model_loader import _plan_tensors as plan_tensors
    from common.model_loader import load_model as load_model_fn
    _plan_tensors = plan_tensors
    load_model = load_model_fn
    core_call = core_call_fn
    evaluation_context = evaluation_context_fn
    gpu_index = gpu_index_fn
    gpu_state = gpu_state_fn
    assert_clean_gpu = assert_clean_gpu_fn
    v3_environment = environment_fn
    method_settings = method_settings_fn
    time_cuda = time_cuda_fn
    verify_identities = verify_identities_fn
    load_sensor_rows = load_sensor_rows_fn
    _RUNTIME_IMPORTS_READY = True


# These are part of the V4 contract.  They are not inferred from a result.
PREDECLARED_QUERY_COUNTS: tuple[int, ...] = (
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_000_000,
    4_000_000,
)
DEFAULT_GLOBAL_QUERY_CAP = 8_000_000
DEFAULT_MEMORY_FRACTION = 0.90
DEFAULT_RUNTIME_CAP_SECONDS = 60.0
DEFAULT_BOUNDARY_POLICY = "predeclared_counts_then_doubling_to_global_cap"
DEFAULT_QUERY_SEED = 20260830
NATIVE_QUERY_COUNT = 40_300
SENSOR_COUNT = 256


@dataclass(frozen=True)
class DummyQuerySpec:
    """Immutable specification for the shared V4 query sequence."""

    version: str = "figure5-v4-dummy-sobol-v1"
    generator: str = "torch.quasirandom.SobolEngine"
    scramble: bool = True
    seed: int = DEFAULT_QUERY_SEED
    dtype: str = "float32"
    dimension: int = 3
    low: tuple[float, ...] = (0.0, 0.0, 0.0)
    high: tuple[float, ...] = (1.0, 1.0, 0.0)
    include_sensor_prefix: bool = True
    sequence_policy: str = "exact_sensor_prefix_then_sobol_suffix"

    def __post_init__(self) -> None:
        if self.generator != "torch.quasirandom.SobolEngine":
            raise ValueError(f"Unsupported V4 dummy generator: {self.generator!r}")
        if self.dtype != "float32":
            raise ValueError("V4 dummy coordinates must be float32")
        if int(self.dimension) < 1:
            raise ValueError("Dummy query dimension must be positive")
        if len(self.low) != int(self.dimension) or len(self.high) != int(self.dimension):
            raise ValueError("Dummy query bounds must match the coordinate dimension")
        low = np.asarray(self.low, dtype=np.float64)
        high = np.asarray(self.high, dtype=np.float64)
        if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)):
            raise ValueError("Dummy query bounds must be finite")
        if np.any(high < low):
            raise ValueError("Dummy query upper bounds must be >= lower bounds")

    def payload(self, *, sensor_coords_sha256: str | None = None) -> dict[str, Any]:
        result = asdict(self)
        result["low"] = [float(value) for value in self.low]
        result["high"] = [float(value) for value in self.high]
        result["sensor_coords_sha256"] = sensor_coords_sha256
        return result


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256(path: Path) -> str:
    """Hash a provenance file without importing the V3 runner."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Small atomic CSV writer used by both audit-only and formal modes."""

    import csv

    if not rows:
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(str(key))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def hash_coordinates(coords: np.ndarray | torch.Tensor) -> str:
    """Hash the exact contiguous little-endian float32 coordinate bytes."""

    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    array = np.asarray(coords, dtype=np.float32)
    if not array.flags.c_contiguous:
        array = np.ascontiguousarray(array)
    # The dtype marker avoids silent cross-dtype hash collisions in manifests.
    return hash_bytes(b"float32|" + array.tobytes(order="C"))


def spec_hash(spec: DummyQuerySpec, *, sensor_coords_sha256: str | None = None) -> str:
    return hash_bytes(_json_bytes(spec.payload(sensor_coords_sha256=sensor_coords_sha256)))


def query_hash(
    coords: np.ndarray,
    spec: DummyQuerySpec,
    *,
    sensor_coords_sha256: str | None = None,
) -> str:
    """Hash both the coordinate bytes and the declared generator contract."""

    return hash_bytes(
        _json_bytes(spec.payload(sensor_coords_sha256=sensor_coords_sha256))
        + b"|coordinates|"
        + np.ascontiguousarray(coords, dtype=np.float32).tobytes(order="C")
    )


def bounds_from_normalized_coords(coords: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized-domain bounds from the canonical dataset mesh."""

    if torch.is_tensor(coords):
        coords = coords.detach().cpu().numpy()
    array = np.asarray(coords, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected [N,D] coordinates, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Canonical coordinates contain non-finite values")
    low = array.min(axis=0).astype(np.float64)
    high = array.max(axis=0).astype(np.float64)
    return low, high


def make_dummy_query_spec(
    coords: np.ndarray | torch.Tensor,
    *,
    seed: int = DEFAULT_QUERY_SEED,
) -> DummyQuerySpec:
    """Build a frozen spec from the canonical normalized combustion mesh."""

    low, high = bounds_from_normalized_coords(coords)
    return DummyQuerySpec(
        seed=int(seed),
        dimension=int(low.size),
        low=tuple(float(value) for value in low),
        high=tuple(float(value) for value in high),
    )


def _validate_sensor_coords(sensor_coords: np.ndarray, spec: DummyQuerySpec) -> np.ndarray:
    sensors = np.asarray(sensor_coords, dtype=np.float32)
    if sensors.ndim != 2 or sensors.shape[1] != spec.dimension:
        raise ValueError(
            f"sensor_coords must have shape [M,{spec.dimension}], got {sensors.shape}"
        )
    if not np.all(np.isfinite(sensors)):
        raise ValueError("Sensor coordinates contain non-finite values")
    low = np.asarray(spec.low, dtype=np.float32)
    high = np.asarray(spec.high, dtype=np.float32)
    # The tolerance is only for float32 round-trip at the canonical bounds;
    # generated coordinates themselves remain exactly inside the bounds.
    tol = np.float32(2e-6)
    if np.any(sensors < low - tol) or np.any(sensors > high + tol):
        raise ValueError("Sensor coordinates fall outside the normalized domain bounds")
    return np.ascontiguousarray(sensors, dtype=np.float32)


def generate_dummy_query_coordinates(
    count: int,
    spec: DummyQuerySpec,
    *,
    sensor_coords: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray:
    """Generate one deterministic shared query sequence of exactly ``count``.

    A sensor prefix is copied byte-for-byte from the canonical mesh.  The
    suffix is drawn directly from a Sobol sequence; it is not a native-grid
    array and is never obtained by slicing/interpolating the 40,300-point mesh.
    Calling this function for different counts with one spec produces a
    prefix-consistent sequence, which lets all methods share the exact same
    coordinates at each requested N.
    """

    count = int(count)
    if count < 1:
        raise ValueError("Dummy query count must be positive")
    sensors = None
    if spec.include_sensor_prefix:
        if sensor_coords is None:
            raise ValueError("sensor_coords is required when include_sensor_prefix=True")
        sensors = _validate_sensor_coords(sensor_coords, spec)
        if count < sensors.shape[0]:
            raise ValueError(
                f"N={count} is smaller than the required sensor prefix ({sensors.shape[0]})"
            )
    prefix_count = 0 if sensors is None else int(sensors.shape[0])
    suffix_count = count - prefix_count

    engine = torch.quasirandom.SobolEngine(
        dimension=int(spec.dimension),
        scramble=bool(spec.scramble),
        seed=int(spec.seed),
    )
    unit = engine.draw(suffix_count).to(dtype=torch.float32).numpy()
    low = np.asarray(spec.low, dtype=np.float32)
    high = np.asarray(spec.high, dtype=np.float32)
    suffix = low[None, :] + unit * (high - low)[None, :]
    suffix = np.ascontiguousarray(suffix, dtype=np.float32)
    if sensors is None:
        return suffix
    return np.ascontiguousarray(np.concatenate((sensors, suffix), axis=0), dtype=np.float32)


def predeclared_query_counts(global_cap: int = DEFAULT_GLOBAL_QUERY_CAP) -> tuple[int, ...]:
    """Return the fixed grid followed by deterministic doubling to the cap."""

    cap = int(global_cap)
    if cap < PREDECLARED_QUERY_COUNTS[-1]:
        raise ValueError(
            f"global_cap must be >= {PREDECLARED_QUERY_COUNTS[-1]:,}; got {cap:,}"
        )
    values = [value for value in PREDECLARED_QUERY_COUNTS if value <= cap]
    current = values[-1]
    while current < cap:
        current = min(cap, current * 2)
        if values[-1] != current:
            values.append(current)
    return tuple(values)


def _first_source_line(path: Path, needles: Sequence[str]) -> int | None:
    if not path.exists():
        return None
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines, start=1):
        if all(needle in line for needle in needles):
            return index
    return None


def _relative_source(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def _architecture_config(cfg: Mapping[str, Any], family: str) -> dict[str, Any]:
    if family == "pointcloud_ffm":
        return dict(cfg)
    section = cfg.get(f"{family}_params", {})
    if not isinstance(section, Mapping):
        return {}
    if family == "latent_fm":
        stage = section.get("stage2", {}) if int(cfg.get("training_stage", 1)) == 2 else section.get("stage1", {})
    else:
        stage = section
    if not isinstance(stage, Mapping):
        return {}
    architecture = stage.get("architecture", {})
    return dict(architecture) if isinstance(architecture, Mapping) else {}


def audit_native_query_support(
    method_cfg: Sequence[Mapping[str, Any]],
    *,
    condition: str = "Cond_T",
) -> list[dict[str, Any]]:
    """Audit all adopted methods from canonical config and architecture code.

    ``LoadedModel.reconstruct`` historically rejects arbitrary baseline query
    subsets.  That adapter policy is recorded but is *not* used as the native
    architecture decision: MLP-RBF and Senseiver are audited from their direct
    query-coordinate ``forward`` methods, as required by V4.
    """

    by_name = {str(item["name"]): item for item in method_cfg}
    source_model = REPO_ROOT / "0_demo_TurbulentCombustion/src/model_baseline.py"
    source_portable = REPO_ROOT / "0_demo_TurbulentCombustion/src/phycoflow_pointcloud/models/portable_core.py"
    source_coherence = REPO_ROOT / "0_demo_TurbulentCombustion/src/evaluate_coherence.py"
    rows: list[dict[str, Any]] = []

    for method in METHODS:
        item = by_name.get(method)
        base = {
            "method": method,
            "condition": condition,
            "status": "ok",
            "native_query_supported": False,
            "native_only": True,
            "query_scaling_eligible": False,
            "family": "",
            "backbone": "",
            "variant": "",
            "architecture_config_path": "",
            "architecture_config_sha256": "",
            "evidence_source": "",
            "evidence_line": "",
            "evidence_symbol": "",
            "decision_basis": "",
            "benchmark_query_path": "",
            "loader_query_subset_api": "not_a_decision_boundary",
        }
        if item is None:
            base.update(status="missing_method_config", decision_basis="method is absent from canonical postprocess_config")
            rows.append(base)
            continue

        run_dir = (
            REPO_ROOT
            / "0_demo_TurbulentCombustion/Save_TrainedModel/_TrainedModels"
            / str(item["directory"])
            / condition
        )
        cfg_path = run_dir / "run_config.yaml"
        base["architecture_config_path"] = _relative_source(cfg_path)
        if not cfg_path.exists():
            base.update(status="missing_run_config", decision_basis="canonical run_config.yaml is unavailable")
            rows.append(base)
            continue
        raw_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        cfg = dict(raw_cfg) if isinstance(raw_cfg, Mapping) else {}
        family = str(cfg.get("baseline_model", "pointcloud_ffm")).strip().lower()
        arch = _architecture_config(cfg, family)
        base["family"] = family
        base["backbone"] = str(cfg.get("backbone", family))
        base["variant"] = str(arch.get("geofno_variant", arch.get("tokenizer", "")))
        base["architecture_config_sha256"] = sha256(cfg_path)

        supported = False
        source = source_model
        line = None
        symbol = ""
        basis = ""
        benchmark_path = ""

        if family == "pointcloud_ffm":
            backbone = str(cfg.get("backbone", "")).strip().lower()
            if backbone == "fno":
                source = source_model
                line = _first_source_line(source, ("expected N = Num_x * Num_y",))
                symbol = "FNO._pointcloud_to_grid"
                basis = (
                    "canonical FNOFFM requires a complete fixed Num_y x Num_x grid; "
                    "FNO._pointcloud_to_grid rejects any N != Num_x*Num_y"
                )
                benchmark_path = "native_reference_only"
            elif backbone in {"perceiver"}:
                supported = True
                source = source_model
                line = _first_source_line(source, ("def _decode_queries_chunked",))
                symbol = "ConditionalPointPerceiver._decode_queries_chunked"
                basis = (
                    "canonical point-cloud Perceiver consumes requested coords as query tokens; "
                    "decoder chunking returns one output per requested query"
                )
                benchmark_path = "core_call -> PointCloudFFM.sample"
            elif backbone in {
                "gl_rbf",
                "gl_rbf_enh",
                "gl_rbf_enh_cq",
                "hybrid_localglobal_rbf",
            }:
                supported = True
                source = source_portable
                line = _first_source_line(source, ("def aggregate_sparse_obs(",))
                symbol = "ConditionalPointHybridLocalGlobalRBF.aggregate_sparse_obs"
                basis = (
                    "canonical GL-RBF/CQ gather operates on arbitrary query_coords and emits "
                    "one readout per requested query; no native-grid reshape"
                )
                benchmark_path = "core_call -> PointCloudFFM.sample"
            else:
                source = source_model
                line = _first_source_line(source, ("class PointCloudFFM",))
                symbol = "PointCloudFFM.sample"
                basis = f"unrecognized canonical point-cloud backbone {backbone!r}; scaling is conservatively disabled"
                benchmark_path = "native_reference_only"
        elif family == "mlp_rbf":
            supported = True
            source = source_model
            line = _first_source_line(source, ("class DeterministicMLPRBFRegressor",))
            symbol = "DeterministicMLPRBFRegressor.forward"
            basis = (
                "canonical deterministic MLP-RBF receives query_coords directly and evaluates "
                "the pointwise backbone at the requested set"
            )
            benchmark_path = "core_call -> bundle.model(query_coords, ... )"
        elif family == "senseiver":
            supported = True
            source = source_model
            line = _first_source_line(source, ("class Senseiver(nn.Module)",))
            symbol = "Senseiver.forward"
            basis = (
                "canonical Senseiver decoder builds positional query tokens from query_coords "
                "and cross-attends each requested query to fixed latent memory"
            )
            benchmark_path = "core_call -> bundle.model(query_coords, ... )"
        elif family == "geofno":
            variant = str(arch.get("geofno_variant", "fno")).strip().lower()
            if variant == "irregular":
                source = source_model
                line = _first_source_line(source, ("class FNOSupervisedIrregular",))
                symbol = "FNOSupervisedIrregular.forward"
                basis = (
                    "canonical irregular Geo-FNO first rasterizes to a fixed latent grid and "
                    "then samples that grid; V4 excludes fixed-grid-then-slice paths"
                )
            else:
                source = source_coherence
                line = _first_source_line(source, ("if variant == \"irregular\":",))
                symbol = "_baseline_reconstruct_deterministic (Geo-FNO fno branch)"
                basis = (
                    "adopted Geo-FNO variant=fno constructs the complete fixed native grid; "
                    "the canonical adapter has no arbitrary query output path"
                )
            benchmark_path = "native_reference_only"
        elif family == "latent_fm":
            source = source_coherence
            line = _first_source_line(source, ("def _baseline_reconstruct_latentfm",))
            symbol = "_baseline_reconstruct_latentfm"
            basis = (
                "canonical Latent FM conditional sampler operates on the fixed num_y x num_x "
                "grid and grid_to_pointcloud is only a flattening conversion"
            )
            benchmark_path = "native_reference_only"
        elif family == "sit":
            tokenizer = str(arch.get("tokenizer", "patch")).strip().lower()
            source = source_model
            line = _first_source_line(source, ("if self.tokenizer == \"patch\":",))
            symbol = "SiTPhysics.forward"
            if tokenizer == "pointnet":
                supported = True
                basis = (
                    "canonical SiT pointnet tokenizer creates one token per requested coordinate; "
                    "query geometry is not rasterized"
                )
                benchmark_path = "canonical pointnet adapter"
            else:
                basis = (
                    "adopted SiT tokenizer=patch uses fixed input_size_h/input_size_w and "
                    "unpatchify; arbitrary query scaling is disabled"
                )
                benchmark_path = "native_reference_only"
        else:
            source = source_coherence
            line = _first_source_line(source, ("def reconstruct_baseline_snapshot",))
            symbol = "reconstruct_baseline_snapshot"
            basis = f"unsupported canonical family {family!r}; scaling is conservatively disabled"
            benchmark_path = "native_reference_only"

        base.update(
            native_query_supported=bool(supported),
            native_only=not supported,
            query_scaling_eligible=bool(supported),
            evidence_source=_relative_source(source),
            evidence_line="" if line is None else int(line),
            evidence_symbol=symbol,
            decision_basis=basis,
            benchmark_query_path=benchmark_path,
        )
        rows.append(base)

    if [row["method"] for row in rows] != list(METHODS):
        raise RuntimeError("Native-query audit did not produce the canonical eight-method order")
    return rows


def _stress_prepared(
    loaded: Any,
    *,
    method: str,
    state: int,
    sensor_rows: list[dict[str, int]],
    query_coords_cpu: np.ndarray,
) -> dict[str, Any]:
    """Prepare one fixed state and one fixed output geometry outside timing."""

    sample = loaded.dataset[state]
    native_coords = sample["coords"].unsqueeze(0).to(loaded.device)
    truth = sample["fields"].unsqueeze(0).to(loaded.device)
    obs_indices_full, obs_field_ids = _plan_tensors(sensor_rows, loaded.device)
    obs_coords = native_coords[:, obs_indices_full[0]]
    obs_values = torch.stack(
        [truth[0, index, field] for index, field in zip(obs_indices_full[0], obs_field_ids[0])]
    ).view(1, -1, 1)
    obs_mask = torch.ones(
        (1, obs_indices_full.shape[1]),
        device=loaded.device,
        dtype=native_coords.dtype,
    )
    query_coords = torch.from_numpy(np.ascontiguousarray(query_coords_cpu, dtype=np.float32))
    query_coords = query_coords.unsqueeze(0).to(loaded.device)
    # The generator's exact sensor prefix gives a state-independent clamp map.
    clamp_indices = torch.arange(
        obs_indices_full.shape[1], device=loaded.device, dtype=torch.long
    ).view(1, -1)
    return {
        "state": int(state),
        "coords": query_coords,
        "truth": None,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_indices": clamp_indices,
        "obs_indices_full": obs_indices_full,
        "obs_field_ids": obs_field_ids,
        "geometry": None,
        "native_coords": native_coords,
        "method": method,
    }


def _measure_geometry_prepare(
    loaded: Any,
    prepared: dict[str, Any],
    *,
    method: str,
    chunk_size: int,
    device: str,
) -> tuple[Any, float, float, float, str]:
    """Measure first-use query-geometry preparation independently."""

    if method != "DMF-Gen":
        return None, 0.0, float("nan"), float("nan"), "not_applicable"

    dev = torch.device(device)
    torch.cuda.reset_peak_memory_stats(dev)
    torch.cuda.synchronize(dev)
    start = time.perf_counter()
    try:
        geometry = loaded.model.prepare_reconstruction_geometry_cache(
            coords=prepared["coords"],
            obs_coords=prepared["obs_coords"],
            obs_mask=prepared["obs_mask"],
            chunk_size=int(chunk_size),
        )
        torch.cuda.synchronize(dev)
    except Exception:
        torch.cuda.synchronize(dev)
        raise
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    allocated = torch.cuda.max_memory_allocated(dev) / 2**20
    reserved = torch.cuda.max_memory_reserved(dev) / 2**20
    return geometry, float(elapsed_ms), float(allocated), float(reserved), "ok"


class ScaleBoundaryReached(RuntimeError):
    """A requested N hit a V4 declared hardware/runtime boundary."""

    def __init__(self, reason: str, *, first_warm_ms: float | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.first_warm_ms = first_warm_ms


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in text or "cuda error: out of memory" in text


def _seed_call(base_seed: int, method: str, count: int, repeat: int, phase: str) -> int:
    return stable_seed(int(base_seed), "scale_stress_v4", method, int(count), int(repeat), phase)


def _invoke_with_seed(
    call: Callable[[], Any],
    *,
    seed: int,
) -> Any:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    return call()


def _latency_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {
            "latency_p10_ms": float("nan"),
            "latency_q25_ms": float("nan"),
            "median_latency_ms": float("nan"),
            "latency_q75_ms": float("nan"),
            "latency_p90_ms": float("nan"),
            "latency_iqr_ms": float("nan"),
        }
    p10, q25, median, q75, p90 = np.quantile(np.asarray(values, dtype=np.float64), [0.10, 0.25, 0.50, 0.75, 0.90])
    return {
        "latency_p10_ms": float(p10),
        "latency_q25_ms": float(q25),
        "median_latency_ms": float(median),
        "latency_q75_ms": float(q75),
        "latency_p90_ms": float(p90),
        "latency_iqr_ms": float(q75 - q25),
    }


def benchmark_stress_count(
    loaded: Any,
    prepared: dict[str, Any],
    *,
    method: str,
    count: int,
    settings: Mapping[str, Any],
    warmups: int,
    minimum_repeats: int,
    minimum_seconds: float,
    runtime_cap_seconds: float,
    memory_fraction: float,
    total_vram_bytes: int,
    device: str,
    base_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run one N with synchronized warmups/repeats and strict boundaries."""

    call = lambda: core_call(loaded, prepared, dict(settings), method)
    dev = torch.device(device)
    repeats: list[dict[str, Any]] = []
    first_warm_ms: float | None = None
    try:
        # Warmups are timed so the runtime cap also covers pathological first
        # model-core executions.  Their values never enter the latency summary.
        for warmup in range(int(warmups)):
            _, elapsed = time_cuda(
                lambda warmup=warmup: _invoke_with_seed(
                    call, seed=_seed_call(base_seed, method, count, warmup, "warmup")
                ),
                device,
            )
            first_warm_ms = float(elapsed) if first_warm_ms is None else first_warm_ms
            if elapsed > float(runtime_cap_seconds) * 1000.0:
                raise ScaleBoundaryReached(
                    f"warm inference exceeded runtime cap ({elapsed / 1000.0:.3f}s > {runtime_cap_seconds:.3f}s)",
                    first_warm_ms=first_warm_ms,
                )
        torch.cuda.synchronize(dev)
        values: list[float] = []
        total_ms = 0.0
        repeat_index = 0
        while len(values) < int(minimum_repeats) or total_ms < float(minimum_seconds) * 1000.0:
            output, elapsed = time_cuda(
                lambda repeat_index=repeat_index: _invoke_with_seed(
                    call,
                    seed=_seed_call(base_seed, method, count, repeat_index, "repeat"),
                ),
                device,
            )
            del output
            if elapsed > float(runtime_cap_seconds) * 1000.0:
                raise ScaleBoundaryReached(
                    f"warm inference exceeded runtime cap ({elapsed / 1000.0:.3f}s > {runtime_cap_seconds:.3f}s)",
                    first_warm_ms=first_warm_ms,
                )
            values.append(float(elapsed))
            total_ms += float(elapsed)
            repeats.append(
                {
                    "method": method,
                    "N": int(count),
                    "repeat": int(repeat_index),
                    "latency_ms": float(elapsed),
                    "state": int(prepared["state"]),
                    "suite": "scale_stress_v4",
                    "throughput_only": True,
                    "accuracy_claim": False,
                }
            )
            repeat_index += 1
            if repeat_index > 100_000:
                raise RuntimeError("V4 timing loop exceeded the declared safety repeat limit")

        # Measure memory on one additional, explicitly isolated core inference.
        # Resetting before the warmup/repeat block would conflate the requested
        # one-inference peak with allocator history across the timing loop.
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
        memory_output = _invoke_with_seed(
            call,
            seed=_seed_call(base_seed, method, count, 0, "memory"),
        )
        torch.cuda.synchronize(dev)
        peak_allocated = torch.cuda.max_memory_allocated(dev) / 2**20
        peak_reserved = torch.cuda.max_memory_reserved(dev) / 2**20
        del memory_output
        memory_limit_mib = float(total_vram_bytes * float(memory_fraction) / 2**20)
        if peak_allocated > memory_limit_mib:
            raise ScaleBoundaryReached(
                f"peak allocated memory exceeded {memory_fraction:.2%} of physical VRAM "
                f"({peak_allocated:.1f} MiB > {memory_limit_mib:.1f} MiB)",
                first_warm_ms=first_warm_ms,
            )
        summary = {
            "method": method,
            "N": int(count),
            "status": "ok",
            "failure_reason": "",
            "state": int(prepared["state"]),
            "batch_size": 1,
            "dtype": "float32",
            "warmups": int(warmups),
            "repeats": len(values),
            "timed_total_ms": float(total_ms),
            "first_warm_core_ms": float(first_warm_ms if first_warm_ms is not None else float("nan")),
            "peak_allocated_mib": float(peak_allocated),
            "peak_reserved_mib": float(peak_reserved),
            "memory_limit_mib": memory_limit_mib,
            "runtime_cap_seconds": float(runtime_cap_seconds),
            "memory_fraction": float(memory_fraction),
            "throughput_only": True,
            "accuracy_claim": False,
            **_latency_summary(values),
            **dict(settings),
        }
        return summary, repeats
    except ScaleBoundaryReached as exc:
        torch.cuda.synchronize(dev)
        peak_allocated = torch.cuda.max_memory_allocated(dev) / 2**20
        peak_reserved = torch.cuda.max_memory_reserved(dev) / 2**20
        summary = {
            "method": method,
            "N": int(count),
            "status": "boundary_failure",
            "failure_reason": exc.reason,
            "state": int(prepared["state"]),
            "batch_size": 1,
            "dtype": "float32",
            "warmups": int(warmups),
            "repeats": len(repeats),
            "timed_total_ms": float(sum(row["latency_ms"] for row in repeats)),
            "first_warm_core_ms": float(exc.first_warm_ms if exc.first_warm_ms is not None else float("nan")),
            "peak_allocated_mib": float(peak_allocated),
            "peak_reserved_mib": float(peak_reserved),
            "memory_limit_mib": float(total_vram_bytes * float(memory_fraction) / 2**20),
            "runtime_cap_seconds": float(runtime_cap_seconds),
            "memory_fraction": float(memory_fraction),
            "throughput_only": True,
            "accuracy_claim": False,
            **_latency_summary([row["latency_ms"] for row in repeats]),
            **dict(settings),
        }
        return summary, repeats
    except Exception as exc:  # noqa: BLE001 - retain every backend failure as a boundary row
        torch.cuda.synchronize(dev)
        peak_allocated = torch.cuda.max_memory_allocated(dev) / 2**20
        peak_reserved = torch.cuda.max_memory_reserved(dev) / 2**20
        reason = "cuda_oom" if _is_cuda_oom(exc) else f"{type(exc).__name__}: {exc}"
        summary = {
            "method": method,
            "N": int(count),
            "status": "failed",
            "failure_reason": reason,
            "state": int(prepared["state"]),
            "batch_size": 1,
            "dtype": "float32",
            "warmups": int(warmups),
            "repeats": len(repeats),
            "timed_total_ms": float(sum(row["latency_ms"] for row in repeats)),
            "first_warm_core_ms": float(first_warm_ms if first_warm_ms is not None else float("nan")),
            "peak_allocated_mib": float(peak_allocated),
            "peak_reserved_mib": float(peak_reserved),
            "memory_limit_mib": float(total_vram_bytes * float(memory_fraction) / 2**20),
            "runtime_cap_seconds": float(runtime_cap_seconds),
            "memory_fraction": float(memory_fraction),
            "throughput_only": True,
            "accuracy_claim": False,
            **_latency_summary([row["latency_ms"] for row in repeats]),
            **dict(settings),
        }
        return summary, repeats


def _empty_failure_row(method: str, count: int, reason: str) -> dict[str, Any]:
    return {
        "method": method,
        "N": int(count),
        "status": "failed",
        "failure_reason": reason,
        "throughput_only": True,
        "accuracy_claim": False,
    }


def strict_scale_qa(
    *,
    support_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
    repeat_rows: Sequence[Mapping[str, Any]],
    geometry_rows: Sequence[Mapping[str, Any]],
    boundary_rows: Sequence[Mapping[str, Any]],
    query_rows: Sequence[Mapping[str, Any]],
    candidate_counts: Sequence[int],
    query_hashes_by_count: Mapping[int, str],
    expected_eligible: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Strict, result-backed QA for a completed V4 run."""

    support_methods = [str(row.get("method")) for row in support_rows]
    eligible = {
        str(row["method"])
        for row in support_rows
        if bool(row.get("native_query_supported")) and bool(row.get("query_scaling_eligible"))
    }
    expected = set(expected_eligible or eligible)
    attempted_methods = {str(row.get("method")) for row in summary_rows}
    successful_rows = [row for row in summary_rows if str(row.get("status")) == "ok"]
    successful_methods = {str(row.get("method")) for row in successful_rows}
    # A method may legitimately terminate at its first declared OOM/runtime
    # boundary.  Treat that recorded prefix as an auditable outcome; requiring
    # every method to reach the global cap would make the strict QA reject the
    # very boundary evidence the V4 protocol is designed to preserve.
    outcome_methods = {
        str(row.get("method"))
        for row in summary_rows
        if row.get("N") not in (None, "", "nan")
    }
    all_eligible_have_outcome = outcome_methods == expected
    attempted_counts = {int(row["N"]) for row in summary_rows if row.get("N") not in (None, "")}
    hashes_match = True
    for count in attempted_counts:
        rows = [row for row in query_rows if int(row.get("N", -1)) == count]
        hashes = {str(row.get("query_sha256")) for row in rows}
        if len(hashes) > 1 or (rows and next(iter(hashes)) != query_hashes_by_count.get(count)):
            hashes_match = False
        for row in summary_rows:
            if row.get("N") in (None, "") or int(row["N"]) != count:
                continue
            if str(row.get("query_sha256", "")) != str(query_hashes_by_count.get(count, "")):
                hashes_match = False
    boundary_ok = True
    for method in expected:
        rows = [row for row in summary_rows if str(row.get("method")) == method]
        if not rows:
            boundary_ok = False
            continue
        ok_counts = [int(row["N"]) for row in rows if str(row.get("status")) == "ok"]
        fail_counts = [
            int(row["N"])
            for row in rows
            if str(row.get("status")) != "ok" and row.get("N") not in (None, "")
        ]
        b = [row for row in boundary_rows if str(row.get("method")) == method]
        if len(b) != 1:
            boundary_ok = False
            continue
        largest = b[0].get("largest_success_N")
        first_failure = b[0].get("first_failure_N")
        if (max(ok_counts) if ok_counts else None) != (None if largest in (None, "") else int(largest)):
            boundary_ok = False
        if (min(fail_counts) if fail_counts else None) != (None if first_failure in (None, "") else int(first_failure)):
            boundary_ok = False

    latency_valid = all(
        float(row.get("median_latency_ms", float("nan"))) > 0
        and float(row.get("latency_iqr_ms", float("nan"))) >= 0
        and int(row.get("repeats", 0)) > 0
        for row in successful_rows
    )
    no_fake_curve = all(bool(row.get("throughput_only")) and not bool(row.get("accuracy_claim")) for row in summary_rows)
    no_unsupported_curve = all(str(row.get("method")) in eligible for row in summary_rows)
    geometry_separate = all("geometry_prepare_ms" in row for row in geometry_rows)
    return {
        "status": "pass"
        if all(
            (
                support_methods == list(METHODS),
                eligible == expected,
                attempted_methods == expected,
                all_eligible_have_outcome,
                {int(value) for value in candidate_counts} >= attempted_counts,
                hashes_match,
                boundary_ok,
                latency_valid,
                no_fake_curve,
                no_unsupported_curve,
                geometry_separate,
                len(repeat_rows) >= len(successful_rows),
            )
        )
        else "fail",
        "support_methods_exact": support_methods == list(METHODS),
        "eligible_methods": sorted(eligible),
        "expected_eligible_methods": sorted(expected),
        "all_eligible_attempted": attempted_methods == expected,
        "all_eligible_have_success": successful_methods == expected,
        "all_eligible_have_outcome": all_eligible_have_outcome,
        "candidate_counts_predeclared": {int(value) for value in candidate_counts} >= attempted_counts,
        "shared_query_hash_per_count": hashes_match,
        "largest_success_first_failure_recorded": boundary_ok,
        "latency_iqr_valid": latency_valid,
        "throughput_only_no_accuracy_claim": no_fake_curve,
        "no_unsupported_scaling_curve": no_unsupported_curve,
        "geometry_preparation_separate": geometry_separate,
        "repeat_rows_present": len(repeat_rows) >= len(successful_rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True, help="Frozen V3 validation plan")
    parser.add_argument("--device", default="cuda:2", help="Explicit clean CUDA device for formal runs")
    parser.add_argument("--run-id")
    parser.add_argument("--state", type=int, default=None, help="One fixed test state for the stress geometry")
    parser.add_argument("--query-seed", type=int, default=DEFAULT_QUERY_SEED)
    parser.add_argument("--global-cap", type=int, default=DEFAULT_GLOBAL_QUERY_CAP)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--minimum-repeats", type=int, default=30)
    parser.add_argument("--minimum-seconds", type=float, default=10.0)
    parser.add_argument("--runtime-cap-seconds", type=float, default=DEFAULT_RUNTIME_CAP_SECONDS)
    parser.add_argument("--memory-fraction", type=float, default=DEFAULT_MEMORY_FRACTION)
    parser.add_argument("--query-chunk-size", type=int, default=8192)
    parser.add_argument("--save-coordinates", action="store_true", help="Persist query .npy files under the V4 root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--audit-only", action="store_true", help="Write only the architecture audit; no CUDA/model load")
    mode.add_argument("--dry-run", action="store_true", help="Print the V4 plan/audit; no output and no CUDA/model load")
    return parser.parse_args()


def _audit_output(root: Path, run_id: str, support_rows: Sequence[Mapping[str, Any]], *, status: str) -> Path:
    output_dir = root / "Dis_SI_Process" / "results" / "ValidationV4" / "ScaleStress" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    write_csv(output_dir / "native_query_support_audit.csv", [dict(row) for row in support_rows])
    atomic = {
        "schema_version": "figure5-validation-v4-scale-stress-audit-1",
        "status": status,
        "run_id": run_id,
        "methods": list(METHODS),
        "support_audit": [dict(row) for row in support_rows],
        "source_policy": "canonical run_config.yaml plus architecture source symbols; loader subset policy is not a decision boundary",
    }
    (output_dir / "support_audit.json").write_text(json.dumps(atomic, indent=2, default=str), encoding="utf-8")
    return output_dir


def _print_dry_run(args: argparse.Namespace, support_rows: Sequence[Mapping[str, Any]]) -> None:
    payload = {
        "schema_version": "figure5-validation-v4-scale-stress-plan-1",
        "device": args.device,
        "predeclared_query_counts": list(PREDECLARED_QUERY_COUNTS),
        "candidate_query_counts": list(predeclared_query_counts(args.global_cap)),
        "global_cap": int(args.global_cap),
        "runtime_cap_seconds": float(args.runtime_cap_seconds),
        "memory_fraction": float(args.memory_fraction),
        "timing": {
            "warmups": int(args.warmups),
            "minimum_repeats": int(args.minimum_repeats),
            "minimum_seconds": float(args.minimum_seconds),
        },
        "support_audit": [dict(row) for row in support_rows],
    }
    print(json.dumps(payload, indent=2, default=str))


def run_scale_stress(args: argparse.Namespace) -> int:
    _ensure_runtime_imports()
    if args.warmups < 20 or args.minimum_repeats < 30 or args.minimum_seconds < 10:
        raise ValueError("Formal V4 scale stress requires >=20 warmups, >=30 repeats, and >=10 measured seconds")
    if not 0.0 < float(args.memory_fraction) <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    if float(args.runtime_cap_seconds) <= 0.0:
        raise ValueError("runtime_cap_seconds must be positive")
    candidate_counts = predeclared_query_counts(args.global_cap)
    plan_path = args.plan.resolve()
    plan = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    post_cfg = load_config()
    support_rows = audit_native_query_support(post_cfg["methods"], condition="Cond_T")
    expected_eligible = [
        row["method"] for row in support_rows if row["query_scaling_eligible"] and row["native_query_supported"]
    ]

    index = gpu_index(args.device)
    before_processes = assert_clean_gpu(index, allow_current=False)
    torch.cuda.set_device(index)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False)
    run_id = args.run_id or f"formal_scale_stress_v4_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    output_dir = REPO_ROOT / "Dis_SI_Process" / "results" / "ValidationV4" / "ScaleStress" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "gpu_state_before.txt").write_text(gpu_state(index), encoding="utf-8")

    identity = verify_identities(plan)
    method_cfg = {row["name"]: row for row in post_cfg["methods"]}
    stress_states = list(map(int, plan["cohorts"]["cost_native_20"]["evaluation_indices"]))
    state = int(args.state if args.state is not None else stress_states[0])
    sensors = load_sensor_rows(plan, [state])[state]

    # Load one eligible canonical dataset solely to establish the normalized
    # domain and exact sensor coordinates.  All later methods validate these
    # bounds, so no method can silently receive a different query sequence.
    first_loaded = None
    first_load_errors: list[str] = []
    for method in expected_eligible:
        try:
            settings = method_settings(post_cfg, method)
            first_loaded = load_model(
                method_cfg[method],
                "Cond_T",
                checkpoint="last",
                split="test",
                device=args.device,
                n_steps=settings["n_steps"],
                ode_solver="euler",
            )
            break
        except Exception as exc:  # noqa: BLE001 - try the next canonical eligible loader
            first_load_errors.append(f"{method}: {type(exc).__name__}: {exc}")
    if first_loaded is None:
        raise RuntimeError(
            "No eligible V4 query-evaluable checkpoint could be loaded to establish domain bounds; "
            + " | ".join(first_load_errors)
        )

    sample = first_loaded.dataset[state]
    native_coords_cpu = sample["coords"].detach().cpu()
    spec = make_dummy_query_spec(native_coords_cpu, seed=args.query_seed)
    native_obs_idx, _ = _plan_tensors(sensors, torch.device("cpu"))
    sensor_coords_cpu = native_coords_cpu[native_obs_idx[0]].numpy().astype(np.float32, copy=False)
    sensor_digest = hash_coordinates(sensor_coords_cpu)
    frozen_spec_hash = spec_hash(spec, sensor_coords_sha256=sensor_digest)
    del first_loaded
    gc.collect()
    torch.cuda.empty_cache()

    query_bank: dict[int, np.ndarray] = {}
    query_hashes: dict[int, str] = {}
    query_rows: list[dict[str, Any]] = []

    def get_query(count: int) -> np.ndarray:
        if count not in query_bank:
            values = generate_dummy_query_coordinates(count, spec, sensor_coords=sensor_coords_cpu)
            query_bank[count] = values
            coord_digest = hash_coordinates(values)
            combined = query_hash(values, spec, sensor_coords_sha256=sensor_digest)
            query_hashes[count] = combined
            query_rows.append(
                {
                    "N": int(count),
                    "sensor_count": int(sensor_coords_cpu.shape[0]),
                    "sensor_prefix_sha256": sensor_digest,
                    "coordinate_sha256": coord_digest,
                    "spec_sha256": frozen_spec_hash,
                    "query_sha256": combined,
                    "generator": spec.generator,
                    "seed": int(spec.seed),
                    "dimension": int(spec.dimension),
                    "low": json.dumps(list(spec.low)),
                    "high": json.dumps(list(spec.high)),
                    "sequence_policy": spec.sequence_policy,
                    "throughput_only": True,
                    "accuracy_claim": False,
                }
            )
            if args.save_coordinates:
                query_dir = output_dir / "queries"
                query_dir.mkdir(parents=True, exist_ok=True)
                np.save(query_dir / f"query_coordinates_N{count}.npy", values, allow_pickle=False)
                query_rows[-1]["saved_path"] = str((query_dir / f"query_coordinates_N{count}.npy").relative_to(output_dir))
        return query_bank[count]

    manifest = {
        "schema_version": "figure5-validation-v4-scale-stress-1",
        "status": "running",
        "formal": True,
        "run_id": run_id,
        "plan": str(plan_path),
        "plan_sha256": sha256(plan_path),
        "identity_checks": identity,
        "environment": v3_environment(args.device),
        "gpu_clean_before": not before_processes,
        "architecture_audit": "native_query_support_audit.csv",
        "protocol": {
            "condition": "Cond_T",
            "state": state,
            "sensor_count": SENSOR_COUNT,
            "batch_size": 1,
            "dtype": "float32",
            "native_query_count": NATIVE_QUERY_COUNT,
            "predeclared_query_counts": list(PREDECLARED_QUERY_COUNTS),
            "candidate_query_counts": list(candidate_counts),
            "global_query_cap": int(args.global_cap),
            "boundary_policy": DEFAULT_BOUNDARY_POLICY,
            "memory_boundary_fraction": float(args.memory_fraction),
            "runtime_cap_seconds": float(args.runtime_cap_seconds),
            "timing_boundary": {
                "name": "warm_model_core_fixed_geometry",
                "included": [
                    "stochastic_prior_or_noise_generation",
                    "value_dependent_conditioning",
                    "all_model_or_flow_evaluations",
                    "adopted_observation_consistency",
                    "final_device_output",
                ],
                "excluded": [
                    "model_loading",
                    "dataset_IO",
                    "host_to_device_transfer",
                    "query_coordinate_generation",
                    "query_coordinate_host_to_device_transfer",
                    "first_use_geometry_preparation",
                    "metrics",
                    "device_to_host_transfer",
                    "disk_IO",
                ],
                "persistent_cache": "only state-independent/query-geometry cache permitted by canonical method",
            },
        },
        "dummy_query_spec": spec.payload(sensor_coords_sha256=sensor_digest),
        "dummy_query_spec_sha256": frozen_spec_hash,
        "native_query_support": support_rows,
        "throughput_only_disclaimer": "N > 40,300 is a dummy-query throughput and memory stress test; no accuracy claim is made.",
    }
    write_csv(output_dir / "native_query_support_audit.csv", [dict(row) for row in support_rows])
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []

    for method in METHODS:
        support = next(row for row in support_rows if row["method"] == method)
        if not support["query_scaling_eligible"]:
            native_rows.append(
                {
                    "method": method,
                    "N": NATIVE_QUERY_COUNT,
                    "status": "native_only_reference",
                    "native_query_supported": False,
                    "native_only": True,
                    "decision_basis": support["decision_basis"],
                    "throughput_only": False,
                    "accuracy_claim": "native_reference_only",
                }
            )
            continue

        loaded = None
        attempted: list[dict[str, Any]] = []
        try:
            settings = method_settings(post_cfg, method)
            loaded = load_model(
                method_cfg[method],
                "Cond_T",
                checkpoint="last",
                split="test",
                device=args.device,
                n_steps=settings["n_steps"],
                ode_solver="euler",
            )
            with evaluation_context(loaded):
                # Validate that every eligible canonical dataset has the same
                # normalized domain and exact sensor coordinates before timing.
                method_coords = loaded.dataset[state]["coords"].detach().cpu().numpy()
                method_low, method_high = bounds_from_normalized_coords(method_coords)
                if not np.allclose(method_low, np.asarray(spec.low), atol=2e-6) or not np.allclose(method_high, np.asarray(spec.high), atol=2e-6):
                    raise RuntimeError(f"{method} canonical coordinate bounds differ from the shared V4 domain")

                for count in candidate_counts:
                    coords_cpu = get_query(count)
                    prepared = _stress_prepared(
                        loaded,
                        method=method,
                        state=state,
                        sensor_rows=sensors,
                        query_coords_cpu=coords_cpu,
                    )
                    try:
                        geometry, geometry_ms, geometry_alloc, geometry_reserved, geometry_status = _measure_geometry_prepare(
                            loaded,
                            prepared,
                            method=method,
                            chunk_size=args.query_chunk_size,
                            device=args.device,
                        )
                        prepared["geometry"] = geometry
                        geometry_rows.append(
                            {
                                "method": method,
                                "N": int(count),
                                "status": geometry_status,
                                "geometry_prepare_ms": float(geometry_ms),
                                "geometry_prepare_peak_allocated_mib": float(geometry_alloc),
                                "geometry_prepare_peak_reserved_mib": float(geometry_reserved),
                                "query_sha256": query_hashes[count],
                                "throughput_only": True,
                                "accuracy_claim": False,
                            }
                        )
                    except Exception as exc:  # noqa: BLE001 - retain geometry failure in SI table
                        reason = "cuda_oom_geometry" if _is_cuda_oom(exc) else f"geometry_{type(exc).__name__}: {exc}"
                        geometry_rows.append(
                            {
                                "method": method,
                                "N": int(count),
                                "status": "failed",
                                "geometry_prepare_ms": float("nan"),
                                "geometry_prepare_peak_allocated_mib": float("nan"),
                                "geometry_prepare_peak_reserved_mib": float("nan"),
                                "query_sha256": query_hashes[count],
                                "failure_reason": reason,
                                "throughput_only": True,
                                "accuracy_claim": False,
                            }
                        )
                        row = _empty_failure_row(method, count, reason)
                        row.update({"query_sha256": query_hashes[count], "geometry_prepare_ms": float("nan")})
                        summary_rows.append(row)
                        attempted.append(row)
                        break

                    summary, repeats = benchmark_stress_count(
                        loaded,
                        prepared,
                        method=method,
                        count=count,
                        settings=settings,
                        warmups=args.warmups,
                        minimum_repeats=args.minimum_repeats,
                        minimum_seconds=args.minimum_seconds,
                        runtime_cap_seconds=args.runtime_cap_seconds,
                        memory_fraction=args.memory_fraction,
                        total_vram_bytes=torch.cuda.get_device_properties(index).total_memory,
                        device=args.device,
                        base_seed=args.query_seed,
                    )
                    summary.update(
                        {
                            "query_sha256": query_hashes[count],
                            "coordinate_sha256": next(row["coordinate_sha256"] for row in query_rows if int(row["N"]) == int(count)),
                            "spec_sha256": frozen_spec_hash,
                            "geometry_prepare_ms": geometry_rows[-1]["geometry_prepare_ms"],
                            "geometry_prepare_peak_allocated_mib": geometry_rows[-1]["geometry_prepare_peak_allocated_mib"],
                            "geometry_prepare_peak_reserved_mib": geometry_rows[-1]["geometry_prepare_peak_reserved_mib"],
                        }
                    )
                    summary_rows.append(summary)
                    repeat_rows.extend(repeats)
                    attempted.append(summary)
                    del prepared
                    gc.collect()
                    torch.cuda.empty_cache()
                    if summary["status"] != "ok":
                        break
                    # The query array stays in query_bank for exact hash reuse;
                    # only its method/device tensor is released here.
                    write_csv(output_dir / "scale_stress_summary.csv", summary_rows)
                    write_csv(output_dir / "scale_stress_repeats.csv", repeat_rows)
                    write_csv(output_dir / "geometry_first_use.csv", geometry_rows)
                    write_csv(output_dir / "query_coordinates_manifest.csv", query_rows)
        except Exception as exc:  # noqa: BLE001 - retain method-unavailable status in manifest
            reason = f"{type(exc).__name__}: {exc}"
            summary_rows.append(
                {
                    "method": method,
                    "N": "",
                    "status": "unavailable",
                    "failure_reason": reason,
                    "throughput_only": True,
                    "accuracy_claim": False,
                }
            )
        finally:
            if loaded is not None:
                loaded.close()
            gc.collect()
            torch.cuda.empty_cache()

        ok_counts = [int(row["N"]) for row in attempted if row.get("status") == "ok"]
        failed_counts = [int(row["N"]) for row in attempted if row.get("status") != "ok" and row.get("N") not in (None, "", "nan")]
        if attempted and any(row.get("status") == "unavailable" for row in attempted):
            term = "model_unavailable"
        elif failed_counts:
            term = "first_failure"
        elif ok_counts and max(ok_counts) >= candidate_counts[-1]:
            term = "global_cap_reached"
        else:
            term = "not_attempted"
        boundary_rows.append(
            {
                "method": method,
                "status": "ok" if attempted and not any(row.get("status") == "unavailable" for row in attempted) else "unavailable",
                "largest_success_N": max(ok_counts) if ok_counts else "",
                "first_failure_N": min(failed_counts) if failed_counts else "",
                "attempted_counts": ";".join(str(int(row["N"])) for row in attempted if row.get("N") not in (None, "", "nan")),
                "termination_reason": term,
                "global_query_cap": int(args.global_cap),
                "memory_boundary_fraction": float(args.memory_fraction),
                "runtime_cap_seconds": float(args.runtime_cap_seconds),
                "throughput_only": True,
                "accuracy_claim": False,
            }
        )

    # Persist all provenance before strict QA so a partial or failed run is
    # still auditable and cannot be mistaken for a complete curve.
    write_csv(output_dir / "native_only_reference.csv", native_rows)
    write_csv(output_dir / "scale_stress_summary.csv", summary_rows)
    write_csv(output_dir / "scale_stress_repeats.csv", repeat_rows)
    write_csv(output_dir / "geometry_first_use.csv", geometry_rows)
    write_csv(output_dir / "boundary_summary.csv", boundary_rows)
    write_csv(output_dir / "query_coordinates_manifest.csv", query_rows)

    final_processes = assert_clean_gpu(index, allow_current=True)
    (output_dir / "gpu_state_after.txt").write_text(gpu_state(index), encoding="utf-8")
    qa = strict_scale_qa(
        support_rows=support_rows,
        summary_rows=summary_rows,
        repeat_rows=repeat_rows,
        geometry_rows=geometry_rows,
        boundary_rows=boundary_rows,
        query_rows=query_rows,
        candidate_counts=candidate_counts,
        query_hashes_by_count=query_hashes,
        expected_eligible=expected_eligible,
    )
    qa.update(
        {
            "identity_pass": all(bool(row.get("pass")) for row in identity),
            "gpu_clean_before": not before_processes,
            "gpu_clean_after": all(int(row["pid"]) == os.getpid() for row in final_processes),
            "fixed_grid_methods_have_no_scaling_curve": not any(
                row.get("method") in {r["method"] for r in native_rows} and row.get("N") != NATIVE_QUERY_COUNT
                for row in summary_rows
            ),
            "native_only_reference_rows": len(native_rows),
            "shared_dummy_spec_hash": frozen_spec_hash,
        }
    )
    qa["status"] = "pass" if all(
        bool(qa.get(key))
        for key in (
            "status",  # strict_scale_qa status itself
            "identity_pass",
            "gpu_clean_before",
            "gpu_clean_after",
            "fixed_grid_methods_have_no_scaling_curve",
        )
    ) else "fail"
    (output_dir / "qa.json").write_text(json.dumps(qa, indent=2, default=str), encoding="utf-8")
    manifest["status"] = "complete" if qa["status"] == "pass" else "qa_failed"
    manifest["completed_at"] = datetime.now(timezone.utc).isoformat()
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "qa": qa}, indent=2, default=str))
    return 0 if qa["status"] == "pass" else 2


def main() -> int:
    args = parse_args()
    post_cfg = load_config()
    support_rows = audit_native_query_support(post_cfg["methods"], condition="Cond_T")
    if args.dry_run:
        _print_dry_run(args, support_rows)
        return 0
    if args.audit_only:
        run_id = args.run_id or f"architecture_audit_v4_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        output_dir = _audit_output(REPO_ROOT, run_id, support_rows, status="complete")
        print(json.dumps({"output_dir": str(output_dir), "support": support_rows}, indent=2, default=str))
        return 0
    return run_scale_stress(args)


if __name__ == "__main__":
    raise SystemExit(main())
