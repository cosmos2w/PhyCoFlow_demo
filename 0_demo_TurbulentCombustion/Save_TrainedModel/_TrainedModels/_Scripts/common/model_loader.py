"""Metadata-driven model loading with one reconstruction interface.

Architecture construction is deliberately delegated to ``evaluate_coherence``
and ``model_baseline``. This module contains no model architecture definitions.
"""
from __future__ import annotations

import argparse
import gc
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .config import ARCHIVE_DIR, SRC_DIR

# Keep third-party JIT/cache writes inside the permitted temporary filesystem.
os.environ.setdefault("KEOPS_CACHE_FOLDER", "/tmp/phycoflow_keops")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phycoflow_matplotlib")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluate_coherence as canonical  # noqa: E402
import model_baseline as baseline_lib  # noqa: E402
from helpers import TurbulentCombustionH5Dataset  # noqa: E402


class MissingDependencyError(RuntimeError):
    """A checkpoint references an external artifact that no longer exists."""


@dataclass
class LoadedModel:
    method: str
    condition: str
    family: str
    backbone: str
    checkpoint_path: Path
    checkpoint_name: str
    config: dict
    dataset: Any
    model: Any
    device: torch.device

    def reconstruct(self, snapshot: int, condition_cfg: dict, sensor_rows: list[dict], *,
                    n_steps: int, ode_solver: str, obs_consistency: str, generation_seed: int) -> dict:
        """Reconstruct using a saved sensor plan and canonical family adapters."""
        torch.manual_seed(int(generation_seed)); np.random.seed(int(generation_seed) & 0xFFFFFFFF)
        torch.cuda.manual_seed_all(int(generation_seed))
        cond_fields = [int(x) for x in condition_cfg["cond_fields"]]
        n_obs = [int(x) for x in condition_cfg["n_obs"]]

        if self.family == "pointcloud_ffm":
            sample = self.dataset[snapshot]
            coords = sample["coords"].unsqueeze(0).to(self.device)
            truth = sample["fields"].unsqueeze(0).to(self.device)
            obs_indices, obs_field_ids = _plan_tensors(sensor_rows, self.device)
            obs_coords = coords[:, obs_indices[0]]
            obs_values = torch.stack([truth[0, idx, fld] for idx, fld in zip(obs_indices[0], obs_field_ids[0])]).view(1, -1, 1)
            obs_mask = torch.ones((1, obs_indices.shape[1]), device=self.device, dtype=coords.dtype)
            kwargs = dict(coords=coords, obs_coords=obs_coords, obs_values=obs_values,
                          obs_mask=obs_mask, obs_field_ids=obs_field_ids, n_steps=n_steps,
                          clamp_indices=obs_indices, ode_solver=ode_solver)
            sig = inspect.signature(self.model.sample)
            applied = obs_consistency if "obs_consistency_mode" in sig.parameters else "native_not_applied"
            if "obs_consistency_mode" in sig.parameters:
                kwargs["obs_consistency_mode"] = obs_consistency
            recon = self.model.sample(**{k: v for k, v in kwargs.items() if k in sig.parameters})
            out = {"coords": coords, "truth": truth, "recon": recon, "obs_coords": obs_coords,
                   "obs_values": obs_values, "obs_mask": obs_mask, "obs_indices": obs_indices,
                   "obs_field_ids": obs_field_ids}
        else:
            # Canonical baseline code owns family-specific inference. Inject the
            # already-saved sparse condition into its shared adapter so sensor
            # selection and generation have independent, explicit seeds.
            planned_indices, planned_fields = _plan_tensors(sensor_rows, self.device)

            def use_saved_plan(*, dataset, coords, truth, cond_fields, n_obs_list):
                obs_coords = coords[:, planned_indices[0]]
                obs_values = torch.stack([
                    truth[0, idx, fld] for idx, fld in zip(planned_indices[0], planned_fields[0])
                ]).view(1, -1, 1)
                obs_mask = torch.ones((1, planned_indices.shape[1]), device=self.device, dtype=coords.dtype)
                return obs_coords, obs_values, obs_mask, planned_indices, planned_fields

            original_builder = canonical._baseline_build_sparse
            canonical._baseline_build_sparse = use_saved_plan
            try:
                torch.manual_seed(int(generation_seed)); np.random.seed(int(generation_seed) & 0xFFFFFFFF)
                torch.cuda.manual_seed_all(int(generation_seed))
                out = canonical.reconstruct_baseline_snapshot(
                    self.model, self.dataset, self.device, snapshot, cond_fields, n_obs, n_steps, ode_solver)
            finally:
                canonical._baseline_build_sparse = original_builder
            applied = "native_not_applied"
            planned = [(int(r["point_index"]), int(r["field_index"])) for r in sensor_rows]
            valid = out["obs_mask"][0].bool()
            actual = list(zip(out["obs_indices"][0, valid].cpu().tolist(), out["obs_field_ids"][0, valid].cpu().tolist()))
            if actual != planned:
                raise RuntimeError("Canonical baseline sensor selection does not match the saved sensor plan.")
        out["obs_consistency_requested"] = obs_consistency
        out["obs_consistency_applied"] = applied
        return out

    def close(self) -> None:
        del self.model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _plan_tensors(rows: list[dict], device):
    ordered = sorted(rows, key=lambda r: int(r["sensor_order"]))
    idx = torch.tensor([[int(r["point_index"]) for r in ordered]], dtype=torch.long, device=device)
    fld = torch.tensor([[int(r["field_index"]) for r in ordered]], dtype=torch.long, device=device)
    return idx, fld


def resolve_checkpoint(run_dir: Path, checkpoint: str, allow_fallback: bool = False) -> tuple[Path | None, str]:
    requested = run_dir / f"{checkpoint}.pt"
    if requested.exists():
        return requested, "ok"
    if allow_fallback:
        other = run_dir / ("best.pt" if checkpoint == "last" else "last.pt")
        if other.exists():
            return other, "ok_fallback"
    return None, "missing checkpoint"


def inspect_artifacts(method: dict, condition: str, checkpoint="last", allow_fallback=False) -> dict:
    run_dir = ARCHIVE_DIR / method["directory"] / condition
    row = {"method": method["name"], "directory_alias": method["directory"], "condition": condition,
           "run_dir": str(run_dir), "checkpoint_requested": checkpoint, "checkpoint_path": "",
           "family": "", "backbone": "", "status": "ok", "detail": ""}
    if not run_dir.is_dir():
        row["status"] = "missing directory"; return row
    cfg_path = run_dir / "run_config.yaml"
    if not cfg_path.exists():
        row["status"] = "missing config"; return row
    ckpt, status = resolve_checkpoint(run_dir, checkpoint, allow_fallback)
    if ckpt is None:
        row["status"] = status; return row
    row["checkpoint_path"] = str(ckpt); row["checkpoint_used"] = ckpt.stem
    try:
        with cfg_path.open("r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        embedded = None
        try:
            embedded = baseline_lib.safe_torch_load(ckpt, map_location="cpu")
        except Exception:
            pass
        family = canonical.infer_checkpoint_family(ckpt, run_dir, embedded)
        row["family"] = family
        row["backbone"] = str(cfg.get("backbone", cfg.get("baseline_model", family)))
        _check_dependencies(cfg, embedded, run_dir)
    except MissingDependencyError as exc:
        row["status"] = "missing dependency"; row["detail"] = str(exc)
    except Exception as exc:
        row["status"] = "load error"; row["detail"] = f"metadata: {type(exc).__name__}: {exc}"
    return row


def _check_dependencies(cfg: dict, checkpoint: Any, run_dir: Path) -> None:
    if str(cfg.get("baseline_model", "")).lower() != "latent_fm" or int(cfg.get("training_stage", 1)) != 2:
        return
    resolved, recorded = canonical.resolve_latentfm_stage1_checkpoint(
        cfg=cfg, checkpoint=checkpoint if isinstance(checkpoint, dict) else None, run_dir=run_dir
    )
    if resolved is None:
        raise MissingDependencyError(
            f"latent-FM shared stage-0/stage-1 checkpoint not found: {recorded or '<none>'}; "
            f"also checked {run_dir.parent / 'Stage0'} and {run_dir.parent / 'Stage1'}"
        )


def load_model(method: dict, condition: str, *, checkpoint="last", allow_fallback=False,
               split="test", device: str | None = None, n_steps=2, ode_solver="euler") -> LoadedModel:
    row = inspect_artifacts(method, condition, checkpoint, allow_fallback)
    if row["status"] != "ok":
        exc = MissingDependencyError if row["status"] == "missing dependency" else RuntimeError
        raise exc(f"{row['status']}: {row['detail']}")
    ckpt = Path(row["checkpoint_path"]); run_dir = ckpt.parent
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    family = row["family"]
    if family == "pointcloud_ffm":
        cfg = canonical.load_run_config(run_dir)
        checkpoint_obj = torch.load(ckpt, map_location=dev, weights_only=False)
        cfg = canonical.resolve_effective_pointcloud_config(run_dir, checkpoint_obj, cfg)
        data_value = cfg.get("data")
        if not data_value:
            raise RuntimeError("PointCloudFFM config has no data path")
        data_path = canonical.resolve_input_path(str(data_value), label="Dataset", extra_base_dirs=[run_dir, canonical.DEMO_DIR])
        stats = run_dir / "dataset_stats.pt"
        dataset = TurbulentCombustionH5Dataset(str(data_path), split=split,
            train_ratio=float(cfg.get("train_ratio", .9)), seed=int(cfg.get("seed", 42)),
            time_stride=int(cfg.get("time_stride", 1)), field_names=cfg.get("FIELD_NAMES", cfg.get("field_names")),
            stats_path=str(stats) if stats.exists() else None)
        model = canonical.build_model(cfg, dataset).to(dev)
        state = checkpoint_obj.get("model", checkpoint_obj) if isinstance(checkpoint_obj, dict) else checkpoint_obj
        # neuraloperator checkpoints may serialize their FNO constructor
        # description as a literal non-tensor ``_metadata`` mapping alongside
        # the learned tensors. It is not a model parameter; remove only this
        # reserved entry while retaining strict validation for every weight.
        if isinstance(state, dict) and "_metadata" in state and not torch.is_tensor(state["_metadata"]):
            state = state.copy()
            state.pop("_metadata")
        model.load_state_dict(state, strict=True); model.eval()
    else:
        args = argparse.Namespace(baseline_model="auto", split=split, n_steps=n_steps, ode_solver=ode_solver)
        _, _, cfg, dataset, model, _, _, _ = canonical.load_baseline_context(args=args, checkpoint_arg=str(ckpt), device=dev)
    loaded = LoadedModel(method["name"], condition, family, row["backbone"], ckpt, ckpt.name, cfg, dataset, model, dev)
    print(f"[LOAD] {method['directory']} / {condition} | family={family} | backbone={loaded.backbone} | checkpoint={ckpt.name} | device={dev}")
    return loaded


def status_from_exception(exc: Exception) -> str:
    if isinstance(exc, MissingDependencyError):
        return "missing dependency"
    text = str(exc).lower()
    if "inference" in text or "sensor plan" in text:
        return "inference error"
    return "load error"
