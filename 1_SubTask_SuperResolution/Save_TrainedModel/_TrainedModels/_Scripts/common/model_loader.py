"""Super-resolution checkpoint adapters with no duplicated architectures."""
from __future__ import annotations
import gc
import hashlib
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
from .dataset_loader import build_run_dataset, read_run_config, resolve_run_config_path
from .recipe_registry import resolve_recipe_dir

os.environ.setdefault("KEOPS_CACHE_FOLDER", "/tmp/phycoflow_keops")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/phycoflow_matplotlib")
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evaluate_ffm as ffm_eval  # noqa: E402
import model_baseline as baseline_lib  # noqa: E402


def checkpoint_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def resolve_checkpoint(run_dir: Path, kind: str, allow_fallback=False):
    requested = run_dir / f"{kind}.pt"
    if requested.exists():
        return requested, "ok"
    if allow_fallback:
        other = run_dir / ("best.pt" if kind == "last" else "last.pt")
        if other.exists():
            return other, "ok_fallback"
    return None, "missing_checkpoint"


def inspect_artifacts(model: dict, recipe_key: str, recipe_spec: dict, checkpoint="last", allow_fallback=False) -> dict:
    model_dir = ARCHIVE_DIR / model["directory"]
    run_dir = resolve_recipe_dir(model_dir, recipe_key, recipe_spec)
    row = {
        "model": model["key"], "model_label": model["label"], "model_directory": model["directory"],
        "recipe": recipe_key, "recipe_directory": run_dir.name, "run_dir": str(run_dir),
        "checkpoint_requested": checkpoint, "checkpoint_path": "", "family": "", "backbone": "",
        "status": "ok", "detail": "",
    }
    if not run_dir.is_dir():
        row["status"] = "missing_directory"; return row
    try:
        config_path = resolve_run_config_path(run_dir)
    except FileNotFoundError:
        row["status"] = "missing_config"; return row
    ckpt, ckpt_status = resolve_checkpoint(run_dir, checkpoint, allow_fallback)
    if ckpt is None:
        row["status"] = ckpt_status; return row
    raw, cfg = read_run_config(run_dir)
    family = "deterministic" if raw.get("baseline_model") else "pointcloud_ffm"
    row.update({
        "checkpoint_path": str(ckpt), "checkpoint_used": ckpt.stem,
        "config_path": str(config_path), "config_format": config_path.suffix.lstrip("."),
        "checkpoint_mtime_ns": ckpt.stat().st_mtime_ns, "family": family,
        "backbone": str(cfg.get("backbone", raw.get("baseline_model", ""))),
        "dataset_mode": cfg.get("dataset_mode", ""), "dataset_name": cfg.get("pdebench_dataset_name", ""),
        "eval_resolution": cfg.get("eval_resolution", "H"),
    })
    if ckpt_status == "ok_fallback":
        row["detail"] = "checkpoint_fallback"
    return row


@dataclass
class LoadedModel:
    model_key: str
    model_label: str
    recipe_key: str
    family: str
    backbone: str
    checkpoint_path: Path
    config: dict
    dataset: Any
    manifest: dict
    model: torch.nn.Module
    device: torch.device
    baseline_bundle: Any = None

    @torch.inference_mode()
    def reconstruct(self, snapshot: int, sensor_rows: list[dict], *, n_steps: int, ode_solver: str,
                    consistency_mode: str, generation_seed: int) -> dict[str, Any]:
        torch.manual_seed(int(generation_seed)); np.random.seed(int(generation_seed) & 0xFFFFFFFF)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(generation_seed))
        sample = self.dataset[int(snapshot)]
        coords = sample["coords"].unsqueeze(0).to(self.device)
        truth = sample["fields"].unsqueeze(0).to(self.device)
        ordered = sorted(sensor_rows, key=lambda row: int(row["sensor_order"]))
        indices = torch.tensor([[int(row["point_index"]) for row in ordered]], dtype=torch.long, device=self.device)
        fields = torch.zeros_like(indices)
        obs_coords = coords[:, indices[0]]
        obs_values = truth[:, indices[0], :1]
        obs_mask = torch.ones(indices.shape, dtype=coords.dtype, device=self.device)
        applied = "native_inference"
        if self.family == "pointcloud_ffm":
            kwargs = {
                "coords": coords, "obs_coords": obs_coords, "obs_values": obs_values,
                "obs_mask": obs_mask, "obs_field_ids": fields, "n_steps": int(n_steps),
                "clamp_indices": indices, "ode_solver": ode_solver,
                "obs_consistency_mode": consistency_mode,
            }
            signature = inspect.signature(self.model.sample)
            call = {k: v for k, v in kwargs.items() if k in signature.parameters}
            if "obs_consistency_mode" not in call:
                applied = "unsupported"
            else:
                applied = consistency_mode
            recon = self.model.sample(**call)
        else:
            recon = self.model(coords, obs_coords, obs_values, obs_mask, fields)
        return {
            "coords": coords[0].cpu().numpy(), "coords_phys": sample["coords_raw"].cpu().numpy(),
            "truth_norm": truth[0].cpu().numpy(), "recon_norm": recon[0].cpu().numpy(),
            "obs_indices": indices[0].cpu().numpy(), "obs_values_norm": obs_values[0, :, 0].cpu().numpy(),
            "obs_consistency_applied": applied,
        }

    def close(self):
        self.model.to("cpu")
        del self.model
        if self.baseline_bundle is not None:
            del self.baseline_bundle
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_model(model_spec: dict, recipe_key: str, recipe_spec: dict, *, checkpoint="last", allow_fallback=False,
               split="test", eval_resolution="H", device: str | None = None) -> LoadedModel:
    row = inspect_artifacts(model_spec, recipe_key, recipe_spec, checkpoint, allow_fallback)
    if row["status"] != "ok":
        raise RuntimeError(row["status"])
    run_dir = Path(row["run_dir"]); ckpt_path = Path(row["checkpoint_path"])
    dev_name = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if dev_name == "auto":
        dev_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    dev = torch.device(dev_name)
    dataset, _, manifest, _ = build_run_dataset(run_dir, model_spec["key"], recipe_key, split=split, eval_resolution=eval_resolution)
    raw, flat = read_run_config(run_dir)
    checkpoint_obj = torch.load(ckpt_path, map_location=dev, weights_only=False)
    bundle = None
    if row["family"] == "pointcloud_ffm":
        cfg = ffm_eval._normalize_eval_config(raw)
        cfg = ffm_eval._apply_checkpoint_fourier_pe_to_cfg(cfg, checkpoint_obj)
        model = ffm_eval._build_model(cfg, dataset).to(dev)
        state = checkpoint_obj.get("model", checkpoint_obj)
        model.load_state_dict(state, strict=True)
    else:
        cfg = baseline_lib.validate_and_normalize_config(raw)
        adapter = baseline_lib.get_baseline_adapter(str(raw["baseline_model"]).lower())
        bundle = adapter.build_for_training(cfg, dev, run_dir, dataset, dataset)
        adapter.load_checkpoint(bundle, checkpoint_obj)
        model = bundle.model
    model.eval()
    print(f"[LOAD] {model_spec['label']} / {recipe_key} | family={row['family']} | "
          f"backbone={row['backbone']} | checkpoint={ckpt_path.name} | eval={eval_resolution}")
    return LoadedModel(model_spec["key"], model_spec["label"], recipe_key, row["family"], row["backbone"],
                       ckpt_path, raw, dataset, manifest, model, dev, bundle)


def status_from_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if "out of memory" in text:
        return "out_of_memory"
    if "missing" in text:
        return text.replace(" ", "_")[:80]
    return "load_or_inference_error"
