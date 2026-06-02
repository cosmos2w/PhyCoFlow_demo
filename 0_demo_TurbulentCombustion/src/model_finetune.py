"""
Utilities for PointCloudFFM post-training fine-tuning.

Quick usage:
    from model_finetune import load_pretrained_ffm, clone_model, set_trainable_scope

    source_run = find_source_run_dir(demo_dir, source_Demo_Num=2)
    model, source_cfg, ckpt = load_pretrained_ffm(source_run, "best", dataset, device)

This module intentionally keeps architecture reconstruction delegated to
``evaluate_ffm._build_model``.  That keeps base checkpoints and RAM checkpoints
on the same loading path and avoids a second, subtly different model builder.
"""

from __future__ import annotations

import copy
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import yaml


def find_source_run_dir(
    demo_dir,
    source_run_dir=None,
    source_Demo_Num=None,
    save_dir_hint="Save_TrainedModel/ffm_tc_pointcloud",
):
    """
    Resolve the pretrained PointCloudFFM run used as the RAM reference policy.

    If ``source_run_dir`` is explicit, it is returned after existence checking.
    Otherwise the newest directory matching
    ``ffm_tc_pointcloud_DemoN<source_Demo_Num>_*`` is selected.
    """
    demo_dir = Path(demo_dir)
    if source_run_dir not in (None, ""):
        path = Path(source_run_dir)
        if not path.is_absolute():
            path = demo_dir / path
        if not path.exists():
            raise FileNotFoundError(f"source_run_dir does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"source_run_dir is not a directory: {path}")
        return path

    if source_Demo_Num is None:
        raise ValueError(
            "source_Demo_Num is required when source_run_dir is not provided."
        )

    save_hint = Path(save_dir_hint)
    save_root = save_hint.parent if save_hint.is_absolute() else demo_dir / save_hint.parent
    pattern = f"{save_hint.name}_DemoN{int(source_Demo_Num)}_*"

    if not save_root.exists():
        raise FileNotFoundError(
            f"Cannot find pretrained source root {save_root}; expected runs like {pattern}"
        )

    candidates = sorted(path for path in save_root.glob(pattern) if path.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"No pretrained source run matching {pattern} under {save_root}. "
            "Set source_run_dir in the RAM config to an existing run directory."
        )
    return candidates[-1]


def load_source_config(source_run_dir: Path) -> dict:
    """
    Load the architecture/config payload saved by a pretrained source run.

    ``run_config.yaml`` is preferred because it preserves YAML values exactly;
    ``args.json`` is kept as a fallback for older runs.
    """
    source_run_dir = Path(source_run_dir)
    yaml_path = source_run_dir / "run_config.yaml"
    json_path = source_run_dir / "args.json"

    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as handle:
            return json.load(handle) or {}

    raise FileNotFoundError(
        f"Could not find run_config.yaml or args.json in source run: {source_run_dir}"
    )


def build_ffm_model_from_config(cfg: dict, dataset) -> torch.nn.Module:
    """
    Rebuild a PointCloudFFM/FNOFFM from a saved config and dataset metadata.

    The actual construction logic lives in ``evaluate_ffm`` so RAM fine-tuning
    and standalone evaluation cannot drift apart.
    """
    from evaluate_ffm import _build_model, _normalize_eval_config

    return _build_model(_normalize_eval_config(dict(cfg)), dataset)


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except pickle.UnpicklingError:
        return torch.load(path, map_location="cpu", weights_only=False)


def _resolve_checkpoint_path(source_run_dir: Path, checkpoint: str) -> Path:
    ckpt = Path(str(checkpoint))
    if ckpt.suffix != ".pt":
        ckpt = ckpt.with_suffix(".pt")
    if not ckpt.is_absolute():
        ckpt = source_run_dir / ckpt
    if not ckpt.exists():
        raise FileNotFoundError(f"Source checkpoint not found: {ckpt}")
    return ckpt


def load_pretrained_ffm(
    source_run_dir: Path,
    checkpoint: str,
    dataset,
    device,
) -> Tuple[nn.Module, dict, dict]:
    """
    Load a pretrained PointCloudFFM/FNOFFM checkpoint and return it on ``device``.
    """
    source_run_dir = Path(source_run_dir)
    source_cfg = load_source_config(source_run_dir)
    model = build_ffm_model_from_config(source_cfg, dataset)

    ckpt_path = _resolve_checkpoint_path(source_run_dir, checkpoint)
    ckpt = _load_checkpoint(ckpt_path)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Some serialized state dicts carry this bookkeeping key literally.
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)
        state_dict.pop("_metadata", None)

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    return model, source_cfg, ckpt


def clone_model(model, device):
    """Deep-copy a model and move the copy to ``device``."""
    return copy.deepcopy(model).to(device)


def sync_params(source, target):
    """Synchronize all parameters and buffers from ``source`` into ``target``."""
    target.load_state_dict(source.state_dict(), strict=True)


@torch.no_grad()
def update_ema_params(source, target, decay: float):
    """
    EMA update for model parameters, with buffers copied exactly.

    RAM uses a lagged old policy for endpoint sampling/targets and a smoother
    evaluation policy for checkpoints.
    """
    decay = float(decay)
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.mul_(decay).add_(src_param.detach().data, alpha=1.0 - decay)

    for src_buffer, tgt_buffer in zip(source.buffers(), target.buffers()):
        tgt_buffer.data.copy_(src_buffer.detach().data)


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _lookup_attr_path(root, path: str):
    obj = root
    for name in path.split("."):
        if not hasattr(obj, name):
            return None
        obj = getattr(obj, name)
    return obj


def _unfreeze_obj(obj) -> None:
    if isinstance(obj, nn.Parameter):
        obj.requires_grad_(True)
    elif isinstance(obj, nn.Module):
        for param in obj.parameters():
            param.requires_grad_(True)


def set_trainable_scope(model: nn.Module, mode: str) -> list[nn.Parameter]:
    """
    Select which policy parameters RAM is allowed to update.

    ``head_glres`` freezes the full model, then unfreezes only the velocity head
    and the lightweight GL-residual adaptation modules used by
    ``gather_mode='topk_rbf_glres'`` when those modules exist.
    """
    mode = str(mode or "head_glres").strip().lower()
    host = _unwrap_model(model)

    if mode == "all":
        for param in host.parameters():
            param.requires_grad_(True)
    elif mode == "head_glres":
        for param in host.parameters():
            param.requires_grad_(False)

        trainable_paths = [
            "model.head",
            "model.coarse_head",
            "model.coarse_film",
            "model.coarse_scale",
            "model.query_readout_in",
            "model.query_latent_readout",
            "model.query_readout_out",
            "model.query_readout_scale",
            "model.sensor_importance",
            "model.sensor_importance_scale",
        ]
        found = []
        for path in trainable_paths:
            obj = _lookup_attr_path(host, path)
            if obj is None:
                continue
            _unfreeze_obj(obj)
            found.append(path)
        print(f"[*] RAM head_glres trainable modules found: {found}")
    else:
        raise ValueError("finetune_mode must be 'head_glres' or 'all'.")

    params = [param for param in host.parameters() if param.requires_grad]
    trainable_count = sum(param.numel() for param in params)
    total_count = sum(param.numel() for param in host.parameters())
    print(
        f"[*] RAM trainable parameters: {trainable_count:,} / {total_count:,} "
        f"({trainable_count / max(total_count, 1):.2%})"
    )
    if not params:
        raise RuntimeError(f"No trainable parameters selected for finetune_mode={mode!r}.")
    return params
