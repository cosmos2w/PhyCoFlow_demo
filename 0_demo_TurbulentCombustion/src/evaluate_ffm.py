import argparse
import json
import os
import re
from pathlib import Path
from typing import Optional

import torch
import yaml
import pickle
import numpy as np

from helpers import TurbulentCombustionH5Dataset, visualize_reconstruction

from Model import (
    ConditionalPointMLPRBF,
    ConditionalPointPerceiver,
    PointCloudFFM,
)
try:
    from Model import FNO, FNOFFM
except ImportError:
    FNO = None
    FNOFFM = None

def parse_args():
    p = argparse.ArgumentParser("Standalone evaluator for trained FFM models.")
    p.add_argument("--Demo-Num", dest="Demo_Num", type=int, required=True, 
                   help="Demo ID to recover.")
    p.add_argument("--demo-root", type=str, default=".", 
                   help="Project/demo root directory.")
    p.add_argument("--split", type=str, default="test", 
                   choices=["train", "val", "test"])
    p.add_argument("--snapshot-index", type=int, default=0, 
                   help="Index within the selected split.")
    
    p.add_argument("--vis-cond-fields", type=int, nargs="+", default=None,
                   help="Override visualization cond_fields. Defaults to YAML vis_cond_fields or cond_fields.")
    p.add_argument("--vis-n-obs-list", type=int, nargs="+", default=None,
                   help="Override visualization n_obs list. Defaults to YAML vis_n_obs_list or n_obs_max_list.")
    p.add_argument("--checkpoint", type=str, default="best", choices=["best", "last"],
                   help="Which checkpoint to load from the recovered run directory.")
    p.add_argument("--n-steps-generation", type=int, default = 8,
                   help="Override generation steps. Defaults to YAML n_steps_generation if present.")
    p.add_argument("--device", type=str, default=None, help="e.g. cuda:0 or cpu")
    
    return p.parse_args()

class IIDGaussianPrior(torch.nn.Module):
    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels, device=coords.device, dtype=coords.dtype)


class RFFGaussianPrior(torch.nn.Module):
    def __init__(self, coord_dim: int = 3, n_features: int = 256, lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega", torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * np.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return np.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat, device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)

def _extract_timestamp(path: Path) -> Optional[str]:
    m = re.search(r"DemoN(\d+)_(\d{8}_\d{6})", path.name)
    if m is None:
        m = re.search(r"demo_N(\d+)_(\d{8}_\d{6})", path.name)
    return m.group(2) if m else None


def _find_latest_yaml(cfg_dir: Path, demo_num: int) -> Path:
    pattern = f"config_pointcloud_ffm_DemoN{demo_num}_*.yaml"
    candidates = sorted(cfg_dir.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No config backup found for Demo_Num={demo_num} in {cfg_dir}"
        )

    def _sort_key(p: Path):
        ts = _extract_timestamp(p)
        return ts if ts is not None else p.stat().st_mtime

    candidates = sorted(candidates, key=_sort_key)
    return candidates[-1]


def _normalize_eval_config(cfg: dict) -> dict:
    cfg = dict(cfg)

    # Backward-compatible defaults
    if cfg.get("cond_fields") is None:
        cfg["cond_fields"] = [cfg.get("cond_field", 2)]
    if cfg.get("n_obs_min_list") is None:
        cfg["n_obs_min_list"] = [cfg.get("n_obs_min", 64)]
    if cfg.get("n_obs_max_list") is None:
        cfg["n_obs_max_list"] = [cfg.get("n_obs_max", 256)]

    if cfg.get("vis_cond_fields") in (None, ""):
        cfg["vis_cond_fields"] = list(cfg["cond_fields"])
    if cfg.get("vis_n_obs_list") in (None, ""):
        cfg["vis_n_obs_list"] = list(cfg["n_obs_max_list"])

    if cfg.get("backbone") is None:
        cfg["backbone"] = "mlp_rbf"

    return cfg


def _build_prior(cfg: dict):
    if cfg.get("prior", "rff") == "iid":
        return IIDGaussianPrior()
    return RFFGaussianPrior(
        coord_dim=3,
        n_features=cfg.get("rff_features", 256),
        lengthscale=cfg.get("rff_lengthscale", 0.15),
    )


def _build_model(cfg: dict, dataset) -> torch.nn.Module:
    prior = _build_prior(cfg)
    backbone_name = cfg.get("backbone", "mlp_rbf")

    if backbone_name == "perceiver":
        backbone = ConditionalPointPerceiver(
            n_fields=dataset.num_fields,
            coord_dim=3,
            latent_dim=cfg.get("latent_dim", 256),
            num_latents=cfg.get("num_latents", 128),
            num_heads=cfg.get("num_heads", 8),
            num_latent_blocks=cfg.get("num_latent_blocks", 4),
            field_embed_dim=cfg.get("field_embed_dim", 128),
            ff_mult=cfg.get("ff_mult", 4),
            attn_dropout=cfg.get("attn_dropout", 0.0),
            mlp_dropout=cfg.get("mlp_dropout", 0.0),
            decode_chunk_size=cfg.get("decode_chunk_size", 4096),
            share_query_proj=cfg.get("share_query_proj", False),
        )
        model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    if backbone_name == "fno":
        if FNO is None or FNOFFM is None:
            raise RuntimeError("YAML says backbone='fno' but FNO/FNOFFM are not available in Model.py")
        Num_x = cfg.get("Num_x", None)
        Num_y = cfg.get("Num_y", None)
        if Num_x is None or Num_y is None:
            raise ValueError("FNO evaluation requires Num_x and Num_y in YAML.")
        backbone = FNO(
            n_fields=dataset.num_fields,
            Num_x=Num_x,
            Num_y=Num_y,
            n_modes_x=cfg.get("fno_modes_x", 32),
            n_modes_y=cfg.get("fno_modes_y", 8),
            hidden_channels=cfg.get("fno_hidden_channels", 64),
            n_layers=cfg.get("fno_n_layers", 4),
        )
        model = FNOFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
        return model

    backbone = ConditionalPointMLPRBF(
        n_fields=dataset.num_fields,
        coord_dim=3,
        hidden_dim=cfg.get("hidden_dim", 256),
        cond_dim=cfg.get("cond_dim", 128),
        field_embed_dim=cfg.get("field_embed_dim", 128),
        rbf_sigma=cfg.get("rbf_sigma", 0.05),
    )
    model = PointCloudFFM(backbone, prior, sigma_min=cfg.get("sigma_min", 1e-4))
    return model


def main():
    args = parse_args()

    demo_root = Path(args.demo_root).resolve()
    cfg_dir = demo_root / "Save_config" / "pointcloud_ffm"

    try:
        yaml_path = _find_latest_yaml(cfg_dir, args.Demo_Num)
    except FileNotFoundError as e:
        print(f"[Warning: !] {e}")
        raise SystemExit(1)

    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    cfg = _normalize_eval_config(cfg)

    train_timestamp = _extract_timestamp(yaml_path)
    if train_timestamp is None:
        print(f"[Warning: !] Could not parse timestamp from config filename: {yaml_path.name}")
        raise SystemExit(1)

    save_dir_cfg = Path(cfg.get("save_dir", "Save_TrainedModel/ffm_tc_pointcloud"))
    model_root = demo_root / save_dir_cfg.parent / f"{save_dir_cfg.name}_DemoN{args.Demo_Num}_{train_timestamp}"

    if not model_root.exists():
        print(f"[Warning: !] Matching model directory not found: {model_root}")
        raise SystemExit(1)

    ckpt_path = model_root / f"{args.checkpoint}.pt"
    if not ckpt_path.exists():
        print(f"[Warning: !] Checkpoint not found: {ckpt_path}")
        raise SystemExit(1)

    device = torch.device(args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu"))

    dataset = TurbulentCombustionH5Dataset(
        cfg.get("data", "Dataset/Merged_CH4COTU1P.h5"),
        split=args.split,
        train_ratio=cfg.get("train_ratio", 0.9),
        seed=cfg.get("seed", 42),
        time_stride=cfg.get("time_stride", 1),
        stats_path=str(model_root / "dataset_stats.pt"),
    )

    try:
        model = _build_model(cfg, dataset).to(device)
    except Exception as e:
        print(f"[Warning: !] Model construction failed: {e}")
        raise SystemExit(1)

    # ckpt = torch.load(ckpt_path, map_location=device)
    try:
        ckpt = torch.load(ckpt_path, map_location=device)
    except pickle.UnpicklingError:
        print("[Warning: !] Restricted torch.load failed; retrying with weights_only=False "
            "for a trusted local checkpoint.")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Some checkpoints may carry "_metadata" as a literal key after serialization.
    # It is not a model parameter and must be removed before load_state_dict(...).
    if isinstance(state_dict, dict) and "_metadata" in state_dict:
        state_dict = dict(state_dict)   # make a plain mutable copy
        state_dict.pop("_metadata", None)

    try:
        model.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"[Warning: !] Checkpoint is incompatible with the reconstructed model: {e}")
        raise SystemExit(1)

    model.eval()

    vis_cond_fields = args.vis_cond_fields if args.vis_cond_fields is not None else cfg["vis_cond_fields"]
    vis_n_obs_list = args.vis_n_obs_list if args.vis_n_obs_list is not None else cfg["vis_n_obs_list"]
    n_steps_generation = (
        args.n_steps_generation if args.n_steps_generation is not None
        else cfg.get("n_steps_generation", 100)
    )
    print(f'\nResults are generated from n_steps={n_steps_generation}\n')

    eval_timestamp = torch.tensor([])  # dummy to avoid importing datetime twice
    from datetime import datetime
    eval_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_dir = demo_root / "Save_reconstruction_files" / "ForOfflineEvaluation" / f"eval_N{args.Demo_Num}_{eval_timestamp}_from_{train_timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = visualize_reconstruction(
        model=model,
        dataset=dataset,
        epoch=int(ckpt.get("epoch", 0)) if isinstance(ckpt, dict) else 0,
        device=device,
        save_dir=str(out_dir),
        cond_fields=vis_cond_fields,
        n_obs=vis_n_obs_list,
        n_steps=n_steps_generation,
        snapshot_index=args.snapshot_index,
        file_prefix=f"snapshot_{args.snapshot_index:04d}",
        save_metrics_json=True,
    )

    summary = {
        "demo_num": int(args.Demo_Num),
        "yaml_path": str(yaml_path),
        "model_root": str(model_root),
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "snapshot_index": int(args.snapshot_index),
        "vis_cond_fields": [int(v) for v in vis_cond_fields],
        "vis_n_obs_list": [int(v) for v in vis_n_obs_list],
        "n_steps_generation": int(n_steps_generation),
        "metrics": metrics,
    }

    with open(out_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("[*] Evaluation finished.")
    print(f"[*] YAML      : {yaml_path}")
    print(f"[*] Checkpoint: {ckpt_path}")
    print(f"[*] Output dir : {out_dir}")
    print(f"[*] Metrics    : {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    main()