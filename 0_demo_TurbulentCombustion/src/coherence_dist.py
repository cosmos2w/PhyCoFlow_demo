"""
Utilities for evaluating data-driven physical coherence terms.

Conventions
-----------
- One snapshot is represented as X in R^{N_pt x C} where C is the number of
  physical fields/channels and N_pt is the number of spatial points.
- All distances are computed in the *normalized field space* used by the model
  unless the caller explicitly chooses to denormalize first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Core 1D Wasserstein utilities
# -----------------------------------------------------------------------------

def empirical_w2_1d_sorted(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Exact empirical squared 2-Wasserstein distance in 1D.

    Both inputs are sorted and compared quantile-by-quantile. This is the main
    reason sliced Wasserstein is practical: after projection to 1D, the OT cost
    reduces to sorting.

    Args:
        x: Tensor of shape [N] or any shape that can be flattened.
        y: Tensor of shape [N] or any shape that can be flattened.

    Returns:
        Scalar tensor with the squared 2-Wasserstein distance.
    """
    x = x.reshape(-1)
    y = y.reshape(-1)

    if x.numel() != y.numel():
        raise ValueError(
            f"empirical_w2_1d_sorted expects equal sample counts, got {x.numel()} and {y.numel()}"
        )

    x_sorted = torch.sort(x)[0]
    y_sorted = torch.sort(y)[0]
    return torch.mean((x_sorted - y_sorted) ** 2)


def per_channel_w2(x_gen: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
    """
    Per-channel 1D Wasserstein anchors.

    Args:
        x_gen: [N_pt, C]
        x_ref: [N_pt, C]

    Returns:
        Tensor of shape [C] with one W2^2 value per channel.
    """
    if x_gen.shape != x_ref.shape:
        raise ValueError(f"Shape mismatch: {tuple(x_gen.shape)} vs {tuple(x_ref.shape)}")

    n_fields = x_gen.shape[1]
    vals = []
    for c in range(n_fields):
        vals.append(empirical_w2_1d_sorted(x_gen[:, c], x_ref[:, c]))
    return torch.stack(vals, dim=0)


# -----------------------------------------------------------------------------
# Projection utilities for joint channel-space discrepancies
# -----------------------------------------------------------------------------

def normalize_directions(theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalize projection directions to lie on the unit sphere."""
    return theta / theta.norm(dim=-1, keepdim=True).clamp_min(eps)


def orthogonality_penalty(theta: torch.Tensor) -> torch.Tensor:
    """
    Soft orthogonality penalty between projection directions.

    For C-channel data, at most C mutually orthogonal directions exist. This
    penalty is therefore used instead of hard orthogonalization when K > C.
    """
    theta = normalize_directions(theta)
    gram = theta @ theta.t()
    eye = torch.eye(theta.shape[0], device=theta.device, dtype=theta.dtype)
    return torch.mean((gram - eye) ** 2)


def project_channels(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """
    Project channel-state vectors onto a batch of directions.

    Args:
        x: [N_pt, C]
        theta: [K, C]

    Returns:
        projected: [N_pt, K]
    """
    return x @ theta.t()


# -----------------------------------------------------------------------------
# Joint Max-Sliced Wasserstein
# -----------------------------------------------------------------------------

def batched_max_swd(
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    num_directions: int = 4,
    n_iter: int = 5,
    lr_theta: float = 0.1,
    ortho_reg: float = 1e-2,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Batched Max-Sliced Wasserstein over the channel dimension.

    This optimizes a small set of projection directions to find linear
    combinations of channels on which the generated and reference snapshots are
    most discrepant. The output is both a scalar coherence penalty and a set of
    interpretable directions theta.

    Args:
        x_gen: [N_pt, C]
        x_ref: [N_pt, C]
        num_directions: requested K. It is clipped to C because only C mutually
            orthogonal directions exist in R^C.
        n_iter: number of ascent steps on theta.
        lr_theta: learning rate for the inner ascent loop.
        ortho_reg: soft orthogonality penalty strength.
        seed: optional random seed for reproducibility.

    Returns:
        Dict containing:
            - score: scalar tensor
            - per_direction_w2: [K_eff]
            - theta: [K_eff, C]
            - theta_init: [K_eff, C]
    """
    if x_gen.shape != x_ref.shape:
        raise ValueError(f"Shape mismatch: {tuple(x_gen.shape)} vs {tuple(x_ref.shape)}")

    n_fields = x_gen.shape[1]
    k_eff = min(int(num_directions), int(n_fields))

    if seed is not None:
        gen = torch.Generator(device=x_gen.device)
        gen.manual_seed(int(seed))
        theta = torch.randn(k_eff, n_fields, device=x_gen.device, dtype=x_gen.dtype, generator=gen)
    else:
        theta = torch.randn(k_eff, n_fields, device=x_gen.device, dtype=x_gen.dtype)

    theta = normalize_directions(theta)
    theta_init = theta.detach().clone()
    theta = theta.clone().detach().requires_grad_(True)

    for _ in range(int(n_iter)):
        theta_n = normalize_directions(theta)
        proj_gen = project_channels(x_gen, theta_n)   # [N_pt, K]
        proj_ref = project_channels(x_ref, theta_n)   # [N_pt, K]

        per_dir = torch.stack([
            empirical_w2_1d_sorted(proj_gen[:, k], proj_ref[:, k])
            for k in range(theta_n.shape[0])
        ])

        # Max-SW uses ascent on the discrepancy, with a small diversity penalty.
        objective = per_dir.mean() - ortho_reg * orthogonality_penalty(theta_n)
        grad = torch.autograd.grad(objective, theta, only_inputs=True)[0]

        with torch.no_grad():
            theta += lr_theta * grad
        theta.requires_grad_(True)

    theta_star = normalize_directions(theta.detach())
    proj_gen = project_channels(x_gen, theta_star)
    proj_ref = project_channels(x_ref, theta_star)
    per_dir = torch.stack([
        empirical_w2_1d_sorted(proj_gen[:, k], proj_ref[:, k])
        for k in range(theta_star.shape[0])
    ])

    return {
        "score": per_dir.mean(),
        "per_direction_w2": per_dir,
        "theta": theta_star,
        "theta_init": theta_init,
    }


# -----------------------------------------------------------------------------
# Pairwise 2D marginal diagnostics
# -----------------------------------------------------------------------------

def _random_unit_vectors_2d(n_proj: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    vec = torch.randn(n_proj, 2, device=device, dtype=dtype)
    return normalize_directions(vec)


def pairwise_2d_swd_matrix(
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    n_proj: int = 32,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """
    Approximate pairwise 2D sliced Wasserstein matrix.

    For each channel pair (i, j), project the 2D joint marginal onto random 1D
    directions and average the exact 1D W2^2 values.

    Args:
        x_gen: [N_pt, C]
        x_ref: [N_pt, C]
        n_proj: number of random 2D slice directions per pair.

    Returns:
        Symmetric matrix [C, C]. Diagonal is zero.
    """
    if x_gen.shape != x_ref.shape:
        raise ValueError(f"Shape mismatch: {tuple(x_gen.shape)} vs {tuple(x_ref.shape)}")

    c = x_gen.shape[1]
    mat = torch.zeros(c, c, device=x_gen.device, dtype=x_gen.dtype)

    for i in range(c):
        for j in range(i + 1, c):
            x_pair_gen = x_gen[:, [i, j]]
            x_pair_ref = x_ref[:, [i, j]]

            if seed is not None:
                gen = torch.Generator(device=x_gen.device)
                gen.manual_seed(int(seed) + i * 1009 + j)
                directions = torch.randn(n_proj, 2, device=x_gen.device, dtype=x_gen.dtype, generator=gen)
                directions = normalize_directions(directions)
            else:
                directions = _random_unit_vectors_2d(n_proj, x_gen.device, x_gen.dtype)

            proj_gen = x_pair_gen @ directions.t()  # [N_pt, n_proj]
            proj_ref = x_pair_ref @ directions.t()  # [N_pt, n_proj]

            vals = torch.stack([
                empirical_w2_1d_sorted(proj_gen[:, p], proj_ref[:, p])
                for p in range(n_proj)
            ])
            score = vals.mean()
            mat[i, j] = score
            mat[j, i] = score

    return mat


# -----------------------------------------------------------------------------
# Main global distributional coherence computation
# -----------------------------------------------------------------------------

@dataclass
class GlobalDistConfig:
    lambda_marg: float = 1.0
    lambda_joint: float = 1.0
    num_directions: int = 4
    n_iter_theta: int = 5
    lr_theta: float = 0.1
    ortho_reg: float = 1e-2
    n_proj_pairwise: int = 32
    include_pairwise: bool = True
    seed: Optional[int] = None


def compute_global_distribution_coherence(
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    cfg: Optional[GlobalDistConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute the global distributional coherence package.

    This implements the recommended default design:
      - per-channel 1D Wasserstein anchors
      - batched Max-SW over full channel-state vectors
      - optional pairwise 2D marginal diagnostics

    Args:
        x_gen: [N_pt, C]
        x_ref: [N_pt, C]
        cfg: configuration dataclass

    Returns:
        Dict with scalar summary and diagnostic tensors.
    """
    cfg = cfg or GlobalDistConfig()

    marg = per_channel_w2(x_gen, x_ref)                              # [C]
    joint = batched_max_swd(
        x_gen=x_gen,
        x_ref=x_ref,
        num_directions=cfg.num_directions,
        n_iter=cfg.n_iter_theta,
        lr_theta=cfg.lr_theta,
        ortho_reg=cfg.ortho_reg,
        seed=cfg.seed,
    )

    out: Dict[str, torch.Tensor] = {
        "mode_score": cfg.lambda_marg * marg.mean() + cfg.lambda_joint * joint["score"],
        "marginal_score": marg.mean(),
        "per_channel_w2": marg,
        "joint_score": joint["score"],
        "per_direction_w2": joint["per_direction_w2"],
        "theta": joint["theta"],
        "theta_init": joint["theta_init"],
    }

    if cfg.include_pairwise:
        pairwise = pairwise_2d_swd_matrix(
            x_gen=x_gen,
            x_ref=x_ref,
            n_proj=cfg.n_proj_pairwise,
            seed=cfg.seed,
        )
        out["pairwise_2d_swd"] = pairwise
        c = pairwise.shape[0]
        denom = max(c * (c - 1), 1)
        out["pairwise_mean"] = pairwise.sum() / denom

    return out


# -----------------------------------------------------------------------------
# Registry for future coherence terms
# -----------------------------------------------------------------------------

COHERENCE_REGISTRY: Dict[str, Callable[..., Dict[str, torch.Tensor]]] = {
    "global_dist": compute_global_distribution_coherence,
}


def compute_coherence(mode: str, x_gen: torch.Tensor, x_ref: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
    """
    Dispatch coherence computation by mode name.
    """
    if mode not in COHERENCE_REGISTRY:
        raise ValueError(f"Unknown coherence mode '{mode}'. Available: {list(COHERENCE_REGISTRY.keys())}")
    return COHERENCE_REGISTRY[mode](x_gen, x_ref, **kwargs)
