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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
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


def empirical_w2_1d_columns(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Vectorized empirical squared 2-Wasserstein distance for column batches.

    Args:
        x, y: tensors with shape [N, K] or [N]. Shapes must match.

    Returns:
        Tensor [K] for 2D inputs. For 1D inputs, returns a length-1 tensor that
        remains scalar-compatible via reshape/squeeze by callers.
    """
    if x.shape != y.shape:
        raise ValueError(f"empirical_w2_1d_columns expects equal shapes, got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.ndim == 1:
        x = x[:, None]
        y = y[:, None]
    elif x.ndim != 2:
        raise ValueError(f"empirical_w2_1d_columns expects [N] or [N, K], got shape {tuple(x.shape)}")

    x_sorted = torch.sort(x, dim=0)[0]
    y_sorted = torch.sort(y, dim=0)[0]
    return torch.mean((x_sorted - y_sorted) ** 2, dim=0)


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

    return empirical_w2_1d_columns(x_gen, x_ref)


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


def make_projection_bank(
    n_fields: int,
    num_directions: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: Optional[int] = None,
    include_axes: bool = False,
    qmc: bool = True,
) -> torch.Tensor:
    """
    Build a deterministic bank of unit directions in channel space.

    Canonical axes are included first when requested. Remaining directions use a
    Sobol quasi-random normal construction when available, with a seeded random
    fallback for older torch builds or unsupported devices.
    """
    if n_fields <= 0:
        raise ValueError(f"n_fields must be positive, got {n_fields}")
    if num_directions <= 0:
        raise ValueError(f"num_directions must be positive, got {num_directions}")
    effective_seed = 0 if seed is None else int(seed)

    parts: List[torch.Tensor] = []
    if include_axes:
        n_axes = min(int(num_directions), int(n_fields))
        parts.append(torch.eye(n_fields, device=device, dtype=dtype)[:n_axes])

    remaining = int(num_directions) - sum(p.shape[0] for p in parts)
    if remaining > 0:
        z: Optional[torch.Tensor] = None
        if qmc:
            try:
                sobol = torch.quasirandom.SobolEngine(
                    dimension=int(n_fields),
                    scramble=True,
                    seed=effective_seed,
                )
                u = sobol.draw(remaining).to(device=device, dtype=dtype)
                eps = torch.finfo(dtype).eps if dtype.is_floating_point else 1e-7
                u = u.clamp(float(eps), 1.0 - float(eps))
                z = math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)
            except Exception:
                z = None

        if z is None:
            gen = torch.Generator(device=device)
            gen.manual_seed(effective_seed)
            z = torch.randn(remaining, n_fields, device=device, dtype=dtype, generator=gen)
        parts.append(z)

    theta = torch.cat(parts, dim=0) if len(parts) > 1 else parts[0]
    return normalize_directions(theta[: int(num_directions)])


def fixed_bank_topk_swd(
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    num_directions: int = 64,
    top_frac: float = 0.10,
    seed: Optional[int] = None,
    include_axes: bool = False,
    exclude_axes_from_score: bool = False,
    qmc: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Fixed-bank robust sliced-Wasserstein joint score.

    Projects snapshots onto a deterministic direction bank and averages the
    largest top_frac of projected empirical W2 values as a stable approximation
    of worst projection discrepancies.
    """
    if x_gen.shape != x_ref.shape:
        raise ValueError(f"Shape mismatch: {tuple(x_gen.shape)} vs {tuple(x_ref.shape)}")

    n_fields = x_gen.shape[1]
    theta = make_projection_bank(
        n_fields=n_fields,
        num_directions=int(num_directions),
        device=x_gen.device,
        dtype=x_gen.dtype,
        seed=seed,
        include_axes=include_axes,
        qmc=qmc,
    )
    proj_gen = project_channels(x_gen, theta)
    proj_ref = project_channels(x_ref, theta)
    per_dir = empirical_w2_1d_columns(proj_gen, proj_ref)

    axis_mask = torch.zeros(per_dir.shape, device=x_gen.device, dtype=torch.bool)
    if include_axes:
        n_axes = min(int(num_directions), int(n_fields))
        axis_mask[:n_axes] = True

    score_mask = torch.ones_like(axis_mask, dtype=torch.bool)
    if exclude_axes_from_score:
        score_mask = ~axis_mask

    eligible_indices = torch.nonzero(score_mask, as_tuple=False).reshape(-1)
    if eligible_indices.numel() > 0:
        eligible_vals = per_dir[eligible_indices]
        k_top = max(1, min(eligible_vals.numel(), int(math.ceil(float(top_frac) * eligible_vals.numel()))))
        top_vals, top_local_indices = torch.topk(eligible_vals, k=k_top, largest=True, sorted=True)
        top_indices = eligible_indices[top_local_indices]
        score = top_vals.mean()
    else:
        top_vals = per_dir.new_empty((0,))
        top_indices = torch.empty((0,), device=x_gen.device, dtype=torch.long)
        score = per_dir.new_tensor(0.0)

    return {
        "score": score,
        "per_direction_w2": per_dir,
        "theta": theta,
        "top_indices": top_indices,
        "top_values": top_vals,
        "axis_direction_mask": axis_mask,
        "joint_score_mask": score_mask,
        "exclude_axes_from_score": torch.tensor(bool(exclude_axes_from_score), device=x_gen.device),
        "top_frac": torch.tensor(float(top_frac), device=x_gen.device, dtype=x_gen.dtype),
    }


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

        per_dir = empirical_w2_1d_columns(proj_gen, proj_ref)

        # Max-SW uses ascent on the discrepancy, with a small diversity penalty.
        objective = per_dir.mean() - ortho_reg * orthogonality_penalty(theta_n)
        grad = torch.autograd.grad(objective, theta, only_inputs=True)[0]

        with torch.no_grad():
            theta += lr_theta * grad
        theta.requires_grad_(True)

    theta_star = normalize_directions(theta.detach())
    proj_gen = project_channels(x_gen, theta_star)
    proj_ref = project_channels(x_ref, theta_star)
    per_dir = empirical_w2_1d_columns(proj_gen, proj_ref)

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

            vals = empirical_w2_1d_columns(proj_gen, proj_ref)
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
    joint_method: str = "topk_swd"
    joint_top_frac: float = 0.10
    joint_qmc: bool = True
    include_axes: bool = False
    lambda_pairwise: float = 0.25
    include_pairwise_in_score: bool = True
    exclude_axis_projections_when_marginal_included: bool = True


def compute_global_distribution_coherence(
    x_gen: torch.Tensor,
    x_ref: torch.Tensor,
    cfg: Optional[GlobalDistConfig] = None,
) -> Dict[str, Any]:
    """
    Compute the global distributional coherence package.

    This implements the recommended default design:
      - per-channel 1D Wasserstein anchors
      - fixed-bank top-k SWD or adaptive Max-SW over channel-state vectors
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
    if cfg.joint_method == "adaptive_maxswd":
        joint = batched_max_swd(
            x_gen=x_gen,
            x_ref=x_ref,
            num_directions=cfg.num_directions,
            n_iter=cfg.n_iter_theta,
            lr_theta=cfg.lr_theta,
            ortho_reg=cfg.ortho_reg,
            seed=cfg.seed,
        )
    elif cfg.joint_method == "topk_swd":
        joint = fixed_bank_topk_swd(
            x_gen=x_gen,
            x_ref=x_ref,
            num_directions=cfg.num_directions,
            top_frac=cfg.joint_top_frac,
            seed=cfg.seed,
            include_axes=cfg.include_axes,
            exclude_axes_from_score=(
                cfg.include_axes
                and cfg.exclude_axis_projections_when_marginal_included
                and float(cfg.lambda_marg) != 0.0
            ),
            qmc=cfg.joint_qmc,
        )
    else:
        raise ValueError(f"Unknown joint_method={cfg.joint_method!r}; expected 'topk_swd' or 'adaptive_maxswd'")

    base_score = cfg.lambda_marg * marg.mean() + cfg.lambda_joint * joint["score"]

    out: Dict[str, Any] = {
        "marginal_score": marg.mean(),
        "per_channel_w2": marg,
        "joint_score": joint["score"],
        "per_direction_w2": joint["per_direction_w2"],
        "theta": joint["theta"],
        "base_score": base_score,
        "joint_method": cfg.joint_method,
        "pairwise_score_included": False,
        "lambda_pairwise": float(cfg.lambda_pairwise),
    }
    if "theta_init" in joint:
        out["theta_init"] = joint["theta_init"]
    if "top_indices" in joint:
        out["top_indices"] = joint["top_indices"]
    if "top_values" in joint:
        out["top_values"] = joint["top_values"]
    if "axis_direction_mask" in joint:
        out["axis_direction_mask"] = joint["axis_direction_mask"]
    if "joint_score_mask" in joint:
        out["joint_score_mask"] = joint["joint_score_mask"]
    if "exclude_axes_from_score" in joint:
        out["exclude_axes_from_score"] = joint["exclude_axes_from_score"]
    if "top_frac" in joint:
        out["top_frac"] = joint["top_frac"]

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

    mode_score = base_score
    if cfg.include_pairwise and cfg.include_pairwise_in_score and "pairwise_mean" in out:
        mode_score = mode_score + cfg.lambda_pairwise * out["pairwise_mean"]
        out["pairwise_score_included"] = True

    out["mode_score"] = mode_score
    out["global_dist_score"] = mode_score

    return out


def coherence_result_to_scalars(result: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract scalar coherence diagnostics without mutating the input result.

    Missing optional diagnostics, such as pairwise_mean when pairwise diagnostics
    are disabled, are represented as NaN so table-writing code can continue.
    """

    def _scalar(name: str, default: float = np.nan) -> float:
        value = result.get(name)
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            flat = value.detach().cpu().reshape(-1)
            if flat.numel() == 0:
                return float(default)
            return float(flat[0].item())
        return float(value)

    return {
        "global_dist_score": _scalar("global_dist_score"),
        "base_score": _scalar("base_score"),
        "marginal_score": _scalar("marginal_score"),
        "joint_score": _scalar("joint_score"),
        "pairwise_mean": _scalar("pairwise_mean"),
        "mode_score": _scalar("mode_score"),
    }


# -----------------------------------------------------------------------------
# Registry for future coherence terms
# -----------------------------------------------------------------------------

COHERENCE_REGISTRY: Dict[str, Callable[..., Dict[str, Any]]] = {
    "global_dist": compute_global_distribution_coherence,
}


def compute_coherence(mode: str, x_gen: torch.Tensor, x_ref: torch.Tensor, **kwargs) -> Dict[str, Any]:
    """
    Dispatch coherence computation by mode name.
    """
    if mode not in COHERENCE_REGISTRY:
        raise ValueError(f"Unknown coherence mode '{mode}'. Available: {list(COHERENCE_REGISTRY.keys())}")
    return COHERENCE_REGISTRY[mode](x_gen, x_ref, **kwargs)
