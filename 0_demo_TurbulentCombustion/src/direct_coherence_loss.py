"""Differentiable direct coherence losses for PointCloudFFM post-training.

The three terms mirror the global distributional diagnostics in
``coherence_dist.py`` but keep tensors attached to autograd:

``self``
    Per-channel marginal empirical 1-D W2^2.
``mutual``
    Pairwise two-field 2-D sliced-Wasserstein, averaged over channel pairs.
``cross``
    Joint all-field fixed-bank top-k sliced-Wasserstein.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn

from coherence_dist import fixed_bank_topk_swd, pairwise_2d_swd_matrix, per_channel_w2
from obs_consistency import (
    apply_endpoint_observation_consistency,
    build_pointwise_observation_maps,
    build_smooth_observation_maps,
    normalize_obs_consistency_mode,
    scatter_observed_values,
)


@dataclass
class DirectCoherenceConfig:
    """Configuration for differentiable direct global coherence.

    ``self_weight`` multiplies the per-channel marginal 1-D W2^2 term.
    ``mutual_weight`` multiplies the pairwise two-field 2-D sliced-Wasserstein
    term, averaged over all upper-triangular channel pairs.
    ``cross_weight`` multiplies the joint all-field fixed-bank top-k
    sliced-Wasserstein term.

    ``channel_weights`` optionally reweights the self/marginal term by channel.
    ``use_denorm`` evaluates coherence in physical units using supplied
    ``mean`` and ``std``; otherwise normalized model space is used.
    """

    enabled: bool = False

    self_weight: float = 1.0
    mutual_weight: float = 0.0
    cross_weight: float = 1.0

    channel_weights: Optional[Sequence[float]] = None

    cross_num_directions: int = 32
    cross_top_frac: float = 0.10
    cross_seed: int = 1234
    cross_include_axes: bool = True
    cross_qmc: bool = True

    mutual_num_directions: int = 16
    mutual_seed: int = 1234

    use_denorm: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def _zero_like_scalar(x: torch.Tensor) -> torch.Tensor:
    return x.sum() * 0.0


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains NaN or Inf values.")


class DirectGlobalCoherenceLoss(nn.Module):
    """Differentiable direct coherence objective for generated clean fields."""

    def __init__(self, cfg: DirectCoherenceConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def _maybe_denormalize(
        self,
        x_gen: torch.Tensor,
        x_ref: torch.Tensor,
        mean: Optional[torch.Tensor],
        std: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.cfg.use_denorm:
            return x_gen, x_ref
        if mean is None or std is None:
            raise ValueError("mean and std are required when DirectCoherenceConfig.use_denorm=True.")
        mean = mean.to(device=x_gen.device, dtype=x_gen.dtype).view(1, 1, -1)
        std = std.to(device=x_gen.device, dtype=x_gen.dtype).view(1, 1, -1)
        _require_finite("mean", mean)
        _require_finite("std", std)
        return x_gen * std + mean, x_ref * std + mean

    def forward(
        self,
        x_gen: torch.Tensor,
        x_ref: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x_gen.ndim != 3 or x_ref.ndim != 3:
            raise ValueError(
                f"DirectGlobalCoherenceLoss expects [B, N, C], got {tuple(x_gen.shape)} and {tuple(x_ref.shape)}"
            )
        if x_gen.shape != x_ref.shape:
            raise ValueError(f"Shape mismatch: {tuple(x_gen.shape)} vs {tuple(x_ref.shape)}")
        _require_finite("x_gen", x_gen)
        _require_finite("x_ref", x_ref)

        x_gen, x_ref = self._maybe_denormalize(x_gen, x_ref, mean, std)
        cfg = self.cfg
        bsz, _, n_channels = x_gen.shape
        base_zero = _zero_like_scalar(x_gen)

        self_loss = base_zero
        if float(cfg.self_weight) != 0.0:
            values = []
            for b in range(bsz):
                per_ch = per_channel_w2(x_gen[b], x_ref[b])
                if cfg.channel_weights is not None:
                    weights = torch.as_tensor(cfg.channel_weights, device=x_gen.device, dtype=x_gen.dtype)
                    if weights.numel() != n_channels:
                        raise ValueError(
                            f"channel_weights length {weights.numel()} does not match channel count {n_channels}."
                        )
                    if torch.any(weights < 0):
                        raise ValueError("channel_weights must be non-negative.")
                    denom = weights.sum().clamp_min(torch.finfo(x_gen.dtype).eps)
                    values.append((per_ch * weights).sum() / denom)
                else:
                    values.append(per_ch.mean())
            self_loss = torch.stack(values).mean()
            _require_finite("self_loss", self_loss)

        mutual_loss = base_zero
        if float(cfg.mutual_weight) != 0.0:
            if n_channels < 2:
                mutual_loss = base_zero
            else:
                values = []
                tri_i, tri_j = torch.triu_indices(n_channels, n_channels, offset=1, device=x_gen.device)
                for b in range(bsz):
                    mat = pairwise_2d_swd_matrix(
                        x_gen=x_gen[b],
                        x_ref=x_ref[b],
                        n_proj=int(cfg.mutual_num_directions),
                        seed=int(cfg.mutual_seed),
                    )
                    values.append(mat[tri_i, tri_j].mean())
                mutual_loss = torch.stack(values).mean()
            _require_finite("mutual_loss", mutual_loss)

        cross_loss = base_zero
        if float(cfg.cross_weight) != 0.0:
            values = []
            for b in range(bsz):
                result = fixed_bank_topk_swd(
                    x_gen=x_gen[b],
                    x_ref=x_ref[b],
                    num_directions=int(cfg.cross_num_directions),
                    top_frac=float(cfg.cross_top_frac),
                    seed=int(cfg.cross_seed),
                    include_axes=bool(cfg.cross_include_axes),
                    exclude_axes_from_score=False,
                    qmc=bool(cfg.cross_qmc),
                )
                values.append(result["score"])
            cross_loss = torch.stack(values).mean()
            _require_finite("cross_loss", cross_loss)

        total = (
            float(cfg.self_weight) * self_loss
            + float(cfg.mutual_weight) * mutual_loss
            + float(cfg.cross_weight) * cross_loss
        )
        _require_finite("total_loss", total)
        components = {
            "self_loss": self_loss,
            "mutual_loss": mutual_loss,
            "cross_loss": cross_loss,
            "total_loss": total,
        }
        return total, components


def sample_coherence_points(
    coords: torch.Tensor,
    fields: torch.Tensor,
    n_points: Optional[int],
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Uniformly subsample empirical field points for coherence evaluation."""
    if coords.ndim != 3 or fields.ndim != 3:
        raise ValueError(f"coords and fields must be [B, N, *], got {tuple(coords.shape)} and {tuple(fields.shape)}")
    if coords.shape[:2] != fields.shape[:2]:
        raise ValueError(f"coords/fields point shape mismatch: {tuple(coords.shape)} vs {tuple(fields.shape)}")
    if n_points is None or int(n_points) >= coords.shape[1]:
        return coords, fields, None

    bsz, n_total, coord_dim = coords.shape
    n_select = max(1, int(n_points))
    indices = []
    for _ in range(bsz):
        indices.append(torch.randperm(n_total, device=coords.device, generator=generator)[:n_select].sort().values)
    idx = torch.stack(indices, dim=0)
    coord_idx = idx.unsqueeze(-1).expand(-1, -1, coord_dim)
    field_idx = idx.unsqueeze(-1).expand(-1, -1, fields.shape[-1])
    return torch.gather(coords, 1, coord_idx), torch.gather(fields, 1, field_idx), idx


def _call_velocity_model(
    ffm_model: nn.Module,
    t: torch.Tensor,
    x_t: torch.Tensor,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    kwargs = {
        "t": t,
        "x_t": x_t,
        "coords": coords,
        "obs_coords": obs_coords,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_field_ids": obs_field_ids,
    }
    if getattr(ffm_model, "requires_full_grid", False):
        if obs_indices is None:
            raise ValueError("This FFM model requires full-grid obs_indices for velocity calls.")
        kwargs["obs_indices"] = obs_indices
    return ffm_model.model(**kwargs)


def _validate_pointwise_indices(
    coords: torch.Tensor,
    obs_indices: Optional[torch.Tensor],
    mode: str,
) -> None:
    if mode not in ("default_hard", "endpoint"):
        return
    if obs_indices is None:
        raise ValueError(f"obs_consistency_mode={mode!r} requires obs_indices.")
    valid = obs_indices >= 0
    if torch.any(valid & (obs_indices >= coords.shape[1])):
        raise ValueError(
            f"obs_consistency_mode={mode!r} requires obs_indices valid for the rollout coordinate set. "
            "Use endpoint_smooth for point-subset coherence rollouts."
        )


def differentiable_rf_rollout(
    ffm_model: nn.Module,
    coords: torch.Tensor,
    obs_coords: torch.Tensor,
    obs_values: torch.Tensor,
    obs_mask: torch.Tensor,
    obs_field_ids: torch.Tensor,
    obs_indices: Optional[torch.Tensor],
    n_steps: int = 2,
    ode_solver: str = "euler",
    obs_consistency_mode: str = "endpoint_smooth",
    obs_consistency_strength: float = 1.0,
    obs_consistency_sigma: float = 0.05,
    obs_consistency_schedule_power: float = 2.0,
    obs_consistency_final_clamp: bool = True,
) -> torch.Tensor:
    """Differentiable terminal rollout for direct coherence training."""
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if ode_solver not in ("euler", "heun"):
        raise ValueError(f"Unsupported ode_solver={ode_solver!r}; expected 'euler' or 'heun'.")
    if getattr(ffm_model, "requires_full_grid", False):
        if obs_indices is None:
            raise ValueError("Full-grid FFM rollouts require obs_indices.")
        if coords.shape[1] != int(getattr(ffm_model.model, "Num_x", 0)) * int(getattr(ffm_model.model, "Num_y", 0)):
            raise ValueError("Full-grid FFM rollouts require full-grid coordinates.")

    mode = normalize_obs_consistency_mode(obs_consistency_mode)
    _validate_pointwise_indices(coords, obs_indices, mode)

    bsz = coords.shape[0]
    x = ffm_model.sample_source(coords)
    value_map = None
    mask_map = None
    if mode == "endpoint":
        value_map, mask_map = build_pointwise_observation_maps(
            coords=coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_indices=obs_indices,
            obs_field_ids=obs_field_ids,
            n_fields=ffm_model.model.n_fields,
        )
    elif mode == "endpoint_smooth":
        value_map, mask_map = build_smooth_observation_maps(
            coords=coords,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            n_fields=ffm_model.model.n_fields,
            sigma=float(obs_consistency_sigma),
        )

    ts = torch.linspace(0.0, 1.0, int(n_steps) + 1, device=coords.device, dtype=coords.dtype)
    for i in range(int(n_steps)):
        t0 = ts[i].expand(bsz)
        dt = ts[i + 1] - ts[i]
        v0 = _call_velocity_model(
            ffm_model, t0, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids, obs_indices
        )
        if mode in ("endpoint", "endpoint_smooth"):
            v0 = apply_endpoint_observation_consistency(
                x_t=x,
                v=v0,
                t=t0,
                value_map=value_map,
                mask_map=mask_map,
                strength=obs_consistency_strength,
                schedule_power=obs_consistency_schedule_power,
            )

        if ode_solver == "heun":
            x_euler = x + dt * v0
            t1 = ts[i + 1].expand(bsz)
            v1 = _call_velocity_model(
                ffm_model, t1, x_euler, coords, obs_coords, obs_values, obs_mask, obs_field_ids, obs_indices
            )
            if mode in ("endpoint", "endpoint_smooth") and float(ts[i + 1].item()) < 1.0:
                v1 = apply_endpoint_observation_consistency(
                    x_t=x_euler,
                    v=v1,
                    t=t1,
                    value_map=value_map,
                    mask_map=mask_map,
                    strength=obs_consistency_strength,
                    schedule_power=obs_consistency_schedule_power,
                )
            x = x + 0.5 * dt * (v0 + v1)
        else:
            x = x + dt * v0

        if mode == "default_hard":
            x = scatter_observed_values(x, obs_values, obs_mask, obs_indices, obs_field_ids, strength=1.0)

    if obs_consistency_final_clamp and mode != "none" and obs_indices is not None:
        valid = obs_indices >= 0
        indices_fit_current_coords = not torch.any(valid & (obs_indices >= coords.shape[1]))
        if indices_fit_current_coords:
            x = scatter_observed_values(x, obs_values, obs_mask, obs_indices, obs_field_ids, strength=1.0)
    return x


def _gradient_vector_from_model(model: nn.Module) -> torch.Tensor:
    pieces = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            pieces.append(torch.zeros_like(param).reshape(-1))
        else:
            pieces.append(param.grad.reshape(-1))
    if not pieces:
        raise RuntimeError("No trainable parameters found for gradient diagnostics.")
    return torch.cat(pieces)


def _weighted_sum_update(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    data_loss: torch.Tensor,
    coherence_loss: torch.Tensor,
    data_weight: float,
    coherence_weight: float,
    grad_clip_norm: Optional[float],
) -> dict:
    optimizer.zero_grad(set_to_none=True)
    total_loss = float(data_weight) * data_loss + float(coherence_weight) * coherence_loss
    total_loss.backward()
    grad_vec = _gradient_vector_from_model(model)
    if not torch.isfinite(grad_vec).all():
        raise FloatingPointError("Weighted-sum gradient contains NaN or Inf values.")
    if grad_clip_norm is not None and float(grad_clip_norm) > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()
    norm = torch.linalg.vector_norm(grad_vec.detach())
    return {
        "data_grad_norm": float("nan"),
        "coherence_grad_norm": float("nan"),
        "gradient_cosine": float("nan"),
        "gradient_conflict": False,
        "config_fallback_used": False,
        "combined_grad_norm": float(norm.detach().cpu()),
    }


class _FixedConFIGWeights:
    """Weight model adapter for conflictfree.ConFIG_update."""

    def __init__(self, weights: torch.Tensor) -> None:
        self.weights = weights.detach().clone()

    def get_weights(
        self,
        gradients: Optional[torch.Tensor] = None,
        losses: Optional[Sequence] = None,
        device: Optional[torch.device | str] = None,
    ) -> torch.Tensor:
        if gradients is None:
            raise ValueError("ConFIG fixed weights require gradients.")
        out = self.weights.to(device=gradients.device if device is None else device, dtype=gradients.dtype)
        if out.numel() != gradients.shape[0]:
            raise ValueError(
                f"ConFIG fixed weights length {out.numel()} does not match gradient count {gradients.shape[0]}."
            )
        return out


def apply_two_objective_update(
    model,
    optimizer,
    data_loss,
    coherence_loss,
    mode,
    data_weight,
    coherence_weight,
    grad_clip_norm,
    config_missing_behavior="error",
) -> dict:
    """Apply weighted-sum or ConFIG two-objective update.

    In ``config`` mode, ``data_weight`` and ``coherence_weight`` are gradient
    pre-scales before the conflict-free update. Callers should pass the same
    scheduled loss weights used for logging/warmup, optionally multiplied by
    ConFIG-specific scale factors.

    For stability, ConFIG is applied only when the two weighted gradients are
    genuinely conflicting. If they are already aligned, the scheduled weighted
    sum is the least surprising descent direction. If ConFIG returns a direction
    that is not a descent direction for an active objective, we also fall back to
    the scheduled weighted sum.
    """
    mode = str(mode or "weighted_sum").strip().lower()
    if mode == "weighted_sum":
        return _weighted_sum_update(
            model, optimizer, data_loss, coherence_loss, data_weight, coherence_weight, grad_clip_norm
        )
    if mode != "config":
        raise ValueError("gradient_balance_mode must be 'weighted_sum' or 'config'.")

    try:
        from conflictfree.grad_operator import ConFIG_update
        from conflictfree.utils import apply_gradient_vector, get_gradient_vector
    except ImportError as exc:
        if str(config_missing_behavior) == "weighted_sum":
            warnings.warn(
                "conflictfree is not installed; falling back to weighted_sum. "
                "Install with: pip install conflictfree",
                RuntimeWarning,
                stacklevel=2,
            )
            return _weighted_sum_update(
                model, optimizer, data_loss, coherence_loss, data_weight, coherence_weight, grad_clip_norm
            )
        raise ImportError(
            "gradient_balance_mode='config' requires the optional conflictfree package. "
            "Install with: pip install conflictfree"
        ) from exc

    config_weights = torch.tensor(
        [float(data_weight), float(coherence_weight)],
        device=data_loss.device,
        dtype=data_loss.dtype,
    ).clamp_min(0.0)
    if not torch.isfinite(config_weights).all():
        raise FloatingPointError("ConFIG objective weights contain NaN or Inf values.")
    if float(config_weights.sum().detach().cpu()) <= 0.0:
        warnings.warn("Both ConFIG objective weights are zero; falling back to data gradient.", RuntimeWarning, stacklevel=2)
        config_weights = torch.tensor([1.0, 0.0], device=data_loss.device, dtype=data_loss.dtype)

    optimizer.zero_grad(set_to_none=True)
    (float(data_weight) * data_loss).backward()
    g_data = get_gradient_vector(model)
    optimizer.zero_grad(set_to_none=True)
    (float(coherence_weight) * coherence_loss).backward()
    g_coherence = get_gradient_vector(model)

    data_ok = torch.isfinite(g_data).all()
    coherence_ok = torch.isfinite(g_coherence).all()
    data_norm = torch.linalg.vector_norm(g_data.detach())
    coherence_norm = torch.linalg.vector_norm(g_coherence.detach())
    denom = (data_norm * coherence_norm).clamp_min(1e-12)
    cosine = torch.dot(g_data.detach().reshape(-1), g_coherence.detach().reshape(-1)) / denom
    weighted_sum_grad = g_data + g_coherence

    optimizer.zero_grad(set_to_none=True)
    fallback = False
    update_mode = "config"
    if not data_ok:
        raise FloatingPointError("Data gradient contains NaN or Inf values.")
    if (not coherence_ok) or float(coherence_norm.detach().cpu()) == 0.0:
        warnings.warn("Invalid or zero coherence gradient; falling back to weighted data gradient.", RuntimeWarning, stacklevel=2)
        apply_gradient_vector(model, g_data)
        fallback = True
        applied_norm = torch.linalg.vector_norm(g_data.detach())
        update_mode = "data_fallback"
    elif float(cosine.detach().cpu()) >= 0.0:
        apply_gradient_vector(model, weighted_sum_grad)
        applied_norm = torch.linalg.vector_norm(weighted_sum_grad.detach())
        update_mode = "weighted_sum_aligned"
    else:
        g_config = ConFIG_update(
            [g_data, g_coherence],
            weight_model=_FixedConFIGWeights(config_weights),
        )
        if not torch.isfinite(g_config).all():
            warnings.warn("ConFIG gradient was invalid; falling back to scheduled weighted-sum gradient.", RuntimeWarning, stacklevel=2)
            apply_gradient_vector(model, weighted_sum_grad)
            fallback = True
            applied_norm = torch.linalg.vector_norm(weighted_sum_grad.detach())
            update_mode = "weighted_sum_invalid_config"
        else:
            descent_data = torch.dot(g_config.detach().reshape(-1), g_data.detach().reshape(-1))
            descent_coherence = torch.dot(g_config.detach().reshape(-1), g_coherence.detach().reshape(-1))
            active_data = float(data_norm.detach().cpu()) > 0.0 and float(config_weights[0].detach().cpu()) > 0.0
            active_coherence = float(coherence_norm.detach().cpu()) > 0.0 and float(config_weights[1].detach().cpu()) > 0.0
            if (active_data and float(descent_data.cpu()) <= 0.0) or (
                active_coherence and float(descent_coherence.cpu()) <= 0.0
            ):
                warnings.warn(
                    "ConFIG gradient was not a descent direction for an active objective; "
                    "falling back to scheduled weighted-sum gradient.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                apply_gradient_vector(model, weighted_sum_grad)
                fallback = True
                applied_norm = torch.linalg.vector_norm(weighted_sum_grad.detach())
                update_mode = "weighted_sum_nondescent_config"
            else:
                apply_gradient_vector(model, g_config)
                applied_norm = torch.linalg.vector_norm(g_config.detach())

    if grad_clip_norm is not None and float(grad_clip_norm) > 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(grad_clip_norm))
    optimizer.step()
    return {
        "data_grad_norm": float(data_norm.detach().cpu()),
        "coherence_grad_norm": float(coherence_norm.detach().cpu()),
        "gradient_cosine": float(cosine.detach().cpu()),
        "gradient_conflict": bool(float(cosine.detach().cpu()) < 0.0),
        "config_fallback_used": fallback,
        "combined_grad_norm": float(applied_norm.detach().cpu()),
        "config_data_weight": float(config_weights[0].detach().cpu()),
        "config_coherence_weight": float(config_weights[1].detach().cpu()),
        "config_update_mode": update_mode,
    }


def gradient_sanity_check(device: str | torch.device = "cpu") -> bool:
    """Small unit-style check that enabled coherence terms backpropagate."""
    x_gen = torch.randn(2, 32, 5, device=device, requires_grad=True)
    x_ref = torch.randn(2, 32, 5, device=device)
    cfg = DirectCoherenceConfig(enabled=True, self_weight=1.0, mutual_weight=0.25, cross_weight=1.0)
    loss, _ = DirectGlobalCoherenceLoss(cfg)(x_gen, x_ref)
    loss.backward()
    return bool(
        x_gen.grad is not None
        and torch.isfinite(x_gen.grad).all()
        and torch.linalg.vector_norm(x_gen.grad).item() > 0.0
    )
