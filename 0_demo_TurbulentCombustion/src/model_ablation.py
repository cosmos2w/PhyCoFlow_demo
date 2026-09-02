"""Cond_T ablation-only model behavior.

The production GL-RBF backbone and rectified-flow wrapper remain unchanged.
Only the narrow scientific interventions for A1--A5 live in this module.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from phycoflow_pointcloud.models.portable_core import ConditionalPointHybridLocalGlobalRBF
from phycoflow_pointcloud.observation import (
    normalize_obs_consistency_mode,
    scatter_observed_values,
)


ABLATION_METADATA: dict[str, dict[str, Any]] = {
    "A1": {
        "ablation_id": "A1",
        "ablation_variant": "deterministic_same_backbone",
        "objective_type": "deterministic_direct_field_mse",
        "deterministic_state": "zeros",
        "deterministic_tau": 0.0,
        "generative_prior_used": False,
    },
    "A2": {
        "ablation_id": "A2",
        "ablation_variant": "no_sensor_global_feedback",
        "objective_type": "rectified_flow",
        "sensor_global_feedback": False,
        "latent_sensor_reinject": True,
        "local_query_conditioning": True,
        "prior": "rff",
    },
    "A3": {
        "ablation_id": "A3",
        "ablation_variant": "no_local_query_conditioning",
        "objective_type": "rectified_flow",
        "sensor_global_feedback": True,
        "local_query_conditioning": False,
        "prior": "rff",
        "bypassed_functions": [
            "aggregate_sparse_obs",
            "_aggregate_chunk",
            "_aggregate_topk_from_geometry",
        ],
    },
    "A4": {
        "ablation_id": "A4",
        "ablation_variant": "iid_gaussian_prior",
        "objective_type": "rectified_flow",
        "sensor_global_feedback": True,
        "local_query_conditioning": True,
        "prior": "iid",
    },
    "A5": {
        "ablation_id": "A5",
        "ablation_variant": "local_sensor_tokens_only",
        "objective_type": "rectified_flow",
        "sensor_to_latent": False,
        "sensor_global_feedback": False,
        "latent_global_conditioning": False,
        "query_latent_readout": False,
        "local_query_conditioning": True,
        "observation_conditioning_route": "sensor_token_topk_only",
        "prior": "rff",
        "bypassed_functions": [
            "_encode_latents",
            "_extract_global_summary",
            "_refine_sensor_tokens",
            "query_latent_readout",
        ],
    },
}

_VARIANT_TO_ID = {
    metadata["ablation_variant"]: ablation_id
    for ablation_id, metadata in ABLATION_METADATA.items()
}


def resolve_ablation(config: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    """Validate and normalize the nested ablation configuration."""
    raw = config.get("ablation")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise TypeError("ablation must be a YAML mapping.")
    if not bool(raw.get("enabled", False)):
        return None
    ablation_id = str(raw.get("id", "")).upper()
    variant = str(raw.get("variant", ""))
    if ablation_id not in ABLATION_METADATA:
        raise ValueError(f"Unknown ablation id {ablation_id!r}; expected A1--A5.")
    expected = ABLATION_METADATA[ablation_id]["ablation_variant"]
    if variant != expected:
        raise ValueError(
            f"Ablation {ablation_id} requires variant={expected!r}, got {variant!r}."
        )
    if _VARIANT_TO_ID.get(variant) != ablation_id:
        raise ValueError(f"Inconsistent ablation id/variant: {ablation_id}/{variant}.")
    return dict(raw)


def ablation_metadata(config: Mapping[str, Any]) -> Optional[dict[str, Any]]:
    resolved = resolve_ablation(config)
    if resolved is None:
        return None
    metadata = dict(ABLATION_METADATA[str(resolved["id"]).upper()])
    metadata.update(
        {
            key: resolved[key]
            for key in (
                "reference_name",
                "reference_config",
                "reference_config_sha256",
                "reference_checkpoint",
                "reference_checkpoint_sha256",
            )
            if key in resolved
        }
    )
    return metadata


class NoSensorGlobalFeedbackBackbone(ConditionalPointHybridLocalGlobalRBF):
    """A2: retain the module schema but bypass latent-to-sensor feedback."""

    def _refine_sensor_tokens(
        self,
        sensor_tokens: torch.Tensor,
        latents: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        del latents
        return sensor_tokens * obs_mask.unsqueeze(-1).to(sensor_tokens.dtype)


class NoLocalQueryConditioningBackbone(ConditionalPointHybridLocalGlobalRBF):
    """A3: retain local modules/widths but transfer no query-local evidence."""

    @staticmethod
    def _zero_local(
        query_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
    ) -> torch.Tensor:
        return refined_sensor_feat.new_zeros(
            query_coords.shape[0],
            query_coords.shape[1],
            refined_sensor_feat.shape[-1],
        )

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        query_feat: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        sensor_importance_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del query_feat, obs_coords, obs_mask, sensor_importance_bias
        return self._zero_local(query_coords, refined_sensor_feat)

    def _aggregate_chunk(
        self,
        query_coords: torch.Tensor,
        query_feat: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        sensor_importance_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del query_feat, obs_coords, obs_mask, sensor_importance_bias
        return self._zero_local(query_coords, refined_sensor_feat)

    def _aggregate_topk_from_geometry(
        self,
        topk_d2: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_valid: torch.Tensor,
        condition_context: Mapping[str, torch.Tensor],
        topk_sensor_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del topk_idx, topk_valid, topk_sensor_feat
        sensor_feat = condition_context["refined_sensor_feat"]
        return sensor_feat.new_zeros(
            topk_d2.shape[0], topk_d2.shape[1], sensor_feat.shape[-1]
        )


class LocalSensorTokensOnlyBackbone(NoSensorGlobalFeedbackBackbone):
    """A5: observations condition queries only through raw sensor-token Top-K.

    All A0 global modules remain instantiated so parameter/state schemas are
    unchanged.  Their execution is bypassed: no observation-dependent latent
    memory is constructed, no global summary is supplied to the head, and the
    query-to-latent readout is disabled.  The point/state branch remains active;
    among observation pathways, only sensor token -> sensor_out_proj -> Top-K/RBF
    reaches a query.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # The readout module was already instantiated by A0 construction.  This
        # execution-only flag preserves all of its parameters/state-dict keys.
        self.use_query_latent_readout = False

    def _encode_latents(
        self,
        sensor_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        del obs_mask
        return sensor_tokens.new_zeros(
            sensor_tokens.shape[0], self.num_latents, self.latent_dim
        )

    def _extract_global_summary(self, latents: torch.Tensor) -> torch.Tensor:
        hidden_dim = self.summary_proj[-1].out_features
        return latents.new_zeros(latents.shape[0], hidden_dim)


class DeterministicDMFRegressor(nn.Module):
    """A1 direct-field wrapper around the unchanged GL_rbf_ENH backbone."""

    def __init__(self, model: nn.Module, sigma_min: float = 1.0e-4):
        super().__init__()
        self.model = model
        self.sigma_min = float(sigma_min)
        self.ablation_metadata = dict(ABLATION_METADATA["A1"])

    def sample_source(self, coords: torch.Tensor) -> torch.Tensor:
        del coords
        raise RuntimeError("A1 is deterministic and has no generative source prior.")

    @staticmethod
    def _neutral_state(x1_or_coords: torch.Tensor, n_fields: int) -> torch.Tensor:
        if x1_or_coords.shape[-1] == n_fields:
            return torch.zeros_like(x1_or_coords)
        return x1_or_coords.new_zeros(
            x1_or_coords.shape[0], x1_or_coords.shape[1], n_fields
        )

    def training_loss(
        self,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del obs_indices
        state = self._neutral_state(x1, self.model.n_fields)
        tau = x1.new_zeros(x1.shape[0])
        prediction = self.model(
            tau, state, coords, obs_coords, obs_values, obs_mask, obs_field_ids
        )
        loss = F.mse_loss(prediction, x1)
        return loss, {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(x1.pow(2).mean().sqrt().detach().cpu()),
        }

    def training_loss_microbatched(
        self,
        *,
        x1: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
        query_microbatch_size: int,
        backward: bool = False,
        reuse_condition_context: bool = True,
        synchronize_timing: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        del obs_indices
        n_query = int(coords.shape[1])
        chunk_size = max(1, int(query_microbatch_size))
        if chunk_size >= n_query:
            loss, metrics = self.training_loss(
                x1, coords, obs_coords, obs_values, obs_mask, obs_field_ids
            )
            if backward:
                loss.backward()
            metrics["query_microbatches"] = 1.0
            return loss.detach() if backward else loss, metrics

        def sync() -> None:
            if synchronize_timing and x1.device.type == "cuda":
                torch.cuda.synchronize(x1.device)

        sync()
        condition_start = time.perf_counter()
        condition_context = None
        if reuse_condition_context:
            condition_context = self.model.prepare_condition_context(
                obs_coords, obs_values, obs_mask, obs_field_ids
            )
        sync()
        condition_ms = (time.perf_counter() - condition_start) * 1000.0
        tau = x1.new_zeros(x1.shape[0])
        total_loss = x1.new_zeros(())
        total_elements = int(x1.numel())
        chunks = 0
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            query_slice = slice(start, end)
            state = torch.zeros_like(x1[:, query_slice])
            if condition_context is None:
                prediction = self.model(
                    tau,
                    state,
                    coords[:, query_slice],
                    obs_coords,
                    obs_values,
                    obs_mask,
                    obs_field_ids,
                )
            else:
                prediction = self.model.forward_query_chunk(
                    tau,
                    state,
                    coords[:, query_slice],
                    condition_context,
                )
            chunk_loss = F.mse_loss(
                prediction, x1[:, query_slice], reduction="sum"
            ) / total_elements
            if backward:
                chunk_loss.backward(
                    retain_graph=condition_context is not None and end < n_query
                )
                total_loss = total_loss + chunk_loss.detach()
            else:
                total_loss = total_loss + chunk_loss
            chunks += 1
        return total_loss, {
            "loss": float(total_loss.detach().cpu()),
            "target_rms": float(x1.pow(2).mean().sqrt().detach().cpu()),
            "condition_context_ms": condition_ms,
            "query_microbatches": float(chunks),
        }

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        n_steps: int = 1,
        clamp_indices: Optional[torch.Tensor] = None,
        ode_solver: str = "euler",
        obs_consistency_mode: str = "default_hard",
        obs_consistency_strength: float = 1.0,
        obs_consistency_sigma: float = 0.05,
        obs_consistency_schedule_power: float = 2.0,
        obs_consistency_final_clamp: bool = True,
        obs_consistency_chunk_size: int = 8192,
        reconstruction_execution_mode: str = "legacy_full",
        reconstruction_query_chunk_size: int = 8192,
        reconstruction_cache_level: str = "static_features",
        reconstruction_geometry_cache: Optional[Any] = None,
    ) -> torch.Tensor:
        del (
            n_steps,
            ode_solver,
            obs_consistency_strength,
            obs_consistency_sigma,
            obs_consistency_schedule_power,
            obs_consistency_chunk_size,
            reconstruction_execution_mode,
            reconstruction_query_chunk_size,
            reconstruction_cache_level,
            reconstruction_geometry_cache,
        )
        state = self._neutral_state(coords, self.model.n_fields)
        tau = coords.new_zeros(coords.shape[0])
        prediction = self.model(
            tau, state, coords, obs_coords, obs_values, obs_mask, obs_field_ids
        )
        mode = normalize_obs_consistency_mode(obs_consistency_mode)
        if mode in {"default_hard", "endpoint"} and clamp_indices is None:
            raise ValueError(f"obs_consistency_mode={mode!r} requires clamp_indices.")
        if obs_consistency_final_clamp and mode != "none" and clamp_indices is not None:
            prediction = scatter_observed_values(
                x=prediction,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )
        return prediction
