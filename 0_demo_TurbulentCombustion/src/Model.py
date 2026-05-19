

import math, os, torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO as NeuralOpFNO  # pip install neuraloperator
from pykeops.torch import LazyTensor

from obs_consistency import (
    apply_endpoint_observation_consistency,
    build_pointwise_observation_maps,
    build_smooth_observation_maps,
    normalize_obs_consistency_mode,
    scatter_observed_values,
)

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

# ------------------------------
# mlp_rbf backbone
# ------------------------------
def make_mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, act=nn.GELU) -> nn.Sequential:
    layers = []
    dim = in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(dim, hidden_dim), act()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, out_dim))
    return nn.Sequential(*layers)


class FourierPositionalEncoding(nn.Module):
    """Sine-cosine frequency encoding for spatial coordinates."""

    def __init__(self, coord_dim: int, num_bands: int = 32, max_freq: float = 64.0):
        super().__init__()
        self.coord_dim = coord_dim
        self.num_bands = num_bands
        self.out_dim = coord_dim * num_bands * 2
        freqs = torch.linspace(1.0, max_freq / 2.0, num_bands)
        self.register_buffer("freqs", freqs)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords[..., : self.coord_dim] * 2.0 - 1.0
        x = coords.unsqueeze(-1) * self.freqs * math.pi
        enc = torch.cat([x.sin(), x.cos()], dim=-1)
        return enc.reshape(*coords.shape[:-1], self.out_dim)

# ------------------------------
# for gathering in GL_rbf
# ------------------------------
def batched_gather_2d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather from x with shape [B, M] using idx with shape [B, N, K].
    Returns shape [B, N, K].
    """
    bsz = x.shape[0]
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(idx)
    return x[batch_idx, idx]


def batched_gather_3d(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    Gather from x with shape [B, M, C] using idx with shape [B, N, K].
    Returns shape [B, N, K, C].
    """
    bsz = x.shape[0]
    batch_idx = torch.arange(bsz, device=x.device).view(bsz, 1, 1).expand_as(idx)
    return x[batch_idx, idx]

class ConditionalPointFFM(nn.Module):
    """
    Instead of one global cond_field_idx, each observation now carries its own field id
    by giving each sensor a learnable field_embed_dim, allowing the model to know 
    what physical property the sensor is measuring, not just where it is.
    """
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        rbf_sigma: float = 0.05,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma
        self.use_fourier_pe = use_fourier_pe
        self.pos_enc = FourierPositionalEncoding(
            coord_dim, num_bands=fourier_pe_num_bands, max_freq=fourier_pe_max_freq
        ) if use_fourier_pe else None
        self.coord_feat_dim = self.pos_enc.out_dim if self.pos_enc is not None else coord_dim

        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        self.point_encoder = make_mlp(self.coord_feat_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3)
        self.obs_encoder = make_mlp(coord_dim + 1 + field_embed_dim, cond_dim, cond_dim, depth=3)
        self.global_encoder = make_mlp(hidden_dim, hidden_dim, hidden_dim, depth=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_fields),
        )

    # For any given query point, the model calculates the physical squared distance to every available sensor. 
    # Using a Radial Basis Function (RBF) kernel, it applies an attention weight: 
    # sensors that are physically closer exert massive influence on the query point, while distant sensors are ignored.
    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        safe_field_ids = obs_field_ids.clamp_min(0)
        obs_field_feat = self.field_embed(safe_field_ids)                 # [B, M, E]
        obs_field_feat = obs_field_feat * obs_mask.unsqueeze(-1)          # zero padded rows

        obs_in = torch.cat([obs_coords, obs_values, obs_field_feat], dim=-1)
        obs_feat = self.obs_encoder(obs_in)
        obs_feat = obs_feat * obs_mask.unsqueeze(-1)

        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        weights = torch.softmax(-d2 / (2 * self.rbf_sigma ** 2 + 1e-12), dim=-1)
        return torch.einsum("bnm,bmd->bnd", weights, obs_feat)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)

        coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t, t_feat], dim=-1))
        local_cond = self.aggregate_sparse_obs(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        global_feat = self.global_encoder(point_feat.mean(dim=1)).unsqueeze(1).expand(bsz, n_pts, -1)

        return self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))


class ConditionalPointMLPRBF(nn.Module):
    """
    Current baseline backbone:
      - per-query-point MLP encoder
      - sensor token encoder
      - RBF-weighted local sensor aggregation
      - one global pooled feature
      - pointwise velocity head

    This is your current model, kept under a clearer name so it can be
    compared directly against the Perceiver backbone.
    """
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        rbf_sigma: float = 0.05,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma
        self.use_fourier_pe = use_fourier_pe
        self.pos_enc = FourierPositionalEncoding(
            coord_dim, num_bands=fourier_pe_num_bands, max_freq=fourier_pe_max_freq
        ) if use_fourier_pe else None
        self.coord_feat_dim = self.pos_enc.out_dim if self.pos_enc is not None else coord_dim

        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        self.point_encoder = make_mlp(self.coord_feat_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3)
        self.obs_encoder = make_mlp(coord_dim + 1 + field_embed_dim, cond_dim, cond_dim, depth=3)
        self.global_encoder = make_mlp(hidden_dim, hidden_dim, hidden_dim, depth=2)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_fields),
        )

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Embed the physical field identity for each sparse sensor.
        safe_field_ids = obs_field_ids.clamp_min(0)
        obs_field_feat = self.field_embed(safe_field_ids)
        obs_field_feat = obs_field_feat * obs_mask.unsqueeze(-1)

        # Encode sparse sensor tokens.
        obs_in = torch.cat([obs_coords, obs_values, obs_field_feat], dim=-1)
        obs_feat = self.obs_encoder(obs_in)
        obs_feat = obs_feat * obs_mask.unsqueeze(-1)

        # RBF weighting from each query point to each sparse sensor.
        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        weights = torch.softmax(-d2 / (2 * self.rbf_sigma ** 2 + 1e-12), dim=-1)
        return torch.einsum("bnm,bmd->bnd", weights, obs_feat)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)

        coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t, t_feat], dim=-1))
        local_cond = self.aggregate_sparse_obs(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        global_feat = self.global_encoder(point_feat.mean(dim=1)).unsqueeze(1).expand(bsz, n_pts, -1)

        return self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))


# ------------------------------
# Perceiver backbone
# ------------------------------
class FeedForward(nn.Module):
    """
    Standard Transformer feed-forward block used after attention.
    """
    def __init__(self, dim: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * ff_mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block with residual connection and FFN.

    q  : [B, Tq, D]
    kv : [B, Tk, D]
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_mult=ff_mult, dropout=mlp_dropout)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Normalize queries and keys/values independently.
        q_in = self.norm_q(q)
        kv_in = self.norm_kv(kv)

        # key_padding_mask: True means "ignore this token".
        attn_out, _ = self.attn(
            q_in,
            kv_in,
            kv_in,
            key_padding_mask=kv_padding_mask,
            need_weights=False,
        )

        x = q + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class SelfAttentionBlock(nn.Module):
    """
    Standard latent self-attention block with residual connection and FFN.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_dropout,
            batch_first=True,
        )
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, ff_mult=ff_mult, dropout=mlp_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_in = self.norm_attn(x)
        attn_out, _ = self.attn(x_in, x_in, x_in, need_weights=False)
        x = x + attn_out
        x = x + self.ff(self.norm_ff(x))
        return x


class ConditionalPointPerceiver(nn.Module):
    """
    Perceiver-style backbone for conditional point-cloud velocity prediction.

    High-level flow:
      1) Build query-state tokens from (coords, x_t, t)
      2) Build sparse sensor tokens from (obs_coords, obs_values, obs_field_ids)
      3) Concatenate them into one input token set
      4) Cross-attend a small learned latent array to the full token set
      5) Process latents with several self-attention blocks
      6) Decode per-point velocity from the latents using output query tokens

    This keeps the external forward signature identical to the existing backbone,
    so the outer flow / RF wrapper does not need to change.
    """
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        latent_dim: int = 256,
        num_latents: int = 128,
        num_heads: int = 8,
        num_latent_blocks: int = 4,
        field_embed_dim: int = 32,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        decode_chunk_size: Optional[int] = 4096,
        share_query_proj: bool = False,
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.decode_chunk_size = decode_chunk_size

        # Field-id embedding lets the model know which physical quantity
        # each sparse sensor measures.
        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        # Query-state token = [coords, x_t, t]
        self.query_in_proj = make_mlp(
            in_dim=coord_dim + n_fields + 1,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Sparse sensor token = [obs_coords, obs_value, field_embedding]
        self.sensor_proj = make_mlp(
            in_dim=coord_dim + 1 + field_embed_dim,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Decoder queries can either share or not share the encoder projection.
        if share_query_proj:
            self.query_out_proj = self.query_in_proj
        else:
            self.query_out_proj = make_mlp(
                in_dim=coord_dim + n_fields + 1,
                hidden_dim=latent_dim,
                out_dim=latent_dim,
                depth=3,
            )

        # Learned latent array used by the Perceiver bottleneck.
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim)
        )

        # Encoder: latents attend to all input tokens.
        self.input_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Latent processing blocks.
        self.latent_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=latent_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_latent_blocks)
        ])

        # Decoder: output query points attend to latent memory.
        self.output_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Final pointwise velocity head.
        self.head = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(latent_dim, n_fields),
        )

    def _build_query_tokens(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        proj: nn.Module,
    ) -> torch.Tensor:
        """
        Build per-point query tokens from coordinates, current field state, and flow time.
        """
        bsz, n_pts, _ = x_t.shape
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        token_in = torch.cat([coords, x_t, t_feat], dim=-1)
        return proj(token_in)

    def _build_sensor_tokens(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build sparse sensor tokens from:
          - sensor location
          - observed scalar value
          - field-id embedding
        """
        safe_field_ids = obs_field_ids.clamp_min(0)
        field_feat = self.field_embed(safe_field_ids)
        field_feat = field_feat * obs_mask.unsqueeze(-1)

        sensor_in = torch.cat([obs_coords, obs_values, field_feat], dim=-1)
        sensor_tokens = self.sensor_proj(sensor_in)

        # Zero padded sensor slots so they do not inject junk features.
        sensor_tokens = sensor_tokens * obs_mask.unsqueeze(-1)
        return sensor_tokens

    def _encode_latents(
        self,
        query_tokens: torch.Tensor,
        sensor_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode all input information into the latent bottleneck.

        query_tokens : [B, N, D]
        sensor_tokens: [B, M, D]
        obs_mask     : [B, M]
        """
        bsz, n_query, _ = query_tokens.shape

        # Concatenate query-state tokens and sparse sensor tokens.
        input_tokens = torch.cat([query_tokens, sensor_tokens], dim=1)  # [B, N+M, D]

        # Query tokens are always valid; only sensor tokens may be padded.
        query_keep_mask = torch.zeros(
            bsz, n_query, device=query_tokens.device, dtype=torch.bool
        )
        sensor_padding_mask = ~obs_mask.bool()
        kv_padding_mask = torch.cat([query_keep_mask, sensor_padding_mask], dim=1)

        # Expand learned latent array across the batch.
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)

        # Encode into latents.
        latents = self.input_cross_attn(
            q=latents,
            kv=input_tokens,
            kv_padding_mask=kv_padding_mask,
        )

        # Process only in latent space from now on.
        for block in self.latent_blocks:
            latents = block(latents)

        return latents

    def _decode_queries_chunked(
        self,
        latents: torch.Tensor,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode per-point outputs in chunks to reduce memory during full-resolution reconstruction. 
        Training usually uses a smaller n_query_points and may not need chunking, but reconstruction on all ~40k points can benefit from it.
        """
        n_pts = coords.shape[1]

        if self.decode_chunk_size is None or n_pts <= self.decode_chunk_size:
            query_tokens = self._build_query_tokens(t, x_t, coords, self.query_out_proj)
            decoded = self.output_cross_attn(q=query_tokens, kv=latents, kv_padding_mask=None)
            return self.head(decoded)

        outputs = []
        for start in range(0, n_pts, self.decode_chunk_size):
            end = min(start + self.decode_chunk_size, n_pts)

            coords_chunk = coords[:, start:end]
            x_t_chunk = x_t[:, start:end]

            query_tokens = self._build_query_tokens(t, x_t_chunk, coords_chunk, self.query_out_proj)
            decoded = self.output_cross_attn(q=query_tokens, kv=latents, kv_padding_mask=None)
            outputs.append(self.head(decoded))

        return torch.cat(outputs, dim=1)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Build query-state tokens for the encoder.
        query_tokens = self._build_query_tokens(t, x_t, coords, self.query_in_proj)

        # Build sparse sensor tokens.
        sensor_tokens = self._build_sensor_tokens(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )

        # Encode all information into latent memory.
        latents = self._encode_latents(
            query_tokens=query_tokens,
            sensor_tokens=sensor_tokens,
            obs_mask=obs_mask,
        )

        # Decode the per-point velocity field from latent memory.
        return self._decode_queries_chunked(
            latents=latents,
            t=t,
            x_t=x_t,
            coords=coords,
        )


# ------------------------------
# Global-Local backbone
# ------------------------------
class ConditionalPointHybridLocalGlobalRBF(nn.Module):
    """
    Hybrid local-global backbone for conditional point-cloud FFM.

    Core Pipeline:
      1) Tokenization: Build sparse sensor tokens from (obs_coords, obs_values, obs_field_ids).
      2) Global Latent Encoding: A learned latent array cross-attends to the sparse sensor tokens,
         processing the field globally.
      3) Double-Dip Refinement: The sparse sensor tokens cross-attend back to the processed latents,
         yielding globally enriched local sensor tokens.
      4) Query Point Aggregation: Gather these enriched sensor tokens to arbitrary query points.
         Supported gather modes:
           - "rbf": Full dense RBF distance-based aggregation.
           - "topk_rbf": Sparse K-Nearest Neighbor RBF aggregation.
           - "topk_rbf_gate": Top-K RBF aggregation modulated by a learned query-sensor content gate.
           - "topk_rbf_ptlocal": Top-K RBF with a lightweight sensor-side local graph refinement.
           - "topk_rbf_glres": Top-K RBF plus cheap global residual readout/scaffold terms.
      5) Global Summary: Extract a global summary from the latents (via 'cls' or 'mean') and 
         concatenate it separately to every query point.
         The latent summary / CLS-like token acts strictly as a concatenated global feature 

    Hardware & Optimization Context:
      - neighbor_backend: Supports "torch" (standard pairwise matrices) and "keops" (LazyTensors).
      - KeOps Integration: The "keops" backend fundamentally eliminates the O(B * N * M) memory 
        bottleneck during pairwise distance computations, reducing it to O(N + M). This allows 
        for massive point clouds and largely removes the need for 'gather_query_chunk_size' loops.
      - Memory Layout: Inputs to KeOps routines are strictly enforced as `.contiguous()` to 
        prevent silent C++ reallocation bottlenecks.
    """
    def __init__(
        self,
        n_fields: int,
        coord_dim: int = 3,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        field_embed_dim: int = 32,
        latent_dim: int = 256,
        num_latents: int = 64,
        num_heads: int = 8,
        num_latent_blocks: int = 3,
        ff_mult: int = 4,
        attn_dropout: float = 0.0,
        mlp_dropout: float = 0.0,
        rbf_sigma: float = 0.05,
        summary_type: str = "cls",   # ["cls", "mean"]

        gather_mode: str = "rbf",    # ["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"]
        gather_topk: int = 32,
        gather_query_chunk_size: Optional[int] = None,
        learnable_rbf_sigma: bool = False,
        neighbor_backend: str = "torch",      # ["auto", "torch", "keops"]

        sensor_local_topk: int = 8,
        sensor_local_dropout: float = 0.0,
        use_fourier_pe: bool = False,
        fourier_pe_num_bands: int = 32,
        fourier_pe_max_freq: float = 64.0,
    ) -> None:
        super().__init__()

        if summary_type not in ["cls", "mean"]:
            raise ValueError(f"summary_type must be 'cls' or 'mean', got {summary_type}")

        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.summary_type = summary_type
        self.use_fourier_pe = use_fourier_pe
        self.pos_enc = FourierPositionalEncoding(
            coord_dim, num_bands=fourier_pe_num_bands, max_freq=fourier_pe_max_freq
        ) if use_fourier_pe else None
        self.coord_feat_dim = self.pos_enc.out_dim if self.pos_enc is not None else coord_dim

        gather_modes = ["rbf", "topk_rbf", "topk_rbf_gate", "topk_rbf_ptlocal", "topk_rbf_glres"]
        if gather_mode not in gather_modes:
            raise ValueError(
                f"gather_mode must be one of {gather_modes}, got {gather_mode}"
            )
        if neighbor_backend not in ["auto", "torch", "keops"]:
            raise ValueError(
                f"neighbor_backend must be one of ['auto', 'torch', 'keops'], got {neighbor_backend}"
            )
        self.gather_mode = gather_mode
        self.gather_topk = int(gather_topk)
        self.gather_query_chunk_size = gather_query_chunk_size
        self.learnable_rbf_sigma = learnable_rbf_sigma
        self.neighbor_backend = neighbor_backend

        if self.gather_mode == "rbf": print(f"\nThe gather mode is {gather_mode} as default choice.\n")
        else: print(f"\nNOTICE: The gather mode is {gather_mode} with top-k {gather_topk} !!!\n")

        # Only build the heavy query-side gate when the gate mode is actually selected.
        if self.gather_mode == "topk_rbf_gate":
            self.query_to_cond = nn.Linear(hidden_dim, cond_dim, bias=False)

            # Scalar query-neighbor reweighting.
            gate_in_dim = cond_dim + cond_dim + coord_dim + 1
            self.gather_gate = nn.Sequential(
                nn.Linear(gate_in_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )

        if self.gather_topk < 1:
            raise ValueError(f"gather_topk must be >= 1, got {self.gather_topk}")
        # Optional learnable locality scale
        if learnable_rbf_sigma:
            self.log_rbf_sigma = nn.Parameter(torch.log(torch.tensor(float(rbf_sigma))))
        # else:
        #     self.register_buffer("_fixed_rbf_sigma", torch.tensor(float(rbf_sigma)))
        #     self.log_rbf_sigma = None

        self.sensor_local_topk = int(sensor_local_topk)
        self.sensor_local_dropout_p = float(sensor_local_dropout)

        if self.sensor_local_topk < 1:
            raise ValueError(f"sensor_local_topk must be >= 1, got {self.sensor_local_topk}")

        # -------------------------
        # Point/query branch
        # -------------------------
        # Query point token from [coords, x_t, t]
        self.point_encoder = make_mlp(
            in_dim=self.coord_feat_dim + n_fields + 1,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=3,
        )

        # -------------------------
        # Sparse sensor branch
        # -------------------------
        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        # Initial sparse sensor token from [obs_coords, obs_value, field_embed]
        self.sensor_in_proj = make_mlp(
            in_dim=coord_dim + 1 + field_embed_dim,
            hidden_dim=latent_dim,
            out_dim=latent_dim,
            depth=3,
        )

        # Project the refined sensor tokens to the local conditioning width
        # used by the RBF gather.
        self.sensor_out_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=cond_dim,
            out_dim=cond_dim,
            depth=2,
        )

        # --------------------------------------------------
        # Optional sensor-side local refinement block Used only in gather_mode == "topk_rbf_ptlocal"
        # This is intentionally placed AFTER sensor_out_proj so it works on cond_dim features, 
        # which keeps memory and compute lower than refining in latent_dim.
        # --------------------------------------------------
        if self.gather_mode == "topk_rbf_ptlocal":
            self.sensor_local_q = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_k = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_v = nn.Linear(cond_dim, cond_dim, bias=False)
            # Relative position encoding: [dx, dy, dz, ||d||]
            self.sensor_local_pos = make_mlp(
                in_dim=coord_dim + 1,
                hidden_dim=cond_dim,
                out_dim=cond_dim,
                depth=2,
            )
            # Lightweight Point-Transformer-style scalar attention over local neighbors.
            self.sensor_local_attn = nn.Sequential(
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
            self.sensor_local_out = nn.Linear(cond_dim, cond_dim, bias=False)
            self.sensor_local_dropout = nn.Dropout(sensor_local_dropout)
            self.sensor_local_norm = nn.LayerNorm(cond_dim)

        if self.gather_mode == "topk_rbf_glres":
            # Cheap query-to-latent readout: attends over L latents, not K sensors per query.
            self.query_readout_in = nn.Linear(hidden_dim, latent_dim, bias=False)
            self.query_latent_readout = CrossAttentionBlock(
                dim=latent_dim,
                num_heads=max(1, min(num_heads, 4)),
                ff_mult=max(1, ff_mult // 2),
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            self.query_readout_out = nn.Linear(latent_dim, hidden_dim, bias=False)
            self.query_readout_scale = nn.Parameter(torch.tensor(0.0))

            # Coarse scaffold is summary-driven and pointwise, so it avoids [B, N, K, C] tensors.
            self.coarse_film = nn.Linear(hidden_dim, 2 * hidden_dim)
            self.coarse_head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(hidden_dim, n_fields),
            )
            self.coarse_scale = nn.Parameter(torch.tensor(0.0))

            # Sensor importance is computed once per refined sensor token, then gathered as scalars.
            self.sensor_importance = nn.Sequential(
                nn.LayerNorm(cond_dim),
                nn.Linear(cond_dim, cond_dim),
                nn.GELU(),
                nn.Linear(cond_dim, 1),
            )
            self.sensor_importance_scale = nn.Parameter(torch.tensor(0.0))

        # -------------------------
        # Latent global processor
        # -------------------------
        self.latents = nn.Parameter(
            torch.randn(num_latents, latent_dim) / math.sqrt(latent_dim)
        )

        # Latents attend to sparse sensor tokens
        self.input_cross_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Process latents in latent space
        self.latent_blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=latent_dim,
                num_heads=num_heads,
                ff_mult=ff_mult,
                attn_dropout=attn_dropout,
                mlp_dropout=mlp_dropout,
            )
            for _ in range(num_latent_blocks)
        ])

        # Double-dip: refined local sensor tokens query the processed latents
        self.sensor_back_attn = CrossAttentionBlock(
            dim=latent_dim,
            num_heads=num_heads,
            ff_mult=ff_mult,
            attn_dropout=attn_dropout,
            mlp_dropout=mlp_dropout,
        )

        # Separate projection for the latent summary used as a global feature
        self.summary_proj = make_mlp(
            in_dim=latent_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            depth=2,
        )

        # -------------------------
        # Final velocity head
        # -------------------------
        self.head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_dim, n_fields),
        )

    def _build_sensor_tokens(
        self,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build sparse sensor tokens from:
          - sensor coordinates
          - observed scalar value
          - field identity embedding
        """
        safe_field_ids = obs_field_ids.clamp_min(0)
        field_feat = self.field_embed(safe_field_ids)                 # [B, M, E]
        field_feat = field_feat * obs_mask.unsqueeze(-1)             # zero padded rows

        sensor_in = torch.cat([obs_coords, obs_values, field_feat], dim=-1)
        sensor_tokens = self.sensor_in_proj(sensor_in)               # [B, M, D]
        sensor_tokens = sensor_tokens * obs_mask.unsqueeze(-1)
        return sensor_tokens

    def _encode_latents(
        self,
        sensor_tokens: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Let the learned latent array absorb and process the sparse sensor set.
        """
        bsz = sensor_tokens.shape[0]

        # Expand learned latents across the batch
        latents = self.latents.unsqueeze(0).expand(bsz, -1, -1)      # [B, L, D]

        # key_padding_mask: True means "ignore this token"
        sensor_padding_mask = ~obs_mask.bool()

        # Latents attend to sparse sensor tokens
        latents = self.input_cross_attn(
            q=latents,
            kv=sensor_tokens,
            kv_padding_mask=sensor_padding_mask,
        )

        # Process in latent space
        for block in self.latent_blocks:
            latents = block(latents)

        return latents

    def _extract_global_summary(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert the latent array into one global summary vector.

        If summary_type == 'cls', the last latent slot is treated as the summary token.
        If summary_type == 'mean', use the mean of all latent slots.
        """
        if self.summary_type == "cls":
            summary = latents[:, -1]         # [B, D]
        else:
            summary = latents.mean(dim=1)    # [B, D]

        return self.summary_proj(summary)    # [B, H]

    def _readout_query_global_chunked(
        self,
        point_feat: torch.Tensor,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        Query-to-latent readout in chunks. This is O(B * N * L), avoiding query-sensor
        [B, N, K, C] feature materialization.
        """
        n_query = point_feat.shape[1]
        chunk_size = self.gather_query_chunk_size
        if chunk_size is None and n_query > 4096:
            chunk_size = 4096

        if chunk_size is None or n_query <= chunk_size:
            q = self.query_readout_in(point_feat)
            readout = self.query_latent_readout(q=q, kv=latents, kv_padding_mask=None)
            return self.query_readout_out(readout)

        outputs = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)
            q = self.query_readout_in(point_feat[:, start:end])
            readout = self.query_latent_readout(q=q, kv=latents, kv_padding_mask=None)
            outputs.append(self.query_readout_out(readout))
        return torch.cat(outputs, dim=1)

    def _predict_global_coarse(
        self,
        point_feat: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        gamma, beta = self.coarse_film(global_feat).chunk(2, dim=-1)
        coarse_feat = (
            point_feat * (1.0 + torch.tanh(gamma).unsqueeze(1))
            + beta.unsqueeze(1)
        )
        return self.coarse_head(coarse_feat)

    def _compute_sensor_importance_bias(
        self,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
    ) -> torch.Tensor:
        bias = self.sensor_importance(refined_sensor_feat).squeeze(-1)
        return bias * obs_mask.to(dtype=bias.dtype)

    def _use_keops(self) -> bool:
        """
        Decide whether to use KeOps.

        - rbf mode can benefit a lot from KeOps soft reductions
        - topk modes can use KeOps KNN search
        """
        if self.neighbor_backend == "torch":
            return False

        if self.neighbor_backend == "keops":
            if LazyTensor is None:
                raise ImportError(
                    "neighbor_backend='keops' was requested, but pykeops is not installed."
                )
            return True

        # auto
        return LazyTensor is not None

    def _aggregate_rbf_keops(
        self,
        query_coords: torch.Tensor,         # [B, N, D]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
    ) -> torch.Tensor:
        """
        Full RBF gather using KeOps sumsoftmaxweight, without building the dense [B, N, M] matrix.
        """
        sigma = torch.exp(self.log_rbf_sigma).clamp_min(1e-6) if self.learnable_rbf_sigma else self.rbf_sigma
        gamma = 1.0 / (2 * sigma ** 2 + 1e-12)

        # --- Force contiguous memory for KeOps ---
        query_coords = query_coords.contiguous()
        obs_coords = obs_coords.contiguous()
        refined_sensor_feat = refined_sensor_feat.contiguous()
        # -----------------------------------------

        # KeOps symbolic tensors
        x_i = LazyTensor(query_coords[:, :, None, :])                 # [B, N, 1, D]
        y_j = LazyTensor(obs_coords[:, None, :, :])                   # [B, 1, M, D]
        v_j = LazyTensor(refined_sensor_feat[:, None, :, :])          # [B, 1, M, Cc]

        # Scalar logits: -gamma * ||x_i - y_j||^2
        sqdist_ij = ((x_i - y_j) ** 2).sum(-1)                        # [B, N, M, 1]
        logits_ij = -gamma * sqdist_ij

        # Mask invalid sensor slots by adding a large negative number
        mask_j = LazyTensor(obs_mask[:, None, :, None].to(query_coords.dtype).contiguous())   # [B, 1, M, 1]
        logits_ij = logits_ij + (mask_j - 1.0) * 1e6

        # Softmax-weighted sum over the sensor axis.
        # With one batch dimension, the j-axis is dim=2.
        local_cond = logits_ij.sumsoftmaxweight(v_j, dim=2)           # [B, N, Cc]
        return local_cond

    def _knn_search_keops(
        self,
        query_coords: torch.Tensor,         # [B, N, D]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Top-k neighbor search using KeOps Kmin_argKmin.
        """

        # --- Force contiguous memory for KeOps ---
        query_coords = query_coords.contiguous()
        obs_coords = obs_coords.contiguous()
        # -----------------------------------------

        x_i = LazyTensor(query_coords[:, :, None, :])                 # [B, N, 1, D]
        y_j = LazyTensor(obs_coords[:, None, :, :])                   # [B, 1, M, D]

        sqdist_ij = ((x_i - y_j) ** 2).sum(-1)                        # [B, N, M, 1]

        # Mask invalid sensor slots
        mask_j = LazyTensor(obs_mask[:, None, :, None].to(query_coords.dtype).contiguous())
        sqdist_ij = sqdist_ij + (1.0 - mask_j) * 1e6

        # With one batch dimension, the j-axis is dim=2.
        topk_d2, topk_idx = sqdist_ij.Kmin_argKmin(K=k, dim=2)

        # KeOps can return indices in a non-long dtype; convert explicitly.
        topk_idx = topk_idx.long()

        topk_sensor_feat = batched_gather_3d(refined_sensor_feat, topk_idx)
        topk_sensor_coords = batched_gather_3d(obs_coords, topk_idx)
        topk_valid = batched_gather_2d(obs_mask, topk_idx).bool()

        return topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _knn_search_torch(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Fallback KNN search using torch.cdist + torch.topk.
        """
        d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
        large = torch.full_like(d2, 1e6)
        d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

        topk_d2, topk_idx = torch.topk(d2, k=k, dim=-1, largest=False)

        topk_sensor_feat = batched_gather_3d(refined_sensor_feat, topk_idx)
        topk_sensor_coords = batched_gather_3d(obs_coords, topk_idx)
        topk_valid = batched_gather_2d(obs_mask, topk_idx).bool()

        return topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _get_topk_neighbors(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unified top-k neighbor retrieval.
        """
        if self._use_keops():
            return self._knn_search_keops(
                query_coords=query_coords,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                k=k,
            )

        return self._knn_search_torch(
            query_coords=query_coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            k=k,
        )

    def _sensor_local_refine(
        self,
        sensor_coords: torch.Tensor,      # [B, M, D]
        sensor_feat: torch.Tensor,        # [B, M, Cc]
        obs_mask: torch.Tensor,           # [B, M]
    ) -> torch.Tensor:
        """
        Point-Transformer-style local refinement on the sensor graph.

        - This operates on M sensors, not N query points, so its memory cost is much
          smaller than query-side gating.
        - It gives each refined sensor token awareness of its local sensor neighborhood
          before the final query-side top-k RBF gather.

        Implementation notes:
        - Uses the existing neighbor backend (torch / keops) through _get_topk_neighbors.
        - Uses K+1 neighbors and drops the first one, which is usually the sensor itself.
        """
        # Search one extra neighbor so we can discard self-neighbor.
        k_search = min(self.sensor_local_topk + 1, sensor_coords.shape[1])

        nbr_d2, _, nbr_feat, nbr_coords, nbr_valid = self._get_topk_neighbors(
            query_coords=sensor_coords,
            obs_coords=sensor_coords,
            refined_sensor_feat=sensor_feat,
            obs_mask=obs_mask,
            k=k_search,
        )

        # Drop the first neighbor slot, which is typically the point itself.
        if k_search > 1:
            nbr_d2 = nbr_d2[:, :, 1:]
            nbr_feat = nbr_feat[:, :, 1:]
            nbr_coords = nbr_coords[:, :, 1:]
            nbr_valid = nbr_valid[:, :, 1:]

        # If there was only one valid sensor total, keep the feature unchanged.
        if nbr_feat.shape[2] == 0:
            return sensor_feat

        q = self.sensor_local_q(sensor_feat).unsqueeze(2)   # [B, M, 1, Cc]
        k = self.sensor_local_k(nbr_feat)                   # [B, M, Ks, Cc]
        v = self.sensor_local_v(nbr_feat)                   # [B, M, Ks, Cc]

        rel = sensor_coords.unsqueeze(2) - nbr_coords       # [B, M, Ks, D]
        rel_dist = torch.sqrt(nbr_d2.clamp_min(0.0)).unsqueeze(-1)  # [B, M, Ks, 1]
        pos = self.sensor_local_pos(torch.cat([rel, rel_dist], dim=-1))  # [B, M, Ks, Cc]

        # Lightweight Point-Transformer-style attention:
        # attention is driven by query-key difference plus relative position.
        attn_logits = self.sensor_local_attn(torch.tanh(q - k + pos)).squeeze(-1)  # [B, M, Ks]
        attn_logits = attn_logits.masked_fill(~nbr_valid, -1e9)
        attn = torch.softmax(attn_logits, dim=-1)

        update = torch.sum(attn.unsqueeze(-1) * (v + pos), dim=2)       # [B, M, Cc]
        out = self.sensor_local_norm(sensor_feat + self.sensor_local_dropout(self.sensor_local_out(update)))

        # Keep padded sensor rows zeroed out.
        out = out * obs_mask.unsqueeze(-1)
        return out

    def _aggregate_chunk(
        self,
        query_coords: torch.Tensor,         # [B, Nc, D]
        query_feat: torch.Tensor,           # [B, Nc, H]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
        sensor_importance_bias: Optional[torch.Tensor] = None,  # [B, M]
    ) -> torch.Tensor:
        """
        Aggregate one query chunk.
        """
        # sigma = self._get_rbf_sigma()
        sigma = torch.exp(self.log_rbf_sigma).clamp_min(1e-6) if self.learnable_rbf_sigma else self.rbf_sigma

        # --------------------------------------------------
        # Default: full RBF gather
        # --------------------------------------------------
        if self.gather_mode == "rbf":
            if self._use_keops():
                return self._aggregate_rbf_keops(
                    query_coords=query_coords,
                    obs_coords=obs_coords,
                    refined_sensor_feat=refined_sensor_feat,
                    obs_mask=obs_mask,
                )

            d2 = torch.cdist(query_coords, obs_coords, p=2.0) ** 2
            large = torch.full_like(d2, 1e6)
            d2 = torch.where(obs_mask.unsqueeze(1) > 0, d2, large)

            logits = -d2 / (2 * sigma ** 2 + 1e-12)
            weights = torch.softmax(logits, dim=-1)
            return torch.einsum("bnm,bmd->bnd", weights, refined_sensor_feat)

        # --------------------------------------------------
        # top-k modes
        # --------------------------------------------------
        k = min(self.gather_topk, obs_coords.shape[1])

        topk_d2, topk_idx, topk_sensor_feat, topk_sensor_coords, topk_valid = self._get_topk_neighbors(
            query_coords=query_coords,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
            k=k,
        )

        logits = -topk_d2 / (2 * sigma ** 2 + 1e-12)

        if self.gather_mode == "topk_rbf_gate":
            query_cond = self.query_to_cond(query_feat)                    # [B, Nc, Cc]
            query_cond = query_cond.unsqueeze(2).expand(-1, -1, k, -1)    # [B, Nc, k, Cc]

            rel = query_coords.unsqueeze(2) - topk_sensor_coords           # [B, Nc, k, D]
            rel_dist = torch.sqrt(topk_d2.clamp_min(0.0)).unsqueeze(-1)    # [B, Nc, k, 1]

            gate_in = torch.cat([query_cond, topk_sensor_feat, rel, rel_dist], dim=-1)
            gate_logits = self.gather_gate(gate_in).squeeze(-1)            # [B, Nc, k]

            logits = logits + gate_logits

        if sensor_importance_bias is not None:
            topk_sensor_bias = batched_gather_2d(sensor_importance_bias, topk_idx)
            logits = logits + self.sensor_importance_scale * topk_sensor_bias

        logits = logits.masked_fill(~topk_valid, -1e9)
        weights = torch.softmax(logits, dim=-1)
        local_cond = torch.sum(weights.unsqueeze(-1) * topk_sensor_feat, dim=2)
        return local_cond

    def aggregate_sparse_obs(
        self,
        query_coords: torch.Tensor,
        query_feat: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        sensor_importance_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Gather the globally enriched local sensor features back to query points.

        Policy:
          - rbf: with KeOps, chunking can usually be disabled
          - topk_rbf: with KeOps, chunking can usually be disabled
          - topk_rbf_gate: still keep optional chunking because gate tensors are [B, N, K, U]
        """
        n_query = query_coords.shape[1]

        if self.gather_mode == "topk_rbf_gate":
            # Gate mode still benefits from chunking because it builds [B, N, K, ...] tensors.
            chunk_size = self.gather_query_chunk_size if self.gather_query_chunk_size is not None else 2048
        else:
            # rbf / topk_rbf / topk_rbf_ptlocal all keep the cheaper gather path.
            chunk_size = self.gather_query_chunk_size

        if chunk_size is None or n_query <= chunk_size:
            return self._aggregate_chunk(
                query_coords=query_coords,
                query_feat=query_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )

        outputs = []
        for start in range(0, n_query, chunk_size):
            end = min(start + chunk_size, n_query)

            local_chunk = self._aggregate_chunk(
                query_coords=query_coords[:, start:end],
                query_feat=query_feat[:, start:end],
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )
            outputs.append(local_chunk)

        return torch.cat(outputs, dim=1)

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Output:
            velocity field of shape [B, N, C]
        """
        bsz, n_pts, _ = x_t.shape

        # -------------------------
        # Query-point features
        # -------------------------
        t_feat = t.view(bsz, 1, 1).expand(bsz, n_pts, 1)
        coord_feat = self.pos_enc(coords) if self.pos_enc is not None else coords
        point_feat = self.point_encoder(torch.cat([coord_feat, x_t, t_feat], dim=-1))  # [B, N, H]

        # -------------------------
        # Local sensor tokens
        # -------------------------
        sensor_tokens = self._build_sensor_tokens(
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
        )  # [B, M, D]

        # -------------------------
        # Global latent processing
        # -------------------------
        latents = self._encode_latents(sensor_tokens=sensor_tokens, obs_mask=obs_mask)  # [B, L, D]

        if self.gather_mode == "topk_rbf_glres":
            global_feat = self._extract_global_summary(latents)                 # [B, H]

        # -------------------------
        # Double-dip refinement:
        # sensor tokens query back into the latent memory
        # -------------------------
        refined_sensor_tokens = self.sensor_back_attn(
            q=sensor_tokens,
            kv=latents,
            kv_padding_mask=None,
        )  # [B, M, D]

        # Zero out padded sensor rows again after attention
        refined_sensor_tokens = refined_sensor_tokens * obs_mask.unsqueeze(-1)

        # Project refined sensor tokens to the local conditioning width
        refined_sensor_feat = self.sensor_out_proj(refined_sensor_tokens)   # [B, M, cond_dim]
        refined_sensor_feat = refined_sensor_feat * obs_mask.unsqueeze(-1)

        if self.gather_mode == "topk_rbf_glres":
            sensor_importance_bias = self._compute_sensor_importance_bias(
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
            )

            local_cond = self.aggregate_sparse_obs(
                query_coords=coords,
                query_feat=point_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
                sensor_importance_bias=sensor_importance_bias,
            )  # [B, N, cond_dim]

            query_global = self._readout_query_global_chunked(point_feat, latents)  # [B, N, H]
            global_for_head = global_feat.unsqueeze(1) + self.query_readout_scale * query_global
            coarse_pred = self.coarse_scale * self._predict_global_coarse(point_feat, global_feat)

            residual = self.head(torch.cat([point_feat, global_for_head, local_cond], dim=-1))
            return coarse_pred + residual

        # Optional sensor-side local graph refinement.
        if self.gather_mode == "topk_rbf_ptlocal":
            refined_sensor_feat = self._sensor_local_refine(
                sensor_coords=obs_coords,
                sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,)

        # -------------------------
        # Gather back to queries
        # -------------------------
        local_cond = self.aggregate_sparse_obs(
            query_coords=coords,
            query_feat=point_feat,
            obs_coords=obs_coords,
            refined_sensor_feat=refined_sensor_feat,
            obs_mask=obs_mask,
        )  # [B, N, cond_dim]

        # -------------------------
        # Separate global summary
        # -------------------------
        global_feat = self._extract_global_summary(latents)                 # [B, H]
        global_feat = global_feat.unsqueeze(1).expand(bsz, n_pts, -1)      # [B, N, H]

        # -------------------------
        # Final velocity prediction
        # -------------------------
        out = self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))
        return out


# ------------------------------
# FNO backbone

# Gaussian splatting condition: FNOs truncate the Fourier series to a specific number of low-frequency modes (n_modes_x, n_modes_y), 
# they are inherently low-pass filters. They struggle to resolve sharp, single-pixel spikes. 
# Feeding a grid of sharp spikes into an FNO often causes ringing artifacts (the Gibbs phenomenon) 
# and makes it difficult for the network to understand the spatial influence of that sensor.

# ------------------------------
class FNO(nn.Module):
    """
    Grid-based FNO backbone compatible with the existing generalized sparse conditioning API.

    Input contract:
        t               : [B]
        x_t             : [B, N, C]
        coords          : [B, N, D]      (unused by FNO forward; kept for API compatibility)
        obs_coords      : [B, M, D]      (unused by FNO forward; kept for API compatibility)
        obs_values      : [B, M, 1]
        obs_mask        : [B, M]
        obs_field_ids   : [B, M]
        obs_indices     : [B, M]         linear point indices in the flattened grid

    Output:
        velocity field  : [B, N, C]

    Notes:
    - The FNO operates on a regular mesh, so x_t is converted from point-cloud layout [B, N, C] to grid layout [B, C, Num_y, Num_x].
    - If the HDF5 point order is not already row-major, the backbone derives
      a stable row-major permutation from coords, applies it internally, and
      returns the velocity in the original point order.
    - Sparse conditioning is rasterized into dense per-field observation maps and mask/support maps before being concatenated to the FNO input.
    - Optional Gaussian splatting can soften one-pixel conditioning impulses before they are concatenated to the spectral input.
    - New FNO runs use three condition-channel groups. This intentionally
      changes the FNO input channel count, so older FNO checkpoints trained
      with the previous two-group conditioning layout are not compatible.
    """

    def __init__(
        self,
        n_fields: int,
        Num_x: int,
        Num_y: int,
        n_modes_x: int = 32,
        n_modes_y: int = 8,
        hidden_channels: int = 64,
        n_layers: int = 4,
        use_grid_positional_embedding: bool = True,
        condition_blur: bool = False,
        condition_blur_kernel: int = 5,
        condition_blur_sigma: float = 1.0,
    ) -> None:
        super().__init__()

        self.n_fields = n_fields
        self.Num_x = int(Num_x)
        self.Num_y = int(Num_y)
        self.condition_blur = bool(condition_blur)
        self.condition_blur_kernel = int(condition_blur_kernel)
        self.condition_blur_sigma = float(condition_blur_sigma)

        if self.condition_blur_kernel < 1 or self.condition_blur_kernel % 2 == 0:
            raise ValueError(
                f"condition_blur_kernel must be a positive odd integer, got {self.condition_blur_kernel}."
            )
        if self.condition_blur_sigma <= 0.0:
            raise ValueError(
                f"condition_blur_sigma must be > 0, got {self.condition_blur_sigma}."
            )

        # Keep the blur kernel off the persistent state dict so old checkpoints
        # still load with strict=True.
        self.register_buffer("_condition_blur_kernel_cache", torch.empty(0), persistent=False)
        self.register_buffer("_grid_order_cache", torch.empty(0, dtype=torch.long), persistent=False)
        self.register_buffer("_point_to_grid_cache", torch.empty(0, dtype=torch.long), persistent=False)

        # FNO input channels:
        #   current state x_t         -> C
        #   scalar time channel       -> 1
        #   normalized observed values        -> C
        #   support-weighted observed values  -> C
        #   soft observation support maps     -> C
        # total = 4C + 1
        in_channels = 4 * n_fields + 1

        self.fno = NeuralOpFNO(
            n_modes=(n_modes_y, n_modes_x),   # tensor layout is [B, C, Num_y, Num_x]
            in_channels=in_channels,
            out_channels=n_fields,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding="grid" if use_grid_positional_embedding else None,
        )

    def _get_grid_permutation(
        self,
        coords: torch.Tensor,
        decimals: int = 6,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return index maps between the dataset point order and row-major grid order.

        grid_order[g] = original point index for row-major grid cell g.
        point_to_grid[p] = row-major grid cell for original point index p.

        The cache assumes the FNO sees one fixed mesh, which is true for this
        turbulent-combustion demo. It is non-persistent and rebuilt after device
        moves or checkpoint loading.
        """
        n_pts = coords.shape[1]
        expected = self.Num_x * self.Num_y
        if n_pts != expected:
            raise ValueError(
                f"FNO backbone expected N = Num_x * Num_y = {expected}, got {n_pts}."
            )

        cached_order = self._grid_order_cache
        cached_point_to_grid = self._point_to_grid_cache
        if (
            cached_order.numel() == n_pts
            and cached_point_to_grid.numel() == n_pts
            and cached_order.device == coords.device
            and cached_point_to_grid.device == coords.device
        ):
            return cached_order, cached_point_to_grid

        coords0 = coords[0, :, :2].detach()
        scale = float(10 ** decimals)
        x = torch.round(coords0[:, 0] * scale) / scale
        y = torch.round(coords0[:, 1] * scale) / scale

        unique_x, x_rank = torch.unique(x, sorted=True, return_inverse=True)
        unique_y, y_rank = torch.unique(y, sorted=True, return_inverse=True)
        if unique_x.numel() != self.Num_x or unique_y.numel() != self.Num_y:
            raise ValueError(
                "FNO could not infer the requested grid from coords: "
                f"detected unique (x, y)=({unique_x.numel()}, {unique_y.numel()}), "
                f"but expected (Num_x, Num_y)=({self.Num_x}, {self.Num_y})."
            )

        point_to_grid = (y_rank.long() * self.Num_x + x_rank.long()).contiguous()
        if torch.unique(point_to_grid).numel() != n_pts:
            raise ValueError(
                "FNO could not infer a complete tensor-product grid from coords. "
                "The coordinate set has duplicate or missing (x, y) grid cells, so "
                "an internal row-major permutation would be ambiguous."
            )

        grid_order = torch.argsort(point_to_grid).contiguous()
        self._grid_order_cache = grid_order
        self._point_to_grid_cache = point_to_grid
        return grid_order, point_to_grid

    def _get_condition_blur_kernel(
        self,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build and cache a depthwise 2D Gaussian kernel used to splat sparse
        conditioning impulses into a small local neighborhood.
        """
        kernel = self._condition_blur_kernel_cache
        if kernel.numel() > 0 and kernel.dtype == dtype and kernel.device == device:
            return kernel

        radius = self.condition_blur_kernel // 2
        coords_1d = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel_1d = torch.exp(-0.5 * (coords_1d / self.condition_blur_sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel_2d = kernel_2d / kernel_2d.sum().clamp_min(1e-12)

        kernel = kernel_2d.view(1, 1, self.condition_blur_kernel, self.condition_blur_kernel)
        kernel = kernel.expand(self.n_fields, 1, -1, -1).contiguous()
        self._condition_blur_kernel_cache = kernel
        return kernel

    def _blur_condition_maps(
        self,
        obs_value_maps: torch.Tensor,
        obs_mask_maps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Replace one-pixel conditioning maps with Gaussian splats.

        Returns three semantically different condition maps:
          - normalized/interpolated values on the original field scale;
          - support-weighted values, i.e. the blurred numerator;
          - soft support, i.e. the blurred mask.

        The support-weighted channel is deliberately separate from the
        normalized channel. It avoids presenting an unqualified finite-support
        normalized plateau/boundary to the spectral model, and lets the FNO
        distinguish "interpolated value" from "confidence/support."
        """
        kernel = self._get_condition_blur_kernel(
            dtype=obs_value_maps.dtype,
            device=obs_value_maps.device,
        )
        padding = self.condition_blur_kernel // 2

        blurred_mask_raw = F.conv2d(
            obs_mask_maps,
            kernel,
            padding=padding,
            groups=self.n_fields,
        )
        blurred_value_num = F.conv2d(
            obs_value_maps,
            kernel,
            padding=padding,
            groups=self.n_fields,
        )

        blurred_value_norm = blurred_value_num / blurred_mask_raw.clamp_min(1e-6)
        blurred_value_norm = torch.where(
            blurred_mask_raw > 0,
            blurred_value_norm,
            torch.zeros_like(blurred_value_norm),
        )
        return blurred_value_norm, blurred_value_num, blurred_mask_raw

    def _pointcloud_to_grid(
        self,
        x: torch.Tensor,
        grid_order: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert original point order [B, N, C] -> row-major grid [B, C, Num_y, Num_x].
        """
        bsz, n_pts, n_fields = x.shape
        expected = self.Num_x * self.Num_y
        if n_pts != expected:
            raise ValueError(
                f"FNO backbone expected N = Num_x * Num_y = {expected}, got {n_pts}."
            )

        x = x[:, grid_order, :]
        x_grid = x.reshape(bsz, self.Num_y, self.Num_x, n_fields)
        x_grid = x_grid.permute(0, 3, 1, 2).contiguous()
        return x_grid

    def _grid_to_pointcloud(
        self,
        x_grid: torch.Tensor,
        point_to_grid: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convert row-major grid [B, C, Num_y, Num_x] -> original point order [B, N, C].
        """
        bsz, n_fields, _, _ = x_grid.shape
        x = x_grid.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(bsz, self.Num_x * self.Num_y, n_fields)
        x = x[:, point_to_grid, :]
        return x

    def _build_condition_maps(
        self,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Rasterize sparse observations into dense grid-aligned maps.

        Returns:
            obs_value_norm_maps    : [B, C, Num_y, Num_x]
            obs_value_weighted_maps: [B, C, Num_y, Num_x]
            obs_support_maps       : [B, C, Num_y, Num_x]
        """
        bsz, _, _ = obs_values.shape
        n_pts = self.Num_x * self.Num_y

        obs_value_maps = torch.zeros(
            bsz, self.n_fields, n_pts, dtype=dtype, device=device
        )
        obs_mask_maps = torch.zeros(
            bsz, self.n_fields, n_pts, dtype=dtype, device=device
        )

        # Scatter sparse sensor values into the appropriate field-channel grid.
        for b in range(bsz):
            valid = obs_mask[b].bool()
            if not valid.any():
                continue

            idx = obs_indices[b, valid].long()
            fld = obs_field_ids[b, valid].long()
            val = obs_values[b, valid, 0]

            obs_value_maps[b, fld, idx] = val
            obs_mask_maps[b, fld, idx] = 1.0

        obs_value_maps = obs_value_maps.reshape(bsz, self.n_fields, self.Num_y, self.Num_x)
        obs_mask_maps = obs_mask_maps.reshape(bsz, self.n_fields, self.Num_y, self.Num_x)

        if self.condition_blur:
            return self._blur_condition_maps(
                obs_value_maps=obs_value_maps,
                obs_mask_maps=obs_mask_maps,
            )

        # Without blur, a point observation is both the normalized value and
        # the support-weighted value, with the binary mask carrying support.
        return obs_value_maps, obs_value_maps, obs_mask_maps

    def forward(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict the velocity field on the full regular grid.

        obs_indices is required because the sparse sensor values must be
        rasterized onto the fixed grid before being fed into the FNO.
        """
        if obs_indices is None:
            raise ValueError(
                "FNO.forward requires obs_indices so sparse observations can be "
                "placed onto the regular grid."
            )

        bsz = x_t.shape[0]
        grid_order, point_to_grid = self._get_grid_permutation(coords)

        # Convert the current state to a row-major grid. If the dataset is
        # stored in a different point order, the permutation is applied here
        # rather than mutating the shared point-cloud dataset.
        x_grid = self._pointcloud_to_grid(x_t, grid_order=grid_order)  # [B, C, Num_y, Num_x]
        # Broadcast time to a full grid channel.
        t_map = t.view(bsz, 1, 1, 1).expand(bsz, 1, self.Num_y, self.Num_x)

        # Convert sparse observations into dense field-aligned maps.
        obs_grid_indices = point_to_grid[obs_indices.long()]
        obs_value_norm_maps, obs_value_weighted_maps, obs_support_maps = self._build_condition_maps(
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_grid_indices,
            dtype=x_t.dtype,
            device=x_t.device,
        )

        # Concatenate:
        #   [current fields, time channel, normalized observed values,
        #    support-weighted observed values, soft support maps]
        fno_in = torch.cat(
            [x_grid, t_map, obs_value_norm_maps, obs_value_weighted_maps, obs_support_maps],
            dim=1,
        )
        # FNO predicts the velocity field on the regular grid.
        vel_grid = self.fno(fno_in)
        # Convert back to the standard point-cloud layout expected by the wrapper.
        vel = self._grid_to_pointcloud(vel_grid, point_to_grid=point_to_grid)
        return vel



# Model wrappers --------------------------------------

# Wrapper for Point-Cloud-Based models
class PointCloudFFM(nn.Module):
    """
    This block implements 1-Rectified Flow instead of the previous noisy
    Functional Flow Matching bridge.

    Core 1-RF idea: (https://github.com/gnobitab/RectifiedFlow)
        1) Draw a source sample x0 ~ prior
        2) Draw a target sample x1 from data
        3) Interpolate linearly: x_t = (1 - t) * x0 + t * x1
        4) Train the velocity model to predict the constant displacement x1 - x0
    """
    def __init__(self, model: nn.Module, prior: nn.Module, sigma_min: float = 1e-4):
        super().__init__()
        self.model = model
        self.prior = prior

        # Kept only so old checkpoints / YAML files do not break / It is not used in 1-RF.
        self.sigma_min = sigma_min

    def sample_source(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Draw a source sample x0 from the chosen prior on the query coordinates.
        This is the pi_0 endpoint in rectified flow.
        """
        return self.prior(coords, self.model.n_fields)

    def simulate(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        Straight-line interpolation between source x0 and target x1.

        x_t = (1 - t) * x0 + t * x1
        """
        alpha = t.view(-1, 1, 1)
        # print(f'alpha.shape: {alpha.shape}')
        # print(f'x0.shape: {x0.shape}')
        # print(f'x1.shape: {x1.shape}')
        return (1.0 - alpha) * x0 + alpha * x1

    def target_vector_field(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """
        1-RF target velocity is the constant straight-line displacement.

        v*(x_t, t) = x1 - x0
        """
        return x1 - x0

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
        # Sample x0 from the source prior for the current query coordinates.
        x0 = self.sample_source(coords)

        # Uniform time for standard 1-RF training.
        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)

        # Straight interpolation and constant target velocity.
        x_t = self.simulate(t, x0, x1)
        target = self.target_vector_field(x0, x1)

        # Predict the velocity under sparse conditioning.
        pred = self.model(t, x_t, coords, obs_coords, obs_values, obs_mask, obs_field_ids)

        # Standard supervised regression loss used in 1-RF.
        loss = F.mse_loss(pred, target)

        return loss, {
            "loss": float(loss.detach().cpu()),
            "target_rms": float(target.pow(2).mean().sqrt().detach().cpu()),
        }

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        n_steps: int = 8,
        clamp_indices: Optional[torch.Tensor] = None,
        ode_solver: str = "euler",
        obs_consistency_mode: str = "default_hard",
        obs_consistency_strength: float = 1.0,
        obs_consistency_sigma: float = 0.05,
        obs_consistency_schedule_power: float = 2.0,
        obs_consistency_final_clamp: bool = True,
        obs_consistency_chunk_size: int = 8192,
    ) -> torch.Tensor:
        """
        Integrate the learned rectified-flow ODE from x0 ~ prior to x1.

        Euler is the default solver because low-step Euler is the main use case
        for 1-RF. Heun is kept as an optional baseline / sanity check.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")

        bsz = coords.shape[0]
        x = self.sample_source(coords)
        obs_consistency_mode = normalize_obs_consistency_mode(obs_consistency_mode)
        if obs_consistency_mode != "none" and clamp_indices is None:
            if obs_consistency_mode in ("default_hard", "endpoint"):
                raise ValueError(
                    f"obs_consistency_mode={obs_consistency_mode!r} requires clamp_indices."
                )

        value_map = None
        mask_map = None
        if obs_consistency_mode == "endpoint":
            value_map, mask_map = build_pointwise_observation_maps(
                coords=coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
            )
        elif obs_consistency_mode == "endpoint_smooth":
            value_map, mask_map = build_smooth_observation_maps(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

        ts = torch.linspace(
            0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype
        )

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]

            # Velocity at the current state.
            v0 = self.model(t0, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
            if obs_consistency_mode in ("endpoint", "endpoint_smooth"):
                # RF clean-endpoint observation masking: guide x1_hat, then
                # convert the consistent endpoint back to a velocity.
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
                # Optional predictor-corrector step.
                x_euler = x + dt * v0
                t1 = ts[i + 1].expand(bsz)
                v1 = self.model(t1, x_euler, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
                if obs_consistency_mode in ("endpoint", "endpoint_smooth") and float(ts[i + 1].item()) < 1.0:
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
                # Default 1-RF benchmark solver.
                x = x + dt * v0

            # default_hard preserves the previous per-step pointwise sensor
            # replacement behavior for SenConsis.
            if obs_consistency_mode == "default_hard" and clamp_indices is not None:
                x = scatter_observed_values(
                    x=x,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_indices=clamp_indices,
                    obs_field_ids=obs_field_ids,
                    strength=1.0,
                )

        if obs_consistency_final_clamp and obs_consistency_mode != "none" and clamp_indices is not None:
            x = scatter_observed_values(
                x=x,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )

        return x

# Wrapper for FNO
class FNOFFM(PointCloudFFM):
    """
    This wrapper keeps the same outer FFM objective as PointCloudFFM but
    requires the full regular grid during both training and sampling, because
    the FNO backbone reshapes [B, N, C] into [B, C, Num_y, Num_x].

    The generalized sparse-conditioning API is preserved, but obs_indices are
    now mandatory so sparse measurements can be rasterized to grid channels.
    """

    def __init__(self, model: nn.Module, prior: nn.Module, sigma_min: float = 1e-4):
        super().__init__(model=model, prior=prior, sigma_min=sigma_min)
        self.requires_full_grid = True

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
        """
        RF training loss for the grid-based FNO backbone.
        obs_indices are required so sparse sensors can be rasterized onto the grid.
        """
        if obs_indices is None:
            raise ValueError("FNOFFM.training_loss requires obs_indices.")

        bsz = x1.shape[0]
        t = torch.rand(bsz, device=x1.device, dtype=x1.dtype)

        # RF source sample
        x0 = self.sample_source(coords)

        # Straight interpolation
        x_t = self.simulate(t, x0, x1)
        target = self.target_vector_field(x0, x1)

        pred = self.model(
            t=t,
            x_t=x_t,
            coords=coords,
            obs_coords=obs_coords,
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
        )

        loss = F.mse_loss(pred, target)
        return loss, {"loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def sample(
        self,
        coords: torch.Tensor,
        obs_coords: torch.Tensor,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        n_steps: int = 100,
        clamp_indices: Optional[torch.Tensor] = None,
        ode_solver: str = "euler",
        obs_consistency_mode: str = "default_hard",
        obs_consistency_strength: float = 1.0,
        obs_consistency_sigma: float = 0.05,
        obs_consistency_schedule_power: float = 2.0,
        obs_consistency_final_clamp: bool = True,
        obs_consistency_chunk_size: int = 8192,
    ) -> torch.Tensor:
        """
        Guided sampling with the FNO backbone.

        clamp_indices serves two roles here:
          1) it tells the backbone where to rasterize sparse observations;
          2) it is also used for hard clamping after each solver step.

        Euler uses one model evaluation per step. Heun uses two. Keeping this
        explicit is important for fair NFE comparisons against GL_rbf.
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be >= 1, got {n_steps}")
        if ode_solver not in ("euler", "heun"):
            raise ValueError(
                f"Unsupported ode_solver={ode_solver!r}; expected 'euler' or 'heun'."
            )
        if clamp_indices is None:
            raise ValueError(
                "FNOFFM.sample requires clamp_indices so sparse observations can be "
                "rasterized onto the grid and clamped during generation."
            )

        bsz = coords.shape[0]
        x = self.prior(coords, self.model.n_fields)
        obs_consistency_mode = normalize_obs_consistency_mode(obs_consistency_mode)

        value_map = None
        mask_map = None
        if obs_consistency_mode == "endpoint":
            value_map, mask_map = build_pointwise_observation_maps(
                coords=coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
            )
        elif obs_consistency_mode == "endpoint_smooth":
            value_map, mask_map = build_smooth_observation_maps(
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                n_fields=self.model.n_fields,
                sigma=obs_consistency_sigma,
                chunk_size=obs_consistency_chunk_size,
            )

        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype)

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]

            v0 = self.model(
                t=t0,
                x_t=x,
                coords=coords,
                obs_coords=obs_coords,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_field_ids=obs_field_ids,
                obs_indices=clamp_indices,
            )
            if obs_consistency_mode in ("endpoint", "endpoint_smooth"):
                # RF clean-endpoint observation masking for the FNO backbone.
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
                v1 = self.model(
                    t=t1,
                    x_t=x_euler,
                    coords=coords,
                    obs_coords=obs_coords,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_field_ids=obs_field_ids,
                    obs_indices=clamp_indices,
                )
                if obs_consistency_mode in ("endpoint", "endpoint_smooth") and float(ts[i + 1].item()) < 1.0:
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

            if obs_consistency_mode == "default_hard":
                x = scatter_observed_values(
                    x=x,
                    obs_values=obs_values,
                    obs_mask=obs_mask,
                    obs_indices=clamp_indices,
                    obs_field_ids=obs_field_ids,
                    strength=1.0,
                )

        if obs_consistency_final_clamp and obs_consistency_mode != "none":
            x = scatter_observed_values(
                x=x,
                obs_values=obs_values,
                obs_mask=obs_mask,
                obs_indices=clamp_indices,
                obs_field_ids=obs_field_ids,
                strength=1.0,
            )

        return x
