# ═════════ Imports ═════════
import abc
import math, torch
import numpy as np
from abc import abstractmethod
from typing import Dict, Optional, Tuple, Sequence

import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint  # noqa: F401  (used indirectly by S3GM `checkpoint(...)`)
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
try:
    from neuralop.models import FNO as NeuralOpFNO  # pip install neuraloperator
except ImportError:
    NeuralOpFNO = None  # resolved lazily; FNO/FNOFFM classes will raise at instantiation
try:
    from pykeops.torch import LazyTensor
except ImportError:
    LazyTensor = None  # keops backend falls back to torch when unavailable

FIELD_NAMES = ("CH4", "CO", "T", "U_1", "p")

__all__ = [
    "FIELD_NAMES",
    "make_mlp",
    "batched_gather_2d",
    "batched_gather_3d",
    "IIDGaussianPrior",
    "RFFGaussianPrior",
    "FeedForward",
    "CrossAttentionBlock",
    "SelfAttentionBlock",
    "ConditionalPointMLPRBF",
    "ConditionalPointPerceiver",
    "ConditionalPointHybridLocalGlobalRBF",
    "ConditionalPointFFM",
    "FNO",
    "PointCloudFFM",
    "FNOFFM",
    "ConvAE",
    "LatentFMUNet",
    "LatentFlowMatching",
    # ── SiT (Scalable Interpolant Transformers) model components ────────
    "SiTTimestepEmbedder",
    "SiTBlock",
    "SiTFinalLayer",
    "SiTPointTokenEmbedder",
    "SiTPointFinalLayer",
    "SiTPhysics",
    "SiTEMA",
    "SiTLearnedGridDeformer",
    # ── S3GM (Self-Supervised Sparse-Sensing Generative Model) components ─
    "SiLU",
    "GroupNorm32",
    "conv_nd",
    "linear",
    "avg_pool_nd",
    "update_ema",
    "zero_module",
    "scale_module",
    "mean_flat",
    "normalization",
    "timestep_embedding",
    "CheckpointFunction",
    "checkpoint",
    "convert_module_to_f16",
    "convert_module_to_f32",
    "make_master_params",
    "model_grads_to_master_grads",
    "master_params_to_model_params",
    "unflatten_master_params",
    "zero_grad",
    "RPENet",
    "RPE",
    "RPEAttention",
    "TimestepBlock",
    "TimestepEmbedAttnThingsSequential",
    "Upsample",
    "Downsample",
    "ResBlock",
    "FactorizedAttentionBlock",
    "UNetVideoModel",
    "ExponentialMovingAverage",
    "SDE",
    "VESDE",
    "VPSDE",
    "S3GMLearnedGridDeformer",
]

# ═════════ §1. Low-level MLP + gather utilities ═════════

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

# ═════════ §2. Prior distributions ═════════

class IIDGaussianPrior(nn.Module):
    """IID standard-normal prior on a point cloud."""

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        bsz, n_pts, _ = coords.shape
        return torch.randn(bsz, n_pts, n_channels,
                           device=coords.device, dtype=coords.dtype)

    def sample(self, shape, device=None, dtype=None):
        return torch.randn(*shape, device=device, dtype=dtype)

class RFFGaussianPrior(nn.Module):
    """Scalable smooth Gaussian-field approximation via random Fourier features."""

    def __init__(self, coord_dim: int = 3, n_features: int = 256,
                 lengthscale: float = 0.15):
        super().__init__()
        self.coord_dim = coord_dim
        self.n_features = n_features
        self.lengthscale = lengthscale
        self.register_buffer("omega",
            torch.randn(coord_dim, n_features) / max(lengthscale, 1e-6))
        self.register_buffer("phase", 2 * math.pi * torch.rand(n_features))

    def _features(self, coords: torch.Tensor) -> torch.Tensor:
        z = coords @ self.omega + self.phase
        return math.sqrt(2.0 / self.n_features) * torch.cos(z)

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        phi = self._features(coords)
        bsz, _, n_feat = phi.shape
        weights = torch.randn(bsz, n_channels, n_feat,
                              device=coords.device, dtype=coords.dtype)
        return torch.einsum("bnf,bcf->bnc", phi, weights)

    def sample(self, shape, device=None, dtype=None):
        # Fallback for dense-grid callers that don't carry coords.
        return torch.randn(*shape, device=device, dtype=dtype)

# ═════════ §3. Attention / Perceiver primitives ═════════

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

# ═════════ §4. Point-cloud backbones (for FFM) ═════════

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
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma

        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        self.point_encoder = make_mlp(coord_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3)
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

        point_feat = self.point_encoder(torch.cat([coords, x_t, t_feat], dim=-1))
        local_cond = self.aggregate_sparse_obs(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        global_feat = self.global_encoder(point_feat.mean(dim=1)).unsqueeze(1).expand(bsz, n_pts, -1)

        return self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))

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

        gather_mode: str = "rbf",    # ["rbf", "topk_rbf", "topk_rbf_gate"]
        gather_topk: int = 32,
        gather_query_chunk_size: Optional[int] = None,
        learnable_rbf_sigma: bool = False,
        neighbor_backend: str = "torch",      # ["auto", "torch", "keops"]
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

        if gather_mode not in ["rbf", "topk_rbf", "topk_rbf_gate"]:
            raise ValueError(
                f"gather_mode must be one of ['rbf', 'topk_rbf', 'topk_rbf_gate'], got {gather_mode}"
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
        else: 
            print(f"\nNOTICE: The gather mode is {gather_mode} with top-k {gather_topk} !!!\n")
            # Project query features to the same local-conditioning space so a
            # lightweight content gate can compare query state and refined sensors.
            self.query_to_cond = nn.Linear(hidden_dim, cond_dim, bias=False)
            # Small gate used only in the optional "topk_rbf_gate" mode.
            # Input = [projected query feat, refined sensor feat, relative coord, distance]
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

        # -------------------------
        # Point/query branch
        # -------------------------
        # Query point token from [coords, x_t, t]
        self.point_encoder = make_mlp(
            in_dim=coord_dim + n_fields + 1,
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

    # def _get_rbf_sigma(self) -> torch.Tensor:
    #     """
    #     Return a positive RBF sigma.
    #     """
    #     if self.learnable_rbf_sigma:
    #         return torch.exp(self.log_rbf_sigma).clamp_min(1e-6)
    #     return self._fixed_rbf_sigma.clamp_min(1e-6)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return topk_d2, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _knn_search_torch(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        return topk_d2, topk_sensor_feat, topk_sensor_coords, topk_valid

    def _get_topk_neighbors(
        self,
        query_coords: torch.Tensor,
        obs_coords: torch.Tensor,
        refined_sensor_feat: torch.Tensor,
        obs_mask: torch.Tensor,
        k: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    def _aggregate_chunk(
        self,
        query_coords: torch.Tensor,         # [B, Nc, D]
        query_feat: torch.Tensor,           # [B, Nc, H]
        obs_coords: torch.Tensor,           # [B, M, D]
        refined_sensor_feat: torch.Tensor,  # [B, M, Cc]
        obs_mask: torch.Tensor,             # [B, M]
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

        topk_d2, topk_sensor_feat, topk_sensor_coords, topk_valid = self._get_topk_neighbors(
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
            chunk_size = self.gather_query_chunk_size if self.gather_query_chunk_size is not None else 2048
        else:
            chunk_size = self.gather_query_chunk_size

        if chunk_size is None or n_query <= chunk_size:
            return self._aggregate_chunk(
                query_coords=query_coords,
                query_feat=query_feat,
                obs_coords=obs_coords,
                refined_sensor_feat=refined_sensor_feat,
                obs_mask=obs_mask,
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
        point_feat = self.point_encoder(torch.cat([coords, x_t, t_feat], dim=-1))  # [B, N, H]

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
    ) -> None:
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.rbf_sigma = rbf_sigma

        self.field_embed = nn.Embedding(n_fields, field_embed_dim)

        self.point_encoder = make_mlp(coord_dim + n_fields + 1, hidden_dim, hidden_dim, depth=3)
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

        point_feat = self.point_encoder(torch.cat([coords, x_t, t_feat], dim=-1))
        local_cond = self.aggregate_sparse_obs(coords, obs_coords, obs_values, obs_mask, obs_field_ids)
        global_feat = self.global_encoder(point_feat.mean(dim=1)).unsqueeze(1).expand(bsz, n_pts, -1)

        return self.head(torch.cat([point_feat, global_feat, local_cond], dim=-1))

# ═════════ §5. FNO backbone ═════════

# ------------------------------
# FNO backbone
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
    - The FNO operates on a regular mesh, so x_t is reshaped from point-cloud layout [B, N, C] to grid layout [B, C, Num_y, Num_x].
    - Sparse conditioning is rasterized into dense per-field observation maps and mask maps before being concatenated to the FNO input.
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
    ) -> None:
        super().__init__()

        self.n_fields = n_fields
        self.Num_x = int(Num_x)
        self.Num_y = int(Num_y)

        # FNO input channels:
        #   current state x_t         -> C
        #   scalar time channel       -> 1
        #   observed value maps       -> C
        #   observed mask maps        -> C
        # total = 3C + 1
        in_channels = 3 * n_fields + 1

        if NeuralOpFNO is None:
            raise ImportError("neuralop is required for FNO/FNOFFM. Install via `pip install neuraloperator`.")
        self.fno = NeuralOpFNO(
            n_modes=(n_modes_y, n_modes_x),   # tensor layout is [B, C, Num_y, Num_x]
            in_channels=in_channels,
            out_channels=n_fields,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding="grid" if use_grid_positional_embedding else None,
        )

    def _pointcloud_to_grid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, N, C] -> [B, C, Num_y, Num_x].
        """
        bsz, n_pts, n_fields = x.shape
        expected = self.Num_x * self.Num_y
        if n_pts != expected:
            raise ValueError(
                f"FNO backbone expected N = Num_x * Num_y = {expected}, got {n_pts}."
            )

        x_grid = x.reshape(bsz, self.Num_y, self.Num_x, n_fields)
        x_grid = x_grid.permute(0, 3, 1, 2).contiguous()
        return x_grid

    def _grid_to_pointcloud(self, x_grid: torch.Tensor) -> torch.Tensor:
        """
        Convert [B, C, Num_y, Num_x] -> [B, N, C].
        """
        bsz, n_fields, _, _ = x_grid.shape
        x = x_grid.permute(0, 2, 3, 1).contiguous()
        x = x.reshape(bsz, self.Num_x * self.Num_y, n_fields)
        return x

    def _build_condition_maps(
        self,
        obs_values: torch.Tensor,
        obs_mask: torch.Tensor,
        obs_field_ids: torch.Tensor,
        obs_indices: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Rasterize sparse observations into dense grid-aligned maps.

        Returns:
            obs_value_maps: [B, C, Num_y, Num_x]
            obs_mask_maps : [B, C, Num_y, Num_x]
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

        return obs_value_maps, obs_mask_maps

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

        # Reshape the current state to a grid.
        x_grid = self._pointcloud_to_grid(x_t)  # [B, C, Num_y, Num_x]
        # Broadcast time to a full grid channel.
        t_map = t.view(bsz, 1, 1, 1).expand(bsz, 1, self.Num_y, self.Num_x)

        # Convert sparse observations into dense field-aligned maps.
        obs_value_maps, obs_mask_maps = self._build_condition_maps(
            obs_values=obs_values,
            obs_mask=obs_mask,
            obs_field_ids=obs_field_ids,
            obs_indices=obs_indices,
            dtype=x_t.dtype,
            device=x_t.device,
        )

        # Concatenate:
        #   [current fields, time channel, observed values, observation masks]
        fno_in = torch.cat([x_grid, t_map, obs_value_maps, obs_mask_maps], dim=1)
        # FNO predicts the velocity field on the regular grid.
        vel_grid = self.fno(fno_in)
        # Convert back to the standard point-cloud layout expected by the wrapper.
        vel = self._grid_to_pointcloud(vel_grid)
        return vel

# ═════════ §6. Rectified-flow wrappers ═════════

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

        ts = torch.linspace(
            0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype
        )

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            dt = ts[i + 1] - ts[i]

            # Velocity at the current state.
            v0 = self.model(t0, x, coords, obs_coords, obs_values, obs_mask, obs_field_ids)

            if ode_solver == "heun":
                # Optional predictor-corrector step.
                x_euler = x + dt * v0
                t1 = ts[i + 1].expand(bsz)
                v1 = self.model(t1, x_euler, coords, obs_coords, obs_values, obs_mask, obs_field_ids)
                x = x + 0.5 * dt * (v0 + v1)
            else:
                # Default 1-RF benchmark solver.
                x = x + dt * v0

            # Keep known sensor values fixed during conditional generation.
            if clamp_indices is not None:
                for b in range(bsz):
                    valid = obs_mask[b].bool()
                    idx = clamp_indices[b, valid].long()
                    fld = obs_field_ids[b, valid].long()
                    val = obs_values[b, valid, 0]
                    x[b, idx, fld] = val

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
    ) -> torch.Tensor:
        """
        Guided sampling with the FNO backbone.

        clamp_indices serves two roles here:
          1) it tells the backbone where to rasterize sparse observations;
          2) it is also used for hard clamping after each Heun step.
        """
        if clamp_indices is None:
            raise ValueError(
                "FNOFFM.sample requires clamp_indices so sparse observations can be "
                "rasterized onto the grid and clamped during generation."
            )

        bsz = coords.shape[0]
        x = self.prior(coords, self.model.n_fields)

        dt = 1.0 / n_steps
        ts = torch.linspace(0.0, 1.0, n_steps + 1, device=coords.device, dtype=coords.dtype)

        for i in range(n_steps):
            t0 = ts[i].expand(bsz)
            t1 = ts[i + 1].expand(bsz)

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

            x_euler = x + dt * v0

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

            x = x + 0.5 * dt * (v0 + v1)

            # Hard-enforce observed values at the measured locations.
            for b in range(bsz):
                valid = obs_mask[b].bool()
                idx = clamp_indices[b, valid].long()
                fld = obs_field_ids[b, valid].long()
                val = obs_values[b, valid, 0]
                x[b, idx, fld] = val

        return x

# ═════════ §7. Latent FM (stage 1 AE + stage 2) ═════════

# -----------------------------------------------------------------------------
# Stage 1: Convolutional Autoencoder
# -----------------------------------------------------------------------------

def _gn_groups(channels: int, max_groups: int = 32) -> int:
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return g

class _ResBlock2d(nn.Module):
    """GN-SiLU-Conv3x3 -> GN-SiLU-Conv3x3 residual block at fixed channel count.

    state_dict layout (matches checkpoint):
        block.0 : GroupNorm(channels)
        block.1 : SiLU
        block.2 : Conv2d(channels, channels, 3, padding=1)
        block.3 : GroupNorm(channels)
        block.4 : SiLU
        block.5 : Conv2d(channels, channels, 3, padding=1)
    """

    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        g = _gn_groups(channels, groups)
        self.block = nn.Sequential(
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(g, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class ConvAE(nn.Module):
    """UNet-style convolutional autoencoder used as the Stage-1 latent space.

    state_dict layout (from best.pt):
        encoder.0                : Conv2d(n_fields, base_ch, 4, 2, 1)   # downsample #1
        encoder.{1..num_res}     : _ResBlock2d(base_ch)
        encoder.{k}              : Conv2d(ch_i, ch_{i+1}, 4, 2, 1)      # downsample
        ... repeated n_levels times ...
        decoder (mirror): ResBlocks, ConvTranspose2d, ..., final ConvT -> n_fields.
    Channel schedule: c_i = min(base_ch * 2**i, latent_ch) for i in 0..n_levels.
    """

    def __init__(self, n_fields: int, base_ch: int = 64, latent_ch: int = 128,
                 n_levels: int = 3, Num_y: int = 100, Num_x: int = 403,
                 deform_coord_dim: int = 0, deform_hidden: int = 128,
                 deform_depth: int = 3, num_res_blocks: int = 2):
        super().__init__()
        self.n_fields = n_fields
        self.base_ch = base_ch
        self.latent_ch = latent_ch
        self.n_levels = n_levels
        self.Num_y = Num_y
        self.Num_x = Num_x
        self.num_res_blocks = num_res_blocks

        # Padding so H, W are divisible by 2**n_levels.
        factor = 2 ** n_levels
        self.H_pad = int(math.ceil(Num_y / factor) * factor)
        self.W_pad = int(math.ceil(Num_x / factor) * factor)

        # Channel schedule.
        chs = [min(base_ch * (2 ** i), latent_ch) for i in range(n_levels + 1)]
        # chs[0] is input-side channel after first conv; chs[-1] == latent_ch.

        # --- Encoder ---
        enc = []
        # encoder.0 : first downsample conv from n_fields to chs[0]
        enc.append(nn.Conv2d(n_fields, chs[0], kernel_size=4, stride=2, padding=1))
        # After first downsample we have chs[0] channels; place num_res_blocks at chs[0].
        for _ in range(num_res_blocks):
            enc.append(_ResBlock2d(chs[0]))
        # Then for each subsequent level: downsample conv then num_res_blocks.
        for i in range(1, n_levels):
            enc.append(nn.Conv2d(chs[i - 1], chs[i], kernel_size=4, stride=2,
                                 padding=1))
            for _ in range(num_res_blocks):
                enc.append(_ResBlock2d(chs[i]))
        self.encoder = nn.Sequential(*enc)

        # --- Decoder (mirrors encoder) ---
        dec = []
        # Start at deepest level (chs[n_levels-1]) with num_res_blocks.
        for _ in range(num_res_blocks):
            dec.append(_ResBlock2d(chs[n_levels - 1]))
        # For each level going up: ConvTranspose then num_res_blocks at lower ch.
        for i in range(n_levels - 1, 0, -1):
            dec.append(nn.ConvTranspose2d(chs[i], chs[i - 1], kernel_size=4,
                                          stride=2, padding=1))
            for _ in range(num_res_blocks):
                dec.append(_ResBlock2d(chs[i - 1]))
        # Final ConvTranspose back to n_fields.
        dec.append(nn.ConvTranspose2d(chs[0], n_fields, kernel_size=4, stride=2,
                                      padding=1))
        self.decoder = nn.Sequential(*dec)

        # --- Optional coordinate deformer (for irregular meshes) ---
        if deform_coord_dim and deform_coord_dim > 0:
            layers = []
            d_in = deform_coord_dim
            for _ in range(deform_depth - 1):
                layers += [nn.Linear(d_in, deform_hidden), nn.SiLU()]
                d_in = deform_hidden
            final = nn.Linear(d_in, 2)
            # Zero-init final layer so `deform(coords) = sigmoid(coords[...,:2]
            # would collapse to ~0.5 everywhere at init. Instead the deformer
            # is used as a residual perturbation (see deform()), with final
            # layer zero-initialized so it starts as the identity map on the
            # first two coord components. This guarantees x_grid is non-trivial
            # from epoch 0, giving a real reconstruction signal to learn from.
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
            layers.append(final)
            self.deformer = nn.Sequential(*layers)
        else:
            self.deformer = None

    # --------- Padding helpers ---------
    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        pad_h = self.H_pad - H
        pad_w = self.W_pad - W
        if pad_h == 0 and pad_w == 0:
            return x
        return F.pad(x, (0, pad_w, 0, pad_h))

    def _crop(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :self.Num_y, :self.Num_x]

    # --------- Core ops ---------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self._pad(x)
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def deform(self, coords: torch.Tensor) -> torch.Tensor:
        assert self.deformer is not None, "ConvAE was built without a deformer"
        # Residual deformer with clamp to [0,1]. splat_to_grid and
        # gather_from_grid both expect normalized [0,1]^2 coords per their
        # docstrings; without bounding, an unconstrained nn.Linear output
        # puts bilinear weights outside [0,1], making scatter-add accumulate
        # wildly large values (loss ~1e11 at epoch 0). Dataset coords are
        # already in [0,1]^3 (z=0 padding), so using the first two channels
        # as the base map and letting the MLP learn a small perturbation is
        # both well-conditioned and matches the reference LatentFM design
        # (which had no deformer at all on regular grids).
        delta = self.deformer(coords)
        base = coords[..., :2]
        return (base + delta).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

# -----------------------------------------------------------------------------
# Stage 2: Latent Flow Matching UNet
# -----------------------------------------------------------------------------

class _TimestepEmbedding(nn.Module):
    """Sinusoidal(t) -> Linear -> SiLU -> Linear  (emb_dim -> emb_dim)."""

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t.float().reshape(-1, 1) * freqs.reshape(1, -1)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb.to(t.dtype if t.is_floating_point() else torch.float32)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(self._sinusoidal(t))

class _AdaGNResBlock(nn.Module):
    """Residual block with AdaGN-style FiLM conditioning on the time embedding.

    state_dict layout (matches checkpoint):
        norm1     : GroupNorm(channels)
        conv1     : Conv2d(channels, channels, 3, padding=1)
        norm2     : GroupNorm(channels)
        conv2     : Conv2d(channels, channels, 3, padding=1)
        emb_proj  : Linear(emb_dim, 2*channels)  # scale, shift
    """

    def __init__(self, channels: int, emb_dim: int, groups: int = 8):
        super().__init__()
        g = _gn_groups(channels, groups)
        self.channels = channels
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.emb_proj = nn.Linear(emb_dim, 2 * channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        scale, shift = self.emb_proj(F.silu(emb)).chunk(2, dim=-1)
        h = self.norm2(h) * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv2(F.silu(h))
        return x + h

class PointNetSensorEncoder(nn.Module):
    """PointNet-style encoder for sparse sensor tokens.

    Adapted from Kashefi et al. (arXiv:2601.03030) "Flow Matching and Diffusion
    Models via PointNet for Generating Fluid Fields on Irregular Geometries":
    shared Conv1d MLPs on per-point tokens followed by a masked global
    max-pool, producing a permutation-invariant sample-level feature.

    To keep the velocity UNet's channel layout identical to the masked-image
    conditioner, the global feature is projected through two heads and tiled
    across the latent (h, w) grid to fill the same (cond_feat, cond_mask)
    slots that LatentFMUNet already consumes.
    """

    def __init__(self, n_fields: int, coord_dim: int = 2,
                 hidden_mult: float = 1.0):
        super().__init__()
        self.n_fields = n_fields
        self.coord_dim = coord_dim
        self.out_feat_ch = n_fields
        self.out_mask_ch = n_fields + 1

        # Per-token input: [coords (D), value (1), field one-hot (F), mask (1)]
        in_ch = coord_dim + 1 + n_fields + 1
        m = hidden_mult
        c1, c2, c3 = int(64 * m), int(128 * m), int(1024 * m)

        def _gn(ch):
            return nn.GroupNorm(_gn_groups(ch), ch)

        self.conv1 = nn.Conv1d(in_ch, c1, 1)
        self.n1 = _gn(c1)
        self.conv2 = nn.Conv1d(c1, c1, 1)
        self.n2 = _gn(c1)
        self.conv3 = nn.Conv1d(c1, c2, 1)
        self.n3 = _gn(c2)
        self.conv4 = nn.Conv1d(c2, c3, 1)
        self.n4 = _gn(c3)

        self.head_feat = nn.Sequential(
            nn.Linear(c3, c2), nn.SiLU(), nn.Linear(c2, self.out_feat_ch),
        )
        self.head_mask = nn.Sequential(
            nn.Linear(c3, c2), nn.SiLU(), nn.Linear(c2, self.out_mask_ch),
        )

    def forward(self, obs_coords_2d: torch.Tensor,
                obs_values: torch.Tensor,
                obs_mask: torch.Tensor,
                obs_field_ids: torch.Tensor,
                h: int, w: int):
        B, M = obs_mask.shape
        dtype = obs_values.dtype
        fld = obs_field_ids.clamp(min=0).long()
        fld_oh = F.one_hot(fld, num_classes=self.n_fields).to(dtype)
        fld_oh = fld_oh * obs_mask.unsqueeze(-1)

        tok = torch.cat(
            [obs_coords_2d.to(dtype), obs_values, fld_oh,
             obs_mask.unsqueeze(-1)],
            dim=-1,
        )  # [B, M, in_ch]
        x = tok.transpose(1, 2)  # [B, in_ch, M]

        x = F.relu(self.n1(self.conv1(x)))
        x = F.relu(self.n2(self.conv2(x)))
        x = F.relu(self.n3(self.conv3(x)))
        x = F.relu(self.n4(self.conv4(x)))  # [B, c3, M]

        # Masked global max-pool (ignore padded sensor slots).
        mask_b1m = obs_mask.unsqueeze(1) > 0.5
        neg_inf = torch.finfo(x.dtype).min
        x = x.masked_fill(~mask_b1m, neg_inf)
        g = x.amax(dim=-1)  # [B, c3]
        # Safety: if a sample has zero valid sensors, max returns -inf.
        g = torch.where(torch.isneginf(g), torch.zeros_like(g), g)

        cond_feat = self.head_feat(g).view(B, -1, 1, 1).expand(-1, -1, h, w).contiguous()
        cond_mask = self.head_mask(g).view(B, -1, 1, 1).expand(-1, -1, h, w).contiguous()
        return cond_feat, cond_mask


class LatentFMUNet(nn.Module):
    """Velocity network operating in the AE's latent grid.

    Sparse-field-as-image conditioning: the masked sparse field and per-field
    mask are stacked as extra input channels at the denoiser's resolution
    (rather than encoded through the frozen AE). Input channels of in_conv:

        latent_ch (z_t) + n_fields (downsampled masked sparse field)
                        + n_fields + 1 (per-field mask + any-sensor aggregate)
    """

    def __init__(self, latent_ch: int, n_fields: int, base_ch: int = 256,
                 ch_mult: Sequence[int] = (1, 2), num_res_blocks: int = 2,
                 num_heads: int = 8, attn_at_level: Optional[int] = None):
        super().__init__()
        self.latent_ch = latent_ch
        self.n_fields = n_fields
        self.base_ch = base_ch
        self.ch_mult = tuple(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.num_heads = num_heads
        # Attention is placed at the bottleneck by default.
        self.attn_at_level = attn_at_level

        emb_dim = base_ch
        self.time_embed = _TimestepEmbedding(emb_dim)

        in_ch = latent_ch + n_fields + (n_fields + 1)
        self.in_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        channels = [base_ch * m for m in self.ch_mult]
        n_lv = len(channels)

        # --- Down path ---
        self.down_blocks = nn.ModuleList()
        self.down_convs = nn.ModuleList()
        prev_ch = base_ch
        for i, ch in enumerate(channels):
            # If ch differs from prev_ch, the first resblock would need a
            # channel change; for the shipped configs channels[0]==base_ch so
            # the first level is identity-in-channel.
            assert ch == prev_ch or i > 0
            blocks = nn.ModuleList(
                [_AdaGNResBlock(ch, emb_dim) for _ in range(num_res_blocks)])
            self.down_blocks.append(blocks)
            if i < n_lv - 1:
                # Downsample conv lifts to next level's channel count.
                self.down_convs.append(
                    nn.Conv2d(ch, channels[i + 1], kernel_size=4,
                              stride=2, padding=1))
                prev_ch = channels[i + 1]
            else:
                prev_ch = ch

        # --- Middle ---
        mid_ch = channels[-1]
        self.mid_res1 = _AdaGNResBlock(mid_ch, emb_dim)
        self.mid_attn = nn.MultiheadAttention(mid_ch, num_heads,
                                              batch_first=True)
        self.mid_attn_norm = nn.GroupNorm(_gn_groups(mid_ch), mid_ch)
        self.mid_res2 = _AdaGNResBlock(mid_ch, emb_dim)

        # --- Up path (mirror) ---
        self.up_blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for i in range(n_lv):
            # Level index from deepest (0) to shallowest (n_lv-1).
            lv = n_lv - 1 - i  # actual resolution level this up block finishes at
            skip_ch = channels[lv]
            if i == 0:
                # First up block: mid_ch + skip_ch (both == channels[-1])
                in_blk = mid_ch + skip_ch
            else:
                # After previous up_conv we are at channels[lv]; concat skip.
                in_blk = channels[lv] + skip_ch
            blocks = nn.ModuleList(
                [_AdaGNResBlock(in_blk, emb_dim) for _ in range(num_res_blocks)])
            self.up_blocks.append(blocks)
            if i < n_lv - 1:
                # Up-sample back to next shallower channel count.
                self.up_convs.append(
                    nn.ConvTranspose2d(in_blk, channels[lv - 1],
                                       kernel_size=4, stride=2, padding=1))

        # --- Output head ---
        final_ch = None
        # After last up_blocks iteration we are at in_blk of that iteration.
        if n_lv == 1:
            final_ch = mid_ch + channels[0]
        else:
            final_ch = channels[0] + channels[0]  # last level: up_conv_out + skip
        self.out_conv = nn.Sequential(
            nn.GroupNorm(_gn_groups(final_ch), final_ch),
            nn.SiLU(),
            nn.Conv2d(final_ch, latent_ch, kernel_size=3, padding=1),
        )

    def forward(self, t: torch.Tensor, z_t: torch.Tensor,
                z_obs: torch.Tensor, z_mask: torch.Tensor) -> torch.Tensor:
        """
        t      : [B] in [0, 1]
        z_t    : [B, latent_ch, h, w]
        z_obs  : [B, latent_ch, h, w]
        z_mask : [B, n_fields + 1, h, w]  (per-field + any-sensor)
        """
        emb = self.time_embed(t)
        h = self.in_conv(torch.cat([z_t, z_obs, z_mask], dim=1))

        skips = []
        n_lv = len(self.down_blocks)
        for i, blocks in enumerate(self.down_blocks):
            for blk in blocks:
                h = blk(h, emb)
            skips.append(h)
            if i < n_lv - 1:
                h = self.down_convs[i](h)

        # --- Middle with self-attention ---
        h = self.mid_res1(h, emb)
        B, C, H, W = h.shape
        h_norm = self.mid_attn_norm(h)
        tokens = h_norm.flatten(2).transpose(1, 2)  # [B, H*W, C]
        attn_out, _ = self.mid_attn(tokens, tokens, tokens, need_weights=False)
        h = h + attn_out.transpose(1, 2).reshape(B, C, H, W)
        h = self.mid_res2(h, emb)

        # --- Up path ---
        # Resize `h` to match each skip's spatial extent before concat. With
        # Conv2d(k=4, s=2, p=1) down and ConvTranspose2d(k=4, s=2, p=1) up,
        # odd input dims floor on the way down (e.g., 13 -> 6) and double
        # back (6 -> 12), so we need a 1-cell correction when the original
        # latent dim was odd. Nearest-neighbor interpolate is weight-free and
        # keeps checkpoint compatibility.
        for i, blocks in enumerate(self.up_blocks):
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for blk in blocks:
                h = blk(h, emb)
            if i < n_lv - 1:
                h = self.up_convs[i](h)

        return self.out_conv(h)

class LatentFlowMatching(nn.Module):
    """Rectified-Flow training & Euler-ODE sampling in the AE latent space."""

    def __init__(self, ae: ConvAE, velocity_net: LatentFMUNet,
                 Num_x: int, Num_y: int,
                 cond_mode: str = "image",
                 pointnet_encoder: Optional["PointNetSensorEncoder"] = None):
        super().__init__()
        self.ae = ae
        self.velocity_net = velocity_net
        self.Num_x = Num_x
        self.Num_y = Num_y
        self.n_fields = ae.n_fields
        self.latent_ch = ae.latent_ch
        self.n_levels = ae.n_levels

        if cond_mode not in ("image", "interp", "pointnet"):
            raise ValueError(
                f"cond_mode must be 'image', 'interp', or 'pointnet', got {cond_mode!r}")
        self.cond_mode = cond_mode
        if cond_mode == "pointnet":
            if pointnet_encoder is None:
                raise ValueError("cond_mode='pointnet' requires pointnet_encoder.")
            self.pointnet = pointnet_encoder
        else:
            self.pointnet = None

    # --- Condition downsampling: sparse-field-as-image conditioning ---
    def _pad_to_ae(self, x: torch.Tensor) -> torch.Tensor:
        """Right/bottom zero-pad to (ae.H_pad, ae.W_pad)."""
        return F.pad(x, (0, self.ae.W_pad - x.shape[-1],
                         0, self.ae.H_pad - x.shape[-2]))

    def _downsample_mask(self, mask_grid: torch.Tensor) -> torch.Tensor:
        """mask_grid: [B, n_fields, H, W]  (any H, W <= AE's H_pad/W_pad).
        Returns [B, n_fields + 1, h, w] at latent resolution: per-field
        max-pool (preserves "any sensor in this cell") + any-sensor aggregate.
        """
        factor = 2 ** self.n_levels
        m = self._pad_to_ae(mask_grid)
        m_ds = F.max_pool2d(m, kernel_size=factor, stride=factor)
        any_m = m_ds.amax(dim=1, keepdim=True)
        return torch.cat([m_ds, any_m], dim=1)

    def _latent_hw(self) -> Tuple[int, int]:
        factor = 2 ** self.n_levels
        return self.ae.H_pad // factor, self.ae.W_pad // factor

    def _encode_condition(self, cond_inputs: dict):
        """Dispatch conditioning on self.cond_mode.

        cond_inputs must contain:
          - 'image' / 'interp' mode: obs_value_grid [B, F, Ny, Nx],
                                     obs_mask_grid  [B, F, Ny, Nx]
          - 'pointnet' mode:         obs_coords_2d [B, M, 2], obs_values [B, M, 1],
                                     obs_mask [B, M], obs_field_ids [B, M]

        Returns (cond_feat, cond_mask) matching LatentFMUNet's channel slots.
        """
        if self.cond_mode in ("image", "interp"):
            obs_value_grid = cond_inputs["obs_value_grid"]
            obs_mask_grid = cond_inputs["obs_mask_grid"]
            factor = 2 ** self.n_levels
            if self.cond_mode == "image":
                # masked sparse field (zero at unobserved cells)
                cond_grid = obs_value_grid * obs_mask_grid
            else:
                # nearest-neighbor Voronoi fill: dense interpolated field
                cond_grid = BASELINE_HELPERS.nearest_fill_grid(obs_value_grid, obs_mask_grid)
            cond_grid = self._pad_to_ae(cond_grid)
            cond_feat = F.avg_pool2d(cond_grid, kernel_size=factor, stride=factor)
            # Mask channels always carry the *actual* sensor positions so the
            # network can distinguish real measurements from interp guesses.
            cond_mask = self._downsample_mask(obs_mask_grid)
            return cond_feat, cond_mask

        # pointnet
        h, w = self._latent_hw()
        return self.pointnet(
            cond_inputs["obs_coords_2d"],
            cond_inputs["obs_values"],
            cond_inputs["obs_mask"],
            cond_inputs["obs_field_ids"],
            h, w,
        )

    def training_loss(self, fields_grid: torch.Tensor, cond_inputs: dict):
        """Rectified-flow MSE loss with configurable sparse conditioning."""
        with torch.no_grad():
            x1 = self.ae.encode(fields_grid)
        cond_feat, cond_mask = self._encode_condition(cond_inputs)

        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=x1.device, dtype=x1.dtype)
        t_ = t.view(B, 1, 1, 1)
        x_t = (1 - t_) * x0 + t_ * x1
        target = x1 - x0

        pred = self.velocity_net(t, x_t, cond_feat, cond_mask)
        loss = F.mse_loss(pred, target)
        info = {"loss": float(loss.detach().cpu())}
        return loss, info

    @torch.no_grad()
    def sample(self, cond_inputs: dict,
               n_steps: int = 8, ode_solver: str = "euler") -> torch.Tensor:
        """ODE integration from t=0 (noise) to t=1 (data) in latent space,
        then AE-decode back to the field grid. Supports 'euler' and 'heun'."""
        if ode_solver not in ("euler", "heun"):
            raise ValueError(
                f"Unsupported ode_solver={ode_solver!r}. Use 'euler' or 'heun'."
            )
        cond_feat, cond_mask = self._encode_condition(cond_inputs)
        B = cond_feat.shape[0]
        shape = (B, self.latent_ch, cond_feat.shape[-2], cond_feat.shape[-1])
        x = torch.randn(shape, device=cond_feat.device, dtype=cond_feat.dtype)

        dt = 1.0 / n_steps
        for k in range(n_steps):
            t = torch.full((B,), k * dt, device=x.device, dtype=x.dtype)
            v = self.velocity_net(t, x, cond_feat, cond_mask)
            if ode_solver == "euler":
                x = x + dt * v
            else:  # heun: 2nd-order predictor-corrector
                t_next = torch.full((B,), (k + 1) * dt,
                                    device=x.device, dtype=x.dtype)
                x_pred = x + dt * v
                v_next = self.velocity_net(t_next, x_pred, cond_feat, cond_mask)
                x = x + 0.5 * dt * (v + v_next)

        out = self.ae.decode(x)
        return out


# ═════════════════════════════════════════════════════════════════════════════
# SiT (Scalable Interpolant Transformers) — Model Components          

# ─────────────────────────────────────────────────────────────────────────────
# [SiT] adaLN-Zero modulation helper
# ─────────────────────────────────────────────────────────────────────────────
def _sit_modulate(x, shift, scale):
    """[SiT] adaLN-Zero: y = x * (1 + scale) + shift (scalar broadcast)."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Sinusoidal timestep embedding
# ─────────────────────────────────────────────────────────────────────────────
class SiTTimestepEmbedder(nn.Module):
    """[SiT] Embeds scalar diffusion/flow timesteps into vector representations."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Core transformer block (adaLN-Zero conditioned)
# ─────────────────────────────────────────────────────────────────────────────
class SiTBlock(nn.Module):
    """[SiT] A SiT block with adaptive layer norm zero (adaLN-Zero) conditioning."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        from timm.models.vision_transformer import Attention, Mlp
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        # qk_norm=True adds per-head LayerNorm on Q and K to bound attention
        # logit magnitude. Without it, Q·Kᵀ norms drift upward over tens of
        # epochs until softmax saturates to one-hot and gradients explode —
        # the signature failure mode of plain ViT/DiT attention on long runs.
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True,
                              qk_norm=True, norm_layer=nn.LayerNorm,
                              **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(_sit_modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(_sit_modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Final output head for patch tokenizer (unpatchify-ready)
# ─────────────────────────────────────────────────────────────────────────────
class SiTFinalLayer(nn.Module):
    """[SiT] The final layer of SiT (patch tokenizer path)."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _sit_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Point-token tokenizer (native irregular-mesh path)
# ─────────────────────────────────────────────────────────────────────────────
class SiTPointTokenEmbedder(nn.Module):
    """[SiT] Per-node input embedder for irregular meshes.

    Replaces PatchEmbed when `tokenizer='pointnet'`. Each mesh node becomes
    one transformer token: its features are the concatenation of
      - the noisy field values at that node        (in_channels)
      - optional sparse-sensor conditioning        (cond_channels)
      - Fourier-feature positional encoding of (x,y) coords
    followed by a small MLP to hidden_size. The DiT blocks never see raw
    coords or a grid — only hidden-size tokens.
    """

    def __init__(self, in_channels: int, cond_channels: int, coord_dim: int,
                 hidden_size: int, fourier_num_freqs: int = 64,
                 fourier_scale: float = 10.0):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        self.coord_dim = coord_dim
        self.fourier_num_freqs = fourier_num_freqs
        # Fixed random Fourier basis (non-trainable) — standard NeRF-style
        # positional encoding. Gaussian(0, scale²) frequencies.
        B = torch.randn(coord_dim, fourier_num_freqs) * fourier_scale
        self.register_buffer("fourier_B", B, persistent=True)

        token_in = in_channels + cond_channels + 2 * fourier_num_freqs
        self.mlp = nn.Sequential(
            nn.Linear(token_in, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, x_nodes, coords, cond_nodes=None):
        # x_nodes: [B, N, in_channels]
        # coords:  [B, N, coord_dim]
        # cond_nodes: [B, N, cond_channels] or None
        proj = coords @ self.fourier_B                   # [B, N, F]
        pos = torch.cat([proj.sin(), proj.cos()], dim=-1)
        parts = [x_nodes]
        if self.cond_channels > 0:
            assert cond_nodes is not None, "cond_channels > 0 requires cond_nodes"
            parts.append(cond_nodes)
        parts.append(pos)
        return self.mlp(torch.cat(parts, dim=-1))


class SiTPointFinalLayer(nn.Module):
    """[SiT] Per-token output head (analog of SiTFinalLayer for point tokens)."""

    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = _sit_modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] 2D / 1D sin-cos positional embedding helpers (non-square grids)
# ─────────────────────────────────────────────────────────────────────────────
def _sit_get_2d_sincos_pos_embed_nonsquare(embed_dim, gh, gw):
    """[SiT] 2D sin-cos positional embedding for non-square grids (gh x gw patches)."""
    assert embed_dim % 2 == 0
    grid_h = np.arange(gh, dtype=np.float32)
    grid_w = np.arange(gw, dtype=np.float32)
    grid_w, grid_h = np.meshgrid(grid_w, grid_h)  # w goes first in meshgrid
    emb_h = _sit_get_1d_sincos_pos_embed(embed_dim // 2, grid_h.flatten())
    emb_w = _sit_get_1d_sincos_pos_embed(embed_dim // 2, grid_w.flatten())
    return np.concatenate([emb_h, emb_w], axis=1)  # (gh*gw, embed_dim)


def _sit_get_1d_sincos_pos_embed(embed_dim, pos):
    """[SiT] 1D sin-cos positional embedding."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Main model: SiT adapted for physical fields (no class labels, no VAE)
# ─────────────────────────────────────────────────────────────────────────────
class SiTPhysics(nn.Module):
    """
    [SiT] SiT adapted for physical field reconstruction from sparse observations.

    Core architecture (patch tokenization, adaLN-Zero DiT blocks, sinusoidal
    timestep embedding, flow-matching transport framework) is unchanged from
    the original SiT. The only difference is a widened patch-embed input:
    the noisy field tensor is channel-concatenated with the sparse-field-
    as-image conditioning (masked sparse field + per-field/any-sensor mask)
    before patchification, mirroring the LatentFM 'image' conditioner. The
    output head still produces only the `in_channels` field channels.

    Changes from the original SiT:
      - No class label embedding (y_embedder removed)
      - in_channels / out_channels set to n_fields (not latent VAE channels)
      - learn_sigma disabled
      - Supports non-square grids via separate H/W patch counts
      - Optional `cond_channels` extra channels concatenated at the patch
        embed for sparse-sensor conditioning (default 0 = unconditional).
    """

    def __init__(
        self,
        # Grid-mode params (used only when tokenizer='patch')
        input_size_h: int = 0,
        input_size_w: int = 0,
        patch_size: int = 2,
        # Common
        in_channels: int = 5,
        cond_channels: int = 0,
        hidden_size: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        # Tokenizer
        tokenizer: str = "patch",
        coord_dim: int = 2,
        fourier_num_freqs: int = 64,
        fourier_scale: float = 10.0,
    ):
        super().__init__()

        if tokenizer not in ("patch", "pointnet"):
            raise ValueError(f"tokenizer must be 'patch' or 'pointnet', got {tokenizer!r}")

        self.tokenizer = tokenizer
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.cond_channels = cond_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.input_size_h = input_size_h
        self.input_size_w = input_size_w

        if tokenizer == "patch":
            # Patch embedding with room for optional sparse conditioning channels.
            # The output head still produces only `in_channels` channels; only the
            # patch embedding sees the widened input.
            from timm.models.vision_transformer import PatchEmbed
            self.x_embedder = PatchEmbed(
                img_size=(input_size_h, input_size_w),
                patch_size=patch_size,
                in_chans=in_channels + cond_channels,
                embed_dim=hidden_size,
                bias=True,
            )
            num_patches = (input_size_h // patch_size) * (input_size_w // patch_size)
            # Fixed sin-cos positional embedding for non-square grids.
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        else:  # pointnet
            self.x_embedder = SiTPointTokenEmbedder(
                in_channels=in_channels,
                cond_channels=cond_channels,
                coord_dim=coord_dim,
                hidden_size=hidden_size,
                fourier_num_freqs=fourier_num_freqs,
                fourier_scale=fourier_scale,
            )
            self.pos_embed = None  # Fourier features are inside the embedder.

        self.t_embedder = SiTTimestepEmbedder(hidden_size)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio)
            for _ in range(depth)
        ])

        if tokenizer == "patch":
            self.final_layer = SiTFinalLayer(hidden_size, patch_size, self.out_channels)
        else:
            self.final_layer = SiTPointFinalLayer(hidden_size, self.out_channels)

        self._initialize_weights(input_size_h, input_size_w, hidden_size, patch_size)

    def _initialize_weights(self, H, W, hidden_size, patch_size):
        """Initialize weights; uses 2D sin-cos pos embed for non-square grids."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        if self.tokenizer == "patch":
            # 2D sin-cos positional embedding for non-square patch grids.
            gh = H // patch_size
            gw = W // patch_size
            pos_embed = _sit_get_2d_sincos_pos_embed_nonsquare(hidden_size, gh, gw)
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

            # Patch embed like nn.Linear.
            w = self.x_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Timestep embedding MLP.
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers.
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, C, H, W)  — supports non-square grids.
        """
        c = self.out_channels
        p = self.patch_size
        gh = self.input_size_h // p
        gw = self.input_size_w // p

        x = x.reshape(x.shape[0], gh, gw, p, p, c)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], c, gh * p, gw * p)
        return imgs

    def forward(self, x, t,
                obs_value_grid=None, obs_mask_grid=None,
                coords=None, obs_value_nodes=None, obs_mask_nodes=None,
                **kwargs):
        """
        Forward pass of SiT for physical fields. Dispatches on self.tokenizer.

        tokenizer='patch' (grid DiT):
          x:              [B, C, H, W] noisy spatial fields
          obs_value_grid: [B, n_fields, H, W] sparse observed values
          obs_mask_grid:  [B, n_fields, H, W] per-field observation mask

        tokenizer='pointnet' (irregular-mesh DiT):
          x:               [B, N, C] noisy fields at mesh nodes
          coords:          [B, N, coord_dim] normalized mesh-node coordinates
          obs_value_nodes: [B, N, n_fields] sparse observed values scattered back
                           to mesh nodes (zero where no sensor)
          obs_mask_nodes:  [B, N, n_fields] per-field observation mask at nodes
        """
        c = self.t_embedder(t)

        if self.tokenizer == "patch":
            if self.cond_channels > 0:
                if obs_value_grid is None or obs_mask_grid is None:
                    cond_feat = torch.zeros(
                        x.shape[0], self.in_channels, x.shape[2], x.shape[3],
                        device=x.device, dtype=x.dtype,
                    )
                    cond_mask = torch.zeros(
                        x.shape[0], self.in_channels + 1, x.shape[2], x.shape[3],
                        device=x.device, dtype=x.dtype,
                    )
                else:
                    # Caller supplies obs_value_grid already in the form they
                    # want the network to see: masked sparse field (zeros at
                    # unobserved) for cond_mode='image', or Voronoi-filled
                    # dense field for cond_mode='interp'. The mask channels
                    # always carry actual sensor locations.
                    cond_feat = obs_value_grid
                    any_mask = obs_mask_grid.amax(dim=1, keepdim=True)
                    cond_mask = torch.cat([obs_mask_grid, any_mask], dim=1)
                x = torch.cat([x, cond_feat, cond_mask], dim=1)

            x = self.x_embedder(x) + self.pos_embed   # (N, T, D)
            for block in self.blocks:
                x = block(x, c)
            x = self.final_layer(x, c)                 # (N, T, p²·C)
            x = self.unpatchify(x)                     # (N, C, H, W)
            return x

        # pointnet tokenizer
        assert coords is not None, "pointnet tokenizer requires `coords` kwarg."
        cond_nodes = None
        if self.cond_channels > 0:
            if obs_value_nodes is None or obs_mask_nodes is None:
                B, N, _ = x.shape
                cond_feat = torch.zeros(B, N, self.in_channels,
                                        device=x.device, dtype=x.dtype)
                cond_mask_full = torch.zeros(B, N, self.in_channels + 1,
                                             device=x.device, dtype=x.dtype)
            else:
                # Caller supplies obs_value_nodes already shaped for the chosen
                # cond_mode (masked vs. Voronoi-filled).
                cond_feat = obs_value_nodes
                any_mask = obs_mask_nodes.amax(dim=-1, keepdim=True)
                cond_mask_full = torch.cat([obs_mask_nodes, any_mask], dim=-1)
            cond_nodes = torch.cat([cond_feat, cond_mask_full], dim=-1)

        tokens = self.x_embedder(x, coords[..., :self.x_embedder.coord_dim], cond_nodes)
        for block in self.blocks:
            tokens = block(tokens, c)
        out = self.final_layer(tokens, c)              # [B, N, C]
        return out


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Learned coordinate deformation for irregular grids (Geo-FNO trick)
# ─────────────────────────────────────────────────────────────────────────────
class SiTLearnedGridDeformer(nn.Module):
    """[SiT] Learned MLP mapping physical 2-D coordinates to latent 2-D coordinates
    in [0, 1], following the Geo-FNO deformable-mapping idea (Li et al. 2023).
    Used only by the SiT baseline when tokenizer='patch' on irregular meshes.

    Note: S3GM has its own variant (`S3GMLearnedGridDeformer`) with a
    different init scheme. Kept as separate classes so existing SiT
    checkpoints remain load-compatible.
    """

    def __init__(self, coord_dim: int = 2, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        dim = coord_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
            dim = hidden_dim
        layers += [nn.Linear(dim, 2), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, coords_2d: torch.Tensor) -> torch.Tensor:
        """[B, N, 2] -> [B, N, 2] in [0, 1]."""
        return self.net(coords_2d)


# ─────────────────────────────────────────────────────────────────────────────
# [SiT] Lightweight EMA used by the SiT baseline
# ─────────────────────────────────────────────────────────────────────────────
class SiTEMA:
    """[SiT] Exponential Moving Average of model parameters (lightweight version)."""

    def __init__(self, parameters, decay=0.9999):
        self.decay = decay
        self.shadow = [p.clone().detach() for p in parameters]
        self.backup = []

    def update(self, parameters):
        for s, p in zip(self.shadow, parameters):
            s.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def store(self, parameters):
        self.backup = [p.clone().detach() for p in parameters]

    def restore(self, parameters):
        for p, b in zip(parameters, self.backup):
            p.data.copy_(b)
        self.backup = []

    def copy_to(self, parameters):
        for p, s in zip(parameters, self.shadow):
            p.data.copy_(s)

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        self.decay = state_dict["decay"]
        # torch.load(..., map_location="cpu") brings shadow back as CPU
        # tensors; move them to the device of the shadow set at __init__
        # so EMA.update matches the live (GPU) parameters.
        if self.shadow:
            device = self.shadow[0].device
            self.shadow = [s.to(device) for s in state_dict["shadow"]]
        else:
            self.shadow = list(state_dict["shadow"])




# ═════════════════════════════════════════════════════════════════════════════
# S3GM (Self-Supervised Sparse-sensing Generative Model) — Model Components
#
# Consolidated from s3gm_core.py. The original s3gm_core.py is left intact;
# training/evaluation scripts import the canonical components from here.
# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# [S3GM] Neural-network utilities (from models/nn.py)
# ─────────────────────────────────────────────────────────────────────────────
class SiLU(nn.Module):
    """[S3GM] Swish/SiLU activation."""
    def forward(self, x):
        return x * torch.sigmoid(x)


class GroupNorm32(nn.GroupNorm):
    """[S3GM] GroupNorm that always runs in fp32 then casts back."""
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def conv_nd(dims, *args, **kwargs):
    """[S3GM] Factory for 1D/2D/3D convolutions."""
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def linear(*args, **kwargs):
    """[S3GM] Thin wrapper over nn.Linear (kept for symmetry with conv_nd)."""
    return nn.Linear(*args, **kwargs)


def avg_pool_nd(dims, *args, **kwargs):
    """[S3GM] Factory for 1D/2D/3D average pooling."""
    if dims == 1:
        return nn.AvgPool1d(*args, **kwargs)
    elif dims == 2:
        return nn.AvgPool2d(*args, **kwargs)
    elif dims == 3:
        return nn.AvgPool3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


def update_ema(target_params, source_params, rate=0.99):
    """[S3GM] In-place EMA update helper (distinct from SiTEMA / ExponentialMovingAverage)."""
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)


def zero_module(module):
    """[S3GM] Zero out a module's parameters in-place and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def scale_module(module, scale):
    """[S3GM] Scale a module's parameters in-place by `scale`."""
    for p in module.parameters():
        p.detach().mul_(scale)
    return module


def mean_flat(tensor, mask=None):
    """[S3GM] Take the mean over every dim except the batch dim."""
    if mask is not None:
        tensor = tensor * mask
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


def normalization(channels):
    """[S3GM] Standard S3GM normalization: GroupNorm with 32 groups."""
    return GroupNorm32(32, channels)


def timestep_embedding(timesteps, dim, max_period=10000):
    """[S3GM] Sinusoidal timestep embedding (function form, for UNet)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CheckpointFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, run_function, length, *args):
        ctx.run_function = run_function
        ctx.input_tensors = list(args[:length])
        ctx.input_params = list(args[length:])
        with torch.no_grad():
            output_tensors = ctx.run_function(*ctx.input_tensors)
        return output_tensors

    @staticmethod
    def backward(ctx, *output_grads):
        ctx.input_tensors = [x.detach().requires_grad_(True) for x in ctx.input_tensors]
        with torch.enable_grad():
            shallow_copies = [x.view_as(x) for x in ctx.input_tensors]
            output_tensors = ctx.run_function(*shallow_copies)
        input_grads = torch.autograd.grad(
            output_tensors,
            ctx.input_tensors + ctx.input_params,
            output_grads,
            allow_unused=True,
        )
        del ctx.input_tensors
        del ctx.input_params
        del output_tensors
        return (None, None) + input_grads


def checkpoint(func, inputs, params, flag):
    """[S3GM] Optional gradient checkpointing wrapper.

    Uses `torch.utils.checkpoint.checkpoint` (non-reentrant) under the hood,
    which picks up parameters via autograd automatically; the `params`
    argument is kept only for API compatibility.
    """
    if flag:
        return torch.utils.checkpoint.checkpoint(func, *inputs, use_reentrant=False)
    else:
        return func(*inputs)


# ─────────────────────────────────────────────────────────────────────────────
# [S3GM] FP16 helpers (from models/fp16_util.py)
# ─────────────────────────────────────────────────────────────────────────────
def convert_module_to_f16(l):
    """[S3GM] In-place conversion of a conv's weights/biases to fp16."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.half()
        l.bias.data = l.bias.data.half()


def convert_module_to_f32(l):
    """[S3GM] Inverse of `convert_module_to_f16`."""
    if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        l.weight.data = l.weight.data.float()
        l.bias.data = l.bias.data.float()


def make_master_params(model_params):
    """[S3GM] Flatten model params into a single master fp32 parameter tensor."""
    master_params = _flatten_dense_tensors(
        [param.detach().float() for param in model_params]
    )
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]


def model_grads_to_master_grads(model_params, master_params):
    """[S3GM] Copy fp16 gradients from `model_params` into the fp32 master buffer."""
    master_params[0].grad = _flatten_dense_tensors(
        [param.grad.data.detach().float() for param in model_params]
    )


def master_params_to_model_params(model_params, master_params):
    """[S3GM] Copy updated fp32 master params back into the fp16 live params."""
    model_params = list(model_params)
    for param, master_param in zip(
        model_params, unflatten_master_params(model_params, master_params)
    ):
        param.detach().copy_(master_param)


def unflatten_master_params(model_params, master_params):
    """[S3GM] Inverse of `make_master_params`: split the flat buffer back out."""
    return _unflatten_dense_tensors(master_params[0].detach(), model_params)


def zero_grad(model_params):
    """[S3GM] Zero out gradients in-place across an iterable of parameters."""
    for param in model_params:
        if param.grad is not None:
            param.grad.detach_()
            param.grad.zero_()


# ─────────────────────────────────────────────────────────────────────────────
# [S3GM] Relative Position Encoding (from models/rpe.py)
# ─────────────────────────────────────────────────────────────────────────────
class RPENet(nn.Module):
    """[S3GM] MLP that produces relative-position biases from (temb, dist)."""

    def __init__(self, channels, num_heads, time_embed_dim):
        super().__init__()
        self.embed_distances = nn.Linear(3, channels)
        self.embed_diffusion_time = nn.Linear(time_embed_dim, channels)
        self.silu = nn.SiLU()
        self.out = nn.Linear(channels, channels)
        self.out.weight.data *= 0.
        self.out.bias.data *= 0.
        self.channels = channels
        self.num_heads = num_heads

    def forward(self, temb, relative_distances):
        distance_embs = torch.stack(
            [torch.log(1 + (relative_distances).clamp(min=0)),
             torch.log(1 + (-relative_distances).clamp(min=0)),
             (relative_distances == 0).float()],
            dim=-1,
        )
        B, T, _ = relative_distances.shape
        C = self.channels
        emb = self.embed_diffusion_time(temb).view(B, T, 1, C) \
            + self.embed_distances(distance_embs)
        return self.out(self.silu(emb)).view(
            *relative_distances.shape, self.num_heads, self.channels // self.num_heads
        )


class RPE(nn.Module):
    """[S3GM] Relative position encoding module (Q/K/V aware)."""

    def __init__(self, channels, num_heads, time_embed_dim, use_rpe_net=False):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // self.num_heads
        self.use_rpe_net = use_rpe_net
        if use_rpe_net:
            self.rpe_net = RPENet(channels, num_heads, time_embed_dim)
        else:
            self.lookup_table_weight = nn.Parameter(
                torch.zeros(2 * self.beta + 1,
                            self.num_heads,
                            self.head_dim))

    def get_R(self, pairwise_distances, temb):
        if self.use_rpe_net:
            return self.rpe_net(temb, pairwise_distances)
        else:
            return self.lookup_table_weight[pairwise_distances]

    def forward(self, x, pairwise_distances, temb, mode):
        if mode == "qk":
            return self.forward_qk(x, pairwise_distances, temb)
        elif mode == "v":
            return self.forward_v(x, pairwise_distances, temb)
        else:
            raise ValueError(f"Unexpected RPE attention mode: {mode}")

    def forward_qk(self, qk, pairwise_distances, temb):
        R = self.get_R(pairwise_distances, temb)
        return torch.einsum("bdhtf,btshf->bdhts", qk, R)

    def forward_v(self, attn, pairwise_distances, temb):
        R = self.get_R(pairwise_distances, temb)
        torch.einsum("bdhts,btshf->bdhtf", attn, R)
        return torch.einsum("bdhts,btshf->bdhtf", attn, R)


class RPEAttention(nn.Module):
    """[S3GM] Self-attention with relative-position encoding on Q/K/V."""

    def __init__(self, channels, num_heads, use_checkpoint=False,
                 time_embed_dim=None, use_rpe_net=None,
                 use_rpe_q=True, use_rpe_k=True, use_rpe_v=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = channels // num_heads
        self.scale = head_dim ** -0.5
        self.use_checkpoint = use_checkpoint

        self.qkv = nn.Linear(channels, channels * 3)
        self.proj_out = zero_module(nn.Linear(channels, channels))
        self.norm = normalization(channels)

        if use_rpe_q or use_rpe_k or use_rpe_v:
            assert use_rpe_net is not None

        def make_rpe_func():
            return RPE(
                channels=channels, num_heads=num_heads,
                time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net,
            )
        self.rpe_q = make_rpe_func() if use_rpe_q else None
        self.rpe_k = make_rpe_func() if use_rpe_k else None
        self.rpe_v = make_rpe_func() if use_rpe_v else None

    def forward(self, x, temb, frame_indices, attn_mask=None, attn_weights_list=None):
        out, attn = checkpoint(
            self._forward, (x, temb, frame_indices, attn_mask),
            self.parameters(), self.use_checkpoint,
        )
        if attn_weights_list is not None:
            B, D, C, T = x.shape
            attn_weights_list.append(attn.detach().view(B * D, -1, T, T).mean(dim=1).abs())
        return out

    def _forward(self, x, temb, frame_indices, attn_mask):
        B, D, C, T = x.shape
        x = x.reshape(B * D, C, T)
        x = self.norm(x)
        x = x.view(B, D, C, T)
        x = torch.einsum("BDCT -> BDTC", x)
        qkv = self.qkv(x).reshape(B, D, T, 3, self.num_heads, C // self.num_heads)
        qkv = torch.einsum("BDTtHF -> tBDHTF", qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q *= self.scale
        attn = (q @ k.transpose(-2, -1))
        if self.rpe_q is not None or self.rpe_k is not None or self.rpe_v is not None:
            pairwise_distances = (frame_indices.unsqueeze(-1) - frame_indices.unsqueeze(-2))
        if self.rpe_k is not None:
            attn += self.rpe_k(q, pairwise_distances, temb=temb, mode="qk")
        if self.rpe_q is not None:
            attn += self.rpe_q(k * self.scale, pairwise_distances, temb=temb, mode="qk").transpose(-1, -2)

        def softmax(w, attn_mask):
            if attn_mask is not None:
                allowed_interactions = attn_mask.view(B, 1, T) * attn_mask.view(B, T, 1)
                allowed_interactions += (1 - attn_mask.view(B, 1, T)) * (1 - attn_mask.view(B, T, 1))
                inf_mask = (1 - allowed_interactions)
                inf_mask[inf_mask == 1] = torch.inf
                w = w - inf_mask.view(B, 1, 1, T, T)
            return torch.softmax(w.float(), dim=-1).type(w.dtype)

        attn = softmax(attn, attn_mask)
        out = attn @ v
        if self.rpe_v is not None:
            out += self.rpe_v(attn, pairwise_distances, temb=temb, mode="v")
        out = torch.einsum("BDHTF -> BDTHF", out).reshape(B, D, T, C)
        out = self.proj_out(out)
        x = x + out
        x = torch.einsum("BDTC -> BDCT", x)
        return x, attn


# ─────────────────────────────────────────────────────────────────────────────
# [S3GM] UNet video model (from models/unet_video.py)
# ─────────────────────────────────────────────────────────────────────────────
class TimestepBlock(nn.Module):
    """[S3GM] Abstract base for blocks that accept (x, emb)."""
    @abstractmethod
    def forward(self, x, emb):
        pass


class TimestepEmbedAttnThingsSequential(nn.Sequential, TimestepBlock):
    """[S3GM] Sequential that threads (emb, attn_mask, T, ...) through its children."""
    def forward(self, x, emb, attn_mask, T=1, frame_indices=None, attn_weights_list=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                kwargs = dict(emb=emb)
                kwargs['emb'] = emb
            elif isinstance(layer, FactorizedAttentionBlock):
                kwargs = dict(
                    temb=emb,
                    attn_mask=attn_mask,
                    T=T,
                    frame_indices=frame_indices,
                    attn_weights_list=attn_weights_list,
                )
            else:
                kwargs = {}
            x = layer(x, **kwargs)
        return x


class Upsample(nn.Module):
    """[S3GM] Nearest-neighbor upsample, optionally followed by a conv."""
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """[S3GM] Strided conv or average-pool downsampler."""
    def __init__(self, channels, use_conv, dims=2):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(dims, channels, channels, 3, stride=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """[S3GM] Residual block with optional scale-shift-norm timestep conditioning."""

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class FactorizedAttentionBlock(nn.Module):
    """[S3GM] Factorized spatial + temporal attention block."""

    def __init__(self, channels, num_heads, use_rpe_net, time_embed_dim=None, use_checkpoint=False):
        super().__init__()
        self.spatial_attention = RPEAttention(
            channels=channels, num_heads=num_heads, use_checkpoint=use_checkpoint,
            use_rpe_q=False, use_rpe_k=False, use_rpe_v=False,
        )
        self.temporal_attention = RPEAttention(
            channels=channels, num_heads=num_heads, use_checkpoint=use_checkpoint,
            time_embed_dim=time_embed_dim, use_rpe_net=use_rpe_net,
        )

    def forward(self, x, attn_mask, temb, T, frame_indices=None, attn_weights_list=None):
        if len(x.shape) == 4:
            BT, C, H, W = x.shape
            B = BT // T
            x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1)
            x = x.reshape(B, H * W, C, T)
            x = self.temporal_attention(
                x, temb, frame_indices,
                attn_mask=attn_mask.flatten(start_dim=2).squeeze(dim=2),
                attn_weights_list=None if attn_weights_list is None else attn_weights_list['temporal'],
            )
            x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)
            x = x.reshape(B, T, C, H * W)
            x = self.spatial_attention(
                x, temb, frame_indices=None,
                attn_weights_list=None if attn_weights_list is None else attn_weights_list['spatial'],
            )
            x = x.reshape(BT, C, H, W)
        elif len(x.shape) == 3:
            BT, C, H = x.shape
            B = BT // T
            x = x.view(B, T, C, H).permute(0, 3, 2, 1)
            x = x.reshape(B, H, C, T)
            x = self.temporal_attention(
                x, temb, frame_indices,
                attn_mask=attn_mask.flatten(start_dim=2).squeeze(dim=2),
                attn_weights_list=None if attn_weights_list is None else attn_weights_list['temporal'],
            )
            x = x.view(B, H, C, T).permute(0, 3, 2, 1)
            x = x.reshape(B, T, C, H)
            x = self.spatial_attention(
                x, temb, frame_indices=None,
                attn_weights_list=None if attn_weights_list is None else attn_weights_list['spatial'],
            )
            x = x.reshape(BT, C, H)
        return x


class UNetVideoModel(nn.Module):
    """[S3GM] Main U-Net video model used by S3GM for score prediction."""

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        image_size=None,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        use_rpe_net=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels + 1
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample
        self.use_rpe_net = use_rpe_net

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedAttnThingsSequential(
                    conv_nd(dims, self.in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                            use_rpe_net=use_rpe_net, time_embed_dim=time_embed_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedAttnThingsSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedAttnThingsSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
            ),
            FactorizedAttentionBlock(
                ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                use_rpe_net=use_rpe_net, time_embed_dim=time_embed_dim,
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims,
                use_checkpoint=use_checkpoint, use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            use_rpe_net=use_rpe_net,
                            time_embed_dim=time_embed_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedAttnThingsSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, *, x0, timesteps, frame_indices=None,
                obs_mask=None, latent_mask=None, return_attn_weights=False):
        if len(x.shape) == 5:
            B, T, C, H, W = x.shape
        elif len(x.shape) == 4:
            B, T, C, H = x.shape
        timesteps = timesteps.view(B, 1).expand(B, T)
        attn_mask = (obs_mask + latent_mask).clip(max=1)
        indicator_template = torch.ones_like(x[:, :, :1, :, :]) if len(x.shape) == 5 else torch.ones_like(x[:, :, :1, :])
        obs_indicator = indicator_template * obs_mask
        x = torch.cat([x * (1 - obs_mask) + x0 * obs_mask, obs_indicator], dim=2)
        x = x.reshape(B * T, self.in_channels, *x.shape[3:])
        timesteps = timesteps.reshape(B * T)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        h = x.type(x.dtype)
        attns = {'spatial': [], 'temporal': [], 'mixed': []} if return_attn_weights else None
        for layer, module in enumerate(self.input_blocks):
            h = module(h, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
            hs.append(h)
        h = self.middle_block(h, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb, attn_mask, T=T, attn_weights_list=attns, frame_indices=frame_indices)
        h = h.type(x.dtype)
        out = self.out(h)
        return out.view(B, T, self.out_channels, *x.shape[2:]), attns

    def get_feature_vectors(self, x, timesteps, y=None):
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result


# ─────────────────────────────────────────────────────────────────────────────
# [S3GM] Exponential Moving Average (from models/ema.py)
# ─────────────────────────────────────────────────────────────────────────────
class ExponentialMovingAverage:
    """[S3GM] EMA helper (distinct from the lightweight SiTEMA class above)."""

    def __init__(self, parameters, decay, use_num_updates=True):
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow_params = [p.clone().detach()
                              for p in parameters if p.requires_grad]
        self.collected_params = []

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            parameters = [p for p in parameters if p.requires_grad]
            for s_param, param in zip(self.shadow_params, parameters):
                s_param.sub_(one_minus_decay * (s_param - param))

    def copy_to(self, parameters):
        parameters = [p for p in parameters if p.requires_grad]
        for s_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)

    def store(self, parameters):
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def state_dict(self):
        return dict(decay=self.decay, num_updates=self.num_updates,
                    shadow_params=self.shadow_params)

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.shadow_params = state_dict['shadow_params']


# ─────────────────────────────────────────────────────────────────────────────
# [S3GM] SDE / VESDE / VPSDE (from sampler/sde.py)
# ─────────────────────────────────────────────────────────────────────────────
class SDE(abc.ABC):
    """[S3GM] Abstract base class for stochastic differential equations."""

    def __init__(self, N):
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        pass

    @abc.abstractmethod
    def sde(self, x, t):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, t):
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape):
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        pass

    def discretize(self, x, t):
        dt = 1 / self.N
        drift, diffusion = self.sde(x, t)
        f = drift * dt
        G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
        return f, G

    def reverse(self, net_fn, probability_flow=False):
        N = self.N
        T = self.T
        sde_fn = self.sde
        discretize_fn = self.discretize

        class RSDE(self.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, t):
                drift, diffusion = sde_fn(x, t)
                score = net_fn(x.float(), t.float())
                drift = drift - diffusion[:, None, None, None] ** 2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(diffusion) if self.probability_flow else diffusion
                return drift, diffusion

            def discretize(self, x, t):
                f, G = discretize_fn(x, t)
                rev_f = f - G[:, None, None, None] ** 2 * net_fn(x, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()


class VESDE(SDE):
    """[S3GM] Variance-Exploding SDE (log-linear sigma schedule)."""

    def __init__(self, config, sigma_min=0.1, sigma_max=20, N=1000):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), N))
        self.N = N
        self.config = config

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        drift = torch.zeros_like(x)
        diffusion = sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min)),
                                                    device=t.device))
        return drift, diffusion

    def marginal_prob(self, x, t):
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        mean = x
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape) * self.sigma_max

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        return -N / 2. * np.log(2 * np.pi * self.sigma_max ** 2) - torch.sum(z ** 2, dim=(1, 2, 3)) / (
                    2 * self.sigma_max ** 2)

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        discrete_sigmas = self.discrete_sigmas.to(t.device)
        sigma = discrete_sigmas[timestep]
        adjacent_sigma = torch.where(timestep == 0, torch.zeros_like(t),
                                     discrete_sigmas[torch.clamp(timestep - 1, min=0)])
        f = torch.zeros_like(x)
        G = torch.sqrt(sigma ** 2 - adjacent_sigma ** 2)
        return f, G


class VPSDE(SDE):
    """[S3GM] Variance-Preserving SDE (linear beta schedule)."""

    def __init__(self, config, beta_min=0.1, beta_max=20, N=1000):
        super().__init__(N)
        self.config = config
        self.beta_0 = beta_min
        self.beta_1 = beta_max
        self.N = N
        self.discrete_betas = torch.linspace(beta_min / N, beta_max / N, N)
        self.alphas = 1. - self.discrete_betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    @property
    def T(self):
        return 1

    def sde(self, x, t):
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion

    def marginal_prob(self, x, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        if len(x.shape) == 4:
            mean = torch.exp(log_mean_coeff[:, None, None, None]) * x
        else:
            mean = torch.exp(log_mean_coeff[:, None, None, None, None]) * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

    def prior_sampling(self, shape):
        return torch.randn(*shape)

    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1, 2, 3)) / 2.
        return logps

    def discretize(self, x, t):
        timestep = (t * (self.N - 1) / self.T).long()
        beta = self.discrete_betas.to(x.device)[timestep]
        alpha = self.alphas.to(x.device)[timestep]
        sqrt_beta = torch.sqrt(beta)
        if len(x.shape) > 4:
            f = torch.sqrt(alpha)[:, None, None, None, None] * x - x
        else:
            f = torch.sqrt(alpha)[:, None, None, None] * x - x
        G = sqrt_beta
        return f, G


# ─────────────────────────────────────────────────────────────────────────────
# [S3GM] Learned grid deformer (Geo-FNO trick, identity-residual init)
# ─────────────────────────────────────────────────────────────────────────────
class S3GMLearnedGridDeformer(nn.Module):
    """[S3GM] Learned MLP mapping normalized 2-D coordinates to latent 2-D
    coordinates in [0, 1].

    Uses an identity-residual init: the final linear is zero-initialized, so at
    step 0 the deformer is exactly the identity on the already-normalized
    input coords. This avoids the degenerate init where a Sigmoid layer with
    standard weights collapses every input to ~(0.5, 0.5) and all airfoil
    points splat to the same grid cell.

    Distinct from `SiTLearnedGridDeformer` (Sigmoid final layer, different
    init) — the two stay as separate classes so existing checkpoints remain
    load-compatible.
    """

    def __init__(self, coord_dim: int = 2, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        layers = []
        dim = coord_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU()]
            dim = hidden_dim
        final = nn.Linear(dim, 2)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        layers.append(final)
        self.net = nn.Sequential(*layers)

    def forward(self, coords_2d: torch.Tensor) -> torch.Tensor:
        """[B, N, 2] -> [B, N, 2] in [0, 1] (identity at init, clamped thereafter)."""
        return (coords_2d + self.net(coords_2d)).clamp(0.0, 1.0)

# =============================================================================
# Unified baseline helpers / adapters
# =============================================================================

import argparse
import contextlib
import copy
import csv
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import matplotlib.pyplot as plt
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import helpers_baseline as BASELINE_HELPERS
    from sit_transport import Transport as LinearVelocityTransport, create_transport, Sampler
except ImportError:
    from . import helpers_baseline as BASELINE_HELPERS
    from .sit_transport import Transport as LinearVelocityTransport, create_transport, Sampler

REPO_ROOT = Path(__file__).resolve().parents[1]

FIELD_NAMES = BASELINE_HELPERS.FIELD_NAMES
TurbulentCombustionH5Dataset = BASELINE_HELPERS.TurbulentCombustionH5Dataset
collate_snapshots = BASELINE_HELPERS.collate_snapshots
validate_regular_grid_compatibility = BASELINE_HELPERS.validate_regular_grid_compatibility
build_sparse_condition = BASELINE_HELPERS.build_sparse_condition
compute_pad_size = BASELINE_HELPERS.compute_pad_size
pointcloud_to_grid_padded = BASELINE_HELPERS.pointcloud_to_grid_padded
grid_to_pointcloud = BASELINE_HELPERS.grid_to_pointcloud
build_obs_grid_mask = BASELINE_HELPERS.build_obs_grid_mask
nearest_fill_grid = BASELINE_HELPERS.nearest_fill_grid
scatter_sensors_to_nodes = BASELINE_HELPERS.scatter_sensors_to_nodes
pointcloud_to_grid = BASELINE_HELPERS.pointcloud_to_grid
gather_from_grid = BASELINE_HELPERS.gather_from_grid
splat_to_grid = BASELINE_HELPERS.splat_to_grid
splat_obs_to_grid = BASELINE_HELPERS.splat_obs_to_grid
_save_single_field_plot = BASELINE_HELPERS._save_single_field_plot
_build_structured_triangulation = BASELINE_HELPERS._build_structured_triangulation

SUPPORTED_BASELINES = {"s3gm", "latent_fm", "sit"}


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def safe_torch_load(path: Path, map_location: str | torch.device = "cpu") -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)
    except (pickle.UnpicklingError, RuntimeError):
        return torch.load(path, map_location=map_location, weights_only=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_device(device_override: Optional[str], device_ids: Sequence[int]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device(f"cuda:{int(device_ids[0])}")
    return torch.device("cpu")


def ensure_absolute(path_value: str | Path, base_dir: Path = REPO_ROOT) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def validate_and_normalize_config(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    baseline_model = str(cfg.get("baseline_model", "")).strip().lower()
    if baseline_model not in SUPPORTED_BASELINES:
        raise ValueError(
            f"Unknown baseline_model={baseline_model!r}. "
            f"Expected one of {sorted(SUPPORTED_BASELINES)}."
        )
    cfg["baseline_model"] = baseline_model

    try:
        training_stage = int(cfg.get("training_stage"))
    except Exception as exc:
        raise ValueError("training_stage must be explicitly set to 1 or 2.") from exc
    if training_stage not in (1, 2):
        raise ValueError("training_stage must be 1 or 2.")
    cfg["training_stage"] = training_stage

    if baseline_model in {"s3gm", "sit"} and training_stage != 1:
        raise ValueError(
            f"baseline_model={baseline_model!r} only supports training_stage=1."
        )

    shared = cfg.setdefault("shared", {})
    shared.setdefault("demo_num", 0)
    shared.setdefault("seed", 42)
    shared.setdefault("device_ids", [0])
    shared.setdefault("reload", False)
    shared.setdefault("paths", {})
    shared["paths"].setdefault("data_path", "Dataset/Merged_CH4COTU1P.h5")
    shared["paths"].setdefault("save_root", "Save_TrainedModel")
    shared["paths"].setdefault("config_backup_root", "Save_config/gen_baseline")
    shared.setdefault("data", {})
    shared["data"].setdefault("dataset_name", "turbulent_combustion")
    shared["data"].setdefault("train_ratio", 0.9)
    shared["data"].setdefault("time_stride", 1)
    shared["data"].setdefault("num_workers", 4)
    shared["data"].setdefault("num_x", 403)
    shared["data"].setdefault("num_y", 100)
    shared.setdefault("conditioning", {})
    shared["conditioning"].setdefault("cond_fields", [2, 3])
    shared["conditioning"].setdefault("n_obs_min_list", [128, 128])
    shared["conditioning"].setdefault("n_obs_max_list", [512, 512])
    if shared["conditioning"].get("vis_cond_fields") is None:
        shared["conditioning"]["vis_cond_fields"] = list(shared["conditioning"]["cond_fields"])
    if shared["conditioning"].get("vis_n_obs_list") is None:
        shared["conditioning"]["vis_n_obs_list"] = list(shared["conditioning"]["n_obs_max_list"])
    shared.setdefault("logging", {})
    shared["logging"].setdefault("eval_every", 5)
    shared["logging"].setdefault("save_every", 200)
    shared.setdefault("overrides", {})
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        shared["overrides"].setdefault(key, None)

    if shared["data"]["dataset_name"] != "turbulent_combustion":
        raise NotImplementedError(
            "The unified baseline flow is wired to the turbulent combustion "
            "dataset in this pass. The adapter structure is ready for more "
            "datasets, but they have not been merged into the new config yet."
        )
    return cfg


def resolve_stage_config(cfg: dict) -> dict:
    cfg = validate_and_normalize_config(cfg)
    baseline = cfg["baseline_model"]
    stage = cfg["training_stage"]
    shared = cfg["shared"]

    if baseline == "latent_fm":
        stage_cfg = copy.deepcopy(cfg["latent_fm_params"][f"stage{stage}"])
    else:
        stage_cfg = copy.deepcopy(cfg[f"{baseline}_params"])

    training = stage_cfg.setdefault("training", {})
    training.setdefault("eval_every", shared["logging"]["eval_every"])
    training.setdefault("save_every", shared["logging"]["save_every"])

    overrides = shared["overrides"]
    for key in ("epochs", "batch_size", "learning_rate", "weight_decay"):
        if overrides.get(key) is not None:
            training[key] = overrides[key]

    return stage_cfg


def build_run_name(cfg: dict, timestamp: str) -> str:
    return (
        f"Baseline_{cfg['baseline_model']}_Stage{cfg['training_stage']}"
        f"_DemoN{int(cfg['shared']['demo_num'])}_{timestamp}"
    )


def list_run_dirs(save_root: Path, cfg: dict) -> list[Path]:
    prefix = (
        f"Baseline_{cfg['baseline_model']}_Stage{cfg['training_stage']}"
        f"_DemoN{int(cfg['shared']['demo_num'])}_"
    )
    if not save_root.exists():
        return []
    return sorted([path for path in save_root.glob(f"{prefix}*") if path.is_dir()])


def find_latest_run_dir(save_root: Path, cfg: dict) -> Optional[Path]:
    run_dirs = list_run_dirs(save_root, cfg)
    return run_dirs[-1] if run_dirs else None


def copy_config_backup(config_path: Path, backup_root: Path, run_name: str) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    target = backup_root / f"{run_name}.yaml"
    target.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def build_dataset(cfg: dict, split: str, stats_path: Path) -> TurbulentCombustionH5Dataset:
    shared = cfg["shared"]
    data_path = ensure_absolute(shared["paths"]["data_path"])
    return TurbulentCombustionH5Dataset(
        str(data_path),
        split=split,
        train_ratio=float(shared["data"]["train_ratio"]),
        seed=int(shared["seed"]),
        time_stride=int(shared["data"]["time_stride"]),
        stats_path=str(stats_path),
    )


def build_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_snapshots,
    )


class TrainingHistoryLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.csv_path = run_dir / "loss_history.csv"
        self.json_path = run_dir / "loss_history.json"
        self.plot_path = run_dir / "loss_history.png"
        self.rows: list[dict[str, Any]] = []
        with open(self.csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "val_loss"])

    def append(self, epoch: int, train_loss: float, val_loss: Optional[float]) -> None:
        row = {
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": None if val_loss is None else float(val_loss),
        }
        self.rows.append(row)
        with open(self.csv_path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                "" if row["val_loss"] is None else row["val_loss"],
            ])
        with open(self.json_path, "w", encoding="utf-8") as handle:
            json.dump(self.rows, handle, indent=2)
        self._save_plot()

    def _save_plot(self) -> None:
        epochs = [row["epoch"] for row in self.rows]
        train_points = [
            (row["epoch"], row["train_loss"])
            for row in self.rows
            if row["train_loss"] is not None and float(row["train_loss"]) > 0.0
        ]
        val_points = [
            (row["epoch"], row["val_loss"])
            for row in self.rows
            if row["val_loss"] is not None and float(row["val_loss"]) > 0.0
        ]

        fig, ax = plt.subplots(figsize=(10, 6))
        if train_points:
            ax.plot(
                [epoch for epoch, _ in train_points],
                [loss for _, loss in train_points],
                label="Train Loss",
                marker="o",
                color="blue",
                markersize=4,
            )
        if val_points:
            ax.plot(
                [epoch for epoch, _ in val_points],
                [loss for _, loss in val_points],
                label="Validation Loss",
                marker="s",
                color="orange",
                markersize=5,
            )

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss History")
        if train_points or val_points:
            ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        if epochs:
            ax.set_xlim(left=1, right=max(epochs))
        if train_points or val_points:
            ax.legend()
        fig.tight_layout()
        fig.savefig(self.plot_path, dpi=150)
        plt.close(fig)


@dataclass
class BaselineBundle:
    baseline_model: str
    training_stage: int
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Optional[Any]
    ema: Optional[Any]
    device: torch.device
    run_dir: Path
    config: dict
    dataset_train: Optional[Any] = None
    dataset_val: Optional[Any] = None
    components: dict[str, Any] = field(default_factory=dict)


class LatentEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self._backup: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply(self, model: nn.Module) -> None:
        self._backup = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self._backup:
                param.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self) -> dict:
        return {name: value.detach().clone() for name, value in self.shadow.items()}

    def load_state_dict(self, state_dict: dict) -> None:
        for name, value in state_dict.items():
            if name in self.shadow:
                self.shadow[name].copy_(value)


def s3gm_loss(net, sde, x_grid, eps: float = 1e-5):
    batch_size = x_grid.shape[0]
    x = x_grid.unsqueeze(1)

    t = torch.rand(batch_size, device=x.device) * (sde.T - eps) + eps
    noise = torch.randn_like(x)
    mean, std = sde.marginal_prob(x, t)
    std_e = std[:, None, None, None, None]
    perturbed = mean + std_e * noise

    obs_mask = torch.zeros(batch_size, 1, 1, 1, 1, device=x.device)
    latent_mask = torch.ones(batch_size, 1, 1, 1, 1, device=x.device)
    frame_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=x.device)

    score, _ = net(
        perturbed,
        x0=x,
        timesteps=std,
        frame_indices=frame_indices,
        obs_mask=obs_mask,
        latent_mask=latent_mask,
    )
    losses = torch.square(score * std_e + noise)
    losses = losses.reshape(batch_size, -1).mean(dim=-1)
    return losses.mean()


def run_epoch_s3gm(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    net = bundle.model
    optimizer = bundle.optimizer if training else None
    sde = bundle.components["sde"]
    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])
    h_pad = int(bundle.components["H_pad"])
    w_pad = int(bundle.components["W_pad"])
    ema = bundle.ema
    all_params_fn = bundle.components["all_params_fn"]

    net.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"S3GM Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    for batch in pbar:
        fields_full = batch["fields"].to(bundle.device)
        x_grid = pointcloud_to_grid_padded(fields_full, num_y, num_x, h_pad, w_pad)
        loss = s3gm_loss(net, sde, x_grid)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(all_params_fn(), max_norm=0.5)
            if torch.isfinite(loss) and torch.isfinite(grad_norm) and grad_norm <= 50.0:
                optimizer.step()
                if ema is not None:
                    ema.update(all_params_fn())
            else:
                optimizer.zero_grad(set_to_none=True)

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


def run_epoch_ae(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    ae = bundle.model
    optimizer = bundle.optimizer if training else None
    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])

    ae.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"LatentFM-AE Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    for batch in pbar:
        fields_full = batch["fields"].to(bundle.device)
        x_grid = pointcloud_to_grid(fields_full, num_y, num_x)
        x_hat, _ = ae(x_grid)
        x_hat = x_hat[:, :, :num_y, :num_x]
        loss = 0.5 * F.l1_loss(x_hat, x_grid) + 0.5 * F.mse_loss(x_hat, x_grid)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(ae.parameters(), max_norm=1.0)
            optimizer.step()

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


def run_epoch_latentfm(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    model = bundle.model
    optimizer = bundle.optimizer if training else None
    ema = bundle.ema
    velocity_net = bundle.components["velocity_net"]
    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])
    conditioning = bundle.config["shared"]["conditioning"]

    cond_fields = conditioning["cond_fields"]
    n_obs_min = conditioning["n_obs_min_list"]
    n_obs_max = conditioning["n_obs_max_list"]

    model.velocity_net.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"LatentFM Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    n_fields = model.n_fields
    n_pts = num_y * num_x

    for batch in pbar:
        coords = batch["coords"].to(bundle.device)
        fields_full = batch["fields"].to(bundle.device)
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=fields_full,
            cond_fields=cond_fields,
            n_obs_min=n_obs_min,
            n_obs_max=n_obs_max,
        )

        fields_grid = pointcloud_to_grid(fields_full, num_y, num_x)
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            num_y,
            num_x,
            num_y,
            num_x,
        )
        ix = (obs_indices % num_x).float() / max(num_x - 1, 1)
        iy = (obs_indices.div(num_x, rounding_mode="floor")).float() / max(num_y - 1, 1)
        obs_coords_2d = torch.stack([ix, iy], dim=-1)

        cond_inputs = {
            "obs_value_grid": obs_value_grid,
            "obs_mask_grid": obs_mask_grid,
            "obs_coords_2d": obs_coords_2d,
            "obs_values": obs_values,
            "obs_mask": obs_mask,
            "obs_field_ids": obs_field_ids,
        }
        loss, _ = model.training_loss(fields_grid, cond_inputs)

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.velocity_net.parameters(), max_norm=1.0)
            optimizer.step()
            if ema is not None:
                ema.update(velocity_net)

        current_loss = float(loss.detach().cpu())
        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


def run_epoch_sit(bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
    net = bundle.model
    optimizer = bundle.optimizer if training else None
    transport = bundle.components["transport"]
    h_pad = int(bundle.components["H_pad"])
    w_pad = int(bundle.components["W_pad"])
    cond_mode = bundle.components["cond_mode"]
    tokenizer = bundle.components["tokenizer"]
    spike_state = bundle.components["spike_state"]
    ema = bundle.ema
    all_params_fn = bundle.components["all_params_fn"]
    n_fields = bundle.components["n_fields"]

    num_y = int(bundle.config["shared"]["data"]["num_y"])
    num_x = int(bundle.config["shared"]["data"]["num_x"])
    cond_fields = bundle.config["shared"]["conditioning"]["cond_fields"]
    n_obs_min = bundle.config["shared"]["conditioning"]["n_obs_min_list"]
    n_obs_max = bundle.config["shared"]["conditioning"]["n_obs_max_list"]
    huber_beta = bundle.components["huber_beta"]

    net.train(training)
    total_loss = 0.0
    count = 0
    pbar = tqdm(loader, desc=f"SiT Epoch {epoch:04d} [{'Train' if training else 'Eval'}]", leave=False)

    for batch in pbar:
        fields_full = batch["fields"].to(bundle.device)
        coords = batch["coords"].to(bundle.device)
        model_kwargs: dict[str, Any] = {}

        if tokenizer == "pointnet":
            x_grid = fields_full
            obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
                coords_full=coords,
                fields_full=fields_full,
                cond_fields=cond_fields,
                n_obs_min=n_obs_min,
                n_obs_max=n_obs_max,
            )
            obs_value_nodes, obs_mask_nodes = scatter_sensors_to_nodes(
                obs_values,
                obs_mask,
                obs_field_ids,
                obs_indices,
                fields_full.shape[0],
                fields_full.shape[1],
                n_fields,
                bundle.device,
                fields_full.dtype,
            )
            model_kwargs["coords"] = coords
            model_kwargs["obs_value_nodes"] = obs_value_nodes
            model_kwargs["obs_mask_nodes"] = obs_mask_nodes
        else:
            x_grid = pointcloud_to_grid_padded(fields_full, num_y, num_x, h_pad, w_pad)
            obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
                coords_full=coords,
                fields_full=fields_full,
                cond_fields=cond_fields,
                n_obs_min=n_obs_min,
                n_obs_max=n_obs_max,
            )
            obs_value_grid, obs_mask_grid = build_obs_grid_mask(
                obs_values,
                obs_mask,
                obs_field_ids,
                obs_indices,
                n_fields,
                fields_full.shape[1],
                num_y,
                num_x,
                h_pad,
                w_pad,
            )
            if cond_mode == "interp":
                obs_value_grid = nearest_fill_grid(obs_value_grid, obs_mask_grid)
            model_kwargs["obs_value_grid"] = obs_value_grid
            model_kwargs["obs_mask_grid"] = obs_mask_grid

        loss_dict = transport.training_losses(
            net,
            x_grid,
            model_kwargs=model_kwargs,
            huber_beta=huber_beta,
        )
        loss = loss_dict["loss"].float().mean()
        current_loss = float(loss.detach().cpu())

        if training and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = float(nn.utils.clip_grad_norm_(all_params_fn(), max_norm=0.5))

            skip = not (math.isfinite(current_loss) and math.isfinite(grad_norm))
            ema_loss = spike_state.get("ema_loss")
            ema_grad = spike_state.get("ema_grad")
            if not skip and ema_loss is not None and current_loss > 5.0 * ema_loss:
                skip = True
            if not skip and ema_grad is not None and grad_norm > 10.0 * ema_grad:
                skip = True

            if skip:
                optimizer.zero_grad(set_to_none=True)
                spike_state["skipped"] = spike_state.get("skipped", 0) + 1
            else:
                optimizer.step()
                if ema is not None:
                    ema.update(all_params_fn())
                beta = 0.99
                spike_state["ema_loss"] = (
                    current_loss if ema_loss is None else beta * ema_loss + (1.0 - beta) * current_loss
                )
                spike_state["ema_grad"] = (
                    grad_norm if ema_grad is None else beta * ema_grad + (1.0 - beta) * grad_norm
                )

        total_loss += current_loss
        count += 1
        pbar.set_postfix_str(f"loss={current_loss:.6e}")

    return total_loss / max(count, 1)


@torch.no_grad()
def visualize_ae_reconstruction(
    ae,
    dataset,
    epoch,
    device,
    save_dir,
    Num_y,
    Num_x,
    snapshot_index=0,
    file_tag=None,
    irregular=False,
):
    ae.eval()
    sample = dataset[snapshot_index]
    fields = sample["fields"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"]

    if irregular:
        coords = sample["coords"].unsqueeze(0).to(device)
        if ae.deformer is not None:
            coords_2d = ae.deform(coords)
        else:
            coords_2d = coords[..., :2]
        x_grid = splat_to_grid(coords_2d, fields, Num_y, Num_x)[0]
        x_hat_grid, _ = ae(x_grid)
        recon_pc = gather_from_grid(x_hat_grid[:, :, :Num_y, :Num_x], coords_2d)
    else:
        x_grid = pointcloud_to_grid(fields, Num_y, Num_x)
        x_hat, _ = ae(x_grid)
        x_hat = x_hat[:, :, :Num_y, :Num_x]
        recon_pc = x_hat.permute(0, 2, 3, 1).reshape(1, -1, ae.n_fields)

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    truth = fields * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon = recon_pc * std.view(1, 1, -1) + mean.view(1, 1, -1)

    truth_np = truth[0].cpu().numpy()
    recon_np = recon[0].cpu().numpy()
    coords_xy = coords_raw.cpu().numpy()[:, :2]

    import matplotlib.tri as mtri
    tri = mtri.Triangulation(coords_xy[:, 0], coords_xy[:, 1])

    metrics = {}
    for c, name in enumerate(dataset.field_names):
        true_f = truth_np[:, c]
        pred_f = recon_np[:, c]
        l2_err = float(np.linalg.norm(true_f - pred_f) / (np.linalg.norm(true_f) + 1e-8))
        metrics[name] = l2_err

        fig, axs = plt.subplots(1, 3, figsize=(18, 4))
        vmin = min(true_f.min(), pred_f.min())
        vmax = max(true_f.max(), pred_f.max())
        for ax, data, title in zip(
            axs,
            [true_f, pred_f, np.abs(true_f - pred_f)],
            [f"{name} Truth", f"{name} AE Recon", f"{name} |Error| (L2={l2_err:.4e})"],
        ):
            im = ax.tricontourf(
                tri,
                data,
                levels=100,
                cmap="coolwarm" if "Error" not in title else "inferno",
                vmin=vmin if "Error" not in title else None,
                vmax=vmax if "Error" not in title else None,
            )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(title)
            ax.set_aspect("equal")

        prefix = file_tag if file_tag else f"epoch_{epoch:04d}"
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f"{prefix}_ae_field_{name}.png"), dpi=150)
        plt.close(fig)

    return metrics


def _score_fn_nograd(net, sde, x_5d, t_vec):
    B = x_5d.shape[0]
    obs_mask = torch.zeros(B, 1, 1, 1, 1, device=x_5d.device)
    latent_mask = torch.ones(B, 1, 1, 1, 1, device=x_5d.device)
    frame_indices = torch.zeros(B, 1, dtype=torch.long, device=x_5d.device)
    _, std = sde.marginal_prob(torch.zeros_like(x_5d), t_vec)
    with torch.no_grad():
        score, _ = net(
            x_5d,
            x0=x_5d,
            timesteps=std,
            frame_indices=frame_indices,
            obs_mask=obs_mask,
            latent_mask=latent_mask,
        )
    return score


def _score_fn_grad(net, sde, x_5d, t_vec):
    B = x_5d.shape[0]
    obs_mask = torch.zeros(B, 1, 1, 1, 1, device=x_5d.device)
    latent_mask = torch.ones(B, 1, 1, 1, 1, device=x_5d.device)
    frame_indices = torch.zeros(B, 1, dtype=torch.long, device=x_5d.device)
    _, std = sde.marginal_prob(torch.zeros_like(x_5d), t_vec)
    score, _ = net(
        x_5d,
        x0=x_5d,
        timesteps=std,
        frame_indices=frame_indices,
        obs_mask=obs_mask,
        latent_mask=latent_mask,
    )
    return score


def dps_sample(
    net,
    sde,
    shape_5d,
    obs_value_grid,
    obs_mask_grid,
    device,
    N_steps=200,
    snr=0.128,
    n_corrector_steps=1,
    alpha_obs=1.0,
    eps=1e-3,
):
    B = shape_5d[0]
    x = sde.prior_sampling(list(shape_5d)).to(device).float()
    timesteps = torch.linspace(sde.T, eps, N_steps + 1, device=device).float()
    obs_5d = obs_value_grid.unsqueeze(1)
    mask_5d = obs_mask_grid.unsqueeze(1)

    for i in tqdm(range(N_steps), desc="S3GM sampling", leave=False):
        t_cur = timesteps[i]
        t_next = timesteps[i + 1]
        vec_t = torch.ones(B, device=device).float() * t_cur

        sigma_cur = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t_cur
        sigma_nxt = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t_next
        step_std = torch.sqrt(torch.clamp(sigma_cur ** 2 - sigma_nxt ** 2, min=0.0))

        for _ in range(n_corrector_steps):
            grad = _score_fn_nograd(net, sde, x, vec_t)
            grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(B, -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(B, -1), dim=-1).mean()
            step_size = (snr * noise_norm / (grad_norm + 1e-12)) ** 2 * 2.0
            step_size = torch.clamp(step_size, max=1.0)
            x = x + step_size * grad + torch.sqrt(2.0 * step_size) * noise

        with torch.enable_grad():
            x_inp = x.detach().clone().requires_grad_(True)
            score = _score_fn_grad(net, sde, x_inp, vec_t)

            with torch.no_grad():
                score_clean = torch.nan_to_num(score.detach(), nan=0.0, posinf=0.0, neginf=0.0)
                G_e = step_std.view(1, 1, 1, 1, 1)
                rev_f = -(G_e ** 2) * score_clean
                x_mean = x_inp.detach() - rev_f
                if i < N_steps - 1:
                    z = torch.randn_like(x_inp)
                    x_next = x_mean + G_e * z
                else:
                    x_next = x_mean

            std_e = sigma_cur.view(1, 1, 1, 1, 1)
            x0_hat = std_e ** 2 * score + x_inp
            residual = (obs_5d - x0_hat) * mask_5d
            dps_loss = alpha_obs * torch.sum(residual ** 2)
            if torch.isfinite(dps_loss) and dps_loss > 0:
                dx = torch.autograd.grad(dps_loss, x_inp)[0]
                dx = torch.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
                dx = torch.clamp(dx, min=-1e8, max=1e8)
            else:
                dx = torch.zeros_like(x_inp)

        x = (x_next - dx).detach()
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    return x.squeeze(1)


def visualize_reconstruction_s3gm(
    net,
    sde,
    dataset,
    Num_x,
    Num_y,
    H_pad,
    W_pad,
    epoch,
    device,
    save_dir,
    cond_fields=(2,),
    n_obs=256,
    N_steps=200,
    snr=0.128,
    n_corrector_steps=1,
    alpha_obs=1.0,
    snapshot_index=0,
    file_tag=None,
    save_metrics_json=True,
    irregular=False,
    deformer=None,
):
    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs, int):
        n_obs = [n_obs] * len(cond_fields)

    n_fields = dataset.num_fields
    n_pts = dataset.num_points
    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs,
        n_obs_max=n_obs,
        valid_mask=valid_mask,
    )

    if irregular:
        coords_2d = coords[..., :2]
        if deformer is not None:
            latent_coords = deformer(coords_2d)
            latent_obs_coords = deformer(obs_coords[..., :2])
        else:
            latent_coords = coords_2d
            latent_obs_coords = obs_coords[..., :2]
        obs_value_grid, obs_mask_grid = splat_obs_to_grid(
            latent_obs_coords,
            obs_values,
            obs_mask,
            obs_field_ids,
            n_fields,
            Num_y,
            Num_x,
        )
        if H_pad > Num_y or W_pad > Num_x:
            obs_value_grid = F.pad(obs_value_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)
            obs_mask_grid = F.pad(obs_mask_grid, (0, W_pad - Num_x, 0, H_pad - Num_y), value=0)
    else:
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            Num_y,
            Num_x,
            H_pad,
            W_pad,
        )

    with torch.enable_grad():
        recon_grid = dps_sample(
            net=net,
            sde=sde,
            shape_5d=(1, 1, n_fields, H_pad, W_pad),
            obs_value_grid=obs_value_grid,
            obs_mask_grid=obs_mask_grid,
            device=device,
            N_steps=N_steps,
            snr=snr,
            n_corrector_steps=n_corrector_steps,
            alpha_obs=alpha_obs,
        )

    if irregular:
        recon = gather_from_grid(recon_grid[:, :, :Num_y, :Num_x], latent_coords)
    else:
        recon = grid_to_pointcloud(recon_grid, Num_y, Num_x)

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon_phys = recon_phys[0].cpu().numpy()
    truth_phys = truth_phys[0].cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    coords_xy = coords_raw[0].cpu().numpy()[:, :2]

    triang = None
    body_polygon = None
    if hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if hasattr(dataset, "airfoil_body_indices") and dataset.airfoil_body_indices is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    metrics = {}
    field_names = dataset.field_names if len(dataset.field_names) == truth_phys.shape[1] else tuple(f"field_{i}" for i in range(truth_phys.shape[1]))
    for c, name in enumerate(field_names):
        sensor_coords = None
        field_sensor_mask = obs_field_ids_cpu == c
        if np.any(field_sensor_mask):
            sensor_coords = coords_xy[obs_indices_cpu[field_sensor_mask]]
        metrics[name] = _save_single_field_plot(
            true_f=truth_phys[:, c],
            pred_f=recon_phys[:, c],
            coords_xy=coords_xy,
            sensor_coords=sensor_coords,
            field_name=name,
            epoch=epoch,
            save_dir=save_dir,
            file_prefix=file_tag,
            triang=triang,
            body_polygon=body_polygon,
        )

    if save_metrics_json:
        prefix = file_tag or f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "N_steps": int(N_steps),
            "snr": float(snr),
            "n_corrector_steps": int(n_corrector_steps),
            "alpha_obs": float(alpha_obs),
            "method": "S3GM",
            "metrics": metrics,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return metrics


def _infer_structured_grid_from_coords(
    coords_xy: np.ndarray,
    decimals: int = 8,
    num_x: Optional[int] = None,
    num_y: Optional[int] = None,
):
    x = np.round(coords_xy[:, 0], decimals=decimals)
    y = np.round(coords_xy[:, 1], decimals=decimals)
    n_pts = coords_xy.shape[0]

    if num_x is not None and num_y is not None:
        nx = int(num_x)
        ny = int(num_y)
        if nx > 0 and ny > 0 and nx * ny == n_pts:
            sort_idx = np.lexsort((x, y))
            unique_x = np.unique(x)
            unique_y = np.unique(y)
            dx = float(np.mean(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
            dy = float(np.mean(np.diff(unique_y))) if len(unique_y) > 1 else 1.0
            return {
                "nx": nx,
                "ny": ny,
                "sort_idx": sort_idx,
                "x_unique": unique_x,
                "y_unique": unique_y,
                "dx": dx,
                "dy": dy,
            }

    unique_x = np.unique(x)
    unique_y = np.unique(y)
    nx = len(unique_x)
    ny = len(unique_y)
    if nx * ny != n_pts:
        raise ValueError(
            f"Coordinates do not form a complete structured 2D grid. Inferred nx={nx}, ny={ny}, N={n_pts}."
        )
    sort_idx = np.lexsort((x, y))
    dx = float(np.mean(np.diff(unique_x))) if nx > 1 else 1.0
    dy = float(np.mean(np.diff(unique_y))) if ny > 1 else 1.0
    return {
        "nx": nx,
        "ny": ny,
        "sort_idx": sort_idx,
        "x_unique": unique_x,
        "y_unique": unique_y,
        "dx": dx,
        "dy": dy,
    }


def _reshape_flat_field_to_grid(field_flat: np.ndarray, grid_info: dict) -> np.ndarray:
    vals = field_flat[grid_info["sort_idx"]]
    return vals.reshape(grid_info["ny"], grid_info["nx"])


def _gaussian_kernel(window_size: int = 11, sigma: float = 1.5, device: str = "cpu"):
    ax = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = torch.outer(g, g)
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, window_size, window_size)


def _ssim2d(u: np.ndarray, v: np.ndarray, data_range: Optional[float] = None, window_size: int = 11, sigma: float = 1.5) -> float:
    x = torch.from_numpy(u).float().unsqueeze(0).unsqueeze(0)
    y = torch.from_numpy(v).float().unsqueeze(0).unsqueeze(0)
    if data_range is None:
        data_range = float(max(u.max(), v.max()) - min(u.min(), v.min()))
    data_range = max(float(data_range), 1e-8)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    kernel = _gaussian_kernel(window_size=window_size, sigma=sigma)
    pad = window_size // 2
    mu_x = F.conv2d(x, kernel, padding=pad)
    mu_y = F.conv2d(y, kernel, padding=pad)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, kernel, padding=pad) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, padding=pad) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, padding=pad) - mu_xy
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
    return float(ssim_map.mean().item())


@torch.no_grad()
def visualize_reconstruction_latentfm(
    model,
    dataset,
    Num_x,
    Num_y,
    epoch,
    device,
    save_dir,
    cond_fields=(2,),
    n_obs=256,
    n_steps=8,
    ode_solver="euler",
    snapshot_index=0,
    file_tag=None,
    save_metrics_json=True,
    irregular=False,
    extra_metrics_list=None,
    save_analysis_npz=False,
    cfg=None,
):
    model.eval()
    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs, int):
        n_obs = [n_obs] * len(cond_fields)

    n_fields = dataset.num_fields
    n_pts = dataset.num_points
    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)
    valid_mask = sample.get("valid_sensor_mask")
    if valid_mask is not None:
        valid_mask = valid_mask.unsqueeze(0).to(device)

    if irregular:
        if model.ae.deformer is not None:
            coords_2d = model.ae.deform(coords)
        else:
            coords_2d = coords[..., :2]
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=truth,
            cond_fields=cond_fields,
            n_obs_min=n_obs,
            n_obs_max=n_obs,
            valid_mask=valid_mask,
            coords_2d=coords_2d,
            Ny=Num_y,
            Nx=Num_x,
        )
        B_, M_ = obs_indices.shape
        batch_idx = torch.arange(B_, device=device).unsqueeze(-1).expand(-1, M_)
        obs_coords_2d = coords_2d[batch_idx, obs_indices]
        obs_value_grid, obs_mask_grid = splat_obs_to_grid(
            obs_coords_2d,
            obs_values,
            obs_mask,
            obs_field_ids,
            n_fields,
            Num_y,
            Num_x,
        )
    else:
        obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
            coords_full=coords,
            fields_full=truth,
            cond_fields=cond_fields,
            n_obs_min=n_obs,
            n_obs_max=n_obs,
            valid_mask=valid_mask,
        )
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            Num_y,
            Num_x,
            Num_y,
            Num_x,
        )
        ix = (obs_indices % Num_x).float() / max(Num_x - 1, 1)
        iy = (obs_indices.div(Num_x, rounding_mode="floor")).float() / max(Num_y - 1, 1)
        obs_coords_2d = torch.stack([ix, iy], dim=-1)

    cond_inputs = {
        "obs_value_grid": obs_value_grid,
        "obs_mask_grid": obs_mask_grid,
        "obs_coords_2d": obs_coords_2d,
        "obs_values": obs_values,
        "obs_mask": obs_mask,
        "obs_field_ids": obs_field_ids,
    }
    recon_grid = model.sample(cond_inputs, n_steps=n_steps, ode_solver=ode_solver)

    if irregular:
        recon = gather_from_grid(recon_grid[:, :, :Num_y, :Num_x], coords_2d)
    else:
        recon = grid_to_pointcloud(recon_grid, Num_y, Num_x)

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon_phys = recon_phys[0].cpu().numpy()
    truth_phys = truth_phys[0].cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    coords_xy = coords_raw[0].cpu().numpy()[:, :2]

    triang = None
    body_polygon = None
    if hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)
    if hasattr(dataset, "airfoil_body_indices") and dataset.airfoil_body_indices is not None:
        body_polygon = coords_xy[dataset.airfoil_body_indices]

    metrics = {}
    field_names = dataset.field_names if len(dataset.field_names) == n_fields else tuple(f"field_{i}" for i in range(n_fields))
    for c, name in enumerate(field_names):
        sensor_coords = None
        field_sensor_mask = obs_field_ids_cpu == c
        if np.any(field_sensor_mask):
            sensor_coords = coords_xy[obs_indices_cpu[field_sensor_mask]]
        metrics[name] = _save_single_field_plot(
            true_f=truth_phys[:, c],
            pred_f=recon_phys[:, c],
            coords_xy=coords_xy,
            sensor_coords=sensor_coords,
            field_name=name,
            epoch=epoch,
            save_dir=save_dir,
            file_prefix=file_tag,
            triang=triang,
            body_polygon=body_polygon,
        )

    extra_metrics_list = extra_metrics_list or []
    extra_metrics = {}
    if len(extra_metrics_list) > 0 or save_analysis_npz:
        try:
            grid_info = _infer_structured_grid_from_coords(
                coords_xy,
                num_x=(cfg or {}).get("Num_x", None),
                num_y=(cfg or {}).get("Num_y", None),
            )
        except ValueError:
            grid_info = None
        if grid_info is not None:
            prefix = file_tag or f"epoch_{epoch:04d}"
            for c, name in enumerate(field_names):
                u = _reshape_flat_field_to_grid(truth_phys[:, c], grid_info)
                v = _reshape_flat_field_to_grid(recon_phys[:, c], grid_info)
                field_metrics = {}
                if "ssim" in extra_metrics_list:
                    field_metrics["ssim"] = _ssim2d(u, v, data_range=float(u.max() - u.min()))
                extra_metrics[name] = field_metrics
                if save_analysis_npz:
                    np.savez_compressed(
                        os.path.join(save_dir, f"{prefix}_field_{name}_analysis.npz"),
                        true_grid=u,
                        pred_grid=v,
                        abs_err_grid=np.abs(v - u),
                        x_unique=grid_info["x_unique"],
                        y_unique=grid_info["y_unique"],
                    )

    if save_metrics_json:
        prefix = file_tag or f"epoch_{epoch:04d}"
        metrics_path = os.path.join(save_dir, f"{prefix}_metrics.json")
        payload = {
            "epoch": int(epoch),
            "snapshot_index": int(snapshot_index),
            "cond_fields": [int(v) for v in cond_fields],
            "n_obs": [int(v) for v in n_obs],
            "n_steps": int(n_steps),
            "ode_solver": ode_solver,
            "method": "LatentFM",
            "metrics": metrics,
            "extra_metrics": extra_metrics,
        }
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    return metrics, extra_metrics


@torch.no_grad()
def sit_conditional_sample(
    net: nn.Module,
    transport: LinearVelocityTransport,
    shape: Sequence[int],
    obs_value_grid: Optional[torch.Tensor] = None,
    obs_mask_grid: Optional[torch.Tensor] = None,
    coords: Optional[torch.Tensor] = None,
    obs_value_nodes: Optional[torch.Tensor] = None,
    obs_mask_nodes: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    n_steps: int = 50,
    sampler_type: str = "euler",
) -> torch.Tensor:
    if isinstance(sampler_type, str):
        sampler_type = {"euler": "Euler", "heun": "Heun"}.get(sampler_type.lower(), sampler_type)
    sampler = Sampler(transport)
    sample_fn = sampler.sample_ode(sampling_method=sampler_type, num_steps=n_steps)
    init = torch.randn(tuple(shape), device=device)
    kwargs: dict[str, Any] = {}
    if coords is not None:
        kwargs["coords"] = coords
    if obs_value_grid is not None:
        kwargs["obs_value_grid"] = obs_value_grid
        kwargs["obs_mask_grid"] = obs_mask_grid
    if obs_value_nodes is not None:
        kwargs["obs_value_nodes"] = obs_value_nodes
        kwargs["obs_mask_nodes"] = obs_mask_nodes
    traj = sample_fn(init, net, **kwargs)
    return traj[-1]


@torch.no_grad()
def visualize_reconstruction_sit(
    net: nn.Module,
    transport: LinearVelocityTransport,
    dataset,
    num_x: int,
    num_y: int,
    h_pad: int,
    w_pad: int,
    epoch: int,
    device: torch.device,
    save_dir: Path,
    cond_fields: Sequence[int],
    n_obs: Sequence[int],
    n_steps: int,
    sampler_type: str,
    snapshot_index: int = 0,
    file_tag: Optional[str] = None,
    tokenizer: str = "patch",
    cond_mode: str = "image",
) -> dict[str, float]:
    if isinstance(cond_fields, int):
        cond_fields = [cond_fields]
    if isinstance(n_obs, int):
        n_obs = [n_obs] * len(cond_fields)

    n_fields = dataset.num_fields
    n_pts = dataset.num_points
    sample = dataset[snapshot_index]
    coords = sample["coords"].unsqueeze(0).to(device)
    coords_raw = sample["coords_raw"].unsqueeze(0).to(device)
    truth = sample["fields"].unsqueeze(0).to(device)

    obs_coords, obs_values, obs_mask, obs_indices, obs_field_ids = build_sparse_condition(
        coords_full=coords,
        fields_full=truth,
        cond_fields=cond_fields,
        n_obs_min=n_obs,
        n_obs_max=n_obs,
    )

    if tokenizer == "pointnet":
        obs_value_nodes, obs_mask_nodes = scatter_sensors_to_nodes(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            1,
            n_pts,
            n_fields,
            device,
            truth.dtype,
        )
        recon = sit_conditional_sample(
            net=net,
            transport=transport,
            shape=(1, n_pts, n_fields),
            coords=coords,
            obs_value_nodes=obs_value_nodes,
            obs_mask_nodes=obs_mask_nodes,
            device=device,
            n_steps=n_steps,
            sampler_type=sampler_type,
        )
    else:
        obs_value_grid, obs_mask_grid = build_obs_grid_mask(
            obs_values,
            obs_mask,
            obs_field_ids,
            obs_indices,
            n_fields,
            n_pts,
            num_y,
            num_x,
            h_pad,
            w_pad,
        )
        if cond_mode == "interp":
            obs_value_grid = nearest_fill_grid(obs_value_grid, obs_mask_grid)
        recon_grid = sit_conditional_sample(
            net=net,
            transport=transport,
            shape=(1, n_fields, h_pad, w_pad),
            obs_value_grid=obs_value_grid,
            obs_mask_grid=obs_mask_grid,
            device=device,
            n_steps=n_steps,
            sampler_type=sampler_type,
        )
        recon = grid_to_pointcloud(recon_grid, num_y, num_x)

    mean = dataset.mean.to(device)
    std = dataset.std.to(device)
    recon_phys = recon * std.view(1, 1, -1) + mean.view(1, 1, -1)
    truth_phys = truth * std.view(1, 1, -1) + mean.view(1, 1, -1)
    recon_phys = recon_phys[0].cpu().numpy()
    truth_phys = truth_phys[0].cpu().numpy()

    valid = obs_mask[0].bool()
    obs_indices_cpu = obs_indices[0, valid].cpu().numpy()
    obs_field_ids_cpu = obs_field_ids[0, valid].cpu().numpy()
    coords_xy = coords_raw[0].cpu().numpy()[:, :2]

    triang = None
    if hasattr(dataset, "grid_shape") and dataset.grid_shape is not None:
        triang = _build_structured_triangulation(coords_xy, dataset.grid_shape)

    metrics = {}
    field_names = dataset.field_names if len(dataset.field_names) == n_fields else tuple(f"field_{i}" for i in range(n_fields))
    for idx, name in enumerate(field_names):
        sensor_coords = None
        field_sensor_mask = obs_field_ids_cpu == idx
        if np.any(field_sensor_mask):
            sensor_coords = coords_xy[obs_indices_cpu[field_sensor_mask]]
        metrics[name] = _save_single_field_plot(
            true_f=truth_phys[:, idx],
            pred_f=recon_phys[:, idx],
            coords_xy=coords_xy,
            sensor_coords=sensor_coords,
            field_name=name,
            epoch=epoch,
            save_dir=str(save_dir),
            file_prefix=file_tag,
            triang=triang,
            body_polygon=None,
        )

    payload = {
        "epoch": int(epoch),
        "snapshot_index": int(snapshot_index),
        "cond_fields": [int(v) for v in cond_fields],
        "n_obs": [int(v) for v in n_obs],
        "n_steps": int(n_steps),
        "sampler_type": sampler_type,
        "method": "SiT",
        "metrics": metrics,
    }
    prefix = file_tag or f"epoch_{epoch:04d}"
    with open(save_dir / f"{prefix}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return metrics


class BaseBaselineAdapter(abc.ABC):
    name: str

    @abstractmethod
    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
        raise NotImplementedError

    @abstractmethod
    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        raise NotImplementedError

    @abstractmethod
    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None) -> dict[str, float]:
        raise NotImplementedError

    @contextlib.contextmanager
    def evaluation_weights(self, bundle: BaselineBundle) -> Iterator[None]:
        yield


class S3GMAdapter(BaseBaselineAdapter):
    name = "s3gm"

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        arch = stage_cfg["architecture"]
        diffusion = stage_cfg["diffusion"]
        training = stage_cfg["training"]
        num_y = int(cfg["shared"]["data"]["num_y"])
        num_x = int(cfg["shared"]["data"]["num_x"])
        h_pad, w_pad = compute_pad_size(num_y, num_x, 2 ** (len(arch["ch_mult"]) - 1))

        net = UNetVideoModel(
            in_channels=train_set.num_fields,
            model_channels=int(arch["nf"]),
            out_channels=train_set.num_fields,
            num_res_blocks=int(arch["num_res_blocks"]),
            attention_resolutions=tuple(arch["attn_resolutions"]),
            image_size=max(h_pad, w_pad),
            dropout=float(arch["dropout"]),
            channel_mult=tuple(arch["ch_mult"]),
            conv_resample=True,
            dims=2,
            num_heads=int(arch["num_heads"]),
            use_rpe_net=True,
            use_checkpoint=bool(arch.get("use_checkpoint", False)),
        ).to(device)

        def all_params_fn():
            return list(net.parameters())

        optimizer = torch.optim.Adam(
            all_params_fn(),
            lr=float(training["learning_rate"]),
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=float(training["weight_decay"]),
        )
        warmup_epochs = 100

        def lr_lambda(epoch_index: int) -> float:
            if epoch_index < warmup_epochs:
                return epoch_index / max(1, warmup_epochs)
            progress = (epoch_index - warmup_epochs) / max(1, int(training["epochs"]) - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        ema = ExponentialMovingAverage(all_params_fn(), decay=float(diffusion["ema_rate"]))
        sde = VESDE(
            config=argparse.Namespace(sde="vesde"),
            sigma_min=float(diffusion["sigma_min"]),
            sigma_max=float(diffusion["sigma_max"]),
            N=int(diffusion["num_scales"]),
        )
        return BaselineBundle(
            baseline_model=self.name,
            training_stage=1,
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            run_dir=run_dir,
            config=cfg,
            dataset_train=train_set,
            dataset_val=val_set,
            components={"sde": sde, "H_pad": h_pad, "W_pad": w_pad, "all_params_fn": all_params_fn},
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        bundle.model.load_state_dict(checkpoint["model"])
        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])
        if bundle.ema is not None and checkpoint.get("ema") is not None:
            bundle.ema.load_state_dict(checkpoint["ema"])
            bundle.ema.shadow_params = [param.to(bundle.device) for param in bundle.ema.shadow_params]

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
        return run_epoch_s3gm(bundle, loader, training, epoch)

    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        return {
            "baseline_model": bundle.baseline_model,
            "training_stage": bundle.training_stage,
            "model": bundle.model.state_dict(),
            "ema": bundle.ema.state_dict() if bundle.ema is not None else None,
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "method": "S3GM_VESDE",
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
        }

    @contextlib.contextmanager
    def evaluation_weights(self, bundle: BaselineBundle) -> Iterator[None]:
        if bundle.ema is None:
            yield
            return
        all_params = bundle.components["all_params_fn"]()
        bundle.ema.store(all_params)
        bundle.ema.copy_to(all_params)
        try:
            yield
        finally:
            bundle.ema.restore(all_params)

    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None) -> dict[str, float]:
        stage_cfg = resolve_stage_config(bundle.config)
        shared_cond = bundle.config["shared"]["conditioning"]
        sampling_cfg = stage_cfg["sampling"]
        n_steps = int(sampling_cfg["sampling_N"] if n_steps is None else n_steps)
        return visualize_reconstruction_s3gm(
            net=bundle.model,
            sde=bundle.components["sde"],
            dataset=dataset,
            Num_x=int(bundle.config["shared"]["data"]["num_x"]),
            Num_y=int(bundle.config["shared"]["data"]["num_y"]),
            H_pad=int(bundle.components["H_pad"]),
            W_pad=int(bundle.components["W_pad"]),
            epoch=epoch,
            device=bundle.device,
            save_dir=str(save_dir),
            cond_fields=shared_cond["vis_cond_fields"],
            n_obs=shared_cond["vis_n_obs_list"],
            N_steps=n_steps,
            snr=float(sampling_cfg["snr"]),
            n_corrector_steps=int(sampling_cfg["n_corrector_steps"]),
            alpha_obs=float(sampling_cfg["alpha_obs"]),
            snapshot_index=snapshot_index,
            file_tag=f"s3gm_N{n_steps}",
            save_metrics_json=True,
        )


class LatentFMAdapter(BaseBaselineAdapter):
    name = "latent_fm"

    def _stage1_checkpoint_path(self, cfg: dict) -> Path:
        stage2_cfg = cfg["latent_fm_params"]["stage2"]
        explicit = stage2_cfg.get("stage1_checkpoint")
        if explicit:
            checkpoint_path = ensure_absolute(explicit)
            if not checkpoint_path.exists():
                raise RuntimeError(
                    f"Latent FM stage 2 requires a valid stage-1 checkpoint, but the configured path does not exist: {checkpoint_path}"
                )
            return checkpoint_path

        stage1_cfg = copy.deepcopy(cfg)
        stage1_cfg["training_stage"] = 1
        save_root = ensure_absolute(cfg["shared"]["paths"]["save_root"])
        latest_stage1_run = find_latest_run_dir(save_root, stage1_cfg)
        if latest_stage1_run is None:
            raise RuntimeError(
                "Latent FM stage 2 requires stage 1 to be completed first. No matching unified stage-1 run directory was found."
            )
        checkpoint_path = latest_stage1_run / "best.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(
                f"Latent FM stage 2 requires stage 1 to be completed first. Expected checkpoint not found: {checkpoint_path}"
            )
        return checkpoint_path

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        stage = int(cfg["training_stage"])
        training = stage_cfg["training"]
        num_y = int(cfg["shared"]["data"]["num_y"])
        num_x = int(cfg["shared"]["data"]["num_x"])

        if stage == 1:
            arch = stage_cfg["architecture"]
            ae = ConvAE(
                n_fields=train_set.num_fields,
                base_ch=int(arch["base_ch"]),
                latent_ch=int(arch["latent_ch"]),
                n_levels=int(arch["n_levels"]),
                Num_y=num_y,
                Num_x=num_x,
                deform_coord_dim=0,
            ).to(device)
            optimizer = torch.optim.AdamW(ae.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]), eta_min=1e-6)
            return BaselineBundle(
                baseline_model=self.name,
                training_stage=1,
                model=ae,
                optimizer=optimizer,
                scheduler=scheduler,
                ema=None,
                device=device,
                run_dir=run_dir,
                config=cfg,
                dataset_train=train_set,
                dataset_val=val_set,
            )

        stage1_checkpoint = self._stage1_checkpoint_path(cfg)
        stage1_ckpt = safe_torch_load(stage1_checkpoint, map_location=device)
        stage1_arch = cfg["latent_fm_params"]["stage1"]["architecture"]
        stage2_arch = stage_cfg["architecture"]
        conditioning_cfg = stage_cfg["conditioning"]

        ae = ConvAE(
            n_fields=train_set.num_fields,
            base_ch=int(stage1_ckpt.get("ae_base_ch", stage1_arch["base_ch"])),
            latent_ch=int(stage1_ckpt.get("ae_latent_ch", stage1_arch["latent_ch"])),
            n_levels=int(stage1_ckpt.get("ae_n_levels", stage1_arch["n_levels"])),
            Num_y=num_y,
            Num_x=num_x,
            deform_coord_dim=int(stage1_ckpt.get("deform_coord_dim", 0)),
            deform_hidden=int(stage1_ckpt.get("deform_hidden", 128)),
            deform_depth=int(stage1_ckpt.get("deform_depth", 3)),
        ).to(device)
        ae.load_state_dict(stage1_ckpt["model"])
        ae.eval()
        ae.requires_grad_(False)

        velocity_net = LatentFMUNet(
            latent_ch=int(stage1_ckpt.get("ae_latent_ch", stage1_arch["latent_ch"])),
            n_fields=train_set.num_fields,
            base_ch=int(stage2_arch["base_ch"]),
            ch_mult=tuple(stage2_arch["ch_mult"]),
            num_res_blocks=int(stage2_arch["num_res_blocks"]),
            num_heads=int(stage2_arch["num_heads"]),
        ).to(device)

        pointnet_encoder = None
        cond_mode = str(conditioning_cfg["cond_mode"])
        if cond_mode == "pointnet":
            pointnet_encoder = PointNetSensorEncoder(
                n_fields=train_set.num_fields,
                coord_dim=2,
                hidden_mult=float(conditioning_cfg["pointnet_hidden_mult"]),
            ).to(device)

        model = LatentFlowMatching(
            ae=ae,
            velocity_net=velocity_net,
            Num_x=num_x,
            Num_y=num_y,
            cond_mode=cond_mode,
            pointnet_encoder=pointnet_encoder,
        ).to(device)

        trainable_params = list(velocity_net.parameters())
        if pointnet_encoder is not None:
            trainable_params += list(pointnet_encoder.parameters())
        optimizer = torch.optim.AdamW(trainable_params, lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training["epochs"]), eta_min=1e-6)
        ema = LatentEMA(velocity_net, decay=float(stage2_arch["ema_decay"]))
        return BaselineBundle(
            baseline_model=self.name,
            training_stage=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            run_dir=run_dir,
            config=cfg,
            dataset_train=train_set,
            dataset_val=val_set,
            components={
                "stage1_checkpoint": str(stage1_checkpoint),
                "velocity_net": velocity_net,
                "pointnet_encoder": pointnet_encoder,
                "cond_mode": cond_mode,
            },
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        if bundle.training_stage == 1:
            bundle.model.load_state_dict(checkpoint["model"])
        else:
            bundle.components["velocity_net"].load_state_dict(checkpoint["velocity_net"])
            pointnet_encoder = bundle.components["pointnet_encoder"]
            if pointnet_encoder is not None and checkpoint.get("pointnet_encoder") is not None:
                pointnet_encoder.load_state_dict(checkpoint["pointnet_encoder"])
            if bundle.ema is not None and checkpoint.get("ema") is not None:
                bundle.ema.load_state_dict(checkpoint["ema"])

        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
        if bundle.training_stage == 1:
            return run_epoch_ae(bundle, loader, training, epoch)
        return run_epoch_latentfm(bundle, loader, training, epoch)

    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        if bundle.training_stage == 1:
            stage1_arch = bundle.config["latent_fm_params"]["stage1"]["architecture"]
            return {
                "baseline_model": bundle.baseline_model,
                "training_stage": bundle.training_stage,
                "model": bundle.model.state_dict(),
                "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
                "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "mean": bundle.dataset_train.mean,
                "std": bundle.dataset_train.std,
                "field_names": bundle.dataset_train.field_names,
                "method": "LatentFM_AE",
                "ae_base_ch": int(stage1_arch["base_ch"]),
                "ae_latent_ch": int(stage1_arch["latent_ch"]),
                "ae_n_levels": int(stage1_arch["n_levels"]),
                "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
                "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
                "dataset": bundle.config["shared"]["data"]["dataset_name"],
            }

        stage1_arch = bundle.config["latent_fm_params"]["stage1"]["architecture"]
        stage2_arch = bundle.config["latent_fm_params"]["stage2"]["architecture"]
        return {
            "baseline_model": bundle.baseline_model,
            "training_stage": bundle.training_stage,
            "velocity_net": bundle.components["velocity_net"].state_dict(),
            "pointnet_encoder": bundle.components["pointnet_encoder"].state_dict() if bundle.components["pointnet_encoder"] is not None else None,
            "ema": bundle.ema.state_dict() if bundle.ema is not None else None,
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "ae_checkpoint": bundle.components["stage1_checkpoint"],
            "mean": bundle.dataset_train.mean,
            "std": bundle.dataset_train.std,
            "field_names": bundle.dataset_train.field_names,
            "method": "LatentFM",
            "ae_base_ch": int(stage1_arch["base_ch"]),
            "ae_latent_ch": int(stage1_arch["latent_ch"]),
            "ae_n_levels": int(stage1_arch["n_levels"]),
            "fm_base_ch": int(stage2_arch["base_ch"]),
            "fm_ch_mult": list(stage2_arch["ch_mult"]),
            "fm_num_res_blocks": int(stage2_arch["num_res_blocks"]),
            "fm_num_heads": int(stage2_arch["num_heads"]),
            "ema_decay": float(stage2_arch["ema_decay"]),
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
            "cond_mode": bundle.components["cond_mode"],
        }

    @contextlib.contextmanager
    def evaluation_weights(self, bundle: BaselineBundle) -> Iterator[None]:
        if bundle.training_stage != 2 or bundle.ema is None:
            yield
            return
        velocity_net = bundle.components["velocity_net"]
        bundle.ema.apply(velocity_net)
        try:
            yield
        finally:
            bundle.ema.restore(velocity_net)

    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None) -> dict[str, float]:
        shared_cond = bundle.config["shared"]["conditioning"]
        if bundle.training_stage == 1:
            return visualize_ae_reconstruction(
                ae=bundle.model,
                dataset=dataset,
                epoch=epoch,
                device=bundle.device,
                save_dir=str(save_dir),
                Num_y=int(bundle.config["shared"]["data"]["num_y"]),
                Num_x=int(bundle.config["shared"]["data"]["num_x"]),
                snapshot_index=snapshot_index,
                file_tag="latent_fm_ae",
                irregular=False,
            )

        stage2_cfg = resolve_stage_config(bundle.config)
        sampling_cfg = stage2_cfg["sampling"]
        n_steps = int(sampling_cfg["benchmark_n_steps"][0] if n_steps is None else n_steps)
        metrics, _ = visualize_reconstruction_latentfm(
            model=bundle.model,
            dataset=dataset,
            Num_x=int(bundle.config["shared"]["data"]["num_x"]),
            Num_y=int(bundle.config["shared"]["data"]["num_y"]),
            epoch=epoch,
            device=bundle.device,
            save_dir=str(save_dir),
            cond_fields=shared_cond["vis_cond_fields"],
            n_obs=shared_cond["vis_n_obs_list"],
            n_steps=n_steps,
            ode_solver=str(sampling_cfg["ode_solver"]),
            snapshot_index=snapshot_index,
            file_tag=f"latentfm_N{n_steps}",
            save_metrics_json=True,
            irregular=False,
        )
        return metrics


class SiTAdapter(BaseBaselineAdapter):
    name = "sit"

    def build_for_training(self, cfg: dict, device: torch.device, run_dir: Path, train_set, val_set) -> BaselineBundle:
        stage_cfg = resolve_stage_config(cfg)
        training = stage_cfg["training"]
        arch = stage_cfg["architecture"]
        transport_cfg = stage_cfg["transport"]
        conditioning_cfg = stage_cfg["conditioning"]
        num_y = int(cfg["shared"]["data"]["num_y"])
        num_x = int(cfg["shared"]["data"]["num_x"])
        h_pad, w_pad = compute_pad_size(num_y, num_x, int(arch["patch_size"]))
        n_fields = train_set.num_fields
        cond_channels = 2 * n_fields + 1

        net = SiTPhysics(
            input_size_h=h_pad,
            input_size_w=w_pad,
            patch_size=int(arch["patch_size"]),
            in_channels=n_fields,
            cond_channels=cond_channels,
            hidden_size=int(arch["hidden_size"]),
            depth=int(arch["depth"]),
            num_heads=int(arch["num_heads"]),
            mlp_ratio=float(arch["mlp_ratio"]),
            tokenizer=str(arch["tokenizer"]),
            coord_dim=int(arch["coord_dim"]),
            fourier_num_freqs=int(arch["fourier_num_freqs"]),
            fourier_scale=float(arch["fourier_scale"]),
        ).to(device)

        def all_params_fn():
            return list(net.parameters())

        def build_param_groups(params: Iterable[torch.nn.Parameter]) -> list[dict[str, Any]]:
            decay, no_decay = [], []
            for param in params:
                if not param.requires_grad:
                    continue
                if param.ndim < 2:
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {"params": decay, "weight_decay": float(training["weight_decay"])},
                {"params": no_decay, "weight_decay": 0.0},
            ]

        optimizer = torch.optim.AdamW(
            build_param_groups(all_params_fn()),
            lr=float(training["learning_rate"]),
            betas=(0.9, 0.95),
            eps=float(training["adam_eps"]),
        )
        warmup_epochs = 200

        def lr_lambda(epoch_index: int) -> float:
            if epoch_index < warmup_epochs:
                return epoch_index / max(1, warmup_epochs)
            progress = (epoch_index - warmup_epochs) / max(1, int(training["epochs"]) - warmup_epochs)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        ema = SiTEMA(all_params_fn(), decay=float(arch["ema_rate"]))
        transport = create_transport(
            path_type=str(transport_cfg["path_type"]),
            prediction=str(transport_cfg["prediction"]),
            loss_weight=transport_cfg.get("loss_weight"),
        )

        return BaselineBundle(
            baseline_model=self.name,
            training_stage=1,
            model=net,
            optimizer=optimizer,
            scheduler=scheduler,
            ema=ema,
            device=device,
            run_dir=run_dir,
            config=cfg,
            dataset_train=train_set,
            dataset_val=val_set,
            components={
                "transport": transport,
                "H_pad": h_pad,
                "W_pad": w_pad,
                "cond_channels": cond_channels,
                "n_fields": n_fields,
                "cond_mode": str(conditioning_cfg["cond_mode"]),
                "tokenizer": str(arch["tokenizer"]),
                "huber_beta": float(training["huber_beta"]),
                "spike_state": {"ema_loss": None, "ema_grad": None, "skipped": 0},
                "all_params_fn": all_params_fn,
            },
        )

    def load_checkpoint(self, bundle: BaselineBundle, checkpoint: dict) -> None:
        bundle.model.load_state_dict(checkpoint["model"], strict=False)
        if bundle.optimizer is not None and checkpoint.get("optimizer") is not None:
            bundle.optimizer.load_state_dict(checkpoint["optimizer"])
        if bundle.scheduler is not None and checkpoint.get("scheduler") is not None:
            bundle.scheduler.load_state_dict(checkpoint["scheduler"])
        if bundle.ema is not None and checkpoint.get("ema") is not None:
            bundle.ema.load_state_dict(checkpoint["ema"])
        if checkpoint.get("spike_state") is not None:
            bundle.components["spike_state"] = dict(checkpoint["spike_state"])

    def run_epoch(self, bundle: BaselineBundle, loader: DataLoader, training: bool, epoch: int) -> float:
        return run_epoch_sit(bundle, loader, training, epoch)

    def build_checkpoint(self, bundle: BaselineBundle, epoch: int, train_loss: float, val_loss: float) -> dict:
        stage_cfg = resolve_stage_config(bundle.config)
        arch = stage_cfg["architecture"]
        return {
            "baseline_model": bundle.baseline_model,
            "training_stage": bundle.training_stage,
            "model": bundle.model.state_dict(),
            "ema": bundle.ema.state_dict() if bundle.ema is not None else None,
            "optimizer": bundle.optimizer.state_dict() if bundle.optimizer is not None else None,
            "scheduler": bundle.scheduler.state_dict() if bundle.scheduler is not None else None,
            "spike_state": dict(bundle.components["spike_state"]),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "method": "SiT_FlowMatching",
            "Num_x": int(bundle.config["shared"]["data"]["num_x"]),
            "Num_y": int(bundle.config["shared"]["data"]["num_y"]),
            "H_pad": int(bundle.components["H_pad"]),
            "W_pad": int(bundle.components["W_pad"]),
            "n_fields": int(bundle.components["n_fields"]),
            "cond_channels": int(bundle.components["cond_channels"]),
            "patch_size": int(arch["patch_size"]),
            "hidden_size": int(arch["hidden_size"]),
            "depth": int(arch["depth"]),
            "sit_num_heads": int(arch["num_heads"]),
            "mlp_ratio": float(arch["mlp_ratio"]),
            "tokenizer": str(arch["tokenizer"]),
            "coord_dim": int(arch["coord_dim"]),
            "fourier_num_freqs": int(arch["fourier_num_freqs"]),
            "fourier_scale": float(arch["fourier_scale"]),
            "cond_mode": str(bundle.components["cond_mode"]),
        }

    @contextlib.contextmanager
    def evaluation_weights(self, bundle: BaselineBundle) -> Iterator[None]:
        if bundle.ema is None:
            yield
            return
        all_params = bundle.components["all_params_fn"]()
        bundle.ema.store(all_params)
        bundle.ema.copy_to(all_params)
        try:
            yield
        finally:
            bundle.ema.restore(all_params)

    def visualize(self, bundle: BaselineBundle, dataset, save_dir: Path, epoch: int, snapshot_index: int, n_steps: Optional[int] = None) -> dict[str, float]:
        stage_cfg = resolve_stage_config(bundle.config)
        shared_cond = bundle.config["shared"]["conditioning"]
        sampling_cfg = stage_cfg["sampling"]
        n_steps = int(sampling_cfg["sampling_N"] if n_steps is None else n_steps)
        return visualize_reconstruction_sit(
            net=bundle.model,
            transport=bundle.components["transport"],
            dataset=dataset,
            num_x=int(bundle.config["shared"]["data"]["num_x"]),
            num_y=int(bundle.config["shared"]["data"]["num_y"]),
            h_pad=int(bundle.components["H_pad"]),
            w_pad=int(bundle.components["W_pad"]),
            epoch=epoch,
            device=bundle.device,
            save_dir=save_dir,
            cond_fields=shared_cond["vis_cond_fields"],
            n_obs=shared_cond["vis_n_obs_list"],
            n_steps=n_steps,
            sampler_type=str(sampling_cfg["ode_solver"]),
            snapshot_index=snapshot_index,
            file_tag=f"sit_N{n_steps}",
            tokenizer=bundle.components["tokenizer"],
            cond_mode=bundle.components["cond_mode"],
        )


def get_baseline_adapter(baseline_model: str) -> BaseBaselineAdapter:
    baseline_model = str(baseline_model).strip().lower()
    registry: dict[str, BaseBaselineAdapter] = {
        "s3gm": S3GMAdapter(),
        "latent_fm": LatentFMAdapter(),
        "sit": SiTAdapter(),
    }
    if baseline_model not in registry:
        raise ValueError(
            f"Unsupported baseline_model={baseline_model!r}. "
            f"Available: {sorted(registry)}"
        )
    return registry[baseline_model]
