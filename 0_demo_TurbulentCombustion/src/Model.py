

import math, os, torch
from pathlib import Path
from typing import Dict, Optional, Tuple, Sequence

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO as NeuralOpFNO  # pip install neuraloperator

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

# -----------------------------------------------------

