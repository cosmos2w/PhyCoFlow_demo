from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from Model import ConditionalPointHybridLocalGlobalRBF, PointCloudFFM


class _RecordingRFFPrior(torch.nn.Module):
    def __init__(self, coord_dim: int = 3, n_features: int = 11):
        super().__init__()
        omega = torch.linspace(-2.0, 2.0, coord_dim * n_features).reshape(coord_dim, n_features)
        phase = torch.linspace(0.0, 2.0 * math.pi, n_features)
        self.register_buffer("omega", omega)
        self.register_buffer("phase", phase)
        self.calls: list[int] = []

    def forward(self, coords: torch.Tensor, n_channels: int) -> torch.Tensor:
        self.calls.append(int(coords.shape[1]))
        phi = math.sqrt(2.0 / self.omega.shape[1]) * torch.cos(coords @ self.omega + self.phase)
        weights = torch.randn(
            coords.shape[0], n_channels, self.omega.shape[1],
            device=coords.device, dtype=coords.dtype,
        )
        return torch.einsum("bnf,bcf->bnc", phi, weights)


def _model() -> PointCloudFFM:
    backbone = ConditionalPointHybridLocalGlobalRBF(
        n_fields=2,
        coord_dim=3,
        hidden_dim=16,
        cond_dim=8,
        field_embed_dim=4,
        latent_dim=16,
        num_latents=8,
        num_heads=4,
        num_latent_blocks=1,
        ff_mult=2,
        gather_mode="topk_rbf_glres",
        gather_topk=3,
        gather_query_chunk_size=4,
        learnable_rbf_sigma=True,
        neighbor_backend="torch",
        use_fourier_pe=True,
        fourier_pe_num_bands=2,
        fourier_pe_max_freq=4.0,
        enhanced_backbone=True,
        sensor_coord_encoding="fourier",
        latent_sensor_reinject=True,
        query_latent_readout=True,
        query_readout_type="coord",
        query_readout_scale_init=1e-2,
        enhanced_head_norm=True,
        glres_scale_init=1e-2,
    )
    return PointCloudFFM(backbone, _RecordingRFFPrior())


def _inputs() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(313)
    coords = torch.rand(2, 31, 3, generator=generator)
    x1 = torch.randn(2, 31, 2, generator=generator)
    obs_indices = torch.tensor([[0, 3, 7, 10, 17], [1, 4, 9, 15, 29]])
    obs_coords = torch.stack([coords[i, obs_indices[i]] for i in range(2)])
    return {
        "x1": x1,
        "coords": coords,
        "obs_coords": obs_coords,
        "obs_values": torch.randn(2, 5, 1, generator=generator),
        "obs_mask": torch.ones(2, 5),
        "obs_field_ids": torch.tensor([[0, 1, 0, 1, 0], [1, 0, 1, 0, 1]]),
        "obs_indices": obs_indices,
    }


def _gradient_snapshot(model: PointCloudFFM):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def test_query_microbatch_matches_loss_all_gradients_and_adam_update():
    torch.manual_seed(17)
    monolithic = _model()
    microbatched = copy.deepcopy(monolithic)
    values = _inputs()
    optimizer_full = torch.optim.Adam(monolithic.parameters(), lr=3e-4)
    optimizer_micro = torch.optim.Adam(microbatched.parameters(), lr=3e-4)

    optimizer_full.zero_grad(set_to_none=True)
    torch.manual_seed(991)
    loss_full, _ = monolithic.training_loss(**values)
    loss_full.backward()
    gradients_full = _gradient_snapshot(monolithic)
    torch.nn.utils.clip_grad_norm_(monolithic.parameters(), max_norm=1.0)
    optimizer_full.step()

    optimizer_micro.zero_grad(set_to_none=True)
    torch.manual_seed(991)
    loss_micro, metrics = microbatched.training_loss_microbatched(
        **values,
        query_microbatch_size=7,
        backward=True,
        reuse_condition_context=True,
    )
    gradients_micro = _gradient_snapshot(microbatched)
    torch.nn.utils.clip_grad_norm_(microbatched.parameters(), max_norm=1.0)
    optimizer_micro.step()

    torch.testing.assert_close(loss_micro, loss_full.detach(), rtol=2e-6, atol=2e-7)
    assert metrics["query_microbatches"] == 5.0
    assert monolithic.prior.calls == [31]
    assert microbatched.prior.calls == [31]
    assert gradients_micro["model.log_rbf_sigma"] is not None
    for name in gradients_full:
        full = gradients_full[name]
        micro = gradients_micro[name]
        assert (full is None) == (micro is None), name
        if full is not None:
            torch.testing.assert_close(micro, full, rtol=8e-5, atol=2e-7, msg=name)
    for (name_full, parameter_full), (name_micro, parameter_micro) in zip(
        monolithic.named_parameters(), microbatched.named_parameters()
    ):
        assert name_full == name_micro
        torch.testing.assert_close(
            # Adam amplifies a few ~1e-9 FP32 cancellation differences in
            # near-zero attention-bias gradients; the largest update delta is
            # still only 5.8e-6 while all raw gradients pass above.
            parameter_micro, parameter_full, rtol=2e-5, atol=6e-6, msg=name_full,
        )


def test_query_microbatch_validation_loss_matches_monolithic_without_gradients():
    torch.manual_seed(23)
    monolithic = _model().eval()
    microbatched = copy.deepcopy(monolithic).eval()
    values = _inputs()
    with torch.no_grad():
        torch.manual_seed(711)
        full, _ = monolithic.training_loss(**values)
        torch.manual_seed(711)
        micro, _ = microbatched.training_loss_microbatched(
            **values,
            query_microbatch_size=7,
            backward=False,
            reuse_condition_context=True,
        )
    torch.testing.assert_close(micro, full, rtol=2e-6, atol=2e-7)
    assert monolithic.prior.calls == [31]
    assert microbatched.prior.calls == [31]
