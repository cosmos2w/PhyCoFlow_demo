#!/usr/bin/env python3
"""Parameter-equivalent MHA-mask versus explicit-SDPA benchmark for Stage 7.

The explicit path reuses every projection and output parameter from each
``nn.MultiheadAttention`` module. Only the execution wrapper changes. This
keeps the scientific checkpoint and architecture fixed while measuring kernel
overhead separately.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import sys
import time
import types
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_pointcloud_cq import make_inputs, sync
from evaluate_pointcloud_fixed_manifest import build_gl_rbf_ffm
from model_ema import ModelEMA
from train_pointcloud_ffm import checkpoint_model_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--query-size", type=int, default=4096)
    parser.add_argument("--query-microbatch-size", type=int, default=2048)
    parser.add_argument("--n-obs", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--parity-batch-size", type=int, default=2)
    parser.add_argument("--parity-query-size", type=int, default=512)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def explicit_sdpa_forward(
    module: torch.nn.MultiheadAttention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    key_padding_mask: torch.Tensor | None = None,
    need_weights: bool = True,
    attn_mask: torch.Tensor | None = None,
    average_attn_weights: bool = True,
    is_causal: bool = False,
) -> tuple[torch.Tensor, None]:
    """MHA-compatible forward using direct scaled_dot_product_attention."""
    del average_attn_weights
    if need_weights:
        raise ValueError("Explicit SDPA benchmark requires need_weights=False.")
    if not module.batch_first or not module._qkv_same_embed_dim:
        raise ValueError("Benchmark supports batch-first, same-width MHA only.")
    if module.bias_k is not None or module.bias_v is not None or module.add_zero_attn:
        raise ValueError("Benchmark does not support bias_k/bias_v/add_zero_attn.")

    embed_dim = module.embed_dim
    q_bias = k_bias = v_bias = None
    if module.in_proj_bias is not None:
        q_bias, k_bias, v_bias = module.in_proj_bias.split(embed_dim)
    q_proj = F.linear(query, module.in_proj_weight[:embed_dim], q_bias)
    k_proj = F.linear(key, module.in_proj_weight[embed_dim : 2 * embed_dim], k_bias)
    v_proj = F.linear(value, module.in_proj_weight[2 * embed_dim :], v_bias)

    batch, q_len, _ = q_proj.shape
    k_len = k_proj.shape[1]
    heads = module.num_heads
    head_dim = embed_dim // heads
    q_proj = q_proj.view(batch, q_len, heads, head_dim).transpose(1, 2)
    k_proj = k_proj.view(batch, k_len, heads, head_dim).transpose(1, 2)
    v_proj = v_proj.view(batch, k_len, heads, head_dim).transpose(1, 2)

    merged_mask: torch.Tensor | None = None
    if key_padding_mask is not None:
        if key_padding_mask.dtype == torch.bool:
            merged_mask = torch.zeros(
                (batch, 1, 1, k_len), dtype=q_proj.dtype, device=q_proj.device
            ).masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        else:
            merged_mask = key_padding_mask[:, None, None, :].to(dtype=q_proj.dtype)
    if attn_mask is not None:
        local_mask = attn_mask
        if local_mask.dtype == torch.bool:
            local_mask = torch.zeros_like(local_mask, dtype=q_proj.dtype).masked_fill(
                local_mask, float("-inf")
            )
        local_mask = local_mask.to(device=q_proj.device, dtype=q_proj.dtype)
        if local_mask.ndim == 2:
            local_mask = local_mask[None, None]
        elif local_mask.ndim == 3:
            local_mask = local_mask.view(batch, heads, q_len, k_len)
        merged_mask = local_mask if merged_mask is None else merged_mask + local_mask

    output = F.scaled_dot_product_attention(
        q_proj,
        k_proj,
        v_proj,
        attn_mask=merged_mask,
        dropout_p=float(module.dropout) if module.training else 0.0,
        is_causal=is_causal,
        scale=1.0 / math.sqrt(head_dim),
    )
    output = output.transpose(1, 2).contiguous().view(batch, q_len, embed_dim)
    return F.linear(output, module.out_proj.weight, module.out_proj.bias), None


def patch_explicit_sdpa(model: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.MultiheadAttention):
            module.forward = types.MethodType(explicit_sdpa_forward, module)
            count += 1
    return count


def build_model(
    config: dict[str, Any], state: dict[str, torch.Tensor], device: torch.device, *, sdpa: bool
) -> tuple[torch.nn.Module, int]:
    model = build_gl_rbf_ffm(config, n_fields=5, device=device)
    model.load_state_dict(state, strict=True)
    count = patch_explicit_sdpa(model) if sdpa else sum(
        isinstance(module, torch.nn.MultiheadAttention) for module in model.modules()
    )
    return model, count


def model_output(model: torch.nn.Module, values: dict[str, torch.Tensor]) -> torch.Tensor:
    return model.model(
        values["t"], values["x_t"], values["coords"], values["obs_coords"],
        values["obs_values"], values["obs_mask"], values["obs_field_ids"],
    )


def parity_check(
    config: dict[str, Any], state: dict[str, torch.Tensor], args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    values = make_inputs(
        args.parity_query_size, args.n_obs, 5, device, seed=7781,
        batch_size=args.parity_batch_size,
    )
    # Exercise a real key-padding mask, not merely an all-valid mask.
    values["obs_mask"][:, -max(1, args.n_obs // 4) :] = 0
    mha, mha_count = build_model(config, state, device, sdpa=False)
    sdpa, sdpa_count = build_model(config, state, device, sdpa=True)
    mha.eval()
    sdpa.eval()
    with torch.no_grad():
        reference = model_output(mha, values)
        candidate = model_output(sdpa, values)
    output_abs = (candidate - reference).abs()

    mha.train()
    sdpa.train()
    mha.zero_grad(set_to_none=True)
    sdpa.zero_grad(set_to_none=True)
    reference_loss = model_output(mha, values).square().mean()
    candidate_loss = model_output(sdpa, values).square().mean()
    reference_loss.backward()
    candidate_loss.backward()
    max_grad_abs = 0.0
    max_grad_rel = 0.0
    worst_grad = ""
    gradients_allclose = True
    sdpa_parameters = dict(sdpa.named_parameters())
    for name, parameter in mha.named_parameters():
        other = sdpa_parameters[name]
        if parameter.grad is None and other.grad is None:
            continue
        difference = (parameter.grad - other.grad).abs()
        local_abs = float(difference.max())
        denominator = max(float(parameter.grad.abs().max()), 1.0e-12)
        local_rel = local_abs / denominator
        if local_abs > max_grad_abs:
            max_grad_abs = local_abs
            worst_grad = name
        max_grad_rel = max(max_grad_rel, local_rel)
        gradients_allclose = gradients_allclose and torch.allclose(
            parameter.grad, other.grad, atol=5.0e-5, rtol=5.0e-3
        )

    result = {
        "mha_module_count": mha_count,
        "sdpa_module_count": sdpa_count,
        "parameter_count_mha": sum(p.numel() for p in mha.parameters()),
        "parameter_count_sdpa": sum(p.numel() for p in sdpa.parameters()),
        "output_max_abs": float(output_abs.max()),
        "output_mean_abs": float(output_abs.mean()),
        "output_max_relative_to_reference_peak": float(output_abs.max())
        / max(float(reference.abs().max()), 1.0e-12),
        "loss_mha": float(reference_loss.detach()),
        "loss_sdpa": float(candidate_loss.detach()),
        "loss_abs_delta": abs(float(reference_loss.detach()) - float(candidate_loss.detach())),
        "gradient_max_abs": max_grad_abs,
        "gradient_max_relative": max_grad_rel,
        "gradients_allclose_atol5e-5_rtol5e-3": gradients_allclose,
        "gradient_worst_parameter": worst_grad,
    }
    result["parameters_identical"] = result["parameter_count_mha"] == result["parameter_count_sdpa"]

    optimizer_reference = copy.deepcopy(mha)
    optimizer_fused = copy.deepcopy(mha)
    generator = torch.Generator(device=device).manual_seed(77331)
    for reference_parameter, fused_parameter in zip(
        optimizer_reference.parameters(), optimizer_fused.parameters()
    ):
        gradient = torch.randn(
            reference_parameter.shape,
            dtype=reference_parameter.dtype,
            device=reference_parameter.device,
            generator=generator,
        ) * 1.0e-3
        reference_parameter.grad = gradient
        fused_parameter.grad = gradient.clone()
    unfused_optimizer = torch.optim.AdamW(
        optimizer_reference.parameters(), lr=1.0e-4, weight_decay=1.0e-6, fused=False
    )
    fused_optimizer = torch.optim.AdamW(
        optimizer_fused.parameters(), lr=1.0e-4, weight_decay=1.0e-6, fused=True
    )
    unfused_optimizer.step()
    fused_optimizer.step()
    optimizer_max_abs = max(
        float((reference_parameter - fused_parameter).abs().max())
        for reference_parameter, fused_parameter in zip(
            optimizer_reference.parameters(), optimizer_fused.parameters()
        )
    )
    result["fused_adamw_one_step_max_parameter_abs"] = optimizer_max_abs
    result["fused_adamw_one_step_passes_atol1e-7_rtol1e-6"] = all(
        torch.allclose(reference_parameter, fused_parameter, atol=1.0e-7, rtol=1.0e-6)
        for reference_parameter, fused_parameter in zip(
            optimizer_reference.parameters(), optimizer_fused.parameters()
        )
    )
    result["passes"] = bool(
        result["parameters_identical"]
        and torch.allclose(candidate, reference, atol=5.0e-5, rtol=5.0e-4)
        and result["gradients_allclose_atol5e-5_rtol5e-3"]
    )
    del mha, sdpa, values, reference, candidate, optimizer_reference, optimizer_fused
    del unfused_optimizer, fused_optimizer
    torch.cuda.empty_cache()
    return result


def benchmark_variant(
    label: str,
    config: dict[str, Any],
    state: dict[str, torch.Tensor],
    values: dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
    *,
    sdpa: bool,
    fused_adamw: bool,
) -> dict[str, Any]:
    model, module_count = build_model(config, state, device, sdpa=sdpa)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.0e-4, weight_decay=1.0e-6, fused=fused_adamw
    )
    ema = ModelEMA(model, decay=float(config.get("model_ema_decay", 0.999)))

    def one_step(measure: bool) -> dict[str, float]:
        optimizer.zero_grad(set_to_none=True)
        if measure:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        torch.manual_seed(99173)
        torch.cuda.manual_seed_all(99173)
        sync(device)
        start = time.perf_counter()
        (_, metrics) = model.training_loss_microbatched(
            x1=values["x_t"], coords=values["coords"],
            obs_coords=values["obs_coords"], obs_values=values["obs_values"],
            obs_mask=values["obs_mask"], obs_field_ids=values["obs_field_ids"],
            obs_indices=None, query_microbatch_size=args.query_microbatch_size,
            backward=True, reuse_condition_context=True, synchronize_timing=True,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        sync(device)
        optimizer_start = time.perf_counter()
        optimizer.step()
        sync(device)
        optimizer_ms = (time.perf_counter() - optimizer_start) * 1000.0
        ema_start = time.perf_counter()
        ema.update(model)
        sync(device)
        ema_ms = (time.perf_counter() - ema_start) * 1000.0
        return {
            "full_step_ms": (time.perf_counter() - start) * 1000.0,
            "condition_context_ms": float(metrics["condition_context_ms"]),
            "query_forward_ms": float(metrics["query_chunk_forward_ms"]),
            "query_backward_ms": float(metrics["query_chunk_backward_ms"]),
            "optimizer_ms": optimizer_ms,
            "ema_ms": ema_ms,
            "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024.0**2,
            "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024.0**2,
        }

    for _ in range(args.warmup):
        one_step(False)
    rows = [one_step(True) for _ in range(args.iterations)]
    result = {
        "label": label,
        "attention": "explicit_sdpa" if sdpa else "nn_mha_mask",
        "adamw": "fused" if fused_adamw else "unfused",
        "mha_module_count": module_count,
        "samples": rows,
    }
    for key in rows[0]:
        values_for_key = [row[key] for row in rows]
        result[key] = statistics.fmean(values_for_key)
        result[f"{key}_median"] = statistics.median(values_for_key)
    del optimizer, ema, model
    torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This formal kernel benchmark requires CUDA.")
    config = yaml.safe_load(args.config.read_text()) or {}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_template = build_gl_rbf_ffm(config, n_fields=5, device=torch.device("cpu"))
    state = checkpoint_model_state(checkpoint, model=state_template)
    del state_template
    parity = parity_check(config, state, args, device)
    if not parity["passes"]:
        raise RuntimeError(f"Explicit SDPA parity failed: {parity}")

    values = make_inputs(
        args.query_size, args.n_obs, 5, device, seed=7782, batch_size=args.batch_size
    )
    variants = [
        benchmark_variant(
            "MHA-mask + unfused AdamW", config, state, values, args, device,
            sdpa=False, fused_adamw=False,
        ),
        benchmark_variant(
            "Explicit SDPA + unfused AdamW", config, state, values, args, device,
            sdpa=True, fused_adamw=False,
        ),
        benchmark_variant(
            "MHA-mask + fused AdamW", config, state, values, args, device,
            sdpa=False, fused_adamw=True,
        ),
    ]
    try:
        variants.append(
            benchmark_variant(
                "Explicit SDPA + fused AdamW", config, state, values, args, device,
                sdpa=True, fused_adamw=True,
            )
        )
        fused_status = "ok"
    except RuntimeError as exc:
        fused_status = f"unsupported: {exc}"

    baseline = variants[0]
    for row in variants:
        row["step_speedup_vs_mha_unfused"] = baseline["full_step_ms"] / row["full_step_ms"]
        row["memory_ratio_vs_mha_unfused"] = row["peak_allocated_mb"] / baseline["peak_allocated_mb"]
    output = {
        "protocol": {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", 0)),
            "checkpoint_weights": "ema" if "model_ema" in checkpoint else "live",
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "batch_size": args.batch_size,
            "query_size": args.query_size,
            "query_microbatch_size": args.query_microbatch_size,
            "n_obs": args.n_obs,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "note": "nn.MultiheadAttention uses need_weights=False; modern PyTorch may internally dispatch it to SDPA. The explicit path measures wrapper/mask overhead while preserving projections.",
        },
        "parity": parity,
        "fused_adamw_status": fused_status,
        "variants": variants,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
