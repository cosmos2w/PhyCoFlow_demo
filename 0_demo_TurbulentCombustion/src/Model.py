"""Historical model import boundary backed by the portable GL_rbf_CQ core.

GL-RBF/CQ and PointCloudFFM symbols are dependency-light direct re-exports.
Older MLP, Perceiver, and FNO baselines are loaded lazily only when requested.
"""

from __future__ import annotations

from phycoflow_pointcloud.models.portable_core import (
    CompactLatentReadout,
    ConditionalPointHybridLocalGlobalRBF,
    ConditionalPointHybridLocalGlobalRBFCQ,
    CrossAttentionBlock,
    FeedForward,
    FourierPositionalEncoding,
    PointCloudFFM,
    SelfAttentionBlock,
    batched_gather_2d,
    batched_gather_3d,
    make_mlp,
)


_LEGACY_EXPORTS = {
    "FIELD_NAMES",
    "ConditionalPointFFM",
    "ConditionalPointMLPRBF",
    "ConditionalPointPerceiver",
    "FNO",
    "FNOFFM",
}


def __getattr__(name: str):
    if name not in _LEGACY_EXPORTS:
        raise AttributeError(f"module 'Model' has no attribute {name!r}")
    from _legacy_model_full import __dict__ as legacy

    value = legacy[name]
    globals()[name] = value
    return value


__all__ = [
    "CompactLatentReadout",
    "ConditionalPointFFM",
    "ConditionalPointHybridLocalGlobalRBF",
    "ConditionalPointHybridLocalGlobalRBFCQ",
    "ConditionalPointMLPRBF",
    "ConditionalPointPerceiver",
    "CrossAttentionBlock",
    "FIELD_NAMES",
    "FNO",
    "FNOFFM",
    "FeedForward",
    "FourierPositionalEncoding",
    "PointCloudFFM",
    "SelfAttentionBlock",
    "batched_gather_2d",
    "batched_gather_3d",
    "make_mlp",
]
