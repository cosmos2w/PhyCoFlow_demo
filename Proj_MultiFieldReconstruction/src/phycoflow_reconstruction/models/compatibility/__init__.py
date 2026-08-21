"""Version-locked checkpoint compatibility adapters.

Compatibility models are intentionally absent from the normal model registry.
They exist only to reproduce named historical runs for evaluation and for the
first post-training migration checks.
"""

from .legacy_tc_demo50 import (
    DEMO50_DATASET_FIELDS,
    DEMO51_CHECKPOINT_FIELDS,
    Demo50CompatibilityManifest,
    LegacyDemo50Model,
    LegacyTCCompatibilityManifest,
    LegacyTCPointCloudModel,
    load_legacy_demo50,
    load_legacy_tc_pointcloud,
)

__all__ = [
    "DEMO50_DATASET_FIELDS",
    "DEMO51_CHECKPOINT_FIELDS",
    "Demo50CompatibilityManifest",
    "LegacyDemo50Model",
    "LegacyTCCompatibilityManifest",
    "LegacyTCPointCloudModel",
    "load_legacy_demo50",
    "load_legacy_tc_pointcloud",
]
