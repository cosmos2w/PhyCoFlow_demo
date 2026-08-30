"""Explicit cache-only representative-snapshot selection for figure panels."""
from __future__ import annotations


def resolve_panel_snapshot(v2: dict, panel: str, fallback: int) -> tuple[int, dict]:
    """Resolve a panel override or the shared qualitative snapshot.

    The returned provenance makes the requested and resolved values explicit.
    This helper performs no cache generation and has no model dependency.
    """
    spec = v2.get("representative_snapshots", {})
    mode = str(spec.get("default_mode", "shared"))
    if mode != "shared":
        raise ValueError("representative_snapshots.default_mode must be 'shared'")
    shared = spec.get("shared_snapshot_index")
    shared = int(fallback if shared is None else shared)
    override = spec.get("overrides", {}).get(str(panel))
    resolved = int(shared if override is None else override)
    if resolved < 0:
        raise ValueError(f"Panel {panel} snapshot must be a non-negative cache index")
    return resolved, {
        "mode": mode,
        "panel": str(panel),
        "shared_snapshot_index": shared,
        "panel_override": None if override is None else int(override),
        "resolved_snapshot_index": resolved,
        "selection_rule": spec.get("selection_rule"),
        "model_inference_performed": False,
    }
