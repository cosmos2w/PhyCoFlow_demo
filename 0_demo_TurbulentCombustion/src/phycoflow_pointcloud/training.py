"""Compatibility import for the canonical ``src/train_pointcloud_ffm.py`` CLI."""

from __future__ import annotations


def main(argv: list[str] | None = None) -> None:
    """Delegate to the single public and historical training implementation."""
    from train_pointcloud_ffm import main as train_main

    train_main(argv)
