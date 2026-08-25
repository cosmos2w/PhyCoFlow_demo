"""Shared helpers for thin compatibility-preserving command wrappers."""

from __future__ import annotations

import contextlib
import sys
import tempfile
from collections.abc import Iterator, Mapping
from pathlib import Path

import yaml


@contextlib.contextmanager
def temporary_yaml(config: Mapping) -> Iterator[Path]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=False)
        path = Path(handle.name)
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


@contextlib.contextmanager
def replaced_argv(argv: list[str]) -> Iterator[None]:
    previous = sys.argv
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous
