"""Deterministic trajectory and frame split helpers."""

from __future__ import annotations

from dataclasses import dataclass

import h5py
import numpy as np

SPLIT_ALIASES = {"val": "validation", "valid": "validation"}


@dataclass(frozen=True)
class SplitSelection:
    split: str
    trajectory_indices: tuple[int, ...]
    frame_indices: tuple[int, ...]
    strategy: str


def normalize_split(split: str) -> str:
    split = SPLIT_ALIASES.get(split.lower(), split.lower())
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"unknown split {split!r}")
    return split


def chronological_frame_indices(num_frames: int, split: str, stride: int = 1) -> np.ndarray:
    split = normalize_split(split)
    if stride < 1:
        raise ValueError("time stride must be positive")
    train_end = int(num_frames * 0.8)
    validation_end = int(num_frames * 0.9)
    bounds = {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, num_frames),
    }
    start, stop = bounds[split]
    return np.arange(start, stop, stride, dtype=np.int64)


def legacy_seeded_random_frame_indices(
    num_frames: int,
    split: str,
    *,
    train_ratio: float = 0.9,
    seed: int = 42,
    stride: int = 1,
) -> np.ndarray:
    """Reproduce the historical point-cloud demo's seeded 90/10 frame split.

    The legacy loader shuffled all eligible frame indices with NumPy's default
    generator, assigned the leading ``int(N * train_ratio)`` frames to training,
    used the complement for both validation and test, then sorted each selection.
    """
    split = normalize_split(split)
    if stride < 1:
        raise ValueError("time stride must be positive")
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be strictly between zero and one")
    indices = np.arange(0, num_frames, stride, dtype=np.int64)
    np.random.default_rng(seed).shuffle(indices)
    train_count = int(indices.size * train_ratio)
    selected = indices[:train_count] if split == "train" else indices[train_count:]
    return np.sort(selected)


def resolve_split(handle: h5py.File, split: str, time_stride: int = 1) -> SplitSelection:
    split = normalize_split(split)
    batch_count, time_count = int(handle["fields"].shape[0]), int(handle["fields"].shape[1])
    if batch_count > 1:
        if f"splits/{split}" not in handle:
            raise KeyError(f"multi-trajectory dataset is missing splits/{split}")
        trajectories = tuple(int(v) for v in handle[f"splits/{split}"][:])
        frames = tuple(range(0, time_count, time_stride))
        return SplitSelection(split, trajectories, frames, "stored_trajectory")

    # A canonical one-trajectory demo may explicitly contain train-only splits.
    if "splits" in handle and any(
        handle[f"splits/{name}"].shape[0] for name in ("validation", "test")
    ):
        trajectories = tuple(int(v) for v in handle[f"splits/{split}"][:])
        return SplitSelection(
            split, trajectories, tuple(range(0, time_count, time_stride)), "stored_trajectory"
        )
    if "splits" in handle and split != "train" and handle[f"splits/{split}"].shape[0] == 0:
        return SplitSelection(split, (), (), "stored_train_only")
    if "splits" in handle and split == "train":
        return SplitSelection(
            split, (0,), tuple(range(0, time_count, time_stride)), "stored_train_only"
        )

    frames = tuple(int(v) for v in chronological_frame_indices(time_count, split, time_stride))
    return SplitSelection(split, (0,), frames, "chronological_frames_80_10_10")
