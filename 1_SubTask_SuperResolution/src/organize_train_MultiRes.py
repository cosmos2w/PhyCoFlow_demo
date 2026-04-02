import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py

"""
If want to train on the Computational Fluid Dynamics (CFD) dataset using the first field (idx 0, which is Density) 
with an even 1:1:1 ratio of Low, Medium, and High-resolution examples:

python organize_train_MultiRes.py \
    --processed-root ./Dataset/PDE_Bench/Processed \
    --dataset-name CFD \
    --selected-field-idx 0 \
    --multires-ratio 1:1:1
"""

def parse_ratio(ratio: str) -> Tuple[int, int, int]:
    """
    Parses a string ratio (e.g., "1:2:1") into a tuple of integers representing
    the split for Low, Medium, and High resolutions.

    Args:
        ratio (str): A string in the format "L:M:H".

    Returns:
        Tuple[int, int, int]: The parsed integer weights.
    """
    parts = ratio.split(":")
    if len(parts) != 3:
        raise ValueError(f"multires ratio must look like '1:1:1', got: {ratio}")
    
    a, b, c = [int(x) for x in parts]
    
    if a < 0 or b < 0 or c < 0:
        raise ValueError("ratio entries must be non-negative")
    if a + b + c == 0:
        raise ValueError("ratio entries cannot all be zero")
        
    return a, b, c


def allocate_counts(n: int, weights: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Distributes a total integer `n` into three buckets proportionally based on `weights`.
    Uses the largest remainder method to handle rounding and ensure exactly `n` is allocated.

    Args:
        n (int): Total number of items (e.g., total training cases).
        weights (Tuple[int, int, int]): The desired distribution weights.

    Returns:
        Tuple[int, int, int]: The exact number of items allocated to each bucket.
    """
    total = sum(weights)
    
    # Calculate exact float distributions
    raw = [n * w / total for w in weights]
    
    # Get the floor of each distribution
    base = [int(x) for x in raw]
    
    # Calculate how many items are left behind due to flooring
    rem = n - sum(base)
    
    # Sort indices by the highest fractional leftover to distribute the remainder fairly
    frac_order = sorted(range(3), key=lambda i: raw[i] - base[i], reverse=True)
    for i in range(rem):
        base[frac_order[i]] += 1
        
    return tuple(base)


def infer_dataset_info(processed_root: Path, dataset_name: str) -> Dict:
    """
    Reads HDF5 dataset files to extract essential metadata without loading all data into RAM.
    
    Args:
        processed_root (Path): Directory containing the processed PDEBench .h5 files.
        dataset_name (str): Name of the dataset ("RD" for Reaction-Diffusion, "CFD" for Computational Fluid Dynamics).

    Returns:
        Dict: Metadata including file paths, field names, shape configurations, and grid resolutions.
    """
    dataset_name = dataset_name.upper()
    
    # Map the dataset name to the expected file structures and specific fields
    if dataset_name == "RD":
        paths = {
            "H": processed_root / "RD_H_Res.h5",
            "M": processed_root / "RD_M_Res.h5",
            "L": processed_root / "RD_L_Res.h5",
        }
        field_names = ["Species_0", "Species_1"]
    elif dataset_name == "CFD":
        paths = {
            "H": processed_root / "CFD_H_res.h5",
            "M": processed_root / "CFD_M_res.h5",
            "L": processed_root / "CFD_L_res.h5",
        }
        field_names = ["density", "pressure", "Vx", "Vy"]
    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    # Validate that all required files actually exist
    for tag, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing processed file for {dataset_name}-{tag}: {p}")

    # Peek into the High-Resolution file to get global shape metrics
    with h5py.File(paths["H"], "r") as f:
        fields = f["fields"]
        n_cases = int(fields.shape[0])
        n_time = int(fields.shape[1])
        n_fields = int(fields.shape[-1])
        time = f["time"][:].tolist()

    res_meta = {}
    
    # Check each resolution file to count unique spatial coordinate points (grid sizing)
    for tag, p in paths.items():
        with h5py.File(p, "r") as f:
            n_pts = int(f["coordinates"].shape[0])
            coords = f["coordinates"][:, 0, 0, :]
            xs = len(set(coords[:, 0].tolist()))
            ys = len(set(coords[:, 1].tolist()))
            res_meta[tag] = {
                "path": str(p),
                "num_points": n_pts,
                "Num_x": xs,
                "Num_y": ys,
            }

    return {
        "dataset_name": dataset_name,
        "processed_root": str(processed_root),
        "paths": {k: str(v) for k, v in paths.items()},
        "field_names": field_names[:n_fields],
        "n_cases": n_cases,
        "n_time": n_time,
        "n_fields": n_fields,
        "time": time,
        "resolutions": res_meta,
    }


def build_manifest(
    processed_root: Path,
    dataset_name: str,
    selected_field_idx: int,
    multires_ratio: str,
    train_fraction: float = 0.9,
) -> Dict:
    """
    Constructs the overall manifest dictionary detailing how the dataset is split 
    across training and validation, and across different resolutions.
    """
    info = infer_dataset_info(processed_root, dataset_name)
    n_cases = info["n_cases"]

    if not (0 <= selected_field_idx < info["n_fields"]):
        raise ValueError(
            f"selected_field_idx={selected_field_idx} out of range for dataset with {info['n_fields']} fields"
        )

    # Simple chronological/sequential split for Train vs Validation
    n_train = int(n_cases * train_fraction)
    train_cases = list(range(0, n_train))
    val_cases = list(range(n_train, n_cases))

    # Split the training cases into Low, Medium, and High resolutions based on ratio
    weights = parse_ratio(multires_ratio)
    nL, nM, nH = allocate_counts(len(train_cases), weights)

    train_cases_by_res = {
        "L": train_cases[:nL],
        "M": train_cases[nL:nL + nM],
        "H": train_cases[nL + nM:nL + nM + nH],
    }

    manifest = {
        "dataset_mode": "pdebench_multires",
        "dataset_name": info["dataset_name"],
        "selected_field_idx": int(selected_field_idx),
        "selected_field_name": info["field_names"][selected_field_idx],
        "multires_ratio": multires_ratio,
        "train_fraction": train_fraction,
        "n_cases": info["n_cases"],
        "n_time": info["n_time"],
        "n_fields_raw": info["n_fields"],
        "field_names_raw": info["field_names"],
        "paths": info["paths"],
        "resolutions": info["resolutions"],
        "split": {
            "train_cases_by_res": train_cases_by_res,
            "val_cases": val_cases, # Validation is kept entirely at high-res by default
        },
        "notes": {
            "train_sampling": "All frames from the listed train cases are used.",
            "val_sampling": "All frames from val cases are reserved for validation/evaluation; high resolution is the default.",
            "sensor_comparison_across_resolutions": "Use canonical physical sensor coordinates and snap them to nearest grid nodes at each resolution.",
        },
    }
    return manifest


def default_manifest_path(processed_root: Path, dataset_name: str, selected_field_idx: int, multires_ratio: str) -> Path:
    """Generates a default file path to save the JSON manifest based on its parameters."""
    safe_ratio = multires_ratio.replace(":", "-")
    out_dir = processed_root / "manifests"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{dataset_name.upper()}_field{selected_field_idx}_ratio_{safe_ratio}.json"


def main():
    p = argparse.ArgumentParser("Organize PDEBench multi-resolution train/val protocol.")

    p.add_argument("--processed-root", type=str, default="Dataset/PDE_Bench/Processed", 
                   help="Path to the directory containing processed H5 files.")
    p.add_argument("--dataset-name", type=str, required=True, choices=["RD", "CFD"], 
                   help="Target dataset to parse.")
    p.add_argument("--selected-field-idx", type=int, required=True, 
                   help="Index of the physical field to target (e.g., 0 for density).")
    p.add_argument("--multires-ratio", type=str, default="1:1:1", 
                   help="Ratio of L:M:H resolution splits for training (e.g., '2:1:1').")
    p.add_argument("--train-fraction", type=float, default=0.9, 
                   help="Percentage of total cases to allocate for training vs validation.")
    p.add_argument("--output", type=str, default=None, 
                   help="Custom output path for the JSON manifest.")
    args = p.parse_args()

    processed_root = Path(args.processed_root)
    manifest = build_manifest(
        processed_root=processed_root,
        dataset_name=args.dataset_name,
        selected_field_idx=args.selected_field_idx,
        multires_ratio=args.multires_ratio,
        train_fraction=args.train_fraction,
    )

    out_path = Path(args.output) if args.output is not None else default_manifest_path(
        processed_root, args.dataset_name, args.selected_field_idx, args.multires_ratio
    )
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[*] Manifest written to: {out_path}")
    print(json.dumps({
        "dataset_name": manifest["dataset_name"],
        "selected_field_name": manifest["selected_field_name"],
        "multires_ratio": manifest["multires_ratio"],
        "train_cases_by_res": {k: len(v) for k, v in manifest["split"]["train_cases_by_res"].items()},
        "val_cases": len(manifest["split"]["val_cases"]),
    }, indent=2))


if __name__ == "__main__":
    main()