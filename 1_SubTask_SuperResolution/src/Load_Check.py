
"""
project/src/load_and_check.py

Opens an HDF5 dataset file and recursively prints its structure:
  - Groups (with attributes)
  - Datasets (with shape, dtype, attributes)

This version is tailored for multi-resolution datasets.
"""

import h5py
from pathlib import Path
import sys

def print_structure(name, obj):
    """
    Callback for h5py.File.visititems().
    name: full path in the file
    obj:   h5py.Group or h5py.Dataset
    """
    if isinstance(obj, h5py.Group):
        attrs = dict(obj.attrs)
        print(f"Group:   {name}")
        if attrs:
            print(f"    attrs: {attrs}")
    elif isinstance(obj, h5py.Dataset):
        attrs = dict(obj.attrs)
        print(f"Dataset: {name}")
        print(f"    shape: {obj.shape}")
        print(f"    dtype: {obj.dtype}")
        if attrs:
            print(f"    attrs: {attrs}")
    else:
        # unlikely, but covers other HDF5 object types
        print(f"{type(obj).__name__}: {name}")

def main():
    # Determine the HDF5 file path relative to this script
    script_dir  = Path(__file__).resolve().parent          # project/src
    project_root = script_dir.parent                        # project/
    h5_path     = project_root / "Dataset" / "PDE_Bench" / "Processed" / "CFD_L_res.h5"

    # Dataset/PDE_Bench/Processed/RD_H_Res.h5
    # Dataset/PDE_Bench/Processed/RD_M_Res.h5
    # Dataset/PDE_Bench/Processed/RD_L_Res.h5
    # Dataset/PDE_Bench/Processed/CFD_H_res.h5
    # Dataset/PDE_Bench/Processed/CFD_M_res.h5
    # Dataset/PDE_Bench/Processed/CFD_L_res.h5

    if not h5_path.exists():
        print(f"ERROR: HDF5 file not found:\n  {h5_path}", file=sys.stderr)
        sys.exit(1)

    # Open file in read‐only mode
    with h5py.File(h5_path, "r") as f:
        print(f"Opened HDF5 file: {h5_path}")
        print("Contents:")
        f.visititems(print_structure)

    print("Done.")

if __name__ == "__main__":
    main()