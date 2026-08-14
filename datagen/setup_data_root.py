"""Safely create the optional large-data symbolic link.

Check without changing anything:
  conda run -n phycoflow_env python datagen/setup_data_root.py --check-only

Create `/data/wanglz/PhyCoFlow/datagen` and `datagen/data` -> that directory:
  conda run -n phycoflow_env python datagen/setup_data_root.py

Custom locations:
  python datagen/setup_data_root.py --target /data/wanglz/my_dataset_root --link datagen/data

The script never replaces an existing regular file/directory or a symlink that
points somewhere else.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


DATAGEN_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Safely configure datagen/data as a symbolic link to large storage.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--target", type=Path, default=Path("/data/wanglz/PhyCoFlow/datagen"), help="Real large-data directory.")
    parser.add_argument("--link", type=Path, default=DATAGEN_ROOT / "data", help="Symbolic-link path exposed to generation scripts.")
    parser.add_argument("--check-only", action="store_true", help="Report permissions, free space, and existing link state without creating anything.")
    args = parser.parse_args()
    target = args.target.expanduser().resolve(strict=False)
    link = args.link.expanduser().absolute()
    existing_parent = target
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    disk = shutil.disk_usage(existing_parent)
    print(f"Target: {target}")
    print(f"Link:   {link}")
    print(f"Existing target parent writable: {os.access(existing_parent, os.W_OK)}")
    print(f"Available target storage: {disk.free / 2**30:.1f} GiB")
    if link.is_symlink():
        current_target = link.resolve(strict=False)
        print(f"Existing symlink target: {current_target}")
        if current_target != target:
            raise SystemExit("Refusing to replace a symlink that points to a different target.")
        print("The symbolic link is already configured correctly.")
        return
    if link.exists():
        raise SystemExit(f"Refusing to replace existing non-symlink path: {link}")
    if args.check_only:
        print("Check complete; no path was created.")
        return
    if not os.access(existing_parent, os.W_OK):
        raise SystemExit(f"Target parent is not writable: {existing_parent}")
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    print(f"Created {link} -> {target}")


if __name__ == "__main__":
    main()

