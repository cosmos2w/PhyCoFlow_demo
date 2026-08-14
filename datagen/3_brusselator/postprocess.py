"""Convert raw Brusselator trajectories to unified HDF5.

Example:
  conda run -n phycoflow_env python datagen/3_brusselator/postprocess.py --raw-dir datagen/data/raw/brusselator/brusselator_canonical --output datagen/data/processed/brusselator.h5

The command shows trajectory progress, validates the result, prints a compact
summary, and writes ``brusselator_README.md`` beside the HDF5 file.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import postprocess_main


if __name__ == "__main__":
    postprocess_main("brusselator")

