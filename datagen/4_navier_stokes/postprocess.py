"""Convert raw Kolmogorov-flow vorticity to unified `(u,v,p)` HDF5.

Example:
  conda run -n phycoflow_env python datagen/4_navier_stokes/postprocess.py --raw-dir datagen/data/raw/kolmogorov/kolmogorov_canonical --output datagen/data/processed/kolmogorov.h5

The command shows trajectory progress, validates the result, prints a compact
summary, and writes ``kolmogorov_README.md`` beside the HDF5 file.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import postprocess_main


if __name__ == "__main__":
    postprocess_main("kolmogorov")

