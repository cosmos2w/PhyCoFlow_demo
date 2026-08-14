"""Visualize KS fields, spectra, phase portrait, and PDE quality.

Processed-HDF5 example:
  conda run -n phycoflow_env python datagen/2_ks/visualize.py --input datagen/data/processed/ks.h5 --trajectory 0 --time-index -1 --output ks_qa.png

Raw-NPZ example:
  conda run -n phycoflow_env python datagen/2_ks/visualize.py --input <raw-dir>/trajectories/trajectory_000000.npz --output ks_raw_qa.pdf
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import visualize_main


if __name__ == "__main__":
    visualize_main("ks")

