"""Visualize Burgers fields and physical-quality diagnostics.

Processed-HDF5 example:
  conda run -n phycoflow_env python datagen/1_burgers/visualize.py --input datagen/data/processed/burgers.h5 --trajectory 0 --time-index -1 --output burgers_qa.png

Raw-NPZ example:
  conda run -n phycoflow_env python datagen/1_burgers/visualize.py --input <raw-dir>/trajectories/trajectory_000000.npz --output burgers_raw_qa.pdf
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import visualize_main


if __name__ == "__main__":
    visualize_main("burgers")

