"""Visualize Brusselator fields and multi-field physical quality.

Processed-HDF5 example:
  conda run -n phycoflow_env python datagen/3_brusselator/visualize.py --input datagen/data/processed/brusselator.h5 --trajectory 0 --time-index -1 --output brusselator_qa.png

Raw-NPZ example:
  conda run -n phycoflow_env python datagen/3_brusselator/visualize.py --input <raw-dir>/trajectories/trajectory_000000.npz --output brusselator_raw_qa.pdf
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import visualize_main


if __name__ == "__main__":
    visualize_main("brusselator")

