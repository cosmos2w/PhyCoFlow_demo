"""Visualize electro-thermal fields, geometry, heating, and spatial scales.

Processed-HDF5 example:
  conda run -n phycoflow_env python datagen/5_electro_thermal/visualize.py --input datagen/data/processed/electro_thermal.h5 --trajectory 0 --time-index 0 --output electro_thermal_qa.png

Raw-NPZ example:
  conda run -n phycoflow_env python datagen/5_electro_thermal/visualize.py --input <raw-dir>/trajectories/trajectory_000000.npz --output electro_thermal_raw_qa.pdf
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import visualize_main


if __name__ == "__main__":
    visualize_main("electro_thermal")
