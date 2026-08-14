"""Convert raw Elder-type mass-transport trajectories to unified HDF5.

Example:
  conda run -n phycoflow_env python datagen/6_mass_transport_fluid/postprocessing.py --raw-dir datagen/data/raw/mass_transport_fluid/mass_transport_fluid_canonical --output datagen/data/processed/mass_transport_fluid.h5

The command shows trajectory progress, validates the result, prints a compact
summary, and writes ``mass_transport_fluid_README.md`` beside the HDF5 file.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import postprocess_main


if __name__ == "__main__":
    postprocess_main("mass_transport_fluid")
