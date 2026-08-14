"""Generate raw transient 2D Elder-type mass-transport trajectories.

Canonical CPU example (the sparse SciPy solver does not use CUDA):
  conda run -n phycoflow_env python datagen/6_mass_transport_fluid/generate.py --backend numpy --device cpu --resolution 128 --record-time 20 --dt 0.25 --save-every 8 --num-trajectories 8

Moderate one-trajectory example ending at year 4:
  conda run -n phycoflow_env python datagen/6_mass_transport_fluid/generate.py --backend numpy --device cpu --resolution 32 --record-time 4 --dt 0.5 --save-every 4 --num-trajectories 1 --dataset-id mass_transport_demo

Run ``python datagen/6_mass_transport_fluid/generate.py --help`` for source,
Darcy, transport, nonlinear, adaptive-step, precision, and path options.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import generate_main


if __name__ == "__main__":
    generate_main("mass_transport_fluid")
