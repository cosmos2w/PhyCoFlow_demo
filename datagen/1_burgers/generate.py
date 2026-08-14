"""Generate raw 1D viscous-Burgers trajectories.

GPU example (physical GPU 1 becomes logical cuda:0):
  CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env python datagen/1_burgers/generate.py --device cuda:0 --num-trajectories 100

CPU reference example:
  conda run -n phycoflow_env python datagen/1_burgers/generate.py --backend numpy --device cpu --num-trajectories 3

Run ``python datagen/1_burgers/generate.py --help`` for all dataset-size,
time-discretization, initial-condition, viscosity, precision, and path options.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import generate_main


if __name__ == "__main__":
    generate_main("burgers")

