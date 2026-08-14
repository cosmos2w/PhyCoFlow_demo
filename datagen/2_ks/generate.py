"""Generate raw 1D Kuramoto--Sivashinsky trajectories.

GPU example (physical GPU 1 becomes logical cuda:0):
  CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env python datagen/2_ks/generate.py --device cuda:0 --num-trajectories 100

CPU reference example:
  conda run -n phycoflow_env python datagen/2_ks/generate.py --backend numpy --device cpu --num-trajectories 3

Run ``python datagen/2_ks/generate.py --help`` for burn-in, record length,
saved cadence, resolution, initial-noise, precision, and path options.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import generate_main


if __name__ == "__main__":
    generate_main("ks")

