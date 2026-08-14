"""Generate raw 2D Kolmogorov-flow vorticity trajectories.

Canonical 192x192 GPU example (physical GPU 1 becomes logical cuda:0):
  CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env python datagen/4_navier_stokes/generate.py --device cuda:0 --resolution 192 --num-trajectories 100

CPU smoke example:
  conda run -n phycoflow_env python datagen/4_navier_stokes/generate.py --backend numpy --device cpu --resolution 32 --burn-in-time 0.02 --record-time 0.04 --num-trajectories 1

Run ``python datagen/4_navier_stokes/generate.py --help`` for Reynolds
number, forcing, perturbation, temporal controls, precision, and size options.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import generate_main


if __name__ == "__main__":
    generate_main("kolmogorov")

