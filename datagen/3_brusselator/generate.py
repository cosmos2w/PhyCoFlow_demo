"""Generate raw 2D Brusselator reaction--diffusion trajectories.

Canonical 192x192 GPU example:
  CUDA_VISIBLE_DEVICES=1 conda run -n phycoflow_env python datagen/3_brusselator/generate.py --device cuda:0 --resolution 192 --num-trajectories 100

CPU smoke example:
  conda run -n phycoflow_env python datagen/3_brusselator/generate.py --backend numpy --device cpu --resolution 32 --record-time 0.1 --num-trajectories 1

Run ``python datagen/3_brusselator/generate.py --help`` for A, B,
diffusivities, noise, temporal controls, precision, and dataset-size options.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import generate_main


if __name__ == "__main__":
    generate_main("brusselator")

