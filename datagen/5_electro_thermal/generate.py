"""Generate raw steady 2D electro-thermal realizations.

Canonical CPU example (the sparse SciPy solver does not use CUDA):
  conda run -n phycoflow_env python datagen/5_electro_thermal/generate.py --backend numpy --device cpu --resolution 128 --num-trajectories 16 --workers 4

Moderate one-realization example:
  conda run -n phycoflow_env python datagen/5_electro_thermal/generate.py --backend numpy --device cpu --resolution 48 --num-trajectories 1 --dataset-id electro_thermal_demo

Run ``python datagen/5_electro_thermal/generate.py --help`` for ellipse,
material, incident-wave, absorbing-layer, coupling, precision, worker, and path
options. Each worker solves one independent trajectory; the parent process owns
all checksummed data writes and manifest updates.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phycoflow_datagen.cli import generate_main


if __name__ == "__main__":
    generate_main("electro_thermal")
