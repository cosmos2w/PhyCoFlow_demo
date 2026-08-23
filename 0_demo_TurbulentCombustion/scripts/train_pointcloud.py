#!/usr/bin/env python3
"""Train a public PointCloudFFM profile."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from phycoflow_pointcloud.training import main

if __name__ == "__main__":
    main()
