"""Make the local datagen package importable without installation."""

from pathlib import Path
import sys


DATAGEN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DATAGEN_ROOT))

