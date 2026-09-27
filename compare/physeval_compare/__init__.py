"""Paired detector/tracker comparison for PhysEval."""

from pathlib import Path
import sys

CORE_ROOT = Path(__file__).resolve().parents[2]
if str(CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(CORE_ROOT))

from .config import MODE_SPECS, ModeSpec

__all__ = ["MODE_SPECS", "ModeSpec"]
