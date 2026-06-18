"""Ensure the repo root (containing the built ``sparc`` package) is importable
when running the test suite in-place without installing the package."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
