#!/usr/bin/env python3
"""Backward-compatibility shim — moved to run_progressive_evaluation_with_regression_guard.py."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from scripts.run_progressive_evaluation_with_regression_guard import main  # noqa: F401

if __name__ == "__main__":
    main()
