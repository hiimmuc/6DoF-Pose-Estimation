#!/usr/bin/env python3
"""
Command-line entry point for 6DoF pose estimation pipeline.

This script serves as the main entry point for running the 6DoF pose estimation pipeline.
It passes all command-line arguments to the pipeline.
"""

import os
import sys

# Ensure the current directory is in the path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from src.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()
