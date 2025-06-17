"""Utility functions for camera operations."""

from typing import Tuple

import numpy as np


def get_default_camera_parameters() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get default camera intrinsic parameters and distortion coefficients.

    In a real application, these would be obtained through camera calibration.

    Returns:
        Tuple containing:
            - Camera intrinsic matrix (3x3)
            - Distortion coefficients
    """
    # Default camera matrix for a 640x480 camera
    camera_matrix = np.array([[500.0, 0, 320.0], [0, 500.0, 240.0], [0, 0, 1]])

    # Default distortion coefficients (no distortion)
    dist_coefs = np.zeros(5)

    return camera_matrix, dist_coefs
