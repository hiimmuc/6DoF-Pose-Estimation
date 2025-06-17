"""Data classes for pose estimation results."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


@dataclass
class PoseResult:
    """
    Data class for storing 6DoF pose estimation results.

    Attributes:
        object_id: Tracking ID of the object.
        class_id: Class ID of the detected object.
        class_name: Name of the detected object class.
        confidence: Detection confidence score.
        x: X-coordinate of object center in mm.
        y: Y-coordinate of object center in mm.
        z: Z-coordinate (depth) of object center in mm.
        roll: Roll angle in degrees.
        pitch: Pitch angle in degrees.
        yaw: Yaw angle in degrees.
        bbox_2d: 2D bounding box [x, y, width, height].
        bbox_rotated: Rotated 2D bounding box [x, y, width, height, angle].
        mask: Binary segmentation mask.
    """

    object_id: int
    class_id: int
    class_name: str
    confidence: float
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    bbox_2d: List[float]
    bbox_rotated: List[float]
    mask: Optional[np.ndarray] = None
    bbox_3d: Optional[np.ndarray] = None
