"""
6DoF Pose Estimation Pipeline

This module provides functionality for estimating 6DoF poses of objects
from images, videos, webcam, or RealSense camera.
"""

from .detection import YOLODetector, get_rotating_bounding_box
from .input_sources import (
    ImageSource,
    InputSource,
    RealSenseSource,
    VideoSource,
    WebcamSource,
    create_input_source,
)
from .main import estimate_poses, run_pipeline
from .pose_estimation import (
    compute_3d_pose,
    create_3d_bbox,
    create_3d_bbox_enhanced,
    get_euler_angles,
    get_robust_depth,
)
from .utils import PerfMonitor
from .visualization import (
    colorize_depth_image,
    create_grid_display,
    create_side_by_side_display,
    draw_3d_axes,
    draw_3d_bbox,
    draw_rotating_box,
)

__all__ = [
    "run_pipeline",
    "estimate_poses",
    "InputSource",
    "ImageSource",
    "VideoSource",
    "WebcamSource",
    "RealSenseSource",
    "create_input_source",
    "YOLODetector",
    "get_rotating_bounding_box",
    "compute_3d_pose",
    "get_euler_angles",
    "create_3d_bbox",
    "create_3d_bbox_enhanced",
    "get_robust_depth",
    "draw_3d_axes",
    "draw_3d_bbox",
    "draw_rotating_box",
    "create_grid_display",
    "create_side_by_side_display",
    "colorize_depth_image",
    "PerfMonitor",
]
