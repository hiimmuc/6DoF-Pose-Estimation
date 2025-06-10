"""
Utilities for 6DoF pose estimation pipeline.

This module provides utilities such as performance monitoring
and other helper functions for the pipeline.
"""

import time
from typing import Dict, List

import cv2
import numpy as np


class PerfMonitor:
    """
    Performance monitoring utility for the 6DoF pipeline.
    Tracks FPS and processing times for different pipeline stages.
    """

    def __init__(self, history_size: int = 30):
        """
        Initialize the performance monitor.

        Args:
            history_size: Number of frames to keep in history for averaging
        """
        self.history_size = history_size
        self.last_time = None
        self.fps_history: List[float] = []
        self.stage_times: Dict[str, List[float]] = {
            "tracking": [],
            "segmentation": [],
            "pose_estimation": [],
            "visualization": [],
        }

    def start_frame(self) -> None:
        """Start timing a new frame"""
        self.last_time = time.time()

    def mark_stage(self, stage_name: str) -> None:
        """
        Mark the completion of a pipeline stage and record its time.

        Args:
            stage_name: Name of the completed stage
        """
        current_time = time.time()
        if self.last_time is not None and stage_name in self.stage_times:
            stage_time = current_time - self.last_time
            self.stage_times[stage_name].append(stage_time)
            if len(self.stage_times[stage_name]) > self.history_size:
                self.stage_times[stage_name].pop(0)
        self.last_time = current_time

    def end_frame(self) -> None:
        """End timing for the current frame and update FPS calculation"""
        current_time = time.time()
        if self.last_time is not None:
            frame_time = current_time - self.last_time
            self.fps_history.append(1.0 / frame_time if frame_time > 0 else 0)
            if len(self.fps_history) > self.history_size:
                self.fps_history.pop(0)

    def get_fps(self) -> float:
        """Get the average FPS over the history window"""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)

    def get_stage_time(self, stage_name: str) -> float:
        """Get the average time for a specific pipeline stage"""
        if stage_name not in self.stage_times or not self.stage_times[stage_name]:
            return 0.0
        return sum(self.stage_times[stage_name]) / len(self.stage_times[stage_name])

    def draw_stats(self, img: np.ndarray) -> np.ndarray:
        """
        Draw performance statistics on an image

        Args:
            img: Input image

        Returns:
            Image with performance stats overlay
        """
        result = img.copy()

        # Prepare stats text
        # fps = self.get_fps()
        tracking_time = self.get_stage_time("tracking") * 1000
        segmentation_time = self.get_stage_time("segmentation") * 1000
        pose_time = self.get_stage_time("pose_estimation") * 1000
        viz_time = self.get_stage_time("visualization") * 1000
        fps = 1 / sum([self.get_stage_time(stage) for stage in self.stage_times])

        # Create semi-transparent overlay for stats
        h, w = result.shape[:2]
        overlay = result.copy()
        cv2.rectangle(overlay, (10, h - 120), (250, h - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result, 0.3, 0, result)

        # Draw text with performance stats
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, f"FPS: {fps:.1f}", (20, h - 90), font, 0.6, (0, 255, 0), 2)
        cv2.putText(
            result, f"Tracking: {tracking_time:.1f} ms", (20, h - 70), font, 0.6, (0, 255, 0), 2
        )
        cv2.putText(
            result,
            f"Segmentation: {segmentation_time:.1f} ms",
            (20, h - 50),
            font,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result, f"Pose Est.: {pose_time:.1f} ms", (20, h - 30), font, 0.6, (0, 255, 0), 2
        )

        return result


def parse_args():
    """Parse command line arguments for the pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="6DoF Pose Estimation Pipeline")
    parser.add_argument(
        "--input",
        type=str,
        choices=["image", "video", "webcam", "realsense"],
        default="webcam",
        help="Input source type (image, video, webcam, realsense)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Path to input source or webcam index",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for YOLO detection",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.5,
        help="IoU threshold for YOLO NMS",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="+",
        default=[0, 41, 67],  # Default classes: person, cup, cell phone
        help="Filter by class IDs",
    )
    parser.add_argument(
        "--tracking-model",
        type=str,
        default="src/checkpoints/YOLO/yolo11n.pt",
        help="Path to YOLO tracking model",
    )
    parser.add_argument(
        "--segmentation-model",
        type=str,
        default="src/checkpoints/YOLO/yolo11n-seg.pt",
        help="Path to YOLO segmentation model",
    )
    parser.add_argument(
        "--axis-length",
        type=float,
        default=0.1,
        help="Length of 3D coordinate axes visualization in meters",
    )
    parser.add_argument(
        "--box-height",
        type=float,
        default=15.0,
        help="Default height for 3D bounding box in regular cameras (in units)",
    )
    parser.add_argument(
        "--max-box-height",
        type=float,
        default=50.0,
        help="Maximum height for 3D bounding box in RealSense mode (in units)",
    )
    parser.add_argument(
        "--show-3d-box",
        action="store_true",
        default=False,
        help="Show 3D bounding box visualization (default: True)",
    )
    parser.add_argument(
        "--no-3d-box",
        dest="show_3d_box",
        action="store_false",
        help="Hide 3D bounding box visualization",
    )
    parser.add_argument(
        "--show-axes",
        action="store_true",
        default=True,
        help="Show 3D coordinate axes visualization (default: True)",
    )
    parser.add_argument(
        "--no-axes",
        dest="show_axes",
        action="store_false",
        help="Hide 3D coordinate axes visualization",
    )
    parser.add_argument(
        "--use-enhanced-bbox",
        action="store_true",
        default=True,
        help="Use enhanced 3D bounding box estimation with depth maps and polygon masks (default: True)",
    )
    parser.add_argument(
        "--no-enhanced-bbox",
        dest="use_enhanced_bbox",
        action="store_false",
        help="Use legacy 3D bounding box estimation method",
    )

    return parser.parse_args()
