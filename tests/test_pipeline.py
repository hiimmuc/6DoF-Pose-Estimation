"""
Test script for 6DoF pose estimation pipeline components.

This script tests the individual components of the pipeline
to ensure they are working correctly.
"""

import os
import sys
import unittest

import cv2
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.pose_estimator import PoseEstimator
from src.data.pose_result import PoseResult
from src.utils.input_source import InputSource
from src.utils.performance import PerformanceMonitor
from src.utils.visualization import create_visualization_grid


class TestPerformanceMonitor(unittest.TestCase):
    """Test the PerformanceMonitor class."""

    def test_timing(self):
        """Test timing functionality."""
        monitor = PerformanceMonitor(window_size=10)

        # Test start and end timer
        start = monitor.start_timer()
        # Simulate work
        _ = [i**2 for i in range(1000)]
        elapsed = monitor.end_timer("test_stage", start)

        self.assertGreater(elapsed, 0)
        self.assertEqual(len(monitor.stages["test_stage"]), 1)

        # Test multiple timings
        for _ in range(5):
            start = monitor.start_timer()
            # Simulate work
            _ = [i**2 for i in range(1000)]
            monitor.end_timer("test_stage", start)

        self.assertEqual(len(monitor.stages["test_stage"]), 6)

        # Test FPS calculation
        fps = monitor.get_fps("test_stage")
        self.assertGreater(fps, 0)

        # Test window size limit
        for _ in range(10):
            start = monitor.start_timer()
            monitor.end_timer("test_stage", start)

        self.assertEqual(len(monitor.stages["test_stage"]), 10)

        # Test reset
        monitor.reset()
        self.assertEqual(len(monitor.stages), 0)


class TestVisualization(unittest.TestCase):
    """Test visualization functions."""

    def test_create_visualization_grid(self):
        """Test creating visualization grid."""
        # Create dummy frames
        h, w = 240, 320
        frames = {
            "tracking_vis": np.zeros((h, w, 3), dtype=np.uint8),
            "segmentation_vis": np.ones((h, w, 3), dtype=np.uint8) * 64,
            "pose_vis": np.ones((h, w, 3), dtype=np.uint8) * 128,
            "depth_vis": np.ones((h, w, 3), dtype=np.uint8) * 192,
        }

        # Create grid
        grid = create_visualization_grid(frames)

        # Check grid dimensions
        self.assertEqual(grid.shape, (h * 2, w * 2, 3))

        # Test missing frame raises exception
        incomplete_frames = frames.copy()
        del incomplete_frames["tracking_vis"]

        with self.assertRaises(ValueError):
            create_visualization_grid(incomplete_frames)


class TestPoseResultClass(unittest.TestCase):
    """Test the PoseResult dataclass."""

    def test_pose_result_creation(self):
        """Test creating a PoseResult instance."""
        result = PoseResult(
            object_id=1,
            class_id=0,
            class_name="person",
            confidence=0.95,
            x=100.0,
            y=200.0,
            z=1000.0,
            roll=10.0,
            pitch=20.0,
            yaw=30.0,
            bbox_2d=[100, 200, 50, 80],
            bbox_rotated=[100, 200, 50, 80, 15.0],
        )

        self.assertEqual(result.object_id, 1)
        self.assertEqual(result.class_name, "person")
        self.assertEqual(result.z, 1000.0)
        self.assertEqual(result.bbox_2d, [100, 200, 50, 80])
        self.assertEqual(result.bbox_rotated, [100, 200, 50, 80, 15.0])


if __name__ == "__main__":
    unittest.main()
