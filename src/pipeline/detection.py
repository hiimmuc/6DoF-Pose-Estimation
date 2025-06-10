"""
Detection and tracking functionality for 6DoF pose estimation pipeline.

This module provides functions for object detection, tracking,
segmentation, and extracting rotated bounding boxes from masks.
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


def get_rotating_bounding_box(mask: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Get minimum rotated bounding box from mask.

    Args:
        mask: Binary mask image

    Returns:
        Dictionary with box info (center, size, angle, corners), or None if no contours found
    """
    # Find contours in the mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    # Find the largest contour by area
    largest_contour = max(contours, key=cv2.contourArea)

    # Get minimum area rectangle
    rect = cv2.minAreaRect(largest_contour)

    # Extract rectangle parameters
    center = rect[0]  # (x, y)
    size = rect[1]  # (width, height)
    angle = rect[2]  # rotation angle in degrees

    # Get box corners
    box_points = cv2.boxPoints(rect)
    box_points = np.intp(box_points)

    return {
        "center": center,
        "size": size,
        "angle": angle,
        "corners": box_points,
        "contour": largest_contour,
        "rect": rect,
    }


class YOLODetector:
    """
    Class for YOLO-based detection, tracking, and segmentation.
    """

    def __init__(
        self,
        tracking_model_path: str,
        segmentation_model_path: str,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
        classes: Optional[List[int]] = None,
    ):
        """
        Initialize YOLO models for tracking and segmentation.

        Args:
            tracking_model_path: Path to YOLO tracking model
            segmentation_model_path: Path to YOLO segmentation model
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS
            classes: List of class IDs to detect (None for all classes)
        """
        self.tracking_model = YOLO(tracking_model_path)
        self.segmentation_model = YOLO(segmentation_model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes  # Will be None if not specified

    def track(self, frame: np.ndarray):
        """
        Run YOLO tracking on a frame.

        Args:
            frame: Input image frame

        Returns:
            YOLO tracking results
        """
        return self.tracking_model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=self.classes,
        )

    def segment(self, frame: np.ndarray):
        """
        Run YOLO segmentation on a frame.

        Args:
            frame: Input image frame

        Returns:
            YOLO segmentation results
        """
        return self.segmentation_model(
            frame, conf=self.conf_threshold, iou=self.iou_threshold, classes=self.classes
        )

    def get_masks_from_results(self, results):
        """
        Extract masks from segmentation results.

        Args:
            results: YOLO segmentation results

        Returns:
            List of masks if available, otherwise empty list
        """
        masks = []
        if hasattr(results[0], "masks") and results[0].masks is not None:
            for mask in results[0].masks.data:
                # Convert tensor mask to numpy array
                numpy_mask = mask.cpu().numpy()
                masks.append(numpy_mask)
        return masks
