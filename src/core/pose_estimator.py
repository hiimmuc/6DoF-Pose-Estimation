"""Core pose estimation functionality."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from src.data.pose_result import PoseResult
from src.utils.performance import PerformanceMonitor
from src.utils.visualization import draw_pose


class PoseEstimator:
    """Core class for estimating 6DoF pose from images and depth data."""

    def __init__(self, tracking_model_path: str, segmentation_model_path: str) -> None:
        """
        Initialize the 6DoF pose estimator.

        Args:
            tracking_model_path: Path to YOLO tracking model.
            segmentation_model_path: Path to YOLO segmentation model.
        """
        self.tracking_model = YOLO(tracking_model_path)
        self.segmentation_model = YOLO(segmentation_model_path)
        self.performance_monitor = PerformanceMonitor()

    def process_frame(
        self,
        rgb_frame: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coefs: np.ndarray,
        depth_frame: Optional[np.ndarray] = None,
        detection_params: Dict = {"conf": 0.5, "iou": 0.5},
        segmentation_params: Dict = {"conf": 0.5, "iou": 0.5},
    ) -> Tuple[List[PoseResult], Dict]:
        """
        Process a frame to detect, track, segment and estimate 6DoF pose of objects.

        Args:
            rgb_frame: RGB input frame.
            camera_matrix: Camera intrinsic matrix.
            dist_coefs: Distortion coefficients.
            depth_frame: Depth frame (optional).
            detection_params: Parameters for object detection.
            segmentation_params: Parameters for segmentation.

        Returns:
            Tuple containing:
                - List of PoseResult objects
                - Dictionary of visualization frames
        """
        # Initialize results and visualizations
        results = []
        visualization = {}

        # Run detection pipeline
        tracking_vis = self._run_tracking(rgb_frame, detection_params, visualization)
        segmentation_vis = self._run_segmentation(rgb_frame, segmentation_params, visualization)

        # Create pose visualization
        pose_vis = rgb_frame.copy()

        # Create depth visualization
        depth_vis = self._prepare_depth_visualization(depth_frame, rgb_frame)
        visualization["depth_vis"] = depth_vis

        # Process detected objects
        start_time = self.performance_monitor.start_timer()
        if hasattr(visualization.get("tracking_results", []), "boxes"):
            results = self._process_detections(
                visualization["tracking_results"],
                visualization.get("segmentation_results"),
                rgb_frame,
                depth_frame,
                camera_matrix,
                dist_coefs,
                pose_vis,
            )

        self.performance_monitor.end_timer("pose_estimation", start_time)

        # Store performance metrics
        start_time = self.performance_monitor.start_timer()
        visualization["fps_info"] = self.performance_monitor.get_all_fps()
        visualization["time_info"] = self.performance_monitor.get_all_times()
        visualization["pose_vis"] = pose_vis
        self.performance_monitor.end_timer("visualization", start_time)

        return results, visualization

    def _run_tracking(
        self, rgb_frame: np.ndarray, detection_params: Dict, visualization: Dict
    ) -> np.ndarray:
        """Run object detection and tracking."""
        start_time = self.performance_monitor.start_timer()
        tracking_results = self.tracking_model.track(
            rgb_frame,
            persist=True,
            conf=detection_params.get("conf", 0.5),
            iou=detection_params.get("iou", 0.5),
        )
        self.performance_monitor.end_timer("tracking", start_time)

        # Store results and create visualization
        visualization["tracking_results"] = tracking_results[0] if tracking_results else None

        tracking_vis = rgb_frame.copy()
        if tracking_results and len(tracking_results) > 0:
            tracking_vis = tracking_results[0].plot()

        visualization["tracking_vis"] = tracking_vis
        return tracking_vis

    def _run_segmentation(
        self, rgb_frame: np.ndarray, segmentation_params: Dict, visualization: Dict
    ) -> np.ndarray:
        """Run instance segmentation."""
        start_time = self.performance_monitor.start_timer()
        segmentation_results = self.segmentation_model(
            rgb_frame,
            conf=segmentation_params.get("conf", 0.5),
            iou=segmentation_params.get("iou", 0.5),
        )
        self.performance_monitor.end_timer("segmentation", start_time)

        # Store results and create visualization
        visualization["segmentation_results"] = (
            segmentation_results[0] if segmentation_results else None
        )

        segmentation_vis = rgb_frame.copy()
        if segmentation_results and len(segmentation_results) > 0:
            segmentation_vis = segmentation_results[0].plot()

        visualization["segmentation_vis"] = segmentation_vis
        return segmentation_vis

    def _prepare_depth_visualization(
        self, depth_frame: Optional[np.ndarray], rgb_frame: np.ndarray
    ) -> np.ndarray:
        """Create visualization of depth data."""
        if depth_frame is not None:
            # Normalize depth for visualization
            depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        else:
            depth_vis = np.zeros_like(rgb_frame)
            cv2.putText(
                depth_vis,
                "No depth data",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            return depth_vis

    def _process_detections(
        self,
        tracking_results,
        segmentation_results,
        rgb_frame: np.ndarray,
        depth_frame: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        dist_coefs: np.ndarray,
        pose_vis: np.ndarray,
    ) -> List[PoseResult]:
        """Process detections to extract pose information."""
        results = []
        boxes = tracking_results.boxes

        for box in boxes:
            # Skip if box has no tracking ID
            if not hasattr(box, "id") or box.id is None:
                continue

            # Extract basic detection info
            track_id = int(box.id.item())
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())

            # Get class name
            class_name = (
                tracking_results.names[class_id]
                if hasattr(tracking_results, "names")
                else f"class_{class_id}"
            )

            # Get bounding box
            x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
            bbox_2d = [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]

            # Find corresponding mask
            mask = self._find_corresponding_mask(
                segmentation_results, x1, y1, x2, y2, rgb_frame.shape[:2]
            )

            # Compute rotated bounding box
            bbox_rotated = self._get_rotated_bbox(mask)

            # Get 3D pose
            x, y, z, corners_3d = self._get_3d_position(
                mask, depth_frame, camera_matrix, bbox_rotated
            )

            # Calculate orientation
            roll, pitch, yaw = self._calculate_orientation(
                corners_3d, bbox_rotated, camera_matrix, dist_coefs
            )

            # Calculate 3D bounding box if depth frame is available
            bbox_3d = None
            if depth_frame is not None:
                # Convert bbox_2d from [x, y, width, height] to [x1, y1, x2, y2]
                x_min, y_min, width, height = bbox_2d
                x_max = x_min + width
                y_max = y_min + height

                bbox = [x_min, y_min, x_max, y_max]
                # bbox = bbox_rotated[:-1]  # Use rotated bbox without angle

                from src.utils.visualization import (
                    generate_3d_box_points,
                    get_deepest_narrowest_depth,
                    get_front_direction,
                )

                # Get depth and direction for 3D box
                box_depth = get_deepest_narrowest_depth(depth_frame, bbox)
                direction = get_front_direction(depth_frame, bbox)

                # Generate 3D box points
                bbox_3d = generate_3d_box_points(bbox, box_depth, direction)

            # Create pose result
            pose_result = PoseResult(
                object_id=track_id,
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                x=float(x),
                y=float(y),
                z=float(z),
                roll=float(roll),
                pitch=float(pitch),
                yaw=float(yaw),
                bbox_2d=bbox_2d,
                bbox_rotated=bbox_rotated,
                mask=mask,
                bbox_3d=bbox_3d,
            )

            results.append(pose_result)

            # Draw pose visualization
            draw_pose(pose_vis, pose_result, camera_matrix, dist_coefs, depth_frame)

        return results

    def _find_corresponding_mask(
        self, segmentation_results, x1: int, y1: int, x2: int, y2: int, img_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Find the mask corresponding to a detection box."""
        # Default mask from bounding box
        mask = np.zeros(img_shape, dtype=np.uint8)

        # If no segmentation results, use bounding box
        if (
            not segmentation_results
            or not hasattr(segmentation_results, "masks")
            or segmentation_results.masks is None
        ):
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask

        # Calculate box center
        box_center_x = (x1 + x2) // 2
        box_center_y = (y1 + y2) // 2

        # Find closest mask
        masks = segmentation_results.masks
        best_mask_idx = None
        min_dist = float("inf")

        for j, segment_mask in enumerate(masks.data):
            mask_points = np.where(segment_mask.cpu().numpy() > 0)
            if len(mask_points[0]) == 0:
                continue

            mask_center_y = int(np.mean(mask_points[0]))
            mask_center_x = int(np.mean(mask_points[1]))

            dist = np.sqrt(
                (mask_center_x - box_center_x) ** 2 + (mask_center_y - box_center_y) ** 2
            )

            if dist < min_dist:
                min_dist = dist
                best_mask_idx = j

        # Use the best matching mask or fall back to bounding box
        if best_mask_idx is not None:
            return masks.data[best_mask_idx].cpu().numpy()
        else:
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            return mask

    def _get_rotated_bbox(self, mask: np.ndarray) -> List[float]:
        """Get the rotated bounding box for an object mask."""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [0, 0, 0, 0, 0]  # Default values

        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)

        # Get rotated rectangle
        rect = cv2.minAreaRect(largest_contour)

        # Convert to [x, y, width, height, angle]
        center, size, angle = rect
        return [center[0], center[1], size[0], size[1], angle]

    def _get_3d_position(
        self,
        mask: np.ndarray,
        depth_frame: Optional[np.ndarray],
        camera_matrix: np.ndarray,
        bbox_rotated: List[float],
    ) -> Tuple[float, float, float, np.ndarray]:
        """Calculate 3D position from depth data."""
        # Get depth at center of mask
        z = self._get_mask_center_depth(mask, depth_frame)

        # Get mask center coordinates
        mask_points = np.where(mask > 0)
        if len(mask_points[0]) == 0:
            return 0.0, 0.0, z, np.zeros((4, 3))

        mask_center_y = int(np.mean(mask_points[0]))
        mask_center_x = int(np.mean(mask_points[1]))

        # Calculate 3D coordinates
        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]

        # Convert mask center to 3D coordinates (in mm)
        x = (mask_center_x - cx) * z / fx
        y = (mask_center_y - cy) * z / fy

        # Get 3D coordinates of bounding box corners
        corners_3d = self._get_3d_bbox_corners(bbox_rotated, z, camera_matrix)

        return x, y, z, corners_3d

    def _get_mask_center_depth(self, mask: np.ndarray, depth_frame: Optional[np.ndarray]) -> float:
        """Calculate the average depth at the masked region."""
        # Default depth if no depth frame is available
        if depth_frame is None:
            return 100.0  # Default 100mm

        # Resize mask if needed to match depth frame dimensions
        if mask.shape[:2] != depth_frame.shape[:2]:
            mask = cv2.resize(mask, (depth_frame.shape[1], depth_frame.shape[0]))

        # Get depth values within the mask
        masked_depth = depth_frame[mask > 0]

        if len(masked_depth) == 0:
            return 100.0  # Default if no depth values available

        # Remove zeros and outliers
        valid_depths = masked_depth[masked_depth > 0]
        if len(valid_depths) == 0:
            return 100.0

        # Use median to be robust against outliers
        return float(np.median(valid_depths))

    def _get_3d_bbox_corners(
        self, bbox_rotated: List[float], depth: float, camera_matrix: np.ndarray
    ) -> np.ndarray:
        """Calculate 3D coordinates of the rotated bounding box corners."""
        x, y, width, height, angle = bbox_rotated

        # Get corners of the rotated rectangle
        box = cv2.boxPoints(((x, y), (width, height), angle))

        # Calculate 3D coordinates of each corner
        corners_3d = []
        for corner in box:
            fx = camera_matrix[0, 0]
            fy = camera_matrix[1, 1]
            cx = camera_matrix[0, 2]
            cy = camera_matrix[1, 2]

            # Convert pixel coordinates to 3D coordinates
            x_3d = (corner[0] - cx) * depth / fx
            y_3d = (corner[1] - cy) * depth / fy
            z_3d = depth

            corners_3d.append([x_3d, y_3d, z_3d])

        return np.array(corners_3d)

    def _calculate_orientation(
        self,
        corners_3d: np.ndarray,
        bbox_rotated: List[float],
        camera_matrix: np.ndarray,
        dist_coefs: np.ndarray,
    ) -> Tuple[float, float, float]:
        """Calculate roll, pitch, yaw angles using PnP."""
        # Define the 2D points (image points)
        x, y, width, height, angle = bbox_rotated
        box = cv2.boxPoints(((x, y), (width, height), angle))
        image_points = np.array(box, dtype=np.float32)

        # Define the 3D points in object coordinate system
        w, h = width / 2, height / 2
        object_points = np.array(
            [[-w, -h, 0], [w, -h, 0], [w, h, 0], [-w, h, 0]], dtype=np.float32
        )

        # Solve the PnP problem
        success, rvec, tvec = cv2.solvePnP(
            object_points, image_points, camera_matrix, dist_coefs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return 0.0, 0.0, 0.0

        # Convert rotation vector to rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rvec)

        # Convert to Euler angles
        pitch = np.arcsin(-rotation_matrix[2, 0])

        if np.cos(pitch) > 1e-10:
            roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        else:
            # Special case: Gimbal lock
            roll = 0
            yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

        # Convert to degrees
        roll_deg = np.degrees(roll)
        pitch_deg = np.degrees(pitch)
        yaw_deg = np.degrees(yaw)

        return roll_deg, pitch_deg, yaw_deg
