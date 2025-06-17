"""
6DoF Pose Estimation Pipeline
"""

import argparse
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from ultralytics import YOLO


class PerformanceMonitor:
    """
    Monitor and report performance metrics for pipeline stages.

    Attributes:
        stages (Dict[str, List[float]]): Dict storing execution times for each pipeline stage.
        window_size (int): Number of frames to consider for moving average calculation.
    """

    def __init__(self, window_size: int = 30) -> None:
        """
        Initialize the performance monitor.

        Args:
            window_size: Number of frames to consider for moving average FPS calculation.
        """
        self.stages = {}
        self.window_size = window_size

    def start_timer(self) -> float:
        """Start timing a stage.

        Returns:
            float: Start time in seconds.
        """
        return time.perf_counter()

    def end_timer(self, stage_name: str, start_time: float) -> float:
        """
        End timing for a stage and record the duration.

        Args:
            stage_name: Name of the pipeline stage.
            start_time: Start time returned by start_timer.

        Returns:
            float: Elapsed time in seconds.
        """
        elapsed = time.perf_counter() - start_time
        if stage_name not in self.stages:
            self.stages[stage_name] = []

        self.stages[stage_name].append(elapsed)

        # Keep only the most recent window_size measurements
        if len(self.stages[stage_name]) > self.window_size:
            self.stages[stage_name] = self.stages[stage_name][-self.window_size :]

        return elapsed

    def get_fps(self, stage_name: str) -> int:
        """
        Get the FPS for a specific stage.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            int: Frames per second for the stage, or 0 if no data is available.
        """
        if stage_name not in self.stages or not self.stages[stage_name]:
            return 0

        avg_time = np.mean(self.stages[stage_name])
        return int(1 / avg_time) if avg_time > 0 else 0

    def get_all_fps(self) -> Dict[str, int]:
        """
        Get FPS for all monitored stages.

        Returns:
            Dict[str, int]: Dictionary mapping stage names to their FPS.
        """
        return {stage: self.get_fps(stage) for stage in self.stages}

    def get_average_time(self, stage_name: str) -> float:
        """
        Get the average execution time for a specific stage in milliseconds.

        Args:
            stage_name: Name of the pipeline stage.

        Returns:
            float: Average execution time in milliseconds, or 0 if no data is available.
        """
        if stage_name not in self.stages or not self.stages[stage_name]:
            return 0.0

        avg_time = np.mean(self.stages[stage_name])
        return avg_time * 1000  # Convert to milliseconds

    def get_all_times(self) -> Dict[str, float]:
        """
        Get average execution times for all monitored stages in milliseconds.

        Returns:
            Dict[str, float]: Dictionary mapping stage names to their average execution time in ms.
        """
        return {stage: self.get_average_time(stage) for stage in self.stages}

    def reset(self) -> None:
        """Reset all performance data."""
        self.stages = {}


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


class InputSource:
    """
    Handle different input sources (image, video, webcam, RealSense).

    This class abstracts away the complexity of dealing with different
    input sources, providing a unified interface for frame acquisition.
    """

    def __init__(self, input_type: str, source_path: str) -> None:
        """
        Initialize the input source.

        Args:
            input_type: Type of input source ('image', 'video', 'webcam', or 'realsense').
            source_path: Path to source or index of webcam.
        """
        self.input_type = input_type.lower()
        self.source_path = source_path
        self.cap = None
        self.realsense_pipeline = None
        self.current_frame = None
        self.depth_frame = None

        # Validate input type
        valid_types = ["image", "video", "webcam", "realsense"]
        if self.input_type not in valid_types:
            raise ValueError(f"Input type must be one of {valid_types}")

        # Initialize the source
        self._initialize_source()

    def _initialize_source(self) -> None:
        """Initialize the appropriate input source."""
        try:
            if self.input_type == "image":
                self.current_frame = cv2.imread(self.source_path)
                if self.current_frame is None:
                    raise ValueError(f"Could not read image from {self.source_path}")

            elif self.input_type in ["video", "webcam"]:
                # If source_path is a number (webcam index), convert to int
                if self.input_type == "webcam" and self.source_path.isdigit():
                    self.source_path = int(self.source_path)

                self.cap = cv2.VideoCapture(self.source_path)
                if not self.cap.isOpened():
                    raise ValueError(f"Could not open video source: {self.source_path}")

            elif self.input_type == "realsense":
                try:
                    import pyrealsense2 as rs

                    # Initialize RealSense pipeline
                    self.realsense_pipeline = rs.pipeline()
                    config = rs.config()

                    # Enable both color and depth streams
                    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

                    # Start the pipeline
                    self.realsense_pipeline.start(config)

                except ImportError:
                    raise ImportError(
                        "pyrealsense2 library not found. Please install it to use RealSense camera."
                    )

        except Exception as e:
            raise RuntimeError(f"Error initializing input source: {str(e)}")

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the next frame from the input source.

        Returns:
            Tuple containing:
                - success (bool): Whether frame was successfully acquired
                - rgb_frame (Optional[np.ndarray]): RGB frame (None if unsuccessful)
                - depth_frame (Optional[np.ndarray]): Depth frame (None if not available)
        """
        rgb_frame = None
        depth_frame = None
        success = False

        try:
            if self.input_type == "image":
                # For image, we always return the same frame
                if self.current_frame is not None:
                    rgb_frame = self.current_frame.copy()
                    success = True
                    # Single image doesn't have depth info, so depth_frame remains None

            elif self.input_type in ["video", "webcam"]:
                # For video and webcam, read the next frame
                success, rgb_frame = self.cap.read()
                # Regular cameras don't provide depth info, depth_frame remains None

            elif self.input_type == "realsense":
                # For RealSense, get both color and depth frames
                frames = self.realsense_pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                depth_frame_rs = frames.get_depth_frame()

                if not color_frame or not depth_frame_rs:
                    return False, None, None

                # Convert frames to numpy arrays
                rgb_frame = np.asanyarray(color_frame.get_data())
                depth_frame = np.asanyarray(depth_frame_rs.get_data())
                success = True

        except Exception as e:
            print(f"Error acquiring frame: {str(e)}")
            return False, None, None

        return success, rgb_frame, depth_frame

    def release(self) -> None:
        """Release resources associated with the input source."""
        try:
            if self.cap is not None:
                self.cap.release()

            if self.realsense_pipeline is not None:
                self.realsense_pipeline.stop()
        except Exception as e:
            print(f"Error releasing resources: {str(e)}")


class PoseEstimator:
    """
    Estimate 6DoF pose using detection, tracking, segmentation, and depth.
    """

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

    def _get_rotated_bbox(self, mask: np.ndarray) -> List[float]:
        """
        Get the rotated bounding box for an object mask.

        Args:
            mask: Binary mask of the object.

        Returns:
            List containing [x, y, width, height, angle] of the rotated bounding box.
        """
        # Find contours in the mask
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return [0, 0, 0, 0, 0]  # Default values if no contour is found

        # Find the largest contour (assuming it's the main object)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the rotated rectangle
        rect = cv2.minAreaRect(largest_contour)

        # Convert to [x, y, width, height, angle]
        center, size, angle = rect
        return [center[0], center[1], size[0], size[1], angle]

    def _get_mask_center_depth(self, mask: np.ndarray, depth_frame: np.ndarray) -> float:
        """
        Calculate the average depth at the masked region.

        Args:
            mask: Binary mask of the object.
            depth_frame: Depth frame.

        Returns:
            float: Average depth value in mm.
        """
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
        """
        Calculate 3D coordinates of the rotated bounding box corners.

        Args:
            bbox_rotated: Rotated bounding box [x, y, width, height, angle].
            depth: Depth of the object center in mm.
            camera_matrix: Camera intrinsic matrix.

        Returns:
            np.ndarray: 3D coordinates of the four corners.
        """
        x, y, width, height, angle = bbox_rotated

        # Get the corners of the rotated rectangle
        box = cv2.boxPoints(((x, y), (width, height), angle))

        # Calculate 3D coordinates of each corner
        corners_3d = []
        for corner in box:
            # Convert to 3D point using depth and camera intrinsics
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

    def _calculate_pose_from_corners(
        self,
        corners_3d: np.ndarray,
        bbox_rotated: List[float],
        camera_matrix: np.ndarray,
        dist_coefs: np.ndarray,
    ) -> Tuple[float, float, float]:
        """
        Calculate roll, pitch, yaw angles from 3D corners using PnP.

        Args:
            corners_3d: 3D coordinates of the bounding box corners.
            bbox_rotated: Rotated bounding box [x, y, width, height, angle].
            camera_matrix: Camera intrinsic matrix.
            dist_coefs: Distortion coefficients.

        Returns:
            Tuple containing roll, pitch, yaw angles in degrees.
        """
        # Define the 2D points (image points)
        x, y, width, height, angle = bbox_rotated
        box = cv2.boxPoints(((x, y), (width, height), angle))
        image_points = np.array(box, dtype=np.float32)

        # Define the 3D points in object coordinate system
        # Assuming a rectangular prism with a flat bottom face
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

        # Convert rotation matrix to Euler angles (roll, pitch, yaw)
        # Using the convention where Roll is rotation around X,
        # Pitch is rotation around Y, and Yaw is rotation around Z
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

    def frame_processor(
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
        results = []
        visualization = {}

        # 1. Run tracking
        start_time = self.performance_monitor.start_timer()
        tracking_results = self.tracking_model.track(
            rgb_frame,
            persist=True,
            conf=detection_params.get("conf", 0.5),
            iou=detection_params.get("iou", 0.5),
        )
        self.performance_monitor.end_timer("tracking", start_time)

        # Create tracking visualization
        tracking_vis = rgb_frame.copy()
        if tracking_results and len(tracking_results) > 0:
            tracking_vis = tracking_results[0].plot()
            visualization["tracking_vis"] = tracking_vis
        else:
            visualization["tracking_vis"] = tracking_vis

        # 2. Run segmentation
        start_time = self.performance_monitor.start_timer()
        segmentation_results = self.segmentation_model(
            rgb_frame,
            conf=segmentation_params.get("conf", 0.5),
            iou=segmentation_params.get("iou", 0.5),
        )
        self.performance_monitor.end_timer("segmentation", start_time)

        # Create segmentation visualization
        segmentation_vis = rgb_frame.copy()
        if segmentation_results and len(segmentation_results) > 0:
            segmentation_vis = segmentation_results[0].plot()
            visualization["segmentation_vis"] = segmentation_vis
        else:
            visualization["segmentation_vis"] = segmentation_vis

        # Create poses visualization
        pose_vis = rgb_frame.copy()

        # Prepare depth visualization if available
        if depth_frame is not None:
            # Normalize depth for visualization
            depth_norm = cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            depth_vis = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
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

        visualization["depth_vis"] = depth_vis

        # 3. Process each tracked object
        start_time = self.performance_monitor.start_timer()

        # Check if we have valid tracking results
        if (
            tracking_results
            and len(tracking_results) > 0
            and hasattr(tracking_results[0], "boxes")
        ):
            # Get boxes and track IDs
            boxes = tracking_results[0].boxes

            # Process each tracked box
            for i, box in enumerate(boxes):
                # Skip if this box has no tracking ID
                if not hasattr(box, "id") or box.id is None:
                    continue

                # Get basic detection info
                track_id = int(box.id.item())
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())

                # Try to get class name
                class_name = (
                    tracking_results[0].names[class_id]
                    if hasattr(tracking_results[0], "names")
                    else f"class_{class_id}"
                )

                # Get the bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy.squeeze().tolist())
                bbox_2d = [x1, y1, x2 - x1, y2 - y1]  # [x, y, width, height]

                # Find corresponding mask from segmentation results
                mask = None
                if (
                    segmentation_results
                    and len(segmentation_results) > 0
                    and hasattr(segmentation_results[0], "masks")
                ):
                    masks = segmentation_results[0].masks

                    if masks is not None:
                        # Convert segment to mask
                        box_center_x = (x1 + x2) // 2
                        box_center_y = (y1 + y2) // 2

                        # Find closest mask to this bounding box center
                        best_mask_idx = None
                        min_dist = float("inf")

                        for j, segment_mask in enumerate(masks.data):
                            # Get mask center
                            mask_points = np.where(segment_mask.cpu().numpy() > 0)
                            if len(mask_points[0]) == 0:
                                continue

                            mask_center_y = int(np.mean(mask_points[0]))
                            mask_center_x = int(np.mean(mask_points[1]))

                            # Calculate distance to box center
                            dist = np.sqrt(
                                (mask_center_x - box_center_x) ** 2
                                + (mask_center_y - box_center_y) ** 2
                            )

                            if dist < min_dist:
                                min_dist = dist
                                best_mask_idx = j

                        if best_mask_idx is not None:
                            mask = masks.data[best_mask_idx].cpu().numpy()

                # If no mask was found, create one from the bounding box
                if mask is None:
                    mask = np.zeros(rgb_frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

                # Compute rotated bounding box
                bbox_rotated = self._get_rotated_bbox(mask)

                # Get depth at center of mask
                if depth_frame is not None:
                    z = self._get_mask_center_depth(mask, depth_frame)
                else:
                    z = 100.0  # Default depth (100mm)

                # Get 3D coordinates of center
                mask_points = np.where(mask > 0)
                if len(mask_points[0]) == 0:
                    continue

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

                # Get 3D coordinates of the bounding box corners
                corners_3d = self._get_3d_bbox_corners(bbox_rotated, z, camera_matrix)

                # Calculate roll, pitch, yaw using PnP
                roll, pitch, yaw = self._calculate_pose_from_corners(
                    corners_3d, bbox_rotated, camera_matrix, dist_coefs
                )

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
                )

                results.append(pose_result)

                # Draw pose visualization
                self._draw_pose(pose_vis, pose_result, camera_matrix, dist_coefs)

        self.performance_monitor.end_timer("pose_estimation", start_time)

        # Don't add FPS info to individual visualizations, we'll add it in the grid
        start_time = self.performance_monitor.start_timer()

        # Store performance info in the visualization dictionary for use in create_visualization_grid
        visualization["fps_info"] = self.performance_monitor.get_all_fps()
        visualization["time_info"] = self.performance_monitor.get_all_times()

        visualization["pose_vis"] = pose_vis
        self.performance_monitor.end_timer("visualization", start_time)

        return results, visualization

    def _draw_pose(
        self,
        image: np.ndarray,
        pose_result: PoseResult,
        camera_matrix: np.ndarray,
        dist_coefs: np.ndarray,
    ) -> None:
        """
        Draw pose axes and rotated bounding box on image.

        Args:
            image: Image to draw on.
            pose_result: Pose estimation result.
            camera_matrix: Camera intrinsic matrix.
            dist_coefs: Distortion coefficients.
        """
        # Draw the rotated bounding box
        x, y, w, h, angle = pose_result.bbox_rotated
        box = cv2.boxPoints(((x, y), (w, h), angle))
        box = np.intp(box)
        cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

        # Draw the axis
        axis_length = min(w, h) / 2

        # Create rotation matrix
        rvec = np.array(
            [
                np.radians(pose_result.pitch),
                np.radians(pose_result.yaw),
                np.radians(pose_result.roll),
            ],
            dtype=np.float32,
        )

        tvec = np.array([pose_result.x, pose_result.y, pose_result.z], dtype=np.float32)

        # Draw coordinate axes
        axis_points = np.float32(
            [
                [0, 0, 0],
                [axis_length, 0, 0],  # X-axis (red)
                [0, axis_length, 0],  # Y-axis (green)
                [0, 0, axis_length],  # Z-axis (blue)
            ]
        )

        # Project 3D points to image plane
        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coefs)
        imgpts = np.int32(imgpts).reshape(-1, 2)

        # Draw axes lines
        origin = tuple(imgpts[0])
        cv2.line(image, origin, tuple(imgpts[1]), (0, 0, 255), 3)  # X-axis: red
        cv2.line(image, origin, tuple(imgpts[2]), (0, 255, 0), 3)  # Y-axis: green
        cv2.line(image, origin, tuple(imgpts[3]), (255, 0, 0), 3)  # Z-axis: blue

        # Draw object ID and pose information
        text_info = f"ID:{pose_result.object_id} {pose_result.class_name}"
        pose_info = f"x:{pose_result.x:.0f} y:{pose_result.y:.0f} z:{pose_result.z:.0f}"
        angle_info = f"r:{pose_result.roll:.1f} p:{pose_result.pitch:.1f} y:{pose_result.yaw:.1f}"

        cv2.putText(
            image,
            text_info,
            (int(x), int(y) - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            image,
            pose_info,
            (int(x), int(y) - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            2,
        )
        cv2.putText(
            image, angle_info, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
        )


def create_visualization_grid(visualizations: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Create a 2x2 grid of visualization frames.

    Args:
        visualizations: Dictionary of visualization frames.

    Returns:
        np.ndarray: Grid image combining all visualizations.
    """
    # Check that we have all required visualizations
    required = ["tracking_vis", "segmentation_vis", "pose_vis", "depth_vis"]
    if not all(key in visualizations for key in required):
        missing = [key for key in required if key not in visualizations]
        raise ValueError(f"Missing required visualizations: {missing}")

    # Ensure all frames have the same size
    h, w = visualizations["tracking_vis"].shape[:2]
    resize_frames = {}

    for name, frame in visualizations.items():
        if name in required and frame.shape[:2] != (h, w):
            resize_frames[name] = cv2.resize(frame, (w, h))
        elif name in required:
            resize_frames[name] = frame

    # Create the 2x2 grid
    top_row = np.hstack((resize_frames["tracking_vis"], resize_frames["segmentation_vis"]))
    bottom_row = np.hstack((resize_frames["pose_vis"], resize_frames["depth_vis"]))
    grid = np.vstack((top_row, bottom_row))

    # Add labels to each quadrant
    labels = ["Tracking", "Segmentation", "6D Pose", "Depth"]
    positions = [(10, 30), (w + 10, 30), (10, h + 30), (w + 10, h + 30)]

    for label, pos in zip(labels, positions):
        cv2.putText(grid, label, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Add compact performance metrics table at the bottom
    if "fps_info" in visualizations:
        fps_info = visualizations["fps_info"]
        time_info = visualizations.get("time_info", {})

        # Make a more compact table
        padding = 5
        font_scale = 0.4
        line_thickness = 1
        row_height = 15

        # Calculate dimensions
        total_width = w * 2  # Full width of the grid

        # Sort stages by processing order with a predefined order
        stage_order = ["tracking", "segmentation", "pose_estimation", "visualization"]
        # Add any remaining stages that aren't in the predefined order
        sorted_stages = [s for s in stage_order if s in fps_info]
        remaining_stages = [s for s in fps_info if s not in sorted_stages]
        sorted_stages.extend(sorted(remaining_stages))

        # Calculate table dimensions
        num_stages = len(sorted_stages)
        table_height = row_height * (num_stages + 1)  # Header + data rows
        table_width = total_width // 2  # Half width for more compact look
        table_x = total_width - table_width - padding  # Right-aligned
        table_y = h * 2 - table_height - padding  # Bottom-aligned

        # Draw semi-transparent background
        overlay = grid.copy()
        cv2.rectangle(
            overlay,
            (table_x - padding, table_y - padding),
            (total_width - padding, h * 2 - padding),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.6, grid, 0.4, 0, grid)

        # Define column widths
        col1_width = table_width * 0.5  # Stage name
        col2_width = table_width * 0.25  # FPS
        col3_width = table_width * 0.25  # Time

        # Draw compact header
        cv2.putText(
            grid,
            "Stage",
            (table_x, table_y + row_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (200, 200, 200),
            line_thickness,
        )
        cv2.putText(
            grid,
            "FPS",
            (int(table_x + col1_width), table_y + row_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (200, 200, 200),
            line_thickness,
        )
        cv2.putText(
            grid,
            "ms",
            (int(table_x + col1_width + col2_width), table_y + row_height - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (200, 200, 200),
            line_thickness,
        )

        # Draw horizontal line under header
        cv2.line(
            grid,
            (table_x - padding, table_y + row_height + 2),
            (total_width - padding, table_y + row_height + 2),
            (150, 150, 150),
            1,
        )

        # Draw data rows
        for i, stage in enumerate(sorted_stages):
            fps = fps_info[stage]
            y_pos = table_y + (i + 2) * row_height

            # Abbreviate stage names to save space
            if stage == "tracking":
                display_name = "Track"
            elif stage == "segmentation":
                display_name = "Seg"
            elif stage == "pose_estimation":
                display_name = "Pose"
            elif stage == "visualization":
                display_name = "Vis"
            else:
                # For other stages, use first 4 chars
                display_name = stage[:4].capitalize()

            # Stage name
            cv2.putText(
                grid,
                display_name,
                (table_x, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness,
            )

            # FPS value with color coding
            fps_color = (0, 255, 0) if fps > 30 else (0, 165, 255) if fps > 15 else (0, 0, 255)
            cv2.putText(
                grid,
                f"{fps}",
                (int(table_x + col1_width), y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                fps_color,
                line_thickness,
            )

            # Time in milliseconds
            time_ms = time_info.get(stage, 0)
            cv2.putText(
                grid,
                f"{time_ms:.1f}",
                (int(table_x + col1_width + col2_width), y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                line_thickness,
            )

    return grid


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="6DoF Pose Estimation Pipeline")

    parser.add_argument(
        "--input",
        type=str,
        choices=["image", "video", "webcam", "realsense"],
        default="webcam",
        help="Input source type",
    )

    parser.add_argument(
        "--source", type=str, default="0", help="Path to input source or webcam index"
    )

    parser.add_argument(
        "--tracking_model",
        type=str,
        default="src/checkpoints/YOLO/yolo11n.pt",
        help="Path to YOLO tracking model",
    )

    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="src/checkpoints/YOLO/yolo11n-seg.pt",
        help="Path to YOLO segmentation model",
    )

    parser.add_argument(
        "--detection_conf", type=float, default=0.5, help="Confidence threshold for detection"
    )

    parser.add_argument(
        "--detection_iou", type=float, default=0.5, help="IoU threshold for detection"
    )

    parser.add_argument(
        "--segmentation_conf",
        type=float,
        default=0.5,
        help="Confidence threshold for segmentation",
    )

    parser.add_argument(
        "--segmentation_iou", type=float, default=0.5, help="IoU threshold for segmentation"
    )

    parser.add_argument("--output", type=str, default=None, help="Path to save output video")

    return parser.parse_args()


def get_camera_parameters() -> Tuple[np.ndarray, np.ndarray]:
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


def main() -> None:
    """
    Main function for the 6DoF pose estimation pipeline.
    """
    # Parse arguments
    args = parse_arguments()

    # Get camera parameters
    camera_matrix, dist_coefs = get_camera_parameters()

    # Initialize input source
    input_source = InputSource(args.input, args.source)

    # Initialize pose estimator
    pose_estimator = PoseEstimator(args.tracking_model, args.segmentation_model)

    # Set up output video writer if specified
    output_writer = None
    if args.output:
        # Get first frame to determine size
        _, first_frame, _ = input_source.get_frame()
        if first_frame is not None:
            h, w = first_frame.shape[:2]
            output_writer = cv2.VideoWriter(
                args.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,  # fps
                (w * 2, h * 2),  # Grid size is 2x2 of original frames
            )

    try:
        # Process frames
        while True:
            # Get next frame
            success, rgb_frame, depth_frame = input_source.get_frame()

            if not success or rgb_frame is None:
                print("Failed to get frame. Exiting.")
                break

            # Set up parameters
            detection_params = {"conf": args.detection_conf, "iou": args.detection_iou}

            segmentation_params = {"conf": args.segmentation_conf, "iou": args.segmentation_iou}

            # Process the frame
            results, visualizations = pose_estimator.frame_processor(
                rgb_frame,
                camera_matrix,
                dist_coefs,
                depth_frame,
                detection_params,
                segmentation_params,
            )

            # Create visualization grid
            vis_grid = create_visualization_grid(visualizations)

            # Display results
            cv2.imshow("6DoF Pose Estimation", vis_grid)

            # Write to output if specified
            if output_writer is not None:
                output_writer.write(vis_grid)

            # Print pose results to console
            for result in results:
                print(
                    f"Object {result.object_id} ({result.class_name}): "
                    f"Position: ({result.x:.1f}, {result.y:.1f}, {result.z:.1f}) mm, "
                    f"Rotation: ({result.roll:.1f}, {result.pitch:.1f}, {result.yaw:.1f}) degrees"
                )

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Clean up
        input_source.release()
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
