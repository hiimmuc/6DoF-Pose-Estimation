#!/usr/bin/env python3
# filepath: /home/namdp/Workspace/VINMOTION/6DoF-PoseEstimation/examples/6dof_pipeline.py

"""
A comprehensive pipeline for 6DoF pose estimation:
- Take input from image, video, webcam, or RealSense camera
- Run YOLO tracking to get tracking results
- Run YOLO segmentation to get mask results
- Display tracking result and mask result side by side
- Get rotating box from the bounding box of mask results
- Compute 3D coordinates using camera matrix
- Output x, y, z, roll, pitch, yaw values and bounding box coordinates
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from ultralytics import YOLO

# Try to import pyrealsense2 for RealSense support
try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("pyrealsense2 not available. RealSense input will not work.")


def parse_arguments():
    """Parse command line arguments"""
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
        default=[0, 41, 67],  # Default class 0 (person)
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
    return parser.parse_args()


class InputSource:
    """Base class for different input sources"""

    def __init__(self):
        self.fps = 30  # Default FPS

    def get_frame(self):
        """Get the next frame from input source"""
        raise NotImplementedError

    def release(self):
        """Release resources"""
        pass

    def get_camera_params(self):
        """Return camera matrix and distortion coefficients"""
        # Default camera parameters (can be overridden by child classes)
        fx, fy = 800, 800  # Focal lengths
        cx, cy = 320, 240  # Principal point

        # Generic camera matrix for a 640x480 camera
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # No distortion
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        return camera_matrix, dist_coeffs


class ImageSource(InputSource):
    """Input source for single images"""

    def __init__(self, image_path):
        super().__init__()
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")
        self.returned = False

    def get_frame(self):
        # Return the image only once
        if not self.returned:
            self.returned = True
            return True, self.image
        return False, None


class VideoSource(InputSource):
    """Input source for video files"""

    def __init__(self, video_path):
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class WebcamSource(InputSource):
    """Input source for webcam"""

    def __init__(self, cam_id=0):
        super().__init__()
        self.cap = cv2.VideoCapture(int(cam_id))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open webcam: {cam_id}")

    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class RealSenseSource(InputSource):
    """Input source for Intel RealSense cameras"""

    def __init__(self):
        super().__init__()
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not available")

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Get depth scale for depth calculations
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Get camera intrinsics
        self.color_profile = self.profile.get_stream(rs.stream.color)
        self.depth_profile = self.profile.get_stream(rs.stream.depth)
        self.color_intrinsics = self.color_profile.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = self.depth_profile.as_video_stream_profile().get_intrinsics()

        # Align depth to color frame
        self.align = rs.align(rs.stream.color)

        self.fps = 30

    def get_frame(self):
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return False, None, None

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return True, color_image, depth_image
        except RuntimeError:
            return False, None, None

    def release(self):
        self.pipeline.stop()

    def get_camera_params(self):
        """Return RealSense camera parameters"""
        camera_matrix = np.array(
            [
                [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # RealSense distortion coefficients
        dist_coeffs = np.array(
            [
                self.color_intrinsics.coeffs[0],  # k1
                self.color_intrinsics.coeffs[1],  # k2
                self.color_intrinsics.coeffs[2],  # p1
                self.color_intrinsics.coeffs[3],  # p2
                self.color_intrinsics.coeffs[4],  # k3
            ],
            dtype=np.float32,
        )

        return camera_matrix, dist_coeffs

    def get_point_3d(self, x: int, y: int, depth_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Get 3D coordinates of a point in the camera coordinate system

        Args:
            x: X coordinate in the image
            y: Y coordinate in the image
            depth_image: Depth image

        Returns:
            Tuple of (X, Y, Z) coordinates in the camera coordinate system (meters)
        """
        # Get depth value at the given pixel
        depth = depth_image[y, x].astype(float) * self.depth_scale  # Convert to meters

        # Deproject from pixel to 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [x, y], depth)

        return tuple(point_3d)


def create_input_source(input_type: str, input_path: str = None) -> InputSource:
    """Factory function to create input sources"""

    if input_type == "image":
        if not input_path:
            raise ValueError("Image path required for image input")
        return ImageSource(input_path)

    elif input_type == "video":
        if not input_path:
            raise ValueError("Video path required for video input")
        return VideoSource(input_path)

    elif input_type == "webcam":
        cam_id = 0 if not input_path else input_path
        return WebcamSource(cam_id)

    elif input_type == "realsense":
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not available")
        return RealSenseSource()

    else:
        raise ValueError(f"Unknown input type: {input_type}")


def get_rotating_bounding_box(mask: np.ndarray) -> Dict[str, Any]:
    """
    Get minimum rotated bounding box from mask

    Args:
        mask: Binary mask image

    Returns:
        Dictionary with box info (center, size, angle, corners)
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


def compute_3d_pose(
    box_info: Dict[str, Any], camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate 6DOF pose from rotated bounding box

    Args:
        box_info: Rotated bounding box information
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients

    Returns:
        Tuple of (rotation vector, translation vector)
    """
    # Get 2D image points from box corners
    image_points = box_info["corners"].astype(np.float32)

    # Create a simple 3D box model (assume the object is flat on a table)
    width, height = box_info["size"]
    # The object model is a flat rectangle with z=0 for all points
    object_points = np.array(
        [
            [-width / 2, -height / 2, 0],
            [width / 2, -height / 2, 0],
            [width / 2, height / 2, 0],
            [-width / 2, height / 2, 0],
        ],
        dtype=np.float32,
    )

    # Solve PnP problem
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    if not success:
        return None, None

    return rvec, tvec


def get_euler_angles(rvec: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation vector to Euler angles (roll, pitch, yaw)

    Args:
        rvec: Rotation vector

    Returns:
        Roll, pitch, yaw angles in degrees
    """
    rmat, _ = cv2.Rodrigues(rvec)
    r = R.from_matrix(rmat)
    angles = r.as_euler("xyz", degrees=True)

    return angles[0], angles[1], angles[2]  # roll, pitch, yaw


def create_3d_bbox(
    box_info: Dict[str, Any],
    rvec: np.ndarray,
    depth: Optional[float] = None,
    use_realsense: bool = False,
    center_depth: Optional[float] = None,
    edge_depth: Optional[float] = None,
    max_height: float = 50.0,
    default_height: float = 15.0,  # Default of 15 units for regular cameras
) -> np.ndarray:
    """
    Create 3D bounding box coordinates based on 2D box information,
    aligned with the object's orientation (roll, pitch, yaw).

    The box is created with the following characteristics:
    1. Width and length match the 2D detected object's dimensions
    2. Box is rotated 90 degrees around the pitch axis to make the height vertical
    3. For RealSense cameras: depth/thickness calculated from depth difference
       between center and edge pixels (with min/max limits)
    4. For regular cameras: default thickness of 15 units

    The resulting 3D box will have:
    - Width and height matching the 2D bounding box dimensions
    - Depth calculated based on camera type (RealSense vs. regular)
    - Orientation aligned with the object's detected orientation

    Args:
        box_info: Dictionary containing rotated bounding box information
        rvec: Rotation vector for alignment with object orientation
        depth: Base depth value for regular cameras (ignored for RealSense)
        use_realsense: Whether using RealSense depth data
        center_depth: Depth at the center of the object (for RealSense)
        edge_depth: Depth at the edge of the object (for RealSense)
        max_height: Maximum height for the 3D bounding box
        default_height: Default height for regular cameras (15 units)

    Returns:
        8x3 array of 3D bounding box coordinates (corners)
    """
    # Get 2D box dimensions - these will be used for width and height to match the 2D box
    width, height = box_info["size"]

    # Calculate the depth/thickness of the 3D box
    if use_realsense and center_depth is not None and edge_depth is not None:
        # Calculate depth based on the difference between center and edge depths (for RealSense)
        box_depth = abs(center_depth - edge_depth)
        box_depth = min(box_depth, max_height)  # Limit maximum depth
        # Use at least a minimum depth if the depth difference is too small
        box_depth = max(box_depth, default_height / 5)
    else:
        # Use default depth for regular cameras
        box_depth = default_height

    # Create 3D box with specific properties:
    # 1. Box is oriented with its height along the Y-axis (vertical orientation)
    # 2. This effectively creates a box rotated 90 degrees around the pitch axis
    # 3. The box width and height match exactly the 2D bounding box dimensions
    # 4. The depth/thickness is determined by:
    #    - For RealSense: difference between center and edge depth measurements
    #    - For regular cameras: default thickness of 15 units
    #
    # Box coordinate system (before object orientation is applied):
    # - X-axis: width of the object (matches 2D width)
    # - Y-axis: depth/thickness of the object (vertical axis)
    # - Z-axis: height of the object (matches 2D height)
    box_points = np.array(
        [
            # Bottom face (bottom = lower Y value)
            [-width / 2, -box_depth / 2, -height / 2],  # bottom front left
            [width / 2, -box_depth / 2, -height / 2],  # bottom front right
            [width / 2, -box_depth / 2, height / 2],  # bottom back right
            [-width / 2, -box_depth / 2, height / 2],  # bottom back left
            # Top face (top = higher Y value)
            [-width / 2, box_depth / 2, -height / 2],  # top front left
            [width / 2, box_depth / 2, -height / 2],  # top front right
            [width / 2, box_depth / 2, height / 2],  # top back right
            [-width / 2, box_depth / 2, height / 2],  # top back left
        ],
        dtype=np.float32,
    )

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvec)

    # Apply rotation to align the box with the object's orientation
    # This preserves the vertical orientation while aligning with object rotation
    rotated_bbox = []
    for point in box_points:
        # Apply rotation from object's orientation
        rotated_point = np.dot(rmat, point)
        rotated_bbox.append(rotated_point)

    # Note: The vertical orientation is maintained because:
    # 1. We created the box with the Y-axis as the vertical axis
    # 2. The rotation matrix from rvec already includes the object's orientation
    # 3. So the resulting box is both vertically oriented and aligned with object rotation

    return np.array(rotated_bbox, dtype=np.float32)


def draw_3d_axes(
    img: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    axis_length: float = 0.1,  # Length of axes in meters
    thickness: int = 3,
) -> np.ndarray:
    """
    Draw 3D coordinate axes to visualize object position and orientation

    Args:
        img: Input image
        rvec: Rotation vector
        tvec: Translation vector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        axis_length: Length of each axis in meters
        thickness: Line thickness

    Returns:
        Image with 3D axes drawn
    """
    result = img.copy()

    # Define coordinate axes in 3D space
    origin = np.array([[0, 0, 0]], dtype=np.float32)

    # Define the endpoints of the three axes
    x_axis = np.array([[axis_length, 0, 0]], dtype=np.float32)
    y_axis = np.array([[0, axis_length, 0]], dtype=np.float32)
    z_axis = np.array([[0, 0, axis_length]], dtype=np.float32)

    # Project the 3D points to the image plane
    origin_2d, _ = cv2.projectPoints(origin, rvec, tvec, camera_matrix, dist_coeffs)
    x_axis_2d, _ = cv2.projectPoints(x_axis, rvec, tvec, camera_matrix, dist_coeffs)
    y_axis_2d, _ = cv2.projectPoints(y_axis, rvec, tvec, camera_matrix, dist_coeffs)
    z_axis_2d, _ = cv2.projectPoints(z_axis, rvec, tvec, camera_matrix, dist_coeffs)

    # Convert to integer pixel coordinates
    origin_2d = tuple(map(int, origin_2d[0][0]))
    x_axis_2d = tuple(map(int, x_axis_2d[0][0]))
    y_axis_2d = tuple(map(int, y_axis_2d[0][0]))
    z_axis_2d = tuple(map(int, z_axis_2d[0][0]))

    # Draw the axes
    cv2.line(result, origin_2d, x_axis_2d, (0, 0, 255), thickness)  # X-axis in red
    cv2.line(result, origin_2d, y_axis_2d, (0, 255, 0), thickness)  # Y-axis in green
    cv2.line(result, origin_2d, z_axis_2d, (255, 0, 0), thickness)  # Z-axis in blue

    # Add text labels for the axes
    cv2.putText(result, "X", x_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(result, "Y", y_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(result, "Z", z_axis_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return result


def draw_3d_bbox(
    img: np.ndarray,
    bbox_3d: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw 3D bounding box on image

    Args:
        img: Input image
        bbox_3d: 8x3 array of 3D bounding box coordinates
        rvec: Rotation vector
        tvec: Translation vector
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        color: Line color (default: yellow)
        thickness: Line thickness

    Returns:
        Image with 3D bounding box drawn
    """
    result = img.copy()

    # Project 3D points to image plane
    bbox_2d, _ = cv2.projectPoints(bbox_3d, rvec, tvec, camera_matrix, dist_coeffs)

    # Convert to integer points
    bbox_2d = bbox_2d.reshape(-1, 2).astype(np.int32)

    # Draw bottom face
    for i in range(4):
        cv2.line(result, tuple(bbox_2d[i]), tuple(bbox_2d[(i + 1) % 4]), color, thickness)

    # Draw top face
    for i in range(4):
        cv2.line(
            result, tuple(bbox_2d[i + 4]), tuple(bbox_2d[((i + 1) % 4) + 4]), color, thickness
        )

    # Draw connecting lines between top and bottom faces
    for i in range(4):
        cv2.line(result, tuple(bbox_2d[i]), tuple(bbox_2d[i + 4]), color, thickness)

    return result


def draw_rotating_box(
    img: np.ndarray,
    box_points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw rotating bounding box on an image

    Args:
        img: Input image
        box_points: 4 corner points of the box
        color: Line color
        thickness: Line thickness

    Returns:
        Image with bounding box drawn
    """
    result = img.copy()

    # Draw the rotated bounding box
    cv2.drawContours(result, [box_points], 0, color, thickness)

    # Draw the center point
    center_x = int(np.mean(box_points[:, 0]))
    center_y = int(np.mean(box_points[:, 1]))
    cv2.circle(result, (center_x, center_y), 5, color, -1)

    return result


def create_grid_display(
    tracking_img: np.ndarray,
    segmentation_img: np.ndarray,
    pose_img: np.ndarray,
    depth_img: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Create a 2x2 grid display showing the complete 6DoF pose estimation pipeline.

    This function arranges the four views in a 2x2 grid:
    - Top-left: Tracking view (object detection and tracking)
    - Top-right: Segmentation view (object instance segmentation)
    - Bottom-left: 3D pose view (with 3D bounding boxes and coordinate axes)
    - Bottom-right: Depth view (for RealSense cameras) or black frame (for regular cameras)

    Each quadrant is labeled for easy identification. All images are resized to match
    dimensions for consistent display.

    Args:
        tracking_img: Image with tracking visualization (object detection + tracking)
        segmentation_img: Image with segmentation visualization (instance segmentation)
        pose_img: Image with 3D pose visualization (3D bounding boxes + axes)
        depth_img: Depth image visualization (provided for RealSense, None for regular cameras)

    Returns:
        2x2 grid combined image with all views and labels
    """
    # Ensure all images are the same size
    h1, w1 = tracking_img.shape[:2]

    # Use the dimensions of the tracking image as reference
    target_h, target_w = h1, w1

    # Resize all images to match the same dimensions
    if tracking_img.shape[:2] != (target_h, target_w):
        tracking_img = cv2.resize(tracking_img, (target_w, target_h))

    if segmentation_img.shape[:2] != (target_h, target_w):
        segmentation_img = cv2.resize(segmentation_img, (target_w, target_h))

    if pose_img.shape[:2] != (target_h, target_w):
        pose_img = cv2.resize(pose_img, (target_w, target_h))

    # Create a black frame for depth if not provided
    if depth_img is None:
        depth_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    elif depth_img.shape[:2] != (target_h, target_w):
        depth_img = cv2.resize(depth_img, (target_w, target_h))

    # Colorize depth image if it's single-channel
    if len(depth_img.shape) == 2 or depth_img.shape[2] == 1:
        # Normalize and apply colormap
        depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_img = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

    # Create the top and bottom rows
    top_row = np.hstack((tracking_img, segmentation_img))
    bottom_row = np.hstack((pose_img, depth_img))

    # Stack the rows vertically
    grid = np.vstack((top_row, bottom_row))

    # Add labels
    margin = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    font_thickness = 2

    # Draw labels with a background box for better visibility
    labels = ["Tracking", "Segmentation", "3D Pose", "Depth"]
    positions = [
        (margin, 25),  # Top-left
        (target_w + margin, 25),  # Top-right
        (margin, target_h + 25),  # Bottom-left
        (target_w + margin, target_h + 25),  # Bottom-right
    ]

    for label, pos in zip(labels, positions):
        # Black background for text
        text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
        cv2.rectangle(
            grid,
            (pos[0] - margin, pos[1] - text_size[1] - margin),
            (pos[0] + text_size[0] + margin, pos[1] + margin),
            (0, 0, 0),
            -1,
        )
        # White text
        cv2.putText(grid, label, pos, font, font_scale, font_color, font_thickness)

    return grid


def create_side_by_side_display(
    tracking_img: np.ndarray, segmentation_img: np.ndarray
) -> np.ndarray:
    """
    Create a side-by-side display of tracking and segmentation results

    Args:
        tracking_img: Image with tracking visualization
        segmentation_img: Image with segmentation visualization

    Returns:
        Combined side-by-side image
    """
    # Resize images if they have different dimensions
    h1, w1 = tracking_img.shape[:2]
    h2, w2 = segmentation_img.shape[:2]

    if h1 != h2 or w1 != w2:
        # Use the larger dimensions for both images
        h = max(h1, h2)
        w = max(w1, w2)

        # Resize images to match
        if h1 != h or w1 != w:
            tracking_img = cv2.resize(tracking_img, (w, h))
        if h2 != h or w2 != w:
            segmentation_img = cv2.resize(segmentation_img, (w, h))

    # Create the side-by-side display
    combined = np.hstack((tracking_img, segmentation_img))

    # Add labels
    cv2.putText(combined, "Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(
        combined, "Segmentation", (w1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
    )

    return combined


def get_robust_depth(
    input_source: RealSenseSource, x: int, y: int, depth_frame: np.ndarray, kernel_size: int = 5
) -> float:
    """
    Get a robust depth estimate by averaging depth values in a small kernel around the point.

    This helps reduce noise in depth measurements from the RealSense camera.

    Args:
        input_source: RealSenseSource instance for 3D point calculation
        x: X coordinate in the image
        y: Y coordinate in the image
        depth_frame: Depth image from RealSense
        kernel_size: Size of the kernel for averaging (default: 5x5)

    Returns:
        Average depth value in meters
    """
    # Ensure kernel size is odd
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1

    # Calculate half kernel size
    half_k = kernel_size // 2

    # Get image dimensions
    h, w = depth_frame.shape[:2]

    # Initialize depth accumulator
    depth_sum = 0.0
    valid_points = 0

    # Iterate over kernel area
    for ky in range(max(0, y - half_k), min(h, y + half_k + 1)):
        for kx in range(max(0, x - half_k), min(w, x + half_k + 1)):
            # Get 3D point
            _, _, z = input_source.get_point_3d(kx, ky, depth_frame)

            # Skip invalid depth values (0 or very large values often indicate invalid measurements)
            if 0.01 < z < 10.0:  # Valid depth range in meters
                depth_sum += z
                valid_points += 1

    # Return average depth or fallback to direct measurement if no valid points found
    if valid_points > 0:
        return depth_sum / valid_points
    else:
        _, _, z = input_source.get_point_3d(x, y, depth_frame)
        return z


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
        self.fps_history = []
        self.stage_times = {
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
        fps = self.get_fps()
        tracking_time = self.get_stage_time("tracking") * 1000
        segmentation_time = self.get_stage_time("segmentation") * 1000
        pose_time = self.get_stage_time("pose_estimation") * 1000
        viz_time = self.get_stage_time("visualization") * 1000

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


def main():
    """
    Main function for the 6DoF Pose Estimation Pipeline

    This pipeline performs the following steps:
    1. Initializes input source (image, video, webcam, or RealSense)
    2. Runs YOLO tracking and segmentation models
    3. For each detected object:
       - Extracts rotated bounding box from segmentation mask
       - Computes 6DoF pose using PnP algorithm
       - Creates 3D bounding box with dimensions matching 2D detection
       - For RealSense: Uses depth data to calculate box thickness
       - For regular cameras: Uses default thickness of 15 units
    4. Displays results in a 2x2 grid:
       - Tracking view (top-left)
       - Segmentation view (top-right)
       - 3D pose visualization (bottom-left)
       - Depth view (bottom-right) or black frame for non-RealSense
    """
    args = parse_arguments()

    # Create performance monitor
    perf_monitor = PerfMonitor()

    # Load YOLO models
    print("Loading YOLO tracking model...")
    tracking_model = YOLO(args.tracking_model)

    print("Loading YOLO segmentation model...")
    segmentation_model = YOLO(args.segmentation_model)

    # Create input source
    try:
        input_source = create_input_source(args.input, args.source)
    except (ValueError, ImportError) as e:
        print(f"Error creating input source: {str(e)}")
        sys.exit(1)

    # Get camera parameters
    camera_matrix, dist_coeffs = input_source.get_camera_params()
    print(f"Camera matrix:\n{camera_matrix}")

    # Initialize performance monitor
    perf_monitor = PerfMonitor(history_size=30)

    try:
        if args.input == "realsense":
            # RealSense specific loop
            while True:
                # Start frame timing
                perf_monitor.start_frame()

                success, color_frame, depth_frame = input_source.get_frame()
                if not success:
                    print("Failed to get frame from RealSense camera")
                    break

                # Run YOLO tracking
                tracking_results = tracking_model.track(
                    color_frame, persist=True, conf=args.conf, iou=args.iou, classes=args.classes
                )

                # Mark tracking stage
                perf_monitor.mark_stage("tracking")

                # Run YOLO segmentation
                segmentation_results = segmentation_model(
                    color_frame, conf=args.conf, iou=args.iou, classes=args.classes
                )

                # Mark segmentation stage
                perf_monitor.mark_stage("segmentation")

                # Visualize tracking results
                tracking_vis = tracking_results[0].plot()

                # Visualize segmentation results
                segmentation_vis = segmentation_results[0].plot()

                # Mark end of segmentation stage
                perf_monitor.mark_stage("segmentation")

                # Create visualization with rotating bounding box and 3D pose
                pose_vis = color_frame.copy()

                # Process segmentation masks
                if (
                    hasattr(segmentation_results[0], "masks")
                    and segmentation_results[0].masks is not None
                ):
                    for i, mask in enumerate(segmentation_results[0].masks.data):
                        # Convert tensor mask to numpy array
                        numpy_mask = mask.cpu().numpy()

                        # Extract rotating bounding box
                        box_info = get_rotating_bounding_box(numpy_mask)
                        if box_info is not None:
                            # Draw rotating bounding box
                            pose_vis = draw_rotating_box(
                                pose_vis, box_info["corners"], color=(255, 0, 255), thickness=2
                            )

                            # Compute 3D pose
                            rvec, tvec = compute_3d_pose(box_info, camera_matrix, dist_coeffs)

                            if rvec is not None and tvec is not None:
                                # Get Euler angles
                                roll, pitch, yaw = get_euler_angles(rvec)

                                # Get 3D coordinates (use RealSense depth data)
                                center_x, center_y = box_info["center"]
                                center_x, center_y = int(center_x), int(center_y)

                                # Use robust depth calculation to reduce noise
                                x, y, _ = input_source.get_point_3d(
                                    center_x, center_y, depth_frame
                                )
                                z = get_robust_depth(input_source, center_x, center_y, depth_frame)

                                # Update translation vector with actual depth data from RealSense
                                tvec[0, 0] = x
                                tvec[1, 0] = y
                                tvec[2, 0] = z

                                # Create 3D bounding box
                                # Get depth at an edge point of the object for thickness calculation
                                # Use the first corner of the rotated bounding box as an edge point
                                edge_x, edge_y = box_info["corners"][0]
                                edge_x, edge_y = int(edge_x), int(edge_y)

                                # Get robust 3D coordinates at that edge point
                                edge_x_3d, edge_y_3d, _ = input_source.get_point_3d(
                                    edge_x, edge_y, depth_frame
                                )
                                edge_z = get_robust_depth(
                                    input_source, edge_x, edge_y, depth_frame
                                )

                                # Create 3D bounding box with dimensions matching the 2D box
                                # and thickness based on the depth difference between center and edge
                                bbox_3d = create_3d_bbox(
                                    box_info,
                                    rvec,
                                    use_realsense=True,
                                    center_depth=z,
                                    edge_depth=edge_z,
                                    max_height=args.max_box_height,
                                    default_height=args.box_height,
                                )

                                # Draw 3D bounding box
                                pose_vis = draw_3d_bbox(
                                    pose_vis,
                                    bbox_3d,
                                    rvec,
                                    tvec,
                                    camera_matrix,
                                    dist_coeffs,
                                    color=(0, 255, 255),  # Yellow color for 3D box
                                    thickness=2,
                                )

                                # Draw 3D coordinate axes
                                pose_vis = draw_3d_axes(
                                    pose_vis,
                                    rvec,
                                    tvec,
                                    camera_matrix,
                                    dist_coeffs,
                                    axis_length=args.axis_length,
                                )

                                # Display pose information
                                text = f"ID: {i} | X: {x:.3f}m Y: {y:.3f}m Z: {z:.3f}m"
                                cv2.putText(
                                    pose_vis,
                                    text,
                                    (center_x, center_y - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2,
                                )

                                euler_text = f"Roll: {roll:.1f} Pitch: {pitch:.1f} Yaw: {yaw:.1f}"
                                cv2.putText(
                                    pose_vis,
                                    euler_text,
                                    (center_x, center_y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2,
                                )

                                # Print to console
                                print(
                                    f"Object {i}: Position [X:{x:.3f}m Y:{y:.3f}m Z:{z:.3f}m] Rotation [Roll:{roll:.1f}° Pitch:{pitch:.1f}° Yaw:{yaw:.1f}°] BBox: {box_info['corners'].tolist()}"
                                )

                # Mark end of pose estimation stage
                perf_monitor.mark_stage("pose_estimation")

                # Create 2x2 grid display with tracking, segmentation, 3D pose, and depth views

                # Enhance depth visualization with better colorization
                if depth_frame is not None:
                    # Normalize depth for better visualization (focusing on relevant depths)
                    depth_norm = depth_frame.copy().astype(np.float32)

                    # Filter out invalid depths (0 values) and clip to reasonable range
                    valid_mask = (depth_norm > 0) & (depth_norm < 10000)
                    if np.any(valid_mask):
                        min_val = np.percentile(depth_norm[valid_mask], 5)  # 5th percentile
                        max_val = np.percentile(depth_norm[valid_mask], 95)  # 95th percentile

                        # Normalize to 0-255 range based on percentile values
                        depth_norm = np.clip(depth_norm, min_val, max_val)
                        depth_norm = ((depth_norm - min_val) / (max_val - min_val) * 255).astype(
                            np.uint8
                        )

                        # Apply colormap
                        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

                        # Add min/max depth labels
                        min_text = f"Min: {min_val*input_source.depth_scale:.2f}m"
                        max_text = f"Max: {max_val*input_source.depth_scale:.2f}m"
                        cv2.putText(
                            depth_colormap,
                            min_text,
                            (10, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                        cv2.putText(
                            depth_colormap,
                            max_text,
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                    else:
                        # Fallback if no valid depths
                        depth_colormap = cv2.applyColorMap(
                            cv2.normalize(depth_frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
                            cv2.COLORMAP_JET,
                        )
                else:
                    # Create black frame if depth is None
                    depth_colormap = np.zeros(
                        (pose_vis.shape[0], pose_vis.shape[1], 3), dtype=np.uint8
                    )

                # Create the 2x2 grid display
                display = create_grid_display(
                    tracking_vis, segmentation_vis, pose_vis, depth_colormap
                )

                # Mark end of visualization stage
                perf_monitor.mark_stage("visualization")

                # Draw performance statistics on the display
                display = perf_monitor.draw_stats(display)

                # Show the 2x2 grid display
                cv2.imshow("6DoF Pose Estimation Pipeline", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # End frame timing
                perf_monitor.end_frame()

        else:
            # Standard loop for image, video, webcam
            while True:
                # Start frame timing
                perf_monitor.start_frame()

                # Get frame
                success, frame = input_source.get_frame()
                if not success:
                    print("Failed to get frame")
                    break

                # Run YOLO tracking
                tracking_results = tracking_model.track(
                    frame, persist=True, conf=args.conf, iou=args.iou, classes=args.classes
                )

                # Mark tracking stage
                perf_monitor.mark_stage("tracking")

                # Run YOLO segmentation
                segmentation_results = segmentation_model(
                    frame, conf=args.conf, iou=args.iou, classes=args.classes
                )

                # Mark segmentation stage
                perf_monitor.mark_stage("segmentation")

                # Visualize tracking results
                tracking_vis = tracking_results[0].plot()

                # Visualize segmentation results
                segmentation_vis = segmentation_results[0].plot()

                # Mark end of segmentation stage
                perf_monitor.mark_stage("segmentation")

                # Create visualization with rotating bounding box and 3D pose
                pose_vis = frame.copy()

                # Process segmentation masks
                if (
                    hasattr(segmentation_results[0], "masks")
                    and segmentation_results[0].masks is not None
                ):
                    for i, mask in enumerate(segmentation_results[0].masks.data):
                        # Convert tensor mask to numpy array
                        numpy_mask = mask.cpu().numpy()

                        # Extract rotating bounding box
                        box_info = get_rotating_bounding_box(numpy_mask)
                        if box_info is not None:
                            # Draw rotating bounding box
                            pose_vis = draw_rotating_box(
                                pose_vis, box_info["corners"], color=(255, 0, 255), thickness=2
                            )

                            # Compute 3D pose
                            rvec, tvec = compute_3d_pose(box_info, camera_matrix, dist_coeffs)

                            if rvec is not None and tvec is not None:
                                # Get Euler angles
                                roll, pitch, yaw = get_euler_angles(rvec)

                                # For non-RealSense, just use the translation vector
                                x, y, z = tvec.flatten()

                                # Create 3D bounding box with default thickness
                                # For non-RealSense cameras, the box size matches 2D dimensions
                                # with the default thickness along Y-axis (vertical)
                                bbox_3d = create_3d_bbox(
                                    box_info,
                                    rvec,
                                    depth=z,
                                    use_realsense=False,
                                    max_height=args.max_box_height,
                                    default_height=args.box_height,
                                )

                                # Draw 3D bounding box
                                pose_vis = draw_3d_bbox(
                                    pose_vis,
                                    bbox_3d,
                                    rvec,
                                    tvec,
                                    camera_matrix,
                                    dist_coeffs,
                                    color=(0, 255, 255),  # Yellow color for 3D box
                                    thickness=2,
                                )

                                # Draw 3D coordinate axes
                                pose_vis = draw_3d_axes(
                                    pose_vis,
                                    rvec,
                                    tvec,
                                    camera_matrix,
                                    dist_coeffs,
                                    axis_length=args.axis_length,
                                )

                                # Display pose information
                                text = f"ID: {i} | X: {x:.3f} Y: {y:.3f} Z: {z:.3f}"
                                cv2.putText(
                                    pose_vis,
                                    text,
                                    (int(box_info["center"][0]), int(box_info["center"][1]) - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2,
                                )

                                euler_text = f"Roll: {roll:.1f} Pitch: {pitch:.1f} Yaw: {yaw:.1f}"
                                cv2.putText(
                                    pose_vis,
                                    euler_text,
                                    (int(box_info["center"][0]), int(box_info["center"][1]) - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 255, 0),
                                    2,
                                )

                                # Print to console
                                print(
                                    f"Object {i}: Position [X:{x:.3f} Y:{y:.3f} Z:{z:.3f}] Rotation [Roll:{roll:.1f}° Pitch:{pitch:.1f}° Yaw:{yaw:.1f}°] BBox: {box_info['corners'].tolist()}"
                                )

                # Mark end of pose estimation stage
                perf_monitor.mark_stage("pose_estimation")

                # Create 2x2 grid display with tracking, segmentation, 3D pose views
                # For standard cameras, depth view will be black (depth_img=None)
                display = create_grid_display(
                    tracking_vis, segmentation_vis, pose_vis, depth_img=None
                )

                # Mark end of visualization stage
                perf_monitor.mark_stage("visualization")

                # Draw performance statistics on the display
                display = perf_monitor.draw_stats(display)

                # Show the 2x2 grid display
                cv2.imshow("6DoF Pose Estimation Pipeline", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                # End frame timing
                perf_monitor.end_frame()

                # Break after processing one frame for image input
                if args.input == "image":
                    cv2.waitKey(0)
                    break

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Release resources
        input_source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
