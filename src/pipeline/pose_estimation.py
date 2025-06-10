"""
3D pose estimation functionality for 6DoF pose estimation pipeline.

This module provides functions for calculating 6DoF poses,
3D bounding boxes, and coordinate transformations. It includes
enhanced methods for 3D bounding box creation using depth data
and polygon masks for more accurate results.
"""

from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

# Import enhanced 3D bounding box functions
from .transform import bbox_2d_to_3d_with_polygon, create_polygon_mask


def compute_3d_pose(
    box_info: Dict[str, Any], camera_matrix: np.ndarray, dist_coeffs: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Estimate 6DOF pose from rotated bounding box.

    Args:
        box_info: Rotated bounding box information
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients

    Returns:
        Tuple of (rotation vector, translation vector), or (None, None) if pose estimation fails
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
    Convert rotation vector to Euler angles (roll, pitch, yaw).

    Args:
        rvec: Rotation vector

    Returns:
        Roll, pitch, yaw angles in degrees
    """
    rmat, _ = cv2.Rodrigues(rvec)
    r = R.from_matrix(rmat)
    angles = r.as_euler("xyz", degrees=True)

    return angles[0], angles[1], angles[2]  # roll, pitch, yaw


def create_pose_6dof(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    """
    Create a 6DoF pose vector from rotation and translation vectors.

    Args:
        rvec: 3x1 rotation vector
        tvec: 3x1 translation vector

    Returns:
        6DoF pose as [x, y, z, rx, ry, rz]
    """
    # Convert rotation vector to angles (form needed by transform.py functions)
    angles = get_euler_angles(rvec)
    rx, ry, rz = [np.radians(angle) for angle in angles]  # Convert to radians

    # Extract translation values
    x, y, z = tvec.flatten()

    return np.array([x, y, z, rx, ry, rz])


def create_3d_bbox_enhanced(
    box_info: Dict[str, Any],
    rvec: np.ndarray,
    tvec: np.ndarray,
    depth_map: Optional[np.ndarray] = None,
    camera_intrinsics: Optional[np.ndarray] = None,
    depth_scale: float = 1.0,
    use_polygon_mask: bool = True,
    object_dimensions: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    """
    Enhanced 3D bounding box creation that leverages depth maps and polygon masks
    for more accurate object dimensions. Aligns with 2D detection and maintains
    consistent orientation with the back face of the 3D box.

    This function uses the advanced bbox_2d_to_3d_with_polygon method from transform.py
    when depth information is available.

    Args:
        box_info: Dictionary containing rotated bounding box information
        rvec: Rotation vector for object orientation
        tvec: Translation vector for object position
        depth_map: Optional depth image (for accurate depth estimation)
        camera_intrinsics: Optional camera intrinsic matrix (required with depth_map)
        depth_scale: Scale factor for depth values
        use_polygon_mask: Whether to use polygon masks for more accurate depth extraction
        object_dimensions: Optional (width, height, depth) in meters

    Returns:
        8x3 array of 3D bounding box coordinates (corners)
    """
    # Check if we have depth information and required parameters for enhanced method
    can_use_enhanced = depth_map is not None and camera_intrinsics is not None

    if can_use_enhanced:
        # Convert rotation and translation to 6DoF pose format
        pose_6dof = create_pose_6dof(rvec, tvec)

        # Use box corners as 2D bbox input for enhanced method
        bbox_2d = box_info["corners"].astype(np.float32)

        # If object_dimensions not provided, use box_info size with an estimated depth
        if object_dimensions is None:
            width, height = box_info["size"]
            # Estimate depth as a proportion of the width (reasonable default)
            depth = width * 0.5
            object_dimensions = (width, height, depth)

        try:
            # Use enhanced method with polygon mask for depth extraction
            bbox_3d = bbox_2d_to_3d_with_polygon(
                pose_6dof=pose_6dof,
                bbox_2d=bbox_2d,
                depth_map=depth_map,
                camera_intrinsics=camera_intrinsics,
                object_dimensions=object_dimensions,
                depth_scale=depth_scale,
                use_polygon_mask=use_polygon_mask,
            )
            return bbox_3d

        except Exception as e:
            print(f"Error in enhanced 3D bbox creation: {e}. Falling back to standard method.")
            # Fall through to standard method

    # If we don't have depth information or the enhanced method failed,
    # use the standard method for backward compatibility
    return create_3d_bbox(box_info, rvec)


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
    5. The back face of the 3D box aligns with the 2D bounding box, with the
       3D box extending forward from the 2D detection

    The resulting 3D box will have:
    - Width and height matching the 2D bounding box dimensions
    - Depth calculated based on camera type (RealSense vs. regular)
    - Orientation aligned with the object's detected orientation
    - Back face aligned exactly with the 2D detection

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
    # 5. The box is positioned so the back face aligns with the 2D bounding box
    #    and the box extends forward from the 2D detection
    #
    # Box coordinate system (before object orientation is applied):
    # - X-axis: width of the object (matches 2D width)
    # - Y-axis: depth/thickness of the object (extends forward from 2D bounding box)
    # - Z-axis: height of the object (matches 2D height)

    # Create the box with an offset in the Y-axis so the back face aligns with the 2D bounding box
    # This means the box extends forward from the 2D bounding box rather than being centered on it
    box_points = np.array(
        [
            # Bottom face (bottom = lower Y value)
            [-width / 2, -box_depth, -height / 2],  # bottom front left
            [width / 2, -box_depth, -height / 2],  # bottom front right
            [width / 2, 0, -height / 2],  # bottom back right (at origin plane)
            [-width / 2, 0, -height / 2],  # bottom back left (at origin plane)
            # Top face (top = higher Y value)
            [-width / 2, -box_depth, height / 2],  # top front left
            [width / 2, -box_depth, height / 2],  # top front right
            [width / 2, 0, height / 2],  # top back right (at origin plane)
            [-width / 2, 0, height / 2],  # top back left (at origin plane)
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


def get_robust_depth(
    input_source, x: int, y: int, depth_frame: np.ndarray, kernel_size: int = 5
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


def get_depth_from_mask(
    depth_map: np.ndarray,
    mask: np.ndarray,
    depth_scale: float = 1.0,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
) -> float:
    """
    Extract robust depth estimate from a region defined by a binary mask.

    Args:
        depth_map: Depth map image
        mask: Binary mask defining the region of interest
        depth_scale: Scale factor for depth values
        min_depth: Minimum valid depth value
        max_depth: Maximum valid depth value

    Returns:
        Median depth value in the masked region
    """
    # Extract depth values in the masked region
    depth_values = depth_map[mask] * depth_scale

    # Filter out invalid depth values
    valid_depths = depth_values[(depth_values > min_depth) & (depth_values < max_depth)]

    if len(valid_depths) == 0:
        # If no valid depths found, return default value
        return 1.0  # Default 1 meter distance

    # Return median depth (more robust to outliers than mean)
    return float(np.median(valid_depths))
