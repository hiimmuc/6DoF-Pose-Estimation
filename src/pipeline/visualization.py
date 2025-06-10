"""
Visualization utilities for 6DoF pose estimation pipeline.

This module provides functions for visualizing tracking results,
segmentation masks, 3D bounding boxes, coordinate axes, and creating
grid displays for pipeline output.
"""

from typing import Optional, Tuple

import cv2
import numpy as np


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
    Draw 3D coordinate axes to visualize object position and orientation.

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
    Draw 3D bounding box on image.

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
    Draw rotating bounding box on an image.

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
    Create a side-by-side display of tracking and segmentation results.

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


def colorize_depth_image(depth_image: np.ndarray, depth_scale: float = 0.001) -> np.ndarray:
    """
    Enhance depth visualization with better colorization.

    Args:
        depth_image: Raw depth image
        depth_scale: Scale factor to convert depth units to meters

    Returns:
        Colorized depth image
    """
    if depth_image is None:
        return None

    # Normalize depth for better visualization (focusing on relevant depths)
    depth_norm = depth_image.copy().astype(np.float32)

    # Filter out invalid depths (0 values) and clip to reasonable range
    valid_mask = (depth_norm > 0) & (depth_norm < 10000)

    if np.any(valid_mask):
        min_val = np.percentile(depth_norm[valid_mask], 5)  # 5th percentile
        max_val = np.percentile(depth_norm[valid_mask], 95)  # 95th percentile

        # Normalize to 0-255 range based on percentile values
        depth_norm = np.clip(depth_norm, min_val, max_val)
        depth_norm = ((depth_norm - min_val) / (max_val - min_val) * 255).astype(np.uint8)

        # Apply colormap
        depth_colormap = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        # Add min/max depth labels
        min_text = f"Min: {min_val * depth_scale:.2f}mm"
        max_text = f"Max: {max_val * depth_scale:.2f}mm"
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
            cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U),
            cv2.COLORMAP_JET,
        )

    return depth_colormap
