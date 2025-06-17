"""Visualization utilities for pose estimation results."""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def draw_pose(
    image: np.ndarray,
    pose_result,  # PoseResult type (avoiding circular import)
    camera_matrix: np.ndarray,
    dist_coefs: np.ndarray,
    depth_frame: Optional[np.ndarray] = None,
) -> None:
    """
    Draw pose axes and bounding boxes (2D rotated and 3D) on image.

    Args:
        image: Image to draw on.
        pose_result: Pose estimation result.
        camera_matrix: Camera intrinsic matrix.
        dist_coefs: Distortion coefficients.
        depth_frame: Optional depth frame for 3D box visualization.
    """
    # Draw the rotated bounding box
    x, y, w, h, angle = pose_result.bbox_rotated
    box = cv2.boxPoints(((x, y), (w, h), angle))
    box = np.intp(box)
    cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

    # Draw 3D bounding box if available
    if pose_result.bbox_3d is not None:
        draw_3d_box(image, pose_result.bbox_3d)
    elif depth_frame is not None:
        # Convert pose_result.bbox_2d from [x, y, width, height] to [x1, y1, x2, y2]
        x, y, width, height = pose_result.bbox_2d
        bbox = [x, y, x + width, y + height]

        # Get depth and direction
        depth = get_deepest_narrowest_depth(depth_frame, bbox)
        direction = get_front_direction(depth_frame, bbox)

        # Generate and draw 3D box
        box_3d_points = generate_3d_box_points(bbox, depth, direction)
        draw_3d_box(image, box_3d_points, color=(255, 165, 0), thickness=2)  # Orange color

    # Draw coordinate axes
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

    # Define axis points
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

    # Draw object information
    text_info = f"ID:{pose_result.object_id} {pose_result.class_name}"
    pose_info = f"x:{pose_result.x:.0f} y:{pose_result.y:.0f} z:{pose_result.z:.0f}"
    angle_info = f"r:{pose_result.roll:.1f} p:{pose_result.pitch:.1f} y:{pose_result.yaw:.1f}"

    cv2.putText(
        image, text_info, (int(x), int(y) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
    )
    cv2.putText(
        image, pose_info, (int(x), int(y) - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2
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
    # Check required visualizations
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

    # Add compact performance metrics table
    _add_performance_table(grid, visualizations, w, h)

    return grid


def _add_performance_table(grid: np.ndarray, visualizations: Dict, w: int, h: int) -> None:
    """Add performance metrics table to visualization grid."""
    if "fps_info" not in visualizations:
        return

    fps_info = visualizations["fps_info"]
    time_info = visualizations.get("time_info", {})

    # Make a compact table
    padding = 5
    font_scale = 0.4
    line_thickness = 1
    row_height = 15

    # Calculate dimensions
    total_width = w * 2  # Full width of the grid

    # Sort stages by processing order
    stage_order = ["tracking", "segmentation", "pose_estimation", "visualization"]
    sorted_stages = [s for s in stage_order if s in fps_info]
    remaining_stages = [s for s in fps_info if s not in sorted_stages]
    sorted_stages.extend(sorted(remaining_stages))

    # Calculate table dimensions
    num_stages = len(sorted_stages)
    table_height = row_height * (num_stages + 1)  # Header + data rows
    table_width = total_width // 2  # Half width
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

    # Draw header
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

    # Draw line under header
    cv2.line(
        grid,
        (table_x - padding, table_y + row_height + 2),
        (total_width - padding, table_y + row_height + 2),
        (150, 150, 150),
        1,
    )

    # Draw data rows
    _draw_performance_rows(
        grid,
        sorted_stages,
        fps_info,
        time_info,
        table_x,
        table_y,
        col1_width,
        col2_width,
        row_height,
        font_scale,
        line_thickness,
    )


def _draw_performance_rows(
    grid: np.ndarray,
    stages: List[str],
    fps_info: Dict[str, int],
    time_info: Dict[str, float],
    table_x: int,
    table_y: int,
    col1_width: float,
    col2_width: float,
    row_height: int,
    font_scale: float,
    line_thickness: int,
) -> None:
    """Draw performance data rows in the table."""
    # Stage name abbreviations
    abbreviations = {
        "tracking": "Track",
        "segmentation": "Seg",
        "pose_estimation": "Pose",
        "visualization": "Vis",
    }

    for i, stage in enumerate(stages):
        fps = fps_info[stage]
        y_pos = table_y + (i + 2) * row_height

        # Use abbreviation or first 4 chars for stage name
        display_name = abbreviations.get(stage, stage[:4].capitalize())

        # Draw stage name
        cv2.putText(
            grid,
            display_name,
            (table_x, y_pos),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            line_thickness,
        )

        # Draw FPS value with color coding
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

        # Draw time in milliseconds
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


def get_deepest_narrowest_depth(
    depth_map: np.ndarray, bbox: List[int], center_ratio: float = 0.2
) -> float:
    """
    Get the deepest depth value from the center region of a bounding box.

    Args:
        depth_map: Depth image.
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        center_ratio: Ratio of center region to consider.

    Returns:
        float: Depth value.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    cx1 = int(x1 + (1 - center_ratio) * w / 2)
    cy1 = int(y1 + (1 - center_ratio) * h / 2)
    cx2 = int(x2 - (1 - center_ratio) * w / 2)
    cy2 = int(y2 - (1 - center_ratio) * h / 2)

    region = depth_map[cy1:cy2, cx1:cx2]
    return np.max(region) if region.size > 0 else np.mean(depth_map[y1:y2, x1:x2])


def get_front_direction(depth_map: np.ndarray, bbox: List[int]) -> Tuple[int, int]:
    """
    Determine the front direction of an object based on depth variations.

    Args:
        depth_map: Depth image.
        bbox: Bounding box coordinates [x1, y1, x2, y2].

    Returns:
        Tuple[int, int]: Direction vector (dx, dy).
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    left_depth = np.mean(depth_map[y1:y2, x1 : x1 + w // 10])
    right_depth = np.mean(depth_map[y1:y2, x2 - w // 10 : x2])
    top_depth = np.mean(depth_map[y1 : y1 + h // 10, x1:x2])
    bottom_depth = np.mean(depth_map[y2 - h // 10 : y2, x1:x2])

    dx = 1 if left_depth > right_depth else -1  # front is to right if left is deeper
    dy = 1 if top_depth > bottom_depth else -1  # front is to bottom if top is deeper

    return dx, dy


def generate_3d_box_points(
    bbox: List[int], depth: float, direction: Tuple[int, int], scale: float = 0.05
) -> np.ndarray:
    """
    Generate points for a 3D bounding box based on 2D box, depth, and direction.

    Args:
        bbox: Bounding box coordinates [x1, y1, x2, y2].
        depth: Depth of the object.
        direction: Direction vector (dx, dy).
        scale: Scale factor for the box depth.

    Returns:
        np.ndarray: 8 points representing the 3D bounding box.
    """
    x1, y1, x2, y2 = bbox
    dx, dy = direction
    offset = int(scale * depth)

    # Front face
    front = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # Back face offset in direction of dx, dy
    back = np.array(
        [
            [x1 + dx * offset, y1 + dy * offset],
            [x2 + dx * offset, y1 + dy * offset],
            [x2 + dx * offset, y2 + dy * offset],
            [x1 + dx * offset, y2 + dy * offset],
        ]
    )

    return np.vstack([front, back]).astype(int)


def draw_3d_box(
    image: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """
    Draw a 3D bounding box on an image.

    Args:
        image: Image to draw on.
        points: 8 points representing the 3D bounding box.
        color: Color of the bounding box.
        thickness: Line thickness.
    """
    # Draw front and back faces
    cv2.polylines(image, [points[:4]], isClosed=True, color=color, thickness=thickness)
    cv2.polylines(image, [points[4:]], isClosed=True, color=color, thickness=thickness)

    # Draw connecting lines between front and back faces
    for i in range(4):
        cv2.line(image, tuple(points[i]), tuple(points[i + 4]), color, thickness)
