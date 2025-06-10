from typing import Optional, Tuple, Union

import numpy as np


def create_polygon_mask(bbox_2d: np.ndarray, depth_shape: Tuple[int, int]) -> np.ndarray:
    """
    Create a binary mask from 4-point polygon for depth extraction.

    Args:
        bbox_2d: 4x2 array of corner points
        depth_shape: (height, width) of depth map

    Returns:
        Binary mask array
    """
    from PIL import Image, ImageDraw

    # Create a PIL image for polygon drawing
    img = Image.new("L", (depth_shape[1], depth_shape[0]), 0)
    draw = ImageDraw.Draw(img)

    # Convert points to tuple format for PIL
    polygon_points = [(int(p[0]), int(p[1])) for p in bbox_2d]

    # Draw filled polygon
    draw.polygon(polygon_points, outline=1, fill=1)

    # Convert back to numpy array
    mask = np.array(img, dtype=bool)

    return mask


def bbox_2d_to_3d_with_polygon(
    pose_6dof: np.ndarray,
    bbox_2d: np.ndarray,
    depth_map: np.ndarray,
    camera_intrinsics: np.ndarray,
    object_dimensions: Optional[Tuple[float, float, float]] = None,
    depth_scale: float = 1.0,
    use_polygon_mask: bool = False,
) -> np.ndarray:
    """
    Enhanced version that can use polygon mask for more accurate depth extraction.

    Args:
        pose_6dof: 6DOF pose as [x, y, z, rx, ry, rz]
        bbox_2d: 2D bounding box as 4x2 array of corner points
        depth_map: Depth map array (H x W)
        camera_intrinsics: 3x3 camera intrinsic matrix
        object_dimensions: Optional (width, height, depth) of object in meters
        depth_scale: Scale factor for depth values
        use_polygon_mask: If True, use polygon mask for depth extraction

    Returns:
        8x3 array of 3D bounding box corners in world coordinates
    """

    # Ensure bbox_2d is numpy array and has correct shape
    bbox_2d = np.array(bbox_2d)
    if bbox_2d.shape != (4, 2):
        raise ValueError("bbox_2d must be a 4x2 array of corner points")

    # Get depth values - either from polygon mask or bounding rectangle
    if use_polygon_mask:
        try:
            mask = create_polygon_mask(bbox_2d, depth_map.shape)
            depth_roi = depth_map[mask] * depth_scale
        except ImportError:
            print("PIL not available, falling back to bounding rectangle method")
            use_polygon_mask = False

    if not use_polygon_mask:
        # Use bounding rectangle method
        x_coords = bbox_2d[:, 0]
        y_coords = bbox_2d[:, 1]
        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        # Ensure bounds are within image
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(depth_map.shape[1], x_max)
        y_max = min(depth_map.shape[0], y_max)

        depth_roi = depth_map[y_min:y_max, x_min:x_max] * depth_scale

    # Filter out invalid depth values
    valid_depths = depth_roi[depth_roi > 0.1]

    if len(valid_depths) == 0:
        raise ValueError("No valid depth values found in bounding box region")

    # Estimate object depth
    estimated_depth = np.median(valid_depths)

    # Extract camera intrinsics
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]

    # Project 2D corner points to 3D camera coordinates
    bbox_3d_camera = []
    for corner in bbox_2d:
        x_cam = (corner[0] - cx) * estimated_depth / fx
        y_cam = (corner[1] - cy) * estimated_depth / fy
        z_cam = estimated_depth
        bbox_3d_camera.append([x_cam, y_cam, z_cam])

    bbox_3d_camera = np.array(bbox_3d_camera)

    # Create 3D bounding box
    if object_dimensions is not None:
        width, height, depth = object_dimensions

        corners_local = np.array(
            [
                [-width / 2, -height / 2, -depth / 2],
                [width / 2, -height / 2, -depth / 2],
                [width / 2, height / 2, -depth / 2],
                [-width / 2, height / 2, -depth / 2],
                [-width / 2, -height / 2, depth / 2],
                [width / 2, -height / 2, depth / 2],
                [width / 2, height / 2, depth / 2],
                [-width / 2, height / 2, depth / 2],
            ]
        )
    else:
        # Estimate dimensions from 2D bounding box points
        edge_lengths = []
        for i in range(4):
            p1 = bbox_2d[i]
            p2 = bbox_2d[(i + 1) % 4]
            edge_length = np.linalg.norm(p2 - p1)
            edge_lengths.append(edge_length)

        # Assume opposite edges represent width and height
        estimated_width_px = (edge_lengths[0] + edge_lengths[2]) / 2
        estimated_height_px = (edge_lengths[1] + edge_lengths[3]) / 2

        estimated_width = estimated_width_px * estimated_depth / fx
        estimated_height = estimated_height_px * estimated_depth / fy
        estimated_depth_dim = estimated_width * 0.5

        corners_local = np.array(
            [
                [-estimated_width / 2, -estimated_height / 2, -estimated_depth_dim / 2],
                [estimated_width / 2, -estimated_height / 2, -estimated_depth_dim / 2],
                [estimated_width / 2, estimated_height / 2, -estimated_depth_dim / 2],
                [-estimated_width / 2, estimated_height / 2, -estimated_depth_dim / 2],
                [-estimated_width / 2, -estimated_height / 2, estimated_depth_dim / 2],
                [estimated_width / 2, -estimated_height / 2, estimated_depth_dim / 2],
                [estimated_width / 2, estimated_height / 2, estimated_depth_dim / 2],
                [-estimated_width / 2, estimated_height / 2, estimated_depth_dim / 2],
            ]
        )

    # Convert 6DOF pose to transformation matrix
    translation = pose_6dof[:3]
    rotation_angles = pose_6dof[3:]

    # Create rotation matrix from Euler angles (ZYX convention)
    rx, ry, rz = rotation_angles

    Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])

    Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])

    Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])

    # Combined rotation matrix
    R = Rz @ Ry @ Rx

    # Apply transformation
    bbox_3d_world = (R @ corners_local.T).T + translation

    return bbox_3d_world


def visualize_3d_bbox(bbox_3d: np.ndarray) -> None:
    """
    Simple visualization helper to print 3D bounding box corners.

    Args:
        bbox_3d: 8x3 array of 3D bounding box corners
    """
    print("3D Bounding Box Corners (x, y, z):")
    corner_names = [
        "Bottom-front-left",
        "Bottom-front-right",
        "Bottom-back-right",
        "Bottom-back-left",
        "Top-front-left",
        "Top-front-right",
        "Top-back-right",
        "Top-back-left",
    ]

    for i, (corner, name) in enumerate(zip(bbox_3d, corner_names)):
        print(f"Corner {i} ({name}): ({corner[0]:.3f}, {corner[1]:.3f}, {corner[2]:.3f})")


# Example usage
if __name__ == "__main__":
    # Example data with 4-point 2D bounding box
    pose_6dof = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3])  # x,y,z,rx,ry,rz

    # 2D bounding box as 4 corner points (could be rotated rectangle)
    bbox_2d = np.array(
        [
            [100, 150],  # top-left
            [200, 140],  # top-right (slightly rotated)
            [210, 300],  # bottom-right
            [110, 310],  # bottom-left
        ]
    )

    # Simulate depth map
    depth_map = np.ones((480, 640)) * 2.5  # 2.5 meters depth
    depth_map[140:310, 100:210] = 2.0  # Object region at 2m depth

    # Camera intrinsics
    camera_intrinsics = np.array([[525.0, 0, 320.0], [0, 525.0, 240.0], [0, 0, 1.0]])

    # Object dimensions in meters
    object_dims = (0.3, 0.4, 0.2)

    try:
        # Method 1: Basic function (backward compatible)
        print("=== Method 1: Basic 4-point bbox ===")
        bbox_3d = bbox_2d_to_3d_with_polygon(
            pose_6dof=pose_6dof,
            bbox_2d=bbox_2d,
            depth_map=depth_map,
            camera_intrinsics=camera_intrinsics,
            object_dimensions=object_dims,
            depth_scale=1.0,
            use_polygon_mask=False,
        )

        visualize_3d_bbox(bbox_3d)
        print(f"Center: {np.mean(bbox_3d, axis=0)}")

        # Method 2: With polygon mask (more accurate for rotated boxes)
        print("\n=== Method 2: With polygon mask ===")
        bbox_3d_polygon = bbox_2d_to_3d_with_polygon(
            pose_6dof=pose_6dof,
            bbox_2d=bbox_2d,
            depth_map=depth_map,
            camera_intrinsics=camera_intrinsics,
            object_dimensions=object_dims,
            depth_scale=1.0,
            use_polygon_mask=True,
        )

        print(f"Center with polygon mask: {np.mean(bbox_3d_polygon, axis=0)}")

        # Example with regular rectangle (backward compatibility)
        print("\n=== Method 3: Regular rectangle ===")
        bbox_2d_rect = np.array(
            [
                [100, 150],  # top-left
                [200, 150],  # top-right
                [200, 300],  # bottom-right
                [100, 300],  # bottom-left
            ]
        )

        bbox_3d_rect = bbox_2d_to_3d_with_polygon(
            pose_6dof=pose_6dof,
            bbox_2d=bbox_2d_rect,
            depth_map=depth_map,
            camera_intrinsics=camera_intrinsics,
            object_dimensions=object_dims,
        )

        print(f"Rectangle center: {np.mean(bbox_3d_rect, axis=0)}")

    except Exception as e:
        print(f"Error: {e}")


# Convenience function for backward compatibility
def bbox_2d_to_3d(
    pose_6dof, bbox_2d, depth_map, camera_intrinsics, object_dimensions=None, depth_scale=1.0
):
    """
    Backward compatible function that works with both formats:
    - bbox_2d as (x_min, y_min, x_max, y_max) tuple
    - bbox_2d as 4x2 array of corner points
    """
    if isinstance(bbox_2d, (tuple, list)) and len(bbox_2d) == 4:
        # Convert (x_min, y_min, x_max, y_max) to 4 corners
        x_min, y_min, x_max, y_max = bbox_2d
        bbox_2d = np.array(
            [
                [x_min, y_min],  # top-left
                [x_max, y_min],  # top-right
                [x_max, y_max],  # bottom-right
                [x_min, y_max],  # bottom-left
            ]
        )

    return bbox_2d_to_3d_with_polygon(
        pose_6dof,
        bbox_2d,
        depth_map,
        camera_intrinsics,
        object_dimensions,
        depth_scale,
        use_polygon_mask=False,
    )

if __name__ == "__main__":
    
    # Rotated/arbitrary 4-point bounding box
    bbox_2d = np.array([
        [100, 150],  # corner 1
        [200, 140],  # corner 2 (rotated)
        [210, 300],  # corner 3
        [110, 310]   # corner 4
    ])
