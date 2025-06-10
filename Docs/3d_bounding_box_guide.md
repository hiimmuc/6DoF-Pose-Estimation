# 3D Bounding Box Visualization in 6DoF Pose Estimation

This document explains how the 3D bounding box visualization works in the 6DoF pose estimation pipeline, with particular focus on how the boxes match 2D detections and maintain vertical orientation.

## Overview

The 6DoF pose estimation pipeline visualizes objects with 3D bounding boxes that show the estimated position and orientation in 3D space. The pipeline now features:

1. 2x2 grid display showing:

    - Tracking view (top-left)
    - Segmentation view (top-right)
    - 3D bounding box view (bottom-left)
    - Depth view (bottom-right) or black frame for non-depth cameras

2. 3D bounding boxes that:
    - Match the 2D detected object dimensions in width and height
    - Maintain a vertical orientation (90-degree rotation around the pitch axis)
    - Have appropriate thickness based on camera type

## Implementation Details

### 1. Box Creation

The `create_3d_bbox()` function generates a 3D bounding box with these key features:

-   **Dimensions**: Width and height match exactly the 2D bounding box dimensions from detection
-   **Orientation**: Box is created with its height along the Y-axis (vertical), effectively rotating it 90 degrees around the pitch axis
-   **Thickness Determination**:
    -   For regular cameras: Uses a default thickness of 15 units (configurable via `--box-height`)
    -   For RealSense cameras: Calculates thickness based on the depth difference between the center and edge of the object (with min/max limits)

### 2. Box Orientation

-   After creating the initial box with vertical orientation, the function applies the object's own rotation
-   This aligns the box with the object's orientation based on the rotation vector (rvec) from the PnP solver
-   The rotation is applied to all points in the box, maintaining proper orientation while aligning with the object's pose

### 3. Visualization in 2x2 Grid

The visualization now uses a 2x2 grid display (`create_grid_display()`):

-   **Tracking View**: Shows object detection and tracking results
-   **Segmentation View**: Shows object instance segmentation masks
-   **3D Bbox View**: Shows 3D bounding boxes and coordinate axes
-   **Depth View**: Shows depth data for RealSense cameras or a black frame for regular cameras

### 4. Box Drawing

The `draw_3d_bbox()` function visualizes the 3D bounding box:

-   Projects the 3D box corners onto the 2D image plane using the camera parameters
-   Draws the box's edges in a specified color (yellow by default)
-   Shows all faces and connecting lines for clear visualization

## Configuration Options

The pipeline accepts several command-line arguments to customize the 3D bounding box:

-   `--box-height`: Default thickness for 3D bounding box in regular cameras (in units, default: 15.0)
-   `--max-box-height`: Maximum thickness for 3D bounding box in RealSense mode (in units, default: 50.0)
-   `--axis-length`: Length of 3D coordinate axes visualization in meters (default: 0.1)

## Example Usage

```bash
# Run with webcam and custom box height
python examples/6dof_pipeline.py --input webcam --box-height 15

# Run with RealSense camera and custom max height
python examples/6dof_pipeline.py --input realsense --max-box-height 40
```

## Box Thickness Determination

### Regular Cameras

For regular cameras (webcam, video, image), the 3D bounding box thickness is set to 15 units by default. This provides a consistent visualization when depth data is not available.

```python
bbox_3d = create_3d_bbox(
    box_info,
    rvec,
    depth=z,
    use_realsense=False,
    max_height=args.max_box_height,
    default_height=args.box_height,  # 15.0 by default
)
```

### RealSense Cameras

For RealSense cameras, the thickness is dynamically calculated as the depth difference between the center of the object and one of its edges:

```python
# Get 3D coordinates at the center point
x, y, z = input_source.get_point_3d(center_x, center_y, depth_frame)

# Get 3D coordinates at an edge point
edge_x, edge_y = box_info["corners"][0]
edge_x_3d, edge_y_3d, edge_z = input_source.get_point_3d(edge_x, edge_y, depth_frame)

# Create 3D bounding box with thickness based on depth difference
bbox_3d = create_3d_bbox(
    box_info,
    rvec,
    use_realsense=True,
    center_depth=z,
    edge_depth=edge_z,
    max_height=args.max_box_height,
    default_height=args.box_height,
)
```

This dynamic thickness calculation provides more realistic 3D bounding boxes that better represent the actual size of the object in 3D space.

# Run with video input and custom settings

python examples/6dof_pipeline.py --input video --source path/to/video.mp4 --box-height 20 --axis-length 0.2

```

## Technical Notes

1. The coordinate system used follows the standard computer vision convention:

    - X-axis: points right (red line in visualization)
    - Y-axis: points down in image but up in 3D space (green line in visualization)
    - Z-axis: points into the image (blue line in visualization)

2. The 90-degree pitch rotation means the box's height extends along the Y-axis (up/down) rather than the Z-axis (depth).

3. For RealSense cameras, the depth difference between the center and edge points provides a more realistic height estimation.
```
