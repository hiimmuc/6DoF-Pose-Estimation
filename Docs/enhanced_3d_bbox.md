# Enhanced 3D Bounding Box Estimation

This document explains the enhanced 3D bounding box estimation methods in the 6DoF pose estimation pipeline.

## Overview

The pipeline now includes two methods for creating 3D bounding boxes:

1. **Legacy Method** (`create_3d_bbox`): Creates a 3D box based on 2D detection size and a fixed or estimated depth.
2. **Enhanced Method** (`create_3d_bbox_enhanced`): Uses depth maps and polygon masks for more accurate 3D box dimensions.

## Enhanced Method Details

The enhanced 3D bounding box estimation method:

-   Uses depth maps to determine object dimensions more accurately
-   Supports polygon masks to extract depth values from precisely the object region
-   Provides more accurate object dimensions in 3D space
-   Maintains alignment with the 2D detection (back face aligned with 2D box)
-   Works with rotated/arbitrary quadrilateral bounding boxes

## Usage

By default, the enhanced method is used when running with a RealSense camera. You can control this with:

```
# Use enhanced method (default)
python run_6dof_pipeline.py --input realsense --use-enhanced-bbox

# Use legacy method
python run_6dof_pipeline.py --input realsense --no-enhanced-bbox
```

## Technical Implementation

The enhanced method uses these key components:

1. **Polygon Mask Creation** (`create_polygon_mask`): Creates a binary mask from a 4-point polygon for depth extraction
2. **3D Box Creation with Polygon** (`bbox_2d_to_3d_with_polygon`): Uses the mask to extract depth values and create the 3D box

### Advantages of Enhanced Method

-   **Better Depth Estimation**: By using the entire object region instead of just a few points, depth estimation is more robust
-   **Better Object Dimensions**: Object dimensions are more accurately estimated by using depth information
-   **Support for Rotated Objects**: Works well with objects at arbitrary orientations
-   **More Accurate Pose**: The resulting 6DoF pose is more accurate due to better depth estimation

## Comparison with Legacy Method

| Feature           | Legacy Method         | Enhanced Method                  |
| ----------------- | --------------------- | -------------------------------- |
| Depth Estimation  | Point samples         | Full polygon region              |
| Object Dimensions | Based on 2D size      | From depth and 2D size           |
| Rotated Objects   | Limited support       | Full support                     |
| Alignment with 2D | Back face aligned     | Back face aligned                |
| Required Inputs   | Box corners, rotation | Box corners, rotation, depth map |
| Processing Speed  | Faster                | Slightly slower                  |
