# Dynamic 3D Coordinate Axes

This document explains the dynamic 3D coordinate axes feature in the 6DoF pose estimation pipeline.

## Overview

The pipeline now supports dynamic sizing of the 3D coordinate axes based on the detected object's dimensions. This provides several benefits:

1. **Visual Clarity**: The axes scale appropriately with the object size, making them more visually meaningful
2. **Proper Proportions**: The axes maintain consistent proportions relative to the object
3. **Better Visualization**: Small objects have smaller axes, large objects have larger axes

## How It Works

When enabled, the dynamic axis sizing:

1. Calculates the dimensions of the 3D bounding box
2. Computes the diagonal length of the box
3. Sets the axis length to 20% of the larger value between:
    - The box diagonal length
    - The maximum dimension (width, height, or depth)
4. Applies this calculated length to the coordinate axes visualization

## Usage

By default, the dynamic axis sizing is enabled. You can control this through:

### Command Line

```bash
# Use dynamic axes (0.2 × box size)
./run_6dof_pipeline.py --axis-length 0

# Use a fixed axis length of 0.15 meters
./run_6dof_pipeline.py --axis-length 0.15
```

### Programmatic API

```python
from src.pipeline import estimate_poses

# With dynamic axis length
results = estimate_poses(
    input_source_type="webcam",
    source_path="0",
    axis_length=0  # Dynamic sizing
)

# With fixed axis length
results = estimate_poses(
    input_source_type="webcam",
    source_path="0",
    axis_length=0.15  # Fixed 15cm length
)
```

## Demo

A dedicated demo is available to showcase this feature:

```bash
# Run the dynamic axes demo
./examples/dynamic_axes_demo.py

# Compare fixed vs dynamic axis length side by side
./examples/dynamic_axes_demo.py --compare --fixed-length 0.1
```

## Benefits for Different Object Sizes

| Object Size | Fixed Axes                                     | Dynamic Axes                                            |
| ----------- | ---------------------------------------------- | ------------------------------------------------------- |
| Very Small  | Axes may be too large, overwhelming the object | Appropriately sized, clearly visible but not dominating |
| Medium      | May be appropriate or slightly off             | Well-proportioned to the object                         |
| Very Large  | May be too small to be clearly visible         | Properly scaled to be clearly visible                   |
