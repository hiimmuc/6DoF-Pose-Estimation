# 6 DoF Object Pose Estimation

This repository provides a comprehensive pipeline for 6 DoF (Degrees of Freedom) object pose estimation using various methods, with a focus on real-time performance.

## Features

### Enhanced 6DoF Pipeline

The main pipeline (`examples/6dof_pipeline.py`) implements a complete workflow for 6DoF pose estimation with the following capabilities:

1. **Input Options**:

    - Webcam
    - Video file
    - Image file
    - Intel RealSense camera (with depth)

2. **Visualization**:

    - 2x2 grid display showing:
        - Tracking view (top-left)
        - Segmentation view (top-right)
        - 3D bounding box view (bottom-left)
        - Depth view (bottom-right) or black frame for non-depth cameras

3. **3D Bounding Box Visualization**:

    - Boxes match 2D detected object dimensions
    - Vertical orientation (90-degree rotation around pitch axis)
    - Dynamic thickness calculation:
        - For RealSense: Based on depth difference between center and edge
        - For regular cameras: Default thickness of 15 units

4. **Performance Monitoring**:
    - Real-time FPS tracking
    - Processing time for each pipeline stage
    - Visual performance overlay

## Usage

### Basic Usage

```bash
# Run with webcam
python examples/6dof_pipeline.py --input webcam

# Run with RealSense camera
python examples/6dof_pipeline.py --input realsense

# Run with video file
python examples/6dof_pipeline.py --input video --source path/to/video.mp4

# Run with image file
python examples/6dof_pipeline.py --input image --source path/to/image.jpg
```

### Advanced Options

```bash
# Set confidence threshold
python examples/6dof_pipeline.py --input webcam --conf 0.6

# Filter specific classes
python examples/6dof_pipeline.py --input webcam --classes 0 41 67

# Customize 3D box height for regular cameras
python examples/6dof_pipeline.py --input webcam --box-height 20

# Set maximum box height for RealSense
python examples/6dof_pipeline.py --input realsense --max-box-height 40

# Change the size of coordinate axes
python examples/6dof_pipeline.py --input webcam --axis-length 0.15
```

## Documentation

-   For more details about 3D bounding box visualization, see [3d_bounding_box_guide.md](./Docs/3d_bounding_box_guide.md)

## Research Directions

We are investigating the following methods:

1. Traditional approach:

    - [x] Segment (DL: YOLO) → Get contour → Extract Rotated Boxes → Pose calculation with PnP
    - [ ] Traditional segmentation → Get contour → Extract Rotated Boxes
    - [ ] Segment (SAM2) → Get BBox → Pose calculation with PnP

2. RGB-only methods:

    - [ ] PoET

3. RGB-D methods:

    - [ ] Any6D

4. Real-time methods:
    - [ ] RNNPose
    - [ ] Yolo v5 6D
    - [ ] GDRNPP (or Fast version)

## Requirements

See `requirements.txt` for detailed dependencies.
