# 6DoF Pose Estimation Pipeline

This pipeline provides functionality for estimating 6-degree-of-freedom (6DoF) poses of objects from different input sources. It combines YOLO-based object detection, tracking, and segmentation with 3D pose estimation techniques.

## Features

-   Support for multiple input sources:
    -   Image files
    -   Video files
    -   Webcam
    -   Intel RealSense depth cameras
-   YOLO-based object detection and tracking
-   Instance segmentation for extracting object masks
-   Rotating bounding box extraction
-   6DoF pose estimation (X, Y, Z, Roll, Pitch, Yaw)
-   Enhanced 3D bounding box estimation using depth maps and polygon masks
-   3D bounding box visualization with back face aligned to 2D bounding box
-   3D coordinate axes visualization
-   Real-time performance monitoring
-   Grid display with tracking, segmentation, pose, and depth views

## Usage

### Running the Pipeline

```bash
# Basic usage with webcam
./run_6dof_pipeline.py

# Using a video file
./run_6dof_pipeline.py --input video --source path/to/video.mp4

# Using an image file
./run_6dof_pipeline.py --input image --source path/to/image.jpg

# Using RealSense camera
./run_6dof_pipeline.py --input realsense

# Hide 3D bounding box visualization
./run_6dof_pipeline.py --no-3d-box

# Hide 3D coordinate axes visualization
./run_6dof_pipeline.py --no-axes

# Hide both 3D bounding box and coordinate axes
./run_6dof_pipeline.py --no-3d-box --no-axes

# Use legacy 3D bounding box estimation (for RealSense)
./run_6dof_pipeline.py --input realsense --no-enhanced-bbox
```

### Command Line Arguments

-   `--input`: Input source type (`image`, `video`, `webcam`, `realsense`)
-   `--source`: Path to input source or webcam index
-   `--conf`: Confidence threshold for YOLO detection (default: 0.5)
-   `--iou`: IoU threshold for YOLO NMS (default: 0.5)
-   `--classes`: Filter by specific class IDs (default: [0, 41, 67] - person, cup, cell phone)
-   `--tracking-model`: Path to YOLO tracking model
-   `--segmentation-model`: Path to YOLO segmentation model
-   `--axis-length`: Length of 3D coordinate axes visualization in meters (default: 0.1)
-   `--box-height`: Default height for 3D bounding box in regular cameras (default: 15.0)
-   `--max-box-height`: Maximum height for 3D bounding box in RealSense mode (default: 50.0)
-   `--show-3d-box`: Show 3D bounding box visualization (default: enabled)
-   `--no-3d-box`: Hide 3D bounding box visualization
-   `--show-axes`: Show 3D coordinate axes visualization (default: enabled)
-   `--no-axes`: Hide 3D coordinate axes visualization
-   `--use-enhanced-bbox`: Use enhanced 3D bounding box estimation with depth data (default: enabled)
-   `--no-enhanced-bbox`: Use legacy 3D bounding box estimation method

## Module Structure

-   `input_sources.py`: Classes for different input sources (image, video, webcam, RealSense)
-   `detection.py`: YOLO detection, tracking, and segmentation functionality
-   `pose_estimation.py`: 3D pose estimation and bounding box calculation
-   `transform.py`: Advanced 3D bounding box estimation with depth map integration
-   `visualization.py`: Visualization utilities for rendering results
-   `utils.py`: Utility functions like performance monitoring
-   `main.py`: Main pipeline runner

## Requirements

-   Python 3.10+
-   OpenCV
-   NumPy
-   SciPy
-   Ultralytics YOLO
-   PyRealSense2 (optional, for RealSense camera support)

## License

See the project's LICENSE file for licensing information.
