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
-   Programmatic API for integration into other applications
-   Structured output format with 6DoF pose and bounding box data

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
-   `--axis-length`: Length of 3D coordinate axes visualization in meters. Set to 0 for auto-sizing (0.2 of box size) (default: 0)
-   `--box-height`: Default height for 3D bounding box in regular cameras (default: 15.0)
-   `--max-box-height`: Maximum height for 3D bounding box in RealSense mode (default: 50.0)
-   `--show-3d-box`: Show 3D bounding box visualization (default: enabled)
-   `--no-3d-box`: Hide 3D bounding box visualization
-   `--show-axes`: Show 3D coordinate axes visualization (default: enabled)
-   `--no-axes`: Hide 3D coordinate axes visualization
-   `--use-enhanced-bbox`: Use enhanced 3D bounding box estimation with depth data (default: enabled)
-   `--no-enhanced-bbox`: Use legacy 3D bounding box estimation method
-   `--verbose`: Print detailed information to console (default: enabled)
-   `--quiet`: Suppress detailed console output

## Module Structure

-   `input_sources.py`: Classes for different input sources (image, video, webcam, RealSense)
-   `detection.py`: YOLO detection, tracking, and segmentation functionality
-   `pose_estimation.py`: 3D pose estimation and bounding box calculation
-   `transform.py`: Advanced 3D bounding box estimation with depth map integration
-   `visualization.py`: Visualization utilities for rendering results
-   `utils.py`: Utility functions like performance monitoring
-   `main.py`: Main pipeline runner

## Programmatic API

The pipeline provides a programmatic API for integration into other applications. This allows you to get 6DoF pose data without visualization.

```python
from src.pipeline import estimate_poses

# Get pose estimation results
results = estimate_poses(
    input_source_type="webcam",  # "image", "video", "webcam", or "realsense"
    source_path="0",             # Path to image/video or webcam index
    verbose=False,               # Whether to print detailed output
    use_enhanced_bbox=True       # Whether to use enhanced 3D box estimation
)

# Process each detected object
for obj in results["objects"]:
    # Get position data
    x = obj["position"]["x"]
    y = obj["position"]["y"]
    z = obj["position"]["z"]

    # Get rotation data
    roll = obj["rotation"]["roll"]
    pitch = obj["rotation"]["pitch"]
    yaw = obj["rotation"]["yaw"]

    # Get bounding box data
    bbox_2d = obj["bbox_2d"]  # 2D corners as [[x,y], [x,y], ...]
    bbox_3d = obj["bbox_3d"]  # 3D corners as [[x,y,z], [x,y,z], ...]

    # Do something with the object data...
    print(f"Object at ({x:.2f}, {y:.2f}, {z:.2f})")
```

For a complete example, see `examples/programmatic_6dof_api.py`.

## Requirements

-   Python 3.10+
-   OpenCV
-   NumPy
-   SciPy
-   Ultralytics YOLO
-   PyRealSense2 (optional, for RealSense camera support)

## License

See the project's LICENSE file for licensing information.
