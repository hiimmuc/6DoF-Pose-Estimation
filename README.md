# 6DoF Pose Estimation Pipeline

A modular and efficient pipeline for performing 6DoF (degrees of freedom) pose estimation using YOLO for object detection, tracking, and segmentation, combined with depth information for accurate 3D pose computation.

## Features

-   **Modular Architecture**: Clean, maintainable code with clear separation of concerns
-   **Multiple Input Sources**: Support for images, videos, webcams, and Intel RealSense cameras
-   **YOLO-based Detection**: High-performance object detection and tracking
-   **Instance Segmentation**: Accurate pixel-wise mask generation for precise object boundaries
-   **3D Pose Estimation**: Complete 6DoF pose calculation (x, y, z, roll, pitch, yaw)
-   **3D Bounding Boxes**: Visualization of objects with 3D bounding boxes
-   **Real-time Performance**: Optimized for real-time applications with performance monitoring
-   **Comprehensive Visualization**: Multi-panel display with tracking, segmentation, pose, and depth views

## Project Structure

```
src/
  ├── core/
  │   ├── __init__.py
  │   └── pose_estimator.py      # Core pose estimation logic
  ├── data/
  │   ├── __init__.py
  │   └── pose_result.py         # Data structures for pose results
  ├── utils/
  │   ├── __init__.py
  │   ├── camera.py              # Camera parameter utilities
  │   ├── input_source.py        # Input handling for various sources
  │   ├── performance.py         # Performance monitoring
  │   └── visualization.py       # Visualization utilities with 3D boxes
  └── main.py                    # Main entry point
examples/
  ├── pose_estimation_example.py # Complete pipeline example
  ├── realsense_pose_estimation.py # RealSense-specific example
  └── yolo_tracking_loop.py      # YOLO tracking example
```

## Installation

1. Clone this repository:

```bash
git clone https://github.com/hiimmuc/6DoF-Pose-Estimation.git
cd 6DoF-PoseEstimation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For Intel RealSense support (optional):

```bash
cd dependencies
./install_realsense_sdk.sh
pip install pyrealsense2
```

4. Download YOLO models:

```bash
mkdir -p src/checkpoints/YOLO
cd src/checkpoints/YOLO
# YOLOv8 models
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-seg.pt
```

## Quick Start

The simplest way to run the pipeline is using our convenience script:

```bash
# Using default webcam
python run_pipeline.py

# Using Intel RealSense camera
python run_pipeline.py --input realsense

# Using a video file
python run_pipeline.py --input video --source path/to/video.mp4

# Save output to a video file
python run_pipeline.py --output results.mp4
```

## Usage Options

### Input Sources

```bash
# Using a webcam (default)
python src/main.py --input webcam --source 0

# Using a video file
python src/main.py --input video --source path/to/video.mp4

# Using an image file
python src/main.py --input image --source path/to/image.jpg

# Using an Intel RealSense camera
python src/main.py --input realsense
```

### Detection Parameters

```bash
# Adjust confidence thresholds
python src/main.py --detection_conf 0.6 --segmentation_conf 0.5

# Adjust IoU thresholds
python src/main.py --detection_iou 0.45 --segmentation_iou 0.45

# Use custom models
python src/main.py --tracking_model path/to/model.pt --segmentation_model path/to/seg_model.pt
```

### Example Applications

```bash
# RealSense specific example with depth visualization
python examples/realsense_pose_estimation.py

# Basic pose estimation with webcam
python examples/pose_estimation_example.py

# Simple YOLO tracking demo
python examples/yolo_tracking_loop.py
```

## Pipeline Architecture

The pipeline follows a modular design with the following stages:

![Pipeline Architecture](Docs/rgb_flow.drawio)

1. **Input Source Handling**

    - Abstract interface for various input sources
    - Transparent handling of RGB and depth streams
    - Automatic resource management

2. **Object Detection & Tracking**

    - YOLO-based detection and tracking
    - Persistent object IDs across frames
    - High-performance inference with GPU acceleration

3. **Instance Segmentation**

    - Pixel-precise object masks
    - Integration with tracking for consistent instance IDs

4. **Pose Estimation**

    - Depth-based 3D position calculation
    - PnP-based orientation estimation
    - Complete 6DoF pose with roll, pitch, yaw

5. **3D Box Generation**

    - Dynamic 3D bounding box creation
    - Depth-aware box projection
    - Direction estimation for box orientation

6. **Performance Monitoring & Visualization**
    - Real-time performance metrics for each pipeline stage
    - Multi-panel visualization with compact overlay
    - Color-coded FPS indicators for performance evaluation

## PoseResult Format

Each detected object produces a `PoseResult` object with the following attributes:

| Attribute              | Type        | Description                              |
| ---------------------- | ----------- | ---------------------------------------- |
| `object_id`            | int         | Unique tracking identifier               |
| `class_id`             | int         | Object class index                       |
| `class_name`           | str         | Human-readable class name                |
| `confidence`           | float       | Detection confidence score               |
| `x`, `y`, `z`          | float       | 3D position in millimeters               |
| `roll`, `pitch`, `yaw` | float       | Rotation angles in degrees               |
| `bbox_2d`              | List[float] | 2D bounding box [x, y, width, height]    |
| `bbox_rotated`         | List[float] | Rotated bounding box [x, y, w, h, angle] |
| `bbox_3d`              | np.ndarray  | 3D bounding box corner points            |
| `mask`                 | np.ndarray  | Binary segmentation mask                 |

## 3D Bounding Box Visualization

The pipeline provides enhanced visualization with 3D bounding boxes that represent the estimated volume and orientation of detected objects.

### Features

-   **Depth-Aware**: Box dimensions are proportional to the object's depth
-   **Direction Estimation**: Automatically determines the front face of objects
-   **Visual Cues**: Color-coded boxes help distinguish between different parts of the object
-   **Adaptive Sizing**: Box dimensions adapt to the object's dimensions in the image

### How It Works

1. **Depth Calculation**: The central depth of the object is determined from the depth map
2. **Direction Estimation**: Front-facing direction is calculated by analyzing depth gradients
3. **Box Generation**: 3D box points are generated using depth and directional information
4. **Perspective Rendering**: The 3D box is projected onto the 2D image plane

### Example

```python
# The 3D box is automatically generated when depth information is available
from src.utils.visualization import generate_3d_box_points, draw_3d_box

# Given a bounding box, depth map and direction vector
bbox = [x1, y1, x2, y2]  # [left, top, right, bottom]
depth = get_deepest_narrowest_depth(depth_map, bbox)
direction = get_front_direction(depth_map, bbox)

# Generate and draw 3D box
box_3d_points = generate_3d_box_points(bbox, depth, direction)
draw_3d_box(image, box_3d_points, color=(255, 165, 0))
```

## Extending the Pipeline

### Adding Custom Detection Models

```python
# In src/main.py
parser.add_argument("--tracking_model", type=str, default="path/to/custom/model.pt")
```

### Implementing Custom Pose Estimation Logic

```python
# Extend the PoseEstimator class
from src.core.pose_estimator import PoseEstimator

class CustomPoseEstimator(PoseEstimator):
    def _calculate_orientation(self, corners_3d, bbox_rotated, camera_matrix, dist_coefs):
        # Custom orientation calculation logic
        # ...
        return custom_roll, custom_pitch, custom_yaw
```

## Performance Considerations

The pipeline is designed for real-time operation but performance depends on hardware and configuration:

### Optimization Tips

-   **Model Selection**: Use smaller YOLO models (nano or small) for better FPS
-   **Resolution**: Lower input resolution significantly improves performance
-   **Selective Processing**: Adjust confidence thresholds to limit detections
-   **Hardware Acceleration**: CUDA GPUs provide significant performance gains

### Performance Metrics

The built-in performance monitor provides real-time metrics for each pipeline stage:

| Pipeline Stage  | Typical FPS (RTX 3080) | Typical FPS (CPU only) |
| --------------- | ---------------------- | ---------------------- |
| Tracking        | 25-30 FPS              | 5-10 FPS               |
| Segmentation    | 20-25 FPS              | 3-7 FPS                |
| Pose Estimation | 50-60 FPS              | 15-30 FPS              |
| Visualization   | 70-80 FPS              | 40-60 FPS              |
| **Overall**     | **15-20 FPS**          | **3-5 FPS**            |

_Note: These are approximate values and depend on specific hardware configurations_

## Prerequisites

-   Python 3.8 or higher
-   CUDA-capable GPU recommended for real-time performance
-   Intel RealSense camera (optional, for depth-based applications)

## Dependencies

The following key libraries are used:

-   **ultralytics**: For YOLO-based detection, tracking, and segmentation
-   **OpenCV**: For image processing and computer vision operations
-   **NumPy**: For numerical operations
-   **PyRealsense2**: For Intel RealSense camera support (optional)

See `requirements.txt` for the complete list of dependencies:

```
numpy>=1.22.0
opencv-python>=4.6.0
ultralytics>=0.0.0
PyYAML>=6.0
matplotlib>=3.5.0
scipy>=1.7.0
tqdm>=4.62.0
pyrealsense2>=2.50.0  # Optional, for RealSense support
```

## Troubleshooting

### Common Issues

1. **"No module named 'pyrealsense2'"**

    - Install the pyrealsense2 package: `pip install pyrealsense2`
    - On Linux, you may need to install the RealSense SDK first

2. **Low FPS Performance**

    - Try using a smaller YOLO model variant
    - Reduce input resolution
    - Ensure GPU acceleration is properly configured

3. **"TypeError: No depth image provided"**

    - When using non-depth cameras, some features like 3D bounding boxes won't be available
    - For full functionality, use a depth camera like Intel RealSense

4. **"CUDA out of memory"**
    - Reduce batch size or use a smaller model
    - Close other GPU-intensive applications

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## Acknowledgments

-   [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO implementation
-   [OpenCV](https://opencv.org/) for computer vision algorithms
-   [Intel RealSense](https://www.intelrealsense.com/) for depth camera support

---

For questions or support, please contact VINMOTION team.

## License

Copyright © 2025 VINMOTION

[Specify your license information here]
