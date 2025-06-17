# 6DoF Pose Estimation Pipeline

A comprehensive pipeline for performing 6DoF (degrees of freedom) pose estimation using object detection, tracking, segmentation, and depth information.

## Features

-   Multiple input sources: image, video, webcam, or Intel RealSense camera
-   YOLO-based object detection and tracking
-   Instance segmentation for accurate mask generation
-   3D pose estimation with roll, pitch, and yaw calculation
-   Performance monitoring for each pipeline stage
-   Visualization of tracking, segmentation, pose, and depth

## Installation

1. Clone this repository:

```bash
git clone <repository-url>
cd 6DoF-PoseEstimation
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. For Intel RealSense support (optional):

```bash
pip install pyrealsense2
```

4. Download YOLO models:

```bash
mkdir -p src/checkpoints/YOLO
# Download models from Ultralytics or use your own pre-trained models
```

## Usage

### Command Line Options

Run the pose estimation pipeline with various input sources:

```bash
# Using a webcam (default)
python examples/pose_estimation_example.py --input webcam --source 0

# Using a video file
python examples/pose_estimation_example.py --input video --source path/to/video.mp4

# Using an image
python examples/pose_estimation_example.py --input image --source path/to/image.jpg

# Using an Intel RealSense camera
python examples/pose_estimation_example.py --input realsense

# Save output to a video file
python examples/pose_estimation_example.py --output output.mp4
```

### Additional Parameters

```bash
# Set detection confidence threshold
--detection_conf 0.6

# Set detection IoU threshold
--detection_iou 0.45

# Set segmentation confidence threshold
--segmentation_conf 0.5

# Set segmentation IoU threshold
--segmentation_iou 0.45

# Specify custom model paths
--tracking_model path/to/tracking/model.pt
--segmentation_model path/to/segmentation/model.pt
```

## Pipeline Overview

The pipeline consists of the following key stages:

1. **Input Handling**: Process frames from images, videos, webcams, or RealSense cameras
2. **Detection and Tracking**: YOLO-based object detection and tracking
3. **Segmentation**: Generate pixel-wise masks for detected objects
4. **Depth Processing**: Extract depth information at mask centers
5. **Pose Estimation**: Calculate 6DoF pose (x, y, z, roll, pitch, yaw)
6. **Visualization**: Display tracking, segmentation, pose, and depth visualizations

## Output Format

The pipeline outputs a list of `PoseResult` objects, each containing:

-   `object_id`: Tracking ID of the object
-   `class_id` and `class_name`: Object class information
-   `confidence`: Detection confidence
-   `x`, `y`, `z`: 3D position in millimeters
-   `roll`, `pitch`, `yaw`: Rotation angles in degrees
-   `bbox_2d`: 2D bounding box coordinates
-   `bbox_rotated`: Rotated bounding box coordinates
-   `mask`: Binary segmentation mask

## Examples

See the `examples/` directory for usage examples:

-   `pose_estimation_example.py`: Complete example of the pipeline
-   `yolo_segmentation.py`: Example of YOLO segmentation
-   `yolo_tracking_loop.py`: Example of YOLO tracking

## License

[Specify your license information here]
