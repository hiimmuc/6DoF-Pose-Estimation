"""
6DoF Pose Estimation Pipeline

This module provides the main entry point for the 6DoF pose estimation pipeline,
which uses YOLO for object detection, tracking, and segmentation,
combined with depth information for 3D pose computation.
"""

import argparse
import os
import sys
from typing import Optional, Tuple

import cv2
import numpy as np

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.pose_estimator import PoseEstimator
from src.utils.camera import get_default_camera_parameters
from src.utils.input_source import InputSource
from src.utils.visualization import create_visualization_grid


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="6DoF Pose Estimation Pipeline")

    parser.add_argument(
        "--input",
        type=str,
        choices=["image", "video", "webcam", "realsense"],
        default="webcam",
        help="Input source type",
    )

    parser.add_argument(
        "--source", type=str, default="0", help="Path to input source or webcam index"
    )

    parser.add_argument(
        "--tracking_model",
        type=str,
        default="src/checkpoints/YOLO/yolo11n.pt",
        help="Path to YOLO tracking model",
    )

    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="src/checkpoints/YOLO/yolo11n-seg.pt",
        help="Path to YOLO segmentation model",
    )

    parser.add_argument(
        "--detection_conf", type=float, default=0.8, help="Confidence threshold for detection"
    )

    parser.add_argument(
        "--detection_iou", type=float, default=0.8, help="IoU threshold for detection"
    )

    parser.add_argument(
        "--segmentation_conf",
        type=float,
        default=0.8,
        help="Confidence threshold for segmentation",
    )

    parser.add_argument(
        "--segmentation_iou", type=float, default=0.8, help="IoU threshold for segmentation"
    )

    parser.add_argument("--output", type=str, default=None, help="Path to save output video")

    return parser.parse_args()


def main() -> None:
    """
    Main function for the 6DoF pose estimation pipeline.
    """
    # Parse arguments
    args = parse_arguments()

    # Get camera parameters
    camera_matrix, dist_coefs = get_default_camera_parameters()

    # Initialize input source
    input_source = InputSource(args.input, args.source)

    # Initialize pose estimator
    pose_estimator = PoseEstimator(args.tracking_model, args.segmentation_model)

    # Set up output video writer if specified
    output_writer = None
    if args.output:
        # Get first frame to determine size
        _, first_frame, _ = input_source.get_frame()
        if first_frame is not None:
            h, w = first_frame.shape[:2]
            output_writer = cv2.VideoWriter(
                args.output,
                cv2.VideoWriter_fourcc(*"mp4v"),
                30,  # fps
                (w * 2, h * 2),  # Grid size is 2x2 of original frames
            )

    try:
        # Process frames
        while True:
            # Get next frame
            success, rgb_frame, depth_frame = input_source.get_frame()

            if not success or rgb_frame is None:
                print("Failed to get frame. Exiting.")
                break

            # Set up parameters
            detection_params = {"conf": args.detection_conf, "iou": args.detection_iou}
            segmentation_params = {"conf": args.segmentation_conf, "iou": args.segmentation_iou}

            # Process the frame
            results, visualizations = pose_estimator.process_frame(
                rgb_frame,
                camera_matrix,
                dist_coefs,
                depth_frame,
                detection_params,
                segmentation_params,
            )

            # Create visualization grid
            vis_grid = create_visualization_grid(visualizations)

            # Display results
            cv2.imshow("6DoF Pose Estimation", vis_grid)

            # Write to output if specified
            if output_writer is not None:
                output_writer.write(vis_grid)

            # Print pose results to console (optional, can be commented out for less verbosity)
            for result in results:
                print(
                    f"Object {result.object_id} ({result.class_name}): "
                    f"Position: ({result.x:.1f}, {result.y:.1f}, {result.z:.1f}) mm, "
                    f"Rotation: ({result.roll:.1f}, {result.pitch:.1f}, {result.yaw:.1f}) degrees"
                )

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        # Clean up
        input_source.release()
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
