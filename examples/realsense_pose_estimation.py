"""
RealSense 6DoF Pose Estimation Example

This example demonstrates how to use the Intel RealSense camera
with the 6DoF pose estimation pipeline. It handles the depth data
to provide accurate 3D pose estimates.
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

# from src.utils.camera import get_default_camera_parameters
from src.utils.visualization import create_visualization_grid


class RealSenseHandler:
    """
    Handler for Intel RealSense cameras, providing RGB and aligned depth frames.

    This class initializes a RealSense camera, configures it for RGB and depth
    streaming, and provides methods to get aligned frames.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        """
        Initialize the RealSense camera handler.

        Args:
            width: Width of the camera frames.
            height: Height of the camera frames.
            fps: Frames per second.
        """
        try:
            import pyrealsense2 as rs

            self.rs = rs
        except ImportError:
            raise ImportError(
                "pyrealsense2 is not installed. Install it using: " "pip install pyrealsense2"
            )

        self.width = width
        self.height = height
        self.fps = fps

        # Initialize RealSense pipeline and config
        self.pipeline = self.rs.pipeline()
        self.config = self.rs.config()

        # Enable color and depth streams
        self.config.enable_stream(self.rs.stream.color, width, height, self.rs.format.bgr8, fps)
        self.config.enable_stream(self.rs.stream.depth, width, height, self.rs.format.z16, fps)

        # Start the pipeline
        self.profile = self.pipeline.start(self.config)

        # Create alignment object
        align_to = self.rs.stream.color
        self.align = self.rs.align(align_to)

        # Give the camera some time to warm up
        for _ in range(5):
            self.pipeline.wait_for_frames()

    def get_intrinsics(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the camera intrinsic parameters from the RealSense camera.

        Returns:
            Tuple containing:
                - Camera intrinsic matrix (3x3)
                - Distortion coefficients
        """
        # Get the depth sensor's depth scale
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        # Get intrinsics from the color stream
        color_stream = self.profile.get_stream(self.rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        # Create camera matrix
        camera_matrix = np.array(
            [[intrinsics.fx, 0, intrinsics.ppx], [0, intrinsics.fy, intrinsics.ppy], [0, 0, 1]]
        )

        # Create distortion coefficients
        dist_coefs = np.array(
            [
                intrinsics.coeffs[0],  # k1
                intrinsics.coeffs[1],  # k2
                intrinsics.coeffs[2],  # p1
                intrinsics.coeffs[3],  # p2
                intrinsics.coeffs[4],  # k3
            ]
        )

        return camera_matrix, dist_coefs

    def get_aligned_frames(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get aligned color and depth frames from the RealSense camera.

        Returns:
            Tuple containing:
                - success (bool): Whether frames were successfully acquired
                - rgb_frame (Optional[np.ndarray]): RGB frame (None if unsuccessful)
                - depth_frame (Optional[np.ndarray]): Depth frame in mm (None if unsuccessful)
        """
        try:
            # Wait for a coherent pair of frames
            frames = self.pipeline.wait_for_frames()

            # Align depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            # Check if frames are valid
            if not color_frame or not depth_frame:
                return False, None, None

            # Convert frames to numpy arrays
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Convert depth to mm (from the depth scale)
            # depth_image = depth_image * self.depth_scale * 1000

            return True, rgb_image, depth_image

        except Exception as e:
            print(f"Error getting frames: {str(e)}")
            return False, None, None

    def release(self) -> None:
        """Release resources and stop the RealSense pipeline."""
        if hasattr(self, "pipeline"):
            self.pipeline.stop()


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="RealSense 6DoF Pose Estimation Example")

    parser.add_argument(
        "--tracking_model",
        type=str,
        default="src/checkpoints/YOLO/yolo11m.pt",
        help="Path to YOLO tracking model",
    )

    parser.add_argument(
        "--segmentation_model",
        type=str,
        default="src/checkpoints/YOLO/yolo11m-seg.pt",
        help="Path to YOLO segmentation model",
    )

    parser.add_argument(
        "--detection_conf", type=float, default=0.7, help="Confidence threshold for detection"
    )

    parser.add_argument(
        "--detection_iou", type=float, default=0.7, help="IoU threshold for detection"
    )

    parser.add_argument(
        "--segmentation_conf",
        type=float,
        default=0.7,
        help="Confidence threshold for segmentation",
    )

    parser.add_argument(
        "--segmentation_iou", type=float, default=0.7, help="IoU threshold for segmentation"
    )

    parser.add_argument("--output", type=str, default=None, help="Path to save output video")

    parser.add_argument("--width", type=int, default=640, help="Width of RealSense camera frames")

    parser.add_argument(
        "--height", type=int, default=480, help="Height of RealSense camera frames"
    )

    parser.add_argument("--fps", type=int, default=30, help="FPS of RealSense camera")

    return parser.parse_args()


def main() -> None:
    """Main function for the RealSense 6DoF pose estimation example."""
    # Parse arguments
    args = parse_arguments()

    print("Initializing RealSense camera...")
    try:
        # Initialize RealSense handler
        realsense = RealSenseHandler(width=args.width, height=args.height, fps=args.fps)

        # Get camera intrinsics
        camera_matrix, dist_coefs = realsense.get_intrinsics()
        print("RealSense camera initialized successfully")
        print(f"Camera matrix:\n{camera_matrix}")

    except Exception as e:
        print(f"Error initializing RealSense camera: {str(e)}")
        print("Make sure the camera is connected and pyrealsense2 is installed")
        return

    try:
        # Initialize pose estimator
        print("Loading YOLO models...")
        pose_estimator = PoseEstimator(args.tracking_model, args.segmentation_model)
        print("Models loaded successfully")

        # Set up output video writer if specified
        output_writer = None
        if args.output:
            # Get first frame to determine size
            _, first_frame, _ = realsense.get_aligned_frames()
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                output_writer = cv2.VideoWriter(
                    args.output,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    30,  # fps
                    (w * 2, h * 2),  # Grid size is 2x2 of original frames
                )
                print(f"Writing output to: {args.output}")

        print("Starting main loop. Press 'q' to exit.")

        # Process frames
        while True:
            # Get aligned RGB and depth frames
            success, rgb_frame, depth_frame = realsense.get_aligned_frames()

            if not success or rgb_frame is None:
                print("Failed to get frames. Retrying...")
                continue

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
            cv2.imshow("RealSense 6DoF Pose Estimation", vis_grid)

            # Write to output if specified
            if output_writer is not None:
                output_writer.write(vis_grid)

            # Print pose results to console (only if objects are detected)
            if results:
                print(f"\nDetected {len(results)} objects:")
                for result in results:
                    print(
                        f"  Object {result.object_id} ({result.class_name}): "
                        f"Position: ({result.x:.1f}, {result.y:.1f}, {result.z:.1f}) mm, "
                        f"Rotation: ({result.roll:.1f}, {result.pitch:.1f}, {result.yaw:.1f}) degrees"
                    )

            # Check for exit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Exiting...")
                break

    except KeyboardInterrupt:
        print("Interrupted by user")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Clean up
        realsense.release()
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
