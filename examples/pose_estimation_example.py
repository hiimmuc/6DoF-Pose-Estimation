"""
6DoF Pose Estimation Example

This script demonstrates the usage of the 6DoF pose estimation pipeline.
It can process input from various sources (image, video, webcam, RealSense)
and visualize the detected objects along with their 6DoF pose.
"""

import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

from src.core.pose_estimator import PoseEstimator
from src.main import parse_arguments
from src.utils.camera import get_default_camera_parameters
from src.utils.input_source import InputSource
from src.utils.visualization import create_visualization_grid


def main():
    """
    Main function for the 6DoF Pose Estimation example.
    """
    # Parse command-line arguments
    args = parse_arguments()

    print(f"Using input type: {args.input} from source: {args.source}")
    print(f"Using tracking model: {args.tracking_model}")
    print(f"Using segmentation model: {args.segmentation_model}")

    # Get camera parameters (in a real application, these would be calibrated)
    camera_matrix, dist_coefs = get_default_camera_parameters()

    # Initialize input source
    try:
        input_source = InputSource(args.input, args.source)
    except Exception as e:
        print(f"Error initializing input source: {str(e)}")
        return

    # Initialize pose estimator
    try:
        pose_estimator = PoseEstimator(args.tracking_model, args.segmentation_model)
    except Exception as e:
        print(f"Error initializing pose estimator: {str(e)}")
        input_source.release()
        return

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
            print(f"Writing output to: {args.output}")

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

    finally:
        # Clean up
        input_source.release()
        if output_writer is not None:
            output_writer.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
