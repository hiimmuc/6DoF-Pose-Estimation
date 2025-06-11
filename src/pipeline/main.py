"""
Main module for running the 6DoF pose estimation pipeline.

This module integrates all components: input handling, detection,
pose estimation, and visualization to run the complete pipeline.
"""

import sys
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from .detection import YOLODetector, get_rotating_bounding_box
from .input_sources import RealSenseSource, create_input_source
from .pose_estimation import (
    compute_3d_pose,
    create_3d_bbox,
    create_3d_bbox_enhanced,
    get_euler_angles,
    get_robust_depth,
)
from .utils import PerfMonitor, parse_args
from .visualization import (
    colorize_depth_image,
    create_grid_display,
    draw_3d_axes,
    draw_3d_bbox,
    draw_rotating_box,
)


def process_frame(
    frame: np.ndarray,
    depth_frame: Optional[np.ndarray],
    detector: YOLODetector,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    input_source,
    perf_monitor: PerfMonitor,
    args,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]]:
    """
    Process a single frame through the 6DoF pose estimation pipeline.

    Args:
        frame: Input color frame
        depth_frame: Input depth frame (None for non-RealSense sources)
        detector: YOLODetector instance
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        input_source: Input source instance
        perf_monitor: Performance monitor instance
        args: Command line arguments
        verbose: Whether to print detailed information to console

    Returns:
        Tuple of (results_dict, visualization_tuple) where:
            results_dict: Dictionary containing detection results with 6DoF poses and bounding boxes
            visualization_tuple: Tuple of (tracking_vis, segmentation_vis, pose_vis, depth_vis)
    """
    # Start frame timing
    perf_monitor.start_frame()

    # Run YOLO tracking
    tracking_results = detector.track(frame)
    perf_monitor.mark_stage("tracking")

    # Run YOLO segmentation
    segmentation_results = detector.segment(frame)
    perf_monitor.mark_stage("segmentation")

    # Visualize tracking results
    tracking_vis = tracking_results[0].plot()

    # Visualize segmentation results
    segmentation_vis = segmentation_results[0].plot()

    # Create visualization with rotating bounding box and 3D pose
    pose_vis = frame.copy()

    is_realsense = depth_frame is not None

    # Initialize results dictionary to store 6DoF poses and bounding boxes
    results_dict = {
        "objects": [],
        "timestamp": time.time(),
        "frame_id": getattr(input_source, "frame_count", 0),
    }

    # Process segmentation masks
    if hasattr(segmentation_results[0], "masks") and segmentation_results[0].masks is not None:
        for i, mask in enumerate(segmentation_results[0].masks.data):
            # Convert tensor mask to numpy array
            numpy_mask = mask.cpu().numpy()

            # Extract rotating bounding box
            box_info = get_rotating_bounding_box(numpy_mask)
            if box_info is None:
                continue

            # Draw rotating bounding box
            pose_vis = draw_rotating_box(
                pose_vis, box_info["corners"], color=(255, 0, 255), thickness=2
            )

            # Compute 3D pose
            rvec, tvec = compute_3d_pose(box_info, camera_matrix, dist_coeffs)
            if rvec is None or tvec is None:
                continue

            # Get Euler angles
            roll, pitch, yaw = get_euler_angles(rvec)

            if is_realsense:
                # Process with RealSense depth data
                # Get 3D coordinates
                center_x, center_y = map(int, box_info["center"])
                x, y, _ = input_source.get_point_3d(center_x, center_y, depth_frame)
                z = get_robust_depth(input_source, center_x, center_y, depth_frame)

                # Update translation vector with actual depth data from RealSense
                tvec[0, 0] = x
                tvec[1, 0] = y
                tvec[2, 0] = z

                # Get camera intrinsics for enhanced 3D box estimation
                camera_intrinsics = input_source.get_camera_intrinsics()

                # Create 3D bounding box using the enhanced method with depth information
                if args.use_enhanced_bbox:
                    # Enhanced 3D bounding box using depth map and polygon mask
                    bbox_3d = create_3d_bbox_enhanced(
                        box_info,
                        rvec,
                        tvec,
                        depth_map=depth_frame,
                        camera_intrinsics=camera_intrinsics,
                        depth_scale=input_source.depth_scale,
                        use_polygon_mask=True,
                    )
                else:
                    # Legacy method using point measurements
                    # Get depth at an edge point for thickness calculation
                    edge_x, edge_y = map(int, box_info["corners"][0])
                    edge_z = get_robust_depth(input_source, edge_x, edge_y, depth_frame)

                    # Create 3D bounding box with legacy method
                    bbox_3d = create_3d_bbox(
                        box_info,
                        rvec,
                        use_realsense=True,
                        center_depth=z,
                        edge_depth=edge_z,
                        max_height=args.max_box_height,
                        default_height=args.box_height,
                    )
            else:
                # Process without depth data (regular camera)
                x, y, z = tvec.flatten()

                # Create 3D bounding box with default thickness
                bbox_3d = create_3d_bbox(
                    box_info,
                    rvec,
                    depth=z,
                    use_realsense=False,
                    max_height=args.max_box_height,
                    default_height=args.box_height,
                )

            # Draw 3D bounding box if enabled
            if args.show_3d_box:
                pose_vis = draw_3d_bbox(
                    pose_vis,
                    bbox_3d,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                    color=(0, 255, 255),  # Yellow color for 3D box
                    thickness=2,
                )

            # Draw 3D coordinate axes if enabled
            if args.show_axes:
                # Calculate axis length as 0.2 of the box size
                try:
                    # First determine the box dimensions from 3D bbox corners
                    x_coords = [point[0] for point in bbox_3d]
                    y_coords = [point[1] for point in bbox_3d]
                    z_coords = [point[2] for point in bbox_3d]

                    # Calculate the 3D box dimensions
                    bbox_width = max(x_coords) - min(x_coords)
                    bbox_height = max(y_coords) - min(y_coords)
                    bbox_depth = max(z_coords) - min(z_coords)

                    # Calculate box diagonal length
                    diagonal_length = np.sqrt(bbox_width**2 + bbox_height**2 + bbox_depth**2)

                    # Set axis length to 0.2 of the box size
                    # Use either the box diagonal or the max dimension (whichever is larger)
                    max_dimension = max(bbox_width, bbox_height, bbox_depth)

                    dynamic_axis_length = 0.0002 * max(diagonal_length, max_dimension)
                    # Ensure a reasonable default if calculations result in too small or invalid values
                    if dynamic_axis_length <= 0.01 or not np.isfinite(dynamic_axis_length):
                        dynamic_axis_length = 0.1  # Default fallback

                    # Use the calculated length but allow the user to override it with args
                    axis_length = args.axis_length if args.axis_length > 0 else dynamic_axis_length
                except Exception:
                    # Fallback to a reasonable default in case of any calculation errors
                    axis_length = 0.1

                pose_vis = draw_3d_axes(
                    pose_vis,
                    rvec,
                    tvec,
                    camera_matrix,
                    dist_coeffs,
                    axis_length=axis_length,
                )

            # Display pose information
            center_x, center_y = map(int, box_info["center"])
            text = f"ID: {i} | X: {x:.3f}m Y: {y:.3f}m Z: {z:.3f}m"
            cv2.putText(
                pose_vis,
                text,
                (center_x, center_y - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            euler_text = f"Roll: {roll:.1f} Pitch: {pitch:.1f} Yaw: {yaw:.1f}"
            cv2.putText(
                pose_vis,
                euler_text,
                (center_x, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # Store object information in results dictionary
            object_info = {
                "id": i,
                "position": {"x": float(x), "y": float(y), "z": float(z)},
                "rotation": {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)},
                "bbox_2d": box_info["corners"].tolist(),
                "bbox_3d": bbox_3d.tolist(),
                "center": box_info["center"],
                "size": box_info["size"],
                "angle": float(box_info["angle"]),
                "class_id": (
                    int(segmentation_results[0].boxes.cls[i].item())
                    if hasattr(segmentation_results[0].boxes, "cls")
                    else -1
                ),
                "confidence": (
                    float(segmentation_results[0].boxes.conf[i].item())
                    if hasattr(segmentation_results[0].boxes, "conf")
                    else 1.0
                ),
            }

            # Add object to results
            results_dict["objects"].append(object_info)

            # Print to console if verbose mode is enabled
            if verbose:
                print(
                    f"Object {i}: Position [X:{x:.3f}m Y:{y:.3f}m Z:{z:.3f}m] "
                    f"Rotation [Roll:{roll:.1f}° Pitch:{pitch:.1f}° Yaw:{yaw:.1f}°] "
                    f"BBox: {box_info['corners'].tolist()}"
                )

    # Mark end of pose estimation stage
    perf_monitor.mark_stage("pose_estimation")

    # Process depth frame if available
    depth_vis = None
    if is_realsense and depth_frame is not None:
        depth_vis = colorize_depth_image(depth_frame, input_source.depth_scale)

    # Mark end of visualization stage
    perf_monitor.mark_stage("visualization")

    # End frame timing
    perf_monitor.end_frame()

    # Create tuple of visualization outputs
    vis_tuple = (tracking_vis, segmentation_vis, pose_vis, depth_vis)

    return results_dict, vis_tuple


def run_pipeline(custom_args=None):
    """
    Run the 6DoF Pose Estimation Pipeline.

    This pipeline performs the following steps:
    1. Initializes input source (image, video, webcam, or RealSense)
    2. Runs YOLO tracking and segmentation models
    3. For each detected object:
       - Extracts rotated bounding box from segmentation mask
       - Computes 6DoF pose using PnP algorithm
       - Creates 3D bounding box with dimensions matching 2D detection
       - For RealSense: Uses depth data to calculate box thickness
       - For regular cameras: Uses default thickness of 15 units
    4. Displays results in a 2x2 grid:
       - Tracking view (top-left)
       - Segmentation view (top-right)
       - 3D pose visualization (bottom-left)
       - Depth view (bottom-right) or black frame for non-RealSense

    Args:
        custom_args: Optional custom arguments to override command line parsing
    """
    # Parse command line arguments if not provided
    args = custom_args if custom_args is not None else parse_args()

    # Create performance monitor
    perf_monitor = PerfMonitor(history_size=30)

    # Create YOLO detector
    try:
        print("Loading YOLO models...")
        detector = YOLODetector(
            tracking_model_path=args.tracking_model,
            segmentation_model_path=args.segmentation_model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            classes=args.classes,
        )
    except Exception as e:
        print(f"Error loading YOLO models: {e}")
        return

    # Create input source
    try:
        input_source = create_input_source(args.input, args.source)
    except (ValueError, ImportError) as e:
        print(f"Error creating input source: {str(e)}")
        sys.exit(1)

    # Get camera parameters
    camera_matrix, dist_coeffs = input_source.get_camera_params()
    print(f"Camera matrix:\n{camera_matrix}")  # Print settings if verbose mode is enabled
    if args.verbose:
        print(f"Visualization settings:")
        print(f"  3D bounding box: {'Enabled' if args.show_3d_box else 'Disabled'}")
        print(f"  3D coordinate axes: {'Enabled' if args.show_axes else 'Disabled'}")
        print(f"  Verbose output: {'Enabled' if args.verbose else 'Disabled'}")

        # Print bounding box mode (only relevant for RealSense)
        if args.input == "realsense":
            print(
                f"  3D bounding box mode: {'Enhanced (with depth map)' if args.use_enhanced_bbox else 'Legacy'}"
            )

    try:
        # Main processing loop
        while True:
            # Get frame based on input source type
            if isinstance(input_source, RealSenseSource):
                success, frame, depth_frame = input_source.get_frame()
            else:
                success, frame = input_source.get_frame()
                depth_frame = None

            if not success:
                print("Failed to get frame from input source")
                break

            # Process the frame
            results, vis_outputs = process_frame(
                frame,
                depth_frame,
                detector,
                camera_matrix,
                dist_coeffs,
                input_source,
                perf_monitor,
                args,
                verbose=args.verbose,
            )

            # Extract visualization outputs
            tracking_vis, segmentation_vis, pose_vis, depth_vis = vis_outputs

            # Create grid display
            display = create_grid_display(tracking_vis, segmentation_vis, pose_vis, depth_vis)

            # Optionally print number of detected objects
            if args.verbose and len(results["objects"]) > 0:
                print(f"Detected {len(results['objects'])} objects in frame {results['frame_id']}")

            # Draw performance statistics
            display = perf_monitor.draw_stats(display)

            # Add help text for toggle options
            h, w = display.shape[:2]
            cv2.putText(
                display,
                "Press 'q' to exit | Use --no-3d-box or --no-axes to toggle visualizations",
                (10, h - 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Show the result
            cv2.imshow("6DoF Pose Estimation Pipeline", display)

            # Check for exit key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # For image input, wait until user presses a key
            if args.input == "image":
                cv2.waitKey(0)
                break

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback

        traceback.print_exc()

    finally:
        # Release resources
        input_source.release()
        cv2.destroyAllWindows()


def estimate_poses(
    input_source_type: str = "webcam",
    source_path: str = "0",
    verbose: bool = False,
    use_enhanced_bbox: bool = True,
    **kwargs,
) -> Dict[str, Any]:
    """
    Programmatic API for 6DoF pose estimation without visualization.
    This function processes a single frame from the specified input and returns pose data.

    Args:
        input_source_type: Type of input source ("image", "video", "webcam", "realsense")
        source_path: Path to input source or webcam index
        verbose: Whether to print detailed output
        use_enhanced_bbox: Whether to use enhanced 3D bounding box estimation
        **kwargs: Additional arguments for pose estimation

    Returns:
        Dictionary containing 6DoF pose and bounding box information
    """
    import argparse

    # Create arguments with default values
    args = argparse.Namespace(
        input=input_source_type,
        source=source_path,
        conf=kwargs.get("conf", 0.5),
        iou=kwargs.get("iou", 0.5),
        classes=kwargs.get("classes", [0, 41, 67]),  # person, cup, cell phone
        tracking_model=kwargs.get("tracking_model", "src/checkpoints/YOLO/yolo11n.pt"),
        segmentation_model=kwargs.get("segmentation_model", "src/checkpoints/YOLO/yolo11n-seg.pt"),
        axis_length=kwargs.get("axis_length", 0),  # Default to auto-sizing (0.2 of box size)
        box_height=kwargs.get("box_height", 15.0),
        max_box_height=kwargs.get("max_box_height", 50.0),
        show_3d_box=kwargs.get("show_3d_box", False),
        show_axes=kwargs.get("show_axes", True),
        use_enhanced_bbox=use_enhanced_bbox,
        verbose=verbose,
    )

    # Create performance monitor
    perf_monitor = PerfMonitor(history_size=10)

    # Create YOLO detector
    try:
        detector = YOLODetector(
            tracking_model_path=args.tracking_model,
            segmentation_model_path=args.segmentation_model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            classes=args.classes,
        )
    except Exception as e:
        if verbose:
            print(f"Error loading YOLO models: {e}")
        return {"error": f"Error loading YOLO models: {str(e)}", "objects": []}

    # Create input source
    try:
        input_source = create_input_source(args.input, args.source)
    except (ValueError, ImportError) as e:
        if verbose:
            print(f"Error creating input source: {str(e)}")
        return {"error": f"Error creating input source: {str(e)}", "objects": []}

    # Get camera parameters
    camera_matrix, dist_coeffs = input_source.get_camera_params()

    try:
        # Get frame
        if isinstance(input_source, RealSenseSource):
            success, frame, depth_frame = input_source.get_frame()
        else:
            success, frame = input_source.get_frame()
            depth_frame = None

        if not success:
            return {"error": "Failed to get frame", "objects": []}

        # Process frame
        results, _ = process_frame(
            frame,
            depth_frame,
            detector,
            camera_matrix,
            dist_coeffs,
            input_source,
            perf_monitor,
            args,
            verbose=verbose,
        )

        # Clean up resources
        input_source.release()

        return results

    except Exception as e:
        if verbose:
            print(f"Error during processing: {str(e)}")
            import traceback

            traceback.print_exc()
        return {"error": f"Error during processing: {str(e)}", "objects": []}


# Command-line entry point
if __name__ == "__main__":
    run_pipeline()
