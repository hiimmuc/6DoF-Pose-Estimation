"""
Usage:
$ python traditional_method.py --input <image|video|webcam|realsense> [--path <path_to_file_or_device_id>] [--output <output_video_path>] [--segmentation <threshold|otsu|canny|color_range>] [--min-area <min_contour_area>] [--fps <display_fps>] [--no-display] [--show-grid] [--objects <object_specs>]

Options:
--input: Type of input source (image, video, webcam, realsense)
--path: Path to image/video file or webcam device ID (if applicable)
--output: Output video file path (for video/webcam/realsense input)
--segmentation: Method for object segmentation (default: threshold). Choices are threshold, otsu, canny, color_range.
--min-area: Minimum contour area to consider (default: 500)
--fps: Display FPS for video/webcam (default: 30)
--no-display: Disable display window
--show-grid: Show 2x2 visualization grid instead of single annotated image
--objects: Object specifications in format "name:width,height,depth;name2:w,h,d" (default: default:0.1,0.05,0.02)

"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

# Try to import pyrealsense2 for RealSense support
try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("pyrealsense2 not available. RealSense input will not work.")


class ObjectInfo:
    """Class to store object information for pose estimation"""

    def __init__(
        self, name: str, dimensions: Tuple[float, float, float], color: Tuple[int, int, int] = None
    ):
        """
        Initialize object information

        Args:
            name: object name/identifier
            dimensions: (width, height, depth) in meters
            color: BGR color for visualization
        """
        self.name = name
        self.width, self.height, self.depth = dimensions
        self.color = color if color else (0, 255, 0)  # Default green

        # Define 3D object points (centered at origin)
        self.object_3d_points = np.array(
            [
                [-self.width / 2, -self.height / 2, 0],
                [self.width / 2, -self.height / 2, 0],
                [self.width / 2, self.height / 2, 0],
                [-self.width / 2, self.height / 2, 0],
                [-self.width / 2, -self.height / 2, -self.depth],
                [self.width / 2, -self.height / 2, -self.depth],
                [self.width / 2, self.height / 2, -self.depth],
                [-self.width / 2, self.height / 2, -self.depth],
            ],
            dtype=np.float32,
        )


class MultiObjectPoseEstimator:
    """
    6DOF Multi-Object Pose Estimation using segmentation -> contour -> rotated bounding box
    """

    def __init__(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray):
        """
        Initialize pose estimator

        Args:
            camera_matrix: 3x3 camera intrinsic matrix
            dist_coeffs: distortion coefficients
        """
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.objects = {}  # Dictionary to store object information

    def add_object(self, obj_info: ObjectInfo):
        """Add an object to track"""
        self.objects[obj_info.name] = obj_info

    def segment_objects(
        self, image: np.ndarray, method: str = "threshold"
    ) -> Dict[str, np.ndarray]:
        """
        Segment objects from image using various methods

        Args:
            image: input image
            method: segmentation method ('threshold', 'contour', 'grabcut', 'color_range')

        Returns:
            dictionary of binary masks for different objects
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            gray = image.copy()
            hsv = None

        masks = {}

        if method == "threshold":
            # Adaptive threshold
            mask = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
            )
            masks["general"] = mask

        elif method == "otsu":
            # Otsu's thresholding
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks["general"] = mask

        elif method == "canny":
            # Canny edge detection + morphological operations
            edges = cv2.Canny(gray, 50, 150)
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            masks["general"] = mask

        elif method == "color_range" and hsv is not None:
            # Color-based segmentation for multiple objects
            # Define color ranges for different objects (can be customized)
            color_ranges = {
                "red_object": [(0, 50, 50), (10, 255, 255)],
                "blue_object": [(100, 50, 50), (130, 255, 255)],
                "green_object": [(40, 50, 50), (80, 255, 255)],
                "yellow_object": [(20, 50, 50), (40, 255, 255)],
            }

            for obj_name, (lower, upper) in color_ranges.items():
                lower = np.array(lower, dtype=np.uint8)
                upper = np.array(upper, dtype=np.uint8)
                mask = cv2.inRange(hsv, lower, upper)

                # Clean up mask
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

                # Only add if mask has significant content
                if cv2.countNonZero(mask) > 100:
                    masks[obj_name] = mask
        else:
            # Simple threshold
            _, mask = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            masks["general"] = mask

        # Clean up masks
        for name, mask in masks.items():
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            masks[name] = mask

        return masks

    def extract_multiple_contours(
        self, mask: np.ndarray, min_area: float = 500
    ) -> List[Tuple[np.ndarray, float]]:
        """
        Extract multiple contours from binary mask

        Args:
            mask: binary mask
            min_area: minimum contour area to consider

        Returns:
            list of (contour, area) tuples sorted by area (largest first)
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # Filter contours by minimum area and sort by area (largest first)
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                valid_contours.append((contour, area))

        # Sort by area (largest first)
        valid_contours.sort(key=lambda x: x[1], reverse=True)

        return valid_contours

    def get_rotated_bounding_box(self, contour: np.ndarray) -> Dict[str, Any]:
        """
        Get minimum rotated bounding box from contour

        Args:
            contour: object contour

        Returns:
            dictionary with box info (center, size, angle, corners)
        """
        # Get minimum area rectangle
        rect = cv2.minAreaRect(contour)

        # Extract rectangle parameters
        center = rect[0]  # (x, y)
        size = rect[1]  # (width, height)
        angle = rect[2]  # rotation angle in degrees

        # Get box corners
        box_points = cv2.boxPoints(rect)
        box_points = np.intp(box_points)

        return {
            "center": center,
            "size": size,
            "angle": angle,
            "corners": box_points,
            "rect": rect,
        }

    def estimate_pose_from_rotated_box(
        self, box_info: Dict[str, Any], obj_info: ObjectInfo
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate 6DOF pose from rotated bounding box

        Args:
            box_info: rotated bounding box information
            obj_info: object information

        Returns:
            rotation vector and translation vector
        """
        # Get 2D image points from box corners
        image_points = box_info["corners"].astype(np.float32)

        # Ensure we have 4 points for the top face of the object
        if len(image_points) != 4:
            raise ValueError("Expected 4 corner points for pose estimation")

        # Use only the top 4 points of the 3D object model
        object_points = obj_info.object_3d_points[:4]

        # Solve PnP problem
        pnp_result = cv2.solvePnP(
            object_points, image_points, self.camera_matrix, self.dist_coeffs
        )

        # Handle different OpenCV versions
        if len(pnp_result) == 3:
            success, rvec, tvec = pnp_result
        else:
            success, rvec, tvec = pnp_result[:3]

        if not success:
            raise ValueError("PnP solver failed")

        return rvec, tvec

    def refine_pose_with_iterative_pnp(
        self,
        box_info: Dict[str, Any],
        obj_info: ObjectInfo,
        initial_rvec: np.ndarray,
        initial_tvec: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Refine pose estimation using iterative PnP

        Args:
            box_info: rotated bounding box information
            obj_info: object information
            initial_rvec: initial rotation vector
            initial_tvec: initial translation vector

        Returns:
            refined rotation and translation vectors
        """
        image_points = box_info["corners"].astype(np.float32)
        object_points = obj_info.object_3d_points[:4]

        # Use iterative PnP for refinement
        pnp_result = cv2.solvePnPRansac(
            object_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            rvec=initial_rvec,
            tvec=initial_tvec,
            useExtrinsicGuess=True,
            iterationsCount=1000,
            reprojectionError=2.0,
        )

        # Handle different OpenCV versions
        if len(pnp_result) >= 3:
            success, rvec, tvec = pnp_result[:3]
        else:
            success = False

        if not success:
            return initial_rvec, initial_tvec

        return rvec, tvec

    def convert_to_pose_matrix(self, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
        """
        Convert rotation and translation vectors to 4x4 pose matrix

        Args:
            rvec: rotation vector
            tvec: translation vector

        Returns:
            4x4 pose transformation matrix
        """
        # Convert rotation vector to rotation matrix
        rmat = cv2.Rodrigues(rvec)[0]

        # Create 4x4 transformation matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rmat
        pose_matrix[:3, 3] = tvec.flatten()

        return pose_matrix

    def get_euler_angles(self, rvec: np.ndarray) -> Tuple[float, float, float]:
        """
        Convert rotation vector to Euler angles (roll, pitch, yaw)

        Args:
            rvec: rotation vector

        Returns:
            roll, pitch, yaw angles in degrees
        """
        rmat, _ = cv2.Rodrigues(rvec)
        r = R.from_matrix(rmat)
        angles = r.as_euler("xyz", degrees=True)

        return angles[0], angles[1], angles[2]  # roll, pitch, yaw

    def estimate_multiple_poses(
        self,
        image: np.ndarray,
        segmentation_method: str = "threshold",
        min_contour_area: float = 500,
    ) -> Dict[str, Any]:
        """
        Complete 6DOF pose estimation pipeline for multiple objects

        Args:
            image: input image
            segmentation_method: method for object segmentation
            min_contour_area: minimum contour area to consider

        Returns:
            dictionary with pose estimation results for all objects
        """
        results = {
            "success": True,
            "masks": {},
            "contour_masks": {},
            "objects": {},
            "total_objects_found": 0,
        }

        try:
            # Step 1: Segment objects
            masks = self.segment_objects(image, segmentation_method)
            results["masks"] = masks

            # Process each mask
            for mask_name, mask in masks.items():
                # Step 2: Extract multiple contours
                contours_info = self.extract_multiple_contours(mask, min_contour_area)

                if not contours_info:
                    continue

                # Create contour visualization mask
                contour_mask = np.zeros_like(mask)
                for i, (contour, area) in enumerate(contours_info):
                    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

                results["contour_masks"][mask_name] = contour_mask

                # Process each contour as a potential object
                mask_objects = []
                for i, (contour, area) in enumerate(contours_info):
                    try:
                        # Step 3: Get rotated bounding box
                        box_info = self.get_rotated_bounding_box(contour)

                        # Find best matching object based on size
                        best_obj = self._find_best_matching_object(box_info)

                        if best_obj is None:
                            # Use default object if no specific match
                            if "default" not in self.objects:
                                continue
                            best_obj = self.objects["default"]

                        # Step 4: Estimate pose
                        rvec, tvec = self.estimate_pose_from_rotated_box(box_info, best_obj)

                        # Step 5: Refine pose (optional)
                        rvec_refined, tvec_refined = self.refine_pose_with_iterative_pnp(
                            box_info, best_obj, rvec, tvec
                        )

                        # Step 6: Convert to various representations
                        pose_matrix = self.convert_to_pose_matrix(rvec_refined, tvec_refined)
                        roll, pitch, yaw = self.get_euler_angles(rvec_refined)

                        object_result = {
                            "success": True,
                            "object_info": best_obj,
                            "contour": contour,
                            "contour_area": area,
                            "rotated_box": box_info,
                            "rotation_vector": rvec_refined,
                            "translation_vector": tvec_refined,
                            "pose_matrix": pose_matrix,
                            "euler_angles": {"roll": roll, "pitch": pitch, "yaw": yaw},
                            "position": {
                                "x": tvec_refined[0][0],
                                "y": tvec_refined[1][0],
                                "z": tvec_refined[2][0],
                            },
                        }

                        mask_objects.append(object_result)

                    except Exception as e:
                        # Continue with other contours if one fails
                        continue

                if mask_objects:
                    results["objects"][mask_name] = mask_objects
                    results["total_objects_found"] += len(mask_objects)

        except Exception as e:
            results["success"] = False
            results["error"] = str(e)

        return results

    def _find_best_matching_object(self, box_info: Dict[str, Any]) -> Optional[ObjectInfo]:
        """
        Find the best matching object based on bounding box size

        Args:
            box_info: rotated bounding box information

        Returns:
            best matching ObjectInfo or None
        """
        if not self.objects:
            return None

        box_size = box_info["size"]  # (width, height)
        box_area = box_size[0] * box_size[1]

        best_match = None
        best_score = float("inf")

        for obj_info in self.objects.values():
            # Compare with object's projected size (simple heuristic)
            obj_projected_area = (obj_info.width * obj_info.height) * 10000  # Rough scale factor
            score = abs(box_area - obj_projected_area) / obj_projected_area

            if score < best_score:
                best_score = score
                best_match = obj_info

        return best_match

    def visualize_results(
        self, image: np.ndarray, results: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Visualize pose estimation results

        Args:
            image: original image
            results: pose estimation results

        Returns:
            tuple of (annotated_image, combined_mask, combined_contour_mask)
        """
        if not results["success"]:
            return image, np.zeros_like(image[:, :, 0]), np.zeros_like(image[:, :, 0])

        vis_image = image.copy()

        # Create combined masks for visualization
        combined_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        combined_contour_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Combine all masks
        for mask_name, mask in results["masks"].items():
            combined_mask = cv2.bitwise_or(combined_mask, mask)

        for mask_name, contour_mask in results["contour_masks"].items():
            combined_contour_mask = cv2.bitwise_or(combined_contour_mask, contour_mask)

        # Draw pose information for each detected object
        object_count = 0
        for mask_name, objects_list in results["objects"].items():
            for obj_result in objects_list:
                if not obj_result["success"]:
                    continue

                object_count += 1
                obj_info = obj_result["object_info"]

                # Draw rotated bounding box
                box_corners = obj_result["rotated_box"]["corners"]
                cv2.drawContours(vis_image, [box_corners], -1, obj_info.color, 2)

                # Draw coordinate axes
                axis_length = max(obj_info.width, obj_info.height) * 0.5
                axis_points = np.array(
                    [
                        [0, 0, 0],
                        [axis_length, 0, 0],  # X-axis (red)
                        [0, axis_length, 0],  # Y-axis (green)
                        [0, 0, -axis_length],  # Z-axis (blue)
                    ],
                    dtype=np.float32,
                )

                projected_axes, _ = cv2.projectPoints(
                    axis_points,
                    obj_result["rotation_vector"],
                    obj_result["translation_vector"],
                    self.camera_matrix,
                    self.dist_coeffs,
                )

                projected_axes = projected_axes.reshape(-1, 2).astype(int)
                origin = tuple(projected_axes[0])

                # Draw axes
                cv2.arrowedLine(
                    vis_image, origin, tuple(projected_axes[1]), (0, 0, 255), 2
                )  # X-red
                cv2.arrowedLine(
                    vis_image, origin, tuple(projected_axes[2]), (0, 255, 0), 2
                )  # Y-green
                cv2.arrowedLine(
                    vis_image, origin, tuple(projected_axes[3]), (255, 0, 0), 2
                )  # Z-blue

                # Add object label
                center = obj_result["rotated_box"]["center"]
                label = f"{obj_info.name}_{object_count}"
                cv2.putText(
                    vis_image,
                    label,
                    (int(center[0] - 20), int(center[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    obj_info.color,
                    2,
                )

        # Add summary text
        total_objects = results["total_objects_found"]
        cv2.putText(
            vis_image,
            f"Objects detected: {total_objects}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return vis_image, combined_mask, combined_contour_mask

    def create_visualization_grid(
        self,
        original: np.ndarray,
        annotated: np.ndarray,
        mask: np.ndarray,
        contour_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Create a 2x2 grid visualization showing all results

        Args:
            original: original image
            annotated: annotated image with pose visualization
            mask: combined segmentation mask
            contour_mask: combined contour mask

        Returns:
            combined visualization grid
        """
        # Resize images to same size
        h, w = original.shape[:2]
        target_h, target_w = h // 2, w // 2

        # Resize all images
        orig_small = cv2.resize(original, (target_w, target_h))
        annot_small = cv2.resize(annotated, (target_w, target_h))

        # Convert masks to 3-channel for concatenation
        mask_colored = cv2.applyColorMap(cv2.resize(mask, (target_w, target_h)), cv2.COLORMAP_JET)
        contour_colored = cv2.applyColorMap(
            cv2.resize(contour_mask, (target_w, target_h)), cv2.COLORMAP_JET
        )

        # Add labels
        cv2.putText(
            orig_small, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )
        cv2.putText(
            annot_small,
            "Pose Estimation",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            mask_colored,
            "Segmentation Mask",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            contour_colored,
            "Contour Mask",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Create 2x2 grid
        top_row = np.hstack([orig_small, annot_small])
        bottom_row = np.hstack([mask_colored, contour_colored])
        grid = np.vstack([top_row, bottom_row])

        return grid


class InputSource:
    """Base class for different input sources"""

    def __init__(self):
        self.fps = 30

    def get_frame(self):
        raise NotImplementedError

    def release(self):
        pass

    def get_camera_params(self):
        """Return default camera parameters"""
        camera_matrix = np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]], dtype=np.float32)

        dist_coeffs = np.array([0.1, -0.2, 0, 0, 0], dtype=np.float32)

        return camera_matrix, dist_coeffs


class ImageSource(InputSource):
    """Input source for single images"""

    def __init__(self, image_path: str):
        super().__init__()
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")

        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")

        self.frame_read = False

    def get_frame(self):
        if not self.frame_read:
            self.frame_read = True
            return True, self.image
        return False, None


class VideoSource(InputSource):
    """Input source for video files"""

    def __init__(self, video_path: str):
        super().__init__()
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class WebcamSource(InputSource):
    """Input source for webcam"""

    def __init__(self, device_id: int = 0):
        super().__init__()
        self.cap = cv2.VideoCapture(device_id)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open webcam with device ID: {device_id}")

        # Set webcam properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.fps = 30

    def get_frame(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class RealSenseSource(InputSource):
    """Input source for Intel RealSense cameras"""

    def __init__(self):
        super().__init__()
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not available")

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color stream
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming
        profile = self.pipeline.start(self.config)

        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        self.fps = 30

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()

            if not color_frame:
                return False, None

            # Convert to numpy array
            color_image = np.asanyarray(color_frame.get_data())
            return True, color_image
        except RuntimeError:
            return False, None

    def release(self):
        self.pipeline.stop()

    def get_camera_params(self):
        """Return RealSense camera parameters"""
        camera_matrix = np.array(
            [
                [self.intrinsics.fx, 0, self.intrinsics.ppx],
                [0, self.intrinsics.fy, self.intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # RealSense distortion coefficients
        dist_coeffs = np.array(
            [
                self.intrinsics.coeffs[0],  # k1
                self.intrinsics.coeffs[1],  # k2
                self.intrinsics.coeffs[2],  # p1
                self.intrinsics.coeffs[3],  # p2
                self.intrinsics.coeffs[4],  # k3
            ],
            dtype=np.float32,
        )

        return camera_matrix, dist_coeffs


def create_input_source(input_type: str, input_path: str = None) -> InputSource:
    """Factory function to create input sources"""

    if input_type == "image":
        if not input_path:
            raise ValueError("Image path required for image input")
        return ImageSource(input_path)

    elif input_type == "video":
        if not input_path:
            raise ValueError("Video path required for video input")
        return VideoSource(input_path)

    elif input_type == "webcam":
        device_id = int(input_path) if input_path else 0
        return WebcamSource(device_id)

    elif input_type == "realsense":
        return RealSenseSource()

    else:
        raise ValueError(f"Unknown input type: {input_type}")


def main():
    """Main function with argument parsing"""

    parser = argparse.ArgumentParser(description="Multi-Object 6DOF Pose Estimation")
    parser.add_argument(
        "--input",
        required=True,
        choices=["image", "video", "webcam", "realsense"],
        help="Input source type",
    )
    parser.add_argument("--path", type=str, help="Path to image/video file or webcam device ID")
    parser.add_argument(
        "--output", type=str, help="Output video file path (for video/webcam/realsense input)"
    )
    parser.add_argument(
        "--segmentation",
        default="threshold",
        choices=["threshold", "otsu", "canny", "color_range"],
        help="Segmentation method",
    )
    parser.add_argument(
        "--min-area", type=float, default=500, help="Minimum contour area (default: 500)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Display FPS for video/webcam (default: 30)"
    )
    parser.add_argument("--no-display", action="store_true", help="Disable display window")
    parser.add_argument("--show-grid", action="store_true", help="Show 2x2 visualization grid")
    parser.add_argument(
        "--objects",
        type=str,
        default="default:0.1,0.05,0.02",
        help='Object specifications in format "name:width,height,depth;name2:w,h,d"',
    )

    args = parser.parse_args()

    try:
        # Create input source
        input_source = create_input_source(args.input, args.path)

        # Get camera parameters
        camera_matrix, dist_coeffs = input_source.get_camera_params()

        # Create pose estimator
        estimator = MultiObjectPoseEstimator(camera_matrix, dist_coeffs)

        # Parse and add objects
        object_specs = args.objects.split(";")
        colors = [
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]

        for i, spec in enumerate(object_specs):
            parts = spec.strip().split(":")
            if len(parts) != 2:
                continue

            name = parts[0]
            try:
                dimensions = tuple(map(float, parts[1].split(",")))
                if len(dimensions) != 3:
                    continue

                color = colors[i % len(colors)]
                obj_info = ObjectInfo(name, dimensions, color)
                estimator.add_object(obj_info)
                print(f"Added object: {name} with dimensions {dimensions}")
            except ValueError:
                print(f"Invalid object specification: {spec}")
                continue

        # Setup video writer if output specified
        video_writer = None
        if args.output and args.input in ["video", "webcam", "realsense"]:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            if args.show_grid:
                # For grid view, we need to determine output size after first frame
                video_writer = None  # Will be initialized after first frame
            else:
                video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (640, 480))

        # Main processing loop
        frame_count = 0
        start_time = time.time()

        while True:
            ret, frame = input_source.get_frame()

            if not ret:
                break

            frame_count += 1

            # Estimate poses for multiple objects
            results = estimator.estimate_multiple_poses(frame, args.segmentation, args.min_area)

            # Visualize results
            vis_frame, combined_mask, combined_contour_mask = estimator.visualize_results(
                frame, results
            )

            # Create display frame
            if args.show_grid:
                display_frame = estimator.create_visualization_grid(
                    frame, vis_frame, combined_mask, combined_contour_mask
                )

                # Initialize video writer with grid dimensions
                if video_writer is None and args.output:
                    h, w = display_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    video_writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))
            else:
                display_frame = vis_frame

            # Add FPS counter
            if frame_count % 30 == 0:  # Update every 30 frames
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                cv2.putText(
                    display_frame,
                    f"FPS: {fps:.1f}",
                    (10, display_frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # Display frame
            if not args.no_display:
                window_name = "6DOF Multi-Object Pose Estimation"
                if args.show_grid:
                    window_name += " - Grid View"

                cv2.imshow(window_name, display_frame)

                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # 'q' or ESC
                    break
                elif key == ord("s"):  # Save current frame
                    cv2.imwrite(f"pose_estimation_frame_{frame_count}.jpg", display_frame)
                    print(f"Saved frame {frame_count}")
                elif key == ord(" "):  # Space to pause
                    cv2.waitKey(0)
                elif key == ord("g"):  # Toggle grid view
                    args.show_grid = not args.show_grid
                    print(f"Grid view: {'ON' if args.show_grid else 'OFF'}")

            # Write to output video
            if video_writer:
                video_writer.write(display_frame)

            # Print pose info for detected objects
            if results["success"] and results["total_objects_found"] > 0:
                print(f"Frame {frame_count}: Found {results['total_objects_found']} objects")

                for mask_name, objects_list in results["objects"].items():
                    for i, obj_result in enumerate(objects_list):
                        if obj_result["success"]:
                            pos = obj_result["position"]
                            angles = obj_result["euler_angles"]
                            obj_name = obj_result["object_info"].name
                            print(
                                f"  {obj_name}_{i+1}: "
                                f"Pos=({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f}) "
                                f"Rot=({angles['roll']:.1f}°, {angles['pitch']:.1f}°, {angles['yaw']:.1f}°)"
                            )
            else:
                if not results["success"]:
                    print(
                        f"Frame {frame_count}: Pose estimation failed - {results.get('error', 'Unknown error')}"
                    )
                else:
                    print(f"Frame {frame_count}: No objects detected")

            # For single image, break after processing
            if args.input == "image":
                if not args.no_display:
                    print("Press any key to exit...")
                    cv2.waitKey(0)
                break

        # Cleanup
        input_source.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()

        print(f"\nProcessed {frame_count} frames")
        if args.output:
            print(f"Output saved to: {args.output}")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
