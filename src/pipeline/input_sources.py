"""
Input source classes for 6DoF pose estimation pipeline.

This module provides classes for different input sources: image files,
video files, webcam, and RealSense cameras.
"""

from typing import Optional, Tuple, Union

import cv2
import numpy as np

# Try to import pyrealsense2 for RealSense support
try:
    import pyrealsense2 as rs

    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False


class InputSource:
    """Base class for different input sources"""

    def __init__(self):
        self.fps = 30  # Default FPS

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Get the next frame from input source

        Returns:
            Tuple of (success flag, frame)
        """
        raise NotImplementedError

    def release(self) -> None:
        """Release resources"""
        pass

    def get_camera_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return camera matrix and distortion coefficients

        Returns:
            Tuple of (camera_matrix, dist_coeffs)
        """
        # Default camera parameters (can be overridden by child classes)
        fx, fy = 800, 800  # Focal lengths
        cx, cy = 320, 240  # Principal point

        # Generic camera matrix for a 640x480 camera
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

        # No distortion
        dist_coeffs = np.zeros((5, 1), dtype=np.float32)

        return camera_matrix, dist_coeffs


class ImageSource(InputSource):
    """Input source for single images"""

    def __init__(self, image_path: str):
        """
        Initialize an image source

        Args:
            image_path: Path to input image
        """
        super().__init__()
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not read image: {image_path}")
        self.returned = False

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Get the image (only once)

        Returns:
            Tuple of (success flag, image)
        """
        # Return the image only once
        if not self.returned:
            self.returned = True
            return True, self.image
        return False, None


class VideoSource(InputSource):
    """Input source for video files"""

    def __init__(self, video_path: str):
        """
        Initialize a video source

        Args:
            video_path: Path to input video
        """
        super().__init__()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Get the next video frame

        Returns:
            Tuple of (success flag, frame)
        """
        return self.cap.read()

    def release(self) -> None:
        """Release video capture resources"""
        self.cap.release()


class WebcamSource(InputSource):
    """Input source for webcam"""

    def __init__(self, cam_id: Union[int, str] = 0):
        """
        Initialize webcam source

        Args:
            cam_id: Camera index or device path
        """
        super().__init__()
        self.cap = cv2.VideoCapture(int(cam_id))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open webcam: {cam_id}")

    def get_frame(self) -> Tuple[bool, np.ndarray]:
        """
        Get the next webcam frame

        Returns:
            Tuple of (success flag, frame)
        """
        return self.cap.read()

    def release(self) -> None:
        """Release webcam resources"""
        self.cap.release()


class RealSenseSource(InputSource):
    """Input source for Intel RealSense cameras"""

    def __init__(self):
        """Initialize RealSense camera source"""
        super().__init__()
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not available")

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Enable color and depth streams
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming
        self.profile = self.pipeline.start(self.config)

        # Get depth scale for depth calculations
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Get camera intrinsics
        self.color_profile = self.profile.get_stream(rs.stream.color)
        self.depth_profile = self.profile.get_stream(rs.stream.depth)
        self.color_intrinsics = self.color_profile.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = self.depth_profile.as_video_stream_profile().get_intrinsics()

        # Align depth to color frame
        self.align = rs.align(rs.stream.color)

        self.fps = 30

    def get_frame(self) -> Tuple[bool, np.ndarray, np.ndarray]:
        """
        Get the next color and depth frames

        Returns:
            Tuple of (success flag, color frame, depth frame)
        """
        try:
            # Wait for a coherent pair of frames: depth and color
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)

            # Align the depth frame to color frame
            aligned_frames = self.align.process(frames)

            # Get aligned frames
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not depth_frame or not color_frame:
                return False, None, None

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            return True, color_image, depth_image
        except RuntimeError:
            return False, None, None

    def release(self) -> None:
        """Release RealSense pipeline resources"""
        self.pipeline.stop()

    def get_camera_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return RealSense camera parameters

        Returns:
            Tuple of (camera_matrix, dist_coeffs)
        """
        camera_matrix = np.array(
            [
                [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

        # RealSense distortion coefficients
        dist_coeffs = np.array(
            [
                self.color_intrinsics.coeffs[0],  # k1
                self.color_intrinsics.coeffs[1],  # k2
                self.color_intrinsics.coeffs[2],  # p1
                self.color_intrinsics.coeffs[3],  # p2
                self.color_intrinsics.coeffs[4],  # k3
            ],
            dtype=np.float32,
        )

        return camera_matrix, dist_coeffs

    def get_camera_intrinsics(self) -> np.ndarray:
        """
        Return the camera intrinsics matrix in the format expected by the transform module.

        Returns:
            3x3 camera intrinsics matrix
        """
        return np.array(
            [
                [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float32,
        )

    def get_point_3d(self, x: int, y: int, depth_image: np.ndarray) -> Tuple[float, float, float]:
        """
        Get 3D coordinates of a point in the camera coordinate system

        Args:
            x: X coordinate in the image
            y: Y coordinate in the image
            depth_image: Depth image

        Returns:
            Tuple of (X, Y, Z) coordinates in the camera coordinate system (meters)
        """
        # Get depth value at the given pixel
        depth = depth_image[y, x].astype(float) * self.depth_scale  # Convert to meters

        # Deproject from pixel to 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(self.color_intrinsics, [x, y], depth)

        return tuple(point_3d)


def create_input_source(input_type: str, input_path: Optional[str] = None) -> InputSource:
    """
    Factory function to create input sources

    Args:
        input_type: Type of input source ("image", "video", "webcam", "realsense")
        input_path: Optional path to input source

    Returns:
        InputSource instance

    Raises:
        ValueError: If input type is unknown
        ImportError: If RealSense is not available
    """
    if input_type == "image":
        if not input_path:
            raise ValueError("Image path required for image input")
        return ImageSource(input_path)

    elif input_type == "video":
        if not input_path:
            raise ValueError("Video path required for video input")
        return VideoSource(input_path)

    elif input_type == "webcam":
        cam_id = 0 if not input_path else input_path
        return WebcamSource(cam_id)

    elif input_type == "realsense":
        if not REALSENSE_AVAILABLE:
            raise ImportError("pyrealsense2 not available")
        return RealSenseSource()

    else:
        raise ValueError(f"Unknown input type: {input_type}")
