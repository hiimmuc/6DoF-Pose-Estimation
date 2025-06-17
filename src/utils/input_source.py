"""Input source handling for various data streams."""

from typing import Optional, Tuple

import cv2
import numpy as np


class InputSource:
    """
    Handle different input sources (image, video, webcam, RealSense).

    Abstracts away the complexity of dealing with different input sources,
    providing a unified interface for frame acquisition.
    """

    def __init__(self, input_type: str, source_path: str) -> None:
        """
        Initialize the input source.

        Args:
            input_type: Type of input ('image', 'video', 'webcam', or 'realsense').
            source_path: Path to source or index of webcam.
        """
        self.input_type = input_type.lower()
        self.source_path = source_path
        self.cap = None
        self.realsense_pipeline = None
        self.current_frame = None

        # Validate input type
        valid_types = ["image", "video", "webcam", "realsense"]
        if self.input_type not in valid_types:
            raise ValueError(f"Input type must be one of {valid_types}")

        # Initialize the source
        self._initialize_source()

    def _initialize_source(self) -> None:
        """Initialize the appropriate input source."""
        try:
            if self.input_type == "image":
                self._init_image()
            elif self.input_type in ["video", "webcam"]:
                self._init_video_webcam()
            elif self.input_type == "realsense":
                self._init_realsense()

        except Exception as e:
            raise RuntimeError(f"Error initializing input source: {str(e)}")

    def _init_image(self) -> None:
        """Initialize image input source."""
        self.current_frame = cv2.imread(self.source_path)
        if self.current_frame is None:
            raise ValueError(f"Could not read image from {self.source_path}")

    def _init_video_webcam(self) -> None:
        """Initialize video or webcam input source."""
        # Convert webcam index from string to int if needed
        if self.input_type == "webcam" and self.source_path.isdigit():
            self.source_path = int(self.source_path)

        self.cap = cv2.VideoCapture(self.source_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {self.source_path}")

    def _init_realsense(self) -> None:
        """Initialize RealSense camera."""
        try:
            import pyrealsense2 as rs

            # Initialize RealSense pipeline
            self.realsense_pipeline = rs.pipeline()
            config = rs.config()

            # Enable color and depth streams
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

            # Start the pipeline
            self.realsense_pipeline.start(config)

        except ImportError:
            raise ImportError(
                "pyrealsense2 library not found. Install it to use RealSense camera."
            )

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get the next frame from the input source.

        Returns:
            Tuple containing:
                - success (bool): Whether frame was successfully acquired
                - rgb_frame (Optional[np.ndarray]): RGB frame
                - depth_frame (Optional[np.ndarray]): Depth frame (None if not available)
        """
        try:
            if self.input_type == "image":
                return self._get_image_frame()
            elif self.input_type in ["video", "webcam"]:
                return self._get_video_frame()
            elif self.input_type == "realsense":
                return self._get_realsense_frame()
        except Exception as e:
            print(f"Error acquiring frame: {str(e)}")
            return False, None, None

    def _get_image_frame(self) -> Tuple[bool, Optional[np.ndarray], None]:
        """Get frame from image source."""
        if self.current_frame is not None:
            return True, self.current_frame.copy(), None
        return False, None, None

    def _get_video_frame(self) -> Tuple[bool, Optional[np.ndarray], None]:
        """Get frame from video or webcam source."""
        success, frame = self.cap.read()
        return success, frame, None

    def _get_realsense_frame(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Get RGB and depth frames from RealSense camera."""
        frames = self.realsense_pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return False, None, None

        # Convert frames to numpy arrays
        rgb_frame = np.asanyarray(color_frame.get_data())
        depth_frame = np.asanyarray(depth_frame.get_data())

        return True, rgb_frame, depth_frame

    def release(self) -> None:
        """Release resources associated with the input source."""
        try:
            if self.cap is not None:
                self.cap.release()

            if self.realsense_pipeline is not None:
                self.realsense_pipeline.stop()
        except Exception as e:
            print(f"Error releasing resources: {str(e)}")
