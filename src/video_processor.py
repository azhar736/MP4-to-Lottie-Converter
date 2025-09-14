"""
Video processing module for extracting frames and analyzing MP4 files.
"""

import cv2
import numpy as np
from PIL import Image
import os
import json
from typing import List, Tuple, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoProcessor:
    """Handles MP4 video processing and frame extraction."""

    def __init__(self, video_path: str):
        """
        Initialize VideoProcessor with video file path.

        Args:
            video_path (str): Path to the MP4 video file
        """
        self.video_path = video_path
        self.cap = None
        self.fps = 0
        self.frame_count = 0
        self.width = 0
        self.height = 0
        self.duration = 0

        self._initialize_video()

    def _initialize_video(self) -> None:
        """Initialize video capture and extract basic properties."""
        try:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Cannot open video file: {self.video_path}")

            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.duration = self.frame_count / self.fps if self.fps > 0 else 0

            logger.info(
                f"Video initialized: {self.width}x{self.height}, {self.fps} FPS, {self.frame_count} frames"
            )

        except Exception as e:
            logger.error(f"Error initializing video: {e}")
            raise

    def extract_frames(
        self, output_dir: str, max_frames: Optional[int] = None
    ) -> List[str]:
        """
        Extract frames from video and save as images.

        Args:
            output_dir (str): Directory to save extracted frames
            max_frames (Optional[int]): Maximum number of frames to extract

        Returns:
            List[str]: List of paths to extracted frame images
        """
        if not self.cap:
            raise ValueError("Video not initialized")

        os.makedirs(output_dir, exist_ok=True)
        frame_paths = []

        # Calculate frame skip for max_frames limit
        frame_skip = 1
        if max_frames and self.frame_count > max_frames:
            frame_skip = self.frame_count // max_frames

        frame_index = 0
        extracted_count = 0

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            if frame_index % frame_skip == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Save frame
                frame_filename = f"frame_{extracted_count:06d}.png"
                frame_path = os.path.join(output_dir, frame_filename)

                pil_image = Image.fromarray(frame_rgb)
                pil_image.save(frame_path, "PNG")

                frame_paths.append(frame_path)
                extracted_count += 1

                if max_frames and extracted_count >= max_frames:
                    break

            frame_index += 1

        logger.info(f"Extracted {len(frame_paths)} frames to {output_dir}")
        return frame_paths

    def get_video_info(self) -> Dict:
        """
        Get comprehensive video information.

        Returns:
            Dict: Video properties and metadata
        """
        return {
            "path": self.video_path,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration": self.duration,
            "aspect_ratio": self.width / self.height if self.height > 0 else 1.0,
        }

    def get_frame_at_time(self, time_seconds: float) -> Optional[np.ndarray]:
        """
        Extract a specific frame at given time.

        Args:
            time_seconds (float): Time in seconds

        Returns:
            Optional[np.ndarray]: Frame data or None if failed
        """
        if not self.cap:
            return None

        frame_number = int(time_seconds * self.fps)
        if frame_number >= self.frame_count:
            return None

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.cap.read()

        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def analyze_motion(self, sample_frames: int = 10) -> Dict:
        """
        Analyze motion patterns in the video for better Lottie conversion.

        Args:
            sample_frames (int): Number of frames to sample for analysis

        Returns:
            Dict: Motion analysis results
        """
        if not self.cap or self.frame_count < 2:
            return {"motion_detected": False, "average_motion": 0, "motion_samples": 0}

        try:
            return self._analyze_motion_optical_flow(sample_frames)
        except Exception as e:
            logger.warning(f"Optical flow motion analysis failed: {e}")
            logger.info("Falling back to frame difference analysis")
            return self._analyze_motion_frame_diff(sample_frames)

    def _analyze_motion_optical_flow(self, sample_frames: int) -> Dict:
        """Analyze motion using optical flow method."""
        motion_values = []
        prev_frame = None

        # Sample frames evenly throughout the video
        frame_step = max(1, self.frame_count // sample_frames)

        for i in range(0, self.frame_count, frame_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()

            if not ret:
                continue

            # Convert to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Use corner detection to find points to track
                corners = cv2.goodFeaturesToTrack(
                    prev_frame,
                    maxCorners=100,
                    qualityLevel=0.3,
                    minDistance=7,
                    blockSize=7,
                )

                if corners is not None and len(corners) > 0:
                    # Calculate optical flow
                    next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                        prev_frame,
                        gray,
                        corners,
                        None,
                        winSize=(15, 15),
                        maxLevel=2,
                        criteria=(
                            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                            10,
                            0.03,
                        ),
                    )

                    if next_pts is not None and status is not None:
                        # Select good points
                        good_new = next_pts[status == 1]
                        good_old = corners[status == 1]

                        if len(good_new) > 0 and len(good_old) > 0:
                            # Calculate motion magnitude
                            motion_vectors = good_new - good_old
                            motion_magnitude = np.mean(
                                np.sqrt(np.sum(motion_vectors**2, axis=1))
                            )
                            motion_values.append(motion_magnitude)

            prev_frame = gray

        avg_motion = np.mean(motion_values) if motion_values else 0

        return {
            "motion_detected": avg_motion > 1.0,
            "average_motion": float(avg_motion),
            "motion_samples": len(motion_values),
        }

    def _analyze_motion_frame_diff(self, sample_frames: int) -> Dict:
        """Analyze motion using frame difference method (fallback)."""
        motion_values = []
        prev_frame = None

        # Sample frames evenly throughout the video
        frame_step = max(1, self.frame_count // sample_frames)

        for i in range(0, self.frame_count, frame_step):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()

            if not ret:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_frame is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_frame, gray)

                # Calculate mean difference as motion indicator
                motion_magnitude = np.mean(diff)
                motion_values.append(motion_magnitude)

            prev_frame = gray

        avg_motion = np.mean(motion_values) if motion_values else 0

        return {
            "motion_detected": avg_motion > 5.0,  # Different threshold for frame diff
            "average_motion": float(avg_motion),
            "motion_samples": len(motion_values),
        }

    def __del__(self):
        """Clean up video capture resources."""
        if self.cap:
            self.cap.release()
