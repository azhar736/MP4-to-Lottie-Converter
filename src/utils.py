"""
Utility functions for file handling, validation, and common operations.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import logging
from pathvalidate import validate_filename, sanitize_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FileUtils:
    """Utility class for file operations."""

    @staticmethod
    def validate_video_file(file_path: str) -> bool:
        """
        Validate if the file is a valid MP4 video.

        Args:
            file_path (str): Path to the video file

        Returns:
            bool: True if valid MP4, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return False

        if not file_path.lower().endswith(".mp4"):
            logger.error(f"File is not an MP4: {file_path}")
            return False

        # Check file size (should be > 0)
        if os.path.getsize(file_path) == 0:
            logger.error(f"File is empty: {file_path}")
            return False

        return True

    @staticmethod
    def create_output_directory(base_path: str, video_name: str) -> str:
        """
        Create organized output directory structure.

        Args:
            base_path (str): Base output directory
            video_name (str): Name of the video file (without extension)

        Returns:
            str: Path to created output directory
        """
        # Sanitize video name for directory
        safe_name = sanitize_filename(video_name)
        output_dir = os.path.join(base_path, f"{safe_name}_lottie_conversion")

        # Create directory structure
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "output"), exist_ok=True)

        logger.info(f"Created output directory: {output_dir}")
        return output_dir

    @staticmethod
    def clean_temp_files(temp_dir: str) -> None:
        """
        Clean up temporary files and directories.

        Args:
            temp_dir (str): Path to temporary directory to clean
        """
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean temp directory {temp_dir}: {e}")

    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """
        Get file size in megabytes.

        Args:
            file_path (str): Path to file

        Returns:
            float: File size in MB
        """
        if not os.path.exists(file_path):
            return 0.0

        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)

    @staticmethod
    def generate_unique_filename(directory: str, base_name: str, extension: str) -> str:
        """
        Generate a unique filename in the given directory.

        Args:
            directory (str): Target directory
            base_name (str): Base filename without extension
            extension (str): File extension (with or without dot)

        Returns:
            str: Unique filename
        """
        if not extension.startswith("."):
            extension = "." + extension

        counter = 1
        original_name = f"{base_name}{extension}"
        filename = original_name

        while os.path.exists(os.path.join(directory, filename)):
            filename = f"{base_name}_{counter}{extension}"
            counter += 1

        return filename


class ValidationUtils:
    """Utility class for validation operations."""

    @staticmethod
    def validate_dimensions(width: int, height: int) -> bool:
        """
        Validate animation dimensions.

        Args:
            width (int): Width in pixels
            height (int): Height in pixels

        Returns:
            bool: True if valid dimensions
        """
        if width <= 0 or height <= 0:
            return False

        if width > 4096 or height > 4096:
            logger.warning("Very large dimensions may cause performance issues")

        return True

    @staticmethod
    def validate_fps(fps: float) -> bool:
        """
        Validate frame rate.

        Args:
            fps (float): Frames per second

        Returns:
            bool: True if valid FPS
        """
        if fps <= 0 or fps > 120:
            return False

        return True

    @staticmethod
    def estimate_output_size(
        frame_count: int, width: int, height: int, quality: int = 85
    ) -> float:
        """
        Estimate output Lottie file size in MB.

        Args:
            frame_count (int): Number of frames
            width (int): Frame width
            height (int): Frame height
            quality (int): JPEG quality (0-100)

        Returns:
            float: Estimated size in MB
        """
        # Rough estimation based on compressed image size
        pixels_per_frame = width * height

        # Quality factor (lower quality = smaller size)
        quality_factor = quality / 100.0

        # Base size per pixel (bytes) - rough estimate
        bytes_per_pixel = 0.5 * quality_factor

        # Calculate total size
        total_bytes = frame_count * pixels_per_frame * bytes_per_pixel

        # Add JSON overhead (approximately 20% of image data)
        total_bytes *= 1.2

        return total_bytes / (1024 * 1024)


class ProgressTracker:
    """Simple progress tracking utility."""

    def __init__(self, total_steps: int, description: str = "Processing"):
        """
        Initialize progress tracker.

        Args:
            total_steps (int): Total number of steps
            description (str): Description of the process
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description
        self.callbacks = []

    def add_callback(self, callback):
        """Add a callback function to be called on progress updates."""
        self.callbacks.append(callback)

    def update(self, step: int = None, message: str = None):
        """
        Update progress.

        Args:
            step (int, optional): Current step number
            message (str, optional): Custom message
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        percentage = (self.current_step / self.total_steps) * 100

        if message:
            log_message = f"{self.description}: {message} ({percentage:.1f}%)"
        else:
            log_message = f"{self.description}: {self.current_step}/{self.total_steps} ({percentage:.1f}%)"

        logger.info(log_message)

        # Call registered callbacks
        for callback in self.callbacks:
            try:
                callback(self.current_step, self.total_steps, percentage, message or "")
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")

    def finish(self, message: str = "Complete"):
        """Mark progress as finished."""
        self.current_step = self.total_steps
        logger.info(f"{self.description}: {message} (100%)")

        for callback in self.callbacks:
            try:
                callback(self.total_steps, self.total_steps, 100.0, message)
            except Exception as e:
                logger.warning(f"Progress callback error: {e}")


class ConfigManager:
    """Configuration management utility."""

    DEFAULT_CONFIG = {
        "max_frames": 150,  # Optimized balance for quality vs size
        "output_quality": 80,  # Higher quality default
        "max_file_size_mb": 5.0,  # Target 5MB for web optimization
        "temp_cleanup": True,
        "preserve_aspect_ratio": True,
        "default_fps": 30.0,
        "quality_priority": True,  # Prioritize quality over size
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path (Optional[str]): Path to config file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()

        if config_path and os.path.exists(config_path):
            self.load_config()

    def load_config(self):
        """Load configuration from file."""
        try:
            import json

            with open(self.config_path, "r") as f:
                user_config = json.load(f)
                self.config.update(user_config)
            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")

    def save_config(self):
        """Save current configuration to file."""
        if not self.config_path:
            return

        try:
            import json

            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value):
        """Set configuration value."""
        self.config[key] = value

    def update(self, updates: dict):
        """Update multiple configuration values."""
        self.config.update(updates)
