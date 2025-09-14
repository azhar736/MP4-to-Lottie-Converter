#!/usr/bin/env python3
"""
MP4 to Lottie Converter

Convert MP4 videos to high-quality Lottie JSON animations.
Supports both GUI and command-line interfaces.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.gui import MP4ToLottieGUI
from src.video_processor import VideoProcessor
from src.lottie_converter import LottieConverter
from src.utils import FileUtils, ValidationUtils, ProgressTracker, ConfigManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("mp4_to_lottie.log"),
    ],
)
logger = logging.getLogger(__name__)


class CommandLineConverter:
    """Command-line interface for MP4 to Lottie conversion."""

    def __init__(self, args):
        """Initialize with command line arguments."""
        self.args = args
        self.config = ConfigManager()

    def convert(self):
        """Perform conversion using command line arguments."""
        try:
            # Validate input file
            if not FileUtils.validate_video_file(self.args.input):
                logger.error(f"Invalid input file: {self.args.input}")
                return False

            # Initialize video processor
            logger.info(f"Processing video: {self.args.input}")
            processor = VideoProcessor(self.args.input)
            video_info = processor.get_video_info()

            logger.info(
                f"Video info: {video_info['width']}x{video_info['height']}, "
                f"{video_info['fps']:.2f} FPS, {video_info['duration']:.2f}s"
            )

            # Create output directory
            video_name = os.path.splitext(os.path.basename(self.args.input))[0]
            if self.args.output:
                output_base = self.args.output
            else:
                output_base = os.path.dirname(self.args.input)

            output_dir = FileUtils.create_output_directory(output_base, video_name)
            frames_dir = os.path.join(output_dir, "frames")

            # Setup progress tracking
            progress = ProgressTracker(3, "CLI Conversion")

            # Extract frames
            logger.info("Extracting frames...")
            max_frames = self.args.max_frames or self.config.get("max_frames", 60)
            frame_paths = processor.extract_frames(frames_dir, max_frames)
            progress.update(1, f"Extracted {len(frame_paths)} frames")

            if not frame_paths:
                logger.error("No frames could be extracted")
                return False

            # Convert to Lottie
            logger.info("Converting to Lottie format...")
            converter = LottieConverter(
                video_info["width"], video_info["height"], video_info["fps"]
            )

            # Use optimization if specified
            max_size = self.args.max_size or self.config.get("max_file_size_mb", 10.0)
            if max_size > 0:
                lottie_data = converter.create_optimized_animation(
                    frame_paths, max_size, video_info["duration"]
                )
            else:
                lottie_data = converter.create_sequence_animation(
                    frame_paths, original_duration=video_info["duration"]
                )

            progress.update(2, "Generated Lottie animation")

            # Save output
            output_file = os.path.join(output_dir, "output", f"{video_name}.json")
            logger.info(f"Saving to: {output_file}")

            if converter.save_animation(lottie_data, output_file):
                progress.update(3, "Conversion complete!")

                # Validate output
                if converter.validate_lottie_format(lottie_data):
                    file_size = FileUtils.get_file_size_mb(output_file)
                    logger.info(
                        f"SUCCESS: Conversion completed! Output: {output_file} ({file_size:.2f}MB)"
                    )

                    # Clean up if requested
                    if not self.args.keep_frames:
                        FileUtils.clean_temp_files(frames_dir)
                        logger.info("Cleaned up temporary frame files")

                    return True
                else:
                    logger.error("Generated file failed validation")
                    return False
            else:
                logger.error("Failed to save Lottie file")
                return False

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return False


def setup_argument_parser():
    """Setup command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert MP4 videos to Lottie JSON animations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Launch GUI
  %(prog)s video.mp4                    # Convert with defaults
  %(prog)s video.mp4 -o /path/output    # Specify output directory
  %(prog)s video.mp4 --max-frames 30    # Limit to 30 frames
  %(prog)s video.mp4 --max-size 5.0     # Limit output to 5MB
        """,
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input MP4 video file (if not provided, GUI will launch)",
    )

    parser.add_argument(
        "-o", "--output", help="Output directory (default: same as input file)"
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        help="Maximum number of frames to extract (default: 60)",
    )

    parser.add_argument(
        "--quality",
        type=int,
        choices=range(10, 101),
        metavar="10-100",
        help="Output quality percentage (default: 85)",
    )

    parser.add_argument(
        "--max-size", type=float, help="Maximum output file size in MB (default: 10.0)"
    )

    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="Keep extracted frame files (default: delete after conversion)",
    )

    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force launch GUI even if input file is provided",
    )

    parser.add_argument(
        "--version", action="version", version="MP4 to Lottie Converter v1.0"
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )

    return parser


def check_dependencies():
    """Check if all required dependencies are available."""
    missing_deps = []

    try:
        import cv2
    except ImportError:
        missing_deps.append("opencv-python")

    try:
        import PIL
    except ImportError:
        missing_deps.append("Pillow")

    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")

    try:
        import pathvalidate
    except ImportError:
        missing_deps.append("pathvalidate")

    if missing_deps:
        logger.error("Missing required dependencies:")
        for dep in missing_deps:
            logger.error(f"  - {dep}")
        logger.error("\nInstall missing dependencies with:")
        logger.error(f"  pip install {' '.join(missing_deps)}")
        return False

    return True


def main():
    """Main application entry point."""
    try:
        # Parse command line arguments
        parser = setup_argument_parser()
        args = parser.parse_args()

        # Configure logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        logger.info("MP4 to Lottie Converter v1.0")

        # Check dependencies
        if not check_dependencies():
            sys.exit(1)

        # Determine whether to use GUI or CLI
        if args.input and not args.gui:
            # Command line mode
            logger.info("Running in command-line mode")
            converter = CommandLineConverter(args)
            success = converter.convert()
            sys.exit(0 if success else 1)
        else:
            # GUI mode
            logger.info("Launching GUI application")
            try:
                app = MP4ToLottieGUI()
                app.run()
            except ImportError as e:
                if "tkinter" in str(e).lower():
                    logger.error("Tkinter is not available. GUI mode requires tkinter.")
                    logger.error(
                        "On some systems, you may need to install python3-tk package."
                    )
                    sys.exit(1)
                else:
                    raise

    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
