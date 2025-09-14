"""
Lottie JSON format converter for creating animations from video frames.
"""

import json
import os
import base64
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LottieConverter:
    """Converts video frames to Lottie JSON animation format."""

    def __init__(self, width: int, height: int, fps: float):
        """
        Initialize LottieConverter with animation properties.

        Args:
            width (int): Animation width in pixels
            height (int): Animation height in pixels
            fps (float): Frames per second
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_rate = fps
        self.assets = []
        self.layers = []

    def create_base_lottie_structure(self, duration_frames: int) -> Dict[str, Any]:
        """
        Create the base Lottie JSON structure.

        Args:
            duration_frames (int): Total number of frames in animation

        Returns:
            Dict[str, Any]: Base Lottie structure
        """
        return {
            "v": "5.7.4",  # Lottie version
            "fr": self.frame_rate,  # Frame rate
            "ip": 0,  # In point (start frame)
            "op": duration_frames,  # Out point (end frame)
            "w": self.width,  # Width
            "h": self.height,  # Height
            "nm": "MP4 to Lottie Animation",  # Name
            "ddd": 0,  # 3D flag
            "assets": self.assets,
            "layers": self.layers,
            "markers": [],
        }

    def create_image_asset(self, image_path: str, asset_id: str) -> Dict[str, Any]:
        """
        Create an image asset from a frame file.

        Args:
            image_path (str): Path to the image file
            asset_id (str): Unique asset identifier

        Returns:
            Dict[str, Any]: Image asset definition
        """
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize if needed to match animation dimensions
                if img.size != (self.width, self.height):
                    img = img.resize(
                        (self.width, self.height), Image.Resampling.LANCZOS
                    )

                # Convert to base64
                import io

                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                img_data = buffer.getvalue()
                img_base64 = base64.b64encode(img_data).decode("utf-8")

                return {
                    "id": asset_id,
                    "w": self.width,
                    "h": self.height,
                    "u": "",  # Path (empty for embedded)
                    "p": f"data:image/png;base64,{img_base64}",  # Base64 data
                    "e": 1,  # Embedded flag
                }

        except Exception as e:
            logger.error(f"Error creating image asset from {image_path}: {e}")
            return None

    def create_image_layer(
        self, asset_id: str, start_frame: int, duration: int, layer_index: int
    ) -> Dict[str, Any]:
        """
        Create an image layer for the Lottie animation.

        Args:
            asset_id (str): Asset ID to reference
            start_frame (int): Frame when layer starts
            duration (int): Duration in frames
            layer_index (int): Layer index (higher = front)

        Returns:
            Dict[str, Any]: Image layer definition
        """
        return {
            "ddd": 0,
            "ind": layer_index,
            "ty": 2,  # Image layer type
            "nm": f"Frame {layer_index}",
            "refId": asset_id,
            "sr": 1,  # Stretch
            "ks": {  # Transform properties
                "o": {"a": 0, "k": 100, "ix": 11},  # Opacity
                "r": {"a": 0, "k": 0, "ix": 10},  # Rotation
                "p": {
                    "a": 0,
                    "k": [self.width / 2, self.height / 2, 0],
                    "ix": 2,
                },  # Position
                "a": {
                    "a": 0,
                    "k": [self.width / 2, self.height / 2, 0],
                    "ix": 1,
                },  # Anchor
                "s": {"a": 0, "k": [100, 100, 100], "ix": 6},  # Scale
            },
            "ao": 0,
            "ip": start_frame,  # In point
            "op": start_frame + duration,  # Out point
            "st": start_frame,  # Start time
            "bm": 0,  # Blend mode
        }

    def create_sequence_animation(
        self,
        frame_paths: List[str],
        frames_per_image: int = 1,
        original_duration: float = None,
    ) -> Dict[str, Any]:
        """
        Create a Lottie animation from a sequence of frame images.

        Args:
            frame_paths (List[str]): List of paths to frame images
            frames_per_image (int): How many frames each image should display
            original_duration (float): Original video duration in seconds

        Returns:
            Dict[str, Any]: Complete Lottie animation JSON
        """
        if not frame_paths:
            raise ValueError("No frame paths provided")

        # Calculate proper timing to match original duration
        if original_duration:
            # Calculate frames per image to match original duration
            total_lottie_frames = int(original_duration * self.fps)
            frames_per_image = max(1, total_lottie_frames // len(frame_paths))
            logger.info(
                f"Adjusting timing: {original_duration}s video → {total_lottie_frames} Lottie frames, {frames_per_image} frames per image"
            )
        else:
            total_lottie_frames = len(frame_paths) * frames_per_image

        lottie_data = self.create_base_lottie_structure(total_lottie_frames)

        # Create assets and layers for each frame
        for i, frame_path in enumerate(frame_paths):
            asset_id = f"image_{i}"

            # Create image asset
            asset = self.create_image_asset(frame_path, asset_id)
            if asset:
                self.assets.append(asset)

                # Create layer for this frame
                start_frame = i * frames_per_image
                layer = self.create_image_layer(
                    asset_id, start_frame, frames_per_image, i
                )
                self.layers.append(layer)

        # Update the base structure with our assets and layers
        lottie_data["assets"] = self.assets
        lottie_data["layers"] = self.layers

        duration_seconds = total_lottie_frames / self.fps
        logger.info(
            f"Created Lottie animation: {len(self.assets)} assets, {len(self.layers)} layers, {duration_seconds:.1f}s duration"
        )
        return lottie_data

    def create_optimized_animation(
        self,
        frame_paths: List[str],
        max_file_size_mb: float = 10.0,
        original_duration: float = None,
    ) -> Dict[str, Any]:
        """
        Create an optimized Lottie animation with aggressive compression for small file sizes.

        Args:
            frame_paths (List[str]): List of paths to frame images
            max_file_size_mb (float): Maximum file size in MB
            original_duration (float): Original video duration in seconds

        Returns:
            Dict[str, Any]: Optimized Lottie animation JSON
        """
        logger.info(
            f"Starting aggressive optimization for max size: {max_file_size_mb}MB"
        )

        # More aggressive optimization strategy for small file sizes
        if max_file_size_mb <= 5.0:
            return self._create_ultra_compressed_animation(
                frame_paths, max_file_size_mb, original_duration
            )

        # Original optimization for larger sizes
        quality_levels = [85, 75, 65, 55, 45, 35, 25]
        frame_skip_levels = [1, 2, 3, 4, 6, 8]
        resolution_scales = [1.0, 0.8, 0.6, 0.5]  # Scale down resolution if needed

        for scale in resolution_scales:
            scaled_width = int(self.width * scale)
            scaled_height = int(self.height * scale)

            for skip in frame_skip_levels:
                for quality in quality_levels:
                    try:
                        # Sample frames based on skip level
                        sampled_frames = frame_paths[::skip]

                        # Create animation with current settings
                        temp_converter = LottieConverter(
                            scaled_width, scaled_height, self.fps
                        )
                        animation = temp_converter._create_compressed_animation(
                            sampled_frames, quality, original_duration, scale
                        )

                        # Estimate file size
                        json_str = json.dumps(animation, separators=(",", ":"))
                        size_mb = len(json_str.encode("utf-8")) / (1024 * 1024)

                        if size_mb <= max_file_size_mb:
                            logger.info(
                                f"Optimized animation: {size_mb:.2f}MB (quality: {quality}%, skip: {skip}, scale: {scale:.1f})"
                            )
                            return animation

                    except Exception as e:
                        logger.warning(
                            f"Failed optimization (skip={skip}, quality={quality}, scale={scale}): {e}"
                        )
                        continue

        # Fallback: create minimal animation with preserved duration
        logger.warning("Using fallback minimal animation due to size constraints")
        minimal_frames = frame_paths[:: max(1, len(frame_paths) // 8)]
        return self.create_sequence_animation(
            minimal_frames, original_duration=original_duration
        )

    def _create_ultra_compressed_animation(
        self,
        frame_paths: List[str],
        max_file_size_mb: float,
        original_duration: float = None,
    ) -> Dict[str, Any]:
        """
        Create ultra-compressed animation for very small file sizes (≤5MB).
        Uses advanced compression techniques while preserving maximum quality.
        """
        logger.info("Using quality-preserving ultra-compression mode")

        # Quality-first approach - start with higher quality and optimize other factors
        quality_levels = [
            85,
            80,
            75,
            70,
            65,
            60,
            55,
            50,
            45,
            40,
        ]  # Higher quality range
        frame_skip_levels = [2, 3, 4, 5, 6, 8, 10]  # More gradual frame reduction
        resolution_scales = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]  # Start with full resolution

        # Intelligent frame selection for quality preservation
        if original_duration and original_duration <= 20:
            # For short videos, use more frames but optimize other aspects
            target_frames = min(len(frame_paths), max(15, int(original_duration * 3)))
        else:
            target_frames = min(len(frame_paths), 25)  # More frames for better quality

        # Quality-preserving compression strategies
        strategies = [
            (
                "premium_quality",
                1.0,
                [85, 80, 75],
                [2, 3],
            ),  # Full resolution, high quality
            (
                "high_quality",
                0.9,
                [80, 75, 70],
                [2, 3, 4],
            ),  # Slight resolution reduction
            ("balanced_quality", 0.8, [75, 70, 65], [3, 4, 5]),  # Balanced approach
            ("optimized", 0.7, [70, 65, 60], [4, 5, 6]),  # More compression
            ("compact", 0.6, [65, 60, 55], [5, 6, 8]),  # Higher compression
            ("minimal", 0.5, [60, 55, 50], [6, 8, 10]),  # Last resort
        ]

        for strategy_name, scale, qualities, skips in strategies:
            scaled_width = int(self.width * scale)
            scaled_height = int(self.height * scale)

            for skip in skips:
                for quality in qualities:
                    try:
                        # Smart frame sampling
                        if len(frame_paths) > target_frames:
                            # Use intelligent frame selection
                            selected_frames = self._select_key_frames(
                                frame_paths, target_frames
                            )
                        else:
                            selected_frames = frame_paths[::skip]

                        # Create ultra-compressed animation
                        temp_converter = LottieConverter(
                            scaled_width, scaled_height, self.fps
                        )
                        animation = temp_converter._create_ultra_compressed_assets(
                            selected_frames, quality, original_duration, scale
                        )

                        # Estimate size with minimal JSON
                        json_str = json.dumps(animation, separators=(",", ":"))
                        size_mb = len(json_str.encode("utf-8")) / (1024 * 1024)

                        if size_mb <= max_file_size_mb:
                            logger.info(
                                f"Ultra-compressed animation: {size_mb:.2f}MB ({strategy_name}: {quality}%, {scale:.1f}x scale, {len(selected_frames)} frames)"
                            )
                            return animation

                    except Exception as e:
                        logger.debug(f"Ultra-compression attempt failed: {e}")
                        continue

        # Final fallback - minimal animation
        logger.warning("Using absolute minimal animation")
        minimal_frames = frame_paths[:: max(1, len(frame_paths) // 6)][:8]
        temp_converter = LottieConverter(
            int(self.width * 0.4), int(self.height * 0.4), self.fps
        )
        return temp_converter._create_ultra_compressed_assets(
            minimal_frames, 15, original_duration, 0.4
        )

    def _select_key_frames(
        self, frame_paths: List[str], target_count: int
    ) -> List[str]:
        """
        Intelligently select key frames for maximum visual impact and quality preservation.
        """
        if len(frame_paths) <= target_count:
            return frame_paths

        # Advanced frame selection strategy
        total_frames = len(frame_paths)

        # Always include first and last frames for temporal consistency
        selected_indices = [0, total_frames - 1]
        remaining_count = target_count - 2

        if remaining_count > 0:
            # Distribute remaining frames with emphasis on temporal consistency
            # Use a combination of even distribution and strategic placement

            # Calculate base step for even distribution
            step = (total_frames - 2) / (remaining_count + 1)

            # Add evenly distributed frames
            for i in range(1, remaining_count + 1):
                index = int(i * step)
                if index not in selected_indices and index < total_frames - 1:
                    selected_indices.append(index)

            # Fill any remaining slots with frames that maximize temporal coverage
            while len(selected_indices) < target_count:
                # Find the largest gap between selected frames
                selected_indices.sort()
                max_gap = 0
                best_insert_pos = 1

                for i in range(len(selected_indices) - 1):
                    gap = selected_indices[i + 1] - selected_indices[i]
                    if gap > max_gap:
                        max_gap = gap
                        best_insert_pos = (
                            selected_indices[i] + selected_indices[i + 1]
                        ) // 2

                if best_insert_pos not in selected_indices:
                    selected_indices.append(best_insert_pos)
                else:
                    break

        # Ensure indices are within bounds and sorted
        selected_indices = sorted([min(i, total_frames - 1) for i in selected_indices])

        # Remove duplicates while preserving order
        unique_indices = []
        for idx in selected_indices:
            if idx not in unique_indices:
                unique_indices.append(idx)

        return [frame_paths[i] for i in unique_indices[:target_count]]

    def _create_ultra_compressed_assets(
        self,
        frame_paths: List[str],
        quality: int,
        original_duration: float = None,
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Create assets with maximum compression techniques.
        """
        # Calculate timing
        if original_duration:
            total_lottie_frames = int(original_duration * self.fps)
            frames_per_image = max(1, total_lottie_frames // len(frame_paths))
        else:
            total_lottie_frames = len(frame_paths)
            frames_per_image = 1

        lottie_data = self.create_base_lottie_structure(total_lottie_frames)
        assets = []
        layers = []

        for i, frame_path in enumerate(frame_paths):
            asset_id = f"img_{i}"  # Shorter asset IDs

            try:
                with Image.open(frame_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Resize to target dimensions
                    if img.size != (self.width, self.height):
                        img = img.resize(
                            (self.width, self.height), Image.Resampling.LANCZOS
                        )

                    # Apply additional optimizations
                    img = self._optimize_image_for_compression(img, quality)

                    # Quality-optimized JPEG compression
                    import io

                    buffer = io.BytesIO()

                    # Advanced JPEG settings for quality preservation
                    jpeg_options = {
                        "format": "JPEG",
                        "quality": quality,
                        "optimize": True,
                        "progressive": True,
                    }

                    # Add quality-preserving options for higher quality settings
                    if quality >= 70:
                        jpeg_options["subsampling"] = (
                            0  # No chroma subsampling for high quality
                        )
                    elif quality >= 60:
                        jpeg_options["subsampling"] = 1  # Moderate subsampling

                    # Apply advanced JPEG compression
                    img.save(buffer, **jpeg_options)
                    img_data = buffer.getvalue()
                    img_base64 = base64.b64encode(img_data).decode("utf-8")

                    # Minimal asset structure
                    asset = {
                        "id": asset_id,
                        "w": self.width,
                        "h": self.height,
                        "u": "",
                        "p": f"data:image/jpeg;base64,{img_base64}",
                        "e": 1,
                    }
                    assets.append(asset)

                    # Create layer with proper timing
                    start_frame = i * frames_per_image
                    layer = self._create_minimal_layer(
                        asset_id, start_frame, frames_per_image, i
                    )
                    layers.append(layer)

            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                continue

        lottie_data["assets"] = assets
        lottie_data["layers"] = layers
        return lottie_data

    def _optimize_image_for_compression(
        self, img: Image.Image, quality: int
    ) -> Image.Image:
        """
        Apply quality-preserving image optimizations for better compression.
        """
        from PIL import ImageFilter, ImageEnhance

        # Only apply aggressive optimizations at very low quality
        if quality < 50:
            # Slight sharpening to counteract compression artifacts
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(1.1)  # Slight sharpening

            # Very subtle noise reduction only at very low quality
            if quality < 40:
                img = img.filter(ImageFilter.GaussianBlur(radius=0.3))  # Reduced blur

        # Enhanced color optimization that preserves quality
        if quality < 60:
            # Use adaptive color quantization that preserves important colors
            img = self._adaptive_color_quantization(img, quality)

        return img

    def _adaptive_color_quantization(
        self, img: Image.Image, quality: int
    ) -> Image.Image:
        """
        Apply adaptive color quantization that preserves visual quality.
        """
        # Calculate color count based on quality
        if quality >= 50:
            color_count = 256  # Full color range
        elif quality >= 40:
            color_count = 220  # Slight reduction
        elif quality >= 30:
            color_count = 180  # Moderate reduction
        else:
            color_count = 150  # More aggressive reduction

        # Use adaptive palette that preserves important colors
        quantized = img.quantize(colors=color_count, method=Image.Quantize.MEDIANCUT)
        return quantized.convert("RGB")

    def _create_minimal_layer(
        self, asset_id: str, start_frame: int, duration: int, layer_index: int
    ) -> Dict[str, Any]:
        """
        Create a minimal layer structure to reduce JSON size.
        """
        return {
            "ddd": 0,
            "ind": layer_index,
            "ty": 2,
            "nm": f"L{layer_index}",  # Shorter names
            "refId": asset_id,
            "sr": 1,
            "ks": {
                "o": {"a": 0, "k": 100},  # Simplified structure
                "r": {"a": 0, "k": 0},
                "p": {"a": 0, "k": [self.width / 2, self.height / 2]},
                "a": {"a": 0, "k": [self.width / 2, self.height / 2]},
                "s": {"a": 0, "k": [100, 100]},
            },
            "ao": 0,
            "ip": start_frame,
            "op": start_frame + duration,
            "st": start_frame,
            "bm": 0,
        }

    def _create_compressed_animation(
        self,
        frame_paths: List[str],
        quality: int,
        original_duration: float = None,
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Create animation with compressed images.

        Args:
            frame_paths (List[str]): Frame paths
            quality (int): JPEG quality (0-100)
            original_duration (float): Original video duration in seconds

        Returns:
            Dict[str, Any]: Animation with compressed assets
        """
        # Calculate proper timing to match original duration
        if original_duration:
            total_lottie_frames = int(original_duration * self.fps)
            frames_per_image = max(1, total_lottie_frames // len(frame_paths))
        else:
            total_lottie_frames = len(frame_paths)
            frames_per_image = 1

        lottie_data = self.create_base_lottie_structure(total_lottie_frames)

        assets = []
        layers = []

        for i, frame_path in enumerate(frame_paths):
            asset_id = f"image_{i}"

            try:
                with Image.open(frame_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Apply scaling if specified
                    target_width = (
                        int(self.width * scale) if scale != 1.0 else self.width
                    )
                    target_height = (
                        int(self.height * scale) if scale != 1.0 else self.height
                    )

                    if img.size != (target_width, target_height):
                        img = img.resize(
                            (target_width, target_height), Image.Resampling.LANCZOS
                        )

                    # Compress as JPEG
                    import io

                    buffer = io.BytesIO()
                    img.save(buffer, format="JPEG", quality=quality, optimize=True)
                    img_data = buffer.getvalue()
                    img_base64 = base64.b64encode(img_data).decode("utf-8")

                    asset = {
                        "id": asset_id,
                        "w": target_width,
                        "h": target_height,
                        "u": "",
                        "p": f"data:image/jpeg;base64,{img_base64}",
                        "e": 1,
                    }
                    assets.append(asset)

                    # Create layer with proper timing
                    start_frame = i * frames_per_image
                    layer = self.create_image_layer(
                        asset_id, start_frame, frames_per_image, i
                    )
                    layers.append(layer)

            except Exception as e:
                logger.error(f"Error processing frame {i}: {e}")
                continue

        lottie_data["assets"] = assets
        lottie_data["layers"] = layers

        return lottie_data

    def save_animation(self, animation_data: Dict[str, Any], output_path: str) -> bool:
        """
        Save Lottie animation to JSON file.

        Args:
            animation_data (Dict[str, Any]): Lottie animation data
            output_path (str): Output file path

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(animation_data, f, separators=(",", ":"))

            file_size = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Lottie animation saved to {output_path} ({file_size:.2f}MB)")
            return True

        except Exception as e:
            logger.error(f"Error saving animation: {e}")
            return False

    def validate_lottie_format(self, animation_data: Dict[str, Any]) -> bool:
        """
        Validate that the animation data follows Lottie format standards.

        Args:
            animation_data (Dict[str, Any]): Animation data to validate

        Returns:
            bool: True if valid, False otherwise
        """
        required_fields = ["v", "fr", "ip", "op", "w", "h", "assets", "layers"]

        for field in required_fields:
            if field not in animation_data:
                logger.error(f"Missing required field: {field}")
                return False

        # Validate dimensions
        if animation_data["w"] <= 0 or animation_data["h"] <= 0:
            logger.error("Invalid dimensions")
            return False

        # Validate frame rate
        if animation_data["fr"] <= 0:
            logger.error("Invalid frame rate")
            return False

        # Validate frame range
        if animation_data["ip"] >= animation_data["op"]:
            logger.error("Invalid frame range")
            return False

        logger.info("Lottie format validation passed")
        return True
