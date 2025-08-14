"""Image preprocessing utilities for document processing.

This module provides utility functions for image preprocessing operations
that can improve the quality and compatibility of images before processing
with multimodal AI models or other analysis tools.
"""

import base64
from pathlib import Path
from typing import Any, Optional, Tuple, Union

from .base import ProcessingError


class ImagePreprocessor:
    """Utility class for image preprocessing operations.

    This class provides static methods for common image preprocessing tasks
    that can improve image analysis results. Operations include resizing,
    format conversion, enhancement, and optimization for API consumption.

    All methods use PIL/Pillow for image manipulation and handle errors
    gracefully with informative error messages.
    """

    @staticmethod
    def resize_image(
        image_path: Union[str, Path],
        max_size: Tuple[int, int] = (1024, 1024),
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Resize image to fit within specified dimensions while maintaining aspect ratio.

        This method resizes images proportionally to fit within the specified
        maximum dimensions. It's useful for reducing file sizes and ensuring
        compatibility with API limitations.

        Args:
            image_path: Path to the input image file.
            max_size: Maximum (width, height) dimensions as tuple (default: 1024x1024).
            output_path: Path for the resized image. If None, creates a new file
                        with '_resized' suffix in the same directory.

        Returns:
            Path to the resized image file as string.

        Raises:
            ProcessingError: If PIL/Pillow is not available or resizing fails.
        """
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "PIL/Pillow is required for image preprocessing. Install with: pip install Pillow"
            ) from exc

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_resized{image_path.suffix}"

        try:
            with Image.open(image_path) as img:
                # Calculate new size maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)

                # Save with optimization
                img.save(output_path, optimize=True, quality=85)
                return str(output_path)
        except Exception as e:
            raise ProcessingError(f"Failed to resize image {image_path}: {str(e)}") from e

    @staticmethod
    def convert_to_rgb(
        image_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """Convert image to RGB format for better compatibility.

        This method converts images to RGB color mode and JPEG format,
        which is widely supported by AI models and APIs. It handles
        transparency by adding a white background.

        Args:
            image_path: Path to the input image file.
            output_path: Path for the converted image. If None, creates a new
                        file with '_rgb.jpg' suffix.

        Returns:
            Path to the converted image file as string.

        Raises:
            ProcessingError: If PIL/Pillow is not available or conversion fails.
        """
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "PIL/Pillow is required for image preprocessing. Install with: pip install Pillow"
            ) from exc

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_rgb.jpg"

        try:
            with Image.open(image_path) as img:
                # Convert to RGB, handling transparency
                if img.mode != "RGB":
                    if img.mode in ("RGBA", "LA"):
                        # Create white background for transparent images
                        background = Image.new("RGB", img.size, (255, 255, 255))
                        if img.mode == "RGBA":
                            background.paste(img, mask=img.split()[-1])  # Use alpha channel as mask
                        else:
                            background.paste(img)
                        img = background
                    else:
                        img = img.convert("RGB")

                img.save(output_path, "JPEG", optimize=True, quality=85)
                return str(output_path)
        except Exception as e:
            raise ProcessingError(f"Failed to convert image {image_path}: {str(e)}") from e

    @staticmethod
    def enhance_image(
        image_path: Union[str, Path],
        brightness: float = 1.0,
        contrast: float = 1.0,
        sharpness: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Enhance image with brightness, contrast, and sharpness adjustments.

        This method applies enhancement filters to improve image quality
        for better analysis results. All enhancement factors default to 1.0
        (no change), with values >1.0 increasing the effect.

        Args:
            image_path: Path to the input image file.
            brightness: Brightness factor (1.0 = no change, >1.0 = brighter, <1.0 = darker).
            contrast: Contrast factor (1.0 = no change, >1.0 = more contrast).
            sharpness: Sharpness factor (1.0 = no change, >1.0 = sharper, <1.0 = softer).
            output_path: Path for the enhanced image. If None, creates a new file
                        with '_enhanced' suffix.

        Returns:
            Path to the enhanced image file as string.

        Raises:
            ProcessingError: If PIL/Pillow is not available or enhancement fails.
        """
        try:
            from PIL import Image, ImageEnhance  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "PIL/Pillow is required for image preprocessing. Install with: pip install Pillow"
            ) from exc

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_enhanced{image_path.suffix}"

        try:
            with Image.open(image_path) as img:
                # Apply enhancements in sequence
                if brightness != 1.0:
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(brightness)

                if contrast != 1.0:
                    enhancer = ImageEnhance.Contrast(img)
                    img = enhancer.enhance(contrast)

                if sharpness != 1.0:
                    enhancer = ImageEnhance.Sharpness(img)
                    img = enhancer.enhance(sharpness)

                img.save(output_path, optimize=True, quality=85)
                return str(output_path)
        except Exception as e:
            raise ProcessingError(f"Failed to enhance image {image_path}: {str(e)}") from e

    @staticmethod
    def crop_image(
        image_path: Union[str, Path],
        bbox: Tuple[int, int, int, int],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Crop image to specified bounding box coordinates.

        This method extracts a rectangular region from the image based on
        the provided bounding box coordinates.

        Args:
            image_path: Path to the input image file.
            bbox: Bounding box as (left, top, right, bottom) coordinates in pixels.
            output_path: Path for the cropped image. If None, creates a new file
                        with '_cropped' suffix.

        Returns:
            Path to the cropped image file as string.

        Raises:
            ProcessingError: If PIL/Pillow is not available or cropping fails.
        """
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "PIL/Pillow is required for image preprocessing. Install with: pip install Pillow"
            ) from exc

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_cropped{image_path.suffix}"

        try:
            with Image.open(image_path) as img:
                # Validate bounding box
                left, top, right, bottom = bbox
                if left < 0 or top < 0 or right > img.width or bottom > img.height:
                    raise ValueError(
                        f"Bounding box {bbox} is outside image dimensions {img.width}x{img.height}"
                    )
                if left >= right or top >= bottom:
                    raise ValueError(f"Invalid bounding box {bbox}: left >= right or top >= bottom")

                cropped = img.crop(bbox)
                cropped.save(output_path, optimize=True, quality=85)
                return str(output_path)
        except Exception as e:
            raise ProcessingError(f"Failed to crop image {image_path}: {str(e)}") from e

    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> dict:
        """Get comprehensive information about an image file.

        This method analyzes an image and returns detailed technical information
        including dimensions, color mode, file size, and calculated properties.

        Args:
            image_path: Path to the image file to analyze.

        Returns:
            Dictionary containing image information with keys:
            - width, height: Image dimensions in pixels
            - mode: Color mode (RGB, RGBA, L, etc.)
            - format: Image format (JPEG, PNG, etc.)
            - size_bytes: File size in bytes
            - aspect_ratio: Width/height ratio
            - total_pixels: Total number of pixels
            - estimated_quality: Quality estimate based on size and resolution

        Raises:
            ProcessingError: If PIL/Pillow is not available or analysis fails.
        """
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "PIL/Pillow is required for image preprocessing. Install with: pip install Pillow"
            ) from exc

        try:
            with Image.open(image_path) as img:
                file_size = Path(image_path).stat().st_size
                total_pixels = img.width * img.height

                return {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "size_bytes": file_size,
                    "aspect_ratio": round(img.width / img.height, 3) if img.height > 0 else 0,
                    "total_pixels": total_pixels,
                    "estimated_quality": ImagePreprocessor._estimate_quality(
                        total_pixels, file_size
                    ),
                    "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                }
        except Exception as e:
            raise ProcessingError(f"Failed to get image info for {image_path}: {str(e)}") from e

    @staticmethod
    def _estimate_quality(
        total_pixels: int, file_size: int  # pylint: disable=unused-argument
    ) -> str:
        """Estimate image quality based on pixel count and file size.

        Args:
            total_pixels: Total number of pixels in the image.
            file_size: File size in bytes.

        Returns:
            Quality estimate string.
        """
        if total_pixels > 8000000:  # 8MP+
            return "very_high"
        if total_pixels > 2000000:  # 2MP+
            return "high"
        if total_pixels > 500000:  # 0.5MP+
            return "medium"
        if total_pixels > 100000:  # 0.1MP+
            return "low"
        return "very_low"

    @staticmethod
    def encode_image_to_base64(image_path: Union[str, Path]) -> str:
        """Encode image file to base64 string for API transmission.

        This method reads an image file and converts it to a base64 encoded
        string suitable for transmission to AI APIs.

        Args:
            image_path: Path to the image file to encode.

        Returns:
            Base64 encoded image data as string.

        Raises:
            ProcessingError: If file reading or encoding fails.
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise ProcessingError(f"Failed to encode image {image_path}: {str(e)}") from e

    @staticmethod
    def is_image_too_large(image_path: Union[str, Path], max_size_mb: float = 20.0) -> bool:
        """Check if image file exceeds specified size limit.

        This method checks if an image file is too large for processing,
        which is useful for API limitations or performance considerations.

        Args:
            image_path: Path to the image file to check.
            max_size_mb: Maximum allowed size in megabytes (default: 20MB).

        Returns:
            True if the image file exceeds the size limit, False otherwise.
        """
        file_size_mb = Path(image_path).stat().st_size / (1024 * 1024)
        return file_size_mb > max_size_mb

    @staticmethod
    def optimize_for_api(
        image_path: Union[str, Path],
        max_size: Tuple[int, int] = (1024, 1024),
        max_file_size_mb: float = 20.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Optimize image for AI API consumption with size and format constraints.

        This method performs a complete optimization workflow to prepare images
        for AI API processing. It combines format conversion, resizing, and
        compression to meet typical API requirements.

        Args:
            image_path: Path to the input image file.
            max_size: Maximum dimensions as (width, height) tuple.
            max_file_size_mb: Maximum file size in megabytes.
            output_path: Path for the optimized image. If None, creates a new file
                        with '_optimized.jpg' suffix.

        Returns:
            Path to the optimized image file as string.

        Raises:
            ProcessingError: If optimization process fails.
        """
        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_optimized.jpg"

        try:
            # Step 1: Convert to RGB format for consistency
            rgb_path = ImagePreprocessor.convert_to_rgb(image_path)
            temp_files_to_clean = [rgb_path] if rgb_path != str(image_path) else []

            # Step 2: Resize if necessary
            resized_path = ImagePreprocessor.resize_image(rgb_path, max_size)
            if resized_path != rgb_path:
                temp_files_to_clean.append(resized_path)

            # Step 3: Compress if file is still too large
            final_path = resized_path
            quality = 85

            while (
                ImagePreprocessor.is_image_too_large(final_path, max_file_size_mb) and quality > 20
            ):
                try:
                    from PIL import Image  # pylint: disable=import-outside-toplevel

                    with Image.open(final_path) as img:
                        compressed_path = str(output_path).replace(".jpg", f"_q{quality}.jpg")
                        img.save(compressed_path, "JPEG", optimize=True, quality=quality)

                        # Clean up previous file if it was temporary
                        if final_path != str(image_path) and final_path in temp_files_to_clean:
                            Path(final_path).unlink(missing_ok=True)

                        final_path = compressed_path
                        quality -= 10

                except ImportError:
                    break

            # Move to final output path if different
            if final_path != str(output_path):
                Path(final_path).rename(output_path)

            # Clean up temporary files
            for temp_file in temp_files_to_clean:
                if temp_file != final_path and temp_file != str(output_path):
                    Path(temp_file).unlink(missing_ok=True)

            return str(output_path)

        except Exception as e:
            raise ProcessingError(f"Failed to optimize image {image_path}: {str(e)}") from e

    @staticmethod
    def batch_optimize_images(
        image_paths: list[Union[str, Path]],
        output_dir: Optional[Union[str, Path]] = None,
        **optimization_kwargs: Any,
    ) -> dict[str, Union[str, Exception]]:
        """Optimize multiple images in batch with error handling.

        This method processes multiple images using the optimize_for_api method
        and provides comprehensive error handling for batch operations.

        Args:
            image_paths: List of paths to image files to optimize.
            output_dir: Directory for optimized images. If None, uses source directories.
            **optimization_kwargs: Additional arguments passed to optimize_for_api.

        Returns:
            Dictionary mapping input paths to either output paths (str) or
            exceptions that occurred during processing.
        """
        results: dict[str, Union[str, Exception]] = {}

        for image_path in image_paths:
            try:
                if output_dir:
                    output_path = Path(output_dir) / f"{Path(image_path).stem}_optimized.jpg"
                    optimization_kwargs["output_path"] = output_path

                result_path = ImagePreprocessor.optimize_for_api(image_path, **optimization_kwargs)
                results[str(image_path)] = str(result_path)

            except Exception as e:  # pylint: disable=broad-exception-caught
                results[str(image_path)] = e

        return results
