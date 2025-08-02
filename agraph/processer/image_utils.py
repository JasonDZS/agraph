import base64
from pathlib import Path
from typing import Optional, Tuple, Union

from .base import ProcessingError


class ImagePreprocessor:
    """Utility class for image preprocessing before analysis."""

    @staticmethod
    def resize_image(
        image_path: Union[str, Path],
        max_size: Tuple[int, int] = (1024, 1024),
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Resize image to fit within max_size while maintaining aspect ratio.

        Args:
            image_path: Path to input image
            max_size: Maximum (width, height) dimensions
            output_path: Path for resized image (if None, overwrites original)

        Returns:
            Path to resized image
        """
        try:
            from PIL import Image
        except ImportError:
            raise ProcessingError("PIL/Pillow is required for image preprocessing. Install with: pip install Pillow")

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_resized{image_path.suffix}"

        try:
            with Image.open(image_path) as img:
                # Calculate new size maintaining aspect ratio
                img.thumbnail(max_size, Image.Resampling.LANCZOS)
                img.save(output_path, optimize=True, quality=85)
                return str(output_path)
        except Exception as e:
            raise ProcessingError(f"Failed to resize image {image_path}: {str(e)}")

    @staticmethod
    def convert_to_rgb(image_path: Union[str, Path], output_path: Optional[Union[str, Path]] = None) -> str:
        """Convert image to RGB format.

        Args:
            image_path: Path to input image
            output_path: Path for converted image (if None, creates new file)

        Returns:
            Path to converted image
        """
        try:
            from PIL import Image
        except ImportError:
            raise ProcessingError("PIL/Pillow is required for image preprocessing. Install with: pip install Pillow")

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_rgb.jpg"

        try:
            with Image.open(image_path) as img:
                # Convert to RGB
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.save(output_path, "JPEG", optimize=True, quality=85)
                return str(output_path)
        except Exception as e:
            raise ProcessingError(f"Failed to convert image {image_path}: {str(e)}")

    @staticmethod
    def enhance_image(
        image_path: Union[str, Path],
        brightness: float = 1.0,
        contrast: float = 1.0,
        sharpness: float = 1.0,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Enhance image with brightness, contrast, and sharpness adjustments.

        Args:
            image_path: Path to input image
            brightness: Brightness factor (1.0 = no change, >1.0 = brighter)
            contrast: Contrast factor (1.0 = no change, >1.0 = more contrast)
            sharpness: Sharpness factor (1.0 = no change, >1.0 = sharper)
            output_path: Path for enhanced image

        Returns:
            Path to enhanced image
        """
        try:
            from PIL import Image, ImageEnhance
        except ImportError:
            raise ProcessingError("PIL/Pillow is required for image preprocessing. Install with: pip install Pillow")

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_enhanced{image_path.suffix}"

        try:
            with Image.open(image_path) as img:
                # Apply enhancements
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
            raise ProcessingError(f"Failed to enhance image {image_path}: {str(e)}")

    @staticmethod
    def crop_image(
        image_path: Union[str, Path], bbox: Tuple[int, int, int, int], output_path: Optional[Union[str, Path]] = None
    ) -> str:
        """Crop image to specified bounding box.

        Args:
            image_path: Path to input image
            bbox: Bounding box as (left, top, right, bottom)
            output_path: Path for cropped image

        Returns:
            Path to cropped image
        """
        try:
            from PIL import Image
        except ImportError:
            raise ProcessingError("PIL/Pillow is required for image preprocessing. Install with: pip install Pillow")

        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_cropped{image_path.suffix}"

        try:
            with Image.open(image_path) as img:
                cropped = img.crop(bbox)
                cropped.save(output_path, optimize=True, quality=85)
                return str(output_path)
        except Exception as e:
            raise ProcessingError(f"Failed to crop image {image_path}: {str(e)}")

    @staticmethod
    def get_image_info(image_path: Union[str, Path]) -> dict:
        """Get basic information about an image.

        Args:
            image_path: Path to the image

        Returns:
            Dictionary with image information
        """
        try:
            from PIL import Image
        except ImportError:
            raise ProcessingError("PIL/Pillow is required for image preprocessing. Install with: pip install Pillow")

        try:
            with Image.open(image_path) as img:
                return {
                    "width": img.width,
                    "height": img.height,
                    "mode": img.mode,
                    "format": img.format,
                    "size_bytes": Path(image_path).stat().st_size,
                    "aspect_ratio": img.width / img.height,
                }
        except Exception as e:
            raise ProcessingError(f"Failed to get image info for {image_path}: {str(e)}")

    @staticmethod
    def encode_image_to_base64(image_path: Union[str, Path]) -> str:
        """Encode image to base64 string.

        Args:
            image_path: Path to the image

        Returns:
            Base64 encoded image data
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            raise ProcessingError(f"Failed to encode image {image_path}: {str(e)}")

    @staticmethod
    def is_image_too_large(image_path: Union[str, Path], max_size_mb: float = 20.0) -> bool:
        """Check if image file is too large.

        Args:
            image_path: Path to the image
            max_size_mb: Maximum size in megabytes

        Returns:
            True if image is too large
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
        """Optimize image for API consumption.

        Args:
            image_path: Path to input image
            max_size: Maximum dimensions
            max_file_size_mb: Maximum file size in MB
            output_path: Path for optimized image

        Returns:
            Path to optimized image
        """
        image_path = Path(image_path)
        if output_path is None:
            output_path = image_path.parent / f"{image_path.stem}_optimized.jpg"

        # Convert to RGB first
        rgb_path = ImagePreprocessor.convert_to_rgb(image_path)

        # Resize if necessary
        resized_path = ImagePreprocessor.resize_image(rgb_path, max_size)

        # Check file size and compress if needed
        quality = 85
        while ImagePreprocessor.is_image_too_large(resized_path, max_file_size_mb) and quality > 20:
            try:
                from PIL import Image

                with Image.open(resized_path) as img:
                    img.save(resized_path, "JPEG", optimize=True, quality=quality)
                quality -= 10
            except ImportError:
                break

        # Move to final output path if different
        if str(resized_path) != str(output_path):
            Path(resized_path).rename(output_path)

        # Clean up temporary files
        if rgb_path != str(image_path):
            Path(rgb_path).unlink(missing_ok=True)

        return str(output_path)
