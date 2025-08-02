"""Tests for image preprocessing utilities."""

import base64
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from agraph.processer.base import ProcessingError
from agraph.processer.image_utils import ImagePreprocessor
from tests.processer.conftest import skip_if_no_module


class TestImagePreprocessor:
    """Test ImagePreprocessor functionality."""

    def test_resize_image_no_pillow(self, temp_dir):
        """Test image resizing when Pillow is not available."""
        image_file = temp_dir / "test.jpg"
        image_file.write_bytes(b"fake image data")

        with patch.dict("sys.modules", {"PIL": None}):
            with pytest.raises(ProcessingError, match="PIL/Pillow is required"):
                ImagePreprocessor.resize_image(image_file)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_resize_image_success(self, temp_dir):
        """Test successful image resizing."""
        image_file = temp_dir / "test.jpg"
        output_file = temp_dir / "resized.jpg"

        # Mock PIL Image with context manager support
        mock_image = Mock()
        mock_image.thumbnail = Mock()
        mock_image.save = Mock()
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            result = ImagePreprocessor.resize_image(image_file, (800, 600), output_file)

            assert result == str(output_file)
            # Check that thumbnail was called with the right size
            args, kwargs = mock_image.thumbnail.call_args
            assert args[0] == (800, 600)
            mock_image.save.assert_called_once_with(output_file, optimize=True, quality=85)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_resize_image_auto_output_path(self, temp_dir):
        """Test image resizing with automatic output path."""
        image_file = temp_dir / "test.jpg"

        mock_image = Mock()
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            result = ImagePreprocessor.resize_image(image_file)

            expected_path = temp_dir / "test_resized.jpg"
            assert result == str(expected_path)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_resize_image_error(self, temp_dir):
        """Test image resizing with processing error."""
        image_file = temp_dir / "corrupted.jpg"

        with patch("PIL.Image.open", side_effect=Exception("Corrupted image")):
            with pytest.raises(ProcessingError, match="Failed to resize image"):
                ImagePreprocessor.resize_image(image_file)

    def test_convert_to_rgb_no_pillow(self, temp_dir):
        """Test RGB conversion when Pillow is not available."""
        image_file = temp_dir / "test.png"
        image_file.write_bytes(b"fake image data")

        with patch.dict("sys.modules", {"PIL": None}):
            with pytest.raises(ProcessingError, match="PIL/Pillow is required"):
                ImagePreprocessor.convert_to_rgb(image_file)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_convert_to_rgb_success(self, temp_dir):
        """Test successful RGB conversion."""
        image_file = temp_dir / "test.png"
        output_file = temp_dir / "rgb.jpg"

        mock_image = Mock()
        mock_image.mode = "RGBA"
        mock_converted = Mock()
        mock_image.convert.return_value = mock_converted
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            result = ImagePreprocessor.convert_to_rgb(image_file, output_file)

            assert result == str(output_file)
            mock_image.convert.assert_called_once_with("RGB")
            mock_converted.save.assert_called_once_with(output_file, "JPEG", optimize=True, quality=85)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_convert_to_rgb_already_rgb(self, temp_dir):
        """Test RGB conversion when image is already RGB."""
        image_file = temp_dir / "test.jpg"

        mock_image = Mock()
        mock_image.mode = "RGB"
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            result = ImagePreprocessor.convert_to_rgb(image_file)

            # Should not call convert when already RGB
            mock_image.convert.assert_not_called()
            mock_image.save.assert_called_once()

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_convert_to_rgb_auto_output_path(self, temp_dir):
        """Test RGB conversion with automatic output path."""
        image_file = temp_dir / "test.png"

        mock_image = Mock()
        mock_image.mode = "RGBA"
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            result = ImagePreprocessor.convert_to_rgb(image_file)

            expected_path = temp_dir / "test_rgb.jpg"
            assert result == str(expected_path)

    def test_enhance_image_no_pillow(self, temp_dir):
        """Test image enhancement when Pillow is not available."""
        image_file = temp_dir / "test.jpg"
        image_file.write_bytes(b"fake image data")

        with patch.dict("sys.modules", {"PIL": None}):
            with pytest.raises(ProcessingError, match="PIL/Pillow is required"):
                ImagePreprocessor.enhance_image(image_file)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_enhance_image_success(self, temp_dir):
        """Test successful image enhancement."""
        image_file = temp_dir / "test.jpg"
        output_file = temp_dir / "enhanced.jpg"

        mock_image = Mock()
        mock_enhancer = Mock()
        mock_enhanced = Mock()
        mock_enhancer.enhance.return_value = mock_enhanced
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            with patch("PIL.ImageEnhance.Brightness", return_value=mock_enhancer):
                with patch("PIL.ImageEnhance.Contrast", return_value=mock_enhancer):
                    with patch("PIL.ImageEnhance.Sharpness", return_value=mock_enhancer):
                        result = ImagePreprocessor.enhance_image(
                            image_file,
                            brightness=1.2,
                            contrast=1.1,
                            sharpness=1.3,
                            output_path=output_file
                        )

                        assert result == str(output_file)
                        # Should enhance all three properties
                        assert mock_enhancer.enhance.call_count == 3

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_enhance_image_no_changes(self, temp_dir):
        """Test image enhancement with default values (no changes)."""
        image_file = temp_dir / "test.jpg"

        mock_image = Mock()
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            with patch("PIL.ImageEnhance.Brightness") as mock_brightness:
                with patch("PIL.ImageEnhance.Contrast") as mock_contrast:
                    with patch("PIL.ImageEnhance.Sharpness") as mock_sharpness:
                        ImagePreprocessor.enhance_image(image_file)

                        # Should not create enhancers for default values
                        mock_brightness.assert_not_called()
                        mock_contrast.assert_not_called()
                        mock_sharpness.assert_not_called()

    def test_crop_image_no_pillow(self, temp_dir):
        """Test image cropping when Pillow is not available."""
        image_file = temp_dir / "test.jpg"
        image_file.write_bytes(b"fake image data")

        with patch.dict("sys.modules", {"PIL": None}):
            with pytest.raises(ProcessingError, match="PIL/Pillow is required"):
                ImagePreprocessor.crop_image(image_file, (10, 10, 100, 100))

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_crop_image_success(self, temp_dir):
        """Test successful image cropping."""
        image_file = temp_dir / "test.jpg"
        output_file = temp_dir / "cropped.jpg"
        bbox = (10, 10, 100, 100)

        mock_image = Mock()
        mock_cropped = Mock()
        mock_image.crop.return_value = mock_cropped
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            result = ImagePreprocessor.crop_image(image_file, bbox, output_file)

            assert result == str(output_file)
            mock_image.crop.assert_called_once_with(bbox)
            mock_cropped.save.assert_called_once_with(output_file, optimize=True, quality=85)

    def test_get_image_info_no_pillow(self, temp_dir):
        """Test getting image info when Pillow is not available."""
        image_file = temp_dir / "test.jpg"
        image_file.write_bytes(b"fake image data")

        with patch.dict("sys.modules", {"PIL": None}):
            with pytest.raises(ProcessingError, match="PIL/Pillow is required"):
                ImagePreprocessor.get_image_info(image_file)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_get_image_info_success(self, temp_dir):
        """Test successful image info extraction."""
        image_file = temp_dir / "test.jpg"
        image_file.write_bytes(b"fake image data" * 100)  # Make file non-empty

        mock_image = Mock()
        mock_image.width = 800
        mock_image.height = 600
        mock_image.mode = "RGB"
        mock_image.format = "JPEG"
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch("PIL.Image.open", return_value=mock_image):
            info = ImagePreprocessor.get_image_info(image_file)

            assert info["width"] == 800
            assert info["height"] == 600
            assert info["mode"] == "RGB"
            assert info["format"] == "JPEG"
            assert info["aspect_ratio"] == 800 / 600
            assert "size_bytes" in info

    def test_encode_image_to_base64(self, temp_dir):
        """Test encoding image to base64."""
        image_file = temp_dir / "test.jpg"
        image_data = b"fake image data"
        image_file.write_bytes(image_data)

        result = ImagePreprocessor.encode_image_to_base64(image_file)

        expected = base64.b64encode(image_data).decode("utf-8")
        assert result == expected

    def test_encode_image_to_base64_error(self, temp_dir):
        """Test base64 encoding with file error."""
        nonexistent_file = temp_dir / "nonexistent.jpg"

        with pytest.raises(ProcessingError, match="Failed to encode image"):
            ImagePreprocessor.encode_image_to_base64(nonexistent_file)

    def test_is_image_too_large_small_file(self, temp_dir):
        """Test checking if small image is too large."""
        image_file = temp_dir / "small.jpg"
        image_file.write_bytes(b"small image data")

        result = ImagePreprocessor.is_image_too_large(image_file, max_size_mb=1.0)

        assert result is False

    def test_is_image_too_large_large_file(self, temp_dir):
        """Test checking if large image is too large."""
        image_file = temp_dir / "large.jpg"
        # Create a file larger than 1MB
        large_data = b"x" * (2 * 1024 * 1024)  # 2MB
        image_file.write_bytes(large_data)

        result = ImagePreprocessor.is_image_too_large(image_file, max_size_mb=1.0)

        assert result is True

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_optimize_for_api_success(self, temp_dir):
        """Test optimizing image for API consumption."""
        image_file = temp_dir / "test.png"
        output_file = temp_dir / "optimized.jpg"

        # Mock the individual methods
        with patch.object(ImagePreprocessor, "convert_to_rgb", return_value=str(temp_dir / "rgb.jpg")):
            with patch.object(ImagePreprocessor, "resize_image", return_value=str(temp_dir / "resized.jpg")):
                with patch.object(ImagePreprocessor, "is_image_too_large", return_value=False):
                    with patch("pathlib.Path.rename") as mock_rename:
                        result = ImagePreprocessor.optimize_for_api(image_file, output_path=output_file)

                        assert result == str(output_file)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_optimize_for_api_auto_output_path(self, temp_dir):
        """Test optimizing image with automatic output path."""
        image_file = temp_dir / "test.png"

        with patch.object(ImagePreprocessor, "convert_to_rgb", return_value=str(temp_dir / "rgb.jpg")):
            with patch.object(ImagePreprocessor, "resize_image", return_value=str(temp_dir / "resized.jpg")):
                with patch.object(ImagePreprocessor, "is_image_too_large", return_value=False):
                    with patch("pathlib.Path.rename"):
                        result = ImagePreprocessor.optimize_for_api(image_file)

                        expected_path = temp_dir / "test_optimized.jpg"
                        assert result == str(expected_path)

    @pytest.mark.skipif(skip_if_no_module("PIL"), reason="Pillow not available")
    def test_optimize_for_api_compress_large_file(self, temp_dir):
        """Test optimizing large image that needs compression."""
        image_file = temp_dir / "large.png"

        # Mock a large file that needs compression
        mock_image = Mock()
        mock_image.__enter__ = Mock(return_value=mock_image)
        mock_image.__exit__ = Mock(return_value=None)

        with patch.object(ImagePreprocessor, "convert_to_rgb", return_value=str(temp_dir / "rgb.jpg")):
            with patch.object(ImagePreprocessor, "resize_image", return_value=str(temp_dir / "resized.jpg")):
                # First call returns True (too large), subsequent calls return False
                with patch.object(ImagePreprocessor, "is_image_too_large", side_effect=[True, False]):
                    with patch("PIL.Image.open", return_value=mock_image):
                        with patch("pathlib.Path.rename"):
                            with patch("pathlib.Path.unlink"):
                                result = ImagePreprocessor.optimize_for_api(image_file)

                                # Should attempt to compress
                                mock_image.save.assert_called()

    def test_optimize_for_api_cleanup_temp_files(self, temp_dir):
        """Test that temporary files are cleaned up."""
        image_file = temp_dir / "test.png"

        with patch.object(ImagePreprocessor, "convert_to_rgb", return_value=str(temp_dir / "temp_rgb.jpg")):
            with patch.object(ImagePreprocessor, "resize_image", return_value=str(temp_dir / "temp_resized.jpg")):
                with patch.object(ImagePreprocessor, "is_image_too_large", return_value=False):
                    with patch("pathlib.Path.rename"):
                        with patch("pathlib.Path.unlink") as mock_unlink:
                            ImagePreprocessor.optimize_for_api(image_file)

                            # Should clean up RGB temp file
                            mock_unlink.assert_called()
