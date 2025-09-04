"""
Test image processor functionality.
"""

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agraph.processor.base import ProcessingError
from agraph.processor.image_processor import (
    ClaudeVisionModel,
    ImageProcessor,
    ImageProcessorFactory,
    OpenAIVisionModel,
)


class TestImageProcessor(unittest.TestCase):
    """Test image document processor."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def test_supported_extensions(self):
        """Test supported file extensions."""
        processor = ImageProcessor()
        extensions = processor.supported_extensions
        expected = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]
        self.assertEqual(extensions, expected)

    def test_can_process_image_files(self):
        """Test image file type detection."""
        processor = ImageProcessor()
        self.assertTrue(processor.can_process("photo.jpg"))
        self.assertTrue(processor.can_process("image.png"))
        self.assertTrue(processor.can_process("animation.gif"))
        self.assertFalse(processor.can_process("document.txt"))

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("agraph.processor.image_processor.OpenAIVisionModel")
    def test_processor_initialization_with_openai(self, mock_openai_model):
        """Test processor initialization with OpenAI model."""
        mock_model = MagicMock()
        mock_openai_model.return_value = mock_model

        processor = ImageProcessor()

        self.assertEqual(processor.model, mock_model)
        mock_openai_model.assert_called_once()

    @patch.dict("os.environ", {}, clear=True)
    def test_processor_initialization_no_api_keys(self):
        """Test processor initialization without API keys raises error."""
        with self.assertRaises(ProcessingError) as context:
            ImageProcessor()

        self.assertIn("No multimodal model available", str(context.exception))

    def test_processor_initialization_with_custom_model(self):
        """Test processor initialization with custom model."""
        mock_model = MagicMock()
        processor = ImageProcessor(model=mock_model)

        self.assertEqual(processor.model, mock_model)

    def test_process_image_with_mock_model(self):
        """Test processing image with mocked model."""
        test_file = Path(self.temp_dir) / "test.jpg"
        test_file.write_bytes(b"fake image data")

        mock_model = MagicMock()
        mock_model.analyze_image.return_value = "Mock image description"

        processor = ImageProcessor(model=mock_model)
        result = processor.process(test_file)

        self.assertEqual(result, "Mock image description")
        mock_model.analyze_image.assert_called_once()

    def test_process_with_custom_prompt(self):
        """Test processing with custom prompt."""
        test_file = Path(self.temp_dir) / "test.png"
        test_file.write_bytes(b"fake image data")

        mock_model = MagicMock()
        mock_model.analyze_image.return_value = "Custom analysis result"

        processor = ImageProcessor(model=mock_model)
        result = processor.process_with_custom_prompt(test_file, "Custom prompt")

        mock_model.analyze_image.assert_called_with(test_file, "Custom prompt")
        self.assertEqual(result, "Custom analysis result")

    def test_extract_text_from_image(self):
        """Test OCR-like text extraction from image."""
        test_file = Path(self.temp_dir) / "text_image.jpg"
        test_file.write_bytes(b"fake image with text")

        mock_model = MagicMock()
        mock_model.analyze_image.return_value = "Extracted text from image"

        processor = ImageProcessor(model=mock_model)
        result = processor.extract_text_from_image(test_file)

        self.assertEqual(result, "Extracted text from image")
        # Check that OCR prompt was used
        args, kwargs = mock_model.analyze_image.call_args
        self.assertIn("extract and transcribe", args[1].lower())

    def test_describe_image_brief(self):
        """Test brief image description."""
        test_file = Path(self.temp_dir) / "test.jpg"
        test_file.write_bytes(b"fake image")

        mock_model = MagicMock()
        mock_model.analyze_image.return_value = "Brief description"

        processor = ImageProcessor(model=mock_model)
        result = processor.describe_image(test_file, detail_level="brief")

        self.assertEqual(result, "Brief description")
        args, kwargs = mock_model.analyze_image.call_args
        self.assertIn("brief", args[1].lower())

    def test_describe_image_comprehensive(self):
        """Test comprehensive image description."""
        test_file = Path(self.temp_dir) / "test.jpg"
        test_file.write_bytes(b"fake image")

        mock_model = MagicMock()
        mock_model.analyze_image.return_value = "Comprehensive analysis"

        processor = ImageProcessor(model=mock_model)
        result = processor.describe_image(test_file, detail_level="comprehensive")

        self.assertEqual(result, "Comprehensive analysis")
        args, kwargs = mock_model.analyze_image.call_args
        self.assertIn("extremely comprehensive", args[1].lower())

    def test_process_unsupported_format(self):
        """Test processing unsupported image format."""
        test_file = Path(self.temp_dir) / "test.svg"
        test_file.write_bytes(b"fake svg")

        mock_model = MagicMock()
        processor = ImageProcessor(model=mock_model)

        with self.assertRaises(ProcessingError) as context:
            processor.process(test_file)

        self.assertIn("Unsupported image format", str(context.exception))

    def test_process_model_failure(self):
        """Test handling of model analysis failure."""
        test_file = Path(self.temp_dir) / "test.jpg"
        test_file.write_bytes(b"fake image")

        mock_model = MagicMock()
        mock_model.analyze_image.side_effect = Exception("Model error")

        processor = ImageProcessor(model=mock_model)

        with self.assertRaises(ProcessingError) as context:
            processor.process(test_file)

        self.assertIn("Failed to process image", str(context.exception))

    @patch("agraph.processor.image_processor.Image")
    def test_extract_metadata_basic(self, mock_image):
        """Test basic metadata extraction."""
        test_file = Path(self.temp_dir) / "test.jpg"
        test_file.write_bytes(b"fake image")

        # Mock PIL Image
        mock_img = MagicMock()
        mock_img.width = 1920
        mock_img.height = 1080
        mock_img.mode = "RGB"
        mock_img.format = "JPEG"
        mock_img.info = {}
        mock_img._getexif.return_value = None
        mock_image.open.return_value.__enter__.return_value = mock_img

        mock_model = MagicMock()
        processor = ImageProcessor(model=mock_model)
        metadata = processor.extract_metadata(test_file)

        self.assertEqual(metadata["width"], 1920)
        self.assertEqual(metadata["height"], 1080)
        self.assertEqual(metadata["mode"], "RGB")
        self.assertEqual(metadata["format"], "JPEG")
        self.assertFalse(metadata["has_transparency"])

    @patch("agraph.processor.image_processor.Image")
    def test_extract_metadata_with_transparency(self, mock_image):
        """Test metadata extraction for images with transparency."""
        test_file = Path(self.temp_dir) / "transparent.png"
        test_file.write_bytes(b"fake png with transparency")

        # Mock PNG with transparency
        mock_img = MagicMock()
        mock_img.width = 500
        mock_img.height = 500
        mock_img.mode = "RGBA"
        mock_img.format = "PNG"
        mock_img.info = {}
        mock_img._getexif.return_value = None
        mock_image.open.return_value.__enter__.return_value = mock_img

        mock_model = MagicMock()
        processor = ImageProcessor(model=mock_model)
        metadata = processor.extract_metadata(test_file)

        self.assertTrue(metadata["has_transparency"])

    def test_extract_metadata_without_pillow(self):
        """Test metadata extraction when PIL is not available."""
        test_file = Path(self.temp_dir) / "test.jpg"
        test_file.write_bytes(b"fake image")

        mock_model = MagicMock()
        processor = ImageProcessor(model=mock_model)

        with patch.dict("sys.modules", {"PIL": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                metadata = processor.extract_metadata(test_file)

                self.assertIn("pillow_error", metadata)

    def test_estimate_image_quality(self):
        """Test image quality estimation."""
        mock_model = MagicMock()
        processor = ImageProcessor(model=mock_model)

        # Mock images of different sizes
        mock_very_high = MagicMock()
        mock_very_high.width = 4000
        mock_very_high.height = 3000  # 12MP

        mock_medium = MagicMock()
        mock_medium.width = 800
        mock_medium.height = 600  # 0.48MP

        mock_low = MagicMock()
        mock_low.width = 200
        mock_low.height = 150  # 0.03MP

        self.assertEqual(processor._estimate_image_quality(mock_very_high), "very_high")
        self.assertEqual(processor._estimate_image_quality(mock_medium), "low")
        self.assertEqual(processor._estimate_image_quality(mock_low), "very_low")

    def test_process_empty_file_raises_error(self):
        """Test that processing empty file raises appropriate error."""
        empty_file = Path(self.temp_dir) / "empty.jpg"
        empty_file.touch()

        mock_model = MagicMock()
        processor = ImageProcessor(model=mock_model)

        with self.assertRaises(ValueError) as context:
            processor.process(empty_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_nonexistent_file_raises_error(self):
        """Test that processing non-existent file raises appropriate error."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.jpg"

        mock_model = MagicMock()
        processor = ImageProcessor(model=mock_model)

        with self.assertRaises(FileNotFoundError):
            processor.process(nonexistent_file)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestOpenAIVisionModel(unittest.TestCase):
    """Test OpenAI Vision model."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    @patch("agraph.processor.image_processor.openai")
    @patch("agraph.processor.image_processor.get_settings")
    def test_openai_model_initialization(self, mock_settings, mock_openai):
        """Test OpenAI model initialization."""
        mock_settings.return_value.openai.api_key = "test_key"
        mock_settings.return_value.openai.api_base = "https://api.openai.com/v1"

        model = OpenAIVisionModel()

        self.assertEqual(model.model, "gpt-4o")
        mock_openai.OpenAI.assert_called_once()

    def test_openai_model_without_openai_package(self):
        """Test OpenAI model initialization without openai package."""
        with patch.dict("sys.modules", {"openai": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with self.assertRaises(ProcessingError) as context:
                    OpenAIVisionModel()

                self.assertIn("openai package", str(context.exception))

    @patch("agraph.processor.image_processor.openai")
    @patch("agraph.processor.image_processor.get_settings")
    def test_analyze_image_base64(self, mock_settings, mock_openai):
        """Test image analysis with base64 data."""
        mock_settings.return_value.openai.api_key = "test_key"
        mock_settings.return_value.openai.api_base = "https://api.openai.com/v1"

        # Mock OpenAI client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Analysis result"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.OpenAI.return_value = mock_client

        model = OpenAIVisionModel()
        result = model.analyze_image_base64("fake_base64_data", "Analyze this image")

        self.assertEqual(result, "Analysis result")
        mock_client.chat.completions.create.assert_called_once()

    @patch("agraph.processor.image_processor.openai")
    @patch("agraph.processor.image_processor.get_settings")
    def test_analyze_image_api_error(self, mock_settings, mock_openai):
        """Test handling of OpenAI API errors."""
        mock_settings.return_value.openai.api_key = "test_key"
        mock_settings.return_value.openai.api_base = "https://api.openai.com/v1"

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.OpenAI.return_value = mock_client

        model = OpenAIVisionModel()

        with self.assertRaises(ProcessingError) as context:
            model.analyze_image_base64("fake_base64", "prompt")

        self.assertIn("OpenAI vision analysis failed", str(context.exception))


class TestClaudeVisionModel(unittest.TestCase):
    """Test Claude Vision model."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    @patch("agraph.processor.image_processor.anthropic")
    def test_claude_model_initialization(self, mock_anthropic):
        """Test Claude model initialization."""
        model = ClaudeVisionModel(api_key="test_key")

        self.assertEqual(model.model, "claude-3-5-sonnet-20241022")
        mock_anthropic.Anthropic.assert_called_once_with(api_key="test_key")

    def test_claude_model_without_anthropic_package(self):
        """Test Claude model initialization without anthropic package."""
        with patch.dict("sys.modules", {"anthropic": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with self.assertRaises(ProcessingError) as context:
                    ClaudeVisionModel()

                self.assertIn("anthropic package", str(context.exception))

    @patch("agraph.processor.image_processor.anthropic")
    def test_analyze_image_base64(self, mock_anthropic):
        """Test image analysis with base64 data."""
        # Mock Anthropic client and response
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Claude analysis result"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        model = ClaudeVisionModel(api_key="test_key")
        result = model.analyze_image_base64("fake_base64_data", "Analyze this image")

        self.assertEqual(result, "Claude analysis result")
        mock_client.messages.create.assert_called_once()

    def test_detect_image_format_from_base64(self):
        """Test image format detection from base64 data."""
        model = ClaudeVisionModel(api_key="test_key")

        # Test PNG format
        png_data = base64.b64encode(b"\x89PNG\r\n\x1a\n").decode()
        self.assertEqual(model._detect_image_format_from_base64(png_data), "png")

        # Test JPEG format
        jpeg_data = base64.b64encode(b"\xff\xd8\xff").decode()
        self.assertEqual(model._detect_image_format_from_base64(jpeg_data), "jpeg")

        # Test GIF format
        gif_data = base64.b64encode(b"GIF89a").decode()
        self.assertEqual(model._detect_image_format_from_base64(gif_data), "gif")

        # Test unknown format (should default to jpeg)
        unknown_data = base64.b64encode(b"unknown format").decode()
        self.assertEqual(model._detect_image_format_from_base64(unknown_data), "jpeg")


class TestImageProcessorFactory(unittest.TestCase):
    """Test image processor factory."""

    @patch("agraph.processor.image_processor.OpenAIVisionModel")
    def test_create_openai_processor(self, mock_openai_model):
        """Test creating processor with OpenAI model."""
        mock_model = MagicMock()
        mock_openai_model.return_value = mock_model

        processor = ImageProcessorFactory.create_openai_processor(api_key="test_key")

        self.assertIsInstance(processor, ImageProcessor)
        self.assertEqual(processor.model, mock_model)
        mock_openai_model.assert_called_once_with(api_key="test_key", model="gpt-4o")

    @patch("agraph.processor.image_processor.ClaudeVisionModel")
    def test_create_claude_processor(self, mock_claude_model):
        """Test creating processor with Claude model."""
        mock_model = MagicMock()
        mock_claude_model.return_value = mock_model

        processor = ImageProcessorFactory.create_claude_processor(api_key="test_key")

        self.assertIsInstance(processor, ImageProcessor)
        self.assertEqual(processor.model, mock_model)
        mock_claude_model.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "test_key"})
    @patch("agraph.processor.image_processor.OpenAIVisionModel")
    def test_create_auto_processor(self, mock_openai_model):
        """Test creating processor with auto-detected model."""
        mock_model = MagicMock()
        mock_openai_model.return_value = mock_model

        processor = ImageProcessorFactory.create_auto_processor()

        self.assertIsInstance(processor, ImageProcessor)


if __name__ == "__main__":
    unittest.main()
