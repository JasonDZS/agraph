"""Image document processor implementation using multimodal AI models.

This module provides functionality for extracting text and descriptions from images
using state-of-the-art multimodal AI models. It supports multiple AI providers
and various image analysis tasks including OCR, description, and content analysis.
"""

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import get_settings
from .base import DocumentProcessor, ProcessingError


class MultimodalModel(ABC):
    """Abstract base class for multimodal AI models.

    This class defines the interface that all multimodal model implementations
    must follow. It provides methods for analyzing images using either file paths
    or base64 encoded data.
    """

    @abstractmethod
    def analyze_image(self, image_path: Union[str, Path], prompt: str, **kwargs: Any) -> str:
        """Analyze an image file and return text description.

        Args:
            image_path: Path to the image file to analyze.
            prompt: Text prompt describing the analysis task.
            **kwargs: Additional model-specific parameters such as:
                - max_tokens: Maximum response length
                - temperature: Creativity/randomness level
                - detail_level: Level of detail in analysis

        Returns:
            Text description or analysis of the image content.

        Raises:
            ProcessingError: If image analysis fails or model is unavailable.
        """

    @abstractmethod
    def analyze_image_base64(self, image_base64: str, prompt: str, **kwargs: Any) -> str:
        """Analyze an image from base64 data and return text description.

        Args:
            image_base64: Base64 encoded image data.
            prompt: Text prompt describing the analysis task.
            **kwargs: Additional model-specific parameters.

        Returns:
            Text description or analysis of the image content.

        Raises:
            ProcessingError: If image analysis fails or model is unavailable.
        """


class OpenAIVisionModel(MultimodalModel):
    """OpenAI GPT-4V integration for advanced image analysis.

    This model uses OpenAI's GPT-4 with vision capabilities to perform
    sophisticated image analysis including object detection, scene understanding,
    text recognition, and detailed content description.

    Features:
    - Advanced scene understanding
    - Text recognition and OCR
    - Object and person detection
    - Artistic and stylistic analysis
    - Configurable detail levels
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):  # pylint: disable=unused-argument
        """Initialize OpenAI vision model.

        Args:
            api_key: OpenAI API key. If None, uses get_settings().openai.api_key.
            model: Model name to use (default: 'gpt-4o' for optimal performance).

        Raises:
            ProcessingError: If OpenAI package is not installed.
        """
        try:
            import openai  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "openai package is required for OpenAI vision. Install with: pip install openai"
            ) from exc

        self.client = openai.OpenAI(api_key=get_settings().openai.api_key, base_url=get_settings().openai.api_base)
        self.model = model

    def analyze_image(self, image_path: Union[str, Path], prompt: str, **kwargs: Any) -> str:
        """Analyze image using OpenAI GPT-4V.

        Args:
            image_path: Path to the image file.
            prompt: Analysis prompt.
            **kwargs: Additional parameters.

        Returns:
            Analysis result from GPT-4V.
        """
        image_base64 = self._encode_image_to_base64(image_path)
        return self.analyze_image_base64(image_base64, prompt, **kwargs)

    def analyze_image_base64(self, image_base64: str, prompt: str, **kwargs: Any) -> str:
        """Analyze image from base64 using OpenAI GPT-4V.

        Args:
            image_base64: Base64 encoded image.
            prompt: Analysis prompt.
            **kwargs: Additional parameters.

        Returns:
            Analysis result from GPT-4V.

        Raises:
            ProcessingError: If API call fails.
        """
        max_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.1)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                            },
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            raise ProcessingError(f"OpenAI vision analysis failed: {str(e)}") from e

    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """Encode image file to base64 string.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64 encoded image data.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


class ClaudeVisionModel(MultimodalModel):
    """Anthropic Claude Vision integration for image analysis.

    This model uses Anthropic's Claude with vision capabilities for detailed
    image analysis. Claude excels at nuanced understanding and provides
    thoughtful, detailed descriptions.

    Features:
    - Detailed scene analysis
    - Contextual understanding
    - Safety-aware content analysis
    - High-quality text extraction
    - Nuanced artistic interpretation
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Claude vision model.

        Args:
            api_key: Anthropic API key. If None, uses environment variable.
            model: Model name to use (default: latest Claude 3.5 Sonnet).

        Raises:
            ProcessingError: If Anthropic package is not installed.
        """
        try:
            import anthropic  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "anthropic package is required for Claude vision. " "Install with: pip install anthropic"
            ) from exc

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_image(self, image_path: Union[str, Path], prompt: str, **kwargs: Any) -> str:
        """Analyze image using Claude Vision.

        Args:
            image_path: Path to the image file.
            prompt: Analysis prompt.
            **kwargs: Additional parameters.

        Returns:
            Analysis result from Claude.
        """
        image_base64 = self._encode_image_to_base64(image_path)
        return self.analyze_image_base64(image_base64, prompt, **kwargs)

    def analyze_image_base64(self, image_base64: str, prompt: str, **kwargs: Any) -> str:
        """Analyze image from base64 using Claude Vision.

        Args:
            image_base64: Base64 encoded image.
            prompt: Analysis prompt.
            **kwargs: Additional parameters.

        Returns:
            Analysis result from Claude.

        Raises:
            ProcessingError: If API call fails.
        """
        max_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.1)

        # Detect image format for Claude API
        image_format = self._detect_image_format_from_base64(image_base64)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": f"image/{image_format}",
                                    "data": image_base64,
                                },
                            },
                            {"type": "text", "text": prompt},
                        ],
                    }
                ],
            )

            # Extract text from response, handling different block types
            text = ""
            if response.content and len(response.content) > 0:
                first_block = response.content[0]
                if hasattr(first_block, "text") and first_block.text is not None:
                    text = first_block.text

            return text
        except Exception as e:
            raise ProcessingError(f"Claude vision analysis failed: {str(e)}") from e

    def _encode_image_to_base64(self, image_path: Union[str, Path]) -> str:
        """Encode image file to base64 string.

        Args:
            image_path: Path to the image file.

        Returns:
            Base64 encoded image data.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _detect_image_format_from_base64(self, image_base64: str) -> str:
        """Detect image format from base64 data using magic numbers.

        Args:
            image_base64: Base64 encoded image data.

        Returns:
            Image format string (png, jpeg, gif, webp).
        """
        image_data = base64.b64decode(image_base64)

        # Check magic numbers for different image formats
        if image_data.startswith(b"\x89PNG"):
            return "png"
        if image_data.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        if image_data.startswith(b"GIF8"):
            return "gif"
        if image_data.startswith(b"RIFF") and b"WEBP" in image_data[:12]:
            return "webp"
        return "jpeg"  # Default fallback


class ImageProcessor(DocumentProcessor):
    """Document processor for image files using multimodal AI models.

    This processor extracts text content and descriptions from images using
    advanced multimodal AI models. It supports various image analysis tasks
    and provides flexible processing options.

    Features:
    - Multiple AI model support (OpenAI GPT-4V, Claude Vision)
    - Automatic model selection based on available API keys
    - Specialized prompts for different analysis tasks
    - Comprehensive image metadata extraction
    - Support for various image formats
    - OCR-like text extraction capabilities

    Supported formats:
    - JPEG (.jpg, .jpeg)
    - PNG (.png)
    - GIF (.gif)
    - BMP (.bmp)
    - TIFF (.tiff, .tif)
    - WebP (.webp)
    """

    def __init__(self, model: Optional[MultimodalModel] = None, default_prompt: Optional[str] = None):
        """Initialize image processor with multimodal model.

        Args:
            model: Multimodal model to use. If None, auto-detects available model.
            default_prompt: Default analysis prompt. If None, uses comprehensive prompt.
        """
        self.model = model or self._get_default_model()
        self.default_prompt = default_prompt or (
            "Please provide a detailed description of this image. Include information about:\n"
            "1. Objects, people, and animals present\n"
            "2. Setting, location, and environment\n"
            "3. Actions and activities taking place\n"
            "4. Colors, lighting, and visual composition\n"
            "5. Any text or writing visible in the image\n"
            "6. Mood, atmosphere, and artistic style\n"
            "Please be thorough and specific in your description."
        )

    def _get_default_model(self) -> MultimodalModel:
        """Automatically select and configure a multimodal model.

        Returns:
            Configured multimodal model instance.

        Raises:
            ProcessingError: If no suitable model is available.
        """
        import os  # pylint: disable=import-outside-toplevel

        # Try OpenAI first (often more widely available)
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIVisionModel()
            except ProcessingError:
                pass

        # Try Claude as fallback
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                return ClaudeVisionModel()
            except ProcessingError:
                pass

        raise ProcessingError(
            "No multimodal model available. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "environment variable and install the required packages (openai or anthropic)."
        )

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported image file extensions.

        Returns:
            List of supported image extensions.
        """
        return [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text description from image using multimodal AI model.

        Args:
            file_path: Path to the image file to process.
            **kwargs: Additional processing parameters:
                - prompt (str): Custom analysis prompt
                - max_tokens (int): Maximum response length
                - temperature (float): Model creativity level
                - detail_level (str): 'brief', 'detailed', or 'comprehensive'

        Returns:
            Text description or analysis of the image content.

        Raises:
            ProcessingError: If image format is unsupported or analysis fails.
        """
        self.validate_file(file_path)

        # Validate image format
        if not self.can_process(file_path):
            raise ProcessingError(f"Unsupported image format: {Path(file_path).suffix}")

        prompt = kwargs.get("prompt", self.default_prompt)

        try:
            return self.model.analyze_image(file_path, prompt, **kwargs)
        except Exception as e:
            raise ProcessingError(f"Failed to process image {file_path}: {str(e)}") from e

    def process_with_custom_prompt(self, file_path: Union[str, Path], prompt: str, **kwargs: Any) -> str:
        """Process image with a custom analysis prompt.

        Args:
            file_path: Path to the image file.
            prompt: Custom prompt for specific analysis needs.
            **kwargs: Additional parameters.

        Returns:
            Analysis result based on the custom prompt.
        """
        return self.process(file_path, prompt=prompt, **kwargs)

    def extract_text_from_image(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract and transcribe text from images (OCR functionality).

        This method uses the multimodal model to identify and transcribe
        any text visible in the image, similar to OCR but with AI understanding.

        Args:
            file_path: Path to the image file.
            **kwargs: Additional parameters.

        Returns:
            Extracted text from the image, or indication if no text found.
        """
        ocr_prompt = (
            "Please extract and transcribe all text visible in this image. "
            "Maintain the original formatting and structure as much as possible. "
            "Include all readable text such as signs, documents, labels, captions, etc. "
            "If no text is visible, respond with 'No text found in image'."
        )
        return self.process(file_path, prompt=ocr_prompt, **kwargs)

    def describe_image(self, file_path: Union[str, Path], detail_level: str = "detailed", **kwargs: Any) -> str:
        """Get image description with specified level of detail.

        Args:
            file_path: Path to the image file.
            detail_level: Level of detail - 'brief', 'detailed', or 'comprehensive'.
            **kwargs: Additional parameters.

        Returns:
            Image description with requested level of detail.
        """
        prompts = {
            "brief": ("Provide a brief, one-sentence description of this image " "focusing on the main subject."),
            "detailed": self.default_prompt,
            "comprehensive": (
                "Provide an extremely comprehensive analysis of this image including:\n"
                "1. Overall scene, setting, and context\n"
                "2. Detailed inventory of all objects and their positions\n"
                "3. All people present, their appearance and activities\n"
                "4. Environmental details (lighting, weather, time of day)\n"
                "5. Colors, textures, and visual composition\n"
                "6. Any text, symbols, or signage visible\n"
                "7. Artistic style, photographic technique, or medium\n"
                "8. Emotional tone, mood, and atmosphere\n"
                "9. Potential cultural, historical, or contextual significance\n"
                "10. Technical aspects (camera angle, composition, quality)\n"
                "Be extremely thorough and include all observable details."
            ),
        }

        prompt = prompts.get(detail_level, prompts["detailed"])
        return self.process(file_path, prompt=prompt, **kwargs)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract comprehensive metadata from image files.

        This method extracts both technical image metadata and file information.
        It uses PIL/Pillow for technical details when available.

        Args:
            file_path: Path to the image file.

        Returns:
            Dictionary containing metadata with keys:
            - file_path: Original file path
            - file_size: File size in bytes
            - file_type: File extension
            - created/modified: File timestamps
            - width/height: Image dimensions
            - mode/format: Image color mode and format
            - has_transparency: Whether image has transparency
            - exif: EXIF data if available
            - pillow_error: Error message if PIL processing fails
        """
        self.validate_file(file_path)
        file_path = Path(file_path)

        stat = file_path.stat()
        metadata = {
            "file_path": str(file_path),
            "file_size": stat.st_size,
            "file_type": file_path.suffix.lower(),
            "created": str(stat.st_ctime),
            "modified": str(stat.st_mtime),
        }

        # Extract technical image metadata using PIL
        try:
            from PIL import Image  # pylint: disable=import-outside-toplevel

            with Image.open(file_path) as img:
                metadata.update(
                    {
                        "width": img.width,
                        "height": img.height,
                        "mode": img.mode,
                        "format": img.format,
                        "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                        "aspect_ratio": round(img.width / img.height, 3) if img.height > 0 else 0,
                    }
                )

                # Extract EXIF data if available
                if hasattr(img, "_getexif") and img._getexif():  # pylint: disable=protected-access
                    exif_data = img._getexif()  # pylint: disable=protected-access
                    # Filter EXIF data to include only basic types
                    filtered_exif = {
                        k: v for k, v in exif_data.items() if isinstance(v, (str, int, float)) and len(str(v)) < 100
                    }
                    if filtered_exif:
                        metadata["exif"] = filtered_exif

                # Additional image analysis
                metadata["total_pixels"] = img.width * img.height
                metadata["estimated_quality"] = self._estimate_image_quality(img)

        except ImportError:
            metadata["pillow_error"] = "PIL/Pillow not available for detailed image metadata"
        except Exception as e:  # pylint: disable=broad-exception-caught
            metadata["metadata_extraction_error"] = str(e)

        return metadata

    def _estimate_image_quality(self, img: Any) -> str:
        """Estimate image quality based on dimensions and file size.

        Args:
            img: PIL Image object.

        Returns:
            Quality estimation string.
        """
        total_pixels = img.width * img.height

        if total_pixels > 8000000:  # 8MP+
            return "very_high"
        if total_pixels > 2000000:  # 2MP+
            return "high"
        if total_pixels > 500000:  # 0.5MP+
            return "medium"
        if total_pixels > 100000:  # 0.1MP+
            return "low"
        return "very_low"


class ImageProcessorFactory:
    """Factory class for creating image processors with specific multimodal models.

    This factory provides convenient methods to create image processors
    configured with specific AI models and settings.
    """

    @staticmethod
    def create_openai_processor(api_key: Optional[str] = None, model: str = "gpt-4o", **kwargs: Any) -> ImageProcessor:
        """Create image processor with OpenAI GPT-4V model.

        Args:
            api_key: OpenAI API key.
            model: OpenAI model name.
            **kwargs: Additional processor parameters.

        Returns:
            ImageProcessor configured with OpenAI model.
        """
        openai_model = OpenAIVisionModel(api_key=api_key, model=model)
        return ImageProcessor(model=openai_model, **kwargs)

    @staticmethod
    def create_claude_processor(
        api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", **kwargs: Any
    ) -> ImageProcessor:
        """Create image processor with Claude Vision model.

        Args:
            api_key: Anthropic API key.
            model: Claude model name.
            **kwargs: Additional processor parameters.

        Returns:
            ImageProcessor configured with Claude model.
        """
        claude_model = ClaudeVisionModel(api_key=api_key, model=model)
        return ImageProcessor(model=claude_model, **kwargs)

    @staticmethod
    def create_auto_processor(**kwargs: Any) -> ImageProcessor:
        """Create image processor with automatically detected model.

        Args:
            **kwargs: Additional processor parameters.

        Returns:
            ImageProcessor with auto-detected model.
        """
        return ImageProcessor(**kwargs)
