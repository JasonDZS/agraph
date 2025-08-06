import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import settings
from .base import DocumentProcessor, ProcessingError


class MultimodalModel(ABC):
    """Abstract base class for multimodal models."""

    @abstractmethod
    def analyze_image(self, image_path: Union[str, Path], prompt: str, **kwargs: Any) -> str:
        """Analyze an image and return text description.

        Args:
            image_path: Path to the image file
            prompt: Text prompt for the analysis
            **kwargs: Additional model-specific parameters

        Returns:
            Text description of the image
        """
        pass

    @abstractmethod
    def analyze_image_base64(self, image_base64: str, prompt: str, **kwargs: Any) -> str:
        """Analyze an image from base64 data and return text description.

        Args:
            image_base64: Base64 encoded image data
            prompt: Text prompt for the analysis
            **kwargs: Additional model-specific parameters

        Returns:
            Text description of the image
        """
        pass


class OpenAIVisionModel(MultimodalModel):
    """OpenAI GPT-4V integration for image analysis."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """Initialize OpenAI vision model.

        Args:
            api_key: OpenAI API key (if None, will use environment variable)
            model: Model name to use
        """
        try:
            import openai
        except ImportError:
            raise ProcessingError("openai package is required for OpenAI vision. Install with: pip install openai")

        self.client = openai.OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE)
        self.model = model

    def analyze_image(self, image_path: Union[str, Path], prompt: str, **kwargs: Any) -> str:
        """Analyze image using OpenAI GPT-4V."""
        image_base64 = self._encode_image(image_path)
        return self.analyze_image_base64(image_base64, prompt, **kwargs)

    def analyze_image_base64(self, image_base64: str, prompt: str, **kwargs: Any) -> str:
        """Analyze image from base64 using OpenAI GPT-4V."""
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
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        ],
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content if content is not None else ""
        except Exception as e:
            raise ProcessingError(f"OpenAI vision analysis failed: {str(e)}")

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")


class ClaudeVisionModel(MultimodalModel):
    """Anthropic Claude Vision integration for image analysis."""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022"):
        """Initialize Claude vision model.

        Args:
            api_key: Anthropic API key (if None, will use environment variable)
            model: Model name to use
        """
        try:
            import anthropic
        except ImportError:
            raise ProcessingError(
                "anthropic package is required for Claude vision. Install with: pip install anthropic"
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def analyze_image(self, image_path: Union[str, Path], prompt: str, **kwargs: Any) -> str:
        """Analyze image using Claude Vision."""
        image_base64 = self._encode_image(image_path)
        return self.analyze_image_base64(image_base64, prompt, **kwargs)

    def analyze_image_base64(self, image_base64: str, prompt: str, **kwargs: Any) -> str:
        """Analyze image from base64 using Claude Vision."""
        max_tokens = kwargs.get("max_tokens", 1000)
        temperature = kwargs.get("temperature", 0.1)

        # Detect image format
        image_format = self._detect_image_format(image_base64)

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
            text = response.content[0].text
            return text if text is not None else ""
        except Exception as e:
            raise ProcessingError(f"Claude vision analysis failed: {str(e)}")

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def _detect_image_format(self, image_base64: str) -> str:
        """Detect image format from base64 data."""
        image_data = base64.b64decode(image_base64)

        # Check magic numbers
        if image_data.startswith(b"\x89PNG"):
            return "png"
        elif image_data.startswith(b"\xff\xd8\xff"):
            return "jpeg"
        elif image_data.startswith(b"GIF8"):
            return "gif"
        elif image_data.startswith(b"RIFF") and b"WEBP" in image_data[:12]:
            return "webp"
        else:
            return "jpeg"  # Default fallback


class ImageProcessor(DocumentProcessor):
    """Processor for image files using multimodal models."""

    def __init__(self, model: Optional[MultimodalModel] = None, default_prompt: Optional[str] = None):
        """Initialize image processor.

        Args:
            model: Multimodal model to use (if None, will try to auto-detect)
            default_prompt: Default prompt for image analysis
        """
        self.model = model or self._get_default_model()
        self.default_prompt = default_prompt or (
            "Please describe this image in detail. Include information about objects, people, "
            "text, colors, composition, and any other relevant visual elements. "
            "If there is text in the image, please transcribe it."
        )

    def _get_default_model(self) -> MultimodalModel:
        """Get default multimodal model based on available API keys."""
        import os

        # Try OpenAI first
        if os.getenv("OPENAI_API_KEY"):
            try:
                return OpenAIVisionModel()
            except ProcessingError:
                pass

        # Try Claude
        if os.getenv("ANTHROPIC_API_KEY"):
            try:
                return ClaudeVisionModel()
            except ProcessingError:
                pass

        raise ProcessingError(
            "No multimodal model available. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY "
            "environment variable and install the required packages."
        )

    @property
    def supported_extensions(self) -> List[str]:
        return [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text from image using multimodal model.

        Args:
            file_path: Path to the image file
            **kwargs: Additional parameters (prompt, max_tokens, temperature, etc.)

        Returns:
            Text description of the image
        """
        self.validate_file(file_path)

        # Validate image format
        if not self.can_process(file_path):
            raise ProcessingError(f"Unsupported image format: {Path(file_path).suffix}")

        prompt = kwargs.get("prompt", self.default_prompt)

        try:
            return self.model.analyze_image(file_path, prompt, **kwargs)
        except Exception as e:
            raise ProcessingError(f"Failed to process image {file_path}: {str(e)}")

    def process_with_custom_prompt(self, file_path: Union[str, Path], prompt: str, **kwargs: Any) -> str:
        """Process image with custom prompt.

        Args:
            file_path: Path to the image file
            prompt: Custom prompt for analysis
            **kwargs: Additional parameters

        Returns:
            Text description based on the prompt
        """
        return self.process(file_path, prompt=prompt, **kwargs)

    def extract_text_from_image(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text from image (OCR-like functionality).

        Args:
            file_path: Path to the image file
            **kwargs: Additional parameters

        Returns:
            Extracted text from the image
        """
        ocr_prompt = (
            "Extract and transcribe all text visible in this image. "
            "Maintain the original formatting and structure as much as possible. "
            "If no text is visible, respond with 'No text found in image'."
        )
        return self.process(file_path, prompt=ocr_prompt, **kwargs)

    def describe_image(self, file_path: Union[str, Path], detail_level: str = "detailed", **kwargs: Any) -> str:
        """Get image description with specified detail level.

        Args:
            file_path: Path to the image file
            detail_level: "brief", "detailed", or "comprehensive"
            **kwargs: Additional parameters

        Returns:
            Image description
        """
        prompts = {
            "brief": "Provide a brief, one-sentence description of this image.",
            "detailed": self.default_prompt,
            "comprehensive": (
                "Provide a comprehensive analysis of this image including: "
                "1. Overall scene and setting, "
                "2. All objects and their positions, "
                "3. People and their activities, "
                "4. Colors, lighting, and mood, "
                "5. Any text or writing visible, "
                "6. Artistic style or photographic qualities, "
                "7. Potential context or story."
            ),
        }

        prompt = prompts.get(detail_level, prompts["detailed"])
        return self.process(file_path, prompt=prompt, **kwargs)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from image file.

        Args:
            file_path: Path to the image file

        Returns:
            Dictionary containing metadata
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

        try:
            from PIL import Image

            with Image.open(file_path) as img:
                metadata.update(
                    {
                        "width": img.width,
                        "height": img.height,
                        "mode": img.mode,
                        "format": img.format,
                        "has_transparency": img.mode in ("RGBA", "LA") or "transparency" in img.info,
                    }
                )

                # Extract EXIF data if available
                if hasattr(img, "_getexif") and img._getexif():
                    exif = img._getexif()
                    metadata["exif"] = {k: v for k, v in exif.items() if isinstance(v, (str, int, float))}

        except ImportError:
            metadata["pillow_error"] = "PIL/Pillow not available for detailed image metadata"
        except Exception as e:
            metadata["metadata_extraction_error"] = str(e)

        return metadata


class ImageProcessorFactory:
    """Factory for creating image processors with different models."""

    @staticmethod
    def create_openai_processor(api_key: Optional[str] = None, model: str = "gpt-4o", **kwargs: Any) -> ImageProcessor:
        """Create image processor with OpenAI model."""
        openai_model = OpenAIVisionModel(api_key=api_key, model=model)
        return ImageProcessor(model=openai_model, **kwargs)

    @staticmethod
    def create_claude_processor(
        api_key: Optional[str] = None, model: str = "claude-3-5-sonnet-20241022", **kwargs: Any
    ) -> ImageProcessor:
        """Create image processor with Claude model."""
        claude_model = ClaudeVisionModel(api_key=api_key, model=model)
        return ImageProcessor(model=claude_model, **kwargs)

    @staticmethod
    def create_auto_processor(**kwargs: Any) -> ImageProcessor:
        """Create image processor with auto-detected model."""
        return ImageProcessor(**kwargs)
