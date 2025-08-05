from pathlib import Path
from typing import Any, Dict, List, Union, cast

import yaml  # type: ignore[import-untyped]

from .base import DocumentProcessor, ProcessingError


class TextProcessor(DocumentProcessor):
    """Processor for plain text and markdown documents."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".txt", ".md", ".markdown"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text from plain text or markdown file.

        Args:
            file_path: Path to the text file
            **kwargs: Additional parameters (encoding, strip_markdown, etc.)

        Returns:
            File content as text
        """
        self.validate_file(file_path)
        file_path = Path(file_path)

        encoding = kwargs.get("encoding", "utf-8")
        strip_markdown = kwargs.get("strip_markdown", False)

        try:
            with open(file_path, "r", encoding=encoding) as file:
                content = file.read()

            # Strip markdown formatting if requested and file is markdown
            if strip_markdown and file_path.suffix.lower() in [".md", ".markdown"]:
                content = self._strip_markdown(content)

            return content

        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=alt_encoding) as file:
                        content = file.read()
                    if strip_markdown and file_path.suffix.lower() in [".md", ".markdown"]:
                        content = self._strip_markdown(content)
                    return content
                except UnicodeDecodeError:
                    continue
            raise ProcessingError(f"Could not decode file {file_path} with any supported encoding")
        except Exception as e:
            raise ProcessingError(f"Failed to process text file {file_path}: {str(e)}")

    def _strip_markdown(self, text: str) -> str:
        """Basic markdown stripping for plain text extraction."""
        import re

        # Remove headers
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove bold and italic
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"\*([^*]+)\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)
        text = re.sub(r"_([^_]+)_", r"\1", text)

        # Remove images first (includes the !)
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove code blocks
        text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove horizontal rules
        text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)

        # Remove blockquotes
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

        # Remove list markers
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)

        # Clean up extra whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)
        text = re.sub(r"[ \t]+", " ", text)

        return text.strip()

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from text file.

        Args:
            file_path: Path to the text file

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
            # Try to detect encoding
            import chardet

            with open(file_path, "rb") as file:
                raw_data = file.read(10000)  # Read first 10KB for detection
                encoding_result = chardet.detect(raw_data)
                metadata["detected_encoding"] = encoding_result.get("encoding", "unknown")
                metadata["encoding_confidence"] = encoding_result.get("confidence", 0.0)
        except ImportError:
            metadata["detected_encoding"] = "unknown (chardet not available)"
        except Exception:
            metadata["detected_encoding"] = "detection_failed"

        try:
            # Count lines and characters
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
                metadata.update(
                    {
                        "line_count": content.count("\n") + 1,
                        "character_count": len(content),
                        "word_count": len(content.split()),
                    }
                )

                # Extract markdown frontmatter if present
                if file_path.suffix.lower() in [".md", ".markdown"]:
                    frontmatter = self._extract_frontmatter(content)
                    if frontmatter:
                        metadata["frontmatter"] = frontmatter

        except Exception as e:
            metadata["content_analysis_error"] = str(e)

        return metadata

    def _extract_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from markdown content."""
        import re

        # Match YAML frontmatter pattern
        pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            return {}

        if yaml is None:
            return {"raw_frontmatter": match.group(1)}

        try:
            result = yaml.safe_load(match.group(1))
            return cast(dict[str, Any], result) if result is not None else {}
        except Exception:
            return {"frontmatter_parse_error": "Invalid YAML"}
