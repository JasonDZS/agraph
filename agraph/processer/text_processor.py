"""Text and Markdown document processor implementation.

This module provides functionality for processing plain text files and Markdown documents.
It includes support for various text encodings, Markdown formatting removal, frontmatter
extraction, and comprehensive text analysis.
"""

from pathlib import Path
from typing import Any, Dict, List, Union, cast

import yaml

from .base import DocumentProcessor, ProcessingError


class TextProcessor(DocumentProcessor):
    """Document processor for plain text and Markdown files.

    This processor handles text-based documents including plain text files
    and Markdown documents. It provides features such as:
    - Multiple encoding detection and fallback
    - Markdown formatting removal
    - YAML frontmatter extraction
    - Text statistics and analysis
    - Content validation and cleaning

    Supported file types:
    - Plain text files (.txt)
    - Markdown files (.md, .markdown)

    Dependencies:
        yaml: For YAML frontmatter parsing in Markdown files.
        chardet: Optional, for encoding detection.
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported text file extensions.

        Returns:
            List containing '.txt', '.md', and '.markdown' extensions.
        """
        return [".txt", ".md", ".markdown"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract and optionally clean text content from text or Markdown files.

        This method reads text files with automatic encoding detection and fallback.
        For Markdown files, it can optionally strip formatting to produce clean text.

        Args:
            file_path: Path to the text file to process.
            **kwargs: Additional processing parameters:
                - encoding (str): Preferred text encoding (default: 'utf-8')
                - strip_markdown (bool): Whether to remove Markdown formatting
                  for .md/.markdown files (default: False)

        Returns:
            File content as text, optionally with Markdown formatting removed.

        Raises:
            ProcessingError: If the file cannot be decoded with any supported
                           encoding or other processing errors occur.
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
                content = self._strip_markdown_formatting(content)

            return content

        except UnicodeDecodeError:
            # Try alternative encodings with fallback
            for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=alt_encoding) as file:
                        content = file.read()
                    if strip_markdown and file_path.suffix.lower() in [".md", ".markdown"]:
                        content = self._strip_markdown_formatting(content)
                    return content
                except UnicodeDecodeError:
                    continue
            raise ProcessingError(f"Could not decode file {file_path} with any supported encoding")
        except Exception as e:
            raise ProcessingError(f"Failed to process text file {file_path}: {str(e)}")

    def _strip_markdown_formatting(self, text: str) -> str:
        """Remove Markdown formatting to produce clean plain text.

        This method removes common Markdown formatting elements while preserving
        the actual text content. It handles headers, emphasis, links, images,
        code blocks, lists, and other formatting elements.

        Args:
            text: Markdown-formatted text to clean.

        Returns:
            Plain text with Markdown formatting removed.
        """
        import re

        # Remove YAML frontmatter first
        text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)

        # Remove headers (# ## ### etc.)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Remove bold and italic formatting
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)  # **bold**
        text = re.sub(r"\*([^*]+)\*", r"\1", text)  # *italic*
        text = re.sub(r"__([^_]+)__", r"\1", text)  # __bold__
        text = re.sub(r"_([^_]+)_", r"\1", text)  # _italic_

        # Remove strikethrough
        text = re.sub(r"~~([^~]+)~~", r"\1", text)

        # Remove images first (includes the !)
        text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)

        # Remove links but keep text
        text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

        # Remove code blocks (triple backticks)
        text = re.sub(r"```[^`]*```", "", text, flags=re.DOTALL)

        # Remove inline code
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Remove horizontal rules
        text = re.sub(r"^---+$", "", text, flags=re.MULTILINE)
        text = re.sub(r"^\*\*\*+$", "", text, flags=re.MULTILINE)

        # Remove blockquotes
        text = re.sub(r"^>\s*", "", text, flags=re.MULTILINE)

        # Remove list markers
        text = re.sub(r"^[\s]*[-*+]\s+", "", text, flags=re.MULTILINE)  # Unordered lists
        text = re.sub(r"^[\s]*\d+\.\s+", "", text, flags=re.MULTILINE)  # Ordered lists

        # Remove table formatting
        text = re.sub(r"\|", " ", text)  # Remove table separators
        text = re.sub(r"^[\s]*:?-+:?[\s]*$", "", text, flags=re.MULTILINE)  # Table separators

        # Clean up extra whitespace
        text = re.sub(r"\n\s*\n", "\n\n", text)  # Multiple newlines to double
        text = re.sub(r"[ \t]+", " ", text)  # Multiple spaces to single
        text = re.sub(r"^\s+", "", text, flags=re.MULTILINE)  # Leading whitespace

        return text.strip()

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract comprehensive metadata from text files.

        This method extracts file system metadata, content statistics, encoding
        information, and for Markdown files, YAML frontmatter if present.

        Args:
            file_path: Path to the text file.

        Returns:
            Dictionary containing metadata with keys:
            - file_path: Original file path
            - file_size: File size in bytes
            - file_type: File extension
            - created/modified: File timestamps
            - detected_encoding: Auto-detected text encoding
            - encoding_confidence: Confidence score for encoding detection
            - line_count, character_count, word_count: Content statistics
            - frontmatter: YAML frontmatter for Markdown files
            - content_analysis_error: Error message if analysis fails
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

        # Try to detect encoding using chardet if available
        try:
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
            # Count lines, characters, and words
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

                # Basic content statistics
                lines = content.split("\n")
                metadata.update(
                    {
                        "line_count": len(lines),
                        "character_count": len(content),
                        "word_count": len(content.split()),
                        "non_empty_line_count": len([line for line in lines if line.strip()]),
                    }
                )

                # Extract markdown frontmatter if present
                if file_path.suffix.lower() in [".md", ".markdown"]:
                    frontmatter = self._extract_yaml_frontmatter(content)
                    if frontmatter:
                        metadata["frontmatter"] = frontmatter

                # Additional text analysis
                metadata.update(self._analyze_text_content(content))

        except Exception as e:
            metadata["content_analysis_error"] = str(e)

        return metadata

    def _extract_yaml_frontmatter(self, content: str) -> Dict[str, Any]:
        """Extract YAML frontmatter from Markdown content.

        YAML frontmatter is metadata at the beginning of Markdown files,
        delimited by '---' lines.

        Args:
            content: Full Markdown content.

        Returns:
            Dictionary containing parsed frontmatter, or empty dict if none found.
        """
        import re

        # Match YAML frontmatter pattern (must be at the start of the file)
        pattern = r"^---\s*\n(.*?)\n---\s*\n"
        match = re.match(pattern, content, re.DOTALL)

        if not match:
            return {}

        frontmatter_text = match.group(1)

        try:
            result = yaml.safe_load(frontmatter_text)
            return cast(dict[str, Any], result) if result is not None else {}
        except Exception as e:
            return {"frontmatter_parse_error": f"Invalid YAML: {str(e)}"}

    def _analyze_text_content(self, content: str) -> Dict[str, Any]:
        """Perform additional analysis on text content.

        Args:
            content: Text content to analyze.

        Returns:
            Dictionary with analysis results.
        """
        analysis = {}

        # Count sentences (basic approximation)
        import re

        sentences = re.split(r"[.!?]+", content)
        analysis["sentence_count"] = len([s for s in sentences if s.strip()])

        # Count paragraphs (double newlines)
        paragraphs = content.split("\n\n")
        analysis["paragraph_count"] = len([p for p in paragraphs if p.strip()])

        # Detect if content is primarily ASCII
        try:
            content.encode("ascii")
            analysis["is_ascii"] = True
        except UnicodeEncodeError:
            analysis["is_ascii"] = False

        # Calculate average word length
        words = content.split()
        if words:
            analysis["average_word_length"] = int(sum(len(word) for word in words) / len(words))
        else:
            analysis["average_word_length"] = 0

        return analysis
