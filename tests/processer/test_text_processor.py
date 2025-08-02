"""Tests for text document processor."""

from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from agraph.processer.base import ProcessingError
from agraph.processer.text_processor import TextProcessor


class TestTextProcessor:
    """Test TextProcessor functionality."""

    def test_supported_extensions(self):
        """Test that TextProcessor supports correct extensions."""
        processor = TextProcessor()
        expected_extensions = [".txt", ".md", ".markdown"]
        assert processor.supported_extensions == expected_extensions

    def test_process_text_file(self, sample_text_file):
        """Test processing a plain text file."""
        processor = TextProcessor()
        result = processor.process(sample_text_file)

        expected_content = "This is a sample text file.\nWith multiple lines.\nFor testing purposes."
        assert result == expected_content

    def test_process_markdown_file(self, sample_markdown_file):
        """Test processing a markdown file without stripping."""
        processor = TextProcessor()
        result = processor.process(sample_markdown_file)

        # Should contain markdown formatting
        assert "# Main Title" in result
        assert "**bold**" in result
        assert "*italic*" in result
        assert "[Link text](https://example.com)" in result

    def test_process_markdown_strip_formatting(self, sample_markdown_file):
        """Test processing markdown file with formatting stripped."""
        processor = TextProcessor()
        result = processor.process(sample_markdown_file, strip_markdown=True)

        # Should not contain markdown formatting
        assert "# Main Title" not in result
        assert "**bold**" not in result
        assert "*italic*" not in result
        assert "Main Title" in result
        assert "bold" in result
        assert "italic" in result

    def test_process_with_custom_encoding(self, temp_dir):
        """Test processing file with custom encoding."""
        # Create file with latin-1 encoding
        latin_file = temp_dir / "latin.txt"
        content = "Café résumé naïve"
        latin_file.write_bytes(content.encode("latin-1"))

        processor = TextProcessor()
        result = processor.process(latin_file, encoding="latin-1")
        assert result == content

    def test_process_unicode_decode_error_fallback(self, temp_dir):
        """Test fallback to alternative encodings on decode error."""
        # Create file with latin-1 content but try to read as utf-8
        latin_file = temp_dir / "latin.txt"
        content = "Café résumé naïve"
        latin_file.write_bytes(content.encode("latin-1"))

        processor = TextProcessor()
        # Should fallback to latin-1 and succeed
        result = processor.process(latin_file)
        assert result == content

    def test_process_unsupported_encoding(self, temp_dir):
        """Test processing binary file with fallback encoding."""
        binary_file = temp_dir / "binary.txt"
        binary_file.write_bytes(b"\x80\x81\x82\x83")

        processor = TextProcessor()
        # Should succeed with fallback encoding (latin-1 can decode any bytes)
        result = processor.process(binary_file)
        assert isinstance(result, str)
        assert len(result) == 4  # Should have 4 characters

    def test_strip_markdown_headers(self):
        """Test stripping markdown headers."""
        processor = TextProcessor()
        text = "# Header 1\n## Header 2\n### Header 3\nRegular text"
        result = processor._strip_markdown(text)

        assert "Header 1" in result
        assert "Header 2" in result
        assert "Header 3" in result
        assert "Regular text" in result
        assert "# Header" not in result
        assert "## Header" not in result

    def test_strip_markdown_emphasis(self):
        """Test stripping markdown emphasis."""
        processor = TextProcessor()
        text = "**bold** and *italic* and __underline__ and _emphasis_"
        result = processor._strip_markdown(text)

        assert result == "bold and italic and underline and emphasis"

    def test_strip_markdown_links(self):
        """Test stripping markdown links."""
        processor = TextProcessor()
        text = "[Link text](https://example.com) and ![Image alt](image.jpg)"
        result = processor._strip_markdown(text)

        # Based on the actual regex patterns:
        # Links: \[([^\]]+)\]\([^)]+\) -> \1 (keeps link text)
        # Images: !\[([^\]]*)\]\([^)]+\) -> \1 (keeps alt text)
        assert result == "Link text and Image alt"

    def test_strip_markdown_code(self):
        """Test stripping markdown code blocks and inline code."""
        processor = TextProcessor()
        text = "```python\nprint('hello')\n```\nSome `inline code` here"
        result = processor._strip_markdown(text)

        assert "print('hello')" not in result
        assert "inline code" in result
        assert "`" not in result

    def test_strip_markdown_lists(self):
        """Test stripping markdown list markers."""
        processor = TextProcessor()
        text = "- Item 1\n* Item 2\n+ Item 3\n1. Numbered 1\n2. Numbered 2"
        result = processor._strip_markdown(text)

        assert "Item 1" in result
        assert "Item 2" in result
        assert "Numbered 1" in result
        assert "-" not in result
        assert "*" not in result
        assert "1." not in result

    def test_extract_metadata_basic(self, sample_text_file):
        """Test extracting basic metadata from text file."""
        processor = TextProcessor()
        metadata = processor.extract_metadata(sample_text_file)

        assert metadata["file_path"] == str(sample_text_file)
        assert metadata["file_type"] == ".txt"
        assert "file_size" in metadata
        assert "created" in metadata
        assert "modified" in metadata
        assert "line_count" in metadata
        assert "character_count" in metadata
        assert "word_count" in metadata

    def test_extract_metadata_encoding_detection(self, sample_text_file):
        """Test encoding detection in metadata extraction."""
        processor = TextProcessor()

        # Mock chardet module to be available
        import types
        mock_chardet = types.ModuleType('chardet')
        mock_chardet.detect = lambda x: {"encoding": "utf-8", "confidence": 0.99}

        with patch.dict("sys.modules", {"chardet": mock_chardet}):
            metadata = processor.extract_metadata(sample_text_file)

            assert metadata["detected_encoding"] == "utf-8"
            assert metadata["encoding_confidence"] == 0.99

    def test_extract_metadata_no_chardet(self, sample_text_file):
        """Test metadata extraction when chardet is not available."""
        processor = TextProcessor()

        with patch.dict("sys.modules", {"chardet": None}):
            metadata = processor.extract_metadata(sample_text_file)

            assert "chardet not available" in metadata["detected_encoding"]

    def test_extract_metadata_frontmatter(self, sample_markdown_file):
        """Test extracting YAML frontmatter from markdown."""
        processor = TextProcessor()
        metadata = processor.extract_metadata(sample_markdown_file)

        assert "frontmatter" in metadata
        frontmatter = metadata["frontmatter"]
        assert frontmatter["title"] == "Sample Document"
        assert frontmatter["author"] == "Test Author"

    def test_extract_frontmatter_no_yaml(self, temp_dir):
        """Test frontmatter extraction when yaml is not available."""
        processor = TextProcessor()
        md_file = temp_dir / "test.md"
        md_file.write_text("---\ntitle: Test\n---\nContent")

        with patch.dict("sys.modules", {"yaml": None}):
            frontmatter = processor._extract_frontmatter(md_file.read_text())

            assert "title" in frontmatter

    def test_extract_frontmatter_invalid_yaml(self, temp_dir):
        """Test frontmatter extraction with invalid YAML."""
        processor = TextProcessor()
        content = "---\ninvalid: yaml: content\n---\nContent"

        with patch("yaml.safe_load", side_effect=Exception("Invalid YAML")):
            frontmatter = processor._extract_frontmatter(content)

            assert "frontmatter_parse_error" in frontmatter

    def test_extract_frontmatter_no_frontmatter(self):
        """Test frontmatter extraction when no frontmatter exists."""
        processor = TextProcessor()
        content = "# Just a regular markdown file\nWith no frontmatter"

        frontmatter = processor._extract_frontmatter(content)
        assert frontmatter == {}

    def test_process_file_not_found(self, nonexistent_file):
        """Test processing non-existent file raises error."""
        processor = TextProcessor()

        with pytest.raises(FileNotFoundError):
            processor.process(nonexistent_file)

    def test_process_empty_file(self, empty_file):
        """Test processing empty file raises error."""
        processor = TextProcessor()

        with pytest.raises(ValueError, match="File is empty"):
            processor.process(empty_file)

    def test_process_with_pathlib_path(self, sample_text_file):
        """Test processing with pathlib.Path object."""
        processor = TextProcessor()
        result = processor.process(Path(sample_text_file))

        expected_content = "This is a sample text file.\nWith multiple lines.\nFor testing purposes."
        assert result == expected_content

    def test_extract_metadata_content_analysis_error(self, temp_dir):
        """Test metadata extraction handles content analysis errors."""
        processor = TextProcessor()

        # Create a file that will cause an error during analysis
        bad_file = temp_dir / "bad.txt"
        bad_file.write_text("content")

        # Mock open to raise an exception
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            metadata = processor.extract_metadata(bad_file)

            assert "content_analysis_error" in metadata
