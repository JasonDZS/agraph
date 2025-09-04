"""
Test text processor functionality.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agraph.processor.base import DependencyError, ProcessingError
from agraph.processor.text_processor import TextProcessor


class TestTextProcessor(unittest.TestCase):
    """Test text document processor."""

    def setUp(self):
        """Set up test environment."""
        self.processor = TextProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.processor.supported_extensions
        expected = [".txt", ".md", ".markdown"]
        self.assertEqual(extensions, expected)

    def test_can_process_text_files(self):
        """Test text file type detection."""
        self.assertTrue(self.processor.can_process("document.txt"))
        self.assertTrue(self.processor.can_process("README.md"))
        self.assertTrue(self.processor.can_process("notes.markdown"))
        self.assertFalse(self.processor.can_process("document.pdf"))

    def test_processor_initialization_with_yaml(self):
        """Test processor initialization with yaml dependency."""
        processor = TextProcessor()
        self.assertIsNotNone(processor)

    def test_processor_initialization_without_yaml(self):
        """Test processor initialization fails without yaml."""
        with patch.dict("sys.modules", {"yaml": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with self.assertRaises(DependencyError):
                    TextProcessor()

    def test_process_simple_text_file(self):
        """Test processing a simple text file."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        content = "This is a test file.\nWith multiple lines."
        test_file.write_text(content, encoding="utf-8")

        result = self.processor.process(test_file)

        self.assertEqual(result, content)

    def test_process_with_custom_encoding(self):
        """Test processing with custom encoding."""
        # Create test file with specific encoding
        test_file = Path(self.temp_dir) / "encoded.txt"
        content = "Text with special chars: café, naïve"
        test_file.write_text(content, encoding="latin-1")

        result = self.processor.process(test_file, encoding="latin-1")

        self.assertEqual(result, content)

    def test_process_encoding_fallback(self):
        """Test automatic encoding fallback for problematic files."""
        # Create file with latin-1 encoding
        test_file = Path(self.temp_dir) / "latin1.txt"
        content = "Café naïve résumé"
        test_file.write_text(content, encoding="latin-1")

        # Process with utf-8 (should fallback to latin-1)
        result = self.processor.process(test_file, encoding="utf-8")

        self.assertEqual(result, content)

    def test_process_markdown_without_stripping(self):
        """Test processing Markdown without formatting removal."""
        test_file = Path(self.temp_dir) / "test.md"
        content = "# Header\n\n**Bold text** and *italic text*"
        test_file.write_text(content)

        result = self.processor.process(test_file)

        self.assertEqual(result, content)

    def test_process_markdown_with_stripping(self):
        """Test processing Markdown with formatting removal."""
        test_file = Path(self.temp_dir) / "test.md"
        content = "# Header\n\n**Bold text** and *italic text*"
        test_file.write_text(content)

        result = self.processor.process(test_file, strip_markdown=True)

        expected = "Header\nBold text and italic text"
        self.assertEqual(result, expected)

    def test_process_markdown_with_frontmatter(self):
        """Test processing Markdown with YAML frontmatter."""
        test_file = Path(self.temp_dir) / "frontmatter.md"
        content = """---
title: Test Document
author: Test Author
---

# Main Content

This is the main content."""
        test_file.write_text(content)

        result = self.processor.process(test_file, strip_markdown=True)

        # Frontmatter should be removed
        self.assertNotIn("title:", result)
        self.assertIn("Main Content", result)

    def test_strip_markdown_formatting(self):
        """Test Markdown formatting removal."""
        markdown_text = """# Header 1

## Header 2

**Bold text** and *italic text*

[Link text](http://example.com)

![Image](image.png)

`inline code`

```
code block
```

- List item 1
- List item 2

1. Numbered item
2. Another item

> Blockquote text

---

Table | Data
------|------
Cell  | Value
"""

        result = self.processor._strip_markdown_formatting(markdown_text)

        # Check that formatting is removed
        self.assertNotIn("#", result)
        self.assertNotIn("**", result)
        self.assertNotIn("*", result)
        self.assertNotIn("[", result)
        self.assertNotIn("]", result)
        self.assertNotIn("`", result)
        self.assertNotIn(">", result)
        # Note: Some dashes may remain in table content, focus on list markers

        # Check that content is preserved
        self.assertIn("Header 1", result)
        self.assertIn("Bold text", result)
        self.assertIn("Link text", result)

    def test_extract_metadata_basic(self):
        """Test basic metadata extraction."""
        test_file = Path(self.temp_dir) / "test.txt"
        content = "This is a test file.\nWith two lines."
        test_file.write_text(content)

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["file_path"], str(test_file))
        self.assertGreater(metadata["file_size"], 0)
        self.assertEqual(metadata["file_type"], ".txt")
        self.assertEqual(metadata["line_count"], 2)
        self.assertEqual(metadata["word_count"], 8)  # "This is a test file With two lines"
        self.assertGreater(metadata["character_count"], 0)

    def test_extract_metadata_with_encoding_detection(self):
        """Test metadata extraction with encoding detection."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Simple ASCII content")

        with patch("chardet.detect") as mock_detect:
            mock_detect.return_value = {"encoding": "utf-8", "confidence": 0.99}

            metadata = self.processor.extract_metadata(test_file)

            self.assertEqual(metadata["detected_encoding"], "utf-8")
            self.assertEqual(metadata["encoding_confidence"], 0.99)

    def test_extract_metadata_without_chardet(self):
        """Test metadata extraction when chardet is not available."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content")

        with patch.dict("sys.modules", {"chardet": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                metadata = self.processor.extract_metadata(test_file)

                self.assertIn("unknown", metadata["detected_encoding"])

    def test_extract_metadata_markdown_frontmatter(self):
        """Test metadata extraction from Markdown with frontmatter."""
        test_file = Path(self.temp_dir) / "with_frontmatter.md"
        content = """---
title: Test Document
author: Test Author
tags: [test, markdown]
published: true
---

# Main Content

This is the main content of the document."""
        test_file.write_text(content)

        metadata = self.processor.extract_metadata(test_file)

        self.assertIn("frontmatter", metadata)
        frontmatter = metadata["frontmatter"]
        self.assertEqual(frontmatter["title"], "Test Document")
        self.assertEqual(frontmatter["author"], "Test Author")
        self.assertEqual(frontmatter["tags"], ["test", "markdown"])

    def test_extract_metadata_invalid_frontmatter(self):
        """Test handling of invalid YAML frontmatter."""
        test_file = Path(self.temp_dir) / "invalid_frontmatter.md"
        content = """---
title: Test Document
invalid: yaml: syntax: error
---

Content here."""
        test_file.write_text(content)

        metadata = self.processor.extract_metadata(test_file)

        self.assertIn("frontmatter", metadata)
        self.assertIn("frontmatter_parse_error", metadata["frontmatter"])

    def test_text_analysis_features(self):
        """Test text content analysis features."""
        test_file = Path(self.temp_dir) / "analysis.txt"
        content = """This is a test document.

It has multiple paragraphs! Each paragraph contains sentences?
This helps with analysis.

Final paragraph here."""
        test_file.write_text(content)

        metadata = self.processor.extract_metadata(test_file)

        # Check analysis features
        self.assertGreater(metadata["sentence_count"], 3)
        self.assertEqual(metadata["paragraph_count"], 3)
        self.assertTrue(metadata["is_ascii"])
        self.assertGreater(metadata["average_word_length"], 0)
        self.assertGreater(metadata["non_empty_line_count"], 0)

    def test_process_empty_file_raises_error(self):
        """Test that processing empty file raises appropriate error."""
        empty_file = Path(self.temp_dir) / "empty.txt"
        empty_file.touch()

        with self.assertRaises(ValueError) as context:
            self.processor.process(empty_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_nonexistent_file_raises_error(self):
        """Test that processing non-existent file raises appropriate error."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.txt"

        with self.assertRaises(FileNotFoundError):
            self.processor.process(nonexistent_file)

    def test_metadata_extraction_with_content_error(self):
        """Test metadata extraction continues when content analysis fails."""
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("Test content")

        # Mock _analyze_text_content to raise exception
        with patch.object(
            self.processor, "_analyze_text_content", side_effect=Exception("Analysis error")
        ):
            metadata = self.processor.extract_metadata(test_file)

            # Should still have basic metadata
            self.assertIn("file_path", metadata)
            self.assertIn("content_analysis_error", metadata)

    def test_unsupported_encoding_fallback(self):
        """Test processing file that can't be decoded with any encoding."""
        # Create binary file that looks like text
        test_file = Path(self.temp_dir) / "binary.txt"
        test_file.write_bytes(b"\x80\x81\x82\x83\x84\x85")  # Invalid UTF-8

        # Mock all encoding attempts to fail
        with patch(
            "builtins.open", side_effect=UnicodeDecodeError("utf-8", b"", 0, 1, "mock error")
        ):
            with self.assertRaises(ProcessingError) as context:
                self.processor.process(test_file)

            self.assertIn("decode", str(context.exception).lower())

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
