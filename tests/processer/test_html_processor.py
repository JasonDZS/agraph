"""Tests for HTML document processor."""

from unittest.mock import patch

import pytest

from agraph.processer.base import ProcessingError
from agraph.processer.html_processor import HTMLProcessor
from tests.processer.conftest import skip_if_no_module


class TestHTMLProcessor:
    """Test HTMLProcessor functionality."""

    def test_supported_extensions(self):
        """Test that HTMLProcessor supports correct extensions."""
        processor = HTMLProcessor()
        expected_extensions = [".html", ".htm"]
        assert processor.supported_extensions == expected_extensions

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_process_html_file_default(self, sample_html_file):
        """Test processing HTML file with default settings."""
        processor = HTMLProcessor()
        result = processor.process(sample_html_file)

        # Should extract text content
        assert "Main Heading" in result
        assert "Subheading" in result
        assert "This is a paragraph" in result
        assert "bold" in result
        assert "italic" in result
        assert "List item 1" in result
        assert "List item 2" in result
        assert "External link" in result

        # Should not contain HTML tags
        assert "<h1>" not in result
        assert "<p>" not in result
        assert "<script>" not in result
        assert "<style>" not in result

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_process_html_preserve_structure(self, sample_html_file):
        """Test processing HTML with structure preservation."""
        processor = HTMLProcessor()
        result = processor.process(sample_html_file, preserve_structure=True)

        # Should have title
        assert "Title: Sample HTML" in result

        # Should have hierarchical headings
        assert "Main Heading" in result
        assert "  Subheading" in result  # Indented subsection

        # Should have list items with bullets
        assert "• List item 1" in result
        assert "• List item 2" in result

        # Should have table structure
        assert "Table:" in result
        assert "Header 1 | Header 2" in result
        assert "Cell 1 | Cell 2" in result

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_process_html_extract_links(self, sample_html_file):
        """Test processing HTML with link extraction."""
        processor = HTMLProcessor()
        result = processor.process(sample_html_file, extract_links=True)

        # Should contain main content
        assert "Main Heading" in result

        # Should contain extracted links section
        assert "Extracted Links:" in result
        assert "External link: https://example.com" in result

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_extract_structured_text(self, sample_html_file):
        """Test structured text extraction."""
        processor = HTMLProcessor()

        # Read and parse HTML
        with open(sample_html_file, "r") as f:
            html_content = f.read()

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            result = processor._extract_structured_text(soup)

            # Should have title
            assert "Title: Sample HTML" in result

            # Should have headings with proper hierarchy
            assert "Main Heading" in result
            assert "  Subheading" in result

            # Should have list items
            assert "• List item 1" in result

            # Should have table content
            assert "Table:" in result
            assert "Header 1 | Header 2" in result
        except ImportError:
            pytest.skip("beautifulsoup4 not available")

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_extract_links(self, sample_html_file):
        """Test link extraction."""
        processor = HTMLProcessor()

        with open(sample_html_file, "r") as f:
            html_content = f.read()

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, "html.parser")
            links = processor._extract_links(soup)

            assert len(links) == 1
            assert "External link: https://example.com" in links
        except ImportError:
            pytest.skip("beautifulsoup4 not available")

    def test_extract_links_various_formats(self, temp_dir):
        """Test extracting different types of links."""
        if skip_if_no_module("bs4"):
            pytest.skip("beautifulsoup4 not available")

        html_content = """
        <html>
        <body>
            <a href="https://example.com">Text link</a>
            <a href="http://test.org">HTTP link</a>
            <a href="ftp://files.com">FTP link</a>
            <a href="/relative/path">Relative link</a>
            <a href="mailto:test@example.com">Email link</a>
            <a href="javascript:void(0)">JavaScript link</a>
            <a href="https://notext.com"></a>
        </body>
        </html>
        """

        html_file = temp_dir / "links.html"
        html_file.write_text(html_content)

        processor = HTMLProcessor()

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        links = processor._extract_links(soup)

        # Should only include http/https/ftp links
        assert "Text link: https://example.com" in links
        assert "HTTP link: http://test.org" in links
        assert "FTP link: ftp://files.com" in links
        assert "https://notext.com" in links  # Link without text

        # Should not include relative, mailto, or javascript links
        relative_links = [link for link in links if "/relative/path" in link]
        assert len(relative_links) == 0

    def test_process_no_beautifulsoup(self, sample_html_file):
        """Test processing when beautifulsoup4 is not available."""
        processor = HTMLProcessor()

        with patch.dict("sys.modules", {"bs4": None}):
            with pytest.raises(ProcessingError, match="beautifulsoup4 is required"):
                processor.process(sample_html_file)

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_process_with_custom_encoding(self, temp_dir):
        """Test processing HTML file with custom encoding."""
        # Create file with latin-1 encoding
        latin_file = temp_dir / "latin.html"
        content = "<html><body><p>Café résumé naïve</p></body></html>"
        latin_file.write_bytes(content.encode("latin-1"))

        processor = HTMLProcessor()
        result = processor.process(latin_file, encoding="latin-1")
        assert "Café résumé naïve" in result

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_process_unicode_decode_error_fallback(self, temp_dir):
        """Test fallback to alternative encodings on decode error."""
        # Create file with latin-1 content
        latin_file = temp_dir / "latin.html"
        content = "<html><body><p>Café résumé naïve</p></body></html>"
        latin_file.write_bytes(content.encode("latin-1"))

        processor = HTMLProcessor()
        # Should fallback to latin-1 and succeed
        result = processor.process(latin_file)
        assert "Café résumé naïve" in result

    def test_process_unsupported_encoding(self, temp_dir):
        """Test processing binary file with fallback encoding."""
        # Create binary file - will be decoded with latin-1 fallback
        binary_file = temp_dir / "binary.html"
        binary_file.write_bytes(b"\x80\x81\x82\x83")

        processor = HTMLProcessor()
        # Should succeed with fallback encoding (latin-1 can decode any bytes)
        result = processor.process(binary_file)
        assert isinstance(result, str)
        assert len(result) == 4  # Should have 4 characters

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_extract_metadata_basic(self, sample_html_file):
        """Test extracting basic metadata from HTML file."""
        processor = HTMLProcessor()
        metadata = processor.extract_metadata(sample_html_file)

        assert metadata["file_path"] == str(sample_html_file)
        assert metadata["file_type"] == ".html"
        assert "file_size" in metadata
        assert "created" in metadata
        assert "modified" in metadata

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_extract_metadata_html_specific(self, sample_html_file):
        """Test extracting HTML-specific metadata."""
        processor = HTMLProcessor()
        metadata = processor.extract_metadata(sample_html_file)

        assert metadata["title"] == "Sample HTML"
        assert "meta_tags" in metadata
        assert metadata["meta_tags"]["description"] == "A sample HTML document"
        assert metadata["meta_tags"]["keywords"] == "sample, test, html"
        assert metadata["meta_tags"]["viewport"] == "width=device-width, initial-scale=1.0"

        # Element counts
        assert metadata["heading_count"] == 2  # h1 and h2
        assert metadata["paragraph_count"] == 1
        assert metadata["link_count"] == 1
        assert metadata["image_count"] == 0
        assert metadata["table_count"] == 1
        assert metadata["list_count"] == 1

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_extract_metadata_with_language(self, temp_dir):
        """Test extracting metadata including language attribute."""
        html_content = """<!DOCTYPE html>
        <html lang="en-US">
        <head><title>Test</title></head>
        <body><p>Content</p></body>
        </html>"""

        html_file = temp_dir / "lang.html"
        html_file.write_text(html_content)

        processor = HTMLProcessor()
        metadata = processor.extract_metadata(html_file)

        assert metadata["language"] == "en-US"

    def test_extract_metadata_no_beautifulsoup(self, sample_html_file):
        """Test metadata extraction when beautifulsoup4 is not available."""
        processor = HTMLProcessor()

        with patch.dict("sys.modules", {"bs4": None}):
            metadata = processor.extract_metadata(sample_html_file)

            assert "parsing_error" in metadata
            assert "beautifulsoup4 not available" in metadata["parsing_error"]

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_extract_metadata_parsing_error(self, temp_dir):
        """Test metadata extraction handles parsing errors."""
        # Create malformed HTML that might cause parsing issues
        bad_html = temp_dir / "bad.html"
        bad_html.write_text("<html><body><p>Unclosed tag")

        processor = HTMLProcessor()

        # Even malformed HTML should be parsed by BeautifulSoup without errors
        # But let's simulate an error
        with patch("bs4.BeautifulSoup", side_effect=Exception("Parse error")):
            metadata = processor.extract_metadata(bad_html)

            assert "parsing_error" in metadata
            assert "Parse error" in metadata["parsing_error"]

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_process_empty_html(self, temp_dir):
        """Test processing empty HTML file."""
        empty_html = temp_dir / "empty.html"
        empty_html.write_text("")

        processor = HTMLProcessor()
        with pytest.raises(ValueError, match="File is empty"):
            processor.process(empty_html)

    @pytest.mark.skipif(skip_if_no_module("bs4"), reason="beautifulsoup4 not available")
    def test_process_minimal_html(self, temp_dir):
        """Test processing minimal HTML content."""
        minimal_html = temp_dir / "minimal.html"
        minimal_html.write_text("<html><body>Just text</body></html>")

        processor = HTMLProcessor()
        result = processor.process(minimal_html)

        assert result.strip() == "Just text"
