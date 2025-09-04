"""
Test HTML processor functionality.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agraph.processor.base import ProcessingError
from agraph.processor.html_processor import HTMLProcessor


class TestHTMLProcessor(unittest.TestCase):
    """Test HTML document processor."""

    def setUp(self):
        """Set up test environment."""
        self.processor = HTMLProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.processor.supported_extensions
        expected = [".html", ".htm"]
        self.assertEqual(extensions, expected)

    def test_can_process_html_files(self):
        """Test HTML file type detection."""
        self.assertTrue(self.processor.can_process("page.html"))
        self.assertTrue(self.processor.can_process("index.htm"))
        self.assertFalse(self.processor.can_process("document.txt"))

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_process_simple_html(self, mock_bs):
        """Test processing simple HTML content."""
        test_file = Path(self.temp_dir) / "simple.html"
        html_content = "<html><body><p>Hello world</p></body></html>"
        test_file.write_text(html_content)

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Hello world"
        mock_bs.return_value = mock_soup

        result = self.processor.process(test_file)

        self.assertEqual(result, "Hello world")

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_process_with_structure_preservation(self, mock_bs):
        """Test processing HTML with structure preservation."""
        test_file = Path(self.temp_dir) / "structured.html"
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Main Title</h1>
            <p>First paragraph</p>
            <h2>Subtitle</h2>
            <p>Second paragraph</p>
        </body>
        </html>
        """
        test_file.write_text(html_content)

        # Mock BeautifulSoup and its methods
        mock_soup = MagicMock()
        mock_title = MagicMock()
        mock_title.get_text.return_value = "Test Page"
        mock_soup.find.return_value = mock_title

        mock_h1 = MagicMock()
        mock_h1.name = "h1"
        mock_h1.get_text.return_value = "Main Title"
        mock_h2 = MagicMock()
        mock_h2.name = "h2"
        mock_h2.get_text.return_value = "Subtitle"
        mock_soup.find_all.return_value = [mock_h1, mock_h2]

        mock_bs.return_value = mock_soup

        with patch.object(
            self.processor, "_extract_structured_text", return_value="Structured content"
        ):
            result = self.processor.process(test_file, preserve_structure=True)

            self.assertEqual(result, "Structured content")

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_process_with_link_extraction(self, mock_bs):
        """Test processing HTML with link extraction."""
        test_file = Path(self.temp_dir) / "links.html"
        html_content = (
            '<html><body><p>Content</p><a href="http://example.com">Link</a></body></html>'
        )
        test_file.write_text(html_content)

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Content Link"
        mock_bs.return_value = mock_soup

        with patch.object(
            self.processor, "_extract_links", return_value=["Link: http://example.com"]
        ):
            result = self.processor.process(test_file, extract_links=True)

            self.assertIn("Content Link", result)
            self.assertIn("Link: http://example.com", result)

    def test_process_without_beautifulsoup(self):
        """Test processing when beautifulsoup4 is not available."""
        test_file = Path(self.temp_dir) / "test.html"
        test_file.write_text("<html><body>Test</body></html>")

        with patch.dict("sys.modules", {"bs4": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with self.assertRaises(ProcessingError) as context:
                    self.processor.process(test_file)

                self.assertIn("beautifulsoup4", str(context.exception))

    def test_read_html_with_encoding_fallback(self):
        """Test HTML reading with encoding fallback."""
        test_file = Path(self.temp_dir) / "encoded.html"
        content = "<html><body>café naïve</body></html>"
        test_file.write_text(content, encoding="latin-1")

        # Should fallback to latin-1 when utf-8 fails
        result = self.processor._read_html_with_encoding_fallback(test_file, "utf-8")

        self.assertIn("café", result)

    def test_read_html_unsupported_encoding(self):
        """Test error handling for unsupported encoding."""
        test_file = Path(self.temp_dir) / "binary.html"
        test_file.write_bytes(b"\x80\x81\x82\x83<html></html>")

        with self.assertRaises(ProcessingError) as context:
            self.processor._read_html_with_encoding_fallback(test_file, "utf-8")

        self.assertIn("decode", str(context.exception).lower())

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_remove_non_content_elements(self, mock_bs):
        """Test removal of non-content elements."""
        mock_soup = MagicMock()
        mock_elements = [MagicMock() for _ in range(3)]
        mock_soup.return_value = mock_elements

        self.processor._remove_non_content_elements(mock_soup)

        # Should call decompose on all found elements
        for element in mock_elements:
            element.decompose.assert_called_once()

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_extract_plain_text(self, mock_bs):
        """Test plain text extraction."""
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "  Line 1  \n\n  Line 2  \n\n\n  "

        result = self.processor._extract_plain_text(mock_soup)

        self.assertEqual(result, "Line 1\nLine 2")

    def test_extract_links(self):
        """Test link extraction from HTML."""
        from bs4 import BeautifulSoup

        html = """
        <html>
        <body>
            <a href="http://example.com">External Link</a>
            <a href="/internal">Internal Link</a>
            <a href="mailto:test@example.com">Email</a>
            <a href="#anchor">Anchor</a>
        </body>
        </html>
        """

        soup = BeautifulSoup(html, "html.parser")
        links = self.processor._extract_links(soup)

        # Should only include external links
        self.assertIn("External Link: http://example.com", links)
        self.assertIn("test@example.com", links)
        self.assertEqual(len(links), 2)

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_extract_metadata_basic(self, mock_bs):
        """Test basic metadata extraction."""
        test_file = Path(self.temp_dir) / "test.html"
        test_file.write_text("<html><head><title>Test Page</title></head></html>")

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_title = MagicMock()
        mock_title.get_text.return_value = "Test Page"
        mock_soup.find.return_value = mock_title
        mock_soup.find_all.return_value = []
        mock_bs.return_value = mock_soup

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["title"], "Test Page")
        self.assertEqual(metadata["file_type"], ".html")
        self.assertIn("heading_count", metadata)

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_extract_metadata_with_meta_tags(self, mock_bs):
        """Test metadata extraction with HTML meta tags."""
        test_file = Path(self.temp_dir) / "meta.html"
        test_file.write_text(
            "<html><head><meta name='description' content='Test description'></head></html>"
        )

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.find.return_value = None  # No title

        mock_meta = MagicMock()
        mock_meta.get.side_effect = lambda key: {
            "name": "description",
            "content": "Test description",
        }.get(key)
        mock_soup.find_all.return_value = [mock_meta]

        mock_bs.return_value = mock_soup

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["meta_tags"]["description"], "Test description")

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_extract_metadata_element_counts(self, mock_bs):
        """Test element counting in metadata."""
        test_file = Path(self.temp_dir) / "elements.html"
        test_file.write_text(
            "<html><body><h1>Title</h1><p>Para</p><a href='#'>Link</a></body></html>"
        )

        # Mock BeautifulSoup
        mock_soup = MagicMock()
        mock_soup.find.return_value = None

        # Mock find_all to return different counts for different elements
        def mock_find_all(tags):
            if tags == ["h1", "h2", "h3", "h4", "h5", "h6"]:
                return [MagicMock()]  # 1 heading
            elif tags == "p":
                return [MagicMock()]  # 1 paragraph
            elif isinstance(tags, str) and "href" in str(tags):
                return [MagicMock()]  # 1 link
            else:
                return []

        mock_soup.find_all.side_effect = mock_find_all
        mock_bs.return_value = mock_soup

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["heading_count"], 1)
        self.assertEqual(metadata["paragraph_count"], 1)

    def test_extract_metadata_without_beautifulsoup(self):
        """Test metadata extraction when beautifulsoup4 is not available."""
        test_file = Path(self.temp_dir) / "test.html"
        test_file.write_text("<html><body>Test</body></html>")

        with patch.dict("sys.modules", {"bs4": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                metadata = self.processor.extract_metadata(test_file)

                self.assertIn("parsing_error", metadata)
                self.assertIn("beautifulsoup4", metadata["parsing_error"])

    def test_extract_metadata_parsing_error(self):
        """Test metadata extraction with parsing errors."""
        test_file = Path(self.temp_dir) / "test.html"
        test_file.write_text("<html><body>Test</body></html>")

        with patch(
            "agraph.processor.html_processor.BeautifulSoup", side_effect=Exception("Parse error")
        ):
            metadata = self.processor.extract_metadata(test_file)

            self.assertIn("parsing_error", metadata)

    def test_calculate_max_depth(self):
        """Test depth calculation for nested structures."""
        # Test with mocked nested structure
        nested_data = {"level1": {"level2": {"level3": "value"}}}

        # This would typically be called on JSON-like structures in HTML
        # For this test, we're testing the depth calculation logic
        depth = self.processor._calculate_max_depth(nested_data)
        self.assertEqual(depth, 3)

    def test_extract_structured_text_components(self):
        """Test individual components of structured text extraction."""
        from bs4 import BeautifulSoup

        # Test title extraction
        html_with_title = "<html><head><title>Page Title</title></head></html>"
        soup = BeautifulSoup(html_with_title, "html.parser")
        titles = self.processor._extract_title(soup)
        self.assertEqual(titles, ["Title: Page Title"])

        # Test heading extraction
        html_with_headings = "<html><body><h1>H1</h1><h2>H2</h2></body></html>"
        soup = BeautifulSoup(html_with_headings, "html.parser")
        headings = self.processor._extract_headings(soup)
        self.assertIn("H1", headings[0])
        self.assertIn("  H2", headings[1])  # H2 should be indented

    def test_extract_paragraphs(self):
        """Test paragraph extraction."""
        from bs4 import BeautifulSoup

        html = "<html><body><p>First para</p><p>Second para</p></body></html>"
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = self.processor._extract_paragraphs(soup)

        self.assertEqual(len(paragraphs), 2)
        self.assertIn("First para", paragraphs)
        self.assertIn("Second para", paragraphs)

    def test_extract_lists(self):
        """Test list extraction."""
        from bs4 import BeautifulSoup

        html = """
        <html><body>
            <ul>
                <li>Item 1</li>
                <li>Item 2</li>
            </ul>
            <ol>
                <li>First</li>
                <li>Second</li>
            </ol>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        list_items = self.processor._extract_lists(soup)

        self.assertIn("• Item 1", list_items)
        self.assertIn("• Item 2", list_items)
        self.assertIn("1. First", list_items)
        self.assertIn("2. Second", list_items)

    def test_extract_tables(self):
        """Test table extraction."""
        from bs4 import BeautifulSoup

        html = """
        <html><body>
            <table>
                <tr><th>Header 1</th><th>Header 2</th></tr>
                <tr><td>Cell 1</td><td>Cell 2</td></tr>
            </table>
        </body></html>
        """
        soup = BeautifulSoup(html, "html.parser")
        table_content = self.processor._extract_tables(soup)

        self.assertIn("Table:", table_content)
        self.assertIn("Header 1 | Header 2", table_content)
        self.assertIn("Cell 1 | Cell 2", table_content)

    def test_process_with_custom_encoding(self):
        """Test processing with custom encoding."""
        test_file = Path(self.temp_dir) / "encoded.html"
        content = "<html><body>café naïve</body></html>"
        test_file.write_text(content, encoding="latin-1")

        with patch("agraph.processor.html_processor.BeautifulSoup") as mock_bs:
            mock_soup = MagicMock()
            mock_soup.get_text.return_value = "café naïve"
            mock_bs.return_value = mock_soup

            result = self.processor.process(test_file, encoding="latin-1")

            self.assertEqual(result, "café naïve")

    def test_process_empty_file_raises_error(self):
        """Test that processing empty file raises appropriate error."""
        empty_file = Path(self.temp_dir) / "empty.html"
        empty_file.touch()

        with self.assertRaises(ValueError) as context:
            self.processor.process(empty_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_nonexistent_file_raises_error(self):
        """Test that processing non-existent file raises appropriate error."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.html"

        with self.assertRaises(FileNotFoundError):
            self.processor.process(nonexistent_file)

    @patch("agraph.processor.html_processor.BeautifulSoup")
    def test_analyze_content_structure(self, mock_bs):
        """Test content structure analysis."""
        test_file = Path(self.temp_dir) / "structured.html"
        test_file.write_text(
            "<html><body><header><nav></nav></header><main></main><footer></footer></body></html>"
        )

        # Mock BeautifulSoup structure analysis
        mock_soup = MagicMock()
        mock_soup.find.side_effect = lambda tag: (
            MagicMock() if tag in ["nav", "footer", "header", "main"] else None
        )
        mock_soup.find_all.return_value = []
        mock_bs.return_value = mock_soup

        metadata = self.processor.extract_metadata(test_file)

        self.assertTrue(metadata["has_navigation"])
        self.assertTrue(metadata["has_footer"])
        self.assertTrue(metadata["has_header"])
        self.assertTrue(metadata["has_main_content"])

    def test_process_html_parsing_error(self):
        """Test handling of HTML parsing errors."""
        test_file = Path(self.temp_dir) / "test.html"
        test_file.write_text("<html><body>Test</body></html>")

        with patch(
            "agraph.processor.html_processor.BeautifulSoup", side_effect=Exception("Parse error")
        ):
            with self.assertRaises(ProcessingError) as context:
                self.processor.process(test_file)

            self.assertIn("Failed to process HTML", str(context.exception))

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
