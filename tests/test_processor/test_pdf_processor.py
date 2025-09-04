"""
Test PDF processor functionality.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agraph.processor.base import ProcessingError
from agraph.processor.pdf_processor import PDFProcessor


class TestPDFProcessor(unittest.TestCase):
    """Test PDF document processor."""

    def setUp(self):
        """Set up test environment."""
        self.processor = PDFProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.processor.supported_extensions
        self.assertEqual(extensions, [".pdf"])

    def test_can_process_pdf_files(self):
        """Test PDF file type detection."""
        self.assertTrue(self.processor.can_process("document.pdf"))
        self.assertTrue(self.processor.can_process("Document.PDF"))
        self.assertFalse(self.processor.can_process("document.txt"))

    def test_process_simple_pdf(self):
        """Test processing a simple PDF file."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"fake pdf content")

        # Mock pypdf import and usage
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "PDF page content"
            mock_reader.pages = [mock_page]
            mock_pypdf.PdfReader.return_value = mock_reader

            result = self.processor.process(test_file)

            self.assertEqual(result, "PDF page content")
            mock_pypdf.PdfReader.assert_called_once()

    def test_process_encrypted_pdf_with_password(self):
        """Test processing encrypted PDF with correct password."""
        test_file = Path(self.temp_dir) / "encrypted.pdf"
        test_file.write_bytes(b"fake encrypted pdf")

        # Mock encrypted PDF
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = True
            mock_page = MagicMock()
            mock_page.extract_text.return_value = "Decrypted content"
            mock_reader.pages = [mock_page]
            mock_pypdf.PdfReader.return_value = mock_reader

            result = self.processor.process(test_file, password="correct_password")

            self.assertEqual(result, "Decrypted content")
            mock_reader.decrypt.assert_called_once_with("correct_password")

    def test_process_encrypted_pdf_without_password(self):
        """Test processing encrypted PDF without password raises error."""
        test_file = Path(self.temp_dir) / "encrypted.pdf"
        test_file.write_bytes(b"fake encrypted pdf")

        # Mock encrypted PDF
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = True
            mock_pypdf.PdfReader.return_value = mock_reader

            with self.assertRaises(ProcessingError) as context:
                self.processor.process(test_file)

            self.assertIn("encrypted", str(context.exception).lower())

    def test_process_encrypted_pdf_wrong_password(self):
        """Test processing encrypted PDF with wrong password raises error."""
        test_file = Path(self.temp_dir) / "encrypted.pdf"
        test_file.write_bytes(b"fake encrypted pdf")

        # Mock encrypted PDF with failed decryption
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = True
            mock_reader.decrypt.return_value = False
            mock_pypdf.PdfReader.return_value = mock_reader

            with self.assertRaises(ProcessingError) as context:
                self.processor.process(test_file, password="wrong_password")

            self.assertIn("password", str(context.exception).lower())

    def test_process_specific_pages(self):
        """Test processing specific pages from PDF."""
        test_file = Path(self.temp_dir) / "multipage.pdf"
        test_file.write_bytes(b"fake multipage pdf")

        # Mock multi-page PDF
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_pages = []
            for i in range(3):
                mock_page = MagicMock()
                mock_page.extract_text.return_value = f"Page {i+1} content"
                mock_pages.append(mock_page)
            mock_reader.pages = mock_pages
            mock_pypdf.PdfReader.return_value = mock_reader

            # Test single page
            result = self.processor.process(test_file, pages=1)
            self.assertEqual(result, "Page 2 content")

            # Test multiple pages
            result = self.processor.process(test_file, pages=[0, 2])
            self.assertEqual(result, "Page 1 content\nPage 3 content")

            # Test page range
            result = self.processor.process(test_file, pages=range(2))
            self.assertEqual(result, "Page 1 content\nPage 2 content")

    def test_process_with_empty_pages(self):
        """Test handling of PDFs with empty pages."""
        test_file = Path(self.temp_dir) / "empty_pages.pdf"
        test_file.write_bytes(b"fake pdf with empty pages")

        # Mock PDF with some empty pages
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_pages = []

            # Page 0: content, Page 1: empty, Page 2: content
            for i, content in enumerate(["Real content", "", "More content"]):
                mock_page = MagicMock()
                mock_page.extract_text.return_value = content
                mock_pages.append(mock_page)

            mock_reader.pages = mock_pages
            mock_pypdf.PdfReader.return_value = mock_reader

            result = self.processor.process(test_file)
            self.assertEqual(result, "Real content\nMore content")

    def test_extract_metadata_basic(self):
        """Test basic metadata extraction."""
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        # Mock PDF with metadata
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_reader.pages = [MagicMock(), MagicMock()]  # 2 pages
            mock_reader.metadata = {
                "/Title": "Test Document",
                "/Author": "Test Author",
                "/Subject": "Test Subject",
                "/Creator": "Test Creator",
                "/Producer": "Test Producer",
                "/CreationDate": "D:20231201120000",
                "/ModDate": "D:20231201130000",
            }
            mock_reader.outline = []
            mock_pypdf.PdfReader.return_value = mock_reader

            metadata = self.processor.extract_metadata(test_file)

            self.assertEqual(metadata["page_count"], 2)
            self.assertEqual(metadata["title"], "Test Document")
            self.assertEqual(metadata["author"], "Test Author")
            self.assertFalse(metadata["is_encrypted"])
            self.assertFalse(metadata["has_bookmarks"])

    def test_extract_metadata_with_bookmarks(self):
        """Test metadata extraction with bookmarks."""
        test_file = Path(self.temp_dir) / "bookmarked.pdf"
        test_file.write_bytes(b"fake pdf with bookmarks")

        # Mock PDF with bookmarks
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_reader.pages = [MagicMock()]
            mock_reader.metadata = {}
            mock_reader.outline = ["Bookmark 1", "Bookmark 2"]
            mock_pypdf.PdfReader.return_value = mock_reader

            metadata = self.processor.extract_metadata(test_file)

            self.assertTrue(metadata["has_bookmarks"])

    def test_extract_metadata_without_pypdf(self):
        """Test metadata extraction when pypdf is not available."""
        with patch.dict("sys.modules", {"pypdf": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                test_file = Path(self.temp_dir) / "test.pdf"
                test_file.write_bytes(b"fake pdf")

                metadata = self.processor.extract_metadata(test_file)

                self.assertIn("error", metadata)
                self.assertIn("pypdf", metadata["error"])

    def test_extract_metadata_error_handling(self):
        """Test metadata extraction error handling."""
        test_file = Path(self.temp_dir) / "corrupted.pdf"
        test_file.write_bytes(b"fake corrupted pdf")

        # Mock pypdf to raise exception
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_pypdf.PdfReader.side_effect = ValueError("Corrupted PDF")

            metadata = self.processor.extract_metadata(test_file)

            self.assertIn("error", metadata)
            self.assertIn("Corrupted PDF", metadata["error"])

    def test_process_missing_pypdf_dependency(self):
        """Test processing when pypdf is not installed."""
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        with patch.dict("sys.modules", {"pypdf": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with self.assertRaises(ProcessingError) as context:
                    self.processor.process(test_file)

                self.assertIn("pypdf", str(context.exception))

    def test_process_nonexistent_file(self):
        """Test processing non-existent file raises error."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.pdf"

        with self.assertRaises(FileNotFoundError):
            self.processor.process(nonexistent_file)

    def test_process_empty_file(self):
        """Test processing empty file raises error."""
        empty_file = Path(self.temp_dir) / "empty.pdf"
        empty_file.touch()

        with self.assertRaises(ValueError) as context:
            self.processor.process(empty_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_invalid_page_numbers(self):
        """Test handling of invalid page numbers."""
        test_file = Path(self.temp_dir) / "test.pdf"
        test_file.write_bytes(b"fake pdf")

        # Mock 2-page PDF
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_reader = MagicMock()
            mock_reader.is_encrypted = False
            mock_pages = []
            for i in range(2):
                mock_page = MagicMock()
                mock_page.extract_text.return_value = f"Page {i+1} content"
                mock_pages.append(mock_page)
            mock_reader.pages = mock_pages
            mock_pypdf.PdfReader.return_value = mock_reader

            # Test invalid page numbers (should be ignored, not crash)
            result = self.processor.process(test_file, pages=[0, 5, 1])  # 5 is invalid
            self.assertEqual(result, "Page 1 content\nPage 2 content")

    def test_process_corrupted_pdf(self):
        """Test handling of corrupted PDF files."""
        test_file = Path(self.temp_dir) / "corrupted.pdf"
        test_file.write_bytes(b"fake corrupted pdf")

        # Mock pypdf to raise exception
        with patch.dict("sys.modules", {"pypdf": MagicMock()}) as mock_modules:
            mock_pypdf = mock_modules["pypdf"]
            mock_pypdf.PdfReader.side_effect = ValueError("Not a valid PDF")

            with self.assertRaises(ProcessingError) as context:
                self.processor.process(test_file)

            self.assertIn("Failed to process PDF", str(context.exception))

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
