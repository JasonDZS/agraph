"""Tests for PDF document processor."""

from unittest.mock import Mock, patch

import pytest

from agraph.processer.base import ProcessingError
from agraph.processer.pdf_processor import PDFProcessor
from tests.processer.conftest import skip_if_no_module


class TestPDFProcessor:
    """Test PDFProcessor functionality."""

    def test_supported_extensions(self):
        """Test that PDFProcessor supports correct extensions."""
        processor = PDFProcessor()
        expected_extensions = [".pdf"]
        assert processor.supported_extensions == expected_extensions

    def test_process_no_pypdf(self, temp_dir):
        """Test processing when pypdf is not available."""
        # Create a dummy PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        processor = PDFProcessor()

        with patch.dict("sys.modules", {"pypdf": None}):
            with pytest.raises(ProcessingError, match="pypdf is required"):
                processor.process(pdf_file)

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_process_pdf_file(self, temp_dir):
        """Test processing a PDF file."""
        # Create a mock PDF file
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        processor = PDFProcessor()

        # Mock pypdf classes
        mock_page = Mock()
        mock_page.extract_text.return_value = "Sample PDF content"

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = processor.process(pdf_file)

            assert result == "Sample PDF content"

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_process_pdf_multiple_pages(self, temp_dir):
        """Test processing PDF with multiple pages."""
        pdf_file = temp_dir / "multipage.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        processor = PDFProcessor()

        # Mock multiple pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page1, mock_page2]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = processor.process(pdf_file)

            expected = "Page 1 content\nPage 2 content"
            assert result == expected

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_process_pdf_specific_pages(self, temp_dir):
        """Test processing specific pages from PDF."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        processor = PDFProcessor()

        # Mock multiple pages
        mock_page1 = Mock()
        mock_page1.extract_text.return_value = "Page 1 content"
        mock_page2 = Mock()
        mock_page2.extract_text.return_value = "Page 2 content"
        mock_page3 = Mock()
        mock_page3.extract_text.return_value = "Page 3 content"

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page1, mock_page2, mock_page3]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            # Test single page
            result = processor.process(pdf_file, pages=1)
            assert result == "Page 2 content"

            # Test multiple specific pages
            result = processor.process(pdf_file, pages=[0, 2])
            expected = "Page 1 content\nPage 3 content"
            assert result == expected

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_process_pdf_out_of_range_pages(self, temp_dir):
        """Test processing with out-of-range page numbers."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        processor = PDFProcessor()

        mock_page = Mock()
        mock_page.extract_text.return_value = "Page content"

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [mock_page]  # Only one page

        with patch("pypdf.PdfReader", return_value=mock_reader):
            # Request page that doesn't exist
            result = processor.process(pdf_file, pages=[0, 5])
            # Should only process existing pages
            assert result == "Page content"

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_process_encrypted_pdf_with_password(self, temp_dir):
        """Test processing encrypted PDF with correct password."""
        pdf_file = temp_dir / "encrypted.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 encrypted content")

        processor = PDFProcessor()

        mock_page = Mock()
        mock_page.extract_text.return_value = "Decrypted content"

        mock_reader = Mock()
        mock_reader.is_encrypted = True
        mock_reader.decrypt.return_value = True
        mock_reader.pages = [mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = processor.process(pdf_file, password="secret")

            assert result == "Decrypted content"
            mock_reader.decrypt.assert_called_once_with("secret")

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_process_encrypted_pdf_no_password(self, temp_dir):
        """Test processing encrypted PDF without password."""
        pdf_file = temp_dir / "encrypted.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 encrypted content")

        processor = PDFProcessor()

        mock_reader = Mock()
        mock_reader.is_encrypted = True

        with patch("pypdf.PdfReader", return_value=mock_reader):
            with pytest.raises(ProcessingError, match="PDF is encrypted but no password provided"):
                processor.process(pdf_file)

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_process_pdf_read_error(self, temp_dir):
        """Test handling PDF read errors."""
        pdf_file = temp_dir / "corrupted.pdf"
        pdf_file.write_bytes(b"corrupted pdf content")

        processor = PDFProcessor()

        with patch("pypdf.PdfReader", side_effect=Exception("PDF read error")):
            with pytest.raises(ProcessingError, match="Failed to process PDF file"):
                processor.process(pdf_file)

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_extract_metadata_basic(self, temp_dir):
        """Test extracting basic metadata from PDF."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        processor = PDFProcessor()

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [Mock(), Mock()]  # 2 pages
        mock_reader.metadata = None

        with patch("pypdf.PdfReader", return_value=mock_reader):
            metadata = processor.extract_metadata(pdf_file)

            assert metadata["file_path"] == str(pdf_file)
            assert metadata["page_count"] == 2
            assert metadata["is_encrypted"] is False

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_extract_metadata_with_pdf_metadata(self, temp_dir):
        """Test extracting PDF metadata when available."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        processor = PDFProcessor()

        # Mock PDF metadata
        mock_metadata = {
            "/Title": "Test Document",
            "/Author": "Test Author",
            "/Subject": "Test Subject",
            "/Creator": "Test Creator",
            "/Producer": "Test Producer",
            "/CreationDate": "D:20231201120000",
            "/ModDate": "D:20231201130000"
        }

        mock_reader = Mock()
        mock_reader.is_encrypted = False
        mock_reader.pages = [Mock()]
        mock_reader.metadata = mock_metadata

        with patch("pypdf.PdfReader", return_value=mock_reader):
            metadata = processor.extract_metadata(pdf_file)

            assert metadata["title"] == "Test Document"
            assert metadata["author"] == "Test Author"
            assert metadata["subject"] == "Test Subject"
            assert metadata["creator"] == "Test Creator"
            assert metadata["producer"] == "Test Producer"
            assert metadata["creation_date"] == "D:20231201120000"
            assert metadata["modification_date"] == "D:20231201130000"

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_extract_metadata_encrypted(self, temp_dir):
        """Test extracting metadata from encrypted PDF."""
        pdf_file = temp_dir / "encrypted.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 encrypted content")

        processor = PDFProcessor()

        mock_reader = Mock()
        mock_reader.is_encrypted = True
        mock_reader.pages = [Mock()]
        mock_reader.metadata = None

        with patch("pypdf.PdfReader", return_value=mock_reader):
            metadata = processor.extract_metadata(pdf_file)

            assert metadata["is_encrypted"] is True

    def test_extract_metadata_no_pypdf(self, temp_dir):
        """Test metadata extraction when pypdf is not available."""
        pdf_file = temp_dir / "test.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 dummy content")

        processor = PDFProcessor()

        with patch.dict("sys.modules", {"pypdf": None}):
            metadata = processor.extract_metadata(pdf_file)

            assert metadata["error"] == "pypdf not available"

    @pytest.mark.skipif(skip_if_no_module("pypdf"), reason="pypdf not available")
    def test_extract_metadata_read_error(self, temp_dir):
        """Test metadata extraction handles read errors."""
        pdf_file = temp_dir / "corrupted.pdf"
        pdf_file.write_bytes(b"corrupted content")

        processor = PDFProcessor()

        with patch("pypdf.PdfReader", side_effect=Exception("Read error")):
            metadata = processor.extract_metadata(pdf_file)

            assert "error" in metadata
            assert "Failed to extract metadata" in metadata["error"]

    def test_process_file_not_found(self, nonexistent_file):
        """Test processing non-existent file raises error."""
        processor = PDFProcessor()

        # Mock pypdf to be available so we can test file validation
        with patch("pypdf.PdfReader") as mock_reader:
            with pytest.raises(FileNotFoundError):
                processor.process(nonexistent_file)

    def test_process_empty_file(self, empty_file):
        """Test processing empty file raises error."""
        # Rename to have PDF extension
        pdf_file = empty_file.parent / "empty.pdf"
        empty_file.rename(pdf_file)

        processor = PDFProcessor()

        # Mock pypdf to be available so we can test file validation
        with patch("pypdf.PdfReader") as mock_reader:
            with pytest.raises(ValueError, match="File is empty"):
                processor.process(pdf_file)
