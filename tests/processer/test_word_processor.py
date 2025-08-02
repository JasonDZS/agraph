"""Tests for Word document processor."""

import subprocess
from unittest.mock import Mock, patch

import pytest

from agraph.processer.base import ProcessingError
from agraph.processer.word_processor import WordProcessor
from tests.processer.conftest import skip_if_no_module


class TestWordProcessor:
    """Test WordProcessor functionality."""

    def test_supported_extensions(self):
        """Test that WordProcessor supports correct extensions."""
        processor = WordProcessor()
        expected_extensions = [".docx", ".doc"]
        assert processor.supported_extensions == expected_extensions

    @pytest.mark.skipif(skip_if_no_module("docx"), reason="python-docx not available")
    def test_process_docx_file(self, temp_dir):
        """Test processing a .docx file."""
        docx_file = temp_dir / "test.docx"
        docx_file.write_bytes(b"dummy docx content")

        processor = WordProcessor()

        # Mock python-docx Document
        mock_paragraph1 = Mock()
        mock_paragraph1.text = "First paragraph"
        mock_paragraph2 = Mock()
        mock_paragraph2.text = "Second paragraph"
        mock_paragraph3 = Mock()
        mock_paragraph3.text = ""  # Empty paragraph should be skipped

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2, mock_paragraph3]
        mock_doc.tables = []

        with patch("docx.Document", return_value=mock_doc):
            result = processor.process(docx_file)

            expected = "First paragraph\nSecond paragraph"
            assert result == expected

    @pytest.mark.skipif(skip_if_no_module("docx"), reason="python-docx not available")
    def test_process_docx_file_with_tables(self, temp_dir):
        """Test processing .docx file with tables."""
        docx_file = temp_dir / "test.docx"
        docx_file.write_bytes(b"dummy docx content")

        processor = WordProcessor()

        # Mock paragraph
        mock_paragraph = Mock()
        mock_paragraph.text = "Document text"

        # Mock table cells
        mock_cell1 = Mock()
        mock_cell1.text = "Cell 1"
        mock_cell2 = Mock()
        mock_cell2.text = "Cell 2"
        mock_cell3 = Mock()
        mock_cell3.text = ""  # Empty cell
        mock_cell4 = Mock()
        mock_cell4.text = "Cell 4"

        # Mock table rows
        mock_row1 = Mock()
        mock_row1.cells = [mock_cell1, mock_cell2]
        mock_row2 = Mock()
        mock_row2.cells = [mock_cell3, mock_cell4]

        # Mock table
        mock_table = Mock()
        mock_table.rows = [mock_row1, mock_row2]

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = [mock_table]

        with patch("docx.Document", return_value=mock_doc):
            result = processor.process(docx_file, include_tables=True)

            expected = "Document text\nCell 1\tCell 2\nCell 4"
            assert result == expected

    @pytest.mark.skipif(skip_if_no_module("docx"), reason="python-docx not available")
    def test_process_docx_file_exclude_tables(self, temp_dir):
        """Test processing .docx file excluding tables."""
        docx_file = temp_dir / "test.docx"
        docx_file.write_bytes(b"dummy docx content")

        processor = WordProcessor()

        mock_paragraph = Mock()
        mock_paragraph.text = "Document text"

        mock_doc = Mock()
        mock_doc.paragraphs = [mock_paragraph]
        mock_doc.tables = [Mock()]  # Table exists but should be ignored

        with patch("docx.Document", return_value=mock_doc):
            result = processor.process(docx_file, include_tables=False)

            assert result == "Document text"

    def test_process_docx_no_python_docx(self, temp_dir):
        """Test processing .docx when python-docx is not available."""
        docx_file = temp_dir / "test.docx"
        docx_file.write_bytes(b"dummy docx content")

        processor = WordProcessor()

        with patch.dict("sys.modules", {"docx": None}):
            with pytest.raises(ProcessingError, match="python-docx is required"):
                processor.process(docx_file)

    @pytest.mark.skipif(skip_if_no_module("docx"), reason="python-docx not available")
    def test_process_docx_read_error(self, temp_dir):
        """Test handling .docx read errors."""
        docx_file = temp_dir / "corrupted.docx"
        docx_file.write_bytes(b"corrupted content")

        processor = WordProcessor()

        with patch("docx.Document", side_effect=Exception("Read error")):
            with pytest.raises(ProcessingError, match="Failed to process .docx file"):
                processor.process(docx_file)

    def test_process_doc_file_with_docx2txt(self, temp_dir):
        """Test processing .doc file using docx2txt."""
        doc_file = temp_dir / "test.doc"
        doc_file.write_bytes(b"dummy doc content")

        processor = WordProcessor()

        # Mock docx2txt module and its process function
        mock_docx2txt = Mock()
        mock_docx2txt.process.return_value = "Extracted text from doc"

        with patch.dict("sys.modules", {"docx2txt": mock_docx2txt}):
            with patch("docx2txt.process", return_value="Extracted text from doc"):
                result = processor.process(doc_file)

                assert result == "Extracted text from doc"

    def test_process_doc_file_with_antiword(self, temp_dir):
        """Test processing .doc file using antiword fallback."""
        doc_file = temp_dir / "test.doc"
        doc_file.write_bytes(b"dummy doc content")

        processor = WordProcessor()

        # Mock docx2txt not being available
        with patch.dict("sys.modules", {"docx2txt": None}):
            # Mock successful antiword execution
            mock_result = Mock()
            mock_result.stdout = "Antiword extracted text"

            with patch("subprocess.run", return_value=mock_result) as mock_subprocess:
                result = processor.process(doc_file)

                assert result == "Antiword extracted text"
                mock_subprocess.assert_called_once_with(
                    ["antiword", str(doc_file)],
                    capture_output=True,
                    text=True,
                    check=True
                )

    def test_process_doc_file_no_tools(self, temp_dir):
        """Test processing .doc file when no tools are available."""
        doc_file = temp_dir / "test.doc"
        doc_file.write_bytes(b"dummy doc content")

        processor = WordProcessor()

        # Mock both docx2txt and antiword not being available
        with patch.dict("sys.modules", {"docx2txt": None}):
            with patch("subprocess.run", side_effect=FileNotFoundError("antiword not found")):
                with pytest.raises(ProcessingError, match="Cannot process .doc files"):
                    processor.process(doc_file)

    def test_process_doc_file_antiword_error(self, temp_dir):
        """Test processing .doc file when antiword fails."""
        doc_file = temp_dir / "test.doc"
        doc_file.write_bytes(b"dummy doc content")

        processor = WordProcessor()

        with patch.dict("sys.modules", {"docx2txt": None}):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "antiword")):
                with pytest.raises(ProcessingError, match="Cannot process .doc files"):
                    processor.process(doc_file)

    def test_process_unsupported_extension(self, temp_dir):
        """Test processing file with unsupported extension."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("plain text")

        processor = WordProcessor()

        with pytest.raises(ProcessingError, match="Unsupported file extension"):
            processor.process(txt_file)

    @pytest.mark.skipif(skip_if_no_module("docx"), reason="python-docx not available")
    def test_extract_metadata_docx(self, temp_dir):
        """Test extracting metadata from .docx file."""
        docx_file = temp_dir / "test.docx"
        docx_file.write_bytes(b"dummy docx content")

        processor = WordProcessor()

        # Mock core properties
        mock_props = Mock()
        mock_props.title = "Test Document"
        mock_props.author = "Test Author"
        mock_props.subject = "Test Subject"
        mock_props.keywords = "test, document"
        mock_props.category = "Test Category"
        mock_props.comments = "Test comments"
        mock_props.created = "2023-01-01T12:00:00"
        mock_props.modified = "2023-01-02T12:00:00"
        mock_props.last_modified_by = "Last Editor"

        mock_doc = Mock()
        mock_doc.core_properties = mock_props
        mock_doc.paragraphs = [Mock(), Mock()]  # 2 paragraphs
        mock_doc.tables = [Mock()]  # 1 table
        mock_doc.sections = [Mock(), Mock(), Mock()]  # 3 sections

        with patch("docx.Document", return_value=mock_doc):
            metadata = processor.extract_metadata(docx_file)

            assert metadata["file_path"] == str(docx_file)
            assert metadata["file_type"] == ".docx"
            assert metadata["title"] == "Test Document"
            assert metadata["author"] == "Test Author"
            assert metadata["subject"] == "Test Subject"
            assert metadata["keywords"] == "test, document"
            assert metadata["category"] == "Test Category"
            assert metadata["comments"] == "Test comments"
            assert metadata["created"] == "2023-01-01T12:00:00"
            assert metadata["modified"] == "2023-01-02T12:00:00"
            assert metadata["last_modified_by"] == "Last Editor"
            assert metadata["paragraph_count"] == 2
            assert metadata["table_count"] == 1
            assert metadata["section_count"] == 3

    @pytest.mark.skipif(skip_if_no_module("docx"), reason="python-docx not available")
    def test_extract_metadata_docx_empty_properties(self, temp_dir):
        """Test extracting metadata from .docx with empty properties."""
        docx_file = temp_dir / "test.docx"
        docx_file.write_bytes(b"dummy docx content")

        processor = WordProcessor()

        # Mock empty core properties
        mock_props = Mock()
        mock_props.title = None
        mock_props.author = None
        mock_props.subject = None
        mock_props.keywords = None
        mock_props.category = None
        mock_props.comments = None
        mock_props.created = None
        mock_props.modified = None
        mock_props.last_modified_by = None

        mock_doc = Mock()
        mock_doc.core_properties = mock_props
        mock_doc.paragraphs = []
        mock_doc.tables = []
        mock_doc.sections = []

        with patch("docx.Document", return_value=mock_doc):
            metadata = processor.extract_metadata(docx_file)

            assert metadata["title"] == ""
            assert metadata["author"] == ""
            assert metadata["created"] == ""

    def test_extract_metadata_docx_no_python_docx(self, temp_dir):
        """Test metadata extraction when python-docx is not available."""
        docx_file = temp_dir / "test.docx"
        docx_file.write_bytes(b"dummy docx content")

        processor = WordProcessor()

        with patch.dict("sys.modules", {"docx": None}):
            metadata = processor.extract_metadata(docx_file)

            assert "error" in metadata
            assert "python-docx not available" in metadata["error"]

    @pytest.mark.skipif(skip_if_no_module("docx"), reason="python-docx not available")
    def test_extract_metadata_docx_read_error(self, temp_dir):
        """Test metadata extraction handles read errors."""
        docx_file = temp_dir / "corrupted.docx"
        docx_file.write_bytes(b"corrupted content")

        processor = WordProcessor()

        with patch("docx.Document", side_effect=Exception("Read error")):
            metadata = processor.extract_metadata(docx_file)

            assert "error" in metadata
            assert "Failed to extract metadata" in metadata["error"]

    def test_extract_metadata_doc_file(self, temp_dir):
        """Test metadata extraction from .doc file."""
        doc_file = temp_dir / "test.doc"
        doc_file.write_bytes(b"dummy doc content")

        processor = WordProcessor()

        metadata = processor.extract_metadata(doc_file)

        assert metadata["file_path"] == str(doc_file)
        assert metadata["file_type"] == ".doc"
        assert metadata["note"] == "Metadata extraction not supported for .doc files"

    def test_process_file_not_found(self, nonexistent_file):
        """Test processing non-existent file raises error."""
        processor = WordProcessor()

        with pytest.raises(FileNotFoundError):
            processor.process(nonexistent_file)

    def test_process_empty_file(self, temp_dir):
        """Test processing empty file raises error."""
        empty_docx = temp_dir / "empty.docx"
        empty_docx.write_text("")

        processor = WordProcessor()

        with pytest.raises(ValueError, match="File is empty"):
            processor.process(empty_docx)
