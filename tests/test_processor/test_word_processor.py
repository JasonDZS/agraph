"""
Test Word processor functionality.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agraph.processor.base import ProcessingError
from agraph.processor.word_processor import WordProcessor


class TestWordProcessor(unittest.TestCase):
    """Test Word document processor."""

    def setUp(self):
        """Set up test environment."""
        self.processor = WordProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.processor.supported_extensions
        expected = [".docx", ".doc"]
        self.assertEqual(extensions, expected)

    def test_can_process_word_files(self):
        """Test Word file type detection."""
        self.assertTrue(self.processor.can_process("document.docx"))
        self.assertTrue(self.processor.can_process("legacy.doc"))
        self.assertFalse(self.processor.can_process("document.txt"))

    def test_process_docx_file(self):
        """Test processing .docx files."""
        test_file = Path(self.temp_dir) / "test.docx"
        test_file.write_bytes(b"fake docx content")

        # Mock docx Document
        with patch.dict("sys.modules", {"docx": MagicMock()}) as mock_modules:
            mock_docx = mock_modules["docx"]
            mock_doc = MagicMock()
            mock_paragraph1 = MagicMock()
            mock_paragraph1.text = "First paragraph"
            mock_paragraph2 = MagicMock()
            mock_paragraph2.text = "Second paragraph"
            mock_doc.paragraphs = [mock_paragraph1, mock_paragraph2]
            mock_doc.tables = []
            mock_docx.Document.return_value = mock_doc

            result = self.processor.process(test_file)

            self.assertEqual(result, "First paragraph\nSecond paragraph")
            mock_docx.Document.assert_called_once_with(str(test_file))

    def test_process_docx_with_tables(self):
        """Test processing .docx files with tables."""
        test_file = Path(self.temp_dir) / "with_tables.docx"
        test_file.write_bytes(b"fake docx with tables")

        # Mock docx Document with table
        with patch.dict("sys.modules", {"docx": MagicMock()}) as mock_modules:
            mock_docx = mock_modules["docx"]
            mock_doc = MagicMock()
            mock_doc.paragraphs = []

            # Mock table
            mock_table = MagicMock()
            mock_row = MagicMock()
            mock_cell1 = MagicMock()
            mock_cell1.text = "Cell 1"
            mock_cell2 = MagicMock()
            mock_cell2.text = "Cell 2"
            mock_row.cells = [mock_cell1, mock_cell2]
            mock_table.rows = [mock_row]
            mock_doc.tables = [mock_table]

            mock_docx.Document.return_value = mock_doc

            result = self.processor.process(test_file, include_tables=True)

            self.assertIn("Cell 1\tCell 2", result)

    def test_process_docx_without_tables(self):
        """Test processing .docx files without including tables."""
        test_file = Path(self.temp_dir) / "with_tables.docx"
        test_file.write_bytes(b"fake docx with tables")

        # Mock docx Document
        with patch.dict("sys.modules", {"docx": MagicMock()}) as mock_modules:
            mock_docx = mock_modules["docx"]
            mock_doc = MagicMock()
            mock_paragraph = MagicMock()
            mock_paragraph.text = "Text content"
            mock_doc.paragraphs = [mock_paragraph]
            mock_doc.tables = [MagicMock()]  # Table exists but should be ignored
            mock_docx.Document.return_value = mock_doc

            result = self.processor.process(test_file, include_tables=False)

            self.assertEqual(result, "Text content")

    def test_process_docx_without_python_docx(self):
        """Test processing .docx when python-docx is not available."""
        test_file = Path(self.temp_dir) / "test.docx"
        test_file.write_bytes(b"fake docx")

        with patch.dict("sys.modules", {"docx": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with self.assertRaises(ProcessingError) as context:
                    self.processor.process(test_file)

                self.assertIn("python-docx", str(context.exception))

    def test_process_doc_file_with_docx2txt(self):
        """Test processing .doc files with docx2txt."""
        test_file = Path(self.temp_dir) / "test.doc"
        test_file.write_bytes(b"fake doc content")

        with patch.dict("sys.modules", {"docx2txt": MagicMock()}) as mock_modules:
            mock_docx2txt = mock_modules["docx2txt"]
            mock_docx2txt.process.return_value = "Extracted doc content"

            result = self.processor.process(test_file)

            self.assertEqual(result, "Extracted doc content")
            mock_docx2txt.process.assert_called_once_with(str(test_file))

    @patch("agraph.processor.word_processor.subprocess")
    def test_process_doc_file_with_antiword(self, mock_subprocess):
        """Test processing .doc files with antiword fallback."""
        test_file = Path(self.temp_dir) / "test.doc"
        test_file.write_bytes(b"fake doc content")

        # Mock docx2txt import failure
        with patch.dict("sys.modules", {"docx2txt": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                # Mock successful antiword execution
                mock_result = MagicMock()
                mock_result.stdout = "Antiword extracted content"
                mock_subprocess.run.return_value = mock_result

                result = self.processor.process(test_file)

                self.assertEqual(result, "Antiword extracted content")
                mock_subprocess.run.assert_called_once()

    def test_process_doc_file_no_dependencies(self):
        """Test processing .doc files when no dependencies are available."""
        test_file = Path(self.temp_dir) / "test.doc"
        test_file.write_bytes(b"fake doc content")

        with patch.dict("sys.modules", {"docx2txt": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with patch(
                    "agraph.processor.word_processor.subprocess.run", side_effect=FileNotFoundError
                ):
                    with self.assertRaises(ProcessingError) as context:
                        self.processor.process(test_file)

                    self.assertIn("Cannot process .doc", str(context.exception))

    def test_process_unsupported_extension(self):
        """Test processing file with unsupported extension."""
        test_file = Path(self.temp_dir) / "test.rtf"
        test_file.write_bytes(b"fake rtf content")

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("Unsupported file extension", str(context.exception))

    def test_extract_table_content(self):
        """Test table content extraction."""
        # Mock table structure
        mock_table = MagicMock()
        mock_row1 = MagicMock()
        mock_row2 = MagicMock()

        # Row 1 cells
        mock_cell1_1 = MagicMock()
        mock_cell1_1.text = "Header 1"
        mock_cell1_2 = MagicMock()
        mock_cell1_2.text = "Header 2"
        mock_row1.cells = [mock_cell1_1, mock_cell1_2]

        # Row 2 cells
        mock_cell2_1 = MagicMock()
        mock_cell2_1.text = "Data 1"
        mock_cell2_2 = MagicMock()
        mock_cell2_2.text = "Data 2"
        mock_row2.cells = [mock_cell2_1, mock_cell2_2]

        mock_table.rows = [mock_row1, mock_row2]

        result = self.processor._extract_table_content(mock_table)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], "Header 1\tHeader 2")
        self.assertEqual(result[1], "Data 1\tData 2")

    def test_extract_table_content_with_empty_cells(self):
        """Test table extraction with empty cells."""
        # Mock table with some empty cells
        mock_table = MagicMock()
        mock_row = MagicMock()

        mock_cell1 = MagicMock()
        mock_cell1.text = "Content"
        mock_cell2 = MagicMock()
        mock_cell2.text = ""  # Empty cell
        mock_cell3 = MagicMock()
        mock_cell3.text = "More content"

        mock_row.cells = [mock_cell1, mock_cell2, mock_cell3]
        mock_table.rows = [mock_row]

        result = self.processor._extract_table_content(mock_table)

        self.assertEqual(result[0], "Content\t\tMore content")

    def test_extract_docx_metadata_basic(self):
        """Test basic metadata extraction from .docx files."""
        test_file = Path(self.temp_dir) / "test.docx"
        test_file.write_bytes(b"fake docx")

        # Mock docx Document and properties
        with patch.dict("sys.modules", {"docx": MagicMock()}) as mock_modules:
            mock_docx = mock_modules["docx"]
            mock_doc = MagicMock()
            mock_props = MagicMock()
            mock_props.title = "Test Document"
            mock_props.author = "Test Author"
            mock_props.subject = "Test Subject"
            mock_props.keywords = "test, document"
            mock_props.created = None
            mock_props.modified = None
            mock_props.last_modified_by = "Test User"
            mock_props.category = ""
            mock_props.comments = ""

            mock_doc.core_properties = mock_props
            mock_doc.paragraphs = [MagicMock(), MagicMock()]  # 2 paragraphs
            mock_doc.tables = [MagicMock()]  # 1 table
            mock_doc.sections = [MagicMock()]  # 1 section
            mock_docx.Document.return_value = mock_doc

            metadata = self.processor.extract_metadata(test_file)

            self.assertEqual(metadata["title"], "Test Document")
            self.assertEqual(metadata["author"], "Test Author")
            self.assertEqual(metadata["paragraph_count"], 2)
            self.assertEqual(metadata["table_count"], 1)

    @patch("agraph.processor.word_processor.docx")
    def test_extract_docx_metadata_with_empty_paragraphs(self, mock_docx):
        """Test metadata extraction counting non-empty paragraphs."""
        test_file = Path(self.temp_dir) / "test.docx"
        test_file.write_bytes(b"fake docx")

        # Mock document with mixed empty/non-empty paragraphs
        mock_doc = MagicMock()
        mock_props = MagicMock()
        mock_props.title = ""
        mock_props.author = ""
        mock_props.subject = ""
        mock_props.keywords = ""
        mock_props.created = None
        mock_props.modified = None
        mock_props.last_modified_by = ""
        mock_props.category = ""
        mock_props.comments = ""

        # Mock paragraphs
        mock_para1 = MagicMock()
        mock_para1.text = "Content paragraph"
        mock_para2 = MagicMock()
        mock_para2.text = ""  # Empty
        mock_para3 = MagicMock()
        mock_para3.text = "   "  # Whitespace only
        mock_para4 = MagicMock()
        mock_para4.text = "Another content paragraph"

        mock_doc.core_properties = mock_props
        mock_doc.paragraphs = [mock_para1, mock_para2, mock_para3, mock_para4]
        mock_doc.tables = []
        mock_doc.sections = []
        mock_docx.Document.return_value = mock_doc

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["paragraph_count"], 4)
        self.assertEqual(metadata["non_empty_paragraph_count"], 2)

    def test_extract_metadata_doc_file(self):
        """Test metadata extraction from .doc files."""
        test_file = Path(self.temp_dir) / "test.doc"
        test_file.write_bytes(b"fake doc content")

        metadata = self.processor.extract_metadata(test_file)

        self.assertIn("note", metadata)
        self.assertIn("not supported", metadata["note"])

    def test_extract_docx_metadata_without_python_docx(self):
        """Test metadata extraction when python-docx is not available."""
        test_file = Path(self.temp_dir) / "test.docx"
        test_file.write_bytes(b"fake docx")

        with patch.dict("sys.modules", {"docx": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                metadata = self.processor.extract_metadata(test_file)

                self.assertIn("error", metadata)
                self.assertIn("python-docx", metadata["error"])

    @patch("agraph.processor.word_processor.docx")
    def test_extract_docx_metadata_processing_error(self, mock_docx):
        """Test metadata extraction with processing errors."""
        test_file = Path(self.temp_dir) / "corrupted.docx"
        test_file.write_bytes(b"fake corrupted docx")

        mock_docx.Document.side_effect = Exception("Corrupted document")

        metadata = self.processor.extract_metadata(test_file)

        self.assertIn("error", metadata)
        self.assertIn("Corrupted document", metadata["error"])

    @patch("agraph.processor.word_processor.docx")
    def test_process_docx_processing_error(self, mock_docx):
        """Test .docx processing with errors."""
        test_file = Path(self.temp_dir) / "corrupted.docx"
        test_file.write_bytes(b"fake corrupted docx")

        mock_docx.Document.side_effect = Exception("Document error")

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("Failed to process .docx", str(context.exception))

    @patch("agraph.processor.word_processor.subprocess")
    def test_process_doc_with_antiword_timeout(self, mock_subprocess):
        """Test .doc processing with antiword timeout."""
        test_file = Path(self.temp_dir) / "slow.doc"
        test_file.write_bytes(b"fake slow doc")

        # Mock docx2txt import failure and antiword timeout
        with patch.dict("sys.modules", {"docx2txt": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                mock_subprocess.run.side_effect = mock_subprocess.TimeoutExpired("antiword", 30)

                with self.assertRaises(ProcessingError) as context:
                    self.processor.process(test_file)

                self.assertIn("Cannot process .doc", str(context.exception))

    @patch("agraph.processor.word_processor.subprocess")
    def test_process_doc_with_antiword_error(self, mock_subprocess):
        """Test .doc processing with antiword execution error."""
        test_file = Path(self.temp_dir) / "bad.doc"
        test_file.write_bytes(b"fake bad doc")

        # Mock docx2txt import failure and antiword error
        with patch.dict("sys.modules", {"docx2txt": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                mock_subprocess.run.side_effect = mock_subprocess.CalledProcessError(1, "antiword")

                with self.assertRaises(ProcessingError) as context:
                    self.processor.process(test_file)

                self.assertIn("Cannot process .doc", str(context.exception))

    def test_extract_table_content_empty_table(self):
        """Test table extraction with completely empty table."""
        # Mock empty table
        mock_table = MagicMock()
        mock_row = MagicMock()
        mock_empty_cell = MagicMock()
        mock_empty_cell.text = ""
        mock_row.cells = [mock_empty_cell, mock_empty_cell]
        mock_table.rows = [mock_row]

        result = self.processor._extract_table_content(mock_table)

        # Empty table should return empty list
        self.assertEqual(result, [])

    def test_process_empty_file_raises_error(self):
        """Test that processing empty file raises appropriate error."""
        empty_file = Path(self.temp_dir) / "empty.docx"
        empty_file.touch()

        with self.assertRaises(ValueError) as context:
            self.processor.process(empty_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_nonexistent_file_raises_error(self):
        """Test that processing non-existent file raises appropriate error."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.docx"

        with self.assertRaises(FileNotFoundError):
            self.processor.process(nonexistent_file)

    @patch("agraph.processor.word_processor.docx2txt")
    def test_process_doc_returns_none(self, mock_docx2txt):
        """Test handling when docx2txt returns None."""
        test_file = Path(self.temp_dir) / "empty_result.doc"
        test_file.write_bytes(b"fake doc")

        mock_docx2txt.process.return_value = None

        result = self.processor.process(test_file)

        self.assertEqual(result, "")

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
