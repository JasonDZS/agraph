"""
Test spreadsheet processor functionality.
"""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agraph.processor.base import ProcessingError
from agraph.processor.spreadsheet_processor import SpreadsheetProcessor


class TestSpreadsheetProcessor(unittest.TestCase):
    """Test spreadsheet document processor."""

    def setUp(self):
        """Set up test environment."""
        self.processor = SpreadsheetProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.processor.supported_extensions
        expected = [".csv", ".xlsx", ".xls"]
        self.assertEqual(extensions, expected)

    def test_can_process_spreadsheet_files(self):
        """Test spreadsheet file type detection."""
        self.assertTrue(self.processor.can_process("data.csv"))
        self.assertTrue(self.processor.can_process("workbook.xlsx"))
        self.assertTrue(self.processor.can_process("legacy.xls"))
        self.assertFalse(self.processor.can_process("document.txt"))

    def test_process_simple_csv(self):
        """Test processing simple CSV file."""
        test_file = Path(self.temp_dir) / "simple.csv"
        csv_content = "Name,Age,City\nJohn,30,New York\nJane,25,Boston"
        test_file.write_text(csv_content)

        result = self.processor.process(test_file)

        self.assertIn("Name", result)
        self.assertIn("John", result)
        self.assertIn("Jane", result)

    def test_process_csv_with_custom_delimiter(self):
        """Test processing CSV with custom delimiter."""
        test_file = Path(self.temp_dir) / "semicolon.csv"
        csv_content = "Name;Age;City\nJohn;30;New York\nJane;25;Boston"
        test_file.write_text(csv_content)

        result = self.processor.process(test_file, delimiter=";")

        self.assertIn("Name", result)
        self.assertIn("John", result)

    def test_process_csv_without_headers(self):
        """Test processing CSV without including headers."""
        test_file = Path(self.temp_dir) / "no_headers.csv"
        csv_content = "Name,Age\nJohn,30\nJane,25"
        test_file.write_text(csv_content)

        result = self.processor.process(test_file, include_headers=False)

        self.assertNotIn("Name", result)
        self.assertIn("John", result)

    def test_process_csv_with_row_limit(self):
        """Test processing CSV with row limit."""
        test_file = Path(self.temp_dir) / "large.csv"
        csv_content = "Name,Age\n" + "\n".join([f"Person{i},{20+i}" for i in range(10)])
        test_file.write_text(csv_content)

        result = self.processor.process(test_file, max_rows=3)

        lines = result.strip().split("\n")
        # Header + 3 data rows = 4 lines
        self.assertEqual(len(lines), 4)

    def test_detect_csv_delimiter(self):
        """Test CSV delimiter detection."""
        # Test comma delimiter
        comma_content = "a,b,c\n1,2,3"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(comma_content)
            f.flush()
            with open(f.name, "r") as file:
                delimiter = self.processor._detect_csv_delimiter(file)
                self.assertEqual(delimiter, ",")

        # Test semicolon delimiter
        semicolon_content = "a;b;c\n1;2;3"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(semicolon_content)
            f.flush()
            with open(f.name, "r") as file:
                delimiter = self.processor._detect_csv_delimiter(file)
                self.assertEqual(delimiter, ";")

    def test_detect_csv_delimiter_fallback(self):
        """Test CSV delimiter detection fallback."""
        # Ambiguous content that can't be detected
        ambiguous_content = "abc\ndef"
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
            f.write(ambiguous_content)
            f.flush()
            with open(f.name, "r") as file:
                delimiter = self.processor._detect_csv_delimiter(file)
                self.assertEqual(delimiter, ",")  # Should fallback to comma

    @patch("agraph.processor.spreadsheet_processor.pd")
    def test_process_excel_single_sheet(self, mock_pd):
        """Test processing Excel file single sheet."""
        test_file = Path(self.temp_dir) / "test.xlsx"
        test_file.write_bytes(b"fake excel content")

        # Mock pandas DataFrame
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.astype.return_value.replace.return_value = mock_df
        mock_df.columns = ["Name", "Age"]
        mock_df.iterrows.return_value = [(0, ["John", "30"]), (1, ["Jane", "25"])]
        mock_pd.read_excel.return_value = mock_df

        result = self.processor.process(test_file)

        mock_pd.read_excel.assert_called_once()
        self.assertIsInstance(result, str)

    @patch("agraph.processor.spreadsheet_processor.pd")
    def test_process_excel_all_sheets(self, mock_pd):
        """Test processing all sheets in Excel file."""
        test_file = Path(self.temp_dir) / "multi_sheet.xlsx"
        test_file.write_bytes(b"fake excel with multiple sheets")

        # Mock ExcelFile
        mock_excel_file = MagicMock()
        mock_excel_file.sheet_names = ["Sheet1", "Sheet2"]
        mock_pd.ExcelFile.return_value = mock_excel_file

        # Mock DataFrames for each sheet
        mock_df = MagicMock()
        mock_df.empty = False
        mock_df.astype.return_value.replace.return_value = mock_df
        mock_df.columns = ["Data"]
        mock_df.iterrows.return_value = [(0, ["Value"])]
        mock_pd.read_excel.return_value = mock_df

        result = self.processor.process(test_file, sheet_name="all")

        self.assertIn("Sheet1", result)
        self.assertIn("Sheet2", result)

    def test_process_excel_without_pandas(self):
        """Test processing Excel when pandas is not available."""
        test_file = Path(self.temp_dir) / "test.xlsx"
        test_file.write_bytes(b"fake excel")

        with patch.dict("sys.modules", {"pandas": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with self.assertRaises(ProcessingError) as context:
                    self.processor.process(test_file)

                self.assertIn("pandas", str(context.exception))

    def test_format_table_rows(self):
        """Test table row formatting with alignment."""
        rows = [["Name", "Age", "City"], ["John", "30", "New York"], ["Jane", "25", "Boston"]]

        result = self.processor._format_table_rows(rows)

        lines = result.split("\n")
        self.assertEqual(len(lines), 3)
        # Check that columns are aligned
        self.assertIn(" | ", result)

    def test_format_table_rows_empty(self):
        """Test formatting empty table."""
        result = self.processor._format_table_rows([])
        self.assertEqual(result, "")

    def test_dataframe_to_formatted_text(self):
        """Test DataFrame to formatted text conversion."""
        with patch("agraph.processor.spreadsheet_processor.pd") as mock_pd:
            # Create mock DataFrame
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.astype.return_value.replace.return_value = mock_df
            mock_df.columns.tolist.return_value = ["Col1", "Col2"]
            mock_df.iterrows.return_value = [(0, ["Val1", "Val2"])]

            result = self.processor._dataframe_to_formatted_text(mock_df, include_headers=True)

            self.assertIsInstance(result, str)

    def test_dataframe_to_formatted_text_empty(self):
        """Test DataFrame to text conversion with empty DataFrame."""
        with patch("agraph.processor.spreadsheet_processor.pd") as mock_pd:
            mock_df = MagicMock()
            mock_df.empty = True

            result = self.processor._dataframe_to_formatted_text(mock_df)

            self.assertEqual(result, "")

    def test_analyze_csv_metadata(self):
        """Test CSV metadata analysis."""
        test_file = Path(self.temp_dir) / "data.csv"
        csv_content = "Name,Age,City\nJohn,30,NYC\nJane,25,Boston"
        test_file.write_text(csv_content)

        metadata = self.processor._analyze_csv_metadata(test_file)

        self.assertEqual(metadata["format"], "csv")
        self.assertEqual(metadata["row_count"], 3)  # Including header
        self.assertEqual(metadata["column_count"], 3)
        self.assertEqual(metadata["detected_delimiter"], ",")

    def test_analyze_csv_metadata_empty_file(self):
        """Test CSV metadata analysis with empty file."""
        test_file = Path(self.temp_dir) / "empty.csv"
        test_file.write_text("")

        metadata = self.processor._analyze_csv_metadata(test_file)

        self.assertIn("content_analysis_error", metadata)

    @patch("agraph.processor.spreadsheet_processor.pd")
    def test_analyze_excel_metadata(self, mock_pd):
        """Test Excel metadata analysis."""
        test_file = Path(self.temp_dir) / "test.xlsx"
        test_file.write_bytes(b"fake excel")

        # Mock ExcelFile
        mock_excel_file = MagicMock()
        mock_excel_file.sheet_names = ["Sheet1", "Data"]
        mock_pd.ExcelFile.return_value = mock_excel_file

        # Mock DataFrame for sheets
        mock_df = MagicMock()
        mock_df.__len__.return_value = 10  # 10 rows
        mock_df.columns.__len__.return_value = 3  # 3 columns
        mock_df.columns.tolist.return_value = ["A", "B", "C"]
        mock_df.empty = False
        mock_df.dtypes.to_dict.return_value = {"A": "object", "B": "int64"}
        mock_pd.read_excel.return_value = mock_df

        metadata = self.processor._analyze_excel_metadata(test_file)

        self.assertEqual(metadata["format"], "excel")
        self.assertEqual(metadata["sheet_count"], 2)
        self.assertEqual(metadata["sheet_names"], ["Sheet1", "Data"])
        self.assertIn("sheets_info", metadata)

    def test_analyze_excel_metadata_without_pandas(self):
        """Test Excel metadata analysis when pandas is not available."""
        test_file = Path(self.temp_dir) / "test.xlsx"
        test_file.write_bytes(b"fake excel")

        with patch.dict("sys.modules", {"pandas": None}):
            with patch("builtins.__import__", side_effect=ImportError):
                with self.assertRaises(ProcessingError):
                    self.processor._analyze_excel_metadata(test_file)

    def test_process_csv_encoding_fallback(self):
        """Test CSV processing with encoding fallback."""
        test_file = Path(self.temp_dir) / "encoded.csv"
        content = "Name,City\nJohn,Café"
        test_file.write_text(content, encoding="latin-1")

        # Should fallback to latin-1 when utf-8 fails
        result = self.processor.process(test_file)

        self.assertIn("Café", result)

    def test_process_csv_unsupported_encoding(self):
        """Test CSV processing with unsupported encoding."""
        test_file = Path(self.temp_dir) / "binary.csv"
        test_file.write_bytes(b"\x80\x81\x82\x83,value")

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("decode", str(context.exception).lower())

    def test_process_empty_csv_file(self):
        """Test processing empty CSV file."""
        test_file = Path(self.temp_dir) / "empty.csv"
        test_file.write_text("")

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_csv_with_only_whitespace(self):
        """Test processing CSV with only whitespace."""
        test_file = Path(self.temp_dir) / "whitespace.csv"
        test_file.write_text("   \n  \n  ")

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_empty_file_raises_error(self):
        """Test that processing empty file raises appropriate error."""
        empty_file = Path(self.temp_dir) / "empty.csv"
        empty_file.touch()

        with self.assertRaises(ValueError) as context:
            self.processor.process(empty_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_nonexistent_file_raises_error(self):
        """Test that processing non-existent file raises appropriate error."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.csv"

        with self.assertRaises(FileNotFoundError):
            self.processor.process(nonexistent_file)

    def test_process_unsupported_extension(self):
        """Test processing file with unsupported extension."""
        test_file = Path(self.temp_dir) / "test.ods"
        test_file.write_bytes(b"fake ods content")

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("Unsupported file extension", str(context.exception))

    @patch("agraph.processor.spreadsheet_processor.pd")
    def test_process_excel_with_error(self, mock_pd):
        """Test Excel processing with pandas error."""
        test_file = Path(self.temp_dir) / "corrupted.xlsx"
        test_file.write_bytes(b"fake corrupted excel")

        mock_pd.read_excel.side_effect = Exception("Excel read error")

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("Failed to process Excel", str(context.exception))

    @patch("agraph.processor.spreadsheet_processor.pd")
    def test_process_all_sheets_with_empty_sheet(self, mock_pd):
        """Test processing all sheets including empty ones."""
        test_file = Path(self.temp_dir) / "mixed_sheets.xlsx"
        test_file.write_bytes(b"fake excel")

        # Mock ExcelFile
        mock_excel_file = MagicMock()
        mock_excel_file.sheet_names = ["Data", "Empty"]
        mock_pd.ExcelFile.return_value = mock_excel_file

        # Mock DataFrames - one with data, one empty
        def mock_read_excel(file_path, sheet_name, nrows):
            if sheet_name == "Data":
                mock_df = MagicMock()
                mock_df.empty = False
                mock_df.astype.return_value.replace.return_value = mock_df
                mock_df.columns.tolist.return_value = ["Col1"]
                mock_df.iterrows.return_value = [(0, ["Value"])]
                return mock_df
            else:  # Empty sheet
                mock_df = MagicMock()
                mock_df.empty = True
                return mock_df

        mock_pd.read_excel.side_effect = mock_read_excel

        result = self.processor.process(test_file, sheet_name="all")

        self.assertIn("Sheet: Data", result)
        self.assertNotIn("Sheet: Empty", result)  # Empty sheets should be skipped

    def test_format_table_rows_with_different_lengths(self):
        """Test table formatting with rows of different lengths."""
        rows = [
            ["Short", "Medium length", "Long column header"],
            ["A", "B", "C"],
            ["Very long content", "X", "Y"],
        ]

        result = self.processor._format_table_rows(rows)

        # All columns should be properly aligned
        lines = result.split("\n")
        self.assertEqual(len(lines), 3)
        # Check that formatting preserves structure
        self.assertIn(" | ", result)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
