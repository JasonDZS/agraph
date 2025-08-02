"""Tests for spreadsheet document processor."""

import csv
from unittest.mock import Mock, patch

import pytest

from agraph.processer.base import ProcessingError
from agraph.processer.spreadsheet_processor import SpreadsheetProcessor
from tests.processer.conftest import skip_if_no_module


class TestSpreadsheetProcessor:
    """Test SpreadsheetProcessor functionality."""

    def test_supported_extensions(self):
        """Test that SpreadsheetProcessor supports correct extensions."""
        processor = SpreadsheetProcessor()
        expected_extensions = [".csv", ".xlsx", ".xls"]
        assert processor.supported_extensions == expected_extensions

    def test_process_csv_file_default(self, sample_csv_file):
        """Test processing CSV file with default settings."""
        processor = SpreadsheetProcessor()
        result = processor.process(sample_csv_file)

        # Should contain header and data
        assert "Name" in result
        assert "Age" in result
        assert "City" in result
        assert "Occupation" in result
        assert "John Doe" in result
        assert "Jane Smith" in result

        # Should be formatted with proper spacing
        lines = result.split("\n")
        assert len(lines) >= 4  # Header + 4 data rows

    def test_process_csv_file_no_headers(self, sample_csv_file):
        """Test processing CSV file without including headers."""
        processor = SpreadsheetProcessor()
        result = processor.process(sample_csv_file, include_headers=False)

        # Should not contain headers
        assert "Name" not in result or result.count("Name") == 0
        # Should contain data
        assert "John Doe" in result

    def test_process_csv_file_max_rows(self, sample_csv_file):
        """Test processing CSV file with row limit."""
        processor = SpreadsheetProcessor()
        result = processor.process(sample_csv_file, max_rows=2)

        lines = result.split("\n")
        # Should have header + 2 rows = 3 lines
        assert len([line for line in lines if line.strip()]) == 3

    def test_process_csv_file_custom_delimiter(self, temp_dir):
        """Test processing CSV file with custom delimiter."""
        # Create CSV with semicolon delimiter
        semicolon_csv = temp_dir / "semicolon.csv"
        content = "Name;Age;City\nJohn;30;New York\nJane;25;London"
        semicolon_csv.write_text(content)

        processor = SpreadsheetProcessor()
        result = processor.process(semicolon_csv, delimiter=";")

        assert "John" in result
        assert "30" in result
        assert "New York" in result

    def test_process_csv_auto_detect_delimiter(self, temp_dir):
        """Test CSV delimiter auto-detection."""
        # Create CSV with tab delimiter
        tab_csv = temp_dir / "tab.csv"
        content = "Name\tAge\tCity\nJohn\t30\tNew York\nJane\t25\tLondon"
        tab_csv.write_text(content)

        processor = SpreadsheetProcessor()
        result = processor.process(tab_csv)

        assert "John" in result
        assert "30" in result
        assert "New York" in result

    def test_process_csv_encoding_fallback(self, temp_dir):
        """Test CSV processing with encoding fallback."""
        # Create CSV with latin-1 encoding
        latin_csv = temp_dir / "latin.csv"
        content = "Name,City\nJean,Paris\nCafé,Naïve"
        latin_csv.write_bytes(content.encode("latin-1"))

        processor = SpreadsheetProcessor()
        result = processor.process(latin_csv)

        assert "Jean" in result
        assert "Paris" in result

    def test_process_csv_unsupported_encoding(self, temp_dir):
        """Test CSV processing with unsupported encoding."""
        # Create file that can't be decoded
        binary_csv = temp_dir / "binary.csv"
        binary_csv.write_bytes(b"\x80\x81\x82\x83")

        processor = SpreadsheetProcessor()

        # Mock open to raise UnicodeDecodeError for all encoding attempts
        original_open = __builtins__['open']
        def mock_open(*args, **kwargs):
            if 'encoding' in kwargs:
                encoding = kwargs['encoding']
                raise UnicodeDecodeError(encoding, b'', 0, 1, f'mock error for {encoding}')
            return original_open(*args, **kwargs)

        with patch("builtins.open", side_effect=mock_open):
            with pytest.raises(ProcessingError, match="Could not decode CSV file"):
                processor.process(binary_csv)

    def test_process_csv_read_error(self, temp_dir):
        """Test CSV processing with read error."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("Name,Age\nJohn,30")

        processor = SpreadsheetProcessor()

        # Mock open to raise an exception
        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            with pytest.raises(ProcessingError, match="Failed to process CSV file"):
                processor.process(csv_file)

    def test_process_empty_csv(self, temp_dir):
        """Test processing empty CSV file."""
        empty_csv = temp_dir / "empty.csv"
        empty_csv.write_text("")

        processor = SpreadsheetProcessor()
        with pytest.raises(ValueError, match="File is empty"):
            processor.process(empty_csv)

    @pytest.mark.skipif(skip_if_no_module("pandas"), reason="pandas not available")
    def test_process_excel_file_default(self, temp_dir):
        """Test processing Excel file with default settings."""
        excel_file = temp_dir / "test.xlsx"
        excel_file.write_bytes(b"dummy excel content")

        processor = SpreadsheetProcessor()

        # Mock pandas DataFrame properly
        mock_df = Mock()
        mock_df.empty = False

        # Create a proper mock for the processed dataframe
        mock_processed_df = Mock()
        mock_processed_df.columns = Mock()
        mock_processed_df.columns.tolist.return_value = ["Name", "Age"]
        mock_processed_df.iterrows.return_value = [
            (0, Mock(tolist=Mock(return_value=["John Doe", "30"]))),
            (1, Mock(tolist=Mock(return_value=["Jane Smith", "25"])))
        ]

        # Chain the mocks properly
        mock_df.astype.return_value.replace.return_value = mock_processed_df

        with patch("pandas.read_excel", return_value=mock_df):
            result = processor.process(excel_file)

            assert result is not None
            assert "Name" in result
            assert "John Doe" in result

    @pytest.mark.skipif(skip_if_no_module("pandas"), reason="pandas not available")
    def test_process_excel_file_specific_sheet(self, temp_dir):
        """Test processing specific sheet from Excel file."""
        excel_file = temp_dir / "test.xlsx"
        excel_file.write_bytes(b"dummy excel content")

        processor = SpreadsheetProcessor()

        mock_df = Mock()
        mock_df.empty = False

        # Create a proper mock for the processed dataframe
        mock_processed_df = Mock()
        mock_processed_df.columns = Mock()
        mock_processed_df.columns.tolist.return_value = ["Name", "Age"]
        mock_processed_df.iterrows.return_value = [
            (0, Mock(tolist=Mock(return_value=["John Doe", "30"])))
        ]

        # Chain the mocks properly
        mock_df.astype.return_value.replace.return_value = mock_processed_df

        with patch("pandas.read_excel", return_value=mock_df) as mock_read:
            processor.process(excel_file, sheet_name="Sheet2")

            mock_read.assert_called_with(excel_file, sheet_name="Sheet2", nrows=1000)

    @pytest.mark.skipif(skip_if_no_module("pandas"), reason="pandas not available")
    def test_process_excel_file_all_sheets(self, temp_dir):
        """Test processing all sheets from Excel file."""
        excel_file = temp_dir / "test.xlsx"
        excel_file.write_bytes(b"dummy excel content")

        processor = SpreadsheetProcessor()

        # Mock ExcelFile
        mock_excel_file = Mock()
        mock_excel_file.sheet_names = ["Sheet1", "Sheet2"]

        # Mock DataFrames
        mock_df1 = Mock()
        mock_df1.empty = False
        mock_df2 = Mock()
        mock_df2.empty = False

        with patch("pandas.ExcelFile", return_value=mock_excel_file):
            with patch("pandas.read_excel", side_effect=[mock_df1, mock_df2]):
                with patch.object(processor, "_dataframe_to_text", return_value="Sheet content"):
                    result = processor.process(excel_file, sheet_name="all")

                    assert "Sheet: Sheet1" in result
                    assert "Sheet: Sheet2" in result

    @pytest.mark.skipif(skip_if_no_module("pandas"), reason="pandas not available")
    def test_process_excel_empty_dataframe(self, temp_dir):
        """Test processing Excel file with empty DataFrame."""
        excel_file = temp_dir / "empty.xlsx"
        excel_file.write_bytes(b"dummy excel content")

        processor = SpreadsheetProcessor()

        mock_df = Mock()
        mock_df.empty = True

        with patch("pandas.read_excel", return_value=mock_df):
            result = processor.process(excel_file)

            assert result == ""

    def test_process_excel_no_pandas(self, temp_dir):
        """Test processing Excel file when pandas is not available."""
        excel_file = temp_dir / "test.xlsx"
        excel_file.write_bytes(b"dummy excel content")

        processor = SpreadsheetProcessor()

        with patch.dict("sys.modules", {"pandas": None}):
            with pytest.raises(ProcessingError, match="pandas is required"):
                processor.process(excel_file)

    @pytest.mark.skipif(skip_if_no_module("pandas"), reason="pandas not available")
    def test_process_excel_read_error(self, temp_dir):
        """Test Excel processing with read error."""
        excel_file = temp_dir / "corrupted.xlsx"
        excel_file.write_bytes(b"corrupted content")

        processor = SpreadsheetProcessor()

        with patch("pandas.read_excel", side_effect=Exception("Read error")):
            with pytest.raises(ProcessingError, match="Failed to process Excel file"):
                processor.process(excel_file)

    def test_process_unsupported_extension(self, temp_dir):
        """Test processing file with unsupported extension."""
        txt_file = temp_dir / "test.txt"
        txt_file.write_text("plain text")

        processor = SpreadsheetProcessor()

        with pytest.raises(ProcessingError, match="Unsupported file extension"):
            processor.process(txt_file)

    def test_dataframe_to_text_with_headers(self):
        """Test converting DataFrame to text with headers."""
        processor = SpreadsheetProcessor()

        # Mock pandas DataFrame
        try:
            import pandas as pd
            df = pd.DataFrame({
                "Name": ["John", "Jane"],
                "Age": [30, 25],
                "City": ["NY", "LA"]
            })

            result = processor._dataframe_to_text(df, include_headers=True)

            assert "Name" in result
            assert "Age" in result
            assert "John" in result
            assert "Jane" in result
        except ImportError:
            pytest.skip("pandas not available")

    def test_dataframe_to_text_without_headers(self):
        """Test converting DataFrame to text without headers."""
        processor = SpreadsheetProcessor()

        try:
            import pandas as pd
            df = pd.DataFrame({
                "Name": ["John", "Jane"],
                "Age": [30, 25]
            })

            result = processor._dataframe_to_text(df, include_headers=False)

            # Should not contain column names as first row
            lines = result.split("\n")
            first_line = lines[0]
            assert "Name" not in first_line
            assert "John" in first_line
        except ImportError:
            pytest.skip("pandas not available")

    def test_dataframe_to_text_empty(self):
        """Test converting empty DataFrame to text."""
        processor = SpreadsheetProcessor()

        try:
            import pandas as pd
            df = pd.DataFrame()

            result = processor._dataframe_to_text(df)

            assert result == ""
        except ImportError:
            pytest.skip("pandas not available")

    def test_extract_metadata_csv(self, sample_csv_file):
        """Test extracting metadata from CSV file."""
        processor = SpreadsheetProcessor()
        metadata = processor.extract_metadata(sample_csv_file)

        assert metadata["file_path"] == str(sample_csv_file)
        assert metadata["file_type"] == ".csv"
        assert "file_size" in metadata
        assert "row_count" in metadata
        assert "column_count" in metadata

        # Should have 5 rows (header + 4 data rows) and 4 columns
        assert metadata["row_count"] == 5
        assert metadata["column_count"] == 4

    @pytest.mark.skipif(skip_if_no_module("pandas"), reason="pandas not available")
    def test_extract_metadata_excel(self, temp_dir):
        """Test extracting metadata from Excel file."""
        excel_file = temp_dir / "test.xlsx"
        excel_file.write_bytes(b"dummy excel content")

        processor = SpreadsheetProcessor()

        # Mock ExcelFile and DataFrame
        mock_excel_file = Mock()
        mock_excel_file.sheet_names = ["Sheet1", "Sheet2"]

        mock_df = Mock()
        mock_df.__len__ = Mock(return_value=10)  # 10 rows

        # Create a proper mock for columns
        mock_columns = Mock()
        mock_columns.tolist.return_value = ["Name", "Age", "City"]
        mock_columns.__len__ = Mock(return_value=3)  # 3 columns
        mock_df.columns = mock_columns

        with patch("pandas.ExcelFile", return_value=mock_excel_file):
            with patch("pandas.read_excel", return_value=mock_df):
                metadata = processor.extract_metadata(excel_file)

                assert metadata["sheet_names"] == ["Sheet1", "Sheet2"]
                assert metadata["sheet_count"] == 2
                assert "sheets_info" in metadata

                sheet_info = metadata["sheets_info"]["Sheet1"]
                assert sheet_info["row_count"] == 10
                assert sheet_info["column_count"] == 3
                assert sheet_info["columns"] == ["Name", "Age", "City"]

    def test_extract_metadata_excel_no_pandas(self, temp_dir):
        """Test metadata extraction when pandas is not available."""
        excel_file = temp_dir / "test.xlsx"
        excel_file.write_bytes(b"dummy excel content")

        processor = SpreadsheetProcessor()

        with patch.dict("sys.modules", {"pandas": None}):
            metadata = processor.extract_metadata(excel_file)

            assert "analysis_error" in metadata
            assert "pandas not available" in metadata["analysis_error"]

    @pytest.mark.skipif(skip_if_no_module("pandas"), reason="pandas not available")
    def test_extract_metadata_excel_read_error(self, temp_dir):
        """Test metadata extraction handles Excel read errors."""
        excel_file = temp_dir / "corrupted.xlsx"
        excel_file.write_bytes(b"corrupted content")

        processor = SpreadsheetProcessor()

        with patch("pandas.ExcelFile", side_effect=Exception("Read error")):
            metadata = processor.extract_metadata(excel_file)

            assert "content_analysis_error" in metadata

    def test_extract_metadata_csv_analysis_error(self, temp_dir):
        """Test CSV metadata extraction handles analysis errors."""
        csv_file = temp_dir / "test.csv"
        csv_file.write_text("Name,Age\nJohn,30")

        processor = SpreadsheetProcessor()

        with patch("builtins.open", side_effect=PermissionError("Access denied")):
            metadata = processor.extract_metadata(csv_file)

            assert "content_analysis_error" in metadata
