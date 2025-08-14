"""Spreadsheet document processor implementation.

This module provides functionality for extracting text and metadata from CSV files
and Excel spreadsheets. It supports various formatting options, multiple sheets,
and comprehensive data analysis.
"""

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

from .base import DocumentProcessor, ProcessingError

if TYPE_CHECKING:
    import pandas as pd


class SpreadsheetProcessor(DocumentProcessor):
    """Document processor for CSV and Excel spreadsheet files.

    This processor handles multiple spreadsheet formats with flexible text extraction:
    - CSV files with automatic delimiter detection
    - Excel files (.xlsx, .xls) with multi-sheet support
    - Configurable formatting and row limits
    - Comprehensive metadata extraction
    - Data structure analysis

    Features:
    - Automatic CSV delimiter detection
    - Multi-sheet Excel processing
    - Configurable row limits for large files
    - Formatted text output with column alignment
    - Header inclusion/exclusion options
    - Encoding fallback for CSV files

    Dependencies:
        pandas: Required for Excel file processing (.xlsx, .xls)
        openpyxl: Required for .xlsx files (installed with pandas[excel])
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported spreadsheet file extensions.

        Returns:
            List containing '.csv', '.xlsx', and '.xls' extensions.
        """
        return [".csv", ".xlsx", ".xls"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text content from spreadsheet files.

        This method processes various spreadsheet formats and returns formatted
        text that preserves the tabular structure in a readable format.

        Args:
            file_path: Path to the spreadsheet file to process.
            **kwargs: Additional processing parameters:
                - sheet_name (str|int|'all'): For Excel files, specify sheet name,
                  index, or 'all' for all sheets (default: 0 for first sheet)
                - max_rows (int): Maximum number of rows to process (default: 1000)
                - include_headers (bool): Whether to include column headers (default: True)
                - delimiter (str): CSV delimiter override (default: auto-detect)
                - encoding (str): Text encoding for CSV files (default: 'utf-8')

        Returns:
            Formatted text content preserving tabular structure with proper alignment.

        Raises:
            ProcessingError: If required dependencies are missing, file format
                           is unsupported, or processing fails.
        """
        self.validate_file(file_path)
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            return self._process_csv_file(file_path, **kwargs)
        if file_path.suffix.lower() in [".xlsx", ".xls"]:
            return self._process_excel_file(file_path, **kwargs)
        raise ProcessingError(f"Unsupported file extension: {file_path.suffix}")

    def _process_csv_file(
        self, file_path: Path, **kwargs: Any
    ) -> str:  # pylint: disable=too-many-locals
        """Process CSV files with automatic delimiter detection and encoding fallback.

        Args:
            file_path: Path to the CSV file.
            **kwargs: Processing parameters.

        Returns:
            Formatted CSV content as text.

        Raises:
            ProcessingError: If file cannot be processed or decoded.
        """
        config = {
            "encoding": kwargs.get("encoding", "utf-8"),
            "delimiter": kwargs.get("delimiter", ","),
            "max_rows": kwargs.get("max_rows", 1000),
            "include_headers": kwargs.get("include_headers", True),
        }

        try:
            with open(file_path, "r", encoding=config["encoding"]) as file:
                # Check if file is empty
                file_content = file.read()
                if not file_content.strip():
                    raise ValueError("File is empty")
                file.seek(0)

                # Auto-detect delimiter if not specified or using default
                if config["delimiter"] == ",":
                    config["delimiter"] = self._detect_csv_delimiter(file)

                reader = csv.reader(file, delimiter=config["delimiter"])
                rows = []

                # Read headers
                try:
                    headers = next(reader)
                    if config["include_headers"]:
                        rows.append(headers)
                except StopIteration:
                    return ""

                # Read data rows with limit
                for i, row in enumerate(reader):
                    if config["max_rows"] and i >= config["max_rows"]:
                        break
                    rows.append(row)

                return self._format_table_rows(rows) if rows else ""

        except UnicodeDecodeError as exc:
            # Try alternative encodings for CSV files
            if config["encoding"] == "utf-8":
                for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        return self._process_csv_file(file_path, encoding=alt_encoding, **kwargs)
                    except UnicodeDecodeError:
                        continue
            raise ProcessingError(
                f"Could not decode CSV file {file_path} with any supported encoding"
            ) from exc
        except Exception as e:
            raise ProcessingError(f"Failed to process CSV file {file_path}: {str(e)}") from e

    def _detect_csv_delimiter(self, file: Any) -> str:
        """Detect the delimiter used in a CSV file.

        Args:
            file: Open file object.

        Returns:
            Detected delimiter character.
        """
        sample = file.read(1024)
        file.seek(0)

        sniffer = csv.Sniffer()
        try:
            dialect = sniffer.sniff(sample)
            return dialect.delimiter
        except csv.Error:
            # Fallback to comma if detection fails
            return ","

    def _process_excel_file(self, file_path: Path, **kwargs: Any) -> str:
        """Process Excel files with multi-sheet support.

        Args:
            file_path: Path to the Excel file.
            **kwargs: Processing parameters.

        Returns:
            Formatted Excel content as text.

        Raises:
            ProcessingError: If pandas is not available or processing fails.
        """
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "pandas is required for Excel processing. Install with: pip install pandas openpyxl"
            ) from exc

        sheet_name = kwargs.get("sheet_name", 0)  # Default to first sheet
        max_rows = kwargs.get("max_rows", 1000)
        include_headers = kwargs.get("include_headers", True)

        try:
            if sheet_name == "all":
                return self._process_all_excel_sheets(file_path, max_rows, include_headers)
            df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=max_rows)
            return self._dataframe_to_formatted_text(df, include_headers)

        except Exception as e:
            raise ProcessingError(f"Failed to process Excel file {file_path}: {str(e)}") from e

    def _process_all_excel_sheets(
        self, file_path: Path, max_rows: int, include_headers: bool
    ) -> str:
        """Process all sheets in an Excel file.

        Args:
            file_path: Path to the Excel file.
            max_rows: Maximum rows per sheet.
            include_headers: Whether to include headers.

        Returns:
            Formatted content from all sheets.
        """
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "pandas is required for Excel processing. Install with: pip install pandas"
            ) from exc

        excel_file = pd.ExcelFile(file_path)
        all_content = []

        for sheet in excel_file.sheet_names:
            df = pd.read_excel(file_path, sheet_name=sheet, nrows=max_rows)
            if not df.empty:
                all_content.append(f"Sheet: {sheet}")
                all_content.append(self._dataframe_to_formatted_text(df, include_headers))
                all_content.append("")  # Empty line between sheets

        return "\n".join(all_content)

    def _dataframe_to_formatted_text(self, df: "pd.DataFrame", include_headers: bool = True) -> str:
        """Convert DataFrame to well-formatted text with proper column alignment.

        Args:
            df: Pandas DataFrame to convert.
            include_headers: Whether to include column headers.

        Returns:
            Formatted text representation of the DataFrame.
        """
        if df.empty:
            return ""

        # Convert all values to strings and handle NaN values
        df_str = df.astype(str).replace("nan", "")

        rows = []

        # Add headers if requested
        if include_headers:
            rows.append(df_str.columns.tolist())

        # Add data rows
        for _, row in df_str.iterrows():
            rows.append(row.tolist())

        return self._format_table_rows(rows)

    def _format_table_rows(self, rows: List[List[str]]) -> str:
        """Format table rows with proper column alignment.

        Args:
            rows: List of rows, each row is a list of cell values.

        Returns:
            Formatted table as string with aligned columns.
        """
        if not rows:
            return ""

        # Calculate column widths for formatting
        max_cols = max(len(row) for row in rows) if rows else 0
        col_widths = [0] * max_cols

        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Format rows with proper spacing
        formatted_rows = []
        for row in rows:
            formatted_cells = []
            for i, cell in enumerate(row):
                width = col_widths[i] if i < len(col_widths) else 0
                formatted_cells.append(str(cell).ljust(width))
            formatted_rows.append(" | ".join(formatted_cells))

        return "\n".join(formatted_rows)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract comprehensive metadata from spreadsheet files.

        This method analyzes spreadsheet structure, content, and provides detailed
        information about the data organization and format.

        Args:
            file_path: Path to the spreadsheet file.

        Returns:
            Dictionary containing metadata with keys:
            - file_path: Original file path
            - file_size: File size in bytes
            - file_type: File extension
            - created/modified: File timestamps
            - CSV specific:
              - row_count, column_count: Data dimensions
            - Excel specific:
              - sheet_names, sheet_count: Sheet information
              - sheets_info: Per-sheet analysis with row/column counts
            - content_analysis_error: Error message if analysis fails
        """
        self.validate_file(file_path)
        file_path = Path(file_path)

        stat = file_path.stat()
        metadata = {
            "file_path": str(file_path),
            "file_size": stat.st_size,
            "file_type": file_path.suffix.lower(),
            "created": str(stat.st_ctime),
            "modified": str(stat.st_mtime),
        }

        if file_path.suffix.lower() == ".csv":
            metadata.update(self._analyze_csv_metadata(file_path))
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            metadata.update(self._analyze_excel_metadata(file_path))

        return metadata

    def _analyze_csv_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Analyze metadata for CSV files.

        Args:
            file_path: Path to the CSV file.

        Returns:
            Dictionary with CSV-specific metadata.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                # Detect delimiter
                delimiter = self._detect_csv_delimiter(file)

                reader = csv.reader(file, delimiter=delimiter)
                row_count = sum(1 for _ in reader)
                file.seek(0)

                # Get column count from first row
                try:
                    first_row = next(reader)
                    col_count = len(first_row)
                except StopIteration:
                    col_count = 0

                return {
                    "format": "csv",
                    "row_count": row_count,
                    "column_count": col_count,
                    "detected_delimiter": delimiter,
                }

        except (OSError, ValueError, UnicodeDecodeError) as e:
            return {"content_analysis_error": str(e)}

    def _analyze_excel_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Analyze metadata for Excel files.

        Args:
            file_path: Path to the Excel file.

        Returns:
            Dictionary with Excel-specific metadata.
        """
        try:
            import pandas as pd  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError(
                "pandas is required for Excel processing. Install with: pip install pandas openpyxl"
            ) from exc

        excel_file = pd.ExcelFile(file_path)

        metadata = {
            "format": "excel",
            "sheet_names": excel_file.sheet_names,
            "sheet_count": len(excel_file.sheet_names),
        }

        # Get detailed info for each sheet
        sheet_info = {}
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                sheet_info[sheet_name] = {
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "columns": df.columns.tolist(),
                    "has_data": not df.empty,
                }

                # Add data type analysis for non-empty sheets
                if not df.empty:
                    sheet_info[sheet_name]["column_types"] = df.dtypes.to_dict()

            except (ValueError, KeyError, OSError) as e:
                sheet_info[sheet_name] = {"error": str(e)}

        metadata["sheets_info"] = sheet_info
        return metadata
