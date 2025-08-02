import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Union

if TYPE_CHECKING:
    import pandas as pd

from .base import DocumentProcessor, ProcessingError


class SpreadsheetProcessor(DocumentProcessor):
    """Processor for CSV and Excel spreadsheet files."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".csv", ".xlsx", ".xls"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text from spreadsheet file.

        Args:
            file_path: Path to the spreadsheet file
            **kwargs: Additional parameters (sheet_name, max_rows, separator, etc.)

        Returns:
            Extracted content as formatted text
        """
        self.validate_file(file_path)
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".csv":
            return self._process_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            return self._process_excel(file_path, **kwargs)
        else:
            raise ProcessingError(f"Unsupported file extension: {file_path.suffix}")

    def _process_csv(self, file_path: Path, **kwargs: Any) -> str:
        """Process CSV files."""
        encoding = kwargs.get("encoding", "utf-8")
        delimiter = kwargs.get("delimiter", ",")
        max_rows = kwargs.get("max_rows", 1000)
        include_headers = kwargs.get("include_headers", True)

        try:
            with open(file_path, "r", encoding=encoding) as file:
                # Check if file is empty
                file_content = file.read()
                if not file_content.strip():
                    raise ValueError("File is empty")
                file.seek(0)

                # Auto-detect delimiter if not specified
                if delimiter == ",":
                    sample = file.read(1024)
                    file.seek(0)
                    sniffer = csv.Sniffer()
                    try:
                        delimiter = sniffer.sniff(sample).delimiter
                    except csv.Error:
                        delimiter = ","

                reader = csv.reader(file, delimiter=delimiter)
                rows = []

                # Read headers
                try:
                    headers = next(reader)
                    if include_headers:
                        rows.append(headers)
                except StopIteration:
                    return ""

                # Read data rows
                for i, row in enumerate(reader):
                    if max_rows and i >= max_rows:
                        break
                    rows.append(row)

                # Format as text
                if not rows:
                    return ""

                # Calculate column widths for formatting
                col_widths = [0] * len(rows[0])
                for row in rows:
                    for i, cell in enumerate(row):
                        if i < len(col_widths):
                            col_widths[i] = max(col_widths[i], len(str(cell)))

                # Format rows
                formatted_rows = []
                for row in rows:
                    formatted_cells = []
                    for i, cell in enumerate(row):
                        width = col_widths[i] if i < len(col_widths) else 0
                        formatted_cells.append(str(cell).ljust(width))
                    formatted_rows.append(" | ".join(formatted_cells))

                return "\n".join(formatted_rows)

        except UnicodeDecodeError:
            # Try alternative encodings only if we haven't already tried them
            if encoding == "utf-8":
                for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        return self._process_csv(file_path, encoding=alt_encoding, **kwargs)
                    except UnicodeDecodeError:
                        continue
            raise ProcessingError(f"Could not decode CSV file {file_path} with any supported encoding")
        except Exception as e:
            raise ProcessingError(f"Failed to process CSV file {file_path}: {str(e)}")

    def _process_excel(self, file_path: Path, **kwargs: Any) -> str:
        """Process Excel files."""
        try:
            import pandas as pd
        except ImportError:
            raise ProcessingError("pandas is required for Excel processing. Install with: pip install pandas openpyxl")

        sheet_name = kwargs.get("sheet_name", 0)  # Default to first sheet
        max_rows = kwargs.get("max_rows", 1000)
        include_headers = kwargs.get("include_headers", True)

        try:
            # Read Excel file
            if sheet_name == "all":
                # Read all sheets
                excel_file = pd.ExcelFile(file_path)
                all_content = []
                for sheet in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet, nrows=max_rows)
                    if not df.empty:
                        all_content.append(f"Sheet: {sheet}")
                        all_content.append(self._dataframe_to_text(df, include_headers))
                        all_content.append("")  # Empty line between sheets
                return "\n".join(all_content)
            else:
                df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=max_rows)
                return self._dataframe_to_text(df, include_headers)

        except Exception as e:
            raise ProcessingError(f"Failed to process Excel file {file_path}: {str(e)}")

    def _dataframe_to_text(self, df: "pd.DataFrame", include_headers: bool = True) -> str:
        """Convert DataFrame to formatted text."""
        if df.empty:
            return ""

        # Convert all values to strings and handle NaN
        df_str = df.astype(str).replace("nan", "")

        rows = []

        # Add headers if requested
        if include_headers:
            rows.append(df_str.columns.tolist())

        # Add data rows
        for _, row in df_str.iterrows():
            rows.append(row.tolist())

        if not rows:
            return ""

        # Calculate column widths for formatting
        col_widths = [0] * len(rows[0])
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(str(cell)))

        # Format rows
        formatted_rows = []
        for row in rows:
            formatted_cells = []
            for i, cell in enumerate(row):
                width = col_widths[i] if i < len(col_widths) else 0
                formatted_cells.append(str(cell).ljust(width))
            formatted_rows.append(" | ".join(formatted_cells))

        return "\n".join(formatted_rows)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from spreadsheet file.

        Args:
            file_path: Path to the spreadsheet file

        Returns:
            Dictionary containing metadata
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
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    reader = csv.reader(file)
                    row_count = sum(1 for _ in reader)
                    file.seek(0)

                    # Get column count from first row
                    try:
                        first_row = next(reader)
                        col_count = len(first_row)
                    except StopIteration:
                        col_count = 0

                    metadata.update(
                        {
                            "row_count": row_count,
                            "column_count": col_count,
                        }
                    )

            except Exception as e:
                metadata["content_analysis_error"] = str(e)

        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            try:
                import pandas as pd

                excel_file = pd.ExcelFile(file_path)

                metadata.update(
                    {
                        "sheet_names": excel_file.sheet_names,
                        "sheet_count": len(excel_file.sheet_names),
                    }
                )

                # Get info for each sheet
                sheet_info = {}
                for sheet_name in excel_file.sheet_names:
                    try:
                        df = pd.read_excel(file_path, sheet_name=sheet_name)
                        sheet_info[sheet_name] = {
                            "row_count": len(df),
                            "column_count": len(df.columns),
                            "columns": df.columns.tolist(),
                        }
                    except Exception as e:
                        sheet_info[sheet_name] = {"error": str(e)}

                metadata["sheets_info"] = sheet_info

            except ImportError:
                metadata["analysis_error"] = "pandas not available for Excel metadata"
            except Exception as e:
                metadata["content_analysis_error"] = str(e)

        return metadata
