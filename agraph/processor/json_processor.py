"""JSON document processor implementation.

This module provides functionality for extracting text and metadata from JSON files,
including regular JSON, JSON Lines (JSONL), and Newline Delimited JSON (NDJSON) formats.
It supports various extraction modes and comprehensive structural analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import DocumentProcessor, ProcessingError


class JSONProcessor(DocumentProcessor):
    """Document processor for JSON files and JSON Lines format.

    This processor handles multiple JSON formats and provides flexible text extraction:
    - Regular JSON files (.json)
    - JSON Lines files (.jsonl, .ndjson)
    - Text value extraction mode
    - Pretty printing and formatting
    - Depth-limited processing for large nested structures
    - Comprehensive metadata and structure analysis

    Features:
    - Multiple JSON format support
    - Text-only extraction for content focus
    - Configurable nesting depth limits
    - Binary data detection and handling
    - Comprehensive structure analysis
    - Error handling for malformed JSON
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported JSON file extensions.

        Returns:
            List containing '.json', '.jsonl', and '.ndjson' extensions.
        """
        return [".json", ".jsonl", ".ndjson"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text content from JSON files.

        This method processes JSON files and extracts text content in various formats.
        It can handle both regular JSON and JSON Lines formats, with options for
        pretty printing and text-only extraction.

        Args:
            file_path: Path to the JSON file to process.
            **kwargs: Additional processing parameters:
                - encoding (str): Text encoding to use (default: 'utf-8')
                - pretty_print (bool): Whether to format JSON with indentation (default: True)
                - extract_values (bool): Extract only text values, not structure (default: False)
                - max_depth (int): Maximum nesting depth to process (default: None)

        Returns:
            Extracted content as formatted text. Format depends on processing options
            and whether the file is JSON Lines format.

        Raises:
            ProcessingError: If file cannot be decoded, contains invalid JSON,
                           or appears to contain binary data.
        """
        self.validate_file(file_path)
        file_path = Path(file_path)

        encoding = kwargs.get("encoding", "utf-8")
        pretty_print = kwargs.get("pretty_print", True)
        extract_values = kwargs.get("extract_values", False)
        max_depth: Optional[int] = kwargs.get("max_depth")

        try:
            with open(file_path, "r", encoding=encoding) as file:
                if file_path.suffix.lower() in [".jsonl", ".ndjson"]:
                    return self._process_jsonlines_file(
                        file, pretty_print, extract_values, max_depth
                    )
                return self._process_regular_json_file(
                    file, pretty_print, extract_values, max_depth
                )

        except UnicodeDecodeError:
            # Try alternative encodings for JSON files
            if encoding == "utf-8":
                for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                    try:
                        with open(file_path, "r", encoding=alt_encoding) as file:
                            if file_path.suffix.lower() in [".jsonl", ".ndjson"]:
                                return self._process_jsonlines_file(
                                    file, pretty_print, extract_values, max_depth
                                )
                            return self._process_regular_json_file(
                                file, pretty_print, extract_values, max_depth
                            )
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
            raise ProcessingError(
                f"Could not decode JSON file {file_path} with any supported encoding"
            ) from UnicodeDecodeError("utf-8", b"", 0, 1, "Could not decode file")
        except Exception as e:
            raise ProcessingError(f"Failed to process JSON file {file_path}: {str(e)}") from e

    def _process_regular_json_file(
        self, file: Any, pretty_print: bool, extract_values: bool, max_depth: Optional[int] = None
    ) -> str:
        """Process regular JSON files (.json).

        Args:
            file: Open file object.
            pretty_print: Whether to format with indentation.
            extract_values: Whether to extract only text values.
            max_depth: Maximum nesting depth to process.

        Returns:
            Processed JSON content as string.

        Raises:
            ProcessingError: If JSON is invalid or contains binary data.
        """
        try:
            # Read content first to check for potential binary data
            content = file.read()
            file.seek(0)  # Reset file pointer for JSON parsing

            # Detect potential binary data (high-byte characters when using latin-1)
            if self._contains_likely_binary_data(content):
                raise ProcessingError(
                    "File appears to contain binary data and cannot be processed as JSON"
                )

            data = json.load(file)

            if extract_values:
                return self._extract_text_values_from_data(data, max_depth)
            if pretty_print:
                return json.dumps(data, indent=2, ensure_ascii=False, default=str)
            return json.dumps(data, ensure_ascii=False, default=str)

        except json.JSONDecodeError as e:
            raise ProcessingError(f"Invalid JSON format: {str(e)}") from e

    def _process_jsonlines_file(
        self, file: Any, pretty_print: bool, extract_values: bool, max_depth: Optional[int] = None
    ) -> str:
        """Process JSON Lines (.jsonl/.ndjson) files.

        Args:
            file: Open file object.
            pretty_print: Whether to format with indentation.
            extract_values: Whether to extract only text values.
            max_depth: Maximum nesting depth to process.

        Returns:
            Processed JSON Lines content as string.
        """
        results = []
        line_num = 0

        for line in file:
            line = line.strip()
            if not line:
                continue

            line_num += 1
            try:
                data = json.loads(line)

                if extract_values:
                    text = self._extract_text_values_from_data(data, max_depth)
                    if text:
                        results.append(f"Line {line_num}: {text}")
                elif pretty_print:
                    formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
                    results.append(f"Line {line_num}:\n{formatted}")
                else:
                    results.append(
                        f"Line {line_num}: {json.dumps(data, ensure_ascii=False, default=str)}"
                    )

            except json.JSONDecodeError as e:
                results.append(f"Line {line_num}: JSON decode error - {str(e)}")

        return "\n\n".join(results)

    def _contains_likely_binary_data(self, content: str) -> bool:
        """Check if content likely contains binary data.

        Args:
            content: String content to check.

        Returns:
            True if content appears to be binary data.
        """
        # Check for high concentration of high-byte characters
        high_byte_chars = sum(1 for char in content if 128 <= ord(char) <= 255)

        # If content is very short and all high-byte characters, likely binary
        if len(content) <= 10 and high_byte_chars == len(content):
            return True

        # If more than 50% high-byte characters in longer content, likely binary
        if len(content) > 10 and high_byte_chars / len(content) > 0.5:
            return True

        return False

    def _extract_text_values_from_data(
        self, data: Any, max_depth: Optional[int] = None, current_depth: int = 0
    ) -> str:
        """Extract only text values from JSON data structure recursively.

        This method traverses the JSON structure and extracts only meaningful
        text content, ignoring structure keys and non-text values.

        Args:
            data: JSON data to process.
            max_depth: Maximum depth to traverse.
            current_depth: Current traversal depth.

        Returns:
            Extracted text values as formatted string.
        """
        # pylint: disable=too-many-branches
        if max_depth is not None and current_depth >= max_depth:
            return str(data)

        text_values = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and value.strip():
                    text_values.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    nested_text = self._extract_text_values_from_data(
                        value, max_depth, current_depth + 1
                    )
                    if nested_text:
                        text_values.append(f"{key}: {nested_text}")
                elif value is not None and not (isinstance(value, str) and not value.strip()):
                    # Include non-empty, non-null values
                    text_values.append(f"{key}: {str(value)}")

        elif isinstance(data, list):
            for item in data:  # Remove unused variable 'i'
                if isinstance(item, str) and item.strip():
                    text_values.append(item)
                elif isinstance(item, (dict, list)):
                    nested_text = self._extract_text_values_from_data(
                        item, max_depth, current_depth + 1
                    )
                    if nested_text:
                        text_values.append(nested_text)
                elif item is not None and not (isinstance(item, str) and not item.strip()):
                    text_values.append(str(item))

        elif isinstance(data, str) and data.strip():
            return data

        elif data is not None and not (isinstance(data, str) and not data.strip()):
            return str(data)

        return "\n".join(text_values)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract comprehensive metadata from JSON files.

        This method analyzes JSON file structure, content, and provides detailed
        information about the data organization and validity.

        Args:
            file_path: Path to the JSON file.

        Returns:
            Dictionary containing metadata with keys:
            - file_path: Original file path
            - file_size: File size in bytes
            - file_type: File extension
            - created/modified: File timestamps
            - format: 'json' or 'jsonlines'
            - is_valid_json: Whether JSON is syntactically valid
            - data_type: Type of root JSON object
            - Structure analysis (for regular JSON):
              - key_count, top_level_keys: Object structure
              - array_length, item_types: Array structure
              - max_depth: Maximum nesting depth
            - Line analysis (for JSON Lines):
              - total_lines, valid_json_lines, invalid_json_lines
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

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                if file_path.suffix.lower() in [".jsonl", ".ndjson"]:
                    metadata.update(self._analyze_jsonlines_metadata(file))
                else:
                    metadata.update(self._analyze_regular_json_metadata(file))

        except Exception as e:  # pylint: disable=broad-exception-caught
            metadata["content_analysis_error"] = str(e)

        return metadata

    def _analyze_jsonlines_metadata(self, file: Any) -> Dict[str, Any]:
        """Analyze metadata for JSON Lines files.

        Args:
            file: Open file object.

        Returns:
            Dictionary with JSON Lines specific metadata.
        """
        line_count = 0
        valid_lines = 0
        invalid_lines = 0
        data_types = set()

        for line in file:
            line = line.strip()
            if line:
                line_count += 1
                try:
                    data = json.loads(line)
                    valid_lines += 1
                    data_types.add(type(data).__name__)
                except json.JSONDecodeError:
                    invalid_lines += 1

        return {
            "format": "jsonlines",
            "total_lines": valid_lines,  # Only count valid lines
            "valid_json_lines": valid_lines,
            "invalid_json_lines": invalid_lines,
            "line_data_types": list(data_types),
        }

    def _analyze_regular_json_metadata(self, file: Any) -> Dict[str, Any]:
        """Analyze metadata for regular JSON files.

        Args:
            file: Open file object.

        Returns:
            Dictionary with regular JSON specific metadata.
        """
        try:
            data = json.load(file)
            metadata = {
                "format": "json",
                "data_type": type(data).__name__,
                "is_valid_json": True,
            }

            # Analyze structure based on data type
            if isinstance(data, dict):
                metadata.update(
                    {
                        "key_count": len(data),
                        "top_level_keys": list(data.keys())[
                            :10
                        ],  # First 10 keys to avoid huge output
                    }
                )
            elif isinstance(data, list):
                metadata.update(
                    {
                        "array_length": len(data),
                        "item_types": list(
                            set(type(item).__name__ for item in data[:100])
                        ),  # Types of first 100 items
                    }
                )

            # Calculate maximum nesting depth
            metadata["max_depth"] = self._calculate_max_depth(data)

            return metadata

        except json.JSONDecodeError as e:
            return {
                "format": "json",
                "is_valid_json": False,
                "json_error": str(e),
            }

    def _calculate_max_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate the maximum nesting depth of a JSON structure.

        Args:
            data: JSON data to analyze.
            current_depth: Current depth in recursion.

        Returns:
            Maximum depth found in the structure.
        """
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(
                self._calculate_max_depth(value, current_depth + 1) for value in data.values()
            )
        if isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_max_depth(item, current_depth + 1) for item in data)
        return current_depth
