import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base import DocumentProcessor, ProcessingError


class JSONProcessor(DocumentProcessor):
    """Processor for JSON documents."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".json", ".jsonl", ".ndjson"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text from JSON file.

        Args:
            file_path: Path to the JSON file
            **kwargs: Additional parameters (pretty_print, extract_values, max_depth, etc.)

        Returns:
            Formatted JSON content as text
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
                    return self._process_jsonlines(file, pretty_print, extract_values, max_depth)
                else:
                    return self._process_json(file, pretty_print, extract_values, max_depth)

        except UnicodeDecodeError:
            # Try alternative encodings
            for alt_encoding in ["latin-1", "cp1252", "iso-8859-1"]:
                try:
                    with open(file_path, "r", encoding=alt_encoding) as file:
                        if file_path.suffix.lower() in [".jsonl", ".ndjson"]:
                            return self._process_jsonlines(file, pretty_print, extract_values, max_depth)
                        else:
                            return self._process_json(file, pretty_print, extract_values, max_depth)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
            raise ProcessingError(f"Could not decode JSON file {file_path} with any supported encoding")
        except Exception as e:
            raise ProcessingError(f"Failed to process JSON file {file_path}: {str(e)}")

    def _process_json(
        self, file: Any, pretty_print: bool, extract_values: bool, max_depth: Optional[int] = None
    ) -> str:
        """Process regular JSON file."""
        try:
            # Read content to check if it's actually valid text
            content = file.read()
            file.seek(0)  # Reset file pointer

            # Check for binary data that might have been incorrectly decoded
            # Characters in the 128-255 range are often binary data when using latin-1
            if any(128 <= ord(char) <= 255 for char in content):
                # If the content is very short and contains only high-byte characters,
                # it's likely binary data
                if len(content) <= 10 and all(128 <= ord(char) <= 255 for char in content):
                    raise ProcessingError("Could not decode JSON file: file appears to contain binary data")

            data = json.load(file)

            if extract_values:
                return self._extract_text_values(data, max_depth)
            elif pretty_print:
                return json.dumps(data, indent=2, ensure_ascii=False, default=str)
            else:
                return json.dumps(data, ensure_ascii=False, default=str)

        except json.JSONDecodeError as e:
            raise ProcessingError(f"Invalid JSON format: {str(e)}")

    def _process_jsonlines(
        self, file: Any, pretty_print: bool, extract_values: bool, max_depth: Optional[int] = None
    ) -> str:
        """Process JSON Lines (.jsonl/.ndjson) file."""
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
                    text = self._extract_text_values(data, max_depth)
                    if text:
                        results.append(f"Line {line_num}: {text}")
                elif pretty_print:
                    formatted = json.dumps(data, indent=2, ensure_ascii=False, default=str)
                    results.append(f"Line {line_num}:\n{formatted}")
                else:
                    results.append(f"Line {line_num}: {json.dumps(data, ensure_ascii=False, default=str)}")

            except json.JSONDecodeError as e:
                results.append(f"Line {line_num}: JSON decode error - {str(e)}")

        return "\n\n".join(results)

    def _extract_text_values(self, data: Any, max_depth: Optional[int] = None, current_depth: int = 0) -> str:
        """Extract only text values from JSON data structure."""
        if max_depth is not None and current_depth >= max_depth:
            return str(data)

        text_values = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, str) and value.strip():
                    text_values.append(f"{key}: {value}")
                elif isinstance(value, (dict, list)):
                    nested_text = self._extract_text_values(value, max_depth, current_depth + 1)
                    if nested_text:
                        text_values.append(f"{key}: {nested_text}")
                elif value is not None and not (isinstance(value, str) and not value.strip()):
                    text_values.append(f"{key}: {str(value)}")

        elif isinstance(data, list):
            for i, item in enumerate(data):
                if isinstance(item, str) and item.strip():
                    text_values.append(item)
                elif isinstance(item, (dict, list)):
                    nested_text = self._extract_text_values(item, max_depth, current_depth + 1)
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
        """Extract metadata from JSON file.

        Args:
            file_path: Path to the JSON file

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

        try:
            with open(file_path, "r", encoding="utf-8") as file:
                if file_path.suffix.lower() in [".jsonl", ".ndjson"]:
                    # JSONL metadata
                    line_count = 0
                    valid_lines = 0
                    invalid_lines = 0

                    for line in file:
                        line = line.strip()
                        if line:
                            line_count += 1
                            try:
                                json.loads(line)
                                valid_lines += 1
                            except json.JSONDecodeError:
                                invalid_lines += 1

                    metadata.update(
                        {
                            "format": "jsonlines",
                            "total_lines": valid_lines,  # Only count valid lines
                            "valid_json_lines": valid_lines,
                            "invalid_json_lines": invalid_lines,
                        }
                    )
                else:
                    # Regular JSON metadata
                    try:
                        data = json.load(file)
                        metadata.update(
                            {
                                "format": "json",
                                "data_type": type(data).__name__,
                                "is_valid_json": True,
                            }
                        )

                        # Analyze structure
                        if isinstance(data, dict):
                            metadata.update(
                                {
                                    "key_count": len(data),
                                    "top_level_keys": list(data.keys())[:10],  # First 10 keys
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

                        # Calculate depth
                        metadata["max_depth"] = self._calculate_depth(data)

                    except json.JSONDecodeError as e:
                        metadata.update(
                            {
                                "format": "json",
                                "is_valid_json": False,
                                "json_error": str(e),
                            }
                        )

        except Exception as e:
            metadata["content_analysis_error"] = str(e)

        return metadata

    def _calculate_depth(self, data: Any, current_depth: int = 0) -> int:
        """Calculate the maximum depth of nested JSON structure."""
        if isinstance(data, dict):
            if not data:
                return current_depth
            return max(self._calculate_depth(value, current_depth + 1) for value in data.values())
        elif isinstance(data, list):
            if not data:
                return current_depth
            return max(self._calculate_depth(item, current_depth + 1) for item in data)
        else:
            return current_depth
