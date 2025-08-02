"""Tests for JSON document processor."""

import json
from pathlib import Path

import pytest

from agraph.processer.base import ProcessingError
from agraph.processer.json_processor import JSONProcessor


class TestJSONProcessor:
    """Test JSONProcessor functionality."""

    def test_supported_extensions(self):
        """Test that JSONProcessor supports correct extensions."""
        processor = JSONProcessor()
        expected_extensions = [".json", ".jsonl", ".ndjson"]
        assert processor.supported_extensions == expected_extensions

    def test_process_json_file_default(self, sample_json_file):
        """Test processing JSON file with default settings."""
        processor = JSONProcessor()
        result = processor.process(sample_json_file)

        # Should be pretty-printed by default
        assert '"name": "Test Document"' in result
        assert '"description": "A sample JSON document for testing"' in result
        # Should have proper indentation
        assert "  " in result

    def test_process_json_file_no_pretty_print(self, sample_json_file):
        """Test processing JSON file without pretty printing."""
        processor = JSONProcessor()
        result = processor.process(sample_json_file, pretty_print=False)

        # Should be compact
        assert '"name":"Test Document"' in result or '"name": "Test Document"' in result
        # Should not have extra indentation
        lines = result.split("\n")
        assert len(lines) == 1  # Single line

    def test_process_json_file_extract_values(self, sample_json_file):
        """Test processing JSON file with value extraction."""
        processor = JSONProcessor()
        result = processor.process(sample_json_file, extract_values=True)

        # Should only contain text values
        assert "Test Document" in result
        assert "A sample JSON document for testing" in result
        assert "Test Author" in result
        assert "This is sample text content" in result
        # Should not contain JSON structure
        assert "{" not in result
        assert "}" not in result

    def test_process_json_file_max_depth(self, sample_json_file):
        """Test processing JSON file with depth limit."""
        processor = JSONProcessor()
        result = processor.process(sample_json_file, extract_values=True, max_depth=1)

        # Should be limited in depth
        assert "Test Document" in result
        # Nested content should be converted to string representation
        assert result is not None

    def test_process_jsonl_file_default(self, sample_jsonl_file):
        """Test processing JSONL file with default settings."""
        processor = JSONProcessor()
        result = processor.process(sample_jsonl_file)

        # Should contain line numbers and formatted JSON
        assert "Line 1:" in result
        assert "Line 2:" in result
        assert "Line 3:" in result
        assert "Line 5:" in result  # Should skip invalid line
        assert "First line of text" in result
        assert "JSON decode error" in result

    def test_process_jsonl_file_extract_values(self, sample_jsonl_file):
        """Test processing JSONL file with value extraction."""
        processor = JSONProcessor()
        result = processor.process(sample_jsonl_file, extract_values=True)

        # Should extract only text values
        assert "First line of text" in result
        assert "Second line of text" in result
        assert "category: A" in result
        assert "category: B" in result

    def test_process_invalid_json(self, temp_dir):
        """Test processing invalid JSON file."""
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text('{"invalid": json content}')

        processor = JSONProcessor()
        with pytest.raises(ProcessingError, match="Invalid JSON format"):
            processor.process(invalid_json)

    def test_process_empty_json_file(self, temp_dir):
        """Test processing empty JSON file."""
        empty_json = temp_dir / "empty.json"
        empty_json.write_text("")

        processor = JSONProcessor()
        with pytest.raises(ValueError, match="File is empty"):
            processor.process(empty_json)

    def test_extract_text_values_dict(self):
        """Test extracting text values from dictionary."""
        processor = JSONProcessor()
        data = {
            "text_field": "Some text",
            "number_field": 42,
            "empty_field": "",
            "null_field": None,
            "nested": {
                "inner_text": "Nested text"
            }
        }

        result = processor._extract_text_values(data)

        assert "text_field: Some text" in result
        assert "number_field: 42" in result
        assert "empty_field" not in result  # Empty strings excluded
        assert "null_field" not in result  # Null values excluded
        assert "inner_text: Nested text" in result

    def test_extract_text_values_list(self):
        """Test extracting text values from list."""
        processor = JSONProcessor()
        data = [
            "First item",
            42,
            "",
            None,
            {
                "nested": "Nested value"
            }
        ]

        result = processor._extract_text_values(data)

        assert "First item" in result
        assert "42" in result
        assert "nested: Nested value" in result

    def test_extract_text_values_max_depth(self):
        """Test extracting text values with depth limit."""
        processor = JSONProcessor()
        data = {
            "level1": {
                "level2": {
                    "level3": "Deep value"
                }
            }
        }

        result = processor._extract_text_values(data, max_depth=2)

        # Should stop at max depth and convert to string
        assert "level1:" in result
        assert "level2:" in result

    def test_extract_metadata_json(self, sample_json_file):
        """Test extracting metadata from JSON file."""
        processor = JSONProcessor()
        metadata = processor.extract_metadata(sample_json_file)

        assert metadata["file_path"] == str(sample_json_file)
        assert metadata["file_type"] == ".json"
        assert metadata["format"] == "json"
        assert metadata["is_valid_json"] is True
        assert metadata["data_type"] == "dict"
        assert "key_count" in metadata
        assert "top_level_keys" in metadata

    def test_extract_metadata_jsonl(self, sample_jsonl_file):
        """Test extracting metadata from JSONL file."""
        processor = JSONProcessor()
        metadata = processor.extract_metadata(sample_jsonl_file)

        assert metadata["file_path"] == str(sample_jsonl_file)
        assert metadata["file_type"] == ".jsonl"
        assert metadata["format"] == "jsonlines"
        assert metadata["total_lines"] == 4  # Only valid JSON lines
        assert metadata["valid_json_lines"] == 4
        assert metadata["invalid_json_lines"] == 1

    def test_extract_metadata_json_array(self, temp_dir):
        """Test extracting metadata from JSON array."""
        json_array_file = temp_dir / "array.json"
        json_array_file.write_text('[{"id": 1}, {"id": 2}, {"id": 3}]')

        processor = JSONProcessor()
        metadata = processor.extract_metadata(json_array_file)

        assert metadata["data_type"] == "list"
        assert metadata["array_length"] == 3
        assert "item_types" in metadata

    def test_extract_metadata_invalid_json(self, temp_dir):
        """Test extracting metadata from invalid JSON."""
        invalid_json = temp_dir / "invalid.json"
        invalid_json.write_text('{"invalid": json}')

        processor = JSONProcessor()
        metadata = processor.extract_metadata(invalid_json)

        assert metadata["is_valid_json"] is False
        assert "json_error" in metadata

    def test_calculate_depth_dict(self):
        """Test calculating depth of nested dictionary."""
        processor = JSONProcessor()

        data = {
            "level1": {
                "level2": {
                    "level3": "value"
                }
            }
        }

        depth = processor._calculate_depth(data)
        assert depth == 3

    def test_calculate_depth_list(self):
        """Test calculating depth of nested list."""
        processor = JSONProcessor()

        data = [
            [
                [
                    "value"
                ]
            ]
        ]

        depth = processor._calculate_depth(data)
        assert depth == 3

    def test_calculate_depth_mixed(self):
        """Test calculating depth of mixed nested structure."""
        processor = JSONProcessor()

        data = {
            "dict": {
                "list": [
                    {
                        "nested": "value"
                    }
                ]
            }
        }

        depth = processor._calculate_depth(data)
        assert depth == 4

    def test_calculate_depth_empty(self):
        """Test calculating depth of empty structures."""
        processor = JSONProcessor()

        assert processor._calculate_depth({}) == 0
        assert processor._calculate_depth([]) == 0
        assert processor._calculate_depth("string") == 0
        assert processor._calculate_depth(42) == 0

    def test_process_unicode_decode_error_fallback(self, temp_dir):
        """Test fallback to alternative encodings on decode error."""
        # Create file with latin-1 content but try to read as utf-8
        latin_file = temp_dir / "latin.json"
        content = '{"café": "résumé"}'
        latin_file.write_bytes(content.encode("latin-1"))

        processor = JSONProcessor()
        # Should fallback to latin-1 and succeed
        result = processor.process(latin_file)
        assert "café" in result

    def test_process_unsupported_encoding(self, temp_dir):
        """Test processing file with completely unsupported encoding."""
        # Create binary file that can't be decoded
        binary_file = temp_dir / "binary.json"
        binary_file.write_bytes(b"\x80\x81\x82\x83")

        processor = JSONProcessor()
        with pytest.raises(ProcessingError, match="Could not decode JSON file"):
            processor.process(binary_file)

    def test_process_with_custom_encoding(self, temp_dir):
        """Test processing file with custom encoding."""
        # Create file with latin-1 encoding
        latin_file = temp_dir / "latin.json"
        content = '{"café": "résumé"}'
        latin_file.write_bytes(content.encode("latin-1"))

        processor = JSONProcessor()
        result = processor.process(latin_file, encoding="latin-1")
        assert "café" in result
