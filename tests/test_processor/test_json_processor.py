"""
Test JSON processor functionality.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agraph.processor.base import ProcessingError
from agraph.processor.json_processor import JSONProcessor


class TestJSONProcessor(unittest.TestCase):
    """Test JSON document processor."""

    def setUp(self):
        """Set up test environment."""
        self.processor = JSONProcessor()
        self.temp_dir = tempfile.mkdtemp()

    def test_supported_extensions(self):
        """Test supported file extensions."""
        extensions = self.processor.supported_extensions
        expected = [".json", ".jsonl", ".ndjson"]
        self.assertEqual(extensions, expected)

    def test_can_process_json_files(self):
        """Test JSON file type detection."""
        self.assertTrue(self.processor.can_process("data.json"))
        self.assertTrue(self.processor.can_process("lines.jsonl"))
        self.assertTrue(self.processor.can_process("stream.ndjson"))
        self.assertFalse(self.processor.can_process("data.txt"))

    def test_process_simple_json_object(self):
        """Test processing simple JSON object."""
        test_file = Path(self.temp_dir) / "simple.json"
        data = {"name": "John", "age": 30, "city": "New York"}
        test_file.write_text(json.dumps(data))

        result = self.processor.process(test_file)

        # Should be pretty printed by default
        self.assertIn("John", result)
        self.assertIn("30", result)
        self.assertIn("New York", result)

    def test_process_json_array(self):
        """Test processing JSON array."""
        test_file = Path(self.temp_dir) / "array.json"
        data = [{"id": 1, "name": "Item 1"}, {"id": 2, "name": "Item 2"}]
        test_file.write_text(json.dumps(data))

        result = self.processor.process(test_file)

        self.assertIn("Item 1", result)
        self.assertIn("Item 2", result)

    def test_process_with_pretty_print_disabled(self):
        """Test processing with pretty printing disabled."""
        test_file = Path(self.temp_dir) / "compact.json"
        data = {"key": "value"}
        test_file.write_text(json.dumps(data))

        result = self.processor.process(test_file, pretty_print=False)

        # Should be compact format
        self.assertEqual(result.strip(), '{"key": "value"}')

    def test_process_with_value_extraction(self):
        """Test text value extraction mode."""
        test_file = Path(self.temp_dir) / "values.json"
        data = {
            "title": "Test Document",
            "description": "This is a test",
            "count": 42,
            "nested": {"content": "Nested content", "number": 123},
        }
        test_file.write_text(json.dumps(data))

        result = self.processor.process(test_file, extract_values=True)

        self.assertIn("Test Document", result)
        self.assertIn("This is a test", result)
        self.assertIn("Nested content", result)
        self.assertIn("42", result)

    def test_process_with_depth_limit(self):
        """Test processing with depth limitation."""
        test_file = Path(self.temp_dir) / "deep.json"
        data = {"level1": {"level2": {"level3": {"content": "Deep content"}}}}
        test_file.write_text(json.dumps(data))

        result = self.processor.process(test_file, extract_values=True, max_depth=2)

        # Should stop at depth 2
        self.assertNotIn("Deep content", result)

    def test_process_jsonlines_file(self):
        """Test processing JSON Lines file."""
        test_file = Path(self.temp_dir) / "data.jsonl"
        lines = [
            '{"id": 1, "text": "First line"}',
            '{"id": 2, "text": "Second line"}',
            "",  # Empty line should be ignored
        ]
        test_file.write_text("\n".join(lines))

        result = self.processor.process(test_file)

        self.assertIn("Line 1:", result)
        self.assertIn("Line 2:", result)
        self.assertIn("First line", result)
        self.assertIn("Second line", result)

    def test_process_jsonlines_with_invalid_lines(self):
        """Test JSON Lines processing with some invalid lines."""
        test_file = Path(self.temp_dir) / "mixed.jsonl"
        lines = ['{"valid": "json"}', "invalid json line", '{"another": "valid line"}']
        test_file.write_text("\n".join(lines))

        result = self.processor.process(test_file)

        self.assertIn("valid", result)
        self.assertIn("JSON decode error", result)
        self.assertIn("another", result)

    def test_process_jsonlines_value_extraction(self):
        """Test JSON Lines with value extraction."""
        test_file = Path(self.temp_dir) / "values.jsonl"
        lines = ['{"message": "Hello world"}', '{"content": "Important data", "priority": 1}']
        test_file.write_text("\n".join(lines))

        result = self.processor.process(test_file, extract_values=True)

        self.assertIn("Hello world", result)
        self.assertIn("Important data", result)

    def test_process_invalid_json(self):
        """Test processing invalid JSON raises error."""
        test_file = Path(self.temp_dir) / "invalid.json"
        test_file.write_text('{"invalid": json}')  # Missing quotes

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("Invalid JSON", str(context.exception))

    def test_process_binary_data_detection(self):
        """Test detection and rejection of binary data."""
        test_file = Path(self.temp_dir) / "binary.json"
        # Write binary data that might be misinterpreted as text
        test_file.write_bytes(b"\x80\x81\x82\x83\x84\x85\x86\x87\x88\x89")

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("binary data", str(context.exception))

    def test_extract_metadata_simple_object(self):
        """Test metadata extraction from simple JSON object."""
        test_file = Path(self.temp_dir) / "simple.json"
        data = {"name": "Test", "value": 123, "active": True}
        test_file.write_text(json.dumps(data))

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["format"], "json")
        self.assertEqual(metadata["data_type"], "dict")
        self.assertTrue(metadata["is_valid_json"])
        self.assertEqual(metadata["key_count"], 3)
        self.assertIn("name", metadata["top_level_keys"])

    def test_extract_metadata_array(self):
        """Test metadata extraction from JSON array."""
        test_file = Path(self.temp_dir) / "array.json"
        data = [{"id": 1}, {"id": 2}, {"id": 3}]
        test_file.write_text(json.dumps(data))

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["data_type"], "list")
        self.assertEqual(metadata["array_length"], 3)
        self.assertIn("dict", metadata["item_types"])

    def test_extract_metadata_nested_structure(self):
        """Test metadata extraction from nested JSON."""
        test_file = Path(self.temp_dir) / "nested.json"
        data = {"level1": {"level2": {"level3": "deep value"}}}
        test_file.write_text(json.dumps(data))

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["max_depth"], 3)

    def test_extract_metadata_jsonlines(self):
        """Test metadata extraction from JSON Lines file."""
        test_file = Path(self.temp_dir) / "data.jsonl"
        lines = [
            '{"type": "message", "content": "Hello"}',
            '{"type": "event", "action": "click"}',
            "invalid json",
        ]
        test_file.write_text("\n".join(lines))

        metadata = self.processor.extract_metadata(test_file)

        self.assertEqual(metadata["format"], "jsonlines")
        self.assertEqual(metadata["total_lines"], 2)  # Only valid lines
        self.assertEqual(metadata["valid_json_lines"], 2)
        self.assertEqual(metadata["invalid_json_lines"], 1)
        self.assertIn("dict", metadata["line_data_types"])

    def test_extract_metadata_invalid_json(self):
        """Test metadata extraction from invalid JSON."""
        test_file = Path(self.temp_dir) / "invalid.json"
        test_file.write_text("invalid json content")

        metadata = self.processor.extract_metadata(test_file)

        self.assertFalse(metadata["is_valid_json"])
        self.assertIn("json_error", metadata)

    def test_calculate_max_depth(self):
        """Test depth calculation for nested structures."""
        # Test empty structures
        self.assertEqual(self.processor._calculate_max_depth({}), 0)
        self.assertEqual(self.processor._calculate_max_depth([]), 0)

        # Test simple structures
        self.assertEqual(self.processor._calculate_max_depth({"key": "value"}), 1)
        self.assertEqual(self.processor._calculate_max_depth(["item"]), 1)

        # Test nested structures
        nested = {"a": {"b": {"c": "value"}}}
        self.assertEqual(self.processor._calculate_max_depth(nested), 3)

    def test_extract_text_values_from_object(self):
        """Test text value extraction from JSON object."""
        data = {
            "title": "Important Title",
            "count": 42,
            "description": "Some description",
            "empty": "",
            "null_value": None,
            "nested": {"content": "Nested content"},
        }

        result = self.processor._extract_text_values_from_data(data)

        self.assertIn("Important Title", result)
        self.assertIn("Some description", result)
        self.assertIn("Nested content", result)
        self.assertIn("42", result)
        # Empty and null values should be excluded

    def test_extract_text_values_from_array(self):
        """Test text value extraction from JSON array."""
        data = ["First item", "Second item", 123, None, ""]

        result = self.processor._extract_text_values_from_data(data)

        self.assertIn("First item", result)
        self.assertIn("Second item", result)
        self.assertIn("123", result)
        # Null and empty values should be excluded

    def test_contains_likely_binary_data(self):
        """Test binary data detection."""
        # Test normal text
        self.assertFalse(self.processor._contains_likely_binary_data("normal text"))

        # Test text with some high-byte characters (should be fine)
        self.assertFalse(self.processor._contains_likely_binary_data("café naïve"))

        # Test mostly high-byte characters (likely binary)
        binary_like = "".join(chr(i) for i in range(128, 138))
        self.assertTrue(self.processor._contains_likely_binary_data(binary_like))

    def test_process_encoding_fallback(self):
        """Test encoding fallback for problematic files."""
        test_file = Path(self.temp_dir) / "latin1.json"
        data = {"text": "café"}
        # Write with latin-1 encoding
        test_file.write_text(json.dumps(data), encoding="latin-1")

        # Should fallback to latin-1 when utf-8 fails
        result = self.processor.process(test_file)

        self.assertIn("café", result)

    def test_process_unsupported_encoding(self):
        """Test processing file with unsupported encoding."""
        test_file = Path(self.temp_dir) / "binary.json"
        test_file.write_bytes(b'\x80\x81\x82\x83{"key": "value"}')

        with self.assertRaises(ProcessingError) as context:
            self.processor.process(test_file)

        self.assertIn("decode", str(context.exception).lower())

    def test_process_empty_file_raises_error(self):
        """Test that processing empty file raises appropriate error."""
        empty_file = Path(self.temp_dir) / "empty.json"
        empty_file.touch()

        with self.assertRaises(ValueError) as context:
            self.processor.process(empty_file)

        self.assertIn("empty", str(context.exception).lower())

    def test_process_nonexistent_file_raises_error(self):
        """Test that processing non-existent file raises appropriate error."""
        nonexistent_file = Path(self.temp_dir) / "nonexistent.json"

        with self.assertRaises(FileNotFoundError):
            self.processor.process(nonexistent_file)

    def test_metadata_extraction_error_handling(self):
        """Test metadata extraction continues when content analysis fails."""
        test_file = Path(self.temp_dir) / "test.json"
        test_file.write_text('{"key": "value"}')

        # Mock file operations to cause error during analysis
        with patch(
            "builtins.open",
            side_effect=[
                open(test_file, "r"),  # For validate_file
                Exception("Analysis error"),  # For metadata analysis
            ],
        ):
            metadata = self.processor.extract_metadata(test_file)

            # Should still have basic metadata
            self.assertIn("file_path", metadata)
            self.assertIn("content_analysis_error", metadata)

    def test_jsonlines_with_mixed_data_types(self):
        """Test JSON Lines with different data types per line."""
        test_file = Path(self.temp_dir) / "mixed.jsonl"
        lines = ['{"type": "object"}', '["array", "data"]', '"simple string"', "42"]
        test_file.write_text("\n".join(lines))

        result = self.processor.process(test_file)

        self.assertIn("object", result)
        self.assertIn("array", result)
        self.assertIn("simple string", result)
        self.assertIn("42", result)

    def test_nested_json_max_depth_calculation(self):
        """Test maximum depth calculation for complex nested structures."""
        deeply_nested = {"a": {"b": {"c": {"d": "value"}}}, "simple": "value"}

        depth = self.processor._calculate_max_depth(deeply_nested)
        self.assertEqual(depth, 4)

    def test_text_values_extraction_with_depth_limit(self):
        """Test text value extraction respects depth limits."""
        data = {
            "level1": {"level2": {"deep_content": "Should not appear"}, "shallow": "Should appear"}
        }

        result = self.processor._extract_text_values_from_data(data, max_depth=2)

        self.assertIn("Should appear", result)
        self.assertNotIn("Should not appear", result)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
