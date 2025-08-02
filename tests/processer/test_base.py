"""Tests for base document processor functionality."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from agraph.processer.base import DocumentProcessor, ProcessingError


class MockProcessor(DocumentProcessor):
    """Mock processor for testing base functionality."""

    @property
    def supported_extensions(self):
        return [".txt", ".mock"]

    def process(self, file_path, **kwargs):
        self.validate_file(file_path)
        return f"Processed {file_path}"

    def extract_metadata(self, file_path):
        self.validate_file(file_path)
        return {"file_path": str(file_path), "type": "mock"}


class TestDocumentProcessor:
    """Test base DocumentProcessor functionality."""

    def test_can_process_supported_extension(self, sample_text_file):
        """Test can_process returns True for supported extensions."""
        processor = MockProcessor()
        assert processor.can_process(sample_text_file) is True

    def test_can_process_unsupported_extension(self, temp_dir):
        """Test can_process returns False for unsupported extensions."""
        processor = MockProcessor()
        unsupported_file = temp_dir / "test.pdf"
        unsupported_file.write_text("content")
        assert processor.can_process(unsupported_file) is False

    def test_can_process_case_insensitive(self, temp_dir):
        """Test can_process is case insensitive."""
        processor = MockProcessor()
        uppercase_file = temp_dir / "test.TXT"
        uppercase_file.write_text("content")
        assert processor.can_process(uppercase_file) is True

    def test_validate_file_exists(self, sample_text_file):
        """Test validate_file passes for existing files."""
        processor = MockProcessor()
        # Should not raise an exception
        processor.validate_file(sample_text_file)

    def test_validate_file_not_exists(self, nonexistent_file):
        """Test validate_file raises FileNotFoundError for non-existent files."""
        processor = MockProcessor()
        with pytest.raises(FileNotFoundError, match="File not found"):
            processor.validate_file(nonexistent_file)

    def test_validate_file_is_directory(self, temp_dir):
        """Test validate_file raises ValueError for directories."""
        processor = MockProcessor()
        with pytest.raises(ValueError, match="Path is not a file"):
            processor.validate_file(temp_dir)

    def test_validate_file_empty(self, empty_file):
        """Test validate_file raises ValueError for empty files."""
        processor = MockProcessor()
        with pytest.raises(ValueError, match="File is empty"):
            processor.validate_file(empty_file)

    def test_process_calls_validate(self, sample_text_file):
        """Test that process method calls validate_file."""
        processor = MockProcessor()
        result = processor.process(sample_text_file)
        assert result == f"Processed {sample_text_file}"

    def test_extract_metadata_calls_validate(self, sample_text_file):
        """Test that extract_metadata method calls validate_file."""
        processor = MockProcessor()
        result = processor.extract_metadata(sample_text_file)
        assert result["file_path"] == str(sample_text_file)
        assert result["type"] == "mock"


class TestProcessingError:
    """Test ProcessingError exception."""

    def test_processing_error_inheritance(self):
        """Test ProcessingError inherits from Exception."""
        error = ProcessingError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_processing_error_message(self):
        """Test ProcessingError stores message correctly."""
        message = "Something went wrong during processing"
        error = ProcessingError(message)
        assert str(error) == message
