"""Tests for document processor factory and manager."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from agraph.processer.base import DocumentProcessor, ProcessingError
from agraph.processer.factory import (
    DocumentProcessorFactory,
    DocumentProcessorManager,
    can_process,
    extract_metadata,
    get_processor,
    get_supported_extensions,
    process_document,
)


class MockProcessorA(DocumentProcessor):
    """Mock processor for testing."""

    @property
    def supported_extensions(self):
        return [".mocka"]

    def process(self, file_path, **kwargs):
        return f"MockA processed {file_path}"

    def extract_metadata(self, file_path):
        return {"processor": "MockA", "file": str(file_path)}


class MockProcessorB(DocumentProcessor):
    """Another mock processor for testing."""

    @property
    def supported_extensions(self):
        return [".mockb", ".mockc"]

    def process(self, file_path, **kwargs):
        return f"MockB processed {file_path}"

    def extract_metadata(self, file_path):
        return {"processor": "MockB", "file": str(file_path)}


class InvalidProcessor:
    """Invalid processor that doesn't inherit from DocumentProcessor."""

    def process(self, file_path):
        return "Invalid"


class TestDocumentProcessorFactory:
    """Test DocumentProcessorFactory functionality."""

    def test_factory_initialization(self):
        """Test factory initialization with default processors."""
        factory = DocumentProcessorFactory()

        # Should have registered default processors
        extensions = factory.get_supported_extensions()

        # Check for some expected extensions
        assert ".txt" in extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".html" in extensions
        assert ".csv" in extensions
        assert ".json" in extensions

    def test_register_processor_valid(self):
        """Test registering a valid processor."""
        factory = DocumentProcessorFactory()

        # Clear existing processors for clean test
        factory._processors = {}

        factory.register_processor(MockProcessorA)

        assert ".mocka" in factory._processors
        assert factory._processors[".mocka"] == MockProcessorA

    def test_register_processor_multiple_extensions(self):
        """Test registering processor with multiple extensions."""
        factory = DocumentProcessorFactory()
        factory._processors = {}

        factory.register_processor(MockProcessorB)

        assert ".mockb" in factory._processors
        assert ".mockc" in factory._processors
        assert factory._processors[".mockb"] == MockProcessorB
        assert factory._processors[".mockc"] == MockProcessorB

    def test_register_processor_invalid(self):
        """Test registering invalid processor raises error."""
        factory = DocumentProcessorFactory()

        with pytest.raises(ValueError, match="Processor must inherit from DocumentProcessor"):
            factory.register_processor(InvalidProcessor)

    def test_get_processor_existing(self, temp_dir):
        """Test getting processor for supported file type."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        test_file = temp_dir / "test.mocka"
        test_file.write_text("content")

        processor = factory.get_processor(test_file)

        assert isinstance(processor, MockProcessorA)

    def test_get_processor_case_insensitive(self, temp_dir):
        """Test processor lookup is case insensitive."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        test_file = temp_dir / "test.MOCKA"
        test_file.write_text("content")

        processor = factory.get_processor(test_file)

        assert isinstance(processor, MockProcessorA)

    def test_get_processor_unsupported(self, temp_dir):
        """Test getting processor for unsupported file type raises error."""
        factory = DocumentProcessorFactory()
        factory._processors = {}

        test_file = temp_dir / "test.unsupported"
        test_file.write_text("content")

        with pytest.raises(ProcessingError, match="No processor available for file type"):
            factory.get_processor(test_file)

    def test_can_process_supported(self, temp_dir):
        """Test can_process returns True for supported files."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        test_file = temp_dir / "test.mocka"
        test_file.write_text("content")

        assert factory.can_process(test_file) is True

    def test_can_process_unsupported(self, temp_dir):
        """Test can_process returns False for unsupported files."""
        factory = DocumentProcessorFactory()
        factory._processors = {}

        test_file = temp_dir / "test.unsupported"
        test_file.write_text("content")

        assert factory.can_process(test_file) is False

    def test_get_supported_extensions(self):
        """Test getting list of supported extensions."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)
        factory.register_processor(MockProcessorB)

        extensions = factory.get_supported_extensions()

        assert ".mocka" in extensions
        assert ".mockb" in extensions
        assert ".mockc" in extensions
        assert len(extensions) == 3

    def test_process_document(self, temp_dir):
        """Test processing document through factory."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        test_file = temp_dir / "test.mocka"
        test_file.write_text("content")

        result = factory.process_document(test_file)

        assert result == f"MockA processed {test_file}"

    def test_process_document_with_kwargs(self, temp_dir):
        """Test processing document with additional kwargs."""
        factory = DocumentProcessorFactory()
        factory._processors = {}

        # Create a processor that accepts kwargs
        class KwargsProcessor(DocumentProcessor):
            @property
            def supported_extensions(self):
                return [".kwargs"]

            def process(self, file_path, **kwargs):
                return f"Processed with option={kwargs.get('option', 'default')}"

            def extract_metadata(self, file_path):
                return {}

        factory.register_processor(KwargsProcessor)

        test_file = temp_dir / "test.kwargs"
        test_file.write_text("content")

        result = factory.process_document(test_file, option="custom")

        assert result == "Processed with option=custom"

    def test_extract_metadata(self, temp_dir):
        """Test extracting metadata through factory."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        test_file = temp_dir / "test.mocka"
        test_file.write_text("content")

        metadata = factory.extract_metadata(test_file)

        assert metadata["processor"] == "MockA"
        assert metadata["file"] == str(test_file)


class TestDocumentProcessorManager:
    """Test DocumentProcessorManager functionality."""

    def test_manager_initialization_default_factory(self):
        """Test manager initialization with default factory."""
        manager = DocumentProcessorManager()

        assert manager.factory is not None
        assert isinstance(manager.factory, DocumentProcessorFactory)

    def test_manager_initialization_custom_factory(self):
        """Test manager initialization with custom factory."""
        custom_factory = DocumentProcessorFactory()
        manager = DocumentProcessorManager(factory=custom_factory)

        assert manager.factory is custom_factory

    def test_manager_process(self, temp_dir):
        """Test processing through manager."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        manager = DocumentProcessorManager(factory=factory)

        test_file = temp_dir / "test.mocka"
        test_file.write_text("content")

        result = manager.process(test_file)

        assert result == f"MockA processed {test_file}"

    def test_manager_extract_metadata(self, temp_dir):
        """Test extracting metadata through manager."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        manager = DocumentProcessorManager(factory=factory)

        test_file = temp_dir / "test.mocka"
        test_file.write_text("content")

        metadata = manager.extract_metadata(test_file)

        assert metadata["processor"] == "MockA"

    def test_manager_can_process(self, temp_dir):
        """Test can_process through manager."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        manager = DocumentProcessorManager(factory=factory)

        test_file = temp_dir / "test.mocka"
        test_file.write_text("content")

        assert manager.can_process(test_file) is True

    def test_manager_get_supported_extensions(self):
        """Test getting supported extensions through manager."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        manager = DocumentProcessorManager(factory=factory)

        extensions = manager.get_supported_extensions()

        assert ".mocka" in extensions

    def test_manager_process_multiple_success(self, temp_dir):
        """Test processing multiple files successfully."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)
        factory.register_processor(MockProcessorB)

        manager = DocumentProcessorManager(factory=factory)

        file1 = temp_dir / "test1.mocka"
        file1.write_text("content1")
        file2 = temp_dir / "test2.mockb"
        file2.write_text("content2")

        results = manager.process_multiple([file1, file2])

        assert results[str(file1)] == f"MockA processed {file1}"
        assert results[str(file2)] == f"MockB processed {file2}"

    def test_manager_process_multiple_with_errors(self, temp_dir):
        """Test processing multiple files with some errors."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        manager = DocumentProcessorManager(factory=factory)

        file1 = temp_dir / "test1.mocka"
        file1.write_text("content1")
        file2 = temp_dir / "test2.unsupported"
        file2.write_text("content2")

        results = manager.process_multiple([file1, file2])

        assert results[str(file1)] == f"MockA processed {file1}"
        assert isinstance(results[str(file2)], Exception)

    def test_manager_batch_extract_metadata_success(self, temp_dir):
        """Test batch metadata extraction successfully."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        manager = DocumentProcessorManager(factory=factory)

        file1 = temp_dir / "test1.mocka"
        file1.write_text("content1")
        file2 = temp_dir / "test2.mocka"
        file2.write_text("content2")

        results = manager.batch_extract_metadata([file1, file2])

        assert results[str(file1)]["processor"] == "MockA"
        assert results[str(file2)]["processor"] == "MockA"

    def test_manager_batch_extract_metadata_with_errors(self, temp_dir):
        """Test batch metadata extraction with some errors."""
        factory = DocumentProcessorFactory()
        factory._processors = {}
        factory.register_processor(MockProcessorA)

        manager = DocumentProcessorManager(factory=factory)

        file1 = temp_dir / "test1.mocka"
        file1.write_text("content1")
        file2 = temp_dir / "test2.unsupported"
        file2.write_text("content2")

        results = manager.batch_extract_metadata([file1, file2])

        assert results[str(file1)]["processor"] == "MockA"
        assert isinstance(results[str(file2)], Exception)


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_processor(self, sample_text_file):
        """Test global get_processor function."""
        processor = get_processor(sample_text_file)

        assert processor is not None
        assert hasattr(processor, "process")
        assert hasattr(processor, "extract_metadata")

    def test_process_document(self, sample_text_file):
        """Test global process_document function."""
        result = process_document(sample_text_file)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_extract_metadata(self, sample_text_file):
        """Test global extract_metadata function."""
        metadata = extract_metadata(sample_text_file)

        assert isinstance(metadata, dict)
        assert "file_path" in metadata

    def test_can_process(self, sample_text_file):
        """Test global can_process function."""
        result = can_process(sample_text_file)

        assert result is True

    def test_can_process_unsupported(self, temp_dir):
        """Test global can_process with unsupported file."""
        unsupported_file = temp_dir / "test.unsupported"
        unsupported_file.write_text("content")

        result = can_process(unsupported_file)

        assert result is False

    def test_get_supported_extensions(self):
        """Test global get_supported_extensions function."""
        extensions = get_supported_extensions()

        assert isinstance(extensions, list)
        assert len(extensions) > 0
        assert ".txt" in extensions
