"""
Test processor factory functionality.
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from agraph.processor.base import DependencyError, DocumentProcessor, ProcessingError
from agraph.processor.factory import DocumentProcessorFactory, DocumentProcessorManager


class MockProcessor(DocumentProcessor):
    """Mock processor for testing."""

    @property
    def supported_extensions(self):
        return [".mock"]

    def process(self, file_path, **kwargs):
        return f"Mock processed: {file_path}"

    def extract_metadata(self, file_path):
        return {"type": "mock", "path": str(file_path)}


class FailingProcessor(DocumentProcessor):
    """Processor that fails during instantiation."""

    @property
    def supported_extensions(self):
        return [".fail"]

    def __init__(self):
        raise ImportError("Missing dependency")

    def process(self, file_path, **kwargs):
        return ""

    def extract_metadata(self, file_path):
        return {}


class TestDocumentProcessorFactory(unittest.TestCase):
    """Test document processor factory."""

    def setUp(self):
        """Set up test environment."""
        self.factory = DocumentProcessorFactory()
        self.temp_dir = tempfile.mkdtemp()

    def test_factory_initialization(self):
        """Test factory initialization with default processors."""
        factory = DocumentProcessorFactory()

        # Check that some default processors are registered
        supported_extensions = factory.get_supported_extensions()
        self.assertGreater(len(supported_extensions), 0)

        # Check for common extensions
        expected_extensions = [".txt", ".pdf", ".docx", ".html", ".json"]
        for ext in expected_extensions:
            self.assertIn(ext, supported_extensions)

    def test_register_custom_processor(self):
        """Test registering a custom processor."""
        factory = DocumentProcessorFactory()

        # Register mock processor
        factory.register_processor(MockProcessor)

        # Check it's available
        self.assertTrue(factory.can_process("test.mock"))
        self.assertIn(".mock", factory.get_supported_extensions())

    def test_register_processor_with_missing_dependencies(self):
        """Test handling of processors with missing dependencies."""
        factory = DocumentProcessorFactory()

        # Should raise DependencyError
        with self.assertRaises(DependencyError):
            factory.register_processor(FailingProcessor)

    def test_register_invalid_processor(self):
        """Test error handling for invalid processor classes."""
        factory = DocumentProcessorFactory()

        # Should raise ValueError for non-processor class
        with self.assertRaises(ValueError):
            factory.register_processor(str)

    def test_get_processor_for_supported_file(self):
        """Test getting processor for supported file type."""
        factory = DocumentProcessorFactory()

        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")

        # Get processor
        processor = factory.get_processor(test_file)
        self.assertIsInstance(processor, DocumentProcessor)

    def test_get_processor_for_unsupported_file(self):
        """Test error when getting processor for unsupported file type."""
        factory = DocumentProcessorFactory()

        # Should raise ProcessingError for unknown extension
        with self.assertRaises(ProcessingError):
            factory.get_processor("test.unknown")

    def test_processor_instance_caching(self):
        """Test that processor instances are cached."""
        factory = DocumentProcessorFactory()

        # Get processor twice
        processor1 = factory.get_processor("test.txt")
        processor2 = factory.get_processor("test.txt")

        # Should be same instance
        self.assertIs(processor1, processor2)

    def test_can_process_method(self):
        """Test file type checking."""
        factory = DocumentProcessorFactory()

        # Test supported types
        self.assertTrue(factory.can_process("test.txt"))
        self.assertTrue(factory.can_process("test.pdf"))

        # Test unsupported type
        self.assertFalse(factory.can_process("test.unknown"))

    def test_process_document_convenience_method(self):
        """Test convenience document processing method."""
        factory = DocumentProcessorFactory()

        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")

        # Process document
        result = factory.process_document(test_file)
        self.assertIsInstance(result, str)
        self.assertIn("test content", result)

    def test_extract_metadata_convenience_method(self):
        """Test convenience metadata extraction method."""
        factory = DocumentProcessorFactory()

        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")

        # Extract metadata
        metadata = factory.extract_metadata(test_file)
        self.assertIsInstance(metadata, dict)
        self.assertIn("file_path", metadata)


class TestDocumentProcessorManager(unittest.TestCase):
    """Test document processor manager."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DocumentProcessorManager()

    def test_manager_initialization(self):
        """Test manager initialization."""
        manager = DocumentProcessorManager()
        self.assertIsNotNone(manager.factory)

    def test_manager_with_custom_factory(self):
        """Test manager with custom factory."""
        factory = DocumentProcessorFactory()
        manager = DocumentProcessorManager(factory=factory)
        self.assertIs(manager.factory, factory)

    def test_single_document_processing(self):
        """Test processing a single document."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")

        # Process document
        result = self.manager.process(test_file)
        self.assertIsInstance(result, str)
        self.assertIn("test content", result)

    def test_metadata_extraction(self):
        """Test metadata extraction."""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")

        # Extract metadata
        metadata = self.manager.extract_metadata(test_file)
        self.assertIsInstance(metadata, dict)

    def test_can_process_check(self):
        """Test file type support checking."""
        self.assertTrue(self.manager.can_process("test.txt"))
        self.assertFalse(self.manager.can_process("test.unknown"))

    def test_get_supported_extensions(self):
        """Test getting supported extensions."""
        extensions = self.manager.get_supported_extensions()
        self.assertIsInstance(extensions, list)
        self.assertGreater(len(extensions), 0)

    def test_batch_processing_success(self):
        """Test successful batch processing."""
        # Create test files
        test_files = []
        for i in range(3):
            test_file = Path(self.temp_dir) / f"test{i}.txt"
            test_file.write_text(f"content {i}")
            test_files.append(test_file)

        # Process batch
        results = self.manager.process_multiple(test_files)

        self.assertEqual(len(results), 3)
        for file_path, result in results.items():
            self.assertIsInstance(result, str)
            self.assertIn("content", result)

    def test_batch_processing_with_failures(self):
        """Test batch processing with some failures."""
        # Create test files (some valid, some invalid)
        valid_file = Path(self.temp_dir) / "valid.txt"
        valid_file.write_text("valid content")

        invalid_file = Path(self.temp_dir) / "nonexistent.txt"

        files = [valid_file, invalid_file]
        results = self.manager.process_multiple(files)

        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[str(valid_file)], str)
        self.assertIsInstance(results[str(invalid_file)], Exception)

    def test_batch_metadata_extraction(self):
        """Test batch metadata extraction."""
        # Create test files
        test_files = []
        for i in range(2):
            test_file = Path(self.temp_dir) / f"test{i}.txt"
            test_file.write_text(f"content {i}")
            test_files.append(test_file)

        # Extract metadata
        results = self.manager.batch_extract_metadata(test_files)

        self.assertEqual(len(results), 2)
        for result in results.values():
            self.assertIsInstance(result, dict)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestGlobalFactoryFunctions(unittest.TestCase):
    """Test global factory convenience functions."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()

    def test_global_get_processor(self):
        """Test global get_processor function."""
        from agraph.processor.factory import get_processor

        processor = get_processor("test.txt")
        self.assertIsInstance(processor, DocumentProcessor)

    def test_global_process_document(self):
        """Test global process_document function."""
        from agraph.processor.factory import process_document

        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")

        result = process_document(test_file)
        self.assertIsInstance(result, str)

    def test_global_extract_metadata(self):
        """Test global extract_metadata function."""
        from agraph.processor.factory import extract_metadata

        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_file.write_text("test content")

        metadata = extract_metadata(test_file)
        self.assertIsInstance(metadata, dict)

    def test_global_can_process(self):
        """Test global can_process function."""
        from agraph.processor.factory import can_process

        self.assertTrue(can_process("test.txt"))
        self.assertFalse(can_process("test.unknown"))

    def test_global_get_supported_extensions(self):
        """Test global get_supported_extensions function."""
        from agraph.processor.factory import get_supported_extensions

        extensions = get_supported_extensions()
        self.assertIsInstance(extensions, list)
        self.assertGreater(len(extensions), 0)

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
