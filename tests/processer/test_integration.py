"""Integration tests for the processer module."""

import pytest

from agraph.processer import (
    DocumentProcessor,
    DocumentProcessorFactory,
    DocumentProcessorManager,
    HTMLProcessor,
    JSONProcessor,
    PDFProcessor,
    ProcessingError,
    SpreadsheetProcessor,
    TextProcessor,
    WordProcessor,
    can_process,
    extract_metadata,
    get_processor,
    get_supported_extensions,
    process_document,
)

# Import image-related classes conditionally to avoid CI issues
try:
    from agraph.processer import ImageProcessor, ImageProcessorFactory
    IMAGE_PROCESSOR_AVAILABLE = True
except Exception:
    IMAGE_PROCESSOR_AVAILABLE = False


class TestModuleImports:
    """Test module imports and public API."""

    def test_import_base_classes(self):
        """Test importing base classes."""
        assert DocumentProcessor is not None
        assert ProcessingError is not None

    def test_import_processors(self):
        """Test importing all processor classes."""
        assert TextProcessor is not None
        assert PDFProcessor is not None
        assert WordProcessor is not None
        assert HTMLProcessor is not None
        assert SpreadsheetProcessor is not None
        assert JSONProcessor is not None

        # Image processor is optional depending on environment
        if IMAGE_PROCESSOR_AVAILABLE:
            assert ImageProcessor is not None
            assert ImageProcessorFactory is not None

    def test_import_factory_classes(self):
        """Test importing factory and manager classes."""
        assert DocumentProcessorFactory is not None
        assert DocumentProcessorManager is not None

    def test_import_convenience_functions(self):
        """Test importing convenience functions."""
        assert process_document is not None
        assert extract_metadata is not None
        assert can_process is not None
        assert get_processor is not None
        assert get_supported_extensions is not None


class TestIntegration:
    """Integration tests for the processer module."""

    def test_end_to_end_text_processing(self, sample_text_file):
        """Test end-to-end text file processing."""
        # Test that we can process a text file from start to finish
        assert can_process(sample_text_file) is True

        processor = get_processor(sample_text_file)
        assert isinstance(processor, TextProcessor)

        content = process_document(sample_text_file)
        assert isinstance(content, str)
        assert len(content) > 0

        metadata = extract_metadata(sample_text_file)
        assert isinstance(metadata, dict)
        assert "file_path" in metadata

    def test_end_to_end_json_processing(self, sample_json_file):
        """Test end-to-end JSON file processing."""
        assert can_process(sample_json_file) is True

        processor = get_processor(sample_json_file)
        assert isinstance(processor, JSONProcessor)

        content = process_document(sample_json_file)
        assert isinstance(content, str)
        assert len(content) > 0

        metadata = extract_metadata(sample_json_file)
        assert isinstance(metadata, dict)
        assert metadata["format"] == "json"

    def test_end_to_end_csv_processing(self, sample_csv_file):
        """Test end-to-end CSV file processing."""
        assert can_process(sample_csv_file) is True

        processor = get_processor(sample_csv_file)
        assert isinstance(processor, SpreadsheetProcessor)

        content = process_document(sample_csv_file)
        assert isinstance(content, str)
        assert "Name" in content
        assert "John Doe" in content

    def test_get_supported_extensions_comprehensive(self):
        """Test that all expected extensions are supported."""
        extensions = get_supported_extensions()

        # Text files
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".markdown" in extensions

        # Documents
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".doc" in extensions

        # Web
        assert ".html" in extensions
        assert ".htm" in extensions

        # Data
        assert ".csv" in extensions
        assert ".xlsx" in extensions
        assert ".xls" in extensions
        assert ".json" in extensions
        assert ".jsonl" in extensions

        # Images (only check if image processor is available)
        if IMAGE_PROCESSOR_AVAILABLE:
            assert ".jpg" in extensions
            assert ".jpeg" in extensions
            assert ".png" in extensions
        else:
            # In CI environments without API keys, image extensions might not be available
            # This is expected behavior
            pass

    def test_manager_batch_processing(self, sample_text_file, sample_json_file, sample_csv_file):
        """Test batch processing with DocumentProcessorManager."""
        manager = DocumentProcessorManager()

        files = [sample_text_file, sample_json_file, sample_csv_file]
        results = manager.process_multiple(files)

        assert len(results) == 3
        for file_path, result in results.items():
            assert isinstance(result, str)
            assert len(result) > 0

    def test_manager_batch_metadata_extraction(self, sample_text_file, sample_json_file):
        """Test batch metadata extraction."""
        manager = DocumentProcessorManager()

        files = [sample_text_file, sample_json_file]
        results = manager.batch_extract_metadata(files)

        assert len(results) == 2
        for file_path, metadata in results.items():
            assert isinstance(metadata, dict)
            assert "file_path" in metadata

    def test_unsupported_file_handling(self, temp_dir):
        """Test handling of unsupported file types."""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("unsupported content")

        assert can_process(unsupported_file) is False

        with pytest.raises(ProcessingError):
            get_processor(unsupported_file)

        with pytest.raises(ProcessingError):
            process_document(unsupported_file)

    def test_error_handling_in_batch_processing(self, sample_text_file, temp_dir):
        """Test error handling in batch processing."""
        manager = DocumentProcessorManager()

        # Include one valid and one invalid file
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("unsupported content")

        files = [sample_text_file, unsupported_file]
        results = manager.process_multiple(files)

        # Valid file should succeed
        assert isinstance(results[str(sample_text_file)], str)

        # Invalid file should have an exception
        assert isinstance(results[str(unsupported_file)], Exception)

    def test_processor_registration_workflow(self, temp_dir):
        """Test custom processor registration workflow."""
        from agraph.processer.base import DocumentProcessor

        # Create a custom processor
        class CustomProcessor(DocumentProcessor):
            @property
            def supported_extensions(self):
                return [".custom"]

            def process(self, file_path, **kwargs):
                return f"Custom processing of {file_path}"

            def extract_metadata(self, file_path):
                return {"custom": True, "file_path": str(file_path)}

        # Create factory and register custom processor
        factory = DocumentProcessorFactory()
        factory.register_processor(CustomProcessor)

        # Test the custom processor
        custom_file = temp_dir / "test.custom"
        custom_file.write_text("custom content")

        assert factory.can_process(custom_file) is True

        processor = factory.get_processor(custom_file)
        assert isinstance(processor, CustomProcessor)

        content = factory.process_document(custom_file)
        assert "Custom processing" in content

        metadata = factory.extract_metadata(custom_file)
        assert metadata["custom"] is True

    def test_module_version_info(self):
        """Test module version information."""
        from agraph.processer import __version__
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_all_exports_available(self):
        """Test that all items in __all__ are available."""
        from agraph.processer import __all__
        import agraph.processer as processer_module

        # Image-related exports that might not be available in CI
        image_related_exports = {
            "ImageProcessor", "ImageProcessorFactory", "ImagePreprocessor",
            "MultimodalModel", "OpenAIVisionModel", "ClaudeVisionModel"
        }

        for item in __all__:
            if item in image_related_exports and not IMAGE_PROCESSOR_AVAILABLE:
                # Skip image-related exports if image processor is not available
                continue
            assert hasattr(processer_module, item), f"Missing export: {item}"

    @pytest.mark.skipif(not IMAGE_PROCESSOR_AVAILABLE, reason="Image processor not available in CI environment")
    def test_image_processor_functionality(self, temp_dir):
        """Test image processor functionality when available."""
        # Create a dummy image file (just for testing file path handling)
        image_file = temp_dir / "test.jpg"
        image_file.write_bytes(b"\xff\xd8\xff\xe0\x00\x10JFIF")  # Minimal JPEG header

        # Test that image files can be processed when image processor is available
        extensions = get_supported_extensions()
        assert ".jpg" in extensions
        assert ".jpeg" in extensions
        assert ".png" in extensions

        # Test that we can get an image processor
        assert can_process(image_file) is True

        # Note: We don't actually process the image to avoid API calls
        # The mock fixture should handle this in actual test runs

    def test_image_processor_graceful_degradation(self):
        """Test that the module works gracefully without image processor."""
        # This test ensures that even if image processor is not available,
        # the core functionality still works
        extensions = get_supported_extensions()

        # Core extensions should always be available
        core_extensions = [".txt", ".pdf", ".docx", ".html", ".csv", ".json"]
        for ext in core_extensions:
            assert ext in extensions, f"Core extension {ext} should always be available"
