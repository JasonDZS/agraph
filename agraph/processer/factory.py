"""Factory classes for creating and managing document processors.

This module implements the Factory pattern for document processing, providing
a centralized way to create appropriate processors for different file types
and manage the processing workflow.

The factory automatically registers all available processors and routes
documents to the appropriate processor based on file extension.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

from .base import DocumentProcessor, ProcessingError
from .html_processor import HTMLProcessor
from .image_processor import ImageProcessor
from .json_processor import JSONProcessor
from .pdf_processor import PDFProcessor
from .spreadsheet_processor import SpreadsheetProcessor
from .text_processor import TextProcessor
from .word_processor import WordProcessor


class DocumentProcessorFactory:
    """Factory class for creating and managing document processors.

    This factory maintains a registry of all available document processors
    and creates appropriate processor instances based on file extensions.
    It supports both built-in processors and custom processor registration.

    The factory uses lazy instantiation - processors are created only when needed,
    which helps with memory efficiency and dependency management.
    """

    def __init__(self) -> None:
        """Initialize the factory and register default processors.

        Creates an empty processor registry and registers all built-in
        processors for immediate use.
        """
        self._processors: Dict[str, Type[DocumentProcessor]] = {}
        self._register_default_processors()

    def _register_default_processors(self) -> None:
        """Register all default document processors.

        This method registers built-in processors for all supported file types.
        Each processor is associated with its supported file extensions.
        """
        self.register_processor(PDFProcessor)
        self.register_processor(WordProcessor)
        self.register_processor(TextProcessor)
        self.register_processor(HTMLProcessor)
        self.register_processor(SpreadsheetProcessor)
        self.register_processor(JSONProcessor)
        self.register_processor(ImageProcessor)

    def register_processor(self, processor_class: Type[DocumentProcessor]) -> None:
        """Register a document processor for specific file extensions.

        This method allows registration of both built-in and custom processors.
        The processor's supported extensions are automatically detected and
        registered in the factory's routing table.

        Args:
            processor_class: The processor class to register. Must inherit
                           from DocumentProcessor.

        Raises:
            ValueError: If the processor class doesn't inherit from DocumentProcessor.
        """
        if not issubclass(processor_class, DocumentProcessor):
            raise ValueError(f"Processor must inherit from DocumentProcessor: {processor_class}")

        processor_instance = processor_class()
        for ext in processor_instance.supported_extensions:
            self._processors[ext.lower()] = processor_class

    def get_processor(self, file_path: Union[str, Path]) -> DocumentProcessor:
        """Get the appropriate processor instance for a file.

        Creates and returns a processor instance capable of handling the
        specified file type. The processor type is determined by file extension.

        Args:
            file_path: Path to the file that needs processing.

        Returns:
            DocumentProcessor instance appropriate for the file type.

        Raises:
            ProcessingError: If no processor is available for the file type.
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self._processors:
            raise ProcessingError(f"No processor available for file type: {extension}")

        processor_class = self._processors[extension]
        return processor_class()

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if a file can be processed by any registered processor.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if a processor is available for this file type.
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        return extension in self._processors

    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions across all registered processors.

        Returns:
            List of supported file extensions (including dots).
        """
        return list(self._processors.keys())

    def process_document(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Process a document using the appropriate processor.

        This is a convenience method that automatically selects the correct
        processor and processes the document in one call.

        Args:
            file_path: Path to the document to process.
            **kwargs: Additional processing parameters passed to the processor.

        Returns:
            Extracted text content from the document.

        Raises:
            ProcessingError: If no processor is available or processing fails.
        """
        processor = self.get_processor(file_path)
        return processor.process(file_path, **kwargs)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from a document using the appropriate processor.

        Args:
            file_path: Path to the document.

        Returns:
            Dictionary containing document metadata.

        Raises:
            ProcessingError: If no processor is available for the file type.
        """
        processor = self.get_processor(file_path)
        return processor.extract_metadata(file_path)


class DocumentProcessorManager:
    """High-level document processing interface.

    This class provides a simplified interface for document processing operations.
    It manages a factory instance and provides convenient methods for processing
    single documents, multiple documents, and extracting metadata.

    This is the recommended entry point for most document processing tasks.
    """

    def __init__(self, factory: Optional[DocumentProcessorFactory] = None):
        """Initialize the document processor manager.

        Args:
            factory: Optional custom factory instance. If None, creates a new
                    factory with default processors registered.
        """
        self.factory = factory or DocumentProcessorFactory()

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Process a document and return its text content.

        Args:
            file_path: Path to the document to process.
            **kwargs: Additional processing parameters specific to the file type.

        Returns:
            Extracted text content from the document.
        """
        return self.factory.process_document(file_path, **kwargs)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from a document.

        Args:
            file_path: Path to the document.

        Returns:
            Dictionary containing document metadata.
        """
        return self.factory.extract_metadata(file_path)

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if a file can be processed.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the file type is supported.
        """
        return self.factory.can_process(file_path)

    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions.

        Returns:
            List of supported file extensions.
        """
        return self.factory.get_supported_extensions()

    def process_multiple(self, file_paths: List[Union[str, Path]], **kwargs: Any) -> Dict[str, Union[str, Exception]]:
        """Process multiple documents in batch.

        This method processes multiple files and returns results for each file.
        If processing fails for a specific file, the exception is captured
        and returned instead of the content.

        Args:
            file_paths: List of file paths to process.
            **kwargs: Additional processing parameters applied to all files.

        Returns:
            Dictionary mapping file paths to either extracted content (str)
            or exceptions that occurred during processing.
        """
        results: Dict[str, Union[str, Exception]] = {}
        for file_path in file_paths:
            try:
                results[str(file_path)] = self.process(file_path, **kwargs)
            except Exception as e:
                results[str(file_path)] = e
        return results

    def batch_extract_metadata(self, file_paths: List[Union[str, Path]]) -> Dict[str, Union[Dict[str, Any], Exception]]:
        """Extract metadata from multiple documents in batch.

        Args:
            file_paths: List of file paths to process.

        Returns:
            Dictionary mapping file paths to either metadata dictionaries
            or exceptions that occurred during extraction.
        """
        results: Dict[str, Union[Dict[str, Any], Exception]] = {}
        for file_path in file_paths:
            try:
                results[str(file_path)] = self.extract_metadata(file_path)
            except Exception as e:
                results[str(file_path)] = e
        return results


# Global factory instance for convenience functions
# This implements the singleton pattern for the default factory
_default_factory = None


def _get_default_factory() -> DocumentProcessorFactory:
    """Get or create the default factory instance.

    Returns:
        The singleton default factory instance.
    """
    global _default_factory
    if _default_factory is None:
        _default_factory = DocumentProcessorFactory()
    return _default_factory


def get_processor(file_path: Union[str, Path]) -> DocumentProcessor:
    """Get a processor for the specified file using the default factory.

    This is a convenience function that uses the global default factory
    to create processor instances.

    Args:
        file_path: Path to the file.

    Returns:
        DocumentProcessor instance appropriate for the file type.
    """
    return _get_default_factory().get_processor(file_path)


def process_document(file_path: Union[str, Path], **kwargs: Any) -> str:
    """Process a document using the default factory.

    This is a convenience function for quick document processing without
    needing to create factory or manager instances.

    Args:
        file_path: Path to the document.
        **kwargs: Additional processing parameters.

    Returns:
        Extracted text content from the document.
    """
    return _get_default_factory().process_document(file_path, **kwargs)


def extract_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract metadata from a document using the default factory.

    Args:
        file_path: Path to the document.

    Returns:
        Dictionary containing document metadata.
    """
    return _get_default_factory().extract_metadata(file_path)


def can_process(file_path: Union[str, Path]) -> bool:
    """Check if a file can be processed using the default factory.

    Args:
        file_path: Path to the file.

    Returns:
        True if the file type is supported.
    """
    return _get_default_factory().can_process(file_path)


def get_supported_extensions() -> List[str]:
    """Get all supported file extensions from the default factory.

    Returns:
        List of supported file extensions.
    """
    return _get_default_factory().get_supported_extensions()
