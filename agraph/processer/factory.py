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
    """Factory class for creating and managing document processors."""

    def __init__(self) -> None:
        self._processors: Dict[str, Type[DocumentProcessor]] = {}
        self._register_default_processors()

    def _register_default_processors(self) -> None:
        """Register all default document processors."""
        self.register_processor(PDFProcessor)
        self.register_processor(WordProcessor)
        self.register_processor(TextProcessor)
        self.register_processor(HTMLProcessor)
        self.register_processor(SpreadsheetProcessor)
        self.register_processor(JSONProcessor)
        self.register_processor(ImageProcessor)

    def register_processor(self, processor_class: Type[DocumentProcessor]) -> None:
        """Register a document processor for specific file extensions.

        Args:
            processor_class: The processor class to register
        """
        if not issubclass(processor_class, DocumentProcessor):
            raise ValueError(f"Processor must inherit from DocumentProcessor: {processor_class}")

        processor_instance = processor_class()
        for ext in processor_instance.supported_extensions:
            self._processors[ext.lower()] = processor_class

    def get_processor(self, file_path: Union[str, Path]) -> DocumentProcessor:
        """Get the appropriate processor for a file.

        Args:
            file_path: Path to the file

        Returns:
            DocumentProcessor instance

        Raises:
            ProcessingError: If no processor is available for the file type
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()

        if extension not in self._processors:
            raise ProcessingError(f"No processor available for file type: {extension}")

        processor_class = self._processors[extension]
        return processor_class()

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if a file can be processed.

        Args:
            file_path: Path to the file

        Returns:
            True if the file can be processed
        """
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        return extension in self._processors

    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions.

        Returns:
            List of supported extensions
        """
        return list(self._processors.keys())

    def process_document(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Process a document and return its text content.

        Args:
            file_path: Path to the document
            **kwargs: Additional processing parameters

        Returns:
            Extracted text content

        Raises:
            ProcessingError: If processing fails
        """
        processor = self.get_processor(file_path)
        return processor.process(file_path, **kwargs)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from a document.

        Args:
            file_path: Path to the document

        Returns:
            Dictionary containing metadata

        Raises:
            ProcessingError: If extraction fails
        """
        processor = self.get_processor(file_path)
        return processor.extract_metadata(file_path)


class DocumentProcessorManager:
    """High-level document processing interface."""

    def __init__(self, factory: Optional[DocumentProcessorFactory] = None):
        """Initialize the document processor.

        Args:
            factory: Optional custom factory. If None, uses default factory.
        """
        self.factory = factory or DocumentProcessorFactory()

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Process a document and return its text content.

        Args:
            file_path: Path to the document
            **kwargs: Additional processing parameters

        Returns:
            Extracted text content
        """
        return self.factory.process_document(file_path, **kwargs)

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from a document.

        Args:
            file_path: Path to the document

        Returns:
            Dictionary containing metadata
        """
        return self.factory.extract_metadata(file_path)

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if a file can be processed.

        Args:
            file_path: Path to the file

        Returns:
            True if the file can be processed
        """
        return self.factory.can_process(file_path)

    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions.

        Returns:
            List of supported extensions
        """
        return self.factory.get_supported_extensions()

    def process_multiple(self, file_paths: List[Union[str, Path]], **kwargs: Any) -> Dict[str, Union[str, Exception]]:
        """Process multiple documents.

        Args:
            file_paths: List of file paths to process
            **kwargs: Additional processing parameters

        Returns:
            Dictionary mapping file paths to extracted content or exceptions
        """
        results: Dict[str, Union[str, Exception]] = {}
        for file_path in file_paths:
            try:
                results[str(file_path)] = self.process(file_path, **kwargs)
            except Exception as e:
                results[str(file_path)] = e
        return results

    def batch_extract_metadata(self, file_paths: List[Union[str, Path]]) -> Dict[str, Union[Dict[str, Any], Exception]]:
        """Extract metadata from multiple documents.

        Args:
            file_paths: List of file paths

        Returns:
            Dictionary mapping file paths to metadata or exceptions
        """
        results: Dict[str, Union[Dict[str, Any], Exception]] = {}
        for file_path in file_paths:
            try:
                results[str(file_path)] = self.extract_metadata(file_path)
            except Exception as e:
                results[str(file_path)] = e
        return results


# Global factory instance for convenience (lazy initialization)
_default_factory = None


def _get_default_factory() -> DocumentProcessorFactory:
    """Get or create the default factory instance."""
    global _default_factory
    if _default_factory is None:
        _default_factory = DocumentProcessorFactory()
    return _default_factory


def get_processor(file_path: Union[str, Path]) -> DocumentProcessor:
    """Get a processor for the specified file using the default factory.

    Args:
        file_path: Path to the file

    Returns:
        DocumentProcessor instance
    """
    return _get_default_factory().get_processor(file_path)


def process_document(file_path: Union[str, Path], **kwargs: Any) -> str:
    """Process a document using the default factory.

    Args:
        file_path: Path to the document
        **kwargs: Additional processing parameters

    Returns:
        Extracted text content
    """
    return _get_default_factory().process_document(file_path, **kwargs)


def extract_metadata(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract metadata from a document using the default factory.

    Args:
        file_path: Path to the document

    Returns:
        Dictionary containing metadata
    """
    return _get_default_factory().extract_metadata(file_path)


def can_process(file_path: Union[str, Path]) -> bool:
    """Check if a file can be processed using the default factory.

    Args:
        file_path: Path to the file

    Returns:
        True if the file can be processed
    """
    return _get_default_factory().can_process(file_path)


def get_supported_extensions() -> List[str]:
    """Get all supported file extensions from the default factory.

    Returns:
        List of supported extensions
    """
    return _get_default_factory().get_supported_extensions()
