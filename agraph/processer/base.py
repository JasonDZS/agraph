"""Base classes and interfaces for document processing.

This module defines the core abstractions used throughout the document processing system.
All document processors must inherit from DocumentProcessor and implement its abstract methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union


class DocumentProcessor(ABC):
    """Abstract base class for all document processors.

    This class defines the interface that all document processors must implement.
    Each processor is responsible for handling specific file types and extracting
    text content and metadata from them.

    Attributes:
        supported_extensions: List of file extensions this processor can handle.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions.

        Returns:
            List of file extensions (including dots) that this processor supports.
            For example: ['.pdf', '.txt', '.docx']
        """
        pass

    @abstractmethod
    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Process a document and extract its text content.

        This is the main method for extracting text from documents. Implementations
        should handle file validation, content extraction, and error handling.

        Args:
            file_path: Path to the document file to process.
            **kwargs: Additional processing parameters specific to each processor.
                     Common parameters include:
                     - encoding: Text encoding for text-based files
                     - password: For encrypted files
                     - pages: Specific pages to extract (for multi-page documents)

        Returns:
            Extracted text content as a string. Should be clean and readable.

        Raises:
            ProcessingError: If processing fails due to corruption, unsupported
                           format, missing dependencies, or other errors.
            FileNotFoundError: If the specified file doesn't exist.
            PermissionError: If the file cannot be read due to permissions.
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata information from the document.

        Metadata extraction should provide useful information about the document
        such as creation date, author, page count, etc. The exact metadata
        available depends on the file format.

        Args:
            file_path: Path to the document file.

        Returns:
            Dictionary containing metadata. Common keys include:
            - file_path: Original file path
            - file_size: File size in bytes
            - file_type: File extension
            - created/modified: Timestamps
            Format-specific metadata varies by processor.

        Note:
            This method should not raise exceptions. If metadata extraction
            fails, it should return a dictionary with error information.
        """
        pass

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the given file type.

        This method checks the file extension against the list of supported
        extensions. It does not validate file content or accessibility.

        Args:
            file_path: Path to the file to check.

        Returns:
            True if the processor supports this file type, False otherwise.
        """
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    def validate_file(self, file_path: Union[str, Path]) -> None:
        """Validate that the file exists and is accessible for processing.

        This method performs basic file system validation before processing.
        It should be called by processors before attempting to read file content.

        Args:
            file_path: Path to the file to validate.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the path is not a file or if the file is empty.
            PermissionError: If the file cannot be read due to permissions.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        if not path.stat().st_size > 0:
            raise ValueError(f"File is empty: {file_path}")


class ProcessingError(Exception):
    """Exception raised when document processing fails.

    This exception is raised when a processor encounters an error during
    document processing that prevents successful extraction of content.
    This includes format errors, corruption, missing dependencies, etc.

    The exception message should provide clear information about what
    went wrong and potentially how to fix it.
    """

    pass
