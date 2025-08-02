from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union


class DocumentProcessor(ABC):
    """Base class for document processors."""

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass

    @abstractmethod
    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Process a document and return its text content.

        Args:
            file_path: Path to the document file
            **kwargs: Additional processing parameters

        Returns:
            Extracted text content

        Raises:
            ProcessingError: If processing fails
        """
        pass

    @abstractmethod
    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from the document.

        Args:
            file_path: Path to the document file

        Returns:
            Dictionary containing metadata
        """
        pass

    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this processor can handle the given file.

        Args:
            file_path: Path to the file

        Returns:
            True if the processor can handle this file type
        """
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions

    def validate_file(self, file_path: Union[str, Path]) -> None:
        """Validate that the file exists and is readable.

        Args:
            file_path: Path to the file

        Raises:
            FileNotFoundError: If file doesn't exist
            PermissionError: If file is not readable
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        if not path.stat().st_size > 0:
            raise ValueError(f"File is empty: {file_path}")


class ProcessingError(Exception):
    """Exception raised when document processing fails."""

    pass
