"""PDF document processor implementation.

This module provides functionality for extracting text and metadata from PDF files.
It supports password-protected PDFs, selective page extraction, and comprehensive
metadata extraction including document properties and structure information.
"""

from pathlib import Path
from typing import Any, Dict, List, Union

from .base import DocumentProcessor, ProcessingError


class PDFProcessor(DocumentProcessor):
    """Document processor for PDF files.

    This processor uses the pypdf library to extract text content and metadata
    from PDF documents. It supports various PDF features including:
    - Password-protected PDFs
    - Selective page extraction
    - Metadata extraction from document properties
    - Error handling for corrupted or invalid PDFs

    Dependencies:
        pypdf: Required for PDF processing functionality.
    """

    @property
    def supported_extensions(self) -> List[str]:
        """Return list of supported PDF file extensions.

        Returns:
            List containing '.pdf' extension.
        """
        return [".pdf"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text content from a PDF file.

        This method extracts text from all pages or specified pages of a PDF document.
        It handles encrypted PDFs if a password is provided and supports selective
        page extraction for large documents.

        Args:
            file_path: Path to the PDF file to process.
            **kwargs: Additional processing parameters:
                - password (str): Password for encrypted PDFs
                - pages (int | List[int] | range): Specific pages to extract.
                  Can be a single page number, list of page numbers, or range object.
                  If None, extracts all pages.

        Returns:
            Extracted text content from the PDF, with pages separated by newlines.

        Raises:
            ProcessingError: If pypdf is not installed, PDF is encrypted without
                           password, file is corrupted, or other processing errors occur.
        """
        try:
            import pypdf  # pylint: disable=import-outside-toplevel
        except ImportError as exc:
            raise ProcessingError("pypdf is required for PDF processing. Install with: pip install pypdf") from exc

        self.validate_file(file_path)

        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                # Handle password-protected PDFs
                password = kwargs.get("password")
                if pdf_reader.is_encrypted:
                    if not password:
                        raise ProcessingError("PDF is encrypted but no password provided")
                    if not pdf_reader.decrypt(password):
                        raise ProcessingError("Invalid password for encrypted PDF")

                # Extract text from specified pages or all pages
                pages = kwargs.get("pages")
                if pages is None:
                    pages = range(len(pdf_reader.pages))
                elif isinstance(pages, int):
                    pages = [pages]

                text_content = []
                total_pages = len(pdf_reader.pages)

                for page_num in pages:
                    if 0 <= page_num < total_pages:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text_content.append(page_text)
                    else:
                        # Log warning but continue processing other pages
                        continue

                return "\n".join(text_content)

        except Exception as e:
            if isinstance(e, ProcessingError):
                raise
            raise ProcessingError(f"Failed to process PDF file {file_path}: {str(e)}") from e

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract comprehensive metadata from a PDF file.

        This method extracts both file system metadata and PDF-specific metadata
        including document properties, page count, encryption status, and more.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Dictionary containing metadata with the following keys:
            - file_path: Original file path
            - file_size: File size in bytes
            - page_count: Number of pages in the PDF
            - is_encrypted: Whether the PDF is password-protected
            - title, author, subject, creator, producer: PDF document properties
            - creation_date, modification_date: Document timestamps
            - error: Error message if metadata extraction fails
        """
        try:
            import pypdf  # pylint: disable=import-outside-toplevel
        except ImportError:
            return {"error": "pypdf not available for metadata extraction"}

        self.validate_file(file_path)

        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                metadata = {
                    "file_path": str(file_path),
                    "file_size": Path(file_path).stat().st_size,
                    "page_count": len(pdf_reader.pages),
                    "is_encrypted": pdf_reader.is_encrypted,
                }

                # Extract PDF metadata if available
                if pdf_reader.metadata:
                    pdf_meta = pdf_reader.metadata
                    metadata.update(
                        {
                            "title": pdf_meta.get("/Title", ""),
                            "author": pdf_meta.get("/Author", ""),
                            "subject": pdf_meta.get("/Subject", ""),
                            "creator": pdf_meta.get("/Creator", ""),
                            "producer": pdf_meta.get("/Producer", ""),
                            "creation_date": str(pdf_meta.get("/CreationDate", "")),
                            "modification_date": str(pdf_meta.get("/ModDate", "")),
                        }
                    )

                # Additional PDF-specific information
                if hasattr(pdf_reader, "outline") and pdf_reader.outline:
                    metadata["has_bookmarks"] = len(pdf_reader.outline) > 0
                else:
                    metadata["has_bookmarks"] = False

                return metadata

        except (OSError, IOError, ValueError) as e:
            return {"error": f"Failed to extract metadata: {str(e)}"}
