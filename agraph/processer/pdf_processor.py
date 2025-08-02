from pathlib import Path
from typing import Any, Dict, List, Union

from .base import DocumentProcessor, ProcessingError


class PDFProcessor(DocumentProcessor):
    """Processor for PDF documents."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def process(self, file_path: Union[str, Path], **kwargs: Any) -> str:
        """Extract text from PDF file.

        Args:
            file_path: Path to the PDF file
            **kwargs: Additional parameters (password, pages, etc.)

        Returns:
            Extracted text content
        """
        try:
            import pypdf
        except ImportError:
            raise ProcessingError("pypdf is required for PDF processing. Install with: pip install pypdf")

        self.validate_file(file_path)

        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)

                # Handle password-protected PDFs
                password = kwargs.get("password")
                if pdf_reader.is_encrypted:
                    if not password:
                        raise ProcessingError("PDF is encrypted but no password provided")
                    pdf_reader.decrypt(password)

                # Extract text from specified pages or all pages
                pages = kwargs.get("pages")
                if pages is None:
                    pages = range(len(pdf_reader.pages))
                elif isinstance(pages, int):
                    pages = [pages]

                text_content = []
                for page_num in pages:
                    if 0 <= page_num < len(pdf_reader.pages):
                        page = pdf_reader.pages[page_num]
                        text_content.append(page.extract_text())

                return "\n".join(text_content)

        except Exception as e:
            raise ProcessingError(f"Failed to process PDF file {file_path}: {str(e)}")

    def extract_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Extract metadata from PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary containing metadata
        """
        try:
            import pypdf
        except ImportError:
            return {"error": "pypdf not available"}

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

                return metadata

        except Exception as e:
            return {"error": f"Failed to extract metadata: {str(e)}"}
