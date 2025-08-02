"""
Document processing module for agraph.

This module provides document processing interfaces for various file formats including:
- PDF files (.pdf)
- Microsoft Word documents (.docx, .doc)
- Plain text and Markdown files (.txt, .md, .markdown)
- HTML files (.html, .htm)
- Spreadsheet files (.csv, .xlsx, .xls)
- JSON files (.json, .jsonl, .ndjson)
- Image files (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .webp) using multimodal AI models

Basic usage:
    from agraph.processer import process_document, extract_metadata, can_process

    # Process a document
    text_content = process_document("document.pdf")

    # Extract metadata
    metadata = extract_metadata("document.pdf")

    # Check if file can be processed
    if can_process("document.pdf"):
        content = process_document("document.pdf")

    # Process images with multimodal models
    image_text = process_document("image.jpg", prompt="Describe this image in detail")

    # Extract text from images (OCR-like)
    from agraph.processer import ImageProcessor
    processor = ImageProcessor()
    extracted_text = processor.extract_text_from_image("image.jpg")

Advanced usage:
    from agraph.processer import DocumentProcessorManager, DocumentProcessorFactory
    from agraph.processer import ImageProcessorFactory

    # Create processor instance
    processor = DocumentProcessorManager()

    # Process multiple files
    results = processor.process_multiple(["file1.pdf", "file2.docx", "image.jpg"])

    # Custom image processors with specific models
    openai_processor = ImageProcessorFactory.create_openai_processor()
    claude_processor = ImageProcessorFactory.create_claude_processor()

    # Process image with specific model
    description = openai_processor.describe_image("image.jpg", detail_level="comprehensive")
"""

from .base import DocumentProcessor, ProcessingError
from .factory import (
    DocumentProcessorFactory,
    DocumentProcessorManager,
    can_process,
    extract_metadata,
    get_processor,
    get_supported_extensions,
    process_document,
)
from .html_processor import HTMLProcessor
from .image_processor import (
    ClaudeVisionModel,
    ImageProcessor,
    ImageProcessorFactory,
    MultimodalModel,
    OpenAIVisionModel,
)
from .image_utils import ImagePreprocessor
from .json_processor import JSONProcessor
from .pdf_processor import PDFProcessor
from .spreadsheet_processor import SpreadsheetProcessor
from .text_processor import TextProcessor
from .word_processor import WordProcessor

__all__ = [
    # Main interfaces
    "process_document",
    "extract_metadata",
    "can_process",
    "get_processor",
    "get_supported_extensions",
    # Classes
    "DocumentProcessorManager",
    "DocumentProcessorFactory",
    "DocumentProcessor",
    "ProcessingError",
    # Individual processors
    "PDFProcessor",
    "WordProcessor",
    "TextProcessor",
    "HTMLProcessor",
    "SpreadsheetProcessor",
    "JSONProcessor",
    # Image processing
    "ImageProcessor",
    "ImageProcessorFactory",
    "ImagePreprocessor",
    "MultimodalModel",
    "OpenAIVisionModel",
    "ClaudeVisionModel",
]

__version__ = "1.0.0"
