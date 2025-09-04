"""
Document processing step implementation.
"""

from pathlib import Path
from typing import Any, List, Union

from ...config import BuildSteps
from ..handler.document_processor import DocumentProcessor
from .base import BuildStep, StepResult
from .context import BuildContext


class DocumentProcessingStep(BuildStep):
    """Step for processing documents into text."""
    
    def __init__(self, document_processor: DocumentProcessor, cache_manager):
        """
        Initialize document processing step.
        
        Args:
            document_processor: Handler for document processing operations
            cache_manager: Cache manager instance
        """
        super().__init__(BuildSteps.DOCUMENT_PROCESSING, cache_manager)
        self.document_processor = document_processor
    
    async def _execute_step(self, context: BuildContext) -> StepResult[List[str]]:
        """
        Execute document processing logic.
        
        Args:
            context: Build context containing documents to process
            
        Returns:
            StepResult containing list of extracted texts
        """
        try:
            # Get documents from context
            documents = context.documents
            if not documents:
                return StepResult.failure_result("No documents provided for processing")
            
            # Validate documents
            if not isinstance(documents, list):
                return StepResult.failure_result("Invalid documents type: expected list")
            
            # Convert Path objects to strings if needed
            document_paths = []
            for i, doc in enumerate(documents):
                if isinstance(doc, Path):
                    document_paths.append(str(doc))
                elif isinstance(doc, str):
                    document_paths.append(doc)
                else:
                    return StepResult.failure_result(
                        f"Invalid document at index {i}: expected str or Path, got {type(doc)}"
                    )
            
            # Execute document processing
            texts = self.document_processor.process_documents(
                document_paths,
                context.use_cache
            )
            
            if not isinstance(texts, list):
                return StepResult.failure_result("Document processing returned invalid result type")
            
            # Validate texts
            for i, text in enumerate(texts):
                if not isinstance(text, str):
                    return StepResult.failure_result(
                        f"Invalid text at index {i}: expected str, got {type(text)}"
                    )
            
            # Calculate processing metrics
            total_chars = sum(len(text) for text in texts)
            avg_text_length = total_chars / len(texts) if texts else 0
            
            # Count different document types processed
            doc_extensions = {}
            for doc_path in document_paths:
                try:
                    ext = Path(doc_path).suffix.lower()
                    doc_extensions[ext] = doc_extensions.get(ext, 0) + 1
                except Exception:
                    doc_extensions['unknown'] = doc_extensions.get('unknown', 0) + 1
            
            return StepResult.success_result(
                texts,
                metadata={
                    "input_documents": len(document_paths),
                    "extracted_texts": len(texts),
                    "total_characters": total_chars,
                    "average_text_length": avg_text_length,
                    "document_types": doc_extensions,
                    "extraction_ratio": len(texts) / len(document_paths) if document_paths else 0
                }
            )
            
        except Exception as e:
            return StepResult.failure_result(f"Document processing failed: {str(e)}")
    
    def _get_cache_input_data(self, context: BuildContext) -> Any:
        """Get input data for cache key generation."""
        return context.documents
    
    def _get_expected_result_type(self) -> type:
        """Get expected result type for cache deserialization."""
        return list