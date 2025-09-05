"""
Document processing handler for knowledge graph builder.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import List, Union

from ...builder.cache import CacheManager
from ...config import DocumentProcessingStatus
from ...logger import logger
from ...processor.factory import DocumentProcessorFactory


class DocumentProcessor:
    """Handles document processing with caching and incremental updates."""

    def __init__(self, cache_manager: CacheManager, processor_factory: DocumentProcessorFactory):
        """Initialize document processor.

        Args:
            cache_manager: Cache manager instance
            processor_factory: Document processor factory
        """
        self.cache_manager = cache_manager
        self.processor_factory = processor_factory

    def process_documents(self, documents: List[Union[str, Path]], use_cache: bool = True) -> List[str]:
        """Process documents and extract text content with incremental updates.

        Args:
            documents: List of document paths
            use_cache: Whether to use caching

        Returns:
            List of extracted text content
        """
        logger.info(f"Processing {len(documents)} documents with incremental updates")

        if not use_cache:
            return self._process_all_documents(documents)

        # Separate processed and unprocessed documents
        processed_docs, unprocessed_docs = self.cache_manager.get_processed_documents(documents)

        logger.info(
            f"Document analysis: {len(processed_docs)} already processed, " f"{len(unprocessed_docs)} need processing"
        )

        # Get cached results for processed documents
        cached_results = self.cache_manager.get_cached_document_results(processed_docs)
        texts = []

        # Add cached results in the same order as original documents
        for doc_path in documents:
            doc_key = str(doc_path)
            if doc_key in cached_results:
                texts.append(cached_results[doc_key])
                logger.debug(f"Using cached result for: {doc_path}")
            else:
                # Process unprocessed document
                if doc_path in unprocessed_docs:
                    try:
                        start_time = time.time()
                        logger.debug(f"Processing new/modified document: {doc_path}")

                        # Update status to processing
                        processing_status = self.cache_manager.get_document_status(doc_path)
                        if processing_status:
                            processing_status.processing_status = "processing"
                            self.cache_manager.update_document_status(doc_path, processing_status)

                        processor = self.processor_factory.get_processor(str(doc_path))
                        content = processor.process(str(doc_path))
                        processing_time = time.time() - start_time

                        texts.append(content)

                        # Save result with status tracking
                        self.cache_manager.save_document_processing_result(doc_path, content, processing_time)

                        logger.debug(
                            f"Successfully processed document {doc_path} - "
                            f"content length: {len(content)} chars, time: {processing_time:.2f}s"
                        )
                    except Exception as e:
                        logger.error(f"Error processing document {doc_path}: {e}")

                        # Update status to failed
                        path = Path(doc_path)
                        if path.exists():
                            file_hash = self.cache_manager.get_file_hash(doc_path)
                            error_status = DocumentProcessingStatus(
                                file_path=str(doc_path),
                                file_hash=file_hash,
                                last_modified=datetime.fromtimestamp(path.stat().st_mtime),
                                processing_status="failed",
                                error_message=str(e),
                            )
                            self.cache_manager.update_document_status(doc_path, error_status)

                        continue

        logger.info(
            f"Document processing completed - processed {len(texts)}/{len(documents)} documents "
            f"({len(unprocessed_docs)} newly processed, {len(processed_docs)} from cache)"
        )

        return texts

    def _process_all_documents(self, documents: List[Union[str, Path]]) -> List[str]:
        """Process all documents without caching (fallback method)."""
        texts = []

        for i, doc_path in enumerate(documents):
            try:
                logger.debug(f"Processing document {i+1}/{len(documents)}: {doc_path}")
                processor = self.processor_factory.get_processor(str(doc_path))
                content = processor.process(str(doc_path))
                texts.append(content)
                logger.debug(f"Successfully processed document {doc_path} - content length: {len(content)} chars")
            except Exception as e:
                logger.error(f"Error processing document {doc_path}: {e}")
                continue

        return texts
