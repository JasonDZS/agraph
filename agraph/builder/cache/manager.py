"""
Cache manager for coordinating cache operations.
"""

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from ...config import (
    BuilderConfig,
    BuildStatus,
    BuildSteps,
    CacheMetadata,
    DocumentProcessingStatus,
)
from ...logger import logger
from .base import CacheBackend
from .file_cache import FileCacheBackend

T = TypeVar("T")


class CacheManager:
    """Manager for cache operations with step awareness."""

    def __init__(self, config: BuilderConfig, backend: Optional[CacheBackend] = None):
        """Initialize cache manager.

        Args:
            config: Builder configuration
            backend: Cache backend (defaults to FileCacheBackend)
        """
        self.config = config
        self.backend = backend or FileCacheBackend(config.cache_dir)
        self._build_status: Optional[BuildStatus] = None
        self._document_status_cache: Dict[str, DocumentProcessingStatus] = {}

    def get_step_result(
        self, step_name: str, input_data: Any, expected_type: Type[T]
    ) -> Optional[T]:
        """Get cached result for a step.

        Args:
            step_name: Name of the build step
            input_data: Input data for the step
            expected_type: Expected return type

        Returns:
            Cached result or None if not found/invalid
        """
        # pylint: disable=too-many-return-statements
        if not self.config.enable_cache:
            return None

        # Generate cache key
        key = self._generate_step_key(step_name, input_data)

        # Check if cached result exists
        if not self.backend.has(key):
            return None

        # Check if cached result is expired
        metadata = self.backend.get_metadata(key)
        if metadata and self.backend.is_expired(metadata, self.config.cache_ttl):
            self.backend.delete(key)
            return None

        # Check if cached result is compatible with current config
        if metadata and not self._is_config_compatible(metadata):
            self.backend.delete(key)
            return None

        # For relation extraction, handle the context-aware result
        if step_name == BuildSteps.RELATION_EXTRACTION:
            cached_data = self.backend.get(key, dict)
            if cached_data and isinstance(cached_data, dict) and "result" in cached_data:
                # New format with entities context
                entities_context = cached_data.get("entities_context", [])
                # Check if backend has the method (specific to FileCache)
                if hasattr(self.backend, "get_relations_with_context"):
                    return self.backend.get_relations_with_context(  # type: ignore
                        cached_data["result"], entities_context, expected_type
                    )
                # Fallback to regular get for other backends
                return self.backend.get(key, expected_type)
            # Old format or direct result - fallback
            return self.backend.get(key, expected_type)

        return self.backend.get(key, expected_type)

    def save_step_result(self, step_name: str, input_data: Any, result: Any) -> None:
        """Save step result to cache.

        Args:
            step_name: Name of the build step
            input_data: Input data for the step
            result: Result to cache
        """
        if not self.config.enable_cache:
            return

        # Generate cache key and metadata
        key = self._generate_step_key(step_name, input_data)
        metadata = CacheMetadata(
            step_name=step_name,
            timestamp=datetime.now(),
            input_hash=self.backend.generate_key(input_data),
            config_hash=self._generate_config_hash(),
        )

        # For relation extraction step, we need to save entities as context
        # so that relations can be properly deserialized
        if (
            step_name == BuildSteps.RELATION_EXTRACTION
            and isinstance(input_data, tuple)
            and len(input_data) == 2
        ):
            _, entities = input_data  # Only need entities for context
            # Save result with entities context for proper relation deserialization
            result_with_context = {"result": result, "entities_context": entities}
            self.backend.set(key, result_with_context, metadata)
        else:
            self.backend.set(key, result, metadata)

    def invalidate_step(self, step_name: str) -> int:
        """Invalidate all cached results for a step.

        Args:
            step_name: Name of the step to invalidate

        Returns:
            Number of invalidated cache entries
        """
        return self.backend.clear(pattern=step_name)

    def invalidate_dependent_steps(self, step_name: str) -> int:
        """Invalidate all steps that depend on the given step.

        Args:
            step_name: Name of the step whose dependents to invalidate

        Returns:
            Number of invalidated cache entries
        """
        dependent_steps = BuildSteps.get_dependent_steps(step_name)
        total_invalidated = 0

        for dep_step in dependent_steps:
            total_invalidated += self.invalidate_step(dep_step)

        return total_invalidated

    def clear_all_cache(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of cleared entries
        """
        return self.backend.clear()

    def get_build_status(self) -> BuildStatus:
        """Get current build status.

        Returns:
            Current build status
        """
        if self._build_status is None:
            self._build_status = self._load_build_status()
        return self._build_status

    def update_build_status(
        self,
        current_step: Optional[str] = None,
        completed_step: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Update build status.

        Args:
            current_step: Currently executing step
            completed_step: Recently completed step
            error_message: Error message if any
        """
        status = self.get_build_status()

        if current_step is not None:
            status.current_step = current_step
            if status.started_at is None:
                status.started_at = datetime.now()

        if completed_step is not None:
            status.last_completed_step = completed_step
            step_index = BuildSteps.get_step_index(completed_step)
            if step_index >= 0:
                status.completed_steps = max(status.completed_steps, step_index + 1)

        if error_message is not None:
            status.error_message = error_message

        status.updated_at = datetime.now()
        self._save_build_status(status)

    def reset_build_status(self) -> None:
        """Reset build status to initial state."""
        self._build_status = BuildStatus()
        self._save_build_status(self._build_status)

    def get_cache_info(self) -> Dict[str, Any]:
        """Get comprehensive cache information.

        Returns:
            Dictionary with cache statistics and status
        """
        backend_info = self.backend.get_cache_info()
        status = self.get_build_status()

        return {
            "backend": backend_info,
            "build_status": status.to_dict(),
            "config": {
                "enabled": self.config.enable_cache,
                "ttl": self.config.cache_ttl,
                "auto_cleanup": self.config.auto_cleanup,
            },
        }

    def _generate_step_key(self, step_name: str, input_data: Any) -> str:
        """Generate cache key for a step."""
        # Include step name and config hash in key
        config_hash = self._generate_config_hash()
        data_hash = self.backend.generate_key(input_data)
        return f"{step_name}_{config_hash}_{data_hash}"

    def _generate_config_hash(self) -> str:
        """Generate hash for current configuration."""
        relevant_config = {
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
            "entity_confidence_threshold": self.config.entity_confidence_threshold,
            "relation_confidence_threshold": self.config.relation_confidence_threshold,
            "cluster_algorithm": self.config.cluster_algorithm,
            "min_cluster_size": self.config.min_cluster_size,
            "llm_model": self.config.llm_model,
        }
        return self.backend.generate_key(relevant_config)

    def _is_config_compatible(self, metadata: CacheMetadata) -> bool:
        """Check if cached result is compatible with current config."""
        current_config_hash = self._generate_config_hash()
        return metadata.config_hash == current_config_hash

    def _load_build_status(self) -> BuildStatus:
        """Load build status from cache."""
        try:
            status_data = self.backend.get("build_status", dict)
            if status_data:
                return BuildStatus.from_dict(status_data)
        except Exception:
            pass

        return BuildStatus()

    def _save_build_status(self, status: BuildStatus) -> None:
        """Save build status to cache."""
        try:
            self.backend.set("build_status", status.to_dict())
            self._build_status = status
        except Exception:
            pass

    # Document-level incremental update methods
    def get_file_hash(self, file_path: Union[str, Path]) -> str:
        """Calculate file hash for change detection."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"File does not exist for hash calculation: {file_path}")
            return ""

        logger.debug(f"Calculating hash for file: {file_path}")
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        file_hash = hasher.hexdigest()
        logger.debug(f"File hash calculated: {file_hash[:16]}... for {file_path}")
        return file_hash

    def get_document_status(
        self, file_path: Union[str, Path]
    ) -> Optional[DocumentProcessingStatus]:
        """Get document processing status."""
        file_key = str(file_path)
        logger.debug(f"Getting document status for: {file_path}")

        # Check memory cache first
        if file_key in self._document_status_cache:
            logger.debug(f"Found document status in memory cache for: {file_path}")
            return self._document_status_cache[file_key]

        # Load from persistent cache
        status_key = f"document_status_{self.backend.generate_key(file_key)}"
        status_data = self.backend.get(status_key, dict)

        if status_data:
            logger.debug(f"Found document status in persistent cache for: {file_path}")
            status = DocumentProcessingStatus.from_dict(status_data)
            self._document_status_cache[file_key] = status
            return status

        logger.debug(f"No document status found for: {file_path}")
        return None

    def update_document_status(
        self, file_path: Union[str, Path], status: DocumentProcessingStatus
    ) -> None:
        """Update document processing status."""
        file_key = str(file_path)
        logger.debug(
            f"Updating document status for {file_path}: status={status.processing_status}, "
            f"hash={status.file_hash[:16]}..."
        )

        # Update memory cache
        self._document_status_cache[file_key] = status

        # Save to persistent cache
        status_key = f"document_status_{self.backend.generate_key(file_key)}"
        self.backend.set(status_key, status.to_dict())

        logger.info(f"Document status updated successfully for: {file_path}")

    def is_document_processed(self, file_path: Union[str, Path]) -> bool:
        """Check if document has been processed and is up-to-date."""
        path = Path(file_path)
        if not path.exists():
            logger.warning(f"Document does not exist, cannot check processing status: {file_path}")
            return False

        status = self.get_document_status(file_path)
        if not status:
            logger.debug(f"No processing status found for document: {file_path}")
            return False

        if status.processing_status != "completed":
            logger.debug(
                f"Document processing not completed (status: {status.processing_status}): {file_path}"
            )
            return False

        # Check if file has been modified since last processing
        current_hash = self.get_file_hash(file_path)
        is_current = status.file_hash == current_hash

        if is_current:
            logger.debug(f"Document is up-to-date: {file_path}")
        else:
            logger.debug(f"Document has been modified since last processing: {file_path}")

        return is_current

    def get_processed_documents(
        self, documents: List[Union[str, Path]]
    ) -> tuple[List[Union[str, Path]], List[Union[str, Path]]]:
        """Separate documents into processed and unprocessed lists."""
        logger.info(f"Analyzing {len(documents)} documents for processing status")
        processed = []
        unprocessed = []

        for doc_path in documents:
            if self.is_document_processed(doc_path):
                processed.append(doc_path)
            else:
                unprocessed.append(doc_path)

        logger.info(
            f"Document analysis complete: {len(processed)} processed, {len(unprocessed)} unprocessed"
        )
        return processed, unprocessed

    def get_cached_document_results(self, documents: List[Union[str, Path]]) -> Dict[str, str]:
        """Get cached processing results for documents."""
        logger.info(f"Retrieving cached results for {len(documents)} documents")
        results = {}
        cache_hits = 0

        for doc_path in documents:
            status = self.get_document_status(doc_path)
            if status and status.processing_status == "completed" and status.extracted_text_hash:
                # Get cached text result
                text_key = f"document_text_{status.extracted_text_hash}"
                cached_text = self.backend.get(text_key, str)
                if cached_text:
                    results[str(doc_path)] = cached_text
                    cache_hits += 1
                    logger.debug(f"Retrieved cached text for document: {doc_path}")
                else:
                    logger.warning(
                        f"Status indicates completed but no cached text found for: {doc_path}"
                    )

        logger.info(f"Cache retrieval complete: {cache_hits}/{len(documents)} hits")
        return results

    def save_document_processing_result(
        self, file_path: Union[str, Path], extracted_text: str, processing_time: float
    ) -> None:
        """Save document processing result with status tracking."""
        path = Path(file_path)
        file_hash = self.get_file_hash(file_path)

        logger.info(
            f"Saving document processing result: {file_path}, "
            f"text_length={len(extracted_text)}, processing_time={processing_time:.2f}s"
        )

        # Generate text hash for content deduplication
        text_hash = hashlib.sha256(extracted_text.encode()).hexdigest()
        logger.debug(f"Generated text hash: {text_hash[:16]}... for {file_path}")

        # Save extracted text
        text_key = f"document_text_{text_hash}"
        self.backend.set(text_key, extracted_text)
        logger.debug(f"Cached extracted text with key: {text_key}")

        # Update document status
        status = DocumentProcessingStatus(
            file_path=str(file_path),
            file_hash=file_hash,
            last_modified=datetime.fromtimestamp(path.stat().st_mtime),
            processing_status="completed",
            extracted_text_hash=text_hash,
            processing_time=processing_time,
        )

        self.update_document_status(file_path, status)
        logger.info(f"Document processing result saved successfully: {file_path}")

    def get_document_processing_summary(self) -> Dict[str, Any]:
        """Get summary of document processing status."""
        logger.debug("Generating document processing summary")
        all_status = []

        # Get all document status keys
        keys = self.backend.list_keys("document_status_")
        logger.debug(f"Found {len(keys)} document status entries")

        for key in keys:
            status_data = self.backend.get(key, dict)
            if status_data:
                all_status.append(DocumentProcessingStatus.from_dict(status_data))

        summary = {
            "total_documents": len(all_status),
            "completed": len([s for s in all_status if s.processing_status == "completed"]),
            "failed": len([s for s in all_status if s.processing_status == "failed"]),
            "processing": len([s for s in all_status if s.processing_status == "processing"]),
            "pending": len([s for s in all_status if s.processing_status == "pending"]),
            "total_processing_time": sum(s.processing_time or 0 for s in all_status),
        }

        logger.info(
            f"Document processing summary: {summary['completed']}/{summary['total_documents']} completed, "
            f"total time: {summary['total_processing_time']:.2f}s"
        )
        return summary

    def clear_document_cache(self, file_path: Optional[Union[str, Path]] = None) -> int:
        """Clear document processing cache."""
        if file_path:
            # Clear specific document
            logger.info(f"Clearing document cache for: {file_path}")
            file_key = str(file_path)
            status_key = f"document_status_{self.backend.generate_key(file_key)}"

            status = self.get_document_status(file_path)
            deleted_count = 0

            if status:
                # Remove text cache
                if status.extracted_text_hash:
                    text_key = f"document_text_{status.extracted_text_hash}"
                    if self.backend.delete(text_key):
                        deleted_count += 1
                        logger.debug(f"Deleted cached text with key: {text_key}")

                # Remove status cache
                if self.backend.delete(status_key):
                    deleted_count += 1
                    logger.debug(f"Deleted status cache with key: {status_key}")

                # Remove from memory cache
                if file_key in self._document_status_cache:
                    del self._document_status_cache[file_key]
                    logger.debug(f"Removed from memory cache: {file_path}")

                logger.info(
                    f"Document cache cleared for {file_path}: {deleted_count} entries deleted"
                )

            if not status:
                logger.debug(f"No cache entries found for document: {file_path}")

            return deleted_count

        # Clear all document cache
        logger.info("Clearing all document processing cache")
        self._document_status_cache.clear()
        status_count = self.backend.clear("document_status_")
        text_count = self.backend.clear("document_text_")
        total_deleted = status_count + text_count

        logger.info(
            f"All document cache cleared: {total_deleted} entries deleted ({status_count} status + {text_count} text)"
        )
        return total_deleted
