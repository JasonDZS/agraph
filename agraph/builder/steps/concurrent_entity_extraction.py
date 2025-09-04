"""
Concurrent entity extraction step implementation.
"""

import asyncio
from typing import Any, List

from ...base.models.entities import Entity
from ...base.models.text import TextChunk
from ...config import BuildSteps
from ...logger import logger
from ..handler.entity_handler import EntityHandler
from .base import StepResult
from .concurrent_base import ConcurrentBuildStep, ParallelProcessingMixin
from .context import BuildContext


class ConcurrentEntityExtractionStep(ConcurrentBuildStep[Entity], ParallelProcessingMixin):
    """
    Concurrent entity extraction step with batch processing and resource management.
    """
    
    def __init__(self, entity_handler: EntityHandler, cache_manager):
        """
        Initialize concurrent entity extraction step.
        
        Args:
            entity_handler: Handler for entity extraction operations
            cache_manager: Cache manager instance
        """
        super().__init__(BuildSteps.ENTITY_EXTRACTION, cache_manager)
        self.entity_handler = entity_handler
    
    def _get_required_resources(self) -> List[str]:
        """Return required resources for entity extraction."""
        return ["entity_extraction", "llm_calls", "documents"]
    
    def _get_batch_size(self) -> int:
        """Return optimal batch size for entity extraction."""
        return self.concurrency_manager.config.entity_batch_size
    
    def _prepare_items_for_batching(self, context: BuildContext) -> List[Any]:
        """
        Prepare chunks for batch processing.
        
        Args:
            context: Build context containing chunks
            
        Returns:
            List of text chunks ready for processing
        """
        chunks = context.chunks
        if not chunks:
            return []
        
        # Validate chunks
        valid_chunks = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, TextChunk):
                logger.warning(f"Invalid chunk at index {i}: expected TextChunk, got {type(chunk)}")
                continue
            if not chunk.content.strip():
                logger.warning(f"Empty chunk at index {i}, skipping")
                continue
            valid_chunks.append(chunk)
        
        logger.info(f"Prepared {len(valid_chunks)} valid chunks for entity extraction")
        return valid_chunks
    
    async def _process_batch(self, batch: List[TextChunk], context: BuildContext) -> List[Entity]:
        """
        Process a batch of text chunks for entity extraction.
        
        Args:
            batch: Batch of text chunks to process
            context: Build context
            
        Returns:
            List of extracted entities from the batch
        """
        try:
            # Group chunks by document for better LLM context
            doc_batches = self._group_chunks_by_document(batch)
            
            # Process each document batch concurrently
            document_results = await self.parallel_map(
                list(doc_batches.items()),
                lambda doc_batch: self._process_document_batch(doc_batch[0], doc_batch[1], context),
                max_concurrency=self.concurrency_manager.config.max_concurrent_documents,
                timeout=self.concurrency_manager.config.llm_call_timeout
            )
            
            # Flatten and deduplicate results
            all_entities = []
            for result in document_results:
                if isinstance(result, Exception):
                    logger.error(f"Document batch processing error: {result}")
                    continue
                elif isinstance(result, list):
                    all_entities.extend(result)
            
            # Remove duplicates while preserving order
            unique_entities = self._deduplicate_entities(all_entities)
            
            logger.debug(f"Processed batch: {len(batch)} chunks → {len(unique_entities)} unique entities")
            return unique_entities
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}")
            return []
    
    def _group_chunks_by_document(self, chunks: List[TextChunk]) -> dict:
        """
        Group text chunks by source document.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Dictionary mapping document ID to list of chunks
        """
        doc_groups = {}
        for chunk in chunks:
            doc_id = chunk.source or "unknown"
            if doc_id not in doc_groups:
                doc_groups[doc_id] = []
            doc_groups[doc_id].append(chunk)
        
        return doc_groups
    
    async def _process_document_batch(
        self, 
        doc_id: str, 
        doc_chunks: List[TextChunk], 
        context: BuildContext
    ) -> List[Entity]:
        """
        Process all chunks from a single document.
        
        Args:
            doc_id: Document identifier
            doc_chunks: List of chunks from the document
            context: Build context
            
        Returns:
            List of entities extracted from the document
        """
        try:
            # Check cache for this document
            cache_key = f"doc_entities_{doc_id}_{hash(tuple(chunk.content for chunk in doc_chunks))}"
            cached_entities = self.cache_manager.backend.get(cache_key, list)
            
            if cached_entities and context.use_cache:
                logger.debug(f"Using cached entities for document {doc_id}: {len(cached_entities)} entities")
                return cached_entities
            
            # Extract entities using the handler
            entities = await self.entity_handler.entity_extractor.extract(doc_chunks)
            
            # Validate extracted entities
            validated_entities = []
            for entity in entities:
                if isinstance(entity, Entity):
                    validated_entities.append(entity)
                else:
                    logger.warning(f"Invalid entity type from extractor: {type(entity)}")
            
            # Cache results for this document
            if context.use_cache:
                self.cache_manager.backend.set(cache_key, validated_entities)
                logger.debug(f"Cached {len(validated_entities)} entities for document {doc_id}")
            
            return validated_entities
            
        except Exception as e:
            logger.error(f"Document {doc_id} processing failed: {str(e)}")
            return []
    
    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate entities while preserving order.
        
        Args:
            entities: List of entities potentially containing duplicates
            
        Returns:
            List of unique entities
        """
        seen = set()
        unique_entities = []
        
        for entity in entities:
            # Create entity key for deduplication
            entity_key = (
                entity.name.lower().strip(),
                entity.entity_type if hasattr(entity, 'entity_type') else 'unknown'
            )
            
            if entity_key not in seen:
                seen.add(entity_key)
                unique_entities.append(entity)
        
        return unique_entities
    
    def _validate_batch_results(self, results: List[Entity]) -> bool:
        """
        Validate entity extraction results.
        
        Args:
            results: Results from batch processing
            
        Returns:
            True if results are valid
        """
        if not isinstance(results, list):
            logger.error("Entity extraction results must be a list")
            return False
        
        for i, entity in enumerate(results):
            if not isinstance(entity, Entity):
                logger.error(f"Invalid entity at index {i}: expected Entity, got {type(entity)}")
                return False
            
            # Validate required entity fields
            if not hasattr(entity, 'name') or not entity.name:
                logger.error(f"Entity at index {i} missing required 'name' field")
                return False
        
        return True
    
    async def _execute_step(self, context: BuildContext) -> StepResult[List[Entity]]:
        """
        Execute concurrent entity extraction with enhanced metadata.
        
        Args:
            context: Build context
            
        Returns:
            StepResult containing extracted entities and detailed metadata
        """
        # Execute the concurrent processing
        result = await super()._execute_step(context)
        
        if result.is_success() and result.data:
            entities = result.data
            
            # Calculate additional metrics
            entity_types = set()
            confidence_scores = []
            text_chunk_coverage = set()
            
            for entity in entities:
                if hasattr(entity, 'entity_type') and entity.entity_type:
                    entity_types.add(entity.entity_type)
                
                if hasattr(entity, 'confidence') and entity.confidence is not None:
                    confidence_scores.append(entity.confidence)
                
                if hasattr(entity, 'text_chunks') and entity.text_chunks:
                    text_chunk_coverage.update(entity.text_chunks)
            
            # Enhanced metadata
            enhanced_metadata = result.metadata.copy() if result.metadata else {}
            enhanced_metadata.update({
                "unique_entity_types": len(entity_types),
                "entity_types": list(entity_types),
                "average_confidence": sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0,
                "confidence_distribution": {
                    "high": len([s for s in confidence_scores if s >= 0.8]),
                    "medium": len([s for s in confidence_scores if 0.5 <= s < 0.8]),
                    "low": len([s for s in confidence_scores if s < 0.5])
                },
                "text_chunk_coverage": len(text_chunk_coverage),
                "extraction_efficiency": len(entities) / enhanced_metadata.get("items_processed", 1) if enhanced_metadata.get("items_processed", 0) > 0 else 0
            })
            
            return StepResult.success_result(entities, metadata=enhanced_metadata)
        
        return result