"""
Entity extraction handler for knowledge graph builder.
"""

import asyncio
import inspect
from typing import Dict, List

from ...base.entities import Entity
from ...base.text import TextChunk
from ...builder.cache import CacheManager
from ...builder.extractors import EntityExtractor
from ...config import BuildSteps
from ...logger import logger


class EntityHandler:
    """Handles entity extraction with incremental processing support."""

    def __init__(self, cache_manager: CacheManager, entity_extractor: EntityExtractor):
        """Initialize entity handler.

        Args:
            cache_manager: Cache manager instance
            entity_extractor: Entity extractor instance
        """
        self.cache_manager = cache_manager
        self.entity_extractor = entity_extractor

    async def extract_entities_from_chunks(
        self, chunks: List[TextChunk], use_cache: bool = True
    ) -> List[Entity]:
        """Extract entities from text chunks with incremental processing support.

        Args:
            chunks: List of text chunks
            use_cache: Whether to use caching

        Returns:
            List of extracted entities
        """
        logger.info(
            f"Extracting entities from {len(chunks)} chunks using {type(self.entity_extractor).__name__}"
        )

        if not use_cache:
            return await self._extract_all_entities(chunks)

        # Try incremental approach: group chunks by document and check cache
        all_entities = []
        chunks_to_process = []
        cached_entities_count = 0

        # Group chunks by source document
        doc_chunks_map: Dict[str, List[TextChunk]] = {}
        for chunk in chunks:
            doc_id = chunk.source
            if doc_id not in doc_chunks_map:
                doc_chunks_map[doc_id] = []
            doc_chunks_map[doc_id].append(chunk)

        for doc_id, doc_chunks in doc_chunks_map.items():
            # Generate cache key for this document's chunks
            doc_chunks_key = f"doc_entities_{self.cache_manager.backend.generate_key([c.content for c in doc_chunks])}"
            cached_entities = self.cache_manager.backend.get(doc_chunks_key, list)

            if cached_entities is not None:
                all_entities.extend(cached_entities)
                cached_entities_count += len(cached_entities)

                # Update text chunks with cached entity references (bidirectional linking)
                chunk_map = {chunk.id: chunk for chunk in doc_chunks}
                for entity in cached_entities:
                    for chunk_id in entity.text_chunks:
                        if chunk_id in chunk_map:
                            chunk_map[chunk_id].entities.add(entity.id)

                logger.debug(f"Using cached entities for {doc_id}: {len(cached_entities)} entities")
            else:
                chunks_to_process.extend(doc_chunks)

        # Process uncached chunks
        if chunks_to_process:
            logger.info(
                f"Processing {len(chunks_to_process)} uncached chunks for entity extraction"
            )

            # Group chunks to process by document for caching
            doc_chunks_to_process: Dict[str, List[TextChunk]] = {}
            for chunk in chunks_to_process:
                doc_id = chunk.source
                if doc_id not in doc_chunks_to_process:
                    doc_chunks_to_process[doc_id] = []
                doc_chunks_to_process[doc_id].append(chunk)

            for doc_id, doc_chunks in doc_chunks_to_process.items():
                # Extract entities for this document's chunks
                extraction_result = self.entity_extractor.extract(doc_chunks)
                if inspect.iscoroutine(extraction_result):
                    try:
                        _ = asyncio.get_event_loop()
                        doc_entities = await extraction_result
                    except RuntimeError:
                        # 降级到同步调用
                        doc_entities = []
                        logger.warning("无法在当前事件循环中执行异步实体提取，跳过此批次")
                else:
                    doc_entities = extraction_result if extraction_result is not None else []

                all_entities.extend(doc_entities)

                # Update text chunks with entity references (bidirectional linking)
                chunk_map = {chunk.id: chunk for chunk in doc_chunks}
                for entity in doc_entities:
                    for chunk_id in entity.text_chunks:
                        if chunk_id in chunk_map:
                            chunk_map[chunk_id].entities.add(entity.id)

                # Cache entities for this document
                doc_entities_id = self.cache_manager.backend.generate_key(
                    [c.content for c in doc_chunks]
                )
                doc_chunks_key = f"doc_entities_{doc_entities_id}"
                self.cache_manager.backend.set(doc_chunks_key, doc_entities)
                logger.debug(f"Cached {len(doc_entities)} entities for {doc_id}")

        # Remove duplicates while preserving order (entities might overlap across documents)
        unique_entities = []
        seen_entities = set()
        for entity in all_entities:
            entity_key = (
                (entity.name.lower(), entity.type)
                if hasattr(entity, "name") and hasattr(entity, "type")
                else str(entity)
            )
            if entity_key not in seen_entities:
                unique_entities.append(entity)
                seen_entities.add(entity_key)

        logger.info(
            f"Entity extraction completed - found {len(unique_entities)} unique entities "
            f"({cached_entities_count} from cache, {len(all_entities) - cached_entities_count} newly extracted)"
        )

        # Save step-level result for compatibility
        if use_cache:
            self.cache_manager.save_step_result(
                BuildSteps.ENTITY_EXTRACTION, chunks, unique_entities
            )

        return unique_entities

    async def _extract_all_entities(self, chunks: List[TextChunk]) -> List[Entity]:
        """Extract entities from all chunks without caching (fallback method)."""
        entities = await self.entity_extractor.extract(chunks)
        return entities
