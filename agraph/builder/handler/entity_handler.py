"""
Entity extraction handler for knowledge graph builder.
"""

import asyncio
import inspect
from typing import Any, Dict, List, Optional

from ...base.models.entities import Entity
from ...base.models.positioning import AlignmentStatus, CharInterval, Position
from ...base.models.text import TextChunk
from ...builder.cache import CacheManager
from ...builder.extractors import EntityExtractor
from ...config import BuildSteps, get_settings
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

    async def extract_entities_from_chunks(self, chunks: List[TextChunk], use_cache: bool = True) -> List[Entity]:
        """Extract entities from text chunks with incremental processing support.

        Args:
            chunks: List of text chunks
            use_cache: Whether to use caching

        Returns:
            List of extracted entities
        """
        logger.info(f"Extracting entities from {len(chunks)} chunks using {type(self.entity_extractor).__name__}")

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

                # Update text chunks with cached entity references and positioning
                chunk_map = {chunk.id: chunk for chunk in doc_chunks}
                self._integrate_positioning_info(cached_entities, chunk_map)

                logger.debug(f"Using cached entities for {doc_id}: {len(cached_entities)} entities")
            else:
                chunks_to_process.extend(doc_chunks)

        # Process uncached chunks
        if chunks_to_process:
            logger.info(f"Processing {len(chunks_to_process)} uncached chunks for entity extraction")

            # Group chunks to process by document for caching
            doc_chunks_to_process: Dict[str, List[TextChunk]] = {}
            for chunk in chunks_to_process:
                doc_id = chunk.source
                if doc_id not in doc_chunks_to_process:
                    doc_chunks_to_process[doc_id] = []
                doc_chunks_to_process[doc_id].append(chunk)

            # Use concurrent processing with semaphore for rate limiting
            settings = get_settings()
            max_concurrent = settings.max_current
            semaphore = asyncio.Semaphore(max_concurrent)

            logger.info(f"Processing {len(doc_chunks_to_process)} documents with max {max_concurrent} concurrent tasks")

            async def process_document(doc_id: str, doc_chunks: List[TextChunk]) -> Any:
                """Process a single document with concurrency control."""
                async with semaphore:
                    try:
                        # Extract entities for this document's chunks
                        extraction_result = self.entity_extractor.extract(doc_chunks)
                        if inspect.iscoroutine(extraction_result):
                            doc_entities = await extraction_result
                        else:
                            doc_entities = extraction_result if extraction_result is not None else []

                        # Update text chunks with entity references and set positioning
                        chunk_map = {chunk.id: chunk for chunk in doc_chunks}
                        self._integrate_positioning_info(doc_entities, chunk_map)

                        # Cache entities for this document
                        doc_entities_id = self.cache_manager.backend.generate_key([c.content for c in doc_chunks])
                        doc_chunks_key = f"doc_entities_{doc_entities_id}"
                        self.cache_manager.backend.set(doc_chunks_key, doc_entities)
                        logger.debug(f"Cached {len(doc_entities)} entities for {doc_id}")

                        return doc_entities
                    except Exception as e:
                        logger.error(f"Error processing document {doc_id}: {e}")
                        return []

            # Process all documents concurrently
            tasks = [process_document(doc_id, doc_chunks) for doc_id, doc_chunks in doc_chunks_to_process.items()]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Task failed with exception: {result}")
                elif isinstance(result, list):
                    all_entities.extend(result)

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
            self.cache_manager.save_step_result(BuildSteps.ENTITY_EXTRACTION, chunks, unique_entities)

        return unique_entities

    async def _extract_all_entities(self, chunks: List[TextChunk]) -> List[Entity]:
        """Extract entities from all chunks without caching (fallback method)."""
        entities = await self.entity_extractor.extract(chunks)

        # Integrate positioning info for non-cached extraction
        chunk_map = {chunk.id: chunk for chunk in chunks}
        self._integrate_positioning_info(entities, chunk_map)

        return entities

    def _integrate_positioning_info(self, entities: List[Entity], chunk_map: Dict[str, TextChunk]) -> None:
        """Integrate positioning information from text chunks into entities.

        This method attempts to find the entity text within the source text chunks
        and sets positioning information based on text matching.

        Args:
            entities: List of entities to update with positioning info
            chunk_map: Mapping of chunk IDs to TextChunk objects
        """
        for entity in entities:
            # Establish bidirectional linking
            for chunk_id in entity.text_chunks:
                if chunk_id in chunk_map:
                    chunk_map[chunk_id].entities.add(entity.id)

            # Set positioning information based on text matching
            if entity.text_chunks and entity.name:
                # Find the first chunk that contains this entity
                for chunk_id in entity.text_chunks:
                    if chunk_id in chunk_map:
                        chunk = chunk_map[chunk_id]
                        position = self._find_entity_position_in_chunk(entity, chunk)
                        if position and position.is_positioned:
                            entity.set_position(position)
                            break  # Use first successful match

    def _find_entity_position_in_chunk(self, entity: Entity, chunk: TextChunk) -> Optional["Position"]:
        """Find entity position within a text chunk.

        Args:
            entity: Entity to locate
            chunk: Text chunk to search in

        Returns:
            Position object if entity is found, None otherwise
        """
        if not entity.name or not chunk.content:
            return None

        entity_text = entity.name.lower().strip()
        chunk_text = chunk.content.lower()

        # Try exact match first
        start_pos = chunk_text.find(entity_text)
        if start_pos != -1:
            end_pos = start_pos + len(entity_text)
            char_interval = CharInterval(start_pos=start_pos, end_pos=end_pos)

            # Calculate confidence based on entity name length and context
            confidence = min(1.0, len(entity_text) / max(10, len(entity_text)))
            confidence = max(0.3, confidence)  # Minimum confidence threshold

            return Position(
                char_interval=char_interval,
                alignment_status=AlignmentStatus.MATCH_EXACT,
                confidence=confidence,
                source_context=f"Found in chunk {chunk.id}",
            )

        # Try matching aliases if exact match fails
        for alias in entity.aliases:
            if alias:
                alias_text = alias.lower().strip()
                start_pos = chunk_text.find(alias_text)
                if start_pos != -1:
                    end_pos = start_pos + len(alias_text)
                    char_interval = CharInterval(start_pos=start_pos, end_pos=end_pos)

                    confidence = min(0.8, len(alias_text) / max(10, len(alias_text)))
                    confidence = max(0.2, confidence)

                    return Position(
                        char_interval=char_interval,
                        alignment_status=AlignmentStatus.MATCH_FUZZY,
                        confidence=confidence,
                        source_context=f"Found alias '{alias}' in chunk {chunk.id}",
                    )

        return None
