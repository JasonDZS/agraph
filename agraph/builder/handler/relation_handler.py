"""
Relation extraction handler for knowledge graph builder.
"""

import asyncio
import inspect
from typing import Any, Dict, List

from ...base.models.entities import Entity
from ...base.models.relations import Relation
from ...base.models.text import TextChunk
from ...builder.cache import CacheManager
from ...builder.extractors import RelationExtractor
from ...config import BuildSteps, get_settings
from ...logger import logger


class RelationHandler:
    """Handles relation extraction with incremental processing support."""

    def __init__(self, cache_manager: CacheManager, relation_extractor: RelationExtractor):
        """Initialize relation handler.

        Args:
            cache_manager: Cache manager instance
            relation_extractor: Relation extractor instance
        """
        self.cache_manager = cache_manager
        self.relation_extractor = relation_extractor

    async def extract_relations_from_chunks(
        self, chunks: List[TextChunk], entities: List[Entity], use_cache: bool = True
    ) -> List[Relation]:
        """Extract relations from text chunks and entities with incremental processing support.

        Args:
            chunks: List of text chunks
            entities: List of entities
            use_cache: Whether to use caching

        Returns:
            List of extracted relations
        """
        logger.info(
            f"Extracting relations from {len(chunks)} chunks and {len(entities)} entities using {type(self.relation_extractor).__name__}"
        )

        if not use_cache:
            return await self._extract_all_relations(chunks, entities)

        # Try incremental approach: group chunks by document and check cache
        all_relations = []
        chunks_to_process = []
        cached_relations_count = 0

        # Group chunks by source document
        doc_chunks_map: Dict[str, List[TextChunk]] = {}
        for chunk in chunks:
            doc_id = chunk.source
            if doc_id not in doc_chunks_map:
                doc_chunks_map[doc_id] = []
            doc_chunks_map[doc_id].append(chunk)

        # Group entities by document (based on which chunks they came from)
        doc_entities_map: Dict[str, List[Entity]] = {}
        for doc_id in doc_chunks_map:
            doc_entities_map[doc_id] = []

        # Distribute entities to documents based on text chunk references
        for entity in entities:
            # Find which documents this entity belongs to based on text_chunks
            entity_docs = set()
            if hasattr(entity, "text_chunks") and entity.text_chunks:
                # Map entity to documents based on its text chunks
                for chunk_id in entity.text_chunks:
                    for chunk in chunks:
                        if chunk.id == chunk_id:
                            entity_docs.add(chunk.source)
                            break

            # If entity has no text chunk references, associate with all documents
            if not entity_docs:
                entity_docs = set(doc_chunks_map.keys())

            # Add entity to relevant document mappings
            for doc_id in entity_docs:
                if doc_id in doc_entities_map:
                    doc_entities_map[doc_id].append(entity)
        # pylint: disable=too-many-nested-blocks
        for doc_id, doc_chunks in doc_chunks_map.items():
            doc_entities = doc_entities_map[doc_id]

            # Generate stable cache key for this document's relations
            doc_chunks_content = sorted([c.content.strip() for c in doc_chunks if c.content])
            doc_entities_content = sorted(
                [
                    (
                        e.name.strip() if hasattr(e, "name") and e.name else "",
                        (
                            e.type.value
                            if hasattr(e, "type") and hasattr(e.type, "value")
                            else str(e.type) if hasattr(e, "type") else ""
                        ),
                    )
                    for e in doc_entities
                ]
            )
            doc_input = (tuple(doc_chunks_content), tuple(doc_entities_content))
            doc_relations_key = f"doc_relations_{self.cache_manager.backend.generate_key(doc_input)}"
            cached_data = self.cache_manager.backend.get(doc_relations_key, dict)

            if cached_data is not None:
                # Handle context-aware cached data (new format)
                if isinstance(cached_data, dict) and "result" in cached_data:
                    entities_context = cached_data.get("entities_context", [])
                    # Create entities map for relation deserialization - handle cached entity format
                    entities_map = {}
                    for entity_data in entities_context:
                        if isinstance(entity_data, dict) and "_data" in entity_data:
                            entity = Entity.from_dict(entity_data["_data"])
                            entities_map[entity.id] = entity
                        elif hasattr(entity_data, "id"):
                            entities_map[entity_data.id] = entity_data

                    # Convert cached relation dicts to Relation objects with entity context
                    cached_relations = [
                        (
                            Relation.from_dict(rel_data["_data"], entities_map=entities_map)
                            if isinstance(rel_data, dict) and "_data" in rel_data
                            else rel_data
                        )
                        for rel_data in cached_data["result"]
                    ]
                else:
                    # Legacy format - relations without entity context (list of relation dicts)
                    cached_relations = []
                    if isinstance(cached_data, list):
                        for rel_data in cached_data:
                            if isinstance(rel_data, dict) and "_data" in rel_data:
                                # This is a cached relation dict - deserialize without entity context
                                # (head_entity and tail_entity will be None)
                                relation = Relation.from_dict(rel_data["_data"])
                                cached_relations.append(relation)
                            elif hasattr(rel_data, "head_entity"):
                                # This is already a Relation object
                                cached_relations.append(rel_data)
                    else:
                        cached_relations = []

                all_relations.extend(cached_relations)
                cached_relations_count += len(cached_relations)

                # Update text chunks with cached relation references (bidirectional linking)
                chunk_map = {chunk.id: chunk for chunk in doc_chunks}
                for relation in cached_relations:
                    for chunk_id in relation.text_chunks:
                        if chunk_id in chunk_map:
                            chunk_map[chunk_id].relations.add(relation.id)

                logger.debug(f"Using cached relations for {doc_id}: {len(cached_relations)} relations")
            else:
                chunks_to_process.extend(doc_chunks)

        # Process uncached chunks
        if chunks_to_process:
            logger.info(f"Processing {len(chunks_to_process)} uncached chunks for relation extraction")

            # Group chunks to process by document
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
                        doc_entities = doc_entities_map[doc_id]

                        # Extract relations for this document's chunks and entities
                        extraction_result = self.relation_extractor.extract(doc_chunks, doc_entities)
                        if inspect.iscoroutine(extraction_result):
                            doc_relations = await extraction_result
                        else:
                            doc_relations = extraction_result if extraction_result is not None else []

                        # Update text chunks with relation references (bidirectional linking)
                        chunk_map = {chunk.id: chunk for chunk in doc_chunks}
                        for relation in doc_relations:
                            for chunk_id in relation.text_chunks:
                                if chunk_id in chunk_map:
                                    chunk_map[chunk_id].relations.add(relation.id)

                        # Cache relations for this document with entity context (use stable content-based keys)
                        doc_chunks_content = sorted([c.content.strip() for c in doc_chunks if c.content])
                        doc_entities_content = sorted(
                            [
                                (
                                    e.name.strip() if hasattr(e, "name") and e.name else "",
                                    (
                                        e.type.value
                                        if hasattr(e, "type") and hasattr(e.type, "value")
                                        else str(e.type) if hasattr(e, "type") else ""
                                    ),
                                )
                                for e in doc_entities
                            ]
                        )
                        doc_input = (tuple(doc_chunks_content), tuple(doc_entities_content))
                        doc_relations_key = f"doc_relations_{self.cache_manager.backend.generate_key(doc_input)}"
                        # Save relations with entity context for proper deserialization
                        result_with_context = {
                            "result": doc_relations,
                            "entities_context": doc_entities,
                        }
                        self.cache_manager.backend.set(doc_relations_key, result_with_context)
                        logger.debug(f"Cached {len(doc_relations)} relations for {doc_id}")

                        return doc_relations
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
                    all_relations.extend(result)

        # Remove duplicate relations while preserving order
        unique_relations = []
        seen_relations = set()
        for relation in all_relations:
            relation_key = (
                getattr(relation, "source", ""),
                getattr(relation, "target", ""),
                getattr(relation, "type", ""),
                getattr(relation, "description", ""),
            )
            if relation_key not in seen_relations:
                unique_relations.append(relation)
                seen_relations.add(relation_key)

        logger.info(
            f"Relation extraction completed - found {len(unique_relations)} unique relations "
            f"({cached_relations_count} from cache, {len(all_relations) - cached_relations_count} newly extracted)"
        )

        # Save step-level result for compatibility
        if use_cache:
            cache_input = (chunks, entities)
            self.cache_manager.save_step_result(BuildSteps.RELATION_EXTRACTION, cache_input, unique_relations)

        return unique_relations

    async def _extract_all_relations(self, chunks: List[TextChunk], entities: List[Entity]) -> List[Relation]:
        """Extract relations from all chunks without caching (fallback method)."""
        relations = await self.relation_extractor.extract(chunks, entities)
        return relations
