"""
Relation extraction handler for knowledge graph builder.
"""

import inspect
from typing import Dict, List

from ...base.entities import Entity
from ...base.relations import Relation
from ...base.text import TextChunk
from ...builder.cache import CacheManager
from ...builder.extractors import RelationExtractor
from ...config import BuildSteps
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

        # Distribute entities to documents (simplified approach - in practice might need more sophisticated mapping)
        for entity in entities:
            # For now, associate entity with all documents (this could be optimized)
            for doc_id in doc_chunks_map:
                doc_entities_map[doc_id].append(entity)
        # pylint: disable=too-many-nested-blocks
        for doc_id, doc_chunks in doc_chunks_map.items():
            doc_entities = doc_entities_map[doc_id]

            # Generate cache key for this document's relations
            doc_input = (doc_chunks, doc_entities)
            doc_relations_key = (
                f"doc_relations_{self.cache_manager.backend.generate_key(doc_input)}"
            )
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
                logger.debug(
                    f"Using cached relations for {doc_id}: {len(cached_relations)} relations"
                )
            else:
                chunks_to_process.extend(doc_chunks)

        # Process uncached chunks
        if chunks_to_process:
            logger.info(
                f"Processing {len(chunks_to_process)} uncached chunks for relation extraction"
            )

            # Group chunks to process by document
            doc_chunks_to_process: Dict[str, List[TextChunk]] = {}
            for chunk in chunks_to_process:
                doc_id = chunk.source
                if doc_id not in doc_chunks_to_process:
                    doc_chunks_to_process[doc_id] = []
                doc_chunks_to_process[doc_id].append(chunk)

            for doc_id, doc_chunks in doc_chunks_to_process.items():
                doc_entities = doc_entities_map[doc_id]

                # Extract relations for this document's chunks and entities
                extraction_result = self.relation_extractor.extract(doc_chunks, doc_entities)
                if inspect.iscoroutine(extraction_result):
                    doc_relations = await extraction_result
                else:
                    doc_relations = extraction_result if extraction_result is not None else []

                all_relations.extend(doc_relations)

                # Cache relations for this document with entity context
                doc_input = (doc_chunks, doc_entities)
                doc_relations_key = (
                    f"doc_relations_{self.cache_manager.backend.generate_key(doc_input)}"
                )
                # Save relations with entity context for proper deserialization
                result_with_context = {"result": doc_relations, "entities_context": doc_entities}
                self.cache_manager.backend.set(doc_relations_key, result_with_context)
                logger.debug(f"Cached {len(doc_relations)} relations for {doc_id}")

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
            self.cache_manager.save_step_result(
                BuildSteps.RELATION_EXTRACTION, cache_input, unique_relations
            )

        return unique_relations

    async def _extract_all_relations(
        self, chunks: List[TextChunk], entities: List[Entity]
    ) -> List[Relation]:
        """Extract relations from all chunks without caching (fallback method)."""
        relations = await self.relation_extractor.extract(chunks, entities)
        return relations
