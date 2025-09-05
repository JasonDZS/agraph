"""
Optimized manager classes for knowledge graph operations with indexing support.

This module provides enhanced manager classes that use indexing and caching
to significantly improve performance for common operations.
"""

import time
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.types import EntityType, RelationType
from ..infrastructure.cache import CacheManager
from ..infrastructure.indexes import IndexManager
from ..models.clusters import Cluster
from ..models.entities import Entity
from ..models.relations import Relation
from ..models.text import TextChunk


class OptimizedEntityManager:
    """
    Optimized manager for entity operations using indexing and caching.

    This manager provides significant performance improvements over the original
    EntityManager by using indexes for fast lookups and caching for expensive operations.
    """

    def __init__(
        self,
        entities: Dict[str, Entity],
        touch_callback: Callable[[], None],
        index_manager: Optional[IndexManager] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        """
        Initialize the OptimizedEntityManager.

        Args:
            entities: Dictionary of entities
            touch_callback: Callback to update graph timestamp
            index_manager: Index manager for fast lookups
            cache_manager: Cache manager for expensive operations
        """
        self.entities = entities
        self._touch = touch_callback
        self.index_manager = index_manager or IndexManager()
        self.cache_manager = cache_manager or CacheManager()

        # Performance metrics
        self._metrics = {"operations_count": 0, "cache_hits": 0, "index_hits": 0, "total_time": 0.0}

    def _record_operation(self, operation_name: str, start_time: float) -> None:
        """Record operation metrics."""
        self._metrics["operations_count"] += 1
        self._metrics["total_time"] += time.time() - start_time

    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the knowledge graph with index updates."""
        start_time = time.time()

        # Add to main storage
        self.entities[entity.id] = entity

        # Update indexes
        self.index_manager.add_entity_to_type_index(entity.id, entity.entity_type)

        # Add entity-cluster relationships to index
        if hasattr(entity, "text_chunks") and entity.text_chunks:
            for chunk_id in entity.text_chunks:
                self.index_manager.add_entity_to_text_chunk_index(entity.id, chunk_id)

        # Invalidate relevant caches
        self.cache_manager.invalidate_by_tags({"entities", f"entity_type_{entity.entity_type}"})

        self._touch()
        self._record_operation("add_entity", start_time)

    def remove_entity(
        self,
        entity_id: str,
        relations: Dict[str, Relation],
        clusters: Dict[str, Cluster],
        text_chunks: Dict[str, TextChunk],
    ) -> bool:
        """Remove an entity using indexes for fast cascading updates."""
        start_time = time.time()

        if entity_id not in self.entities:
            return False

        entity = self.entities[entity_id]

        # Use index to efficiently find and remove related objects
        removed_data = self.index_manager.remove_entity_from_all_indexes(entity_id, entity.entity_type)

        # Remove relations using index information
        for relation_id in removed_data["relations"]:
            relations.pop(relation_id, None)

        # Update clusters using index information
        for cluster_id in removed_data["clusters"]:
            cluster = clusters.get(cluster_id)
            if cluster:
                cluster.remove_entity(entity_id)

        # Update text chunks using index information
        for text_chunk_id in removed_data["text_chunks"]:
            text_chunk = text_chunks.get(text_chunk_id)
            if text_chunk:
                text_chunk.remove_entity(entity_id)

        # Remove from main storage
        del self.entities[entity_id]

        # Invalidate relevant caches
        self.cache_manager.invalidate_by_tags({"entities", f"entity_type_{entity.entity_type}", f"entity_{entity_id}"})

        self._touch()
        self._record_operation("remove_entity", start_time)
        return True

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID with caching."""
        cache_key = f"entity_{entity_id}"

        # Try cache first
        cached_entity = self.cache_manager.get(cache_key)
        if cached_entity is not None:
            self._metrics["cache_hits"] += 1
            return cached_entity  # type: ignore[no-any-return]

        # Get from main storage
        entity = self.entities.get(entity_id)

        # Cache the result if found
        if entity:
            self.cache_manager.put(
                cache_key,
                entity,
                ttl=300,  # 5 minutes TTL
                tags={f"entity_{entity_id}", "entities"},
            )

        return entity

    def get_entities_by_type(self, entity_type: Union[EntityType, str]) -> List[Entity]:
        """Get all entities of a specific type using index (O(1) lookup)."""
        start_time = time.time()

        # Use cache for expensive type queries
        cache_key = f"entities_by_type_{entity_type}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            self._metrics["cache_hits"] += 1
            self._record_operation("get_entities_by_type_cached", start_time)
            return cached_result  # type: ignore[no-any-return]

        # Use index for fast lookup
        entity_ids = self.index_manager.get_entities_by_type(entity_type)
        self._metrics["index_hits"] += 1

        # Get actual entity objects
        entities = []
        for entity_id in entity_ids:
            entity = self.entities.get(entity_id)
            if entity:
                entities.append(entity)

        # Cache the result
        self.cache_manager.put(
            cache_key,
            entities,
            ttl=180,  # 3 minutes TTL
            tags={f"entity_type_{entity_type}", "entities"},
        )

        self._record_operation("get_entities_by_type", start_time)
        return entities

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by name or description with caching."""
        start_time = time.time()

        # Use cache for search results
        cache_key = f"search_entities_{hash(query)}_{limit}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            self._metrics["cache_hits"] += 1
            self._record_operation("search_entities_cached", start_time)
            return cached_result  # type: ignore[no-any-return]

        query_lower = query.lower()
        matches = []

        # Search through entities (this could be further optimized with a search index)
        for entity in self.entities.values():
            if (
                query_lower in entity.name.lower()
                or query_lower in entity.description.lower()
                or any(query_lower in alias.lower() for alias in entity.aliases)
            ):
                matches.append(entity)

                if len(matches) >= limit:
                    break

        # Cache the search result
        self.cache_manager.put(
            cache_key,
            matches,
            ttl=120,  # 2 minutes TTL for search results
            tags={"search_results", "entities"},
        )

        self._record_operation("search_entities", start_time)
        return matches

    def get_entity_statistics(self) -> Dict[str, int]:
        """Get entity statistics with caching."""
        cache_key = "entity_statistics"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            self._metrics["cache_hits"] += 1
            return cached_result  # type: ignore[no-any-return]

        # Calculate statistics
        type_counts: Dict[str, int] = {}
        for entity in self.entities.values():
            entity_type = str(entity.entity_type)
            type_counts[entity_type] = type_counts.get(entity_type, 0) + 1

        statistics = {
            "total_entities": len(self.entities),
            **type_counts,  # Flatten type_counts into the main dict
        }

        # Cache for 5 minutes
        self.cache_manager.put(cache_key, statistics, ttl=300, tags={"statistics", "entities"})

        return statistics

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        return {
            **self._metrics,
            "average_operation_time": (
                self._metrics["total_time"] / self._metrics["operations_count"]
                if self._metrics["operations_count"] > 0
                else 0
            ),
            "cache_hit_ratio": (
                self._metrics["cache_hits"] / self._metrics["operations_count"]
                if self._metrics["operations_count"] > 0
                else 0
            ),
            "index_statistics": self.index_manager.get_statistics(),
            "cache_statistics": self.cache_manager.get_statistics(),
        }


class OptimizedRelationManager:
    """
    Optimized manager for relation operations using indexing and caching.
    """

    def __init__(
        self,
        relations: Dict[str, Relation],
        touch_callback: Callable[[], None],
        index_manager: Optional[IndexManager] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        """Initialize the OptimizedRelationManager."""
        self.relations = relations
        self._touch = touch_callback
        self.index_manager = index_manager or IndexManager()
        self.cache_manager = cache_manager or CacheManager()

        # Performance metrics
        self._metrics = {"operations_count": 0, "cache_hits": 0, "index_hits": 0, "total_time": 0.0}

    def _record_operation(self, operation_name: str, start_time: float) -> None:
        """Record operation metrics."""
        self._metrics["operations_count"] += 1
        self._metrics["total_time"] += time.time() - start_time

    def add_relation(self, relation: Relation) -> None:
        """Add a relation to the knowledge graph with index updates."""
        start_time = time.time()

        # Add to main storage
        self.relations[relation.id] = relation

        # Update indexes
        if relation.head_entity and relation.tail_entity:
            self.index_manager.add_relation_to_index(relation.id, relation.head_entity.id, relation.tail_entity.id)

        # Invalidate relevant caches
        self.cache_manager.invalidate_by_tags({"relations", f"relation_type_{relation.relation_type}"})

        self._touch()
        self._record_operation("add_relation", start_time)

    def remove_relation(
        self, relation_id: str, clusters: Dict[str, Cluster], text_chunks: Dict[str, TextChunk]
    ) -> bool:
        """Remove a relation using indexes."""
        start_time = time.time()

        if relation_id not in self.relations:
            return False

        relation = self.relations[relation_id]

        # Remove from indexes
        self.index_manager.remove_relation_from_index(relation_id)

        # Remove relation from clusters
        for cluster in clusters.values():
            cluster.remove_relation(relation_id)

        # Remove relation from text chunks
        for text_chunk in text_chunks.values():
            text_chunk.remove_relation(relation_id)

        # Remove from main storage
        del self.relations[relation_id]

        # Invalidate relevant caches
        self.cache_manager.invalidate_by_tags(
            {"relations", f"relation_type_{relation.relation_type}", f"relation_{relation_id}"}
        )

        self._touch()
        self._record_operation("remove_relation", start_time)
        return True

    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get a relation by ID with caching."""
        cache_key = f"relation_{relation_id}"

        # Try cache first
        cached_relation = self.cache_manager.get(cache_key)
        if cached_relation is not None:
            self._metrics["cache_hits"] += 1
            return cached_relation  # type: ignore[no-any-return]

        # Get from main storage
        relation = self.relations.get(relation_id)

        # Cache the result if found
        if relation:
            self.cache_manager.put(
                cache_key,
                relation,
                ttl=300,  # 5 minutes TTL
                tags={f"relation_{relation_id}", "relations"},
            )

        return relation

    def get_relations_by_type(self, relation_type: Union[RelationType, str]) -> List[Relation]:
        """Get all relations of a specific type with caching."""
        cache_key = f"relations_by_type_{relation_type}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            self._metrics["cache_hits"] += 1
            return cached_result  # type: ignore[no-any-return]

        # Filter relations by type
        relations = [relation for relation in self.relations.values() if relation.relation_type == relation_type]

        # Cache the result
        self.cache_manager.put(
            cache_key,
            relations,
            ttl=180,  # 3 minutes TTL
            tags={f"relation_type_{relation_type}", "relations"},
        )

        return relations

    def get_entity_relations(self, entity_id: str, direction: str = "both") -> List[Relation]:
        """Get all relations connected to an entity using index (O(1) lookup)."""
        start_time = time.time()

        # Use cache
        cache_key = f"entity_relations_{entity_id}_{direction}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result is not None:
            self._metrics["cache_hits"] += 1
            self._record_operation("get_entity_relations_cached", start_time)
            return cached_result  # type: ignore[no-any-return]

        # Use index for fast lookup
        relation_ids = self.index_manager.get_entity_relations(entity_id)
        self._metrics["index_hits"] += 1

        relations = []
        for relation_id in relation_ids:
            relation = self.relations.get(relation_id)
            if relation:
                # Apply direction filter
                if direction == "both":
                    relations.append(relation)
                elif direction == "outgoing" and relation.head_entity and relation.head_entity.id == entity_id:
                    relations.append(relation)
                elif direction == "incoming" and relation.tail_entity and relation.tail_entity.id == entity_id:
                    relations.append(relation)

        # Cache the result
        self.cache_manager.put(
            cache_key,
            relations,
            ttl=240,  # 4 minutes TTL
            tags={f"entity_{entity_id}", "relations"},
        )

        self._record_operation("get_entity_relations", start_time)
        return relations

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for monitoring."""
        return {
            **self._metrics,
            "average_operation_time": (
                self._metrics["total_time"] / self._metrics["operations_count"]
                if self._metrics["operations_count"] > 0
                else 0
            ),
            "cache_hit_ratio": (
                self._metrics["cache_hits"] / self._metrics["operations_count"]
                if self._metrics["operations_count"] > 0
                else 0
            ),
            "index_statistics": self.index_manager.get_statistics(),
            "cache_statistics": self.cache_manager.get_statistics(),
        }
