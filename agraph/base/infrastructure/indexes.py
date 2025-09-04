"""
Index management system for AGraph knowledge graph.

This module provides indexing capabilities to optimize query performance
by maintaining efficient lookup structures for entities, relations, and their relationships.
"""

import threading
from collections import defaultdict
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Set, TypeVar, Union, cast

from ..core.types import EntityType

if TYPE_CHECKING:
    from ..graphs.legacy import KnowledgeGraph
    from ..graphs.optimized import OptimizedKnowledgeGraph


# Type variable for generic function types
F = TypeVar("F", bound=Callable[..., Any])


class IndexType(Enum):
    """Types of indexes maintained by the IndexManager."""

    ENTITY_TYPE = "entity_type"
    RELATION_ENTITY = "relation_entity"
    ENTITY_RELATIONS = "entity_relations"
    ENTITY_CLUSTERS = "entity_clusters"
    ENTITY_TEXT_CHUNKS = "entity_text_chunks"
    CLUSTER_ENTITIES = "cluster_entities"


class IndexManager:
    """
    Manages all indexes for efficient knowledge graph operations.

    This class maintains various indexes to optimize common query patterns:
    - Entity type lookup
    - Entity-relation relationships
    - Entity-cluster relationships
    - Entity-text chunk relationships
    """

    def __init__(self) -> None:
        """Initialize the IndexManager with empty indexes."""
        # Entity type index: EntityType -> Set[entity_id]
        self._entity_type_index: Dict[Union[EntityType, str], Set[str]] = defaultdict(set)

        # Relation-entity index: relation_id -> (head_entity_id, tail_entity_id)
        self._relation_entity_index: Dict[str, tuple[str, str]] = {}

        # Entity-relations index: entity_id -> Set[relation_id]
        self._entity_relations_index: Dict[str, Set[str]] = defaultdict(set)

        # Entity-clusters index: entity_id -> Set[cluster_id]
        self._entity_clusters_index: Dict[str, Set[str]] = defaultdict(set)

        # Entity-text chunks index: entity_id -> Set[text_chunk_id]
        self._entity_text_chunks_index: Dict[str, Set[str]] = defaultdict(set)

        # Cluster-entities index: cluster_id -> Set[entity_id]
        self._cluster_entities_index: Dict[str, Set[str]] = defaultdict(set)

        # Text chunk-entities index: text_chunk_id -> Set[entity_id]
        self._text_chunk_entities_index: Dict[str, Set[str]] = defaultdict(set)

        # Thread safety lock
        self._lock = threading.RWLock() if hasattr(threading, "RWLock") else threading.Lock()

        # Index statistics
        self._stats = {"total_indexes": 0, "index_hits": 0, "index_misses": 0}

    def _with_write_lock(self, func: F) -> F:
        """Decorator to ensure write operations are thread-safe."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if hasattr(self._lock, "writer"):
                with self._lock.writer():
                    return func(*args, **kwargs)
            else:
                with self._lock:
                    return func(*args, **kwargs)

        return cast(F, wrapper)

    def _with_read_lock(self, func: F) -> F:
        """Decorator to ensure read operations are thread-safe."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if hasattr(self._lock, "reader"):
                with self._lock.reader():
                    return func(*args, **kwargs)
            else:
                with self._lock:
                    return func(*args, **kwargs)

        return cast(F, wrapper)

    # Entity Type Index Operations
    def add_entity_to_type_index(self, entity_id: str, entity_type: Union[EntityType, str]) -> None:
        """Add an entity to the type index."""

        @self._with_write_lock
        def _add() -> None:
            self._entity_type_index[entity_type].add(entity_id)
            self._stats["total_indexes"] += 1

        _add()

    def remove_entity_from_type_index(
        self, entity_id: str, entity_type: Union[EntityType, str]
    ) -> None:
        """Remove an entity from the type index."""

        @self._with_write_lock
        def _remove() -> None:
            self._entity_type_index[entity_type].discard(entity_id)
            if not self._entity_type_index[entity_type]:
                del self._entity_type_index[entity_type]
            self._stats["total_indexes"] = max(0, self._stats["total_indexes"] - 1)

        _remove()

    def get_entities_by_type(self, entity_type: Union[EntityType, str]) -> Set[str]:
        """Get all entity IDs of a specific type."""

        @self._with_read_lock
        def _get() -> set[str]:
            result = self._entity_type_index.get(entity_type, set()).copy()
            if result:
                self._stats["index_hits"] += 1
            else:
                self._stats["index_misses"] += 1
            return result

        return _get()

    # Entity-Relation Index Operations
    def add_relation_to_index(
        self, relation_id: str, head_entity_id: str, tail_entity_id: str
    ) -> None:
        """Add a relation to the entity-relation indexes."""

        @self._with_write_lock
        def _add() -> None:
            # Store relation -> (head, tail) mapping
            self._relation_entity_index[relation_id] = (head_entity_id, tail_entity_id)

            # Store entity -> relations mappings
            self._entity_relations_index[head_entity_id].add(relation_id)
            self._entity_relations_index[tail_entity_id].add(relation_id)

            self._stats["total_indexes"] += 2

        _add()

    def remove_relation_from_index(self, relation_id: str) -> Optional[tuple[str, str]]:
        """Remove a relation from the entity-relation indexes."""

        @self._with_write_lock
        def _remove() -> Optional[tuple[str, str]]:
            return self._remove_relation_from_index_internal(relation_id)

        return _remove()

    def _remove_relation_from_index_internal(self, relation_id: str) -> Optional[tuple[str, str]]:
        """Internal method to remove relation from index without acquiring locks."""
        # Get the entity IDs first
        entity_ids = self._relation_entity_index.get(relation_id)
        if entity_ids:
            head_entity_id, tail_entity_id = entity_ids

            # Remove from relation-entity index
            del self._relation_entity_index[relation_id]

            # Remove from entity-relations indexes
            self._entity_relations_index[head_entity_id].discard(relation_id)
            self._entity_relations_index[tail_entity_id].discard(relation_id)

            # Clean up empty sets
            if not self._entity_relations_index[head_entity_id]:
                del self._entity_relations_index[head_entity_id]
            if not self._entity_relations_index[tail_entity_id]:
                del self._entity_relations_index[tail_entity_id]

            self._stats["total_indexes"] = max(0, self._stats["total_indexes"] - 2)
            return entity_ids
        return None

    def get_entity_relations(self, entity_id: str) -> Set[str]:
        """Get all relation IDs connected to an entity."""

        @self._with_read_lock
        def _get() -> set[str]:
            result = self._entity_relations_index.get(entity_id, set()).copy()
            if result:
                self._stats["index_hits"] += 1
            else:
                self._stats["index_misses"] += 1
            return result

        return _get()

    def get_relation_entities(self, relation_id: str) -> Optional[tuple[str, str]]:
        """Get the head and tail entity IDs for a relation."""

        @self._with_read_lock
        def _get() -> Optional[tuple[str, str]]:
            result = self._relation_entity_index.get(relation_id)
            if result:
                self._stats["index_hits"] += 1
            else:
                self._stats["index_misses"] += 1
            return result

        return _get()

    # Entity-Cluster Index Operations
    def add_entity_to_cluster_index(self, entity_id: str, cluster_id: str) -> None:
        """Add an entity-cluster relationship to the index."""

        @self._with_write_lock
        def _add() -> None:
            self._entity_clusters_index[entity_id].add(cluster_id)
            self._cluster_entities_index[cluster_id].add(entity_id)
            self._stats["total_indexes"] += 2

        _add()

    def remove_entity_from_cluster_index(self, entity_id: str, cluster_id: str) -> None:
        """Remove an entity-cluster relationship from the index."""

        @self._with_write_lock
        def _remove() -> None:
            self._remove_entity_from_cluster_index_internal(entity_id, cluster_id)

        _remove()

    def _remove_entity_from_cluster_index_internal(self, entity_id: str, cluster_id: str) -> None:
        """Internal method to remove entity-cluster relationship without acquiring locks."""
        self._entity_clusters_index[entity_id].discard(cluster_id)
        self._cluster_entities_index[cluster_id].discard(entity_id)

        # Clean up empty sets
        if not self._entity_clusters_index[entity_id]:
            del self._entity_clusters_index[entity_id]
        if not self._cluster_entities_index[cluster_id]:
            del self._cluster_entities_index[cluster_id]

        self._stats["total_indexes"] = max(0, self._stats["total_indexes"] - 2)

    def get_entity_clusters(self, entity_id: str) -> Set[str]:
        """Get all cluster IDs containing an entity."""

        @self._with_read_lock
        def _get() -> set[str]:
            result = self._entity_clusters_index.get(entity_id, set()).copy()
            if result:
                self._stats["index_hits"] += 1
            else:
                self._stats["index_misses"] += 1
            return result

        return _get()

    def get_cluster_entities(self, cluster_id: str) -> Set[str]:
        """Get all entity IDs in a cluster."""

        @self._with_read_lock
        def _get() -> set[str]:
            result = self._cluster_entities_index.get(cluster_id, set()).copy()
            if result:
                self._stats["index_hits"] += 1
            else:
                self._stats["index_misses"] += 1
            return result

        return _get()

    # Entity-TextChunk Index Operations
    def add_entity_to_text_chunk_index(self, entity_id: str, text_chunk_id: str) -> None:
        """Add an entity-text chunk relationship to the index."""

        @self._with_write_lock
        def _add() -> None:
            self._entity_text_chunks_index[entity_id].add(text_chunk_id)
            self._text_chunk_entities_index[text_chunk_id].add(entity_id)
            self._stats["total_indexes"] += 2

        _add()

    def remove_entity_from_text_chunk_index(self, entity_id: str, text_chunk_id: str) -> None:
        """Remove an entity-text chunk relationship from the index."""

        @self._with_write_lock
        def _remove() -> None:
            self._remove_entity_from_text_chunk_index_internal(entity_id, text_chunk_id)

        _remove()

    def _remove_entity_from_text_chunk_index_internal(
        self, entity_id: str, text_chunk_id: str
    ) -> None:
        """Internal method to remove entity-text chunk relationship without acquiring locks."""
        self._entity_text_chunks_index[entity_id].discard(text_chunk_id)
        self._text_chunk_entities_index[text_chunk_id].discard(entity_id)

        # Clean up empty sets
        if not self._entity_text_chunks_index[entity_id]:
            del self._entity_text_chunks_index[entity_id]
        if not self._text_chunk_entities_index[text_chunk_id]:
            del self._text_chunk_entities_index[text_chunk_id]

        self._stats["total_indexes"] = max(0, self._stats["total_indexes"] - 2)

    def get_entity_text_chunks(self, entity_id: str) -> Set[str]:
        """Get all text chunk IDs connected to an entity."""

        @self._with_read_lock
        def _get() -> set[str]:
            result = self._entity_text_chunks_index.get(entity_id, set()).copy()
            if result:
                self._stats["index_hits"] += 1
            else:
                self._stats["index_misses"] += 1
            return result

        return _get()

    def get_text_chunk_entities(self, text_chunk_id: str) -> Set[str]:
        """Get all entity IDs in a text chunk."""

        @self._with_read_lock
        def _get() -> set[str]:
            result = self._text_chunk_entities_index.get(text_chunk_id, set()).copy()
            if result:
                self._stats["index_hits"] += 1
            else:
                self._stats["index_misses"] += 1
            return result

        return _get()

    # Bulk Operations
    def remove_entity_from_all_indexes(
        self, entity_id: str, entity_type: Union[EntityType, str] = None
    ) -> Dict[str, Set[str]]:
        """Remove an entity from all indexes and return what was removed."""

        @self._with_write_lock
        def _remove_all() -> dict[str, set[str]]:
            removed_data: dict[str, set[str]] = {
                "relations": set(),
                "clusters": set(),
                "text_chunks": set(),
            }

            # Remove from entity type index if type is provided
            if entity_type is not None:
                self._entity_type_index[entity_type].discard(entity_id)
                if not self._entity_type_index[entity_type]:
                    del self._entity_type_index[entity_type]

            # Remove from relation indexes
            relation_ids = self._entity_relations_index.get(entity_id, set()).copy()
            for relation_id in relation_ids:
                entity_pair = self._relation_entity_index.get(relation_id)
                if entity_pair:
                    head_id, tail_id = entity_pair
                    # Only remove the relation if this entity is involved
                    if entity_id in (head_id, tail_id):
                        self._remove_relation_from_index_internal(relation_id)
                        removed_data["relations"].add(relation_id)

            # Remove from cluster indexes
            cluster_ids = self._entity_clusters_index.get(entity_id, set()).copy()
            for cluster_id in cluster_ids:
                self._remove_entity_from_cluster_index_internal(entity_id, cluster_id)
                removed_data["clusters"].add(cluster_id)

            # Remove from text chunk indexes
            text_chunk_ids = self._entity_text_chunks_index.get(entity_id, set()).copy()
            for text_chunk_id in text_chunk_ids:
                self._remove_entity_from_text_chunk_index_internal(entity_id, text_chunk_id)
                removed_data["text_chunks"].add(text_chunk_id)

            return removed_data

        return _remove_all()

    # Statistics and Maintenance
    def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics for monitoring and debugging."""

        @self._with_read_lock
        def _get_stats() -> dict[str, Any]:
            return {
                "total_indexes": self._stats["total_indexes"],
                "index_hits": self._stats["index_hits"],
                "index_misses": self._stats["index_misses"],
                "hit_ratio": (
                    self._stats["index_hits"]
                    / (self._stats["index_hits"] + self._stats["index_misses"])
                    if (self._stats["index_hits"] + self._stats["index_misses"]) > 0
                    else 0.0
                ),
                "entity_types_count": len(self._entity_type_index),
                "relations_count": len(self._relation_entity_index),
                "entity_relations_count": len(self._entity_relations_index),
                "entity_clusters_count": len(self._entity_clusters_index),
                "cluster_entities_count": len(self._cluster_entities_index),
                "entity_text_chunks_count": len(self._entity_text_chunks_index),
                "text_chunk_entities_count": len(self._text_chunk_entities_index),
            }

        return _get_stats()

    def clear_all_indexes(self) -> None:
        """Clear all indexes. Use with caution!"""

        @self._with_write_lock
        def _clear() -> None:
            self._clear_all_indexes_internal()

        _clear()

    def rebuild_indexes(
        self, knowledge_graph: Union["KnowledgeGraph", "OptimizedKnowledgeGraph"]
    ) -> None:
        """Rebuild all indexes from the knowledge graph data."""

        @self._with_write_lock
        def _rebuild() -> None:
            # Clear existing indexes (this method is already lock-free internally)
            self._clear_all_indexes_internal()

            # Rebuild entity type index (lock-free internal operation)
            for entity_id, entity in knowledge_graph.entities.items():
                self._entity_type_index[entity.entity_type].add(entity_id)

            # Rebuild relation indexes (lock-free internal operation)
            for relation_id, relation in knowledge_graph.relations.items():
                if relation.head_entity and relation.tail_entity:
                    head_id, tail_id = relation.head_entity.id, relation.tail_entity.id
                    # Store relation -> (head, tail) mapping
                    self._relation_entity_index[relation_id] = (head_id, tail_id)
                    # Store entity -> relations mappings
                    self._entity_relations_index[head_id].add(relation_id)
                    self._entity_relations_index[tail_id].add(relation_id)

            # Rebuild cluster indexes (lock-free internal operation)
            for cluster_id, cluster in knowledge_graph.clusters.items():
                for entity_id in cluster.entities:
                    self._entity_clusters_index[entity_id].add(cluster_id)
                    self._cluster_entities_index[cluster_id].add(entity_id)

            # Rebuild text chunk indexes (lock-free internal operation)
            for text_chunk_id, text_chunk in knowledge_graph.text_chunks.items():
                for entity_id in text_chunk.entities:
                    self._entity_text_chunks_index[entity_id].add(text_chunk_id)
                    self._text_chunk_entities_index[text_chunk_id].add(entity_id)

            # Update statistics
            total_indexes = (
                sum(len(entities) for entities in self._entity_type_index.values())
                + len(self._relation_entity_index) * 2
                + sum(len(entities) for entities in self._entity_clusters_index.values()) * 2
                + sum(len(entities) for entities in self._entity_text_chunks_index.values()) * 2
            )
            self._stats["total_indexes"] = total_indexes

        _rebuild()

    def _clear_all_indexes_internal(self) -> None:
        """Internal method to clear all indexes without acquiring locks."""
        self._entity_type_index.clear()
        self._relation_entity_index.clear()
        self._entity_relations_index.clear()
        self._entity_clusters_index.clear()
        self._cluster_entities_index.clear()
        self._entity_text_chunks_index.clear()
        self._text_chunk_entities_index.clear()

        # Reset statistics
        self._stats = {"total_indexes": 0, "index_hits": 0, "index_misses": 0}


# Global index manager instance (optional singleton pattern)
_global_index_manager: Optional[IndexManager] = None


def get_global_index_manager() -> IndexManager:
    """Get the global IndexManager instance (creates one if it doesn't exist)."""
    # pylint: disable=global-statement
    global _global_index_manager
    if _global_index_manager is None:
        _global_index_manager = IndexManager()
    return _global_index_manager


def set_global_index_manager(index_manager: IndexManager) -> None:
    """Set the global IndexManager instance."""
    # pylint: disable=global-statement
    global _global_index_manager
    _global_index_manager = index_manager
