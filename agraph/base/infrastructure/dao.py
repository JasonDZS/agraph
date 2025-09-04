"""
Data Access Layer (DAO) for AGraph.

This module provides an abstraction layer between managers and data storage,
enabling different storage backends (memory, database, file system) to be
used interchangeably without changing business logic.

Enhanced with event system integration for automatic event publishing on data changes.
"""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from threading import RLock

# Forward declarations
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Union

from ..core.result import ErrorCode, Result
from ..core.types import ClusterType, EntityType, RelationType
from ..events.events import (
    EventManager,
    EventType,
    create_entity_event,
    create_relation_event,
    create_system_event,
)
from ..models.clusters import Cluster
from ..models.entities import Entity
from ..models.relations import Relation
from ..models.text import TextChunk


class TransactionContext:
    """Context for managing transactional operations."""

    def __init__(self, dao: "DataAccessLayer"):
        self.dao = dao
        self.operations: List[Callable] = []
        self.rollback_operations: List[Callable] = []
        self.committed = False
        self.rolled_back = False

    def add_operation(self, operation: Callable, rollback_operation: Callable) -> None:
        """Add an operation to the transaction."""
        self.operations.append(operation)
        self.rollback_operations.append(rollback_operation)

    def commit(self) -> Result[bool]:
        """Commit all operations in the transaction."""
        if self.committed or self.rolled_back:
            return Result.fail(
                ErrorCode.INVALID_OPERATION, "Transaction already committed or rolled back"
            )

        try:
            for operation in self.operations:
                operation()
            self.committed = True
            return Result.ok(True)
        except Exception as e:
            self.rollback()
            return Result.internal_error(e)

    def rollback(self) -> Result[bool]:
        """Rollback all operations in the transaction."""
        if self.committed:
            return Result.fail(
                ErrorCode.INVALID_OPERATION, "Cannot rollback already committed transaction"
            )

        try:
            for rollback_op in reversed(self.rollback_operations):
                try:
                    rollback_op()
                except Exception:
                    # Continue rolling back even if individual operations fail
                    pass
            self.rolled_back = True
            return Result.ok(True)
        except Exception as e:
            return Result.internal_error(e)


class DataAccessLayer(ABC):
    """
    Abstract base class for data access layers.

    This interface defines the low-level data operations that all
    storage backends must implement. It provides transaction support
    and ensures data consistency.

    Enhanced with event system integration for automatic event publishing.
    """

    # pylint: disable=too-many-public-methods

    def __init__(self, event_manager: Optional[EventManager] = None) -> None:
        self._lock = RLock()
        self._transaction_context: Optional[TransactionContext] = None
        self._event_manager = event_manager
        self._source_name = self.__class__.__name__

    def set_event_manager(self, event_manager: EventManager) -> None:
        """Set the event manager for this DAO."""
        self._event_manager = event_manager

    def _publish_event(
        self,
        event_type: EventType,
        target_type: str,
        target_id: str,
        data: Optional[Dict[str, Any]] = None,
        transaction_id: Optional[str] = None,
    ) -> None:
        """Publish an event if event manager is available."""
        if self._event_manager:
            if target_type == "entity":
                event = create_entity_event(
                    event_type, target_id, data, self._source_name, transaction_id
                )
            elif target_type == "relation":
                event = create_relation_event(
                    event_type, target_id, data, self._source_name, transaction_id
                )
            else:
                event = create_system_event(event_type, self._source_name, data)

            # Publish asynchronously to avoid blocking DAO operations
            self._event_manager.publish(event, synchronous=False)

    # Transaction Management
    @contextmanager
    def transaction(self) -> Generator[TransactionContext, None, None]:
        """Create a transaction context."""
        if self._transaction_context is not None:
            raise ValueError("Nested transactions are not supported")

        self._transaction_context = TransactionContext(self)
        try:
            yield self._transaction_context
            # Auto-commit if not explicitly handled
            if (
                not self._transaction_context.committed
                and not self._transaction_context.rolled_back
            ):
                result = self._transaction_context.commit()
                if not result.is_ok():
                    raise RuntimeError(f"Transaction commit failed: {result.error_message}")
        except Exception:
            if self._transaction_context and not self._transaction_context.rolled_back:
                self._transaction_context.rollback()
            raise
        finally:
            self._transaction_context = None

    # Entity Operations
    @abstractmethod
    def get_entities(self) -> Dict[str, "Entity"]:
        """Get all entities."""

    @abstractmethod
    def get_entity_by_id(self, entity_id: str) -> Optional["Entity"]:
        """Get an entity by ID."""

    @abstractmethod
    def save_entity(self, entity: "Entity") -> None:
        """Save an entity."""

    @abstractmethod
    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID."""

    @abstractmethod
    def get_entities_by_type(self, entity_type: Union[EntityType, str]) -> List["Entity"]:
        """Get entities by type."""

    @abstractmethod
    def search_entities(self, query: str, limit: int = 10) -> List["Entity"]:
        """Search entities by query."""

    # Relation Operations
    @abstractmethod
    def get_relations(self) -> Dict[str, "Relation"]:
        """Get all relations."""

    @abstractmethod
    def get_relation_by_id(self, relation_id: str) -> Optional["Relation"]:
        """Get a relation by ID."""

    @abstractmethod
    def save_relation(self, relation: "Relation") -> None:
        """Save a relation."""

    @abstractmethod
    def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation by ID."""

    @abstractmethod
    def get_relations_by_type(self, relation_type: Union[RelationType, str]) -> List["Relation"]:
        """Get relations by type."""

    @abstractmethod
    def get_entity_relations(self, entity_id: str) -> List["Relation"]:
        """Get all relations connected to an entity."""

    # Cluster Operations
    @abstractmethod
    def get_clusters(self) -> Dict[str, "Cluster"]:
        """Get all clusters."""

    @abstractmethod
    def get_cluster_by_id(self, cluster_id: str) -> Optional["Cluster"]:
        """Get a cluster by ID."""

    @abstractmethod
    def save_cluster(self, cluster: "Cluster") -> None:
        """Save a cluster."""

    @abstractmethod
    def delete_cluster(self, cluster_id: str) -> bool:
        """Delete a cluster by ID."""

    @abstractmethod
    def get_clusters_by_type(self, cluster_type: Union[ClusterType, str]) -> List["Cluster"]:
        """Get clusters by type."""

    @abstractmethod
    def get_cluster_entities(self, cluster_id: str) -> List["Entity"]:
        """Get all entities in a cluster."""

    @abstractmethod
    def add_entity_to_cluster(self, cluster_id: str, entity_id: str) -> bool:
        """Add an entity to a cluster."""

    @abstractmethod
    def remove_entity_from_cluster(self, cluster_id: str, entity_id: str) -> bool:
        """Remove an entity from a cluster."""

    # Text Chunk Operations
    @abstractmethod
    def get_text_chunks(self) -> Dict[str, "TextChunk"]:
        """Get all text chunks."""

    @abstractmethod
    def get_text_chunk_by_id(self, chunk_id: str) -> Optional["TextChunk"]:
        """Get a text chunk by ID."""

    @abstractmethod
    def save_text_chunk(self, chunk: "TextChunk") -> None:
        """Save a text chunk."""

    @abstractmethod
    def delete_text_chunk(self, chunk_id: str) -> bool:
        """Delete a text chunk by ID."""

    @abstractmethod
    def get_text_chunks_by_source(self, source: str) -> List["TextChunk"]:
        """Get text chunks by source."""

    @abstractmethod
    def get_chunk_entities(self, chunk_id: str) -> List["Entity"]:
        """Get all entities referenced in a text chunk."""

    @abstractmethod
    def get_chunk_relations(self, chunk_id: str) -> List["Relation"]:
        """Get all relations referenced in a text chunk."""

    # Utility Operations
    @abstractmethod
    def clear_all(self) -> None:
        """Clear all data."""

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""

    @abstractmethod
    def backup_data(self) -> Dict[str, Any]:
        """Create a backup of all data."""

    @abstractmethod
    def restore_data(self, backup: Dict[str, Any]) -> None:
        """Restore data from backup."""


class MemoryDataAccessLayer(DataAccessLayer):
    """
    In-memory implementation of DataAccessLayer.

    This is the default implementation that stores all data in memory.
    It provides fast access but data is not persisted between sessions.

    Enhanced with event system integration.
    """

    # pylint: disable=too-many-public-methods

    def __init__(self, event_manager: Optional[EventManager] = None) -> None:
        super().__init__(event_manager)
        self._entities: Dict[str, "Entity"] = {}
        self._relations: Dict[str, "Relation"] = {}
        self._clusters: Dict[str, "Cluster"] = {}
        self._text_chunks: Dict[str, "TextChunk"] = {}
        # Cluster-entity relationships: cluster_id -> set of entity_ids
        self._cluster_entities: Dict[str, Set[str]] = {}

    # Entity Operations
    def get_entities(self) -> Dict[str, "Entity"]:
        """Get all entities."""
        with self._lock:
            return self._entities.copy()

    def get_entity_by_id(self, entity_id: str) -> Optional["Entity"]:
        """Get an entity by ID."""
        with self._lock:
            return self._entities.get(entity_id)

    def save_entity(self, entity: "Entity") -> None:
        """Save an entity."""
        with self._lock:
            is_new_entity = entity.id not in self._entities

            if self._transaction_context:
                old_entity = self._entities.get(entity.id)

                def _save_operation():
                    self._entities[entity.id] = entity
                    # Publish event after successful save
                    event_type = (
                        EventType.ENTITY_ADDED if is_new_entity else EventType.ENTITY_UPDATED
                    )
                    entity_data = entity.to_dict() if hasattr(entity, "to_dict") else vars(entity)
                    self._publish_event(
                        event_type,
                        "entity",
                        entity.id,
                        entity_data,
                        getattr(self._transaction_context, "transaction_id", None),
                    )

                def _rollback_operation():
                    if old_entity is None:
                        self._entities.pop(entity.id, None)
                    else:
                        self._entities[entity.id] = old_entity

                self._transaction_context.add_operation(_save_operation, _rollback_operation)
            else:
                self._entities[entity.id] = entity
                # Publish event immediately for non-transactional operations
                event_type = EventType.ENTITY_ADDED if is_new_entity else EventType.ENTITY_UPDATED
                entity_data = entity.to_dict() if hasattr(entity, "to_dict") else vars(entity)
                self._publish_event(event_type, "entity", entity.id, entity_data)

    def add_entity(self, entity: "Entity") -> None:
        """Add an entity (alias for save_entity)."""
        self.save_entity(entity)

    def remove_entity(self, entity_id: str) -> bool:
        """Remove an entity (alias for delete_entity)."""
        return self.delete_entity(entity_id)

    def delete_entity(self, entity_id: str) -> bool:
        """Delete an entity by ID."""
        with self._lock:
            if entity_id not in self._entities:
                return False

            old_entity = self._entities[entity_id]
            entity_data = (
                old_entity.to_dict() if hasattr(old_entity, "to_dict") else vars(old_entity)
            )

            if self._transaction_context:

                def _delete_operation():
                    self._entities.pop(entity_id, None)
                    # Publish event after successful delete
                    self._publish_event(
                        EventType.ENTITY_REMOVED,
                        "entity",
                        entity_id,
                        entity_data,
                        getattr(self._transaction_context, "transaction_id", None),
                    )

                def _rollback_operation():
                    self._entities[entity_id] = old_entity

                self._transaction_context.add_operation(_delete_operation, _rollback_operation)
            else:
                del self._entities[entity_id]
                # Publish event immediately for non-transactional operations
                self._publish_event(EventType.ENTITY_REMOVED, "entity", entity_id, entity_data)
            return True

    def get_entities_by_type(self, entity_type: Union[EntityType, str]) -> List["Entity"]:
        """Get entities by type."""
        with self._lock:
            # Convert entity_type to string value for comparison
            if hasattr(entity_type, "value"):
                target_type = entity_type.value
            else:
                target_type = str(entity_type)

            return [
                entity for entity in self._entities.values() if entity.entity_type == target_type
            ]

    def search_entities(self, query: str, limit: int = 10) -> List["Entity"]:
        """Search entities by query."""
        with self._lock:
            query_lower = query.lower()
            matches = []

            for entity in self._entities.values():
                if (
                    query_lower in entity.name.lower()
                    or query_lower in entity.description.lower()
                    or any(query_lower in alias.lower() for alias in entity.aliases)
                ):
                    matches.append(entity)
                    if len(matches) >= limit:
                        break

            return matches

    # Relation Operations
    def get_relations(self) -> Dict[str, "Relation"]:
        """Get all relations."""
        with self._lock:
            return self._relations.copy()

    def get_relation_by_id(self, relation_id: str) -> Optional["Relation"]:
        """Get a relation by ID."""
        with self._lock:
            return self._relations.get(relation_id)

    def save_relation(self, relation: "Relation") -> None:
        """Save a relation."""
        with self._lock:
            is_new_relation = relation.id not in self._relations

            if self._transaction_context:
                old_relation = self._relations.get(relation.id)

                def _save_operation():
                    self._relations[relation.id] = relation
                    # Publish event after successful save
                    event_type = (
                        EventType.RELATION_ADDED if is_new_relation else EventType.RELATION_UPDATED
                    )
                    relation_data = (
                        relation.to_dict() if hasattr(relation, "to_dict") else vars(relation)
                    )
                    self._publish_event(
                        event_type,
                        "relation",
                        relation.id,
                        relation_data,
                        getattr(self._transaction_context, "transaction_id", None),
                    )

                def _rollback_operation():
                    if old_relation is None:
                        self._relations.pop(relation.id, None)
                    else:
                        self._relations[relation.id] = old_relation

                self._transaction_context.add_operation(_save_operation, _rollback_operation)
            else:
                self._relations[relation.id] = relation
                # Publish event immediately for non-transactional operations
                event_type = (
                    EventType.RELATION_ADDED if is_new_relation else EventType.RELATION_UPDATED
                )
                relation_data = (
                    relation.to_dict() if hasattr(relation, "to_dict") else vars(relation)
                )
                self._publish_event(event_type, "relation", relation.id, relation_data)

    def add_relation(self, relation: "Relation") -> None:
        """Add a relation (alias for save_relation)."""
        self.save_relation(relation)

    def remove_relation(self, relation_id: str) -> bool:
        """Remove a relation (alias for delete_relation)."""
        return self.delete_relation(relation_id)

    def delete_relation(self, relation_id: str) -> bool:
        """Delete a relation by ID."""
        with self._lock:
            if relation_id not in self._relations:
                return False

            old_relation = self._relations[relation_id]
            relation_data = (
                old_relation.to_dict() if hasattr(old_relation, "to_dict") else vars(old_relation)
            )

            if self._transaction_context:

                def _delete_operation():
                    self._relations.pop(relation_id, None)
                    # Publish event after successful delete
                    self._publish_event(
                        EventType.RELATION_REMOVED,
                        "relation",
                        relation_id,
                        relation_data,
                        getattr(self._transaction_context, "transaction_id", None),
                    )

                def _rollback_operation():
                    self._relations[relation_id] = old_relation

                self._transaction_context.add_operation(_delete_operation, _rollback_operation)
            else:
                del self._relations[relation_id]
                # Publish event immediately for non-transactional operations
                self._publish_event(
                    EventType.RELATION_REMOVED, "relation", relation_id, relation_data
                )
            return True

    def get_relations_by_type(self, relation_type: Union[RelationType, str]) -> List["Relation"]:
        """Get relations by type."""
        with self._lock:
            # Convert relation_type to string value for comparison
            if hasattr(relation_type, "value"):
                target_type = relation_type.value
            else:
                target_type = str(relation_type)

            return [
                relation
                for relation in self._relations.values()
                if relation.relation_type == target_type
            ]

    def get_entity_relations(self, entity_id: str) -> List["Relation"]:
        """Get all relations connected to an entity."""
        with self._lock:
            relations = []
            for relation in self._relations.values():
                if (relation.head_entity and relation.head_entity.id == entity_id) or (
                    relation.tail_entity and relation.tail_entity.id == entity_id
                ):
                    relations.append(relation)
            return relations

    # Cluster Operations
    def get_clusters(self) -> Dict[str, "Cluster"]:
        """Get all clusters."""
        with self._lock:
            return self._clusters.copy()

    def get_cluster_by_id(self, cluster_id: str) -> Optional["Cluster"]:
        """Get a cluster by ID."""
        with self._lock:
            return self._clusters.get(cluster_id)

    def save_cluster(self, cluster: "Cluster") -> None:
        """Save a cluster."""
        with self._lock:
            if self._transaction_context:
                old_cluster = self._clusters.get(cluster.id)
                self._transaction_context.add_operation(
                    lambda: setattr(self, "_clusters", {**self._clusters, cluster.id: cluster}),
                    lambda: (
                        self._clusters.pop(cluster.id, None)
                        if old_cluster is None
                        else setattr(self, "_clusters", {**self._clusters, cluster.id: old_cluster})
                    ),
                )
            else:
                self._clusters[cluster.id] = cluster

    def add_cluster(self, cluster: "Cluster") -> None:
        """Add a cluster (alias for save_cluster)."""
        self.save_cluster(cluster)

    def remove_cluster(self, cluster_id: str) -> bool:
        """Remove a cluster (alias for delete_cluster)."""
        return self.delete_cluster(cluster_id)

    def delete_cluster(self, cluster_id: str) -> bool:
        """Delete a cluster by ID."""
        with self._lock:
            if cluster_id not in self._clusters:
                return False

            if self._transaction_context:
                old_cluster = self._clusters[cluster_id]
                self._transaction_context.add_operation(
                    lambda: self._clusters.pop(cluster_id, None),
                    lambda: setattr(self, "_clusters", {**self._clusters, cluster_id: old_cluster}),
                )
            else:
                del self._clusters[cluster_id]
            return True

    def get_clusters_by_type(self, cluster_type: Union[ClusterType, str]) -> List["Cluster"]:
        """Get clusters by type."""
        with self._lock:
            # Convert cluster_type to string value for comparison
            if hasattr(cluster_type, "value"):
                target_type = cluster_type.value
            else:
                target_type = str(cluster_type)

            return [
                cluster
                for cluster in self._clusters.values()
                if cluster.cluster_type == target_type
            ]

    def get_cluster_entities(self, cluster_id: str) -> List["Entity"]:
        """Get all entities in a cluster."""
        with self._lock:
            entity_ids = self._cluster_entities.get(cluster_id, set())
            return [
                self._entities[entity_id] for entity_id in entity_ids if entity_id in self._entities
            ]

    def add_entity_to_cluster(self, cluster_id: str, entity_id: str) -> bool:
        """Add an entity to a cluster."""
        with self._lock:
            # Verify cluster exists
            if cluster_id not in self._clusters:
                return False

            # Verify entity exists
            if entity_id not in self._entities:
                return False

            # Add entity to cluster
            if cluster_id not in self._cluster_entities:
                self._cluster_entities[cluster_id] = set()
            self._cluster_entities[cluster_id].add(entity_id)
            return True

    def remove_entity_from_cluster(self, cluster_id: str, entity_id: str) -> bool:
        """Remove an entity from a cluster."""
        with self._lock:
            if cluster_id in self._cluster_entities:
                self._cluster_entities[cluster_id].discard(entity_id)
                return True
            return False

    # Text Chunk Operations
    def get_text_chunks(self) -> Dict[str, "TextChunk"]:
        """Get all text chunks."""
        with self._lock:
            return self._text_chunks.copy()

    def get_text_chunk_by_id(self, chunk_id: str) -> Optional["TextChunk"]:
        """Get a text chunk by ID."""
        with self._lock:
            return self._text_chunks.get(chunk_id)

    def save_text_chunk(self, chunk: "TextChunk") -> None:
        """Save a text chunk."""
        with self._lock:
            if self._transaction_context:
                old_chunk = self._text_chunks.get(chunk.id)
                self._transaction_context.add_operation(
                    lambda: setattr(self, "_text_chunks", {**self._text_chunks, chunk.id: chunk}),
                    lambda: (
                        self._text_chunks.pop(chunk.id, None)
                        if old_chunk is None
                        else setattr(
                            self, "_text_chunks", {**self._text_chunks, chunk.id: old_chunk}
                        )
                    ),
                )
            else:
                self._text_chunks[chunk.id] = chunk

    def add_text_chunk(self, chunk: "TextChunk") -> None:
        """Add a text chunk (alias for save_text_chunk)."""
        self.save_text_chunk(chunk)

    def remove_text_chunk(self, chunk_id: str) -> bool:
        """Remove a text chunk (alias for delete_text_chunk)."""
        return self.delete_text_chunk(chunk_id)

    def delete_text_chunk(self, chunk_id: str) -> bool:
        """Delete a text chunk by ID."""
        with self._lock:
            if chunk_id not in self._text_chunks:
                return False

            if self._transaction_context:
                old_chunk = self._text_chunks[chunk_id]
                self._transaction_context.add_operation(
                    lambda: self._text_chunks.pop(chunk_id, None),
                    lambda: setattr(
                        self, "_text_chunks", {**self._text_chunks, chunk_id: old_chunk}
                    ),
                )
            else:
                del self._text_chunks[chunk_id]
            return True

    def get_text_chunks_by_source(self, source: str) -> List["TextChunk"]:
        """Get text chunks by source."""
        with self._lock:
            return [chunk for chunk in self._text_chunks.values() if chunk.source == source]

    def get_chunk_entities(self, chunk_id: str) -> List["Entity"]:
        """Get all entities referenced in a text chunk."""
        with self._lock:
            # For now, return empty list as we don't track chunk-entity relationships
            # This would need to be implemented based on how chunk-entity relationships are tracked
            return []

    def get_chunk_relations(self, chunk_id: str) -> List["Relation"]:
        """Get all relations referenced in a text chunk."""
        with self._lock:
            # For now, return empty list as we don't track chunk-relation relationships
            # This would need to be implemented based on how chunk-relation relationships are tracked
            return []

    # Utility Operations
    def clear_all(self) -> None:
        """Clear all data."""
        with self._lock:
            if self._transaction_context:
                old_data = {
                    "entities": self._entities.copy(),
                    "relations": self._relations.copy(),
                    "clusters": self._clusters.copy(),
                    "text_chunks": self._text_chunks.copy(),
                }

                def _clear_all() -> None:
                    self._entities.clear()
                    self._relations.clear()
                    self._clusters.clear()
                    self._text_chunks.clear()

                def _restore_all() -> None:
                    setattr(self, "_entities", old_data["entities"])
                    setattr(self, "_relations", old_data["relations"])
                    setattr(self, "_clusters", old_data["clusters"])
                    setattr(self, "_text_chunks", old_data["text_chunks"])

                self._transaction_context.add_operation(_clear_all, _restore_all)
            else:
                self._entities.clear()
                self._relations.clear()
                self._clusters.clear()
                self._text_chunks.clear()

    def get_statistics(self) -> Dict[str, Any]:
        """Get storage statistics."""
        with self._lock:
            return {
                "entities_count": len(self._entities),
                "relations_count": len(self._relations),
                "clusters_count": len(self._clusters),
                "text_chunks_count": len(self._text_chunks),
                "storage_type": "memory",
            }

    def backup_data(self) -> Dict[str, Any]:
        """Create a backup of all data."""
        with self._lock:
            return {
                "entities": {eid: entity.to_dict() for eid, entity in self._entities.items()},
                "relations": {rid: relation.to_dict() for rid, relation in self._relations.items()},
                "clusters": {cid: cluster.to_dict() for cid, cluster in self._clusters.items()},
                "text_chunks": {tid: chunk.to_dict() for tid, chunk in self._text_chunks.items()},
            }

    def restore_data(self, backup: Dict[str, Any]) -> None:
        """Restore data from backup."""
        with self._lock:
            self.clear_all()

            # Restore entities first
            for eid, entity_data in backup.get("entities", {}).items():
                entity = Entity.from_dict(entity_data)
                self._entities[eid] = entity

            # Restore relations with entity references
            for rid, relation_data in backup.get("relations", {}).items():
                relation = Relation.from_dict(relation_data, entities_map=self._entities)
                self._relations[rid] = relation

            # Restore clusters
            for cid, cluster_data in backup.get("clusters", {}).items():
                cluster = Cluster.from_dict(cluster_data)
                self._clusters[cid] = cluster

            # Restore text chunks
            for tid, chunk_data in backup.get("text_chunks", {}).items():
                chunk = TextChunk.from_dict(chunk_data)
                self._text_chunks[tid] = chunk
