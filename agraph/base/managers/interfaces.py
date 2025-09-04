"""
Abstract interfaces for AGraph manager classes.

This module defines the unified interfaces that all managers should implement,
promoting consistency and enabling interchangeable implementations.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Generic, List, Optional, TypeVar, Union

from ..core.result import Result
from ..core.types import ClusterType, EntityType, RelationType

T = TypeVar("T")

if TYPE_CHECKING:
    from ..models.clusters import Cluster
    from ..models.entities import Entity
    from ..models.relations import Relation
    from ..models.text import TextChunk
    from ..transactions.batch import BatchContext


class Manager(Generic[T], ABC):
    """
    Abstract base class for all managers in the AGraph system.

    This interface defines the standard operations that all managers
    should support, ensuring consistent behavior across different
    types of graph components.
    """

    @abstractmethod
    def add(self, item: T) -> Result[T]:
        """
        Add an item to the managed collection.

        Args:
            item: The item to add

        Returns:
            Result containing the added item or error
        """

    @abstractmethod
    def remove(self, item_id: str) -> Result[bool]:
        """
        Remove an item from the managed collection.

        Args:
            item_id: ID of the item to remove

        Returns:
            Result indicating success/failure
        """

    @abstractmethod
    def get(self, item_id: str) -> Result[Optional[T]]:
        """
        Get an item by its ID.

        Args:
            item_id: ID of the item to retrieve

        Returns:
            Result containing the item or None if not found
        """

    @abstractmethod
    def list_all(self) -> Result[List[T]]:
        """
        List all items in the managed collection.

        Returns:
            Result containing list of all items
        """

    @abstractmethod
    def list_by_criteria(self, criteria: Dict[str, Any]) -> Result[List[T]]:
        """
        List items matching the given criteria.

        Args:
            criteria: Dictionary of criteria to match

        Returns:
            Result containing list of matching items
        """

    @abstractmethod
    def search(self, query: str, limit: int = 10) -> Result[List[T]]:
        """
        Search for items matching the query.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            Result containing list of matching items
        """

    @abstractmethod
    def count(self) -> Result[int]:
        """
        Get the total count of items.

        Returns:
            Result containing the count
        """

    @abstractmethod
    def exists(self, item_id: str) -> Result[bool]:
        """
        Check if an item exists.

        Args:
            item_id: ID of the item to check

        Returns:
            Result indicating whether the item exists
        """

    @abstractmethod
    def validate(self, item: T) -> Result[bool]:
        """
        Validate an item according to business rules.

        Args:
            item: Item to validate

        Returns:
            Result indicating whether the item is valid
        """

    @abstractmethod
    def get_statistics(self) -> Result[Dict[str, Any]]:
        """
        Get statistics about the managed collection.

        Returns:
            Result containing statistics dictionary
        """


class EntityManager(Manager["Entity"], ABC):
    """Abstract interface for entity management operations."""

    @abstractmethod
    def list_by_type(self, entity_type: Union[EntityType, str]) -> Result[List["Entity"]]:
        """
        List entities by type.

        Args:
            entity_type: Type of entities to list

        Returns:
            Result containing list of entities of the specified type
        """

    @abstractmethod
    def get_related_entities(
        self, entity_id: str, relation_types: Optional[List[RelationType]] = None
    ) -> Result[List["Entity"]]:
        """
        Get entities related to the given entity.

        Args:
            entity_id: ID of the source entity
            relation_types: Optional filter for relation types

        Returns:
            Result containing list of related entities
        """

    @abstractmethod
    def update_confidence(self, entity_id: str, confidence: float) -> Result[bool]:
        """
        Update the confidence score of an entity.

        Args:
            entity_id: ID of the entity
            confidence: New confidence score

        Returns:
            Result indicating success/failure
        """


class RelationManager(Manager["Relation"], ABC):
    """Abstract interface for relation management operations."""

    @abstractmethod
    def list_by_type(self, relation_type: Union[RelationType, str]) -> Result[List["Relation"]]:
        """
        List relations by type.

        Args:
            relation_type: Type of relations to list

        Returns:
            Result containing list of relations of the specified type
        """

    @abstractmethod
    def get_entity_relations(
        self, entity_id: str, direction: str = "both"
    ) -> Result[List["Relation"]]:
        """
        Get all relations connected to an entity.

        Args:
            entity_id: ID of the entity
            direction: Direction filter ("incoming", "outgoing", "both")

        Returns:
            Result containing list of related relations
        """

    @abstractmethod
    def find_path(
        self, source_id: str, target_id: str, max_depth: int = 3
    ) -> Result[List["Relation"]]:
        """
        Find a path between two entities.

        Args:
            source_id: ID of the source entity
            target_id: ID of the target entity
            max_depth: Maximum path length to search

        Returns:
            Result containing list of relations forming a path
        """


class ClusterManager(Manager["Cluster"], ABC):
    """Abstract interface for cluster management operations."""

    @abstractmethod
    def list_by_type(self, cluster_type: Union[ClusterType, str]) -> Result[List["Cluster"]]:
        """
        List clusters by type.

        Args:
            cluster_type: Type of clusters to list

        Returns:
            Result containing list of clusters of the specified type
        """

    @abstractmethod
    def get_cluster_entities(self, cluster_id: str) -> Result[List["Entity"]]:
        """
        Get all entities in a cluster.

        Args:
            cluster_id: ID of the cluster

        Returns:
            Result containing list of entities in the cluster
        """

    @abstractmethod
    def add_entity_to_cluster(self, cluster_id: str, entity_id: str) -> Result[bool]:
        """
        Add an entity to a cluster.

        Args:
            cluster_id: ID of the cluster
            entity_id: ID of the entity

        Returns:
            Result indicating success/failure
        """

    @abstractmethod
    def remove_entity_from_cluster(self, cluster_id: str, entity_id: str) -> Result[bool]:
        """
        Remove an entity from a cluster.

        Args:
            cluster_id: ID of the cluster
            entity_id: ID of the entity

        Returns:
            Result indicating success/failure
        """


class TextChunkManager(Manager["TextChunk"], ABC):
    """Abstract interface for text chunk management operations."""

    @abstractmethod
    def list_by_source(self, source: str) -> Result[List["TextChunk"]]:
        """
        List text chunks by source.

        Args:
            source: Source identifier

        Returns:
            Result containing list of text chunks from the source
        """

    @abstractmethod
    def get_chunk_entities(self, chunk_id: str) -> Result[List["Entity"]]:
        """
        Get all entities referenced in a text chunk.

        Args:
            chunk_id: ID of the text chunk

        Returns:
            Result containing list of entities in the chunk
        """

    @abstractmethod
    def get_chunk_relations(self, chunk_id: str) -> Result[List["Relation"]]:
        """
        Get all relations referenced in a text chunk.

        Args:
            chunk_id: ID of the text chunk

        Returns:
            Result containing list of relations in the chunk
        """


class BatchOperationManager(ABC):
    """
    Interface for batch operations across multiple managers.

    This interface allows for transactional operations that span
    multiple types of graph components.
    """

    @abstractmethod
    def begin_batch(self) -> Result["BatchContext"]:
        """
        Begin a batch operation context.

        Returns:
            Result containing batch context
        """

    @abstractmethod
    def commit_batch(self, context: "BatchContext") -> Result[Dict[str, Any]]:
        """
        Commit all operations in a batch.

        Args:
            context: Batch context

        Returns:
            Result containing operation summary
        """

    @abstractmethod
    def rollback_batch(self, context: "BatchContext") -> Result[bool]:
        """
        Rollback all operations in a batch.

        Args:
            context: Batch context

        Returns:
            Result indicating success/failure
        """


class ManagerFactory(ABC):
    """
    Abstract factory for creating manager instances.

    This factory allows for different implementations of managers
    (e.g., in-memory, database-backed, optimized versions) to be
    created based on configuration or runtime requirements.
    """

    @abstractmethod
    def create_entity_manager(self, config: Optional[Dict[str, Any]] = None) -> EntityManager:
        """Create an entity manager instance."""

    @abstractmethod
    def create_relation_manager(self, config: Optional[Dict[str, Any]] = None) -> RelationManager:
        """Create a relation manager instance."""

    @abstractmethod
    def create_cluster_manager(self, config: Optional[Dict[str, Any]] = None) -> ClusterManager:
        """Create a cluster manager instance."""

    @abstractmethod
    def create_text_chunk_manager(
        self, config: Optional[Dict[str, Any]] = None
    ) -> TextChunkManager:
        """Create a text chunk manager instance."""

    @abstractmethod
    def create_batch_operation_manager(
        self, config: Optional[Dict[str, Any]] = None
    ) -> BatchOperationManager:
        """Create a batch operation manager instance."""
