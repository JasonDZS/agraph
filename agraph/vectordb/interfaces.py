"""
Vector store interface definitions.

This module defines small, focused interfaces that follow the Interface Segregation Principle,
allowing implementations to only depend on the interfaces they actually use.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from ..base.clusters import Cluster
from ..base.entities import Entity
from ..base.relations import Relation
from ..base.text import TextChunk


class VectorStoreCore(ABC):
    """Core vector store interface with basic lifecycle methods."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize vector storage connection and configuration."""

    @abstractmethod
    async def close(self) -> None:
        """Close vector storage connection."""

    @abstractmethod
    async def get_stats(self) -> Dict[str, int]:
        """Get vector storage statistics.

        Returns:
            Dictionary containing counts of each object type
        """

    @abstractmethod
    async def clear_all(self) -> bool:
        """Clear all data.

        Returns:
            Whether the operation succeeded
        """

    def is_initialized(self) -> bool:
        """Check if vector storage is initialized.

        Returns:
            Whether it is initialized
        """
        return getattr(self, "_is_initialized", False)


class EntityStore(ABC):
    """Interface for entity storage operations."""

    @abstractmethod
    async def add_entity(self, entity: Entity, embedding: Optional[List[float]] = None) -> bool:
        """Add entity to vector storage."""

    @abstractmethod
    async def update_entity(self, entity: Entity, embedding: Optional[List[float]] = None) -> bool:
        """Update entity information."""

    @abstractmethod
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete entity."""

    @abstractmethod
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""

    @abstractmethod
    async def search_entities(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Entity, float]]:
        """Search for similar entities."""

    @abstractmethod
    async def batch_add_entities(
        self,
        entities: List[Entity],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[bool]:
        """Batch add entities."""


class RelationStore(ABC):
    """Interface for relation storage operations."""

    @abstractmethod
    async def add_relation(
        self, relation: Relation, embedding: Optional[List[float]] = None
    ) -> bool:
        """Add relation to vector storage."""

    @abstractmethod
    async def update_relation(
        self, relation: Relation, embedding: Optional[List[float]] = None
    ) -> bool:
        """Update relation information."""

    @abstractmethod
    async def delete_relation(self, relation_id: str) -> bool:
        """Delete relation."""

    @abstractmethod
    async def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""

    @abstractmethod
    async def search_relations(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Relation, float]]:
        """Search for similar relations."""

    @abstractmethod
    async def batch_add_relations(
        self,
        relations: List[Relation],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[bool]:
        """Batch add relations."""


class ClusterStore(ABC):
    """Interface for cluster storage operations."""

    @abstractmethod
    async def add_cluster(self, cluster: Cluster, embedding: Optional[List[float]] = None) -> bool:
        """Add cluster to vector storage."""

    @abstractmethod
    async def update_cluster(
        self, cluster: Cluster, embedding: Optional[List[float]] = None
    ) -> bool:
        """Update cluster information."""

    @abstractmethod
    async def delete_cluster(self, cluster_id: str) -> bool:
        """Delete cluster."""

    @abstractmethod
    async def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get cluster by ID."""

    @abstractmethod
    async def search_clusters(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Cluster, float]]:
        """Search for similar clusters."""

    @abstractmethod
    async def batch_add_clusters(
        self,
        clusters: List[Cluster],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[bool]:
        """Batch add clusters."""


class TextChunkStore(ABC):
    """Interface for text chunk storage operations."""

    @abstractmethod
    async def add_text_chunk(
        self, text_chunk: TextChunk, embedding: Optional[List[float]] = None
    ) -> bool:
        """Add text chunk to vector storage."""

    @abstractmethod
    async def update_text_chunk(
        self, text_chunk: TextChunk, embedding: Optional[List[float]] = None
    ) -> bool:
        """Update text chunk information."""

    @abstractmethod
    async def delete_text_chunk(self, chunk_id: str) -> bool:
        """Delete text chunk."""

    @abstractmethod
    async def get_text_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """Get text chunk by ID."""

    @abstractmethod
    async def search_text_chunks(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[TextChunk, float]]:
        """Search for similar text chunks."""

    @abstractmethod
    async def batch_add_text_chunks(
        self,
        text_chunks: List[TextChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[bool]:
        """Batch add text chunks."""


class VectorStore(VectorStoreCore, EntityStore, RelationStore, ClusterStore, TextChunkStore):
    """Complete vector store interface combining all storage capabilities.

    This interface provides the full functionality by combining all the specialized interfaces.
    Implementations can inherit from this for complete functionality, or implement only
    the specific interfaces they need.
    """

    def __init__(self, collection_name: str = "knowledge_graph", **_: Any) -> None:
        """Initialize vector storage.

        Args:
            collection_name: Collection/table name
            **_: Additional initialization parameters
        """
        self.collection_name = collection_name
        self._is_initialized = False

    def _validate_embedding(self, embedding: Optional[List[float]]) -> bool:
        """Validate vector embedding format.

        Args:
            embedding: Vector embedding

        Returns:
            Whether it is valid
        """
        if embedding is None:
            return True
        return isinstance(embedding, list) and all(isinstance(x, (int, float)) for x in embedding)

    async def __aenter__(self) -> "VectorStore":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.close()
