"""
Storage interfaces following Single Responsibility Principle

Each interface has a single, well-defined responsibility:
- GraphConnection: Connection management
- GraphCRUD: Basic CRUD operations
- GraphQuery: Query operations
- GraphBackup: Backup and restore operations
- GraphExport: Export functionality
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation
from ..text import TextChunk


class GraphConnection(ABC):
    """Interface for storage connection management"""

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to storage backend

        Returns:
            bool: True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from storage backend"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if connected to storage backend

        Returns:
            bool: True if connected
        """
        pass


class GraphCRUD(ABC):
    """Interface for basic graph CRUD operations"""

    @abstractmethod
    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """
        Save knowledge graph

        Args:
            graph: Knowledge graph to save

        Returns:
            bool: True if save successful
        """
        pass

    @abstractmethod
    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """
        Load knowledge graph

        Args:
            graph_id: Graph ID

        Returns:
            KnowledgeGraph: Loaded graph, None if not found
        """
        pass

    @abstractmethod
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete knowledge graph

        Args:
            graph_id: Graph ID

        Returns:
            bool: True if delete successful
        """
        pass

    @abstractmethod
    def list_graphs(self) -> List[Dict[str, Any]]:
        """
        List all graphs

        Returns:
            List[Dict[str, Any]]: Graph metadata list
        """
        pass


class GraphEntityCRUD(ABC):
    """Interface for entity CRUD operations"""

    @abstractmethod
    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        Add entity to graph

        Args:
            graph_id: Graph ID
            entity: Entity to add

        Returns:
            bool: True if add successful
        """
        pass

    @abstractmethod
    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        Update entity in graph

        Args:
            graph_id: Graph ID
            entity: Entity to update

        Returns:
            bool: True if update successful
        """
        pass

    @abstractmethod
    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """
        Remove entity from graph

        Args:
            graph_id: Graph ID
            entity_id: Entity ID to remove

        Returns:
            bool: True if remove successful
        """
        pass


class GraphRelationCRUD(ABC):
    """Interface for relation CRUD operations"""

    @abstractmethod
    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        Add relation to graph

        Args:
            graph_id: Graph ID
            relation: Relation to add

        Returns:
            bool: True if add successful
        """
        pass

    @abstractmethod
    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        Update relation in graph

        Args:
            graph_id: Graph ID
            relation: Relation to update

        Returns:
            bool: True if update successful
        """
        pass

    @abstractmethod
    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """
        Remove relation from graph

        Args:
            graph_id: Graph ID
            relation_id: Relation ID to remove

        Returns:
            bool: True if remove successful
        """
        pass


class GraphTextChunkCRUD(ABC):
    """Interface for text chunk CRUD operations"""

    @abstractmethod
    def add_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """
        Add text chunk to graph

        Args:
            graph_id: Graph ID
            text_chunk: TextChunk to add

        Returns:
            bool: True if add successful
        """
        pass

    @abstractmethod
    def update_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """
        Update text chunk in graph

        Args:
            graph_id: Graph ID
            text_chunk: TextChunk to update

        Returns:
            bool: True if update successful
        """
        pass

    @abstractmethod
    def remove_text_chunk(self, graph_id: str, chunk_id: str) -> bool:
        """
        Remove text chunk from graph

        Args:
            graph_id: Graph ID
            chunk_id: Text chunk ID to remove

        Returns:
            bool: True if remove successful
        """
        pass

    @abstractmethod
    def batch_add_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """
        Batch add text chunks to graph

        Args:
            graph_id: Graph ID
            text_chunks: List of TextChunk objects to add

        Returns:
            bool: True if batch add successful
        """
        pass

    @abstractmethod
    def batch_update_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """
        Batch update text chunks in graph

        Args:
            graph_id: Graph ID
            text_chunks: List of TextChunk objects to update

        Returns:
            bool: True if batch update successful
        """
        pass

    @abstractmethod
    def batch_remove_text_chunks(self, graph_id: str, chunk_ids: List[str]) -> bool:
        """
        Batch remove text chunks from graph

        Args:
            graph_id: Graph ID
            chunk_ids: List of text chunk IDs to remove

        Returns:
            bool: True if batch remove successful
        """
        pass


class GraphQuery(ABC):
    """Interface for graph query operations"""

    @abstractmethod
    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """
        Query entities based on conditions

        Args:
            conditions: Query conditions

        Returns:
            List[Entity]: Matching entities
        """
        pass

    @abstractmethod
    def query_relations(
        self,
        head_entity: Optional[str] = None,
        tail_entity: Optional[str] = None,
        relation_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Relation]:
        """
        Query relations based on conditions

        Args:
            head_entity: Head entity ID
            tail_entity: Tail entity ID
            relation_type: Relation type
            **kwargs: Additional query parameters

        Returns:
            List[Relation]: Matching relations
        """
        pass

    @abstractmethod
    def query_text_chunks(self, conditions: Dict[str, Any]) -> List[TextChunk]:
        """
        Query text chunks based on conditions

        Args:
            conditions: Query conditions

        Returns:
            List[TextChunk]: Matching text chunks
        """
        pass


class GraphTextChunkQuery(ABC):
    """Interface for advanced text chunk query operations"""

    @abstractmethod
    def get_text_chunk(self, graph_id: str, chunk_id: str) -> Optional[TextChunk]:
        """
        Get text chunk by ID

        Args:
            graph_id: Graph ID
            chunk_id: Text chunk ID

        Returns:
            TextChunk: Text chunk object, None if not found
        """
        pass

    @abstractmethod
    def get_chunks_by_entity(self, graph_id: str, entity_id: str) -> List[TextChunk]:
        """
        Get all text chunks connected to a specific entity

        Args:
            graph_id: Graph ID
            entity_id: Entity ID

        Returns:
            List[TextChunk]: Text chunks connected to the entity
        """
        pass

    @abstractmethod
    def get_chunks_by_relation(self, graph_id: str, relation_id: str) -> List[TextChunk]:
        """
        Get all text chunks connected to a specific relation

        Args:
            graph_id: Graph ID
            relation_id: Relation ID

        Returns:
            List[TextChunk]: Text chunks connected to the relation
        """
        pass

    @abstractmethod
    def get_chunks_by_source(self, graph_id: str, source: str) -> List[TextChunk]:
        """
        Get all text chunks from a specific source

        Args:
            graph_id: Graph ID
            source: Source document or origin

        Returns:
            List[TextChunk]: Text chunks from the specified source
        """
        pass

    @abstractmethod
    def get_chunks_by_type(self, graph_id: str, chunk_type: str) -> List[TextChunk]:
        """
        Get all text chunks of a specific type

        Args:
            graph_id: Graph ID
            chunk_type: Type of text chunks to retrieve

        Returns:
            List[TextChunk]: Text chunks of the specified type
        """
        pass

    @abstractmethod
    def get_chunks_by_language(self, graph_id: str, language: str) -> List[TextChunk]:
        """
        Get all text chunks in a specific language

        Args:
            graph_id: Graph ID
            language: Language code (e.g., 'zh', 'en')

        Returns:
            List[TextChunk]: Text chunks in the specified language
        """
        pass

    @abstractmethod
    def search_text_chunks(
        self,
        graph_id: str,
        query: str,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 10,
    ) -> List[TextChunk]:
        """
        Search text chunks by content or metadata

        Args:
            graph_id: Graph ID
            query: Search query string
            chunk_type: Optional chunk type filter
            language: Optional language filter
            limit: Maximum number of results

        Returns:
            List[TextChunk]: Matching text chunks
        """
        pass


class GraphTextChunkEmbedding(ABC):
    """Interface for text chunk embedding operations"""

    @abstractmethod
    def add_chunk_embedding(self, graph_id: str, chunk_id: str, embedding: List[float]) -> bool:
        """
        Add or update embedding for a text chunk

        Args:
            graph_id: Graph ID
            chunk_id: Text chunk ID
            embedding: Vector embedding of the text chunk

        Returns:
            bool: True if add/update successful
        """
        pass

    @abstractmethod
    def get_chunk_embedding(self, graph_id: str, chunk_id: str) -> Optional[List[float]]:
        """
        Get embedding for a text chunk

        Args:
            graph_id: Graph ID
            chunk_id: Text chunk ID

        Returns:
            Optional[List[float]]: Vector embedding, None if not found
        """
        pass

    @abstractmethod
    def search_chunks_by_embedding(
        self,
        graph_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[tuple[TextChunk, float]]:
        """
        Search text chunks by embedding similarity

        Args:
            graph_id: Graph ID
            query_embedding: Query vector embedding
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            chunk_type: Optional filter by chunk type
            language: Optional filter by language

        Returns:
            List[tuple[TextChunk, float]]: List of (chunk, similarity_score) tuples
        """
        pass

    @abstractmethod
    def search_chunks_hybrid(
        self,
        graph_id: str,
        query_text: str,
        query_embedding: Optional[List[float]] = None,
        top_k: int = 10,
        text_weight: float = 0.3,
        embedding_weight: float = 0.7,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[tuple[TextChunk, float]]:
        """
        Hybrid search combining text and embedding similarity

        Args:
            graph_id: Graph ID
            query_text: Query text string
            query_embedding: Optional query vector embedding
            top_k: Maximum number of results to return
            text_weight: Weight for text similarity (0.0 to 1.0)
            embedding_weight: Weight for embedding similarity (0.0 to 1.0)
            chunk_type: Optional filter by chunk type
            language: Optional filter by language

        Returns:
            List[tuple[TextChunk, float]]: List of (chunk, combined_score) tuples
        """
        pass


class GraphTextChunkAnalysis(ABC):
    """Interface for text chunk analysis operations"""

    @abstractmethod
    def get_chunk_context(self, graph_id: str, chunk_id: str) -> Dict[str, Any]:
        """
        Get contextual information for a text chunk

        Args:
            graph_id: Graph ID
            chunk_id: Text chunk ID

        Returns:
            Dict[str, Any]: Context information
        """
        pass

    @abstractmethod
    def get_chunk_neighbors(self, graph_id: str, chunk_id: str, max_distance: int = 2) -> List[tuple[TextChunk, int]]:
        """
        Get neighboring text chunks through entity/relation connections

        Args:
            graph_id: Graph ID
            chunk_id: Text chunk ID
            max_distance: Maximum connection distance

        Returns:
            List[tuple[TextChunk, int]]: List of (chunk, distance) tuples
        """
        pass

    @abstractmethod
    def find_orphaned_chunks(self, graph_id: str) -> List[TextChunk]:
        """
        Find text chunks that have no connections to entities or relations

        Args:
            graph_id: Graph ID

        Returns:
            List[TextChunk]: Chunks with no connections
        """
        pass

    @abstractmethod
    def find_highly_connected_chunks(self, graph_id: str, min_connections: int = 5) -> List[tuple[TextChunk, int]]:
        """
        Find text chunks with many entity/relation connections

        Args:
            graph_id: Graph ID
            min_connections: Minimum number of connections

        Returns:
            List[tuple[TextChunk, int]]: List of (chunk, connection_count) tuples
        """
        pass

    @abstractmethod
    def validate_chunk_connections(self, graph_id: str) -> Dict[str, Any]:
        """
        Validate that all chunk connections point to existing entities and relations

        Args:
            graph_id: Graph ID

        Returns:
            Dict[str, Any]: Validation results
        """
        pass

    @abstractmethod
    def get_chunk_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about text chunks in the graph

        Args:
            graph_id: Graph ID

        Returns:
            Dict[str, Any]: Comprehensive chunk statistics
        """
        pass

    @abstractmethod
    def cluster_chunks_by_similarity(self, graph_id: str, threshold: float = 0.5) -> List[List[str]]:
        """
        Cluster text chunks based on similarity

        Args:
            graph_id: Graph ID
            threshold: Similarity threshold for clustering

        Returns:
            List[List[str]]: List of clusters, each containing chunk IDs
        """
        pass


class GraphBackup(ABC):
    """Interface for graph backup and restore operations"""

    @abstractmethod
    def backup_graph(self, graph_id: str, backup_path: str) -> bool:
        """
        Backup graph to file

        Args:
            graph_id: Graph ID
            backup_path: Path to backup file

        Returns:
            bool: True if backup successful
        """
        pass

    @abstractmethod
    def restore_graph(self, backup_path: str) -> Optional[str]:
        """
        Restore graph from backup file

        Args:
            backup_path: Path to backup file

        Returns:
            str: Restored graph ID, None if failed
        """
        pass


class GraphExport(ABC):
    """Interface for graph export operations"""

    @abstractmethod
    def export_graph(self, graph_id: str, format: str = "json") -> Optional[Dict[str, Any]]:
        """
        Export graph in specified format

        Args:
            graph_id: Graph ID
            format: Export format ('json', 'csv', 'graphml', etc.)

        Returns:
            Dict[str, Any]: Exported data, None if failed
        """
        pass


class GraphStatistics(ABC):
    """Interface for graph statistics operations"""

    @abstractmethod
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get graph statistics

        Args:
            graph_id: Graph ID

        Returns:
            Dict[str, Any]: Statistics data
        """
        pass


# Composition interfaces for different use cases


class BasicGraphStorage(GraphConnection, GraphCRUD):
    """Basic graph storage interface with connection and CRUD operations"""

    pass


class QueryableGraphStorage(BasicGraphStorage, GraphQuery, GraphTextChunkQuery):
    """Graph storage with query capabilities"""

    pass


class TextChunkGraphStorage(
    BasicGraphStorage, GraphTextChunkCRUD, GraphTextChunkQuery, GraphTextChunkEmbedding, GraphTextChunkAnalysis
):
    """Graph storage with comprehensive text chunk capabilities"""

    pass


class FullGraphStorage(
    BasicGraphStorage,
    GraphEntityCRUD,
    GraphRelationCRUD,
    GraphTextChunkCRUD,
    GraphQuery,
    GraphTextChunkQuery,
    GraphTextChunkEmbedding,
    GraphTextChunkAnalysis,
    GraphBackup,
    GraphExport,
    GraphStatistics,
):
    """Full-featured graph storage interface"""

    pass


class ReadOnlyGraphStorage(GraphConnection, GraphQuery, GraphStatistics):
    """Read-only graph storage interface"""

    pass


# Vector Storage Interfaces


class VectorStorageConnection(ABC):
    """Interface for vector storage connection management"""

    @abstractmethod
    def connect(self) -> bool:
        """Connect to vector storage backend"""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from vector storage backend"""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to vector storage backend"""
        pass


class VectorStorageCRUD(ABC):
    """Interface for vector CRUD operations"""

    @abstractmethod
    def add_vector(self, vector_id: str, vector: Any, metadata: Optional[Dict] = None) -> bool:
        """Add vector to storage"""
        pass

    @abstractmethod
    def get_vector(self, vector_id: str) -> Optional[Any]:
        """Get vector from storage"""
        pass

    @abstractmethod
    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from storage"""
        pass

    @abstractmethod
    def save_vectors(self, vectors: Dict[str, Any], metadata: Optional[Dict] = None) -> bool:
        """Batch save vectors"""
        pass

    @abstractmethod
    def load_vectors(self) -> tuple[Dict[str, Any], Dict]:
        """Load all vectors"""
        pass

    @abstractmethod
    def clear(self) -> bool:
        """Clear all vectors"""
        pass


class VectorStorageQuery(ABC):
    """Interface for vector query operations"""

    @abstractmethod
    def search_similar_vectors(
        self, query_vector: Any, top_k: int = 10, threshold: float = 0.0
    ) -> List[tuple[str, float]]:
        """Search for similar vectors"""
        pass

    @staticmethod
    def compute_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            float: Similarity score (0-1)
        """
        try:
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(max(0.0, min(1.0, similarity)))
        except Exception as e:
            logger.error("Error computing cosine similarity: %s", e)
            return 0.0


class VectorStorage(VectorStorageConnection, VectorStorageCRUD, VectorStorageQuery):
    """Complete vector storage interface"""

    def save(self) -> None:
        """
        Save the current state of the vector storage.

        This method should be implemented to persist the current state
        to a file or database.
        """
