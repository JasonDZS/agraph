"""
In-memory vector store implementation.

Provides an in-memory vector store implementation for testing and small-scale applications.
Uses simple cosine similarity for vector search.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from ..base.models.clusters import Cluster
from ..base.models.entities import Entity
from ..base.models.relations import Relation
from ..base.models.text import TextChunk
from ..logger import logger
from .constants import DEFAULT_EMBEDDING_DIMENSION, ERROR_MESSAGES
from .embeddings import create_openai_embedding_function
from .exceptions import VectorStoreError
from .interfaces import VectorStore
from .mixins import EmbeddingStatsMixin, HybridSearchMixin
from .query_builder import EmbeddingGenerator, ResultProcessor


class MemoryVectorStore(VectorStore, EmbeddingStatsMixin, HybridSearchMixin):
    """In-memory vector store implementation.

    Stores all data in memory and provides basic vector search functionality.
    Suitable for testing and small datasets.
    """

    # pylint: disable=too-many-public-methods,too-many-ancestors

    def __init__(
        self,
        collection_name: str = "knowledge_graph",
        use_openai_embeddings: bool = False,
        openai_embedding_config: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        """Initialize memory vector store.

        Args:
            collection_name: Collection name
            use_openai_embeddings: Whether to use OpenAI embeddings
            openai_embedding_config: OpenAI embedding function configuration parameters
            **_: Additional parameters
        """
        super().__init__(collection_name, **_)

        # Data storage
        self._entities: Dict[str, Entity] = {}
        self._relations: Dict[str, Relation] = {}
        self._clusters: Dict[str, Cluster] = {}
        self._text_chunks: Dict[str, TextChunk] = {}

        # Vector storage
        self._entity_embeddings: Dict[str, List[float]] = {}
        self._relation_embeddings: Dict[str, List[float]] = {}
        self._cluster_embeddings: Dict[str, List[float]] = {}
        self._text_chunk_embeddings: Dict[str, List[float]] = {}

        # Handle embedding function configuration
        self._openai_embedding: Optional[Any] = None
        if use_openai_embeddings:
            try:
                embedding_config = openai_embedding_config or {}
                self._openai_embedding = create_openai_embedding_function(**embedding_config)
            except Exception as e:
                # If OpenAI is not available, fall back to simple embedding
                logger.error(f"Warning: Failed to create OpenAI embedding function: {e}")
                logger.error("Falling back to simple character-based embeddings.")
                self._openai_embedding = None
                raise VectorStoreError(ERROR_MESSAGES["openai_embedding_failed"].format(error=e)) from e

    async def initialize(self) -> None:
        """Initialize vector store."""
        self._is_initialized = True

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Cosine similarity [-1, 1]
        """
        if len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(a * a for a in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _generate_text_embedding(self, text: str) -> List[float]:
        """Generate simple embedding vector for text.

        This is a very simple implementation.
        In practice, professional embedding models should be used.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        return EmbeddingGenerator.generate_simple_embedding(text, DEFAULT_EMBEDDING_DIMENSION)

    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding vector using OpenAI API.

        Args:
            text: Input text

        Returns:
            Embedding vector
        """
        if self._openai_embedding:
            try:
                result = await self._openai_embedding.embed_single(text)
                return list(result)  # Ensure it's List[float]
            except Exception as e:
                print(f"Warning: OpenAI embedding failed, falling back to simple embedding: {e}")
                raise VectorStoreError(ERROR_MESSAGES["openai_embedding_failed"].format(error=e)) from e

        # Fall back to simple embedding
        return self._generate_text_embedding(text)

    # Entity operations
    async def add_entity(self, entity: Entity, embedding: Optional[List[float]] = None) -> bool:
        """Add entity to vector store."""
        try:
            self._entities[entity.id] = entity

            if embedding is None:
                # Generate embedding using name and description
                text = f"{entity.name} {entity.description}"
                if self._openai_embedding:
                    embedding = await self._generate_openai_embedding(text)
                else:
                    embedding = self._generate_text_embedding(text)

            if not self._validate_embedding(embedding):
                raise VectorStoreError(
                    ERROR_MESSAGES["invalid_embedding"].format(object_type="entity", object_id=entity.id)
                )

            self._entity_embeddings[entity.id] = embedding
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to add entity {entity.id}: {e}") from e

    async def update_entity(self, entity: Entity, embedding: Optional[List[float]] = None) -> bool:
        """Update entity information."""
        if entity.id not in self._entities:
            return False
        return await self.add_entity(entity, embedding)

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete entity."""
        if entity_id in self._entities:
            del self._entities[entity_id]
            self._entity_embeddings.pop(entity_id, None)
            return True
        return False

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        return self._entities.get(entity_id)

    async def search_entities(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Entity, float]]:
        """Search similar entities."""
        if isinstance(query, str):
            query_embedding = self._generate_text_embedding(query)
        else:
            query_embedding = query

        results = []
        for entity_id, entity in self._entities.items():
            # Apply filter conditions
            if not ResultProcessor.apply_filters(entity, filter_dict):
                continue

            entity_embedding = self._entity_embeddings.get(entity_id)
            if entity_embedding:
                similarity = self._cosine_similarity(query_embedding, entity_embedding)
                results.append((entity, similarity))

        # Sort by similarity and return top k
        results = ResultProcessor.sort_by_similarity(results, descending=True)
        return results[:top_k]

    # Relation operations
    async def add_relation(self, relation: Relation, embedding: Optional[List[float]] = None) -> bool:
        """Add relation to vector store."""
        try:
            self._relations[relation.id] = relation

            if embedding is None:
                # Generate embedding using description and relation_type
                text = f"{relation.relation_type} {relation.description}"
                if self._openai_embedding:
                    embedding = await self._generate_openai_embedding(text)
                else:
                    embedding = self._generate_text_embedding(text)

            if not self._validate_embedding(embedding):
                raise VectorStoreError(
                    ERROR_MESSAGES["invalid_embedding"].format(object_type="relation", object_id=relation.id)
                )

            self._relation_embeddings[relation.id] = embedding
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to add relation {relation.id}: {e}") from e

    async def update_relation(self, relation: Relation, embedding: Optional[List[float]] = None) -> bool:
        """Update relation information."""
        if relation.id not in self._relations:
            return False
        return await self.add_relation(relation, embedding)

    async def delete_relation(self, relation_id: str) -> bool:
        """Delete relation."""
        if relation_id in self._relations:
            del self._relations[relation_id]
            self._relation_embeddings.pop(relation_id, None)
            return True
        return False

    async def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        return self._relations.get(relation_id)

    async def search_relations(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Relation, float]]:
        """Search similar relations."""
        if isinstance(query, str):
            query_embedding = self._generate_text_embedding(query)
        else:
            query_embedding = query

        results = []
        for relation_id, relation in self._relations.items():
            # Apply filter conditions
            if not ResultProcessor.apply_filters(relation, filter_dict):
                continue

            relation_embedding = self._relation_embeddings.get(relation_id)
            if relation_embedding:
                similarity = self._cosine_similarity(query_embedding, relation_embedding)
                results.append((relation, similarity))

        results = ResultProcessor.sort_by_similarity(results, descending=True)
        return results[:top_k]

    # Cluster operations
    async def add_cluster(self, cluster: Cluster, embedding: Optional[List[float]] = None) -> bool:
        """Add cluster to vector store."""
        try:
            self._clusters[cluster.id] = cluster

            if embedding is None:
                text = f"{cluster.name} {cluster.description}"
                if self._openai_embedding:
                    embedding = await self._generate_openai_embedding(text)
                else:
                    embedding = self._generate_text_embedding(text)

            if not self._validate_embedding(embedding):
                raise VectorStoreError(
                    ERROR_MESSAGES["invalid_embedding"].format(object_type="cluster", object_id=cluster.id)
                )

            self._cluster_embeddings[cluster.id] = embedding
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to add cluster {cluster.id}: {e}") from e

    async def update_cluster(self, cluster: Cluster, embedding: Optional[List[float]] = None) -> bool:
        """Update cluster information."""
        if cluster.id not in self._clusters:
            return False
        return await self.add_cluster(cluster, embedding)

    async def delete_cluster(self, cluster_id: str) -> bool:
        """Delete cluster."""
        if cluster_id in self._clusters:
            del self._clusters[cluster_id]
            self._cluster_embeddings.pop(cluster_id, None)
            return True
        return False

    async def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get cluster by ID."""
        return self._clusters.get(cluster_id)

    async def search_clusters(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Cluster, float]]:
        """Search similar clusters."""
        if isinstance(query, str):
            query_embedding = self._generate_text_embedding(query)
        else:
            query_embedding = query

        results = []
        for cluster_id, cluster in self._clusters.items():
            if not ResultProcessor.apply_filters(cluster, filter_dict):
                continue

            cluster_embedding = self._cluster_embeddings.get(cluster_id)
            if cluster_embedding:
                similarity = self._cosine_similarity(query_embedding, cluster_embedding)
                results.append((cluster, similarity))

        results = ResultProcessor.sort_by_similarity(results, descending=True)
        return results[:top_k]

    # TextChunk operations
    async def add_text_chunk(self, text_chunk: TextChunk, embedding: Optional[List[float]] = None) -> bool:
        """Add text chunk to vector store."""
        try:
            self._text_chunks[text_chunk.id] = text_chunk

            if embedding is None:
                text = f"{text_chunk.title} {text_chunk.content}"
                if self._openai_embedding:
                    embedding = await self._generate_openai_embedding(text)
                else:
                    embedding = self._generate_text_embedding(text)

            if not self._validate_embedding(embedding):
                raise VectorStoreError(
                    ERROR_MESSAGES["invalid_embedding"].format(object_type="text_chunk", object_id=text_chunk.id)
                )

            self._text_chunk_embeddings[text_chunk.id] = embedding
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to add text_chunk {text_chunk.id}: {e}") from e

    async def update_text_chunk(self, text_chunk: TextChunk, embedding: Optional[List[float]] = None) -> bool:
        """Update text chunk information."""
        if text_chunk.id not in self._text_chunks:
            return False
        return await self.add_text_chunk(text_chunk, embedding)

    async def delete_text_chunk(self, chunk_id: str) -> bool:
        """Delete text chunk."""
        if chunk_id in self._text_chunks:
            del self._text_chunks[chunk_id]
            self._text_chunk_embeddings.pop(chunk_id, None)
            return True
        return False

    async def get_text_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """Get text chunk by ID."""
        return self._text_chunks.get(chunk_id)

    async def search_text_chunks(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[TextChunk, float]]:
        """Search similar text chunks."""
        if isinstance(query, str):
            query_embedding = self._generate_text_embedding(query)
        else:
            query_embedding = query

        results = []
        for chunk_id, chunk in self._text_chunks.items():
            if not ResultProcessor.apply_filters(chunk, filter_dict):
                continue

            chunk_embedding = self._text_chunk_embeddings.get(chunk_id)
            if chunk_embedding:
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                results.append((chunk, similarity))

        results = ResultProcessor.sort_by_similarity(results, descending=True)
        return results[:top_k]

    # Batch operations
    async def batch_add_entities(
        self,
        entities: List[Entity],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[bool]:
        """Batch add entities."""
        results = []
        if embeddings is None:
            embedding_list: List[Optional[List[float]]] = [None] * len(entities)
        else:
            embedding_list = list(embeddings)

        for entity, embedding in zip(entities, embedding_list):
            result = await self.add_entity(entity, embedding)
            results.append(result)

        return results

    async def batch_add_relations(
        self,
        relations: List[Relation],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[bool]:
        """Batch add relations."""
        results = []
        if embeddings is None:
            embedding_list: List[Optional[List[float]]] = [None] * len(relations)
        else:
            embedding_list = list(embeddings)

        for relation, embedding in zip(relations, embedding_list):
            result = await self.add_relation(relation, embedding)
            results.append(result)

        return results

    async def batch_add_clusters(
        self,
        clusters: List[Cluster],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[bool]:
        """Batch add clusters."""
        results = []
        if embeddings is None:
            embedding_list: List[Optional[List[float]]] = [None] * len(clusters)
        else:
            embedding_list = list(embeddings)

        for cluster, embedding in zip(clusters, embedding_list):
            result = await self.add_cluster(cluster, embedding)
            results.append(result)

        return results

    async def batch_add_text_chunks(
        self,
        text_chunks: List[TextChunk],
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[bool]:
        """Batch add text chunks."""
        results = []
        if embeddings is None:
            embedding_list: List[Optional[List[float]]] = [None] * len(text_chunks)
        else:
            embedding_list = list(embeddings)

        for chunk, embedding in zip(text_chunks, embedding_list):
            result = await self.add_text_chunk(chunk, embedding)
            results.append(result)

        return results

    # Utility methods
    async def get_stats(self) -> Dict[str, int]:
        """Get vector store statistics."""
        return {
            "entities": len(self._entities),
            "relations": len(self._relations),
            "clusters": len(self._clusters),
            "text_chunks": len(self._text_chunks),
        }

    async def clear_all(self) -> bool:
        """Clear all data."""
        try:
            self._entities.clear()
            self._relations.clear()
            self._clusters.clear()
            self._text_chunks.clear()

            self._entity_embeddings.clear()
            self._relation_embeddings.clear()
            self._cluster_embeddings.clear()
            self._text_chunk_embeddings.clear()

            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to clear all data: {e}") from e

    async def close(self) -> None:
        """Close vector store connection."""
        # Close OpenAI embedding function connection
        if self._openai_embedding:
            try:
                await self._openai_embedding.close()
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    # Event loop is already closed, cleanup is not needed
                    logger.warning("Event loop is closed, OpenAI embedding cleanup skipped.")
                else:
                    raise VectorStoreError(f"Failed to close OpenAI embedding function: {e}") from e

        # Set initialization status to False
        self._is_initialized = False
