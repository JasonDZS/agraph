"""
ChromaDB vector store implementation.

ChromaDB-based vector store implementation with support for persistence
and high-performance vector search.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        "ChromaDB is not installed. Please install it with: "
        "pip install '.[vectordb]' or pip install chromadb>=0.5.0"
    ) from e

from ..base.clusters import Cluster
from ..base.entities import Entity
from ..base.relations import Relation
from ..base.text import TextChunk
from ..config import get_settings
from ..utils import get_type_value
from .constants import COLLECTION_SUFFIXES, ERROR_MESSAGES, MAX_CONTENT_LENGTH_IN_METADATA
from .embeddings import (
    ChromaEmbeddingFunction,
    OpenAIEmbeddingFunction,
    create_openai_embedding_function,
)
from .exceptions import VectorStoreError
from .interfaces import VectorStore
from .mixins import EmbeddingStatsMixin, HybridSearchMixin
from .query_builder import QueryBuilder, ResultProcessor


class ChromaVectorStore(VectorStore, EmbeddingStatsMixin, HybridSearchMixin):
    """ChromaDB-based vector store implementation.

    Uses ChromaDB as the vector database backend, supporting persistent storage,
    high-performance vector search and multiple embedding models.
    """

    # pylint: disable=too-many-public-methods,too-many-ancestors
    def __init__(
        self,
        collection_name: str = "knowledge_graph",
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        embedding_function: Optional[Any] = None,
        use_openai_embeddings: bool = False,
        openai_embedding_config: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> None:
        """Initialize ChromaDB vector store.

        Args:
            collection_name: Collection name
            persist_directory: Persistent directory path, None for in-memory mode
            host: ChromaDB server address (for client mode)
            port: ChromaDB server port (for client mode)
            embedding_function: Custom embedding function, None uses default
            use_openai_embeddings: Whether to use OpenAI embeddings,
                True will automatically create OpenAI embedding function
            openai_embedding_config: OpenAI embedding function configuration parameters
            **_: Additional ChromaDB configuration parameters
        """
        super().__init__(collection_name, **_)

        self.persist_directory = persist_directory
        self.host = host
        self.port = port
        # Handle embedding function configuration
        if embedding_function is None and use_openai_embeddings:
            # Create OpenAI embedding function
            try:
                embedding_function = create_openai_embedding_function(
                    **(openai_embedding_config or {})
                )
            except Exception as e:
                raise VectorStoreError(
                    ERROR_MESSAGES["openai_embedding_failed"].format(error=e)
                ) from e

        if isinstance(embedding_function, OpenAIEmbeddingFunction):
            # Wrap in ChromaEmbeddingFunction
            self.embedding_function: Any = ChromaEmbeddingFunction(embedding_function)
            self._openai_embedding: Optional[OpenAIEmbeddingFunction] = embedding_function
        else:
            # Use provided embedding function or default
            self.embedding_function = embedding_function
            self._openai_embedding = None

        self._client: Optional[Any] = None
        self._collections: Dict[str, Any] = {}

        # Collection name mapping
        self._collection_names = {
            data_type: f"{collection_name}{suffix}"
            for data_type, suffix in COLLECTION_SUFFIXES.items()
        }

    async def initialize(self) -> None:
        """Initialize ChromaDB client and collections."""
        try:
            # Create ChromaDB client
            if self.host and self.port:
                # Client mode
                self._client = chromadb.HttpClient(host=self.host, port=self.port)
            elif self.persist_directory:
                # Persistent mode
                self._client = chromadb.PersistentClient(
                    path=self.persist_directory,
                    settings=Settings(anonymized_telemetry=False, allow_reset=True),
                )
            else:
                # In-memory mode
                self._client = chromadb.Client(
                    settings=Settings(anonymized_telemetry=False, allow_reset=True)
                )

            # Create or get collections
            for data_type, collection_name in self._collection_names.items():
                try:
                    collection = self._client.get_collection(
                        name=collection_name, embedding_function=self.embedding_function
                    )
                except (ValueError, chromadb.errors.InvalidCollectionException):
                    # Collection doesn't exist, create new collection
                    collection = self._client.create_collection(
                        name=collection_name, embedding_function=self.embedding_function
                    )

                self._collections[data_type] = collection

            self._is_initialized = True
        except Exception as e:
            raise VectorStoreError(ERROR_MESSAGES["chromadb_init_failed"].format(error=e)) from e

    async def close(self) -> None:
        """Close ChromaDB connection."""
        self._client = None
        self._collections = {}
        self._is_initialized = False

    def _get_collection(self, data_type: str) -> Any:
        """Get collection for specified data type.

        Args:
            data_type: Data type ('entity', 'relation', 'cluster', 'text_chunk')

        Returns:
            ChromaDB collection object
        """
        if not self._is_initialized:
            raise VectorStoreError(ERROR_MESSAGES["not_initialized"])

        if data_type not in self._collections:
            raise VectorStoreError(ERROR_MESSAGES["unknown_data_type"].format(data_type=data_type))

        return self._collections[data_type]

    def _prepare_entity_data(self, entity: Entity) -> Tuple[str, str, Dict[str, Any]]:
        """Prepare entity data for ChromaDB storage.

        Args:
            entity: Entity object

        Returns:
            (id, document, metadata) tuple
        """
        document = f"{entity.name} {entity.description}"
        metadata = {
            "name": entity.name,
            "entity_type": get_type_value(entity.entity_type),
            "description": entity.description,
            "properties": json.dumps(entity.properties),
            "aliases": json.dumps(entity.aliases),
            "confidence": entity.confidence,
            "source": entity.source,
            "text_chunks": json.dumps(list(entity.text_chunks)),
            "created_at": entity.created_at.isoformat(),
            "updated_at": entity.updated_at.isoformat(),
        }
        return entity.id, document, metadata

    def _prepare_relation_data(self, relation: Relation) -> Tuple[str, str, Dict[str, Any]]:
        """Prepare relation data for ChromaDB storage."""
        document = f"{relation.relation_type} {relation.description}"
        metadata = {
            "head_entity_id": relation.head_entity.id if relation.head_entity else None,
            "tail_entity_id": relation.tail_entity.id if relation.tail_entity else None,
            "relation_type": get_type_value(relation.relation_type),
            "description": relation.description,
            "properties": json.dumps(relation.properties),
            "confidence": relation.confidence,
            "source": relation.source,
            "text_chunks": json.dumps(list(relation.text_chunks)),
            "created_at": relation.created_at.isoformat(),
            "updated_at": relation.updated_at.isoformat(),
        }
        return relation.id, document, metadata

    def _prepare_cluster_data(self, cluster: Cluster) -> Tuple[str, str, Dict[str, Any]]:
        """Prepare cluster data for ChromaDB storage."""
        document = f"{cluster.name} {cluster.description}"
        metadata = {
            "name": cluster.name,
            "cluster_type": get_type_value(cluster.cluster_type),
            "description": cluster.description,
            "entities": json.dumps(list(cluster.entities)),
            "relations": json.dumps(list(cluster.relations)),
            "centroid_entity_id": cluster.centroid_entity_id,
            "parent_cluster_id": cluster.parent_cluster_id,
            "child_clusters": json.dumps(list(cluster.child_clusters)),
            "cohesion_score": cluster.cohesion_score,
            "properties": json.dumps(cluster.properties),
            "confidence": cluster.confidence,
            "source": cluster.source,
            "text_chunks": json.dumps(list(cluster.text_chunks)),
            "created_at": cluster.created_at.isoformat(),
            "updated_at": cluster.updated_at.isoformat(),
        }
        return cluster.id, document, metadata

    def _prepare_text_chunk_data(self, text_chunk: TextChunk) -> Tuple[str, str, Dict[str, Any]]:
        """Prepare text chunk data for ChromaDB storage."""
        document = f"{text_chunk.title} {text_chunk.content}"
        metadata = {
            "title": text_chunk.title,
            "content": text_chunk.content[:MAX_CONTENT_LENGTH_IN_METADATA],
            "source": text_chunk.source,
            "start_index": text_chunk.start_index if text_chunk.start_index is not None else -1,
            "end_index": text_chunk.end_index if text_chunk.end_index is not None else -1,
            "chunk_type": text_chunk.chunk_type,
            "language": text_chunk.language,
            "confidence": text_chunk.confidence,
            "metadata": json.dumps(text_chunk.metadata),
            "entities": json.dumps(list(text_chunk.entities)),
            "relations": json.dumps(list(text_chunk.relations)),
            "created_at": text_chunk.created_at.isoformat(),
            "updated_at": text_chunk.updated_at.isoformat(),
        }
        return text_chunk.id, document, metadata

    def _execute_chroma_query(
        self,
        collection: Any,
        query: Union[str, List[float]],
        top_k: int,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute ChromaDB query with common parameters.

        Args:
            collection: ChromaDB collection
            query: Query string or embedding vector
            top_k: Number of results to return
            filter_dict: Filter conditions

        Returns:
            ChromaDB query result
        """
        query_builder = QueryBuilder()
        query_builder.with_query(query).with_top_k(top_k).with_filters(filter_dict).with_includes(
            ["metadatas", "distances"]
        )

        query_params = query_builder.build()

        result = collection.query(
            query_embeddings=query_params.get("query_embeddings"),
            query_texts=query_params.get("query_texts"),
            n_results=query_params["n_results"],
            where=query_params.get("where"),
            include=query_params["include"],
        )
        # Ensure the result is a proper Dict[str, Any]
        return dict(result) if result is not None else {}

    # Entity operations
    async def add_entity(self, entity: Entity, embedding: Optional[List[float]] = None) -> bool:
        """Add entity to ChromaDB."""
        try:
            collection = self._get_collection("entity")
            entity_id, document, metadata = self._prepare_entity_data(entity)

            collection.upsert(
                ids=[entity_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding is not None else None,
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to add entity {entity.id}: {e}") from e

    async def update_entity(self, entity: Entity, embedding: Optional[List[float]] = None) -> bool:
        """Update entity information with incremental changes."""
        try:
            # Get existing entity
            existing_entity = await self.get_entity(entity.id)
            if not existing_entity:
                # Entity doesn't exist, create new one
                return await self.add_entity(entity, embedding)

            # Merge updates with existing data
            updated_entity = Entity(
                id=entity.id,
                name=entity.name if entity.name else existing_entity.name,
                entity_type=(
                    entity.entity_type if entity.entity_type else existing_entity.entity_type
                ),
                description=(
                    entity.description if entity.description else existing_entity.description
                ),
                properties={**existing_entity.properties, **entity.properties},
                aliases=(
                    list(set(existing_entity.aliases + entity.aliases))
                    if entity.aliases
                    else existing_entity.aliases
                ),
                confidence=(
                    entity.confidence if entity.confidence != 1.0 else existing_entity.confidence
                ),
                source=entity.source if entity.source else existing_entity.source,
                text_chunks=existing_entity.text_chunks.union(entity.text_chunks),
                created_at=existing_entity.created_at,  # Preserve original creation time
                updated_at=datetime.now(),  # Update modification time
            )

            # Save the merged entity
            collection = self._get_collection("entity")
            entity_id, document, metadata = self._prepare_entity_data(updated_entity)

            collection.upsert(
                ids=[entity_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding is not None else None,
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to update entity {entity.id}: {e}") from e

    async def delete_entity(self, entity_id: str) -> bool:
        """Delete entity."""
        try:
            collection = self._get_collection("entity")
            collection.delete(ids=[entity_id])
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete entity {entity_id}: {e}") from e

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID."""
        try:
            collection = self._get_collection("entity")
            result = collection.get(ids=[entity_id], include=["metadatas"])

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0]
            return self._reconstruct_entity_from_metadata(entity_id, metadata)
        except Exception as e:
            raise VectorStoreError(f"Failed to get entity {entity_id}: {e}") from e

    def _reconstruct_entity_from_metadata(self, entity_id: str, metadata: Dict[str, Any]) -> Entity:
        """Reconstruct Entity object from ChromaDB metadata."""
        return Entity(
            id=entity_id,
            name=metadata.get("name", ""),
            entity_type=metadata.get("entity_type", "UNKNOWN"),
            description=metadata.get("description", ""),
            properties=json.loads(metadata.get("properties", "{}")),
            aliases=json.loads(metadata.get("aliases", "[]")),
            confidence=metadata.get("confidence", 1.0),
            source=metadata.get("source", ""),
            text_chunks=set(json.loads(metadata.get("text_chunks", "[]"))),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
        )

    async def search_entities(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Entity, float]]:
        """Search similar entities."""
        try:
            collection = self._get_collection("entity")
            result = self._execute_chroma_query(collection, query, top_k, filter_dict)

            entities_with_scores = []
            if (
                result.get("ids")
                and result["ids"][0]
                and result.get("metadatas")
                and result["metadatas"][0]
            ):
                for entity_id, metadata, distance in zip(
                    result["ids"][0],
                    result["metadatas"][0],
                    (
                        result["distances"][0]
                        if result.get("distances")
                        else [0.0] * len(result["ids"][0])
                    ),
                ):
                    entity = self._reconstruct_entity_from_metadata(entity_id, dict(metadata))
                    similarity = ResultProcessor.calculate_similarity(distance)
                    entities_with_scores.append((entity, similarity))

            return entities_with_scores
        except Exception as e:
            raise VectorStoreError(f"Failed to search entities: {e}") from e

    # Relation operations
    async def add_relation(
        self, relation: Relation, embedding: Optional[List[float]] = None
    ) -> bool:
        """Add relation to ChromaDB."""
        try:
            collection = self._get_collection("relation")
            relation_id, document, metadata = self._prepare_relation_data(relation)

            collection.upsert(
                ids=[relation_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding is not None else None,
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to add relation {relation.id}: {e}") from e

    async def update_relation(
        self, relation: Relation, embedding: Optional[List[float]] = None
    ) -> bool:
        """Update relation information with incremental changes."""
        try:
            # Get existing relation
            existing_relation = await self.get_relation(relation.id)
            if not existing_relation:
                # Relation doesn't exist, create new one
                return await self.add_relation(relation, embedding)

            # Merge updates with existing data
            updated_relation = Relation(
                id=relation.id,
                head_entity=(
                    relation.head_entity if relation.head_entity else existing_relation.head_entity
                ),
                tail_entity=(
                    relation.tail_entity if relation.tail_entity else existing_relation.tail_entity
                ),
                relation_type=(
                    relation.relation_type
                    if relation.relation_type
                    else existing_relation.relation_type
                ),
                description=(
                    relation.description if relation.description else existing_relation.description
                ),
                properties={**existing_relation.properties, **relation.properties},
                confidence=(
                    relation.confidence
                    if relation.confidence != 0.8
                    else existing_relation.confidence
                ),
                source=relation.source if relation.source else existing_relation.source,
                text_chunks=existing_relation.text_chunks.union(relation.text_chunks),
                created_at=existing_relation.created_at,  # Preserve original creation time
                updated_at=datetime.now(),  # Update modification time
            )

            # Save the merged relation
            collection = self._get_collection("relation")
            relation_id, document, metadata = self._prepare_relation_data(updated_relation)

            collection.upsert(
                ids=[relation_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding is not None else None,
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to update relation {relation.id}: {e}") from e

    async def delete_relation(self, relation_id: str) -> bool:
        """Delete relation."""
        try:
            collection = self._get_collection("relation")
            collection.delete(ids=[relation_id])
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete relation {relation_id}: {e}") from e

    async def get_relation(self, relation_id: str) -> Optional[Relation]:
        """Get relation by ID."""
        try:
            collection = self._get_collection("relation")
            result = collection.get(ids=[relation_id], include=["metadatas"])

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0]
            return await self._reconstruct_relation_from_metadata(relation_id, metadata)
        except Exception as e:
            raise VectorStoreError(f"Failed to get relation {relation_id}: {e}") from e

    async def _reconstruct_relation_from_metadata(
        self, relation_id: str, metadata: Dict[str, Any]
    ) -> Relation:
        """Reconstruct Relation object from ChromaDB metadata."""
        # Fetch entity references
        head_entity = None
        tail_entity = None

        head_entity_id = metadata.get("head_entity_id")
        tail_entity_id = metadata.get("tail_entity_id")

        if head_entity_id:
            head_entity = await self.get_entity(head_entity_id)

        if tail_entity_id:
            tail_entity = await self.get_entity(tail_entity_id)

        return Relation(
            id=relation_id,
            head_entity=head_entity,
            tail_entity=tail_entity,
            relation_type=metadata.get("relation_type", "RELATED_TO"),
            description=metadata.get("description", ""),
            properties=json.loads(metadata.get("properties", "{}")),
            confidence=metadata.get("confidence", 0.8),
            source=metadata.get("source", ""),
            text_chunks=set(json.loads(metadata.get("text_chunks", "[]"))),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
        )

    async def search_relations(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Relation, float]]:
        """Search similar relations."""
        # pylint: disable=too-many-locals
        try:
            collection = self._get_collection("relation")

            query_params: Dict[str, Any] = {
                "n_results": top_k,
                "include": ["metadatas", "distances"],
            }

            if isinstance(query, str):
                query_params["query_texts"] = [query]
            else:
                query_params["query_embeddings"] = [query]

            if filter_dict:
                where_clause = {}
                for key, value in filter_dict.items():
                    where_clause[key] = {"$eq": value}
                query_params["where"] = where_clause

            result = collection.query(
                query_embeddings=query_params.get("query_embeddings"),
                query_texts=query_params.get("query_texts"),
                n_results=query_params["n_results"],
                where=query_params.get("where"),
                include=query_params["include"],
            )

            relations_with_scores = []
            if (
                result.get("ids")
                and result["ids"][0]
                and result.get("metadatas")
                and result["metadatas"][0]
            ):
                for relation_id, metadata, distance in zip(
                    result["ids"][0],
                    result["metadatas"][0],
                    (
                        result["distances"][0]
                        if result.get("distances")
                        else [0.0] * len(result["ids"][0])
                    ),
                ):
                    relation = await self._reconstruct_relation_from_metadata(
                        relation_id, dict(metadata)
                    )
                    similarity = max(0.0, 1.0 - distance)
                    relations_with_scores.append((relation, similarity))

            return relations_with_scores
        except Exception as e:
            raise VectorStoreError(f"Failed to search relations: {e}") from e

    # Cluster operations
    async def add_cluster(self, cluster: Cluster, embedding: Optional[List[float]] = None) -> bool:
        """Add cluster to ChromaDB."""
        try:
            collection = self._get_collection("cluster")
            cluster_id, document, metadata = self._prepare_cluster_data(cluster)

            collection.upsert(
                ids=[cluster_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding is not None else None,
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to add cluster {cluster.id}: {e}") from e

    async def update_cluster(
        self, cluster: Cluster, embedding: Optional[List[float]] = None
    ) -> bool:
        """Update cluster information with incremental changes."""
        try:
            # Get existing cluster
            existing_cluster = await self.get_cluster(cluster.id)
            if not existing_cluster:
                # Cluster doesn't exist, create new one
                return await self.add_cluster(cluster, embedding)

            # Merge updates with existing data
            updated_cluster = Cluster(
                id=cluster.id,
                name=cluster.name if cluster.name else existing_cluster.name,
                cluster_type=(
                    cluster.cluster_type if cluster.cluster_type else existing_cluster.cluster_type
                ),
                description=(
                    cluster.description if cluster.description else existing_cluster.description
                ),
                entities=existing_cluster.entities.union(cluster.entities),
                relations=existing_cluster.relations.union(cluster.relations),
                centroid_entity_id=(
                    cluster.centroid_entity_id
                    if cluster.centroid_entity_id
                    else existing_cluster.centroid_entity_id
                ),
                parent_cluster_id=(
                    cluster.parent_cluster_id
                    if cluster.parent_cluster_id
                    else existing_cluster.parent_cluster_id
                ),
                child_clusters=existing_cluster.child_clusters.union(cluster.child_clusters),
                cohesion_score=(
                    cluster.cohesion_score
                    if cluster.cohesion_score != 0.0
                    else existing_cluster.cohesion_score
                ),
                properties={**existing_cluster.properties, **cluster.properties},
                confidence=(
                    cluster.confidence if cluster.confidence != 0.8 else existing_cluster.confidence
                ),
                source=cluster.source if cluster.source else existing_cluster.source,
                text_chunks=existing_cluster.text_chunks.union(cluster.text_chunks),
                created_at=existing_cluster.created_at,  # Preserve original creation time
                updated_at=datetime.now(),  # Update modification time
            )

            # Save the merged cluster
            collection = self._get_collection("cluster")
            cluster_id, document, metadata = self._prepare_cluster_data(updated_cluster)

            collection.upsert(
                ids=[cluster_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding is not None else None,
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to update cluster {cluster.id}: {e}") from e

    async def delete_cluster(self, cluster_id: str) -> bool:
        """Delete cluster."""
        try:
            collection = self._get_collection("cluster")
            collection.delete(ids=[cluster_id])
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete cluster {cluster_id}: {e}") from e

    async def get_cluster(self, cluster_id: str) -> Optional[Cluster]:
        """Get cluster by ID."""
        try:
            collection = self._get_collection("cluster")
            result = collection.get(ids=[cluster_id], include=["metadatas"])

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0]
            return self._reconstruct_cluster_from_metadata(cluster_id, metadata)
        except Exception as e:
            raise VectorStoreError(f"Failed to get cluster {cluster_id}: {e}") from e

    def _reconstruct_cluster_from_metadata(
        self, cluster_id: str, metadata: Dict[str, Any]
    ) -> Cluster:
        """Reconstruct Cluster object from ChromaDB metadata."""
        return Cluster(
            id=cluster_id,
            name=metadata.get("name", ""),
            cluster_type=metadata.get("cluster_type", "OTHER"),
            description=metadata.get("description", ""),
            entities=set(json.loads(metadata.get("entities", "[]"))),
            relations=set(json.loads(metadata.get("relations", "[]"))),
            centroid_entity_id=metadata.get("centroid_entity_id", ""),
            parent_cluster_id=metadata.get("parent_cluster_id", ""),
            child_clusters=set(json.loads(metadata.get("child_clusters", "[]"))),
            cohesion_score=metadata.get("cohesion_score", 0.0),
            properties=json.loads(metadata.get("properties", "{}")),
            confidence=metadata.get("confidence", 0.8),
            source=metadata.get("source", ""),
            text_chunks=set(json.loads(metadata.get("text_chunks", "[]"))),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
        )

    async def search_clusters(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Cluster, float]]:
        """Search similar clusters."""
        # pylint: disable=too-many-locals
        try:
            collection = self._get_collection("cluster")

            query_params: Dict[str, Any] = {
                "n_results": top_k,
                "include": ["metadatas", "distances"],
            }

            if isinstance(query, str):
                query_params["query_texts"] = [query]
            else:
                query_params["query_embeddings"] = [query]

            if filter_dict:
                where_clause = {}
                for key, value in filter_dict.items():
                    where_clause[key] = {"$eq": value}
                query_params["where"] = where_clause

            result = collection.query(
                query_embeddings=query_params.get("query_embeddings"),
                query_texts=query_params.get("query_texts"),
                n_results=query_params["n_results"],
                where=query_params.get("where"),
                include=query_params["include"],
            )

            clusters_with_scores = []
            if (
                result.get("ids")
                and result["ids"][0]
                and result.get("metadatas")
                and result["metadatas"][0]
            ):
                for cluster_id, metadata, distance in zip(
                    result["ids"][0],
                    result["metadatas"][0],
                    (
                        result["distances"][0]
                        if result.get("distances")
                        else [0.0] * len(result["ids"][0])
                    ),
                ):
                    cluster = self._reconstruct_cluster_from_metadata(cluster_id, dict(metadata))
                    similarity = max(0.0, 1.0 - distance)
                    clusters_with_scores.append((cluster, similarity))

            return clusters_with_scores
        except Exception as e:
            raise VectorStoreError(f"Failed to search clusters: {e}") from e

    # TextChunk operations
    async def add_text_chunk(
        self, text_chunk: TextChunk, embedding: Optional[List[float]] = None
    ) -> bool:
        """Add text chunk to ChromaDB."""
        try:
            collection = self._get_collection("text_chunk")
            chunk_id, document, metadata = self._prepare_text_chunk_data(text_chunk)

            collection.upsert(
                ids=[chunk_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding is not None else None,
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to add text_chunk {text_chunk.id}: {e}") from e

    async def update_text_chunk(
        self, text_chunk: TextChunk, embedding: Optional[List[float]] = None
    ) -> bool:
        """Update text chunk information with incremental changes."""
        try:
            # Get existing text chunk
            existing_chunk = await self.get_text_chunk(text_chunk.id)
            if not existing_chunk:
                # Text chunk doesn't exist, create new one
                return await self.add_text_chunk(text_chunk, embedding)

            # Merge updates with existing data
            updated_chunk = TextChunk(
                id=text_chunk.id,
                content=text_chunk.content if text_chunk.content else existing_chunk.content,
                title=text_chunk.title if text_chunk.title else existing_chunk.title,
                source=text_chunk.source if text_chunk.source else existing_chunk.source,
                start_index=(
                    text_chunk.start_index
                    if text_chunk.start_index is not None
                    else existing_chunk.start_index
                ),
                end_index=(
                    text_chunk.end_index
                    if text_chunk.end_index is not None
                    else existing_chunk.end_index
                ),
                chunk_type=(
                    text_chunk.chunk_type if text_chunk.chunk_type else existing_chunk.chunk_type
                ),
                language=text_chunk.language if text_chunk.language else existing_chunk.language,
                confidence=(
                    text_chunk.confidence
                    if text_chunk.confidence != 1.0
                    else existing_chunk.confidence
                ),
                metadata={**existing_chunk.metadata, **text_chunk.metadata},
                entities=existing_chunk.entities.union(text_chunk.entities),
                relations=existing_chunk.relations.union(text_chunk.relations),
                created_at=existing_chunk.created_at,  # Preserve original creation time
                updated_at=datetime.now(),  # Update modification time
            )

            # Save the merged text chunk
            collection = self._get_collection("text_chunk")
            chunk_id, document, metadata = self._prepare_text_chunk_data(updated_chunk)

            collection.upsert(
                ids=[chunk_id],
                documents=[document],
                metadatas=[metadata],
                embeddings=[embedding] if embedding is not None else None,
            )
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to update text_chunk {text_chunk.id}: {e}") from e

    async def delete_text_chunk(self, chunk_id: str) -> bool:
        """Delete text chunk."""
        try:
            collection = self._get_collection("text_chunk")
            collection.delete(ids=[chunk_id])
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete text_chunk {chunk_id}: {e}") from e

    async def get_text_chunk(self, chunk_id: str) -> Optional[TextChunk]:
        """Get text chunk by ID."""
        try:
            collection = self._get_collection("text_chunk")
            result = collection.get(ids=[chunk_id], include=["metadatas"])

            if not result["ids"]:
                return None

            metadata = result["metadatas"][0]
            return self._reconstruct_text_chunk_from_metadata(chunk_id, metadata)
        except Exception as e:
            raise VectorStoreError(f"Failed to get text_chunk {chunk_id}: {e}") from e

    def _reconstruct_text_chunk_from_metadata(
        self, chunk_id: str, metadata: Dict[str, Any]
    ) -> TextChunk:
        """Reconstruct TextChunk object from ChromaDB metadata."""
        # Convert -1 back to None for start_index and end_index
        start_index = metadata.get("start_index")
        if start_index == -1:
            start_index = None
        end_index = metadata.get("end_index")
        if end_index == -1:
            end_index = None

        return TextChunk(
            id=chunk_id,
            content=metadata.get("content", ""),
            title=metadata.get("title", ""),
            source=metadata.get("source", ""),
            start_index=start_index,
            end_index=end_index,
            chunk_type=metadata.get("chunk_type", "par."),
            language=metadata.get("language", "zh"),
            confidence=metadata.get("confidence", 1.0),
            metadata=json.loads(metadata.get("metadata", "{}")),
            entities=set(json.loads(metadata.get("entities", "[]"))),
            relations=set(json.loads(metadata.get("relations", "[]"))),
            created_at=datetime.fromisoformat(metadata["created_at"]),
            updated_at=datetime.fromisoformat(metadata["updated_at"]),
        )

    async def search_text_chunks(
        self,
        query: Union[str, List[float]],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[TextChunk, float]]:
        """Search similar text chunks."""
        # pylint: disable=too-many-locals
        try:
            collection = self._get_collection("text_chunk")

            query_params: Dict[str, Any] = {
                "n_results": top_k,
                "include": ["metadatas", "distances"],
            }

            if isinstance(query, str):
                query_params["query_texts"] = [query]
            else:
                query_params["query_embeddings"] = [query]

            if filter_dict:
                where_clause = {}
                for key, value in filter_dict.items():
                    where_clause[key] = {"$eq": value}
                query_params["where"] = where_clause

            result = collection.query(
                query_embeddings=query_params.get("query_embeddings"),
                query_texts=query_params.get("query_texts"),
                n_results=query_params["n_results"],
                where=query_params.get("where"),
                include=query_params["include"],
            )

            chunks_with_scores = []
            if (
                result.get("ids")
                and result["ids"][0]
                and result.get("metadatas")
                and result["metadatas"][0]
            ):
                for chunk_id, metadata, distance in zip(
                    result["ids"][0],
                    result["metadatas"][0],
                    (
                        result["distances"][0]
                        if result.get("distances")
                        else [0.0] * len(result["ids"][0])
                    ),
                ):
                    chunk = self._reconstruct_text_chunk_from_metadata(chunk_id, dict(metadata))
                    similarity = max(0.0, 1.0 - distance)
                    chunks_with_scores.append((chunk, similarity))

            return chunks_with_scores
        except Exception as e:
            raise VectorStoreError(f"Failed to search text_chunks: {e}") from e

    # Batch operations
    async def batch_add_entities(
        self,
        entities: List[Entity],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[bool]:
        """Batch add entities."""
        # pylint: disable=too-many-locals
        if batch_size is None:
            batch_size = get_settings().embedding.batch_size
        try:
            collection = self._get_collection("entity")

            ids, documents, metadatas, statuses = [], [], [], []
            for entity in entities:
                entity_id, document, metadata = self._prepare_entity_data(entity)
                ids.append(entity_id)
                documents.append(document)
                metadatas.append(metadata)

            for i in range(0, len(entities), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_documents = documents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=embeddings[i : i + batch_size] if embeddings else None,
                )
                statuses.extend([True] * len(batch_ids))
            return statuses
        except Exception as e:
            raise VectorStoreError(f"Failed to batch add entities: {e}") from e

    async def batch_add_relations(
        self,
        relations: List[Relation],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[bool]:
        """Batch add relations."""
        # pylint: disable=too-many-locals
        if batch_size is None:
            batch_size = get_settings().embedding.batch_size
        try:
            collection = self._get_collection("relation")

            ids, documents, metadatas, statuses = [], [], [], []
            for relation in relations:
                relation_id, document, metadata = self._prepare_relation_data(relation)
                ids.append(relation_id)
                documents.append(document)
                metadatas.append(metadata)

            for i in range(0, len(relations), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_documents = documents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=embeddings[i : i + batch_size] if embeddings else None,
                )
                statuses.extend([True] * len(batch_ids))
            return statuses

        except Exception as e:
            raise VectorStoreError(f"Failed to batch add relations: {e}") from e

    async def batch_add_clusters(
        self,
        clusters: List[Cluster],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[bool]:
        """Batch add clusters."""
        # pylint: disable=too-many-locals
        if batch_size is None:
            batch_size = get_settings().embedding.batch_size
        try:
            collection = self._get_collection("cluster")

            ids, documents, metadatas, statuses = [], [], [], []
            for cluster in clusters:
                cluster_id, document, metadata = self._prepare_cluster_data(cluster)
                ids.append(cluster_id)
                documents.append(document)
                metadatas.append(metadata)

            for i in range(0, len(clusters), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_documents = documents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=embeddings[i : i + batch_size] if embeddings else None,
                )
                statuses.extend([True] * len(batch_ids))
            return statuses
        except Exception as e:
            raise VectorStoreError(f"Failed to batch add clusters: {e}") from e

    async def batch_add_text_chunks(
        self,
        text_chunks: List[TextChunk],
        embeddings: Optional[List[List[float]]] = None,
        batch_size: Optional[int] = None,
    ) -> List[bool]:
        """Batch add text chunks."""
        # pylint: disable=too-many-locals
        if batch_size is None:
            batch_size = get_settings().embedding.batch_size
        try:
            collection = self._get_collection("text_chunk")

            ids, documents, metadatas, statuses = [], [], [], []
            for chunk in text_chunks:
                chunk_id, document, metadata = self._prepare_text_chunk_data(chunk)
                ids.append(chunk_id)
                documents.append(document)
                metadatas.append(metadata)

            for i in range(0, len(text_chunks), batch_size):
                batch_ids = ids[i : i + batch_size]
                batch_documents = documents[i : i + batch_size]
                batch_metadatas = metadatas[i : i + batch_size]
                collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    metadatas=batch_metadatas,
                    embeddings=embeddings[i : i + batch_size] if embeddings else None,
                )
                statuses.extend([True] * len(batch_ids))
            return statuses
        except Exception as e:
            raise VectorStoreError(f"Failed to batch add text_chunks: {e}") from e

    # Utility methods
    async def get_stats(self) -> Dict[str, int]:
        """Get vector store statistics."""
        stats = {}

        for data_type, collection in self._collections.items():
            count = collection.count()
            stats[f"{data_type}s"] = count  # Add 's' to make it plural for consistency

        return stats

    async def clear_all(self) -> bool:
        """Clear all data."""
        try:
            for collection in self._collections.values():
                # ChromaDB doesn't have a direct clear method, need to delete all documents
                result = collection.get(include=[])  # Only get IDs
                if result["ids"]:
                    collection.delete(ids=result["ids"])
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to clear all data: {e}") from e
