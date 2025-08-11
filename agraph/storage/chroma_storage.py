"""
ChromaDB-based graph storage implementation.

Provides ChromaDB vector database storage for knowledge graphs with support
for storing and retrieving node and edge vectors.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np

try:
    import chromadb
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection
    from chromadb.config import Settings
    from chromadb.errors import ChromaError

    CHROMADB_AVAILABLE = True
except ImportError:
    chromadb = None  # type: ignore
    Settings = type(None)  # type: ignore
    ChromaError = Exception  # type: ignore
    ClientAPI = type(None)  # type: ignore
    Collection = type(None)  # type: ignore
    CHROMADB_AVAILABLE = False

from ..config import settings
from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation
from ..text import TextChunk
from .interfaces import (
    GraphConnection,
    GraphCRUD,
    GraphEntityCRUD,
    GraphQuery,
    GraphRelationCRUD,
    GraphTextChunkCRUD,
    GraphTextChunkEmbedding,
    GraphTextChunkQuery,
    VectorStorage,
)


class ChromaDBGraphStorage(
    VectorStorage,
    GraphConnection,
    GraphCRUD,
    GraphEntityCRUD,
    GraphRelationCRUD,
    GraphTextChunkCRUD,
    GraphTextChunkQuery,
    GraphTextChunkEmbedding,
    GraphQuery,
):
    """ChromaDB-based graph storage implementation with comprehensive graph operations."""

    def __init__(self, persist_directory: Optional[str] = None, collection_prefix: str = "agraph"):
        """
        Initialize ChromaDB storage.

        Args:
            persist_directory: Directory to persist ChromaDB data. If None, uses workdir from settings.
            collection_prefix: Prefix for ChromaDB collections.
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB is required but not installed. Install with: pip install chromadb")

        self._is_connected = False

        if persist_directory is None:
            persist_directory = os.path.join(settings.workdir, "chroma_db")

        self.persist_directory = persist_directory
        self.collection_prefix = collection_prefix
        self.client: Optional[ClientAPI] = None
        self.entity_collection: Optional[Collection] = None
        self.relation_collection: Optional[Collection] = None
        self.graph_metadata_collection: Optional[Collection] = None
        self.text_collection: Optional[Collection] = None

        self._ensure_persist_dir()
        self.connect()

    def _ensure_persist_dir(self) -> None:
        """Ensure persist directory exists."""
        if not os.path.exists(self.persist_directory):
            os.makedirs(self.persist_directory, exist_ok=True)
            logger.info(f"Created ChromaDB persist directory: {self.persist_directory}", self.persist_directory)

    # GraphConnection interface

    def connect(self) -> bool:
        """Connect to ChromaDB."""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB is not available")
            return False

        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory, settings=Settings(anonymized_telemetry=False, allow_reset=True)
            )

            # Initialize collections
            self.entity_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_entities", metadata={"description": "Entity vectors and metadata"}
            )

            self.relation_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_relations", metadata={"description": "Relation vectors and metadata"}
            )

            self.graph_metadata_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_graphs", metadata={"description": "Graph metadata"}
            )

            self.text_collection = self.client.get_or_create_collection(
                name=f"{self.collection_prefix}_texts", metadata={"description": "Original text embeddings"}
            )

            self._is_connected = True
            logger.info(f"Connected to ChromaDB at {self.persist_directory}")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        if self.client:
            # ChromaDB client doesn't require explicit disconnect
            self.client = None
            self.entity_collection = None
            self.relation_collection = None
            self.graph_metadata_collection = None
            self.text_collection = None

        self._is_connected = False
        logger.info("Disconnected from ChromaDB")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._is_connected and self.client is not None

    # GraphCRUD interface

    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """Save knowledge graph to ChromaDB."""
        if not self.is_connected():
            if not self.connect():
                return False

        try:
            # Save graph metadata
            graph_metadata = {
                "id": graph.id,
                "name": graph.name,
                "created_at": graph.created_at.isoformat(),
                "updated_at": graph.updated_at.isoformat(),
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
            }

            if self.graph_metadata_collection is None:
                logger.error("Graph metadata collection is not initialized")
                return False

            # Type-safe metadata conversion for ChromaDB
            safe_graph_metadata = {
                "id": str(graph.id),
                "name": str(graph.name),
                "created_at": str(graph.created_at.isoformat()),
                "updated_at": str(graph.updated_at.isoformat()),
                "entity_count": int(len(graph.entities)),
                "relation_count": int(len(graph.relations)),
            }

            # Type-safe metadatas for ChromaDB
            metadatas_list = [safe_graph_metadata]

            self.graph_metadata_collection.upsert(
                ids=[graph.id], metadatas=metadatas_list, documents=[json.dumps(graph_metadata)]
            )

            # Save entities with vectors
            entity_ids = []
            entity_metadatas = []
            entity_documents = []
            entity_embeddings = []

            for entity in graph.entities.values():
                entity_ids.append(f"{graph.id}_{entity.id}")

                # Type-safe metadata conversion for ChromaDB
                safe_entity_metadata = {
                    "graph_id": str(graph.id),
                    "entity_id": str(entity.id),
                    "name": str(entity.name),
                    "entity_type": str(entity.entity_type),
                    "confidence": float(entity.confidence),
                    "aliases": str(json.dumps(entity.aliases)),
                    "properties": str(json.dumps(entity.properties)),
                }
                entity_metadatas.append(safe_entity_metadata)

                # Use entity description or name as document
                entity_documents.append(entity.description or entity.name)

                # Use embedding if available, otherwise create a dummy embedding
                if hasattr(entity, "embedding") and getattr(entity, "embedding", None) is not None:
                    entity_embeddings.append(np.array(getattr(entity, "embedding")).tolist())
                else:
                    # Create a dummy embedding vector (384 dimensions for default embedding)
                    dummy_embedding = np.random.random(384).tolist()
                    entity_embeddings.append(dummy_embedding)

            if entity_ids and self.entity_collection is not None:
                # Type-safe metadatas
                self.entity_collection.upsert(
                    ids=entity_ids,
                    metadatas=entity_metadatas,
                    documents=entity_documents,
                    embeddings=entity_embeddings,
                )

            # Save relations with vectors
            relation_ids = []
            relation_metadatas = []
            relation_documents = []
            relation_embeddings = []

            for relation in graph.relations.values():
                relation_ids.append(f"{graph.id}_{relation.id}")

                # Type-safe metadata conversion for ChromaDB
                safe_relation_metadata = {
                    "graph_id": str(graph.id),
                    "relation_id": str(relation.id),
                    "relation_type": str(relation.relation_type),
                    "head_entity_id": str(relation.head_entity.id) if relation.head_entity else None,
                    "tail_entity_id": str(relation.tail_entity.id) if relation.tail_entity else None,
                    "confidence": float(relation.confidence),
                    "properties": str(json.dumps(relation.properties)),
                }
                relation_metadatas.append(safe_relation_metadata)

                # Use relation description or type as document
                relation_documents.append(relation.description or str(relation.relation_type))

                # Use embedding if available, otherwise create a dummy embedding
                if hasattr(relation, "embedding") and getattr(relation, "embedding", None) is not None:
                    relation_embeddings.append(np.array(getattr(relation, "embedding")).tolist())
                else:
                    # Create a dummy embedding vector (384 dimensions for default embedding)
                    dummy_embedding = np.random.random(384).tolist()
                    relation_embeddings.append(dummy_embedding)

            if relation_ids and self.relation_collection is not None:
                # Type-safe metadatas and embeddings
                self.relation_collection.upsert(
                    ids=relation_ids,
                    metadatas=relation_metadatas,
                    documents=relation_documents,
                    embeddings=relation_embeddings,
                )

            logger.info(f"Graph {graph.id} saved to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error saving graph to ChromaDB: {e}")
            return False

    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load knowledge graph from ChromaDB."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return None

        try:
            if self.graph_metadata_collection is None:
                logger.error("Graph metadata collection is not initialized")
                return None

            # Load graph metadata
            graph_results = self.graph_metadata_collection.get(ids=[graph_id], include=["metadatas"])

            if not graph_results["ids"]:
                logger.warning(f"Graph {graph_id} not found")
                return None

            # Type-safe access to metadata
            metadatas = graph_results.get("metadatas")
            if not metadatas or len(metadatas) == 0:
                logger.error("Graph metadata is missing")
                return None

            graph_metadata = metadatas[0]
            graph_name = graph_metadata.get("name")
            if not isinstance(graph_name, str):
                logger.error("Invalid graph name in metadata")
                return None

            # Create knowledge graph
            graph = KnowledgeGraph(id=graph_id, name=graph_name)

            # Parse timestamps with type safety
            created_at = graph_metadata.get("created_at")
            if created_at and isinstance(created_at, str):
                try:
                    graph.created_at = datetime.fromisoformat(created_at)
                except ValueError:
                    logger.warning(f"Invalid created_at format: {created_at}")

            updated_at = graph_metadata.get("updated_at")
            if updated_at and isinstance(updated_at, str):
                try:
                    graph.updated_at = datetime.fromisoformat(updated_at)
                except ValueError:
                    logger.warning(f"Invalid updated_at format: {updated_at}")

            # Load entities
            if self.entity_collection is None:
                logger.error("Entity collection is not initialized")
                return graph

            entity_results = self.entity_collection.get(
                where={"graph_id": graph_id}, include=["metadatas", "embeddings"]
            )

            # Type-safe access to entity results
            entity_metadatas = entity_results.get("metadatas")
            if entity_metadatas is not None:
                for i, entity_id in enumerate(entity_results["ids"]):
                    if i >= len(entity_metadatas):
                        continue

                    metadata = entity_metadatas[i]
                    try:
                        embedding = (
                            entity_results["embeddings"][i]
                            if "embeddings" in entity_results
                            and entity_results["embeddings"] is not None
                            and i < len(entity_results["embeddings"])
                            else None
                        )
                    except (IndexError, TypeError):
                        embedding = None

                    # Type-safe metadata access
                    entity_id_val = metadata.get("entity_id")
                    entity_name = metadata.get("name")
                    entity_type = metadata.get("entity_type")
                    confidence = metadata.get("confidence", 1.0)

                    if (
                        not isinstance(entity_id_val, str)
                        or not isinstance(entity_name, str)
                        or not isinstance(entity_type, str)
                    ):
                        logger.warning(f"Invalid entity metadata at index {i}")
                        continue

                    if not isinstance(confidence, (int, float)):
                        confidence = 1.0

                    entity = Entity(
                        id=entity_id_val,
                        name=entity_name,
                        entity_type=entity_type,
                        confidence=float(confidence),
                        aliases=json.loads(str(metadata.get("aliases", "[]"))),
                        properties=json.loads(str(metadata.get("properties", "{}"))),
                    )

                    # Set embedding if available (Note: Entity class doesn't have embedding attribute)
                    # if embedding is not None and len(embedding) > 0:
                    #     entity.embedding = embedding

                    graph.add_entity(entity)

            # Load relations
            if self.relation_collection is None:
                logger.error("Relation collection is not initialized")
                return graph

            relation_results = self.relation_collection.get(
                where={"graph_id": graph_id}, include=["metadatas", "embeddings"]
            )

            # Type-safe access to relation results
            relation_metadatas = relation_results.get("metadatas")
            if relation_metadatas is not None:
                for i, relation_id in enumerate(relation_results["ids"]):
                    if i >= len(relation_metadatas):
                        continue

                    metadata = relation_metadatas[i]
                    try:
                        embedding = (
                            relation_results["embeddings"][i]
                            if "embeddings" in relation_results
                            and relation_results["embeddings"] is not None
                            and i < len(relation_results["embeddings"])
                            else None
                        )
                    except (IndexError, TypeError):
                        embedding = None

                    # Type-safe metadata access
                    relation_id_val = metadata.get("relation_id")
                    relation_type = metadata.get("relation_type")
                    confidence = metadata.get("confidence", 1.0)

                    if not isinstance(relation_id_val, str) or not isinstance(relation_type, str):
                        logger.warning(f"Invalid relation metadata at index {i}")
                        continue

                    if not isinstance(confidence, (int, float)):
                        confidence = 1.0

                    head_entity_id = metadata.get("head_entity_id")
                    tail_entity_id = metadata.get("tail_entity_id")
                    head_entity = graph.entities.get(head_entity_id) if isinstance(head_entity_id, str) else None
                    tail_entity = graph.entities.get(tail_entity_id) if isinstance(tail_entity_id, str) else None

                    relation = Relation(
                        id=relation_id_val,
                        head_entity=head_entity,
                        tail_entity=tail_entity,
                        relation_type=relation_type,
                        confidence=float(confidence),
                        properties=json.loads(str(metadata.get("properties", "{}"))),
                    )

                    # Set embedding if available (Note: Relation class doesn't have embedding attribute)
                    if embedding is not None and len(embedding) > 0:
                        relation.embedding = embedding

                    graph.add_relation(relation)

            logger.info(f"Graph {graph_id} loaded from ChromaDB")
            return graph

        except Exception as e:
            import traceback

            logger.error(f"Error loading graph from ChromaDB: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def delete_graph(self, graph_id: str) -> bool:
        """Delete knowledge graph from ChromaDB."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return False

        try:
            # Delete entities
            if self.entity_collection is not None:
                entity_results = self.entity_collection.get(where={"graph_id": graph_id})
                if entity_results["ids"]:
                    self.entity_collection.delete(ids=entity_results["ids"])

            # Delete relations
            if self.relation_collection is not None:
                relation_results = self.relation_collection.get(where={"graph_id": graph_id})
                if relation_results["ids"]:
                    self.relation_collection.delete(ids=relation_results["ids"])

            # Delete graph metadata
            if self.graph_metadata_collection is not None:
                self.graph_metadata_collection.delete(ids=[graph_id])

            logger.info(f"Graph {graph_id} deleted from ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error deleting graph from ChromaDB: {e}")
            return False

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all available graphs."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            if self.graph_metadata_collection is None:
                logger.error("Graph metadata collection is not initialized")
                return []

            results = self.graph_metadata_collection.get(include=["metadatas"])

            graph_list = []
            metadatas = results.get("metadatas")
            if metadatas is not None:
                for metadata in metadatas:
                    # Type-safe metadata access
                    graph_id = metadata.get("id")
                    name = metadata.get("name")
                    created_at = metadata.get("created_at")
                    updated_at = metadata.get("updated_at")
                    entity_count = metadata.get("entity_count", 0)
                    relation_count = metadata.get("relation_count", 0)

                    if not isinstance(graph_id, str) or not isinstance(name, str):
                        continue

                    graph_info = {
                        "id": graph_id,
                        "name": name,
                        "created_at": created_at if isinstance(created_at, str) else None,
                        "updated_at": updated_at if isinstance(updated_at, str) else None,
                        "entity_count": entity_count if isinstance(entity_count, int) else 0,
                        "relation_count": relation_count if isinstance(relation_count, int) else 0,
                    }
                    graph_list.append(graph_info)

            # Sort by updated_at with type safety
            def sort_key(x: Dict[str, Any]) -> str:
                updated_at = x.get("updated_at", "")
                return updated_at if isinstance(updated_at, str) else ""

            graph_list.sort(key=sort_key, reverse=True)
            return graph_list

        except Exception as e:
            logger.error(f"Error listing graphs from ChromaDB: {e}")
            return []

    # GraphQuery interface

    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """Query entities based on specified conditions."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            if self.entity_collection is None:
                logger.error("Entity collection is not initialized")
                return []

            where_clause = {}

            # Build where clause from conditions - ChromaDB only supports one condition at a time
            if "graph_id" in conditions:
                where_clause["graph_id"] = conditions["graph_id"]

            # Get results
            results = self.entity_collection.get(where=where_clause, include=["metadatas", "embeddings"])

            entities = []
            metadatas = results.get("metadatas")
            if metadatas is not None:
                for i, entity_id in enumerate(results["ids"]):
                    if i >= len(metadatas):
                        continue

                    metadata = metadatas[i]
                    try:
                        embedding = (
                            results["embeddings"][i]
                            if "embeddings" in results
                            and results["embeddings"] is not None
                            and i < len(results["embeddings"])
                            else None
                        )
                    except (IndexError, TypeError):
                        embedding = None

                    # Apply additional filters after retrieval
                    if "entity_type" in conditions:
                        if metadata.get("entity_type") != conditions["entity_type"]:
                            continue

                    if "name" in conditions:
                        name_filter = conditions["name"].lower()
                        entity_name = metadata.get("name")
                        if not isinstance(entity_name, str) or name_filter not in entity_name.lower():
                            continue

                    if "min_confidence" in conditions:
                        confidence = metadata.get("confidence", 1.0)
                        if not isinstance(confidence, (int, float)) or confidence < conditions["min_confidence"]:
                            continue

                    # Type-safe metadata access
                    entity_id_val = metadata.get("entity_id")
                    entity_name = metadata.get("name")
                    entity_type = metadata.get("entity_type")
                    confidence = metadata.get("confidence", 1.0)

                    if (
                        not isinstance(entity_id_val, str)
                        or not isinstance(entity_name, str)
                        or not isinstance(entity_type, str)
                    ):
                        continue

                    if not isinstance(confidence, (int, float)):
                        confidence = 1.0

                    entity = Entity(
                        id=entity_id_val,
                        name=entity_name,
                        entity_type=entity_type,
                        confidence=float(confidence),
                        aliases=json.loads(str(metadata.get("aliases", "[]"))),
                        properties=json.loads(str(metadata.get("properties", "{}"))),
                    )

                    # Note: Entity class doesn't have embedding attribute
                    # if embedding is not None and len(embedding) > 0:
                    #     entity.embedding = embedding

                    entities.append(entity)

            # Apply limit
            limit = conditions.get("limit", 100)
            return entities[:limit]

        except Exception as e:
            logger.error(f"Error querying entities from ChromaDB: {e}")
            return []

    def query_relations(
        self,
        head_entity: Optional[str] = None,
        tail_entity: Optional[str] = None,
        relation_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Relation]:
        """Query relations based on specified conditions."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            if self.relation_collection is None:
                logger.error("Relation collection is not initialized")
                return []

            where_clause = {}

            # Build where clause
            if "graph_id" in kwargs:
                where_clause["graph_id"] = kwargs["graph_id"]

            if head_entity:
                where_clause["head_entity_id"] = head_entity

            if tail_entity:
                where_clause["tail_entity_id"] = tail_entity

            if relation_type:
                where_clause["relation_type"] = str(relation_type)

            # Get results
            results = self.relation_collection.get(where=where_clause, include=["metadatas", "embeddings"])

            relations = []
            metadatas = results.get("metadatas")
            if metadatas is not None:
                for i, relation_id in enumerate(results["ids"]):
                    if i >= len(metadatas):
                        continue

                    metadata = metadatas[i]
                    embeddings = results.get("embeddings")
                    embedding = embeddings[i] if embeddings and len(embeddings) > i else None

                    # Type-safe metadata access
                    relation_id_val = metadata.get("relation_id")
                    relation_type = metadata.get("relation_type")
                    confidence = metadata.get("confidence", 1.0)

                    if not isinstance(relation_id_val, str) or not isinstance(relation_type, str):
                        continue

                    if not isinstance(confidence, (int, float)):
                        confidence = 1.0

                    # Note: We don't have access to entity objects here
                    # This is a limitation - in a real implementation, you might want to
                    # load the entities separately or include them in the metadata
                    relation = Relation(
                        id=relation_id_val,
                        head_entity=None,  # Would need to be loaded separately
                        tail_entity=None,  # Would need to be loaded separately
                        relation_type=relation_type,
                        confidence=float(confidence),
                        properties=json.loads(str(metadata.get("properties", "{}"))),
                    )

                    # Note: Relation class doesn't have embedding attribute
                    # if embedding is not None and len(embedding) > 0:
                    #     relation.embedding = embedding

                    relations.append(relation)

            return relations

        except Exception as e:
            logger.error(f"Error querying relations from ChromaDB: {e}")
            return []

    # VectorStorage interface

    def add_vector(self, vector_id: str, vector: Any, metadata: Optional[Dict] = None) -> bool:
        """Add vector to storage."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return False

        try:
            # Determine which collection to use based on metadata or vector_id
            collection_name = "entities"  # default
            if metadata and "type" in metadata:
                collection_name = metadata["type"]
            elif "_relation_" in vector_id:
                collection_name = "relations"

            collection = self.entity_collection if collection_name == "entities" else self.relation_collection

            if collection is None:
                logger.error(f"Collection for {collection_name} is not initialized")
                return False

            # Convert vector to list if needed
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            elif not isinstance(vector, list):
                vector = list(vector)

            # Ensure metadata has proper types for ChromaDB
            safe_metadata = {}
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        safe_metadata[key] = value
                    else:
                        safe_metadata[key] = str(value)

            collection.upsert(
                ids=[vector_id],
                embeddings=[vector],
                metadatas=[safe_metadata],
                documents=[safe_metadata.get("document", vector_id)],
            )

            return True

        except Exception as e:
            logger.error(f"Error adding vector {vector_id} to ChromaDB: {e}")
            return False

    def get_vector(self, vector_id: str) -> Optional[Any]:
        """Get vector from storage."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return None

        try:
            # Try entity collection first
            if self.entity_collection is None:
                logger.error("Entity collection is not initialized")
                return None
            results = self.entity_collection.get(ids=[vector_id], include=["embeddings"])

            if results["ids"] and results.get("embeddings"):
                embeddings = results["embeddings"]
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]

            # Try relation collection
            if self.relation_collection is None:
                logger.error("Relation collection is not initialized")
                return None
            results = self.relation_collection.get(ids=[vector_id], include=["embeddings"])

            if results["ids"] and results.get("embeddings"):
                embeddings = results["embeddings"]
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]

            return None

        except Exception as e:
            logger.error(f"Error getting vector {vector_id} from ChromaDB: {e}")
            return None

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from storage."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return False

        try:
            # Try to delete from both collections
            try:
                self.entity_collection.delete(ids=[vector_id])
            except Exception:
                pass

            try:
                self.relation_collection.delete(ids=[vector_id])
            except Exception:
                pass

            return True

        except Exception as e:
            logger.error(f"Error deleting vector {vector_id} from ChromaDB: {e}")
            return False

    def save_vectors(self, vectors: Dict[str, Any], metadata: Optional[Dict] = None) -> bool:
        """Batch save vectors."""
        try:
            for vector_id, vector in vectors.items():
                if not self.add_vector(vector_id, vector, metadata):
                    return False
            return True

        except Exception as e:
            logger.error(f"Error batch saving vectors to ChromaDB: {e}")
            return False

    def load_vectors(self) -> Tuple[Dict[str, Any], Dict]:
        """Load all vectors."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return {}, {}

        try:
            vectors = {}
            metadata: Dict[str, Any] = {"entity_vectors": {}, "relation_vectors": {}}

            # Load entity vectors with type safety
            entity_results = self.entity_collection.get(include=["embeddings", "metadatas"])
            entity_embeddings = entity_results.get("embeddings") if entity_results.get("embeddings") else []
            entity_metadatas = entity_results.get("metadatas") if entity_results.get("metadatas") else []

            for i, vector_id in enumerate(entity_results["ids"]):
                if entity_embeddings and i < len(entity_embeddings) and entity_embeddings[i] is not None:
                    vectors[vector_id] = entity_embeddings[i]
                if entity_metadatas and i < len(entity_metadatas) and entity_metadatas[i] is not None:
                    metadata["entity_vectors"][vector_id] = entity_metadatas[i]

            # Load relation vectors with type safety
            relation_results = self.relation_collection.get(include=["embeddings", "metadatas"])
            relation_embeddings = relation_results.get("embeddings") if relation_results.get("embeddings") else []
            relation_metadatas = relation_results.get("metadatas") if relation_results.get("metadatas") else []

            for i, vector_id in enumerate(relation_results["ids"]):
                if relation_embeddings and i < len(relation_embeddings) and relation_embeddings[i] is not None:
                    vectors[vector_id] = relation_embeddings[i]
                if relation_metadatas and i < len(relation_metadatas) and relation_metadatas[i] is not None:
                    metadata["relation_vectors"][vector_id] = relation_metadatas[i]

            return vectors, metadata

        except Exception as e:
            logger.error(f"Error loading vectors from ChromaDB: {e}")
            return {}, {}

    def clear(self) -> bool:
        """Clear all vectors."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return False

        try:
            # Delete collections and recreate them
            self.client.delete_collection(f"{self.collection_prefix}_entities")
            self.client.delete_collection(f"{self.collection_prefix}_relations")
            self.client.delete_collection(f"{self.collection_prefix}_graphs")
            self.client.delete_collection(f"{self.collection_prefix}_texts")

            # Recreate collections
            self.entity_collection = self.client.create_collection(
                name=f"{self.collection_prefix}_entities", metadata={"description": "Entity vectors and metadata"}
            )

            self.relation_collection = self.client.create_collection(
                name=f"{self.collection_prefix}_relations", metadata={"description": "Relation vectors and metadata"}
            )

            self.graph_metadata_collection = self.client.create_collection(
                name=f"{self.collection_prefix}_graphs", metadata={"description": "Graph metadata"}
            )

            self.text_collection = self.client.create_collection(
                name=f"{self.collection_prefix}_texts", metadata={"description": "Original text embeddings"}
            )

            return True

        except Exception as e:
            logger.error(f"Error clearing ChromaDB: {e}")
            return False

    def search_similar_vectors(
        self, query_vector: Any, top_k: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            elif not isinstance(query_vector, list):
                query_vector = list(query_vector)

            results = []

            # Search in entity collection
            entity_results = self.entity_collection.query(
                query_embeddings=[query_vector], n_results=min(top_k, 100), include=["distances"]
            )

            for i, vector_id in enumerate(entity_results["ids"][0]):
                distance = entity_results["distances"][0][i]
                # Convert distance to similarity (ChromaDB uses L2 distance by default)
                similarity = 1.0 / (1.0 + distance)
                if similarity >= threshold:
                    results.append((vector_id, similarity))

            # Search in relation collection
            relation_results = self.relation_collection.query(
                query_embeddings=[query_vector], n_results=min(top_k, 100), include=["distances"]
            )

            for i, vector_id in enumerate(relation_results["ids"][0]):
                distance = relation_results["distances"][0][i]
                similarity = 1.0 / (1.0 + distance)
                if similarity >= threshold:
                    results.append((vector_id, similarity))

            # search in text collection if available
            if self.text_collection:
                text_results = self.text_collection.query(
                    query_embeddings=[query_vector], n_results=min(top_k, 100), include=["distances"]
                )

                for i, vector_id in enumerate(text_results["ids"][0]):
                    distance = text_results["distances"][0][i]
                    similarity = 1.0 / (1.0 + distance)
                    if similarity >= threshold:
                        results.append((vector_id, similarity))

            # Sort by similarity and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error searching similar vectors in ChromaDB: {e}")
            return []

    def save(self) -> None:
        """Save current state - ChromaDB auto-persists."""
        pass

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information and statistics."""
        if not self.is_connected():
            return {"error": "Not connected to ChromaDB"}

        try:
            # Get collection sizes
            entity_count = self.entity_collection.count()
            relation_count = self.relation_collection.count()
            graph_count = self.graph_metadata_collection.count()
            text_count = self.text_collection.count() if self.text_collection else 0

            # Calculate directory size
            total_size = 0
            if os.path.exists(self.persist_directory):
                for root, dirs, files in os.walk(self.persist_directory):
                    for file in files:
                        total_size += os.path.getsize(os.path.join(root, file))

            return {
                "persist_directory": self.persist_directory,
                "entity_count": entity_count,
                "relation_count": relation_count,
                "graph_count": graph_count,
                "text_count": text_count,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "is_connected": self.is_connected(),
                "collections": {
                    "entities": f"{self.collection_prefix}_entities",
                    "relations": f"{self.collection_prefix}_relations",
                    "graphs": f"{self.collection_prefix}_graphs",
                    "texts": f"{self.collection_prefix}_texts",
                },
            }

        except Exception as e:
            logger.error(f"Error getting ChromaDB storage info: {e}")
            return {"error": str(e)}

    # Additional CRUD methods for entities and relations

    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """Add entity to specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error(f"Graph {graph_id} not found")
                return False

            graph.add_entity(entity)
            return self.save_graph(graph)

        except Exception as e:
            logger.error(f"Error adding entity to ChromaDB: {e}")
            return False

    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """Add relation to specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error(f"Graph {graph_id} not found")
                return False

            graph.add_relation(relation)
            return self.save_graph(graph)

        except Exception as e:
            logger.error(f"Error adding relation to ChromaDB: {e}")
            return False

    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """Update entity in specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error(f"Graph {graph_id} not found")
                return False

            if entity.id in graph.entities:
                graph.entities[entity.id] = entity
                graph.updated_at = datetime.now()
                return self.save_graph(graph)
            return self.add_entity(graph_id, entity)

        except Exception as e:
            logger.error(f"Error updating entity in ChromaDB: {e}")
            return False

    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """Update relation in specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error(f"Graph {graph_id} not found")
                return False

            if relation.id in graph.relations:
                graph.relations[relation.id] = relation
                graph.updated_at = datetime.now()
                return self.save_graph(graph)
            return self.add_relation(graph_id, relation)

        except Exception as e:
            logger.error(f"Error updating relation in ChromaDB: {e}")
            return False

    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """Remove entity from specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error(f"Graph {graph_id} not found")
                return False

            success = graph.remove_entity(entity_id)
            if success:
                return self.save_graph(graph)
            return False

        except Exception as e:
            logger.error(f"Error removing entity from ChromaDB: {e}")
            return False

    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """Remove relation from specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error(f"Graph {graph_id} not found")
                return False

            success = graph.remove_relation(relation_id)
            if success:
                return self.save_graph(graph)
            return False

        except Exception as e:
            logger.error(f"Error removing relation from ChromaDB: {e}")
            return False

    # Text vector operations

    def add_text_vector(
        self, text_id: str, text_content: str, embedding: List[float], metadata: Optional[Dict] = None
    ) -> bool:
        """Add text vector to ChromaDB.

        Args:
            text_id: Unique identifier for the text
            text_content: Original text content
            embedding: Text embedding vector
            metadata: Additional metadata

        Returns:
            bool: Success status
        """
        if not self.is_connected():
            if not self.connect():
                return False

        try:
            text_metadata = metadata or {}
            text_metadata.update(
                {"text_id": text_id, "char_count": len(text_content), "created_at": datetime.now().isoformat()}
            )

            if self.text_collection is None:
                logger.error("Text collection is not initialized")
                return False

            self.text_collection.upsert(
                ids=[text_id], metadatas=[text_metadata], documents=[text_content], embeddings=[embedding]
            )

            logger.debug(f"Text vector {text_id} added to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error adding text vector {text_id} to ChromaDB: {e}")
            return False

    def get_text_vector(self, text_id: str) -> Optional[Tuple[str, List[float], Dict]]:
        """Get text vector from ChromaDB.

        Args:
            text_id: Text identifier

        Returns:
            Optional[Tuple[str, List[float], Dict]]: Text content, embedding, and metadata
        """
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return None

        try:
            results = self.text_collection.get(ids=[text_id], include=["documents", "embeddings", "metadatas"])

            if not results["ids"]:
                return None

            text_content = results["documents"][0]
            embedding = results["embeddings"][0] if results["embeddings"] else []
            metadata = results["metadatas"][0] if results["metadatas"] else {}

            return text_content, embedding, metadata

        except Exception as e:
            logger.error(f"Error getting text vector {text_id} from ChromaDB: {e}")
            return None

    def search_similar_texts(
        self, query_vector: List[float], top_k: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, str, float]]:
        """Search for similar texts.

        Args:
            query_vector: Query embedding vector
            top_k: Number of results to return
            threshold: Similarity threshold

        Returns:
            List[Tuple[str, str, float]]: List of (text_id, text_content, similarity_score)
        """
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            results = self.text_collection.query(
                query_embeddings=[query_vector], n_results=min(top_k, 100), include=["documents", "distances"]
            )

            similar_texts = []
            for i, text_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # Convert distance to similarity
                similarity = 1.0 / (1.0 + distance)
                if similarity >= threshold:
                    text_content = results["documents"][0][i]
                    similar_texts.append((text_id, text_content, similarity))

            # Sort by similarity descending
            similar_texts.sort(key=lambda x: x[2], reverse=True)
            return similar_texts

        except Exception as e:
            logger.error(f"Error searching similar texts in ChromaDB: {e}")
            return []

    def delete_text_vector(self, text_id: str) -> bool:
        """Delete text vector from ChromaDB.

        Args:
            text_id: Text identifier

        Returns:
            bool: Success status
        """
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return False

        try:
            self.text_collection.delete(ids=[text_id])
            logger.debug(f"Text vector {text_id} deleted from ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error deleting text vector {text_id} from ChromaDB: {e}")
            return False

    def batch_add_text_vectors(self, texts_data: List[Tuple[str, str, List[float], Optional[Dict]]]) -> bool:
        """Batch add text vectors.

        Args:
            texts_data: List of (text_id, text_content, embedding, metadata) tuples

        Returns:
            bool: Success status
        """
        if not self.is_connected():
            if not self.connect():
                return False

        try:
            if not texts_data:
                return True

            ids = []
            documents = []
            embeddings = []
            metadatas = []

            for text_id, text_content, embedding, metadata in texts_data:
                text_metadata = metadata or {}
                text_metadata.update(
                    {"text_id": text_id, "char_count": len(text_content), "created_at": datetime.now().isoformat()}
                )

                ids.append(text_id)
                documents.append(text_content)
                embeddings.append(embedding)
                metadatas.append(text_metadata)

            self.text_collection.upsert(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)

            logger.info(f"Batch added {len(texts_data)} text vectors to ChromaDB")
            return True

        except Exception as e:
            logger.error(f"Error batch adding text vectors to ChromaDB: {e}")
            return False

    # TextChunk CRUD Operations

    def add_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """Add text chunk to graph."""
        if not self.is_connected():
            if not self.connect():
                return False

        try:
            chunk_metadata = {
                "graph_id": graph_id,
                "chunk_id": text_chunk.id,
                "chunk_type": text_chunk.chunk_type,
                "language": text_chunk.language,
                "source": text_chunk.source,
                "char_count": len(text_chunk.content),
                "metadata": json.dumps(text_chunk.metadata),
                "entities": json.dumps(list(text_chunk.entities)),
                "relations": json.dumps(list(text_chunk.relations)),
                "created_at": (
                    text_chunk.created_at.isoformat() if text_chunk.created_at else datetime.now().isoformat()
                ),
                "updated_at": (
                    text_chunk.updated_at.isoformat() if text_chunk.updated_at else datetime.now().isoformat()
                ),
            }

            # Use embedding if available, otherwise create a dummy one
            if hasattr(text_chunk, "embedding") and text_chunk.embedding:
                embedding = text_chunk.embedding
            else:
                raise ValueError("Embedding for text chunk is required but not found.")

            self.text_collection.upsert(
                ids=[f"{graph_id}_{text_chunk.id}"],
                metadatas=[chunk_metadata],
                documents=[text_chunk.content],
                embeddings=[embedding],
            )

            logger.debug(f"Text chunk {text_chunk.id} added to graph {graph_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding text chunk to ChromaDB: {e}")
            return False

    def update_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """Update text chunk in graph."""
        return self.add_text_chunk(graph_id, text_chunk)  # ChromaDB upsert handles updates

    def remove_text_chunk(self, graph_id: str, chunk_id: str) -> bool:
        """Remove text chunk from graph."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return False

        try:
            self.text_collection.delete(ids=[f"{graph_id}_{chunk_id}"])
            logger.debug(f"Text chunk {chunk_id} removed from graph {graph_id}")
            return True

        except Exception as e:
            logger.error(f"Error removing text chunk from ChromaDB: {e}")
            return False

    def batch_add_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """Batch add text chunks to graph."""
        if not self.is_connected():
            if not self.connect():
                return False

        try:
            if not text_chunks:
                return True

            ids = []
            metadatas = []
            documents = []
            embeddings = []

            for chunk in text_chunks:
                chunk_metadata = {
                    "graph_id": graph_id,
                    "chunk_id": chunk.id,
                    "chunk_type": chunk.chunk_type,
                    "language": chunk.language,
                    "source": chunk.source,
                    "char_count": len(chunk.content),
                    "metadata": json.dumps(chunk.metadata),
                    "entities": json.dumps(list(chunk.entities)),
                    "relations": json.dumps(list(chunk.relations)),
                    "created_at": chunk.created_at.isoformat() if chunk.created_at else datetime.now().isoformat(),
                    "updated_at": chunk.updated_at.isoformat() if chunk.updated_at else datetime.now().isoformat(),
                }
                if hasattr(chunk, "embedding") and chunk.embedding:
                    embedding = chunk.embedding
                else:
                    # Create a dummy embedding if not provided
                    raise ValueError("Embedding for text chunk is required but not found.")

                ids.append(f"{graph_id}_{chunk.id}")
                metadatas.append(chunk_metadata)
                documents.append(chunk.content)
                embeddings.append(embedding)

            self.text_collection.upsert(ids=ids, metadatas=metadatas, documents=documents, embeddings=embeddings)

            logger.info(f"Batch added {len(text_chunks)} text chunks to graph {graph_id}")
            return True

        except Exception as e:
            logger.error(f"Error batch adding text chunks to ChromaDB: {e}")
            return False

    def batch_update_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """Batch update text chunks in graph."""
        return self.batch_add_text_chunks(graph_id, text_chunks)  # ChromaDB upsert handles updates

    def batch_remove_text_chunks(self, graph_id: str, chunk_ids: List[str]) -> bool:
        """Batch remove text chunks from graph."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return False

        try:
            if not chunk_ids:
                return True

            ids_to_delete = [f"{graph_id}_{chunk_id}" for chunk_id in chunk_ids]
            self.text_collection.delete(ids=ids_to_delete)

            logger.info(f"Batch removed {len(chunk_ids)} text chunks from graph {graph_id}")
            return True

        except Exception as e:
            logger.error(f"Error batch removing text chunks from ChromaDB: {e}")
            return False

    # Query Operations

    def query_text_chunks(self, conditions: Dict[str, Any]) -> List[TextChunk]:
        """Query text chunks based on conditions."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            where_clause = {}

            # Build where clause from conditions - ChromaDB only supports one condition at a time
            if "graph_id" in conditions:
                where_clause["graph_id"] = conditions["graph_id"]

            # Get results
            results = self.text_collection.get(where=where_clause, include=["metadatas", "documents", "embeddings"])

            text_chunks = []
            for i, chunk_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                content = results["documents"][i]
                try:
                    embedding = (
                        results["embeddings"][i]
                        if "embeddings" in results and results["embeddings"] is not None
                        else None
                    )
                except (IndexError, TypeError):
                    embedding = None

                # Apply additional filters after retrieval
                if "chunk_type" in conditions:
                    if metadata.get("chunk_type") != conditions["chunk_type"]:
                        continue

                if "language" in conditions:
                    if metadata.get("language") != conditions["language"]:
                        continue

                if "source" in conditions:
                    if metadata.get("source") != conditions["source"]:
                        continue

                if "min_char_count" in conditions:
                    if metadata.get("char_count", 0) < conditions["min_char_count"]:
                        continue

                if "max_char_count" in conditions:
                    if metadata.get("char_count", 0) > conditions["max_char_count"]:
                        continue

                chunk = TextChunk(
                    id=str(metadata.get("chunk_id", "")),
                    content=content,
                    chunk_type=str(metadata.get("chunk_type", "text")),
                    language=str(metadata.get("language", "zh")),
                    source=str(metadata.get("source", "")),
                    metadata=json.loads(str(metadata.get("metadata", "{}"))),
                    entities=set(json.loads(str(metadata.get("entities", "[]")))),
                    relations=set(json.loads(str(metadata.get("relations", "[]")))),
                )

                if "created_at" in metadata:
                    chunk.created_at = datetime.fromisoformat(str(metadata["created_at"]))
                if "updated_at" in metadata:
                    chunk.updated_at = datetime.fromisoformat(str(metadata["updated_at"]))

                if embedding is not None and len(embedding) > 0:
                    chunk.embedding = cast(
                        List[float], embedding.tolist() if hasattr(embedding, "tolist") else embedding
                    )

                text_chunks.append(chunk)

            # Apply limit
            limit = conditions.get("limit", 100)
            return text_chunks[:limit]

        except Exception as e:
            logger.error(f"Error querying text chunks from ChromaDB: {e}")
            return []

    # TextChunk Query Operations

    def get_text_chunk(self, graph_id: str, chunk_id: str) -> Optional[TextChunk]:
        """Get text chunk by ID."""
        conditions = {"graph_id": graph_id, "chunk_id": chunk_id}
        chunks = self.query_text_chunks(conditions)
        return chunks[0] if chunks else None

    def get_chunks_by_entity(self, graph_id: str, entity_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific entity."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            # Query all chunks for the graph
            results = self.text_collection.get(
                where={"graph_id": graph_id}, include=["metadatas", "documents", "embeddings"]
            )

            text_chunks = []
            for i, chunk_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                entity_ids = json.loads(str(metadata.get("entities", "[]")))

                # Check if entity_id is in the chunk's entity_ids
                if entity_id in entity_ids:
                    content = results["documents"][i]
                    try:
                        embedding = (
                            results["embeddings"][i]
                            if "embeddings" in results and results["embeddings"] is not None
                            else None
                        )
                    except (IndexError, TypeError):
                        embedding = None

                    chunk = TextChunk(
                        id=str(metadata.get("chunk_id", "")),
                        content=content,
                        chunk_type=str(metadata.get("chunk_type", "text")),
                        language=str(metadata.get("language", "zh")),
                        source=str(metadata.get("source", "")),
                        metadata=json.loads(str(metadata.get("metadata", "{}"))),
                        entities=set(entity_ids),
                        relations=set(json.loads(str(metadata.get("relations", "[]")))),
                    )

                    if "created_at" in metadata:
                        chunk.created_at = datetime.fromisoformat(str(metadata["created_at"]))
                    if "updated_at" in metadata:
                        chunk.updated_at = datetime.fromisoformat(str(metadata["updated_at"]))

                    if embedding is not None and len(embedding) > 0:
                        chunk.embedding = cast(
                            List[float], embedding.tolist() if hasattr(embedding, "tolist") else embedding
                        )

                    text_chunks.append(chunk)

            return text_chunks

        except Exception as e:
            logger.error(f"Error getting chunks by entity from ChromaDB: {e}")
            return []

    def get_chunks_by_relation(self, graph_id: str, relation_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific relation."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            # Query all chunks for the graph
            results = self.text_collection.get(
                where={"graph_id": graph_id}, include=["metadatas", "documents", "embeddings"]
            )

            text_chunks = []
            for i, chunk_id in enumerate(results["ids"]):
                metadata = results["metadatas"][i]
                relation_ids = json.loads(str(metadata.get("relations", "[]")))

                # Check if relation_id is in the chunk's relation_ids
                if relation_id in relation_ids:
                    content = results["documents"][i]
                    try:
                        embedding = (
                            results["embeddings"][i]
                            if "embeddings" in results and results["embeddings"] is not None
                            else None
                        )
                    except (IndexError, TypeError):
                        embedding = None

                    chunk = TextChunk(
                        id=str(metadata.get("chunk_id", "")),
                        content=content,
                        chunk_type=str(metadata.get("chunk_type", "text")),
                        language=str(metadata.get("language", "zh")),
                        source=str(metadata.get("source", "")),
                        metadata=json.loads(str(metadata.get("metadata", "{}"))),
                        entities=set(json.loads(str(metadata.get("entities", "[]")))),
                        relations=set(relation_ids),
                    )

                    if "created_at" in metadata:
                        chunk.created_at = datetime.fromisoformat(str(metadata["created_at"]))
                    if "updated_at" in metadata:
                        chunk.updated_at = datetime.fromisoformat(str(metadata["updated_at"]))

                    if embedding is not None and len(embedding) > 0:
                        chunk.embedding = cast(
                            List[float], embedding.tolist() if hasattr(embedding, "tolist") else embedding
                        )

                    text_chunks.append(chunk)

            return text_chunks

        except Exception as e:
            logger.error(f"Error getting chunks by relation from ChromaDB: {e}")
            return []

    def get_chunks_by_source(self, graph_id: str, source: str) -> List[TextChunk]:
        """Get all text chunks from a specific source."""
        conditions = {"graph_id": graph_id, "source": source}
        return self.query_text_chunks(conditions)

    def get_chunks_by_type(self, graph_id: str, chunk_type: str) -> List[TextChunk]:
        """Get all text chunks of a specific type."""
        conditions = {"graph_id": graph_id, "chunk_type": chunk_type}
        return self.query_text_chunks(conditions)

    def get_chunks_by_language(self, graph_id: str, language: str) -> List[TextChunk]:
        """Get all text chunks in a specific language."""
        conditions = {"graph_id": graph_id, "language": language}
        return self.query_text_chunks(conditions)

    def search_text_chunks(
        self,
        graph_id: str,
        query: str,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 10,
    ) -> List[TextChunk]:
        """Search text chunks by content or metadata."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            where_clause = {"graph_id": graph_id}

            # Get all chunks first, then filter by content
            results = self.text_collection.get(where=where_clause, include=["metadatas", "documents", "embeddings"])

            text_chunks = []
            if results["ids"]:
                for i, chunk_id in enumerate(results["ids"]):
                    metadata = results["metadatas"][i]
                    content = results["documents"][i]

                    # Apply filters
                    if chunk_type and metadata.get("chunk_type") != chunk_type:
                        continue

                    if language and metadata.get("language") != language:
                        continue

                    # Simple text search
                    if query.lower() not in content.lower():
                        continue

                    try:
                        embedding = (
                            results["embeddings"][i]
                            if "embeddings" in results and results["embeddings"] is not None
                            else None
                        )
                    except (IndexError, TypeError):
                        embedding = None

                    chunk = TextChunk(
                        id=str(metadata.get("chunk_id", "")),
                        content=content,
                        chunk_type=str(metadata.get("chunk_type", "text")),
                        language=str(metadata.get("language", "zh")),
                        source=str(metadata.get("source", "")),
                        metadata=json.loads(str(metadata.get("metadata", "{}"))),
                        entities=set(json.loads(str(metadata.get("entities", "[]")))),
                        relations=set(json.loads(str(metadata.get("relations", "[]")))),
                    )

                    if "created_at" in metadata:
                        chunk.created_at = datetime.fromisoformat(str(metadata["created_at"]))
                    if "updated_at" in metadata:
                        chunk.updated_at = datetime.fromisoformat(str(metadata["updated_at"]))

                    if embedding is not None and len(embedding) > 0:
                        chunk.embedding = cast(
                            List[float], embedding.tolist() if hasattr(embedding, "tolist") else embedding
                        )

                    text_chunks.append(chunk)

                    if len(text_chunks) >= limit:
                        break

            return text_chunks

        except Exception as e:
            logger.error(f"Error searching text chunks in ChromaDB: {e}")
            return []

    # TextChunk Embedding Operations

    def add_chunk_embedding(self, graph_id: str, chunk_id: str, embedding: List[float]) -> bool:
        """Add or update embedding for a text chunk."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return False

        try:
            # Get existing chunk data
            results = self.text_collection.get(ids=[f"{graph_id}_{chunk_id}"], include=["metadatas", "documents"])

            if not results["ids"]:
                logger.error(f"Text chunk {chunk_id} not found in graph {graph_id}")
                return False

            # Update with new embedding
            self.text_collection.upsert(
                ids=[f"{graph_id}_{chunk_id}"],
                metadatas=results["metadatas"],
                documents=results["documents"],
                embeddings=[embedding],
            )

            logger.debug(f"Embedding updated for chunk {chunk_id} in graph {graph_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding chunk embedding to ChromaDB: {e}")
            return False

    def get_chunk_embedding(self, graph_id: str, chunk_id: str) -> Optional[List[float]]:
        """Get embedding for a text chunk."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return None

        try:
            results = self.text_collection.get(ids=[f"{graph_id}_{chunk_id}"], include=["embeddings"])

            if results["ids"] and len(results["embeddings"]) > 0:
                embedding = results["embeddings"][0]
                if embedding is not None:
                    return cast(List[float], embedding.tolist() if hasattr(embedding, "tolist") else embedding)

            return None

        except Exception as e:
            logger.error(f"Error getting chunk embedding from ChromaDB: {e}")
            return None

    def search_chunks_by_embedding(
        self,
        graph_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[Tuple[TextChunk, float]]:
        """Search text chunks by embedding similarity."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            where_clause = {"graph_id": graph_id}

            if chunk_type:
                where_clause["chunk_type"] = chunk_type

            if language:
                where_clause["language"] = language

            results = self.text_collection.query(
                where=where_clause,
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["metadatas", "documents", "embeddings", "distances"],
            )

            chunk_similarities = []
            if results["ids"] and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    # Convert distance to similarity
                    similarity = 1.0 / (1.0 + distance)

                    if similarity >= threshold:
                        metadata = results["metadatas"][0][i]
                        content = results["documents"][0][i]
                        embedding = None
                        try:
                            if (
                                results["embeddings"]
                                and len(results["embeddings"]) > 0
                                and len(results["embeddings"][0]) > i
                            ):
                                embedding = results["embeddings"][0][i]
                        except Exception:
                            embedding = None

                        chunk = TextChunk(
                            id=str(metadata.get("chunk_id", "")),
                            content=content,
                            chunk_type=str(metadata.get("chunk_type", "text")),
                            language=str(metadata.get("language", "zh")),
                            source=str(metadata.get("source", "")),
                            metadata=json.loads(str(metadata.get("metadata", "{}"))),
                            entities=set(json.loads(str(metadata.get("entities", "[]")))),
                            relations=set(json.loads(str(metadata.get("relations", "[]")))),
                        )

                        if "created_at" in metadata:
                            chunk.created_at = datetime.fromisoformat(str(metadata["created_at"]))
                        if "updated_at" in metadata:
                            chunk.updated_at = datetime.fromisoformat(str(metadata["updated_at"]))

                        try:
                            if embedding is not None:
                                # Handle numpy array safely
                                if hasattr(embedding, "__len__") and len(embedding) > 0:
                                    embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else embedding
                                    if isinstance(embedding_list, list):
                                        chunk.embedding = cast(List[float], embedding_list)
                        except Exception:
                            # If there's any issue with embedding, skip it
                            pass

                        chunk_similarities.append((chunk, similarity))

            # Sort by similarity descending
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
            return chunk_similarities

        except Exception as e:
            logger.error(f"Error searching chunks by embedding in ChromaDB: {e}")
            return []

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
    ) -> List[Tuple[TextChunk, float]]:
        """Hybrid search combining text and embedding similarity."""
        if not self.is_connected():
            logger.error("Not connected to ChromaDB")
            return []

        try:
            # Normalize weights
            total_weight = text_weight + embedding_weight
            text_weight = text_weight / total_weight
            embedding_weight = embedding_weight / total_weight

            # Text search results
            text_results = self.search_text_chunks(
                graph_id=graph_id,
                query=query_text,
                chunk_type=chunk_type,
                language=language,
                limit=top_k * 2,  # Get more results for combining
            )

            # Embedding search results (if embedding provided)
            embedding_results = []
            if query_embedding:
                embedding_results = self.search_chunks_by_embedding(
                    graph_id=graph_id,
                    query_embedding=query_embedding,
                    top_k=top_k * 2,
                    chunk_type=chunk_type,
                    language=language,
                )

            # Combine results
            chunk_scores = {}

            # Add text search scores
            for i, chunk in enumerate(text_results):
                # Simple scoring based on rank
                score = text_weight * (1.0 - i / len(text_results))
                chunk_scores[chunk.id] = (chunk, score)

            # Add embedding search scores
            for chunk, similarity in embedding_results:
                if chunk.id in chunk_scores:
                    # Combine scores
                    existing_chunk, existing_score = chunk_scores[chunk.id]
                    new_score = existing_score + embedding_weight * similarity
                    chunk_scores[chunk.id] = (existing_chunk, new_score)
                else:
                    # New chunk from embedding search
                    score = embedding_weight * similarity
                    chunk_scores[chunk.id] = (chunk, score)

            # Sort by combined score
            final_results = list(chunk_scores.values())
            final_results.sort(key=lambda x: x[1], reverse=True)

            return final_results[:top_k]

        except Exception as e:
            logger.error(f"Error in hybrid search in ChromaDB: {e}")
            return []
