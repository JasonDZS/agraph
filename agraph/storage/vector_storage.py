"""
Vector storage implementation module.

Provides JSON file-based vector storage implementation with similarity search
and support for Entity, Relation, and TextChunk CRUD and Query operations.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

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


class JsonVectorStorage(
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
    """JSON file-based vector storage implementation with comprehensive graph operations."""

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize JSON vector storage.

        Args:
            file_path: Path to the vector storage file. If None, uses default path.
        """
        if file_path is None:
            file_path = os.path.join(settings.workdir, "vectors.json")
        self.file_path = file_path
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict = {}
        self._is_connected = False

        # Storage for graph entities
        self.entities: Dict[str, Entity] = {}  # graph_id -> {entity_id: Entity}
        self.relations: Dict[str, Relation] = {}  # graph_id -> {relation_id: Relation}
        self.text_chunks: Dict[str, TextChunk] = {}  # graph_id -> {chunk_id: TextChunk}

        # Index for efficient querying
        self._entity_by_graph: Dict[str, Dict[str, Entity]] = {}
        self._relation_by_graph: Dict[str, Dict[str, Relation]] = {}
        self._chunk_by_graph: Dict[str, Dict[str, TextChunk]] = {}

        # Store complete knowledge graphs
        self._graphs: Dict[str, KnowledgeGraph] = {}

        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Ensure storage file exists."""
        if not os.path.exists(self.file_path):
            dir_path = os.path.dirname(self.file_path)
            if dir_path:  # Only create directory if there is a directory part
                os.makedirs(dir_path, exist_ok=True)
            self._save_to_file({}, {}, {}, {}, {}, {})
        else:
            # If file exists, try to load data
            self.vectors, self.metadata, entities_data, relations_data, chunks_data, graphs_data = (
                self._load_from_file()
            )
            self._deserialize_graph_data(entities_data, relations_data, chunks_data)
            self._deserialize_graphs_data(graphs_data)

    def _serialize_graph_data(self) -> Tuple[Dict, Dict, Dict]:
        """Serialize graph data for storage."""
        entities_data = {}
        relations_data = {}
        chunks_data = {}

        for graph_id, entities in self._entity_by_graph.items():
            entities_data[graph_id] = {entity_id: entity.to_dict() for entity_id, entity in entities.items()}

        for graph_id, relations in self._relation_by_graph.items():
            relations_data[graph_id] = {relation_id: relation.to_dict() for relation_id, relation in relations.items()}

        for graph_id, chunks in self._chunk_by_graph.items():
            chunks_data[graph_id] = {chunk_id: chunk.to_dict() for chunk_id, chunk in chunks.items()}

        return entities_data, relations_data, chunks_data

    def _deserialize_graph_data(self, entities_data: Dict, relations_data: Dict, chunks_data: Dict) -> None:
        """Deserialize graph data from storage."""
        # First load entities
        for graph_id, entities in entities_data.items():
            if graph_id not in self._entity_by_graph:
                self._entity_by_graph[graph_id] = {}
            for entity_id, entity_dict in entities.items():
                self._entity_by_graph[graph_id][entity_id] = Entity.from_dict(entity_dict)

        # Then load relations (they need entity references)
        for graph_id, relations in relations_data.items():
            if graph_id not in self._relation_by_graph:
                self._relation_by_graph[graph_id] = {}
            entities_map = self._entity_by_graph.get(graph_id, {})
            for relation_id, relation_dict in relations.items():
                self._relation_by_graph[graph_id][relation_id] = Relation.from_dict(relation_dict, entities_map)

        # Finally load text chunks
        for graph_id, chunks in chunks_data.items():
            if graph_id not in self._chunk_by_graph:
                self._chunk_by_graph[graph_id] = {}
            for chunk_id, chunk_dict in chunks.items():
                self._chunk_by_graph[graph_id][chunk_id] = TextChunk.from_dict(chunk_dict)

    def _deserialize_graphs_data(self, graphs_data: Dict) -> None:
        """Deserialize complete knowledge graphs from storage."""
        for graph_id, graph_dict in graphs_data.items():
            graph = KnowledgeGraph.from_dict(graph_dict)
            self._graphs[graph_id] = graph

    def _save_to_file(
        self,
        vectors: Dict[str, np.ndarray],
        metadata: Dict,
        entities_data: Dict,
        relations_data: Dict,
        chunks_data: Dict,
        graphs_data: Optional[Dict] = None,
    ) -> None:
        """Save vectors and graph data to file."""
        try:
            data = {
                "vectors": {k: v.tolist() for k, v in vectors.items()},
                "metadata": metadata,
                "entities": entities_data,
                "relations": relations_data,
                "text_chunks": chunks_data,
                "graphs": graphs_data or {},
            }
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Error saving data to file: %s", e)

    def _load_from_file(self) -> Tuple[Dict[str, np.ndarray], Dict, Dict, Dict, Dict, Dict]:
        """Load vectors and graph data from file."""
        try:
            if not os.path.exists(self.file_path):
                return {}, {}, {}, {}, {}, {}

            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            vectors = {k: np.array(v, dtype=np.float32) for k, v in data.get("vectors", {}).items()}
            metadata = data.get("metadata", {})
            entities_data = data.get("entities", {})
            relations_data = data.get("relations", {})
            chunks_data = data.get("text_chunks", {})
            graphs_data = data.get("graphs", {})

            return vectors, metadata, entities_data, relations_data, chunks_data, graphs_data
        except Exception as e:
            logger.error("Error loading data from file: %s", e)
            return {}, {}, {}, {}, {}, {}

    # VectorStorageConnection methods

    def connect(self) -> bool:
        """Connect to vector storage."""
        try:
            self._ensure_file_exists()
            # Load existing data
            self.vectors, self.metadata, entities_data, relations_data, chunks_data, graphs_data = (
                self._load_from_file()
            )
            self._deserialize_graph_data(entities_data, relations_data, chunks_data)
            self._deserialize_graphs_data(graphs_data)
            self._is_connected = True
            logger.info("Connected to vector storage at %s", self.file_path)
            return True
        except Exception as e:
            logger.error("Failed to connect to vector storage: %s", e)
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from vector storage."""
        # Save current state
        if self._is_connected:
            self.save()
        self._is_connected = False
        logger.info("Disconnected from vector storage")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._is_connected

    # VectorStorageCRUD methods

    def add_vector(self, vector_id: str, vector: Any, metadata: Optional[Dict] = None) -> bool:
        """Add vector to storage."""
        try:
            if isinstance(vector, (list, tuple)):
                vector = np.array(vector, dtype=np.float32)
            elif isinstance(vector, np.ndarray):
                vector = vector.astype(np.float32)
            else:
                logger.error("Invalid vector type: %s", type(vector))
                return False

            self.vectors[vector_id] = vector
            if metadata:
                if "vector_metadata" not in self.metadata:
                    self.metadata["vector_metadata"] = {}
                self.metadata["vector_metadata"][vector_id] = metadata

            # 自动保存
            if self._is_connected:
                self.save()
            return True
        except Exception as e:
            logger.error("Error adding vector %s: %s", vector_id, e)
            return False

    def get_vector(self, vector_id: str) -> Optional[Any]:
        """Get vector from storage."""
        return self.vectors.get(vector_id)

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from storage."""
        try:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
                if "vector_metadata" in self.metadata and vector_id in self.metadata["vector_metadata"]:
                    del self.metadata["vector_metadata"][vector_id]

                # Auto-save
                if self._is_connected:
                    self.save()
                return True
            return False
        except Exception as e:
            logger.error("Error deleting vector %s: %s", vector_id, e)
            return False

    def save_vectors(self, vectors: Dict[str, Any], metadata: Optional[Dict] = None) -> bool:
        """Batch save vectors."""
        try:
            # Update vectors in memory
            for vector_id, vector in vectors.items():
                if isinstance(vector, (list, tuple)):
                    vector = np.array(vector, dtype=np.float32)
                elif isinstance(vector, np.ndarray):
                    vector = vector.astype(np.float32)

                self.vectors[vector_id] = vector

            # Update metadata
            if metadata:
                self.metadata.update(metadata)

            # Save to file
            if self._is_connected:
                self.save()
            return True
        except Exception as e:
            logger.error("Error saving vectors: %s", e)
            return False

    def load_vectors(self) -> Tuple[Dict[str, Any], Dict]:
        """Load all vectors."""
        try:
            self.vectors, self.metadata, _, _, _, _ = self._load_from_file()
            return dict(self.vectors), self.metadata
        except Exception as e:
            logger.error("Error loading vectors: %s", e)
            return {}, {}

    def clear(self) -> bool:
        """Clear all vectors."""
        try:
            self.vectors.clear()
            self.metadata.clear()
            if self._is_connected:
                self.save()
            return True
        except Exception as e:
            logger.error("Error clearing vectors: %s", e)
            return False

    # VectorStorageQuery methods

    def search_similar_vectors(
        self, query_vector: Any, top_k: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        try:
            if isinstance(query_vector, (list, tuple)):
                query_vector = np.array(query_vector, dtype=np.float32)
            elif not isinstance(query_vector, np.ndarray):
                logger.error("Invalid query vector type: %s", type(query_vector))
                return []

            similarities = []
            for vector_id, vector in self.vectors.items():
                similarity = JsonVectorStorage.compute_similarity(query_vector, vector)
                if similarity >= threshold:
                    similarities.append((vector_id, similarity))

            # 按相似度降序排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.error("Error searching similar vectors: %s", e)
            return []

    @staticmethod
    def compute_similarity(vector1: Any, vector2: Any) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            if isinstance(vector1, (list, tuple)):
                vector1 = np.array(vector1, dtype=np.float32)
            if isinstance(vector2, (list, tuple)):
                vector2 = np.array(vector2, dtype=np.float32)

            dot_product = np.dot(vector1, vector2)
            norm_a = np.linalg.norm(vector1)
            norm_b = np.linalg.norm(vector2)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(dot_product / (norm_a * norm_b))
        except Exception as e:
            logger.error("Error computing similarity: %s", e)
            return 0.0

    # Additional utility methods

    def save(self) -> None:
        """Save current state to file."""
        entities_data, relations_data, chunks_data = self._serialize_graph_data()
        graphs_data = {graph_id: graph.to_dict() for graph_id, graph in self._graphs.items()}
        self._save_to_file(self.vectors, self.metadata, entities_data, relations_data, chunks_data, graphs_data)

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information and statistics."""
        try:
            file_size = 0
            if os.path.exists(self.file_path):
                file_size = os.path.getsize(self.file_path)

            total_entities = sum(len(entities) for entities in self._entity_by_graph.values())
            total_relations = sum(len(relations) for relations in self._relation_by_graph.values())
            total_chunks = sum(len(chunks) for chunks in self._chunk_by_graph.values())

            return {
                "file_path": self.file_path,
                "vector_count": len(self.vectors),
                "entity_count": total_entities,
                "relation_count": total_relations,
                "text_chunk_count": total_chunks,
                "graph_count": len(
                    set(
                        list(self._entity_by_graph.keys())
                        + list(self._relation_by_graph.keys())
                        + list(self._chunk_by_graph.keys())
                    )
                ),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "is_connected": self.is_connected(),
            }
        except Exception as e:
            logger.error("Error getting storage info: %s", e)
            return {}

    # GraphCRUD interface methods

    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """Save knowledge graph to storage."""
        try:
            # Store the graph
            self._graphs[graph.id] = graph

            # Store entities, relations, and text chunks separately for efficient access
            if graph.id not in self._entity_by_graph:
                self._entity_by_graph[graph.id] = {}
            if graph.id not in self._relation_by_graph:
                self._relation_by_graph[graph.id] = {}
            if graph.id not in self._chunk_by_graph:
                self._chunk_by_graph[graph.id] = {}

            # Copy entities
            for entity in graph.entities.values():
                self._entity_by_graph[graph.id][entity.id] = entity
                # Store entity embedding if available
                if hasattr(entity, "embedding") and entity.embedding:
                    embedding_key = f"{graph.id}:entity:{entity.id}"
                    self.vectors[embedding_key] = np.array(entity.embedding, dtype=np.float32)

            # Copy relations
            for relation in graph.relations.values():
                self._relation_by_graph[graph.id][relation.id] = relation
                # Store relation embedding if available
                if hasattr(relation, "embedding") and relation.embedding:
                    embedding_key = f"{graph.id}:relation:{relation.id}"
                    self.vectors[embedding_key] = np.array(relation.embedding, dtype=np.float32)

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.info(f"Graph {graph.id} saved to JSON storage")
            return True

        except Exception as e:
            logger.error(f"Error saving graph {graph.id}: {e}")
            return False

    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load knowledge graph from storage."""
        try:
            if graph_id in self._graphs:
                return self._graphs[graph_id]

            # If not in memory, try to reconstruct from components
            if (
                graph_id in self._entity_by_graph
                or graph_id in self._relation_by_graph
                or graph_id in self._chunk_by_graph
            ):

                # Create new graph
                graph = KnowledgeGraph(id=graph_id, name=f"Graph_{graph_id}")

                # Add entities
                if graph_id in self._entity_by_graph:
                    for entity in self._entity_by_graph[graph_id].values():
                        graph.add_entity(entity)

                # Add relations
                if graph_id in self._relation_by_graph:
                    for relation in self._relation_by_graph[graph_id].values():
                        graph.add_relation(relation)

                # Store reconstructed graph
                self._graphs[graph_id] = graph
                return graph

            logger.warning(f"Graph {graph_id} not found")
            return None

        except Exception as e:
            logger.error(f"Error loading graph {graph_id}: {e}")
            return None

    def delete_graph(self, graph_id: str) -> bool:
        """Delete knowledge graph from storage."""
        try:
            # Remove from graphs storage
            if graph_id in self._graphs:
                del self._graphs[graph_id]

            # Remove entities
            if graph_id in self._entity_by_graph:
                for entity_id in self._entity_by_graph[graph_id].keys():
                    embedding_key = f"{graph_id}:entity:{entity_id}"
                    if embedding_key in self.vectors:
                        del self.vectors[embedding_key]
                del self._entity_by_graph[graph_id]

            # Remove relations
            if graph_id in self._relation_by_graph:
                for relation_id in self._relation_by_graph[graph_id].keys():
                    embedding_key = f"{graph_id}:relation:{relation_id}"
                    if embedding_key in self.vectors:
                        del self.vectors[embedding_key]
                del self._relation_by_graph[graph_id]

            # Remove text chunks
            if graph_id in self._chunk_by_graph:
                for chunk_id in self._chunk_by_graph[graph_id].keys():
                    embedding_key = f"{graph_id}:{chunk_id}"
                    if embedding_key in self.vectors:
                        del self.vectors[embedding_key]
                del self._chunk_by_graph[graph_id]

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.info(f"Graph {graph_id} deleted from JSON storage")
            return True

        except Exception as e:
            logger.error(f"Error deleting graph {graph_id}: {e}")
            return False

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all available graphs."""
        try:
            graph_list = []

            # Get graphs from stored graphs
            for graph_id, graph in self._graphs.items():
                graph_info = {
                    "id": graph.id,
                    "name": graph.name,
                    "created_at": graph.created_at.isoformat() if graph.created_at else None,
                    "updated_at": graph.updated_at.isoformat() if graph.updated_at else None,
                    "entity_count": len(graph.entities),
                    "relation_count": len(graph.relations),
                }
                graph_list.append(graph_info)

            # Add graphs that exist in components but not as complete graphs
            all_graph_ids = set(
                list(self._entity_by_graph.keys())
                + list(self._relation_by_graph.keys())
                + list(self._chunk_by_graph.keys())
            )

            for graph_id in all_graph_ids:
                if not any(g["id"] == graph_id for g in graph_list):
                    graph_info = {
                        "id": graph_id,
                        "name": f"Graph_{graph_id}",
                        "created_at": None,
                        "updated_at": None,
                        "entity_count": len(self._entity_by_graph.get(graph_id, {})),
                        "relation_count": len(self._relation_by_graph.get(graph_id, {})),
                    }
                    graph_list.append(graph_info)

            # Sort by updated_at
            graph_list.sort(key=lambda x: str(x.get("updated_at", "")), reverse=True)
            return graph_list

        except Exception as e:
            logger.error(f"Error listing graphs: {e}")
            return []

    # Entity CRUD Operations
    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """Add entity to graph."""
        try:
            if graph_id not in self._entity_by_graph:
                self._entity_by_graph[graph_id] = {}

            self._entity_by_graph[graph_id][entity.id] = entity

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Added entity %s to graph %s", entity.id, graph_id)
            return True
        except Exception as e:
            logger.error("Error adding entity %s to graph %s: %s", entity.id, graph_id, e)
            return False

    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """Update entity in graph."""
        try:
            if graph_id not in self._entity_by_graph:
                logger.warning("Graph %s not found", graph_id)
                return False

            if entity.id not in self._entity_by_graph[graph_id]:
                logger.warning("Entity %s not found in graph %s", entity.id, graph_id)
                return False

            entity.updated_at = datetime.now()
            self._entity_by_graph[graph_id][entity.id] = entity

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Updated entity %s in graph %s", entity.id, graph_id)
            return True
        except Exception as e:
            logger.error("Error updating entity %s in graph %s: %s", entity.id, graph_id, e)
            return False

    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """Remove entity from graph."""
        try:
            if graph_id not in self._entity_by_graph:
                logger.warning("Graph %s not found", graph_id)
                return False

            if entity_id not in self._entity_by_graph[graph_id]:
                logger.warning("Entity %s not found in graph %s", entity_id, graph_id)
                return False

            # Remove entity
            del self._entity_by_graph[graph_id][entity_id]

            # Clean up relations that reference this entity
            if graph_id in self._relation_by_graph:
                relations_to_remove = []
                for relation_id, relation in self._relation_by_graph[graph_id].items():
                    if (relation.head_entity and relation.head_entity.id == entity_id) or (
                        relation.tail_entity and relation.tail_entity.id == entity_id
                    ):
                        relations_to_remove.append(relation_id)

                for relation_id in relations_to_remove:
                    del self._relation_by_graph[graph_id][relation_id]

            # Clean up text chunk connections
            if graph_id in self._chunk_by_graph:
                for chunk in self._chunk_by_graph[graph_id].values():
                    chunk.remove_entity(entity_id)

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Removed entity %s from graph %s", entity_id, graph_id)
            return True
        except Exception as e:
            logger.error("Error removing entity %s from graph %s: %s", entity_id, graph_id, e)
            return False

    # Relation CRUD Operations
    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """Add relation to graph."""
        try:
            if graph_id not in self._relation_by_graph:
                self._relation_by_graph[graph_id] = {}

            self._relation_by_graph[graph_id][relation.id] = relation

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Added relation %s to graph %s", relation.id, graph_id)
            return True
        except Exception as e:
            logger.error("Error adding relation %s to graph %s: %s", relation.id, graph_id, e)
            return False

    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """Update relation in graph."""
        try:
            if graph_id not in self._relation_by_graph:
                logger.warning("Graph %s not found", graph_id)
                return False

            if relation.id not in self._relation_by_graph[graph_id]:
                logger.warning("Relation %s not found in graph %s", relation.id, graph_id)
                return False

            relation.updated_at = datetime.now()
            self._relation_by_graph[graph_id][relation.id] = relation

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Updated relation %s in graph %s", relation.id, graph_id)
            return True
        except Exception as e:
            logger.error("Error updating relation %s in graph %s: %s", relation.id, graph_id, e)
            return False

    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """Remove relation from graph."""
        try:
            if graph_id not in self._relation_by_graph:
                logger.warning("Graph %s not found", graph_id)
                return False

            if relation_id not in self._relation_by_graph[graph_id]:
                logger.warning("Relation %s not found in graph %s", relation_id, graph_id)
                return False

            # Remove relation
            del self._relation_by_graph[graph_id][relation_id]

            # Clean up text chunk connections
            if graph_id in self._chunk_by_graph:
                for chunk in self._chunk_by_graph[graph_id].values():
                    chunk.remove_relation(relation_id)

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Removed relation %s from graph %s", relation_id, graph_id)
            return True
        except Exception as e:
            logger.error("Error removing relation %s from graph %s: %s", relation_id, graph_id, e)
            return False

    # TextChunk CRUD Operations
    def add_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """Add text chunk to graph."""
        try:
            if graph_id not in self._chunk_by_graph:
                self._chunk_by_graph[graph_id] = {}

            self._chunk_by_graph[graph_id][text_chunk.id] = text_chunk

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Added text chunk %s to graph %s", text_chunk.id, graph_id)
            return True
        except Exception as e:
            logger.error("Error adding text chunk %s to graph %s: %s", text_chunk.id, graph_id, e)
            return False

    def update_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """Update text chunk in graph."""
        try:
            if graph_id not in self._chunk_by_graph:
                logger.warning("Graph %s not found", graph_id)
                return False

            if text_chunk.id not in self._chunk_by_graph[graph_id]:
                logger.warning("Text chunk %s not found in graph %s", text_chunk.id, graph_id)
                return False

            text_chunk.updated_at = datetime.now()
            self._chunk_by_graph[graph_id][text_chunk.id] = text_chunk

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Updated text chunk %s in graph %s", text_chunk.id, graph_id)
            return True
        except Exception as e:
            logger.error("Error updating text chunk %s in graph %s: %s", text_chunk.id, graph_id, e)
            return False

    def remove_text_chunk(self, graph_id: str, chunk_id: str) -> bool:
        """Remove text chunk from graph."""
        try:
            if graph_id not in self._chunk_by_graph:
                logger.warning("Graph %s not found", graph_id)
                return False

            if chunk_id not in self._chunk_by_graph[graph_id]:
                logger.warning("Text chunk %s not found in graph %s", chunk_id, graph_id)
                return False

            # Get the chunk before removing it to clean up connections
            chunk = self._chunk_by_graph[graph_id][chunk_id]

            # Remove chunk
            del self._chunk_by_graph[graph_id][chunk_id]

            # Clean up entity connections
            if graph_id in self._entity_by_graph:
                for entity_id in chunk.entities:
                    if entity_id in self._entity_by_graph[graph_id]:
                        self._entity_by_graph[graph_id][entity_id].remove_text_chunk(chunk_id)

            # Clean up relation connections
            if graph_id in self._relation_by_graph:
                for relation_id in chunk.relations:
                    if relation_id in self._relation_by_graph[graph_id]:
                        self._relation_by_graph[graph_id][relation_id].remove_text_chunk(chunk_id)

            # Remove embedding if exists
            embedding_key = f"{graph_id}:{chunk_id}"
            if embedding_key in self.vectors:
                del self.vectors[embedding_key]

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Removed text chunk %s from graph %s", chunk_id, graph_id)
            return True
        except Exception as e:
            logger.error("Error removing text chunk %s from graph %s: %s", chunk_id, graph_id, e)
            return False

    def batch_add_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """Batch add text chunks to graph."""
        try:
            if graph_id not in self._chunk_by_graph:
                self._chunk_by_graph[graph_id] = {}

            for text_chunk in text_chunks:
                self._chunk_by_graph[graph_id][text_chunk.id] = text_chunk

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Batch added %d text chunks to graph %s", len(text_chunks), graph_id)
            return True
        except Exception as e:
            logger.error("Error batch adding text chunks to graph %s: %s", graph_id, e)
            return False

    def batch_update_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """Batch update text chunks in graph."""
        try:
            if graph_id not in self._chunk_by_graph:
                logger.warning("Graph %s not found", graph_id)
                return False

            for text_chunk in text_chunks:
                if text_chunk.id in self._chunk_by_graph[graph_id]:
                    text_chunk.updated_at = datetime.now()
                    self._chunk_by_graph[graph_id][text_chunk.id] = text_chunk

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Batch updated %d text chunks in graph %s", len(text_chunks), graph_id)
            return True
        except Exception as e:
            logger.error("Error batch updating text chunks in graph %s: %s", graph_id, e)
            return False

    def batch_remove_text_chunks(self, graph_id: str, chunk_ids: List[str]) -> bool:
        """Batch remove text chunks from graph."""
        try:
            if graph_id not in self._chunk_by_graph:
                logger.warning("Graph %s not found", graph_id)
                return False

            for chunk_id in chunk_ids:
                if chunk_id in self._chunk_by_graph[graph_id]:
                    # Get the chunk before removing it to clean up connections
                    chunk = self._chunk_by_graph[graph_id][chunk_id]

                    # Remove chunk
                    del self._chunk_by_graph[graph_id][chunk_id]

                    # Clean up entity connections
                    if graph_id in self._entity_by_graph:
                        for entity_id in chunk.entities:
                            if entity_id in self._entity_by_graph[graph_id]:
                                self._entity_by_graph[graph_id][entity_id].remove_text_chunk(chunk_id)

                    # Clean up relation connections
                    if graph_id in self._relation_by_graph:
                        for relation_id in chunk.relations:
                            if relation_id in self._relation_by_graph[graph_id]:
                                self._relation_by_graph[graph_id][relation_id].remove_text_chunk(chunk_id)

                    # Remove embedding if exists
                    embedding_key = f"{graph_id}:{chunk_id}"
                    if embedding_key in self.vectors:
                        del self.vectors[embedding_key]

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Batch removed %d text chunks from graph %s", len(chunk_ids), graph_id)
            return True
        except Exception as e:
            logger.error("Error batch removing text chunks from graph %s: %s", graph_id, e)
            return False

    # Query Operations
    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """Query entities based on conditions."""
        try:
            graph_id = conditions.get("graph_id")
            if not graph_id:
                logger.warning("graph_id is required for entity query")
                return []

            if graph_id not in self._entity_by_graph:
                return []

            entities = list(self._entity_by_graph[graph_id].values())

            # Apply filters
            if "entity_type" in conditions:
                entities = [e for e in entities if e.entity_type == conditions["entity_type"]]

            if "name" in conditions:
                name_filter = conditions["name"].lower()
                entities = [e for e in entities if name_filter in e.name.lower()]

            if "source" in conditions:
                entities = [e for e in entities if e.source == conditions["source"]]

            if "min_confidence" in conditions:
                entities = [e for e in entities if e.confidence >= conditions["min_confidence"]]

            if "text_chunk_id" in conditions:
                chunk_id = conditions["text_chunk_id"]
                entities = [e for e in entities if chunk_id in e.text_chunks]

            # Apply limit
            limit = conditions.get("limit", 100)
            return entities[:limit]

        except Exception as e:
            logger.error("Error querying entities: %s", e)
            return []

    def query_relations(
        self,
        head_entity: Optional[str] = None,
        tail_entity: Optional[str] = None,
        relation_type: Optional[Any] = None,
        **kwargs: Any,
    ) -> List[Relation]:
        """Query relations based on conditions."""
        try:
            graph_id = kwargs.get("graph_id")
            if not graph_id:
                logger.warning("graph_id is required for relation query")
                return []

            if graph_id not in self._relation_by_graph:
                return []

            relations = list(self._relation_by_graph[graph_id].values())

            # Apply filters
            if head_entity:
                relations = [r for r in relations if r.head_entity and r.head_entity.id == head_entity]

            if tail_entity:
                relations = [r for r in relations if r.tail_entity and r.tail_entity.id == tail_entity]

            if relation_type:
                relations = [r for r in relations if r.relation_type == relation_type]

            if "source" in kwargs:
                relations = [r for r in relations if r.source == kwargs["source"]]

            if "min_confidence" in kwargs:
                relations = [r for r in relations if r.confidence >= kwargs["min_confidence"]]

            if "text_chunk_id" in kwargs:
                chunk_id = kwargs["text_chunk_id"]
                relations = [r for r in relations if chunk_id in r.text_chunks]

            # Apply limit
            limit = kwargs.get("limit", 100)
            return relations[:limit]

        except Exception as e:
            logger.error("Error querying relations: %s", e)
            return []

    def query_text_chunks(self, conditions: Dict[str, Any]) -> List[TextChunk]:
        """Query text chunks based on conditions."""
        try:
            graph_id = conditions.get("graph_id")
            if not graph_id:
                logger.warning("graph_id is required for text chunk query")
                return []

            if graph_id not in self._chunk_by_graph:
                return []

            chunks = list(self._chunk_by_graph[graph_id].values())

            # Apply filters
            if "chunk_type" in conditions:
                chunks = [c for c in chunks if c.chunk_type == conditions["chunk_type"]]

            if "language" in conditions:
                chunks = [c for c in chunks if c.language == conditions["language"]]

            if "source" in conditions:
                chunks = [c for c in chunks if c.source == conditions["source"]]

            if "min_confidence" in conditions:
                chunks = [c for c in chunks if c.confidence >= conditions["min_confidence"]]

            if "content" in conditions:
                content_filter = conditions["content"].lower()
                chunks = [c for c in chunks if content_filter in c.content.lower()]

            if "entity_id" in conditions:
                entity_id = conditions["entity_id"]
                chunks = [c for c in chunks if entity_id in c.entities]

            if "relation_id" in conditions:
                relation_id = conditions["relation_id"]
                chunks = [c for c in chunks if relation_id in c.relations]

            # Apply limit
            limit = conditions.get("limit", 100)
            return chunks[:limit]

        except Exception as e:
            logger.error("Error querying text chunks: %s", e)
            return []

    # TextChunk Query Operations
    def get_text_chunk(self, graph_id: str, chunk_id: str) -> Optional[TextChunk]:
        """Get text chunk by ID."""
        try:
            if graph_id in self._chunk_by_graph and chunk_id in self._chunk_by_graph[graph_id]:
                return self._chunk_by_graph[graph_id][chunk_id]
            return None
        except Exception as e:
            logger.error("Error getting text chunk %s from graph %s: %s", chunk_id, graph_id, e)
            return None

    def get_chunks_by_entity(self, graph_id: str, entity_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific entity."""
        try:
            if graph_id not in self._chunk_by_graph:
                return []

            return [chunk for chunk in self._chunk_by_graph[graph_id].values() if entity_id in chunk.entities]
        except Exception as e:
            logger.error("Error getting chunks by entity %s in graph %s: %s", entity_id, graph_id, e)
            return []

    def get_chunks_by_relation(self, graph_id: str, relation_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific relation."""
        try:
            if graph_id not in self._chunk_by_graph:
                return []

            return [chunk for chunk in self._chunk_by_graph[graph_id].values() if relation_id in chunk.relations]
        except Exception as e:
            logger.error("Error getting chunks by relation %s in graph %s: %s", relation_id, graph_id, e)
            return []

    def get_chunks_by_source(self, graph_id: str, source: str) -> List[TextChunk]:
        """Get all text chunks from a specific source."""
        try:
            if graph_id not in self._chunk_by_graph:
                return []

            return [chunk for chunk in self._chunk_by_graph[graph_id].values() if chunk.source == source]
        except Exception as e:
            logger.error("Error getting chunks by source %s in graph %s: %s", source, graph_id, e)
            return []

    def get_chunks_by_type(self, graph_id: str, chunk_type: str) -> List[TextChunk]:
        """Get all text chunks of a specific type."""
        try:
            if graph_id not in self._chunk_by_graph:
                return []

            return [chunk for chunk in self._chunk_by_graph[graph_id].values() if chunk.chunk_type == chunk_type]
        except Exception as e:
            logger.error("Error getting chunks by type %s in graph %s: %s", chunk_type, graph_id, e)
            return []

    def get_chunks_by_language(self, graph_id: str, language: str) -> List[TextChunk]:
        """Get all text chunks in a specific language."""
        try:
            if graph_id not in self._chunk_by_graph:
                return []

            return [chunk for chunk in self._chunk_by_graph[graph_id].values() if chunk.language == language]
        except Exception as e:
            logger.error("Error getting chunks by language %s in graph %s: %s", language, graph_id, e)
            return []

    def search_text_chunks(
        self,
        graph_id: str,
        query: str,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 10,
    ) -> List[TextChunk]:
        """Search text chunks by content or metadata."""
        try:
            if graph_id not in self._chunk_by_graph:
                return []

            query_lower = query.lower()
            chunks = []

            for chunk in self._chunk_by_graph[graph_id].values():
                # Apply filters first
                if chunk_type and chunk.chunk_type != chunk_type:
                    continue
                if language and chunk.language != language:
                    continue

                # Search in content and title
                if query_lower in chunk.content.lower() or query_lower in chunk.title.lower():
                    chunks.append(chunk)

                if len(chunks) >= limit:
                    break

            return chunks
        except Exception as e:
            logger.error("Error searching text chunks in graph %s: %s", graph_id, e)
            return []

    # TextChunk Embedding Operations
    def add_chunk_embedding(self, graph_id: str, chunk_id: str, embedding: List[float]) -> bool:
        """Add or update embedding for a text chunk."""
        try:
            # Verify chunk exists
            if graph_id not in self._chunk_by_graph or chunk_id not in self._chunk_by_graph[graph_id]:
                logger.warning("Text chunk %s not found in graph %s", chunk_id, graph_id)
                return False

            # Store embedding with composite key
            embedding_key = f"{graph_id}:{chunk_id}"
            embedding_array = np.array(embedding, dtype=np.float32)
            self.vectors[embedding_key] = embedding_array

            # Update chunk's embedding field
            chunk = self._chunk_by_graph[graph_id][chunk_id]
            chunk.embedding = embedding
            chunk.updated_at = datetime.now()

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug("Added embedding for chunk %s in graph %s", chunk_id, graph_id)
            return True
        except Exception as e:
            logger.error("Error adding embedding for chunk %s in graph %s: %s", chunk_id, graph_id, e)
            return False

    def get_chunk_embedding(self, graph_id: str, chunk_id: str) -> Optional[List[float]]:
        """Get embedding for a text chunk."""
        try:
            # Try to get from vectors storage first
            embedding_key = f"{graph_id}:{chunk_id}"
            if embedding_key in self.vectors:
                return list(self.vectors[embedding_key].tolist())

            # Fallback to chunk's embedding field
            if graph_id in self._chunk_by_graph and chunk_id in self._chunk_by_graph[graph_id]:
                chunk = self._chunk_by_graph[graph_id][chunk_id]
                return chunk.embedding

            return None
        except Exception as e:
            logger.error("Error getting embedding for chunk %s in graph %s: %s", chunk_id, graph_id, e)
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
        try:
            if graph_id not in self._chunk_by_graph:
                return []

            query_vector = np.array(query_embedding, dtype=np.float32)
            results = []

            for chunk_id, chunk in self._chunk_by_graph[graph_id].items():
                # Apply filters
                if chunk_type and chunk.chunk_type != chunk_type:
                    continue
                if language and chunk.language != language:
                    continue

                # Get chunk embedding
                embedding_key = f"{graph_id}:{chunk_id}"
                chunk_embedding = None

                if embedding_key in self.vectors:
                    chunk_embedding = self.vectors[embedding_key]
                elif chunk.embedding:
                    chunk_embedding = np.array(chunk.embedding, dtype=np.float32)

                if chunk_embedding is not None:
                    similarity = JsonVectorStorage.compute_similarity(query_vector, chunk_embedding)
                    if similarity >= threshold:
                        results.append((chunk, similarity))

            # Sort by similarity (descending) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error("Error searching chunks by embedding in graph %s: %s", graph_id, e)
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
        try:
            if graph_id not in self._chunk_by_graph:
                return []

            query_lower = query_text.lower()
            query_vector = None
            if query_embedding:
                query_vector = np.array(query_embedding, dtype=np.float32)

            results = []

            for chunk_id, chunk in self._chunk_by_graph[graph_id].items():
                # Apply filters
                if chunk_type and chunk.chunk_type != chunk_type:
                    continue
                if language and chunk.language != language:
                    continue

                # Calculate text similarity (simple keyword matching)
                text_similarity = 0.0
                content_lower = chunk.content.lower()
                title_lower = chunk.title.lower()

                if query_lower in content_lower or query_lower in title_lower:
                    # Simple scoring: exact match gets higher score
                    if query_lower == content_lower or query_lower == title_lower:
                        text_similarity = 1.0
                    else:
                        # Partial match scoring based on length ratio
                        content_match_ratio = query_lower.count(" ") + 1
                        text_similarity = min(0.9, content_match_ratio / (content_lower.count(" ") + 1))

                # Calculate embedding similarity
                embedding_similarity = 0.0
                if query_vector is not None:
                    embedding_key = f"{graph_id}:{chunk_id}"
                    chunk_embedding = None

                    if embedding_key in self.vectors:
                        chunk_embedding = self.vectors[embedding_key]
                    elif chunk.embedding:
                        chunk_embedding = np.array(chunk.embedding, dtype=np.float32)

                    if chunk_embedding is not None:
                        embedding_similarity = JsonVectorStorage.compute_similarity(query_vector, chunk_embedding)

                # Combine scores
                combined_score = text_similarity * text_weight + embedding_similarity * embedding_weight

                if combined_score > 0:
                    results.append((chunk, combined_score))

            # Sort by combined score (descending) and return top_k
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error("Error in hybrid search for graph %s: %s", graph_id, e)
            return []

    # Additional text vector operations to match ChromaDB interface

    def add_text_vector(
        self, text_id: str, text_content: str, embedding: List[float], metadata: Optional[Dict] = None
    ) -> bool:
        """Add text vector to storage.

        Args:
            text_id: Unique identifier for the text
            text_content: Original text content
            embedding: Text embedding vector
            metadata: Additional metadata

        Returns:
            bool: Success status
        """
        try:
            # Store the embedding
            embedding_key = f"text:{text_id}"
            self.vectors[embedding_key] = np.array(embedding, dtype=np.float32)

            # Store metadata including content
            if "text_metadata" not in self.metadata:
                self.metadata["text_metadata"] = {}

            text_metadata = metadata or {}
            text_metadata.update(
                {
                    "text_id": text_id,
                    "content": text_content,
                    "char_count": len(text_content),
                    "created_at": datetime.now().isoformat(),
                }
            )

            self.metadata["text_metadata"][text_id] = text_metadata

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug(f"Text vector {text_id} added to JSON storage")
            return True

        except Exception as e:
            logger.error(f"Error adding text vector {text_id}: {e}")
            return False

    def get_text_vector(self, text_id: str) -> Optional[Tuple[str, List[float], Dict]]:
        """Get text vector from storage.

        Args:
            text_id: Text identifier

        Returns:
            Optional[Tuple[str, List[float], Dict]]: Text content, embedding, and metadata
        """
        try:
            embedding_key = f"text:{text_id}"
            if embedding_key not in self.vectors:
                return None

            embedding = self.vectors[embedding_key].tolist()

            # Get metadata
            text_metadata = self.metadata.get("text_metadata", {}).get(text_id, {})
            text_content = text_metadata.get("content", "")

            return text_content, embedding, text_metadata

        except Exception as e:
            logger.error(f"Error getting text vector {text_id}: {e}")
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
        try:
            query_array = np.array(query_vector, dtype=np.float32)
            results = []

            text_metadata = self.metadata.get("text_metadata", {})

            for vector_id, vector in self.vectors.items():
                if not vector_id.startswith("text:"):
                    continue

                text_id = vector_id[5:]  # Remove "text:" prefix
                similarity = JsonVectorStorage.compute_similarity(query_array, vector)

                if similarity >= threshold:
                    text_content = text_metadata.get(text_id, {}).get("content", "")
                    results.append((text_id, text_content, similarity))

            # Sort by similarity descending
            results.sort(key=lambda x: x[2], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error(f"Error searching similar texts: {e}")
            return []

    def delete_text_vector(self, text_id: str) -> bool:
        """Delete text vector from storage.

        Args:
            text_id: Text identifier

        Returns:
            bool: Success status
        """
        try:
            embedding_key = f"text:{text_id}"
            if embedding_key in self.vectors:
                del self.vectors[embedding_key]

            if "text_metadata" in self.metadata and text_id in self.metadata["text_metadata"]:
                del self.metadata["text_metadata"][text_id]

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.debug(f"Text vector {text_id} deleted from JSON storage")
            return True

        except Exception as e:
            logger.error(f"Error deleting text vector {text_id}: {e}")
            return False

    def batch_add_text_vectors(self, texts_data: List[Tuple[str, str, List[float], Optional[Dict]]]) -> bool:
        """Batch add text vectors.

        Args:
            texts_data: List of (text_id, text_content, embedding, metadata) tuples

        Returns:
            bool: Success status
        """
        try:
            if not texts_data:
                return True

            # Ensure text_metadata exists
            if "text_metadata" not in self.metadata:
                self.metadata["text_metadata"] = {}

            for text_id, text_content, embedding, metadata in texts_data:
                # Store embedding
                embedding_key = f"text:{text_id}"
                self.vectors[embedding_key] = np.array(embedding, dtype=np.float32)

                # Store metadata
                text_metadata = metadata or {}
                text_metadata.update(
                    {
                        "text_id": text_id,
                        "content": text_content,
                        "char_count": len(text_content),
                        "created_at": datetime.now().isoformat(),
                    }
                )

                self.metadata["text_metadata"][text_id] = text_metadata

            # Auto-save if connected
            if self._is_connected:
                self.save()

            logger.info(f"Batch added {len(texts_data)} text vectors to JSON storage")
            return True

        except Exception as e:
            logger.error(f"Error batch adding text vectors: {e}")
            return False
