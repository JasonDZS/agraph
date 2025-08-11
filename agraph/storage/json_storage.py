"""
JSON file storage implementation.

Provides JSON file-based storage for knowledge graphs.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..config import settings
from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation
from ..text import TextChunk
from .base_storage import GraphStorage


class JsonStorage(GraphStorage):
    """JSON file-based graph storage implementation."""

    def __init__(self, storage_dir: Optional[str] = None):
        """
        Initialize JSON storage.

        Args:
            storage_dir: Directory to store JSON files. If None, uses workdir from settings.
        """
        super().__init__()
        if storage_dir is not None:
            self.storage_dir = storage_dir
        else:
            self.storage_dir = settings.workdir
        self.graphs_file = os.path.join(self.storage_dir, "graphs.json")
        self.ensure_storage_dir()

    def ensure_storage_dir(self) -> None:
        """Ensure storage directory exists."""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)
            logger.info("Created storage directory: %s", self.storage_dir)

    def connect(self) -> bool:
        """Connect to file system storage."""
        try:
            self.ensure_storage_dir()
            self._is_connected = True
            logger.info("Connected to JSON storage at %s", self.storage_dir)
            return True
        except Exception as e:
            logger.error("Failed to connect to JSON storage: %s", e)
            return False

    def disconnect(self) -> None:
        """Disconnect from storage."""
        self._is_connected = False
        logger.info("Disconnected from JSON storage")

    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """Save knowledge graph to JSON file."""
        if not self.is_connected():
            if not self.connect():
                return False

        try:
            # 保存图谱数据到单独文件
            graph_file = os.path.join(self.storage_dir, f"{graph.id}.json")
            graph_data = graph.to_dict()

            with open(graph_file, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            # 更新图谱索引
            self._update_graph_index(graph)

            logger.info("Graph %s saved to %s", graph.id, graph_file)
            return True

        except Exception as e:
            logger.error("Error saving graph to JSON: %s", e)
            return False

    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """Load knowledge graph from JSON file."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return None

        try:
            graph_file = os.path.join(self.storage_dir, f"{graph_id}.json")

            if not os.path.exists(graph_file):
                logger.warning("Graph file not found: %s", graph_file)
                return None

            with open(graph_file, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            graph = KnowledgeGraph.from_dict(graph_data)
            logger.info("Graph %s loaded from %s", graph_id, graph_file)
            return graph

        except Exception as e:
            logger.error("Error loading graph from JSON: %s", e)
            return None

    def delete_graph(self, graph_id: str) -> bool:
        """Delete knowledge graph from storage."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return False

        try:
            graph_file = os.path.join(self.storage_dir, f"{graph_id}.json")

            if os.path.exists(graph_file):
                os.remove(graph_file)
                logger.info("Graph file deleted: %s", graph_file)

            # Remove from index
            self._remove_from_graph_index(graph_id)

            return True

        except Exception as e:
            logger.error("Error deleting graph: %s", e)
            return False

    def list_graphs(self) -> List[Dict[str, Any]]:
        """List all available graphs."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return []

        try:
            if not os.path.exists(self.graphs_file):
                return []

            with open(self.graphs_file, "r", encoding="utf-8") as f:
                graphs_index = json.load(f)

            result: list[dict[str, Any]] = graphs_index.get("graphs", [])
            return result

        except Exception as e:
            logger.error("Error listing graphs: %s", e)
            return []

    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """Query entities based on specified conditions."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return []

        try:
            graph_id = conditions.get("graph_id")
            if not graph_id:
                logger.error("graph_id is required for entity query")
                return []

            graph = self.load_graph(graph_id)
            if not graph:
                return []

            entities = list(graph.entities.values())

            # Apply filter conditions
            if "entity_type" in conditions:
                entity_type = conditions["entity_type"]
                entities = [e for e in entities if getattr(e.entity_type, "value", str(e.entity_type)) == entity_type]

            if "name" in conditions:
                name_filter = conditions["name"].lower()
                entities = [e for e in entities if name_filter in e.name.lower()]

            if "min_confidence" in conditions:
                min_confidence = conditions["min_confidence"]
                entities = [e for e in entities if e.confidence >= min_confidence]

            # Limit results
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
        """Query relations based on specified conditions."""
        if not self.is_connected():
            logger.error("Not connected to storage")
            return []

        try:
            graph_id = kwargs.get("graph_id")
            if not graph_id:
                logger.error("graph_id is required for relation query")
                return []

            graph = self.load_graph(graph_id)
            if not graph:
                return []

            relations = list(graph.relations.values())

            # Apply filter conditions
            if head_entity:
                relations = [r for r in relations if r.head_entity and r.head_entity.id == head_entity]

            if tail_entity:
                relations = [r for r in relations if r.tail_entity and r.tail_entity.id == tail_entity]

            if relation_type:
                relations = [r for r in relations if r.relation_type == relation_type]

            return relations

        except Exception as e:
            logger.error("Error querying relations: %s", e)
            return []

    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """Add entity to specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            graph.add_entity(entity)
            return self.save_graph(graph)

        except Exception as e:
            logger.error("Error adding entity: %s", e)
            return False

    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """Add relation to specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            graph.add_relation(relation)
            return self.save_graph(graph)

        except Exception as e:
            logger.error("Error adding relation: %s", e)
            return False

    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """Update entity in specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            if entity.id in graph.entities:
                graph.entities[entity.id] = entity
                graph.updated_at = datetime.now()
                return self.save_graph(graph)
            return self.add_entity(graph_id, entity)

        except Exception as e:
            logger.error("Error updating entity: %s", e)
            return False

    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """Update relation in specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            if relation.id in graph.relations:
                graph.relations[relation.id] = relation
                graph.updated_at = datetime.now()
                return self.save_graph(graph)
            return self.add_relation(graph_id, relation)

        except Exception as e:
            logger.error("Error updating relation: %s", e)
            return False

    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """Remove entity from specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            success = graph.remove_entity(entity_id)
            if success:
                return self.save_graph(graph)
            return False

        except Exception as e:
            logger.error("Error removing entity: %s", e)
            return False

    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """Remove relation from specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            success = graph.remove_relation(relation_id)
            if success:
                return self.save_graph(graph)
            return False

        except Exception as e:
            logger.error("Error removing relation: %s", e)
            return False

    def _update_graph_index(self, graph: KnowledgeGraph) -> None:
        """Update graph index with new graph information."""
        try:
            graphs_index: Dict[str, Any] = {"graphs": []}

            if os.path.exists(self.graphs_file):
                with open(self.graphs_file, "r", encoding="utf-8") as f:
                    graphs_index = json.load(f)

            # Remove existing graph record
            graphs_list = graphs_index.get("graphs", [])
            graphs_list = [g for g in graphs_list if g.get("id") != graph.id]

            # Add new record
            graph_info = {
                "id": graph.id,
                "name": graph.name,
                "created_at": graph.created_at.isoformat(),
                "updated_at": graph.updated_at.isoformat(),
                "entity_count": len(graph.entities),
                "relation_count": len(graph.relations),
            }
            graphs_list.append(graph_info)

            # Sort by update time
            graphs_list.sort(key=lambda x: x["updated_at"], reverse=True)

            graphs_index["graphs"] = graphs_list

            with open(self.graphs_file, "w", encoding="utf-8") as f:
                json.dump(graphs_index, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Error updating graph index: %s", e)

    def _remove_from_graph_index(self, graph_id: str) -> None:
        """Remove graph from index."""
        try:
            if not os.path.exists(self.graphs_file):
                return

            with open(self.graphs_file, "r", encoding="utf-8") as f:
                graphs_index = json.load(f)

            graphs_list = graphs_index.get("graphs", [])
            graphs_list = [g for g in graphs_list if g.get("id") != graph_id]

            graphs_index["graphs"] = graphs_list

            with open(self.graphs_file, "w", encoding="utf-8") as f:
                json.dump(graphs_index, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error("Error removing from graph index: %s", e)

    def compact_storage(self) -> None:
        """Compact storage by removing invalid index entries."""
        try:
            # Clean up non-existent graph indexes
            graphs_list = self.list_graphs()
            valid_graphs = []

            for graph_info in graphs_list:
                graph_id = graph_info.get("id")
                graph_file = os.path.join(self.storage_dir, f"{graph_id}.json")

                if os.path.exists(graph_file):
                    valid_graphs.append(graph_info)
                else:
                    logger.info("Removing invalid graph index entry: %s", graph_id)

            # Update index
            graphs_index = {"graphs": valid_graphs}
            with open(self.graphs_file, "w", encoding="utf-8") as f:
                json.dump(graphs_index, f, ensure_ascii=False, indent=2)

            logger.info("Storage compaction completed")

        except Exception as e:
            logger.error("Error compacting storage: %s", e)

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information and statistics."""
        try:
            total_size = 0
            file_count = 0

            for filename in os.listdir(self.storage_dir):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.storage_dir, filename)
                    total_size += os.path.getsize(file_path)
                    file_count += 1

            graphs_count = len(self.list_graphs())

            return {
                "storage_dir": self.storage_dir,
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "file_count": file_count,
                "graphs_count": graphs_count,
                "is_connected": self.is_connected(),
            }

        except Exception as e:
            logger.error("Error getting storage info: %s", e)
            return {}

    # TextChunk specific implementations

    def add_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """Add text chunk to specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            graph.add_text_chunk(text_chunk)
            return self.save_graph(graph)

        except Exception as e:
            logger.error("Error adding text chunk: %s", e)
            return False

    def update_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """Update text chunk in specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            if text_chunk.id in graph.text_chunks:
                graph.text_chunks[text_chunk.id] = text_chunk
                graph.updated_at = datetime.now()
                return self.save_graph(graph)
            return self.add_text_chunk(graph_id, text_chunk)

        except Exception as e:
            logger.error("Error updating text chunk: %s", e)
            return False

    def remove_text_chunk(self, graph_id: str, chunk_id: str) -> bool:
        """Remove text chunk from specified graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            success = graph.remove_text_chunk(chunk_id)
            if success:
                return self.save_graph(graph)
            return False

        except Exception as e:
            logger.error("Error removing text chunk: %s", e)
            return False

    def query_text_chunks(self, conditions: Dict[str, Any]) -> List[TextChunk]:
        """Query text chunks based on specified conditions."""

        if not self.is_connected():
            logger.error("Not connected to storage")
            return []

        try:
            graph_id = conditions.get("graph_id")
            if not graph_id:
                logger.error("graph_id is required for text chunk query")
                return []

            graph = self.load_graph(graph_id)
            if not graph:
                return []

            chunks = list(graph.text_chunks.values())

            # Apply filter conditions
            if "chunk_type" in conditions:
                chunk_type = conditions["chunk_type"]
                chunks = [c for c in chunks if c.chunk_type == chunk_type]

            if "language" in conditions:
                language = conditions["language"]
                chunks = [c for c in chunks if c.language == language]

            if "source" in conditions:
                source = conditions["source"]
                chunks = [c for c in chunks if c.source == source]

            if "min_confidence" in conditions:
                min_confidence = conditions["min_confidence"]
                chunks = [c for c in chunks if c.confidence >= min_confidence]

            if "entity_ids" in conditions:
                entity_ids = set(conditions["entity_ids"])
                chunks = [c for c in chunks if entity_ids.intersection(c.entities)]

            if "relation_ids" in conditions:
                relation_ids = set(conditions["relation_ids"])
                chunks = [c for c in chunks if relation_ids.intersection(c.relations)]

            # Limit results
            limit = conditions.get("limit", 100)
            return chunks[:limit]

        except Exception as e:
            logger.error("Error querying text chunks: %s", e)
            return []

    def get_text_chunk(self, graph_id: str, chunk_id: str) -> Optional[TextChunk]:
        """Get text chunk by ID."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return None
            return graph.get_text_chunk(chunk_id)

        except Exception as e:
            logger.error("Error getting text chunk: %s", e)
            return None

    def get_chunks_by_entity(self, graph_id: str, entity_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific entity."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return []
            return graph.get_entity_text_chunks(entity_id)

        except Exception as e:
            logger.error("Error getting chunks by entity: %s", e)
            return []

    def get_chunks_by_relation(self, graph_id: str, relation_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific relation."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return []
            return graph.get_relation_text_chunks(relation_id)

        except Exception as e:
            logger.error("Error getting chunks by relation: %s", e)
            return []

    def batch_add_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """Batch add text chunks to graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            success_count = 0
            for chunk in text_chunks:
                if graph.add_text_chunk(chunk):
                    success_count += 1

            if success_count > 0:
                self.save_graph(graph)
                logger.info("Successfully added %d/%d text chunks", success_count, len(text_chunks))

            return success_count == len(text_chunks)

        except Exception as e:
            logger.error("Error batch adding text chunks: %s", e)
            return False

    def batch_update_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """Batch update text chunks in graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            success_count = 0
            for chunk in text_chunks:
                if chunk.id in graph.text_chunks:
                    graph.text_chunks[chunk.id] = chunk
                    success_count += 1
                else:
                    # If chunk doesn't exist, add it
                    if graph.add_text_chunk(chunk):
                        success_count += 1

            if success_count > 0:
                graph.updated_at = datetime.now()
                self.save_graph(graph)
                logger.info("Successfully updated %d/%d text chunks", success_count, len(text_chunks))

            return success_count == len(text_chunks)

        except Exception as e:
            logger.error("Error batch updating text chunks: %s", e)
            return False

    def batch_remove_text_chunks(self, graph_id: str, chunk_ids: List[str]) -> bool:
        """Batch remove text chunks from graph."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                logger.error("Graph %s not found", graph_id)
                return False

            success_count = 0
            for chunk_id in chunk_ids:
                if graph.remove_text_chunk(chunk_id):
                    success_count += 1

            if success_count > 0:
                self.save_graph(graph)
                logger.info("Successfully removed %d/%d text chunks", success_count, len(chunk_ids))

            return success_count == len(chunk_ids)

        except Exception as e:
            logger.error("Error batch removing text chunks: %s", e)
            return False

    def get_chunks_by_source(self, graph_id: str, source: str) -> List[TextChunk]:
        """Get all text chunks from a specific source."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return []
            return graph.get_text_chunks_by_source(source)

        except Exception as e:
            logger.error("Error getting chunks by source: %s", e)
            return []

    def get_chunks_by_type(self, graph_id: str, chunk_type: str) -> List[TextChunk]:
        """Get all text chunks of a specific type."""
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return []
            return graph.get_text_chunks_by_type(chunk_type)

        except Exception as e:
            logger.error("Error getting chunks by type: %s", e)
            return []

    def get_chunks_by_language(self, graph_id: str, language: str) -> List[TextChunk]:
        """Get all text chunks in a specific language."""
        try:
            conditions = {"graph_id": graph_id, "language": language}
            return self.query_text_chunks(conditions)

        except Exception as e:
            logger.error("Error getting chunks by language: %s", e)
            return []

    # Embedding-related implementations (basic file-based storage)

    def add_chunk_embedding(self, graph_id: str, chunk_id: str, embedding: List[float]) -> bool:
        """Add or update embedding for a text chunk."""
        try:
            chunk = self.get_text_chunk(graph_id, chunk_id)
            if not chunk:
                logger.error("Text chunk %s not found in graph %s", chunk_id, graph_id)
                return False

            chunk.embedding = embedding
            return self.update_text_chunk(graph_id, chunk)

        except Exception as e:
            logger.error("Error adding chunk embedding: %s", e)
            return False

    def get_chunk_embedding(self, graph_id: str, chunk_id: str) -> Optional[List[float]]:
        """Get embedding for a text chunk."""
        try:
            chunk = self.get_text_chunk(graph_id, chunk_id)
            if chunk:
                return chunk.embedding
            return None

        except Exception as e:
            logger.error("Error getting chunk embedding: %s", e)
            return None

    def search_chunks_by_embedding(
        self,
        graph_id: str,
        query_embedding: List[float],
        top_k: int = 10,
        threshold: float = 0.0,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
    ) -> List[tuple]:
        """Search text chunks by embedding similarity."""
        try:
            conditions = {"graph_id": graph_id}
            if chunk_type:
                conditions["chunk_type"] = chunk_type
            if language:
                conditions["language"] = language

            chunks = self.query_text_chunks(conditions)
            chunk_similarities = []

            for chunk in chunks:
                if chunk.embedding is not None:
                    similarity = self._compute_cosine_similarity(query_embedding, chunk.embedding)
                    if similarity >= threshold:
                        chunk_similarities.append((chunk, similarity))

            # Sort by similarity (descending)
            chunk_similarities.sort(key=lambda x: x[1], reverse=True)
            return chunk_similarities[:top_k]

        except Exception as e:
            logger.error("Error searching chunks by embedding: %s", e)
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
    ) -> List[tuple]:
        """Hybrid search combining text and embedding similarity."""
        try:
            # Validate weights
            if abs(text_weight + embedding_weight - 1.0) > 0.001:
                logger.warning("Text and embedding weights do not sum to 1.0, normalizing...")
                total_weight = text_weight + embedding_weight
                text_weight /= total_weight
                embedding_weight /= total_weight

            conditions = {"graph_id": graph_id}
            if chunk_type:
                conditions["chunk_type"] = chunk_type
            if language:
                conditions["language"] = language

            chunks = self.query_text_chunks(conditions)
            chunk_scores = []

            query_lower = query_text.lower()

            for chunk in chunks:
                total_score = 0.0

                # Text similarity score
                text_score = 0.0
                if query_lower in chunk.content.lower():
                    text_score = 0.8  # High score for exact match
                elif query_lower in chunk.title.lower():
                    text_score = 0.6  # Medium score for title match
                else:
                    # Simple word overlap scoring
                    query_words = set(query_lower.split())
                    content_words = set(chunk.content.lower().split())
                    if query_words and content_words:
                        overlap = len(query_words.intersection(content_words))
                        text_score = overlap / len(query_words)

                total_score += text_score * text_weight

                # Embedding similarity score
                if query_embedding is not None and chunk.embedding is not None:
                    embedding_score = self._compute_cosine_similarity(query_embedding, chunk.embedding)
                    total_score += embedding_score * embedding_weight

                if total_score > 0:
                    chunk_scores.append((chunk, total_score))

            # Sort by combined score (descending)
            chunk_scores.sort(key=lambda x: x[1], reverse=True)
            return chunk_scores[:top_k]

        except Exception as e:
            logger.error("Error in hybrid search: %s", e)
            return []

    def get_chunk_neighbors(self, graph_id: str, chunk_id: str, max_distance: int = 2) -> List[tuple]:
        """Get neighboring text chunks through entity/relation connections."""
        try:
            chunk = self.get_text_chunk(graph_id, chunk_id)
            if not chunk:
                return []

            graph = self.load_graph(graph_id)
            if not graph:
                return []

            neighbors = []
            visited_chunks = {chunk_id}

            # BFS to find neighbors within max_distance
            current_level = [(chunk, 0)]

            while current_level and any(distance < max_distance for _, distance in current_level):
                next_level = []

                for current_chunk, distance in current_level:
                    if distance < max_distance:
                        # Find chunks connected through shared entities
                        for entity_id in current_chunk.entities:
                            connected_chunks = graph.get_entity_text_chunks(entity_id)
                            for connected_chunk in connected_chunks:
                                if connected_chunk.id not in visited_chunks:
                                    neighbors.append((connected_chunk, distance + 1))
                                    next_level.append((connected_chunk, distance + 1))
                                    visited_chunks.add(connected_chunk.id)

                        # Find chunks connected through shared relations
                        for relation_id in current_chunk.relations:
                            connected_chunks = graph.get_relation_text_chunks(relation_id)
                            for connected_chunk in connected_chunks:
                                if connected_chunk.id not in visited_chunks:
                                    neighbors.append((connected_chunk, distance + 1))
                                    next_level.append((connected_chunk, distance + 1))
                                    visited_chunks.add(connected_chunk.id)

                current_level = next_level

            return neighbors

        except Exception as e:
            logger.error("Error getting chunk neighbors: %s", e)
            return []

    def _compute_cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            import numpy as np

            vec1_array = np.array(vec1, dtype=np.float32)
            vec2_array = np.array(vec2, dtype=np.float32)

            dot_product = np.dot(vec1_array, vec2_array)
            norm1 = np.linalg.norm(vec1_array)
            norm2 = np.linalg.norm(vec2_array)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(max(0.0, min(1.0, similarity)))

        except Exception as e:
            logger.error("Error computing cosine similarity: %s", e)
            return 0.0
