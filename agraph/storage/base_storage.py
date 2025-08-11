"""
Graph storage base class.

Provides abstract base class for graph storage implementations.
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, cast

from ..entities import Entity
from ..graph import KnowledgeGraph
from ..logger import logger
from ..relations import Relation
from ..text import TextChunk


class GraphStorage(ABC):
    """Abstract base class for graph storage implementations."""

    def __init__(self) -> None:
        """Initialize the graph storage."""
        self.connection = None
        self._is_connected = False

    def is_connected(self) -> bool:
        """
        Check if connected to storage backend.

        Returns:
            bool: True if connected
        """
        return self._is_connected

    def set_connected(self, value: bool) -> None:
        """
        Set connection status.

        Args:
            value: Connection status to set
        """
        self._is_connected = value

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to storage backend.

        Returns:
            bool: True if connection successful
        """

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from storage backend."""

    @abstractmethod
    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """
        Save knowledge graph to storage.

        Args:
            graph: Knowledge graph to save

        Returns:
            bool: True if save successful
        """

    @abstractmethod
    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """
        Load knowledge graph from storage.

        Args:
            graph_id: Graph identifier

        Returns:
            KnowledgeGraph: Loaded knowledge graph, None if not found
        """

    @abstractmethod
    def delete_graph(self, graph_id: str) -> bool:
        """
        Delete knowledge graph from storage.

        Args:
            graph_id: Graph identifier

        Returns:
            bool: True if delete successful
        """

    @abstractmethod
    def list_graphs(self) -> List[Dict[str, Any]]:
        """
        List all available graphs.

        Returns:
            List[Dict[str, Any]]: Graph metadata list
        """

    @abstractmethod
    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """
        Query entities based on conditions.

        Args:
            conditions: Query conditions

        Returns:
            List[Entity]: Matching entities
        """

    @abstractmethod
    def query_relations(
        self,
        head_entity: Optional[str] = None,
        tail_entity: Optional[str] = None,
        relation_type: Optional[Any] = None,
    ) -> List[Relation]:
        """
        Query relations based on conditions.

        Args:
            head_entity: Head entity ID
            tail_entity: Tail entity ID
            relation_type: Relation type

        Returns:
            List[Relation]: Matching relations
        """

    @abstractmethod
    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        Add entity to graph.

        Args:
            graph_id: Graph identifier
            entity: Entity object to add

        Returns:
            bool: True if add successful
        """

    @abstractmethod
    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        Add relation to graph.

        Args:
            graph_id: Graph identifier
            relation: Relation object to add

        Returns:
            bool: True if add successful
        """

    @abstractmethod
    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        Update entity in graph.

        Args:
            graph_id: Graph identifier
            entity: Entity object to update

        Returns:
            bool: True if update successful
        """

    @abstractmethod
    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        Update relation in graph.

        Args:
            graph_id: Graph identifier
            relation: Relation object to update

        Returns:
            bool: True if update successful
        """

    @abstractmethod
    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """
        Remove entity from graph.

        Args:
            graph_id: Graph identifier
            entity_id: Entity ID to remove

        Returns:
            bool: True if remove successful
        """

    @abstractmethod
    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """
        Remove relation from graph.

        Args:
            graph_id: Graph identifier
            relation_id: Relation ID to remove

        Returns:
            bool: True if remove successful
        """

    @abstractmethod
    def add_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """
        Add text chunk to graph.

        Args:
            graph_id: Graph identifier
            text_chunk: TextChunk object to add

        Returns:
            bool: True if add successful
        """

    @abstractmethod
    def update_text_chunk(self, graph_id: str, text_chunk: TextChunk) -> bool:
        """
        Update text chunk in graph.

        Args:
            graph_id: Graph identifier
            text_chunk: TextChunk object to update

        Returns:
            bool: True if update successful
        """

    @abstractmethod
    def remove_text_chunk(self, graph_id: str, chunk_id: str) -> bool:
        """
        Remove text chunk from graph.

        Args:
            graph_id: Graph identifier
            chunk_id: Text chunk ID to remove

        Returns:
            bool: True if remove successful
        """

    @abstractmethod
    def query_text_chunks(self, conditions: Dict[str, Any]) -> List[TextChunk]:
        """
        Query text chunks based on conditions.

        Args:
            conditions: Query conditions (e.g., source, chunk_type, entity_ids, etc.)

        Returns:
            List[TextChunk]: Matching text chunks
        """

    @abstractmethod
    def get_text_chunk(self, graph_id: str, chunk_id: str) -> Optional[TextChunk]:
        """
        Get text chunk by ID.

        Args:
            graph_id: Graph identifier
            chunk_id: Text chunk ID

        Returns:
            TextChunk: Text chunk object, None if not found
        """

    @abstractmethod
    def get_chunks_by_entity(self, graph_id: str, entity_id: str) -> List[TextChunk]:
        """
        Get all text chunks connected to a specific entity.

        Args:
            graph_id: Graph identifier
            entity_id: Entity ID

        Returns:
            List[TextChunk]: Text chunks connected to the entity
        """

    @abstractmethod
    def get_chunks_by_relation(self, graph_id: str, relation_id: str) -> List[TextChunk]:
        """
        Get all text chunks connected to a specific relation.

        Args:
            graph_id: Graph identifier
            relation_id: Relation ID

        Returns:
            List[TextChunk]: Text chunks connected to the relation
        """

    @abstractmethod
    def batch_add_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """
        Batch add text chunks to graph.

        Args:
            graph_id: Graph identifier
            text_chunks: List of TextChunk objects to add

        Returns:
            bool: True if batch add successful
        """

    @abstractmethod
    def batch_update_text_chunks(self, graph_id: str, text_chunks: List[TextChunk]) -> bool:
        """
        Batch update text chunks in graph.

        Args:
            graph_id: Graph identifier
            text_chunks: List of TextChunk objects to update

        Returns:
            bool: True if batch update successful
        """

    @abstractmethod
    def batch_remove_text_chunks(self, graph_id: str, chunk_ids: List[str]) -> bool:
        """
        Batch remove text chunks from graph.

        Args:
            graph_id: Graph identifier
            chunk_ids: List of text chunk IDs to remove

        Returns:
            bool: True if batch remove successful
        """

    @abstractmethod
    def get_chunks_by_source(self, graph_id: str, source: str) -> List[TextChunk]:
        """
        Get all text chunks from a specific source.

        Args:
            graph_id: Graph identifier
            source: Source document or origin

        Returns:
            List[TextChunk]: Text chunks from the specified source
        """

    @abstractmethod
    def get_chunks_by_type(self, graph_id: str, chunk_type: str) -> List[TextChunk]:
        """
        Get all text chunks of a specific type.

        Args:
            graph_id: Graph identifier
            chunk_type: Type of text chunks to retrieve

        Returns:
            List[TextChunk]: Text chunks of the specified type
        """

    @abstractmethod
    def get_chunks_by_language(self, graph_id: str, language: str) -> List[TextChunk]:
        """
        Get all text chunks in a specific language.

        Args:
            graph_id: Graph identifier
            language: Language code (e.g., 'zh', 'en')

        Returns:
            List[TextChunk]: Text chunks in the specified language
        """

    @abstractmethod
    def add_chunk_embedding(self, graph_id: str, chunk_id: str, embedding: List[float]) -> bool:
        """
        Add or update embedding for a text chunk.

        Args:
            graph_id: Graph identifier
            chunk_id: Text chunk ID
            embedding: Vector embedding of the text chunk

        Returns:
            bool: True if add/update successful
        """

    @abstractmethod
    def get_chunk_embedding(self, graph_id: str, chunk_id: str) -> Optional[List[float]]:
        """
        Get embedding for a text chunk.

        Args:
            graph_id: Graph identifier
            chunk_id: Text chunk ID

        Returns:
            Optional[List[float]]: Vector embedding, None if not found
        """

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
        Search text chunks by embedding similarity.

        Args:
            graph_id: Graph identifier
            query_embedding: Query vector embedding
            top_k: Maximum number of results to return
            threshold: Minimum similarity threshold
            chunk_type: Optional filter by chunk type
            language: Optional filter by language

        Returns:
            List[tuple[TextChunk, float]]: List of (chunk, similarity_score) tuples
        """

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
        Hybrid search combining text and embedding similarity.

        Args:
            graph_id: Graph identifier
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

    @abstractmethod
    def get_chunk_neighbors(self, graph_id: str, chunk_id: str, max_distance: int = 2) -> List[tuple[TextChunk, int]]:
        """
        Get neighboring text chunks through entity/relation connections.

        Args:
            graph_id: Graph identifier
            chunk_id: Text chunk ID
            max_distance: Maximum connection distance

        Returns:
            List[tuple[TextChunk, int]]: List of (chunk, distance) tuples
        """

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get graph statistics.

        Args:
            graph_id: Graph identifier

        Returns:
            Dict[str, Any]: Statistics information
        """
        try:
            graph = self.load_graph(graph_id)
            if graph:
                stats = graph.get_basic_statistics()
                # Add text chunk statistics if available
                try:
                    text_chunks = self.query_text_chunks({"graph_id": graph_id})
                    stats["text_chunks"] = len(text_chunks)
                    if text_chunks:
                        total_length = sum(chunk.get_text_length() for chunk in text_chunks)
                        stats["total_text_length"] = total_length
                        stats["avg_chunk_length"] = total_length / len(text_chunks)

                        # Statistics by chunk type
                        chunk_types: Dict[str, int] = {}
                        for chunk in text_chunks:
                            chunk_types[chunk.chunk_type] = chunk_types.get(chunk.chunk_type, 0) + 1
                        stats["chunk_types"] = chunk_types

                        # Statistics by language
                        languages: Dict[str, int] = {}
                        for chunk in text_chunks:
                            languages[chunk.language] = languages.get(chunk.language, 0) + 1
                        stats["languages"] = languages

                except Exception:
                    # If text chunk queries are not supported, skip silently
                    pass

                return stats
            return {}
        except Exception as e:
            logger.error("Error getting graph statistics: %s", e)
            return {}

    def backup_graph(self, graph_id: str, backup_path: str) -> bool:
        """
        Backup graph to file.

        Args:
            graph_id: Graph identifier
            backup_path: Backup file path

        Returns:
            bool: True if backup successful
        """
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return False

            # Subclasses can override this method for specific backup logic
            graph_data = graph.to_dict()

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(graph_data, f, ensure_ascii=False, indent=2)

            logger.info("Graph %s backed up to %s", graph_id, backup_path)
            return True

        except Exception as e:
            logger.error("Error backing up graph %s: %s", graph_id, e)
            return False

    def restore_graph(self, backup_path: str) -> Optional[str]:
        """
        Restore graph from backup file.

        Args:
            backup_path: Backup file path

        Returns:
            str: Restored graph ID, None if failed
        """
        try:

            with open(backup_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)

            graph = KnowledgeGraph.from_dict(graph_data)

            if self.save_graph(graph):
                logger.info("Graph restored from %s with ID %s", backup_path, graph.id)
                return graph.id

            return None

        except Exception as e:
            logger.error("Error restoring graph from %s: %s", backup_path, e)
            return None

    def export_graph(self, graph_id: str, outformat: str = "json") -> Optional[Dict[str, Any]]:
        """
        Export graph data in specified format.

        Args:
            graph_id: Graph identifier
            outformat: Export format ('json', 'csv', 'graphml', etc.)

        Returns:
            Dict[str, Any]: Exported data, None if failed
        """
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return None

            if outformat.lower() == "json":
                return graph.to_dict()
            if outformat.lower() == "csv":
                return self._export_to_csv_format(graph)
            if outformat.lower() == "graphml":
                return self._export_to_graphml_format(graph)
            logger.warning("Unsupported export outformat: %s", outformat)
            return None

        except Exception as e:
            logger.error("Error exporting graph %s: %s", graph_id, e)
            return None

    def _export_to_csv_format(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Export graph data to CSV format."""
        from ..utils import get_type_value

        entities_data = []
        for entity in graph.entities.values():
            entities_data.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": get_type_value(entity.entity_type),
                    "description": entity.description,
                    "confidence": entity.confidence,
                    "source": entity.source,
                }
            )

        relations_data = []
        for relation in graph.relations.values():
            if relation.head_entity is None or relation.tail_entity is None:
                continue
            relations_data.append(
                {
                    "id": relation.id,
                    "head_entity": relation.head_entity.name,
                    "tail_entity": relation.tail_entity.name,
                    "relation_type": get_type_value(relation.relation_type),
                    "confidence": relation.confidence,
                    "source": relation.source,
                }
            )

        # Add text chunks data if available
        text_chunks_data = []
        try:
            text_chunks = self.query_text_chunks({"graph_id": graph.id})
            for chunk in text_chunks:
                text_chunks_data.append(
                    {
                        "id": chunk.id,
                        "title": chunk.title,
                        "content": chunk.get_summary(),  # Use summary for CSV
                        "source": chunk.source,
                        "chunk_type": chunk.chunk_type,
                        "language": chunk.language,
                        "confidence": chunk.confidence,
                        "text_length": chunk.get_text_length(),
                        "entity_count": len(chunk.entities),
                        "relation_count": len(chunk.relations),
                    }
                )
        except Exception:
            # If text chunk queries are not supported, skip silently
            pass

        result = {"entities": entities_data, "relations": relations_data}
        if text_chunks_data:
            result["text_chunks"] = text_chunks_data

        return result

    def _export_to_graphml_format(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Export graph data to GraphML format."""
        from ..utils import get_type_value

        # Simplified GraphML format
        nodes = []
        edges = []

        for entity in graph.entities.values():
            nodes.append(
                {
                    "id": entity.id,
                    "label": entity.name,
                    "type": get_type_value(entity.entity_type),
                    "description": entity.description,
                }
            )

        for relation in graph.relations.values():
            if relation.head_entity is None or relation.tail_entity is None:
                continue
            edges.append(
                {
                    "id": relation.id,
                    "source": relation.head_entity.id,
                    "target": relation.tail_entity.id,
                    "label": get_type_value(relation.relation_type),
                    "confidence": relation.confidence,
                }
            )

        return {"graph": {"nodes": nodes, "edges": edges}}

    def search_text_chunks(
        self,
        graph_id: str,
        query: str,
        chunk_type: Optional[str] = None,
        language: Optional[str] = None,
        limit: int = 10,
    ) -> List[TextChunk]:
        """
        Search text chunks by content or metadata.

        Args:
            graph_id: Graph identifier
            query: Search query string
            chunk_type: Optional chunk type filter
            language: Optional language filter
            limit: Maximum number of results

        Returns:
            List[TextChunk]: Matching text chunks
        """
        conditions = {"graph_id": graph_id}
        if chunk_type:
            conditions["chunk_type"] = chunk_type
        if language:
            conditions["language"] = language

        try:
            all_chunks = self.query_text_chunks(conditions)

            # Simple text search in content and title
            matching_chunks = []
            query_lower = query.lower()

            for chunk in all_chunks:
                if query_lower in chunk.content.lower() or query_lower in chunk.title.lower():
                    matching_chunks.append(chunk)

                if len(matching_chunks) >= limit:
                    break

            return matching_chunks

        except Exception as e:
            logger.error("Error searching text chunks: %s", e)
            return []

    def get_chunk_context(self, graph_id: str, chunk_id: str) -> Dict[str, Any]:
        """
        Get contextual information for a text chunk including connected entities and relations.

        Args:
            graph_id: Graph identifier
            chunk_id: Text chunk ID

        Returns:
            Dict[str, Any]: Context information including entities, relations, and similar chunks
        """
        try:
            chunk = self.get_text_chunk(graph_id, chunk_id)
            if not chunk:
                return {}

            context: Dict[str, Any] = {"chunk": chunk.to_dict(), "entities": [], "relations": [], "similar_chunks": []}

            # Get connected entities
            if chunk.entities:
                try:
                    entity_conditions = {"graph_id": graph_id, "ids": list(chunk.entities)}
                    entities = self.query_entities(entity_conditions)
                    context["entities"] = [entity.to_dict() for entity in entities]
                except Exception:
                    pass

            # Get connected relations
            if chunk.relations:
                try:
                    relations = self.query_relations()
                    matching_relations = [r for r in relations if r.id in chunk.relations]
                    context["relations"] = [relation.to_dict() for relation in matching_relations]
                except Exception:
                    pass

            # Find similar chunks (chunks that share entities or relations)
            try:
                all_chunks = self.query_text_chunks({"graph_id": graph_id})
                similar_chunks = []

                for other_chunk in all_chunks:
                    if other_chunk.id != chunk_id:
                        similarity = chunk.calculate_similarity(other_chunk)
                        if similarity > 0.1:  # Threshold for similarity
                            similar_chunks.append({"chunk": other_chunk.to_dict(), "similarity": similarity})

                # Sort by similarity and take top 5
                similar_chunks.sort(key=lambda x: cast(float, x["similarity"]), reverse=True)
                context["similar_chunks"] = similar_chunks[:5]

            except Exception:
                pass

            return context

        except Exception as e:
            logger.error("Error getting chunk context: %s", e)
            return {}

    def analyze_chunk_similarity_distribution(self, graph_id: str, chunk_id: str) -> Dict[str, Any]:
        """
        Analyze similarity distribution for a text chunk.

        Args:
            graph_id: Graph identifier
            chunk_id: Text chunk ID

        Returns:
            Dict[str, Any]: Similarity distribution analysis
        """
        try:
            target_chunk = self.get_text_chunk(graph_id, chunk_id)
            if not target_chunk:
                return {}

            all_chunks = self.query_text_chunks({"graph_id": graph_id})
            similarities = []

            for chunk in all_chunks:
                if chunk.id != chunk_id:
                    similarity = target_chunk.calculate_similarity(chunk)
                    similarities.append(similarity)

            if not similarities:
                return {"chunk_id": chunk_id, "analysis": "No other chunks to compare"}

            similarities.sort(reverse=True)

            return {
                "chunk_id": chunk_id,
                "total_chunks": len(similarities),
                "max_similarity": max(similarities),
                "min_similarity": min(similarities),
                "avg_similarity": sum(similarities) / len(similarities),
                "top_10_similarities": similarities[:10],
                "highly_similar_count": sum(1 for s in similarities if s > 0.7),
                "moderately_similar_count": sum(1 for s in similarities if 0.3 <= s <= 0.7),
                "low_similar_count": sum(1 for s in similarities if s < 0.3),
            }

        except Exception as e:
            logger.error("Error analyzing chunk similarity distribution: %s", e)
            return {}

    def get_chunk_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Get comprehensive statistics about text chunks in the graph.

        Args:
            graph_id: Graph identifier

        Returns:
            Dict[str, Any]: Comprehensive chunk statistics
        """
        try:
            chunks = self.query_text_chunks({"graph_id": graph_id})
            if not chunks:
                return {"total_chunks": 0}

            # Basic statistics
            total_chunks = len(chunks)
            total_content_length = sum(chunk.get_text_length() for chunk in chunks)
            avg_content_length = total_content_length / total_chunks if total_chunks > 0 else 0

            # Statistics by type
            type_stats: Dict[str, Dict[str, Any]] = {}
            for chunk in chunks:
                chunk_type = chunk.chunk_type
                if chunk_type not in type_stats:
                    type_stats[chunk_type] = {"count": 0, "total_length": 0}
                type_stats[chunk_type]["count"] += 1
                type_stats[chunk_type]["total_length"] += chunk.get_text_length()

            # Calculate averages for each type
            for type_name, stats in type_stats.items():
                stats["avg_length"] = stats["total_length"] / stats["count"]

            # Statistics by language
            language_stats: Dict[str, int] = {}
            for chunk in chunks:
                language = chunk.language
                language_stats[language] = language_stats.get(language, 0) + 1

            # Statistics by source
            source_stats: Dict[str, int] = {}
            for chunk in chunks:
                source = chunk.source if chunk.source else "unknown"
                source_stats[source] = source_stats.get(source, 0) + 1

            # Entity and relation connection statistics
            chunks_with_entities = sum(1 for chunk in chunks if chunk.entities)
            chunks_with_relations = sum(1 for chunk in chunks if chunk.relations)
            avg_entities_per_chunk = sum(len(chunk.entities) for chunk in chunks) / total_chunks
            avg_relations_per_chunk = sum(len(chunk.relations) for chunk in chunks) / total_chunks

            # Confidence distribution
            confidences = [chunk.confidence for chunk in chunks]
            confidence_stats = {
                "avg_confidence": sum(confidences) / len(confidences),
                "max_confidence": max(confidences),
                "min_confidence": min(confidences),
                "high_confidence_count": sum(1 for c in confidences if c >= 0.8),
                "medium_confidence_count": sum(1 for c in confidences if 0.5 <= c < 0.8),
                "low_confidence_count": sum(1 for c in confidences if c < 0.5),
            }

            # Embedding statistics
            chunks_with_embeddings = sum(1 for chunk in chunks if chunk.embedding is not None)
            embedding_coverage = chunks_with_embeddings / total_chunks if total_chunks > 0 else 0

            return {
                "total_chunks": total_chunks,
                "total_content_length": total_content_length,
                "avg_content_length": avg_content_length,
                "type_statistics": type_stats,
                "language_statistics": language_stats,
                "source_statistics": source_stats,
                "connection_statistics": {
                    "chunks_with_entities": chunks_with_entities,
                    "chunks_with_relations": chunks_with_relations,
                    "avg_entities_per_chunk": avg_entities_per_chunk,
                    "avg_relations_per_chunk": avg_relations_per_chunk,
                },
                "confidence_statistics": confidence_stats,
                "embedding_statistics": {
                    "chunks_with_embeddings": chunks_with_embeddings,
                    "embedding_coverage": embedding_coverage,
                },
            }

        except Exception as e:
            logger.error("Error getting chunk statistics: %s", e)
            return {}

    def find_orphaned_chunks(self, graph_id: str) -> List[TextChunk]:
        """
        Find text chunks that have no connections to entities or relations.

        Args:
            graph_id: Graph identifier

        Returns:
            List[TextChunk]: Chunks with no entity or relation connections
        """
        try:
            chunks = self.query_text_chunks({"graph_id": graph_id})
            orphaned_chunks = []

            for chunk in chunks:
                if not chunk.entities and not chunk.relations:
                    orphaned_chunks.append(chunk)

            return orphaned_chunks

        except Exception as e:
            logger.error("Error finding orphaned chunks: %s", e)
            return []

    def find_highly_connected_chunks(self, graph_id: str, min_connections: int = 5) -> List[tuple[TextChunk, int]]:
        """
        Find text chunks with many entity/relation connections.

        Args:
            graph_id: Graph identifier
            min_connections: Minimum number of connections to be considered highly connected

        Returns:
            List[tuple[TextChunk, int]]: List of (chunk, connection_count) tuples
        """
        try:
            chunks = self.query_text_chunks({"graph_id": graph_id})
            highly_connected = []

            for chunk in chunks:
                connection_count = len(chunk.entities) + len(chunk.relations)
                if connection_count >= min_connections:
                    highly_connected.append((chunk, connection_count))

            # Sort by connection count (descending)
            highly_connected.sort(key=lambda x: x[1], reverse=True)
            return highly_connected

        except Exception as e:
            logger.error("Error finding highly connected chunks: %s", e)
            return []

    def validate_chunk_connections(self, graph_id: str) -> Dict[str, Any]:
        """
        Validate that all chunk connections point to existing entities and relations.

        Args:
            graph_id: Graph identifier

        Returns:
            Dict[str, Any]: Validation results including any broken connections
        """
        try:
            chunks = self.query_text_chunks({"graph_id": graph_id})
            graph = self.load_graph(graph_id)

            if not graph:
                return {"error": "Graph not found"}

            validation_results: Dict[str, Any] = {
                "total_chunks": len(chunks),
                "chunks_with_broken_entity_refs": [],
                "chunks_with_broken_relation_refs": [],
                "broken_entity_refs": 0,
                "broken_relation_refs": 0,
                "valid_chunks": 0,
            }

            for chunk in chunks:
                chunk_has_issues = False

                # Check entity references
                for entity_id in chunk.entities:
                    if entity_id not in graph.entities:
                        validation_results["chunks_with_broken_entity_refs"].append(
                            {"chunk_id": chunk.id, "broken_entity_id": entity_id}
                        )
                        validation_results["broken_entity_refs"] += 1
                        chunk_has_issues = True

                # Check relation references
                for relation_id in chunk.relations:
                    if relation_id not in graph.relations:
                        validation_results["chunks_with_broken_relation_refs"].append(
                            {"chunk_id": chunk.id, "broken_relation_id": relation_id}
                        )
                        validation_results["broken_relation_refs"] += 1
                        chunk_has_issues = True

                if not chunk_has_issues:
                    validation_results["valid_chunks"] += 1

            return validation_results

        except Exception as e:
            logger.error("Error validating chunk connections: %s", e)
            return {"error": str(e)}

    def get_chunk_coverage_by_entities(self, graph_id: str) -> Dict[str, Any]:
        """
        Get coverage statistics showing how entities are represented in text chunks.

        Args:
            graph_id: Graph identifier

        Returns:
            Dict[str, Any]: Coverage statistics for entities in text chunks
        """
        try:
            graph = self.load_graph(graph_id)
            chunks = self.query_text_chunks({"graph_id": graph_id})

            if not graph or not chunks:
                return {}

            entity_coverage = {}
            total_entities = len(graph.entities)

            # Count how many chunks mention each entity
            for entity_id, entity in graph.entities.items():
                chunk_count = sum(1 for chunk in chunks if entity_id in chunk.entities)
                entity_coverage[entity_id] = {
                    "entity_name": entity.name,
                    "chunk_count": chunk_count,
                    "coverage_percentage": (chunk_count / len(chunks)) * 100 if chunks else 0,
                }

            # Overall statistics
            entities_in_chunks = sum(
                1 for entity_id in graph.entities if any(entity_id in chunk.entities for chunk in chunks)
            )

            coverage_stats = {
                "total_entities": total_entities,
                "entities_mentioned_in_chunks": entities_in_chunks,
                "entity_coverage_percentage": (entities_in_chunks / total_entities) * 100 if total_entities > 0 else 0,
                "entity_details": entity_coverage,
            }

            return coverage_stats

        except Exception as e:
            logger.error("Error getting chunk coverage by entities: %s", e)
            return {}

    def cluster_chunks_by_similarity(self, graph_id: str, threshold: float = 0.5) -> List[List[str]]:
        """
        Cluster text chunks based on similarity (shared entities and relations).

        Args:
            graph_id: Graph identifier
            threshold: Similarity threshold for clustering

        Returns:
            List[List[str]]: List of clusters, each containing chunk IDs
        """
        try:
            chunks = self.query_text_chunks({"graph_id": graph_id})

            if len(chunks) < 2:
                return [[chunk.id] for chunk in chunks]

            # Build similarity matrix
            chunk_list = list(chunks)
            n = len(chunk_list)
            clusters = []
            used_chunks = set()

            for i in range(n):
                if chunk_list[i].id in used_chunks:
                    continue

                cluster = [chunk_list[i].id]
                used_chunks.add(chunk_list[i].id)

                for j in range(i + 1, n):
                    if chunk_list[j].id in used_chunks:
                        continue

                    similarity = chunk_list[i].calculate_similarity(chunk_list[j])
                    if similarity >= threshold:
                        cluster.append(chunk_list[j].id)
                        used_chunks.add(chunk_list[j].id)

                clusters.append(cluster)

            return clusters

        except Exception as e:
            logger.error("Error clustering chunks by similarity: %s", e)
            return []
