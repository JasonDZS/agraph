"""Knowledge retriever module.

Dedicated to knowledge graph retrieval functionality, separated from the building process.
"""

import asyncio
import json
import os.path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ..config import settings
from ..embeddings import GraphEmbedding, OpenAIEmbedding
from ..graph import KnowledgeGraph
from ..logger import logger
from ..storage import JsonVectorStorage, VectorStorage
from ..text import TextChunk
from ..utils import get_type_value
from .base import RetrievalEntity, RetrievalRelation, RetrievalResult, RetrievalTextChunk


class KnowledgeRetriever:
    """Knowledge retriever - dedicated to knowledge graph retrieval functionality."""

    def __init__(
        self,
        graph: Optional[KnowledgeGraph] = None,
        graph_embedding: Optional[GraphEmbedding] = None,
        vector_storage: Optional[VectorStorage] = None,
    ):
        """Initialize knowledge retriever.

        Args:
            graph: Knowledge graph instance.
            graph_embedding: Graph embedding instance for vector retrieval
                (mutually exclusive with vector_storage).
            vector_storage: Vector storage instance for direct vector retrieval.
        """
        if graph is not None:
            self.graph = graph
        else:
            with open(os.path.join(settings.workdir, "graph.json"), "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            # Load graph data from JSON string
            self.graph = KnowledgeGraph().from_dict(graph_data)

        if vector_storage is not None:
            self.vector_storage = vector_storage
        else:
            self.vector_storage = JsonVectorStorage(os.path.join(settings.workdir, "vectors.json"))

        if graph_embedding is not None:
            self.graph_embedding = graph_embedding
        else:
            self.graph_embedding = OpenAIEmbedding(
                vector_storage=self.vector_storage,
                openai_api_key=settings.OPENAI_API_KEY,
                openai_api_base=settings.OPENAI_API_BASE,
                embedding_model=settings.EMBEDDING_MODEL,
            )

        # Use direct vector_storage if provided, otherwise get from graph_embedding
        if not self.vector_storage and self.graph_embedding:
            self.vector_storage = self.graph_embedding.vector_storage

        # Initialize retrieval statistics
        self.retrieval_stats: Dict[str, Any] = {
            "total_searches": 0,
            "entity_searches": 0,
            "relation_searches": 0,
            "text_chunk_searches": 0,
            "knowledge_searches": 0,
            "errors": 0,
            "search_history": [],
            "last_updated": datetime.now(),
        }

    async def search_entities(
        self, query: str, top_k: int = 10, similarity_threshold: float = 0.5
    ) -> List[RetrievalEntity]:
        """Search for similar entities based on embedding.

        Args:
            query: Query text.
            top_k: Number of results to return.
            similarity_threshold: Similarity threshold.

        Returns:
            List[RetrievalEntity]: List of entities with similarity scores.
        """
        if not self.vector_storage:
            logger.warning("No vector storage configured for search")
            return []

        try:
            # Use text embedding for similarity search
            if self.graph_embedding:
                query_embedding = await self.graph_embedding.embed_text(query)
            else:
                logger.warning("No graph embedding available, cannot embed query text")
                return []

            if query_embedding is None:
                return []

            # Use vector storage to search for similar entities
            similar_vectors = self.vector_storage.search_similar_vectors(
                query_embedding, top_k + 10, similarity_threshold  # Get more, then filter
            )
            logger.info(f"search similar vectors: {similar_vectors}")

            # Filter entity vectors, ensure entities exist in the graph
            filtered_results = []
            for vector_id, similarity in similar_vectors:
                if vector_id.startswith("entity_"):
                    entity_id = vector_id[len("entity_") :]  # Remove prefix
                    if self.graph and entity_id in self.graph.entities:
                        entity = self.graph.get_entity(entity_id)
                        if entity is not None:
                            filtered_results.append(
                                RetrievalEntity(
                                    entity=entity,
                                    score=similarity,
                                )
                            )
                        if len(filtered_results) >= top_k:
                            break

            # Track search statistics
            self._track_search("entity_search", query, len(filtered_results), True)

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            self._track_search("entity_search", query, 0, False, str(e))
            return []

    async def search_relations(
        self, query: str, top_k: int = 10, similarity_threshold: float = 0.5
    ) -> List[RetrievalRelation]:
        """Search for similar relations based on embedding.

        Args:
            query: Query text.
            top_k: Number of results to return.
            similarity_threshold: Similarity threshold.

        Returns:
            List[RetrievalRelation]: List of relations with similarity scores.
        """
        if not self.vector_storage:
            logger.warning("No vector storage configured for search")
            return []

        try:
            # Use text embedding for similarity search
            if self.graph_embedding:
                query_embedding = await self.graph_embedding.embed_text(query)
            else:
                logger.warning("No graph embedding available, cannot embed query text")
                return []

            if query_embedding is None:
                return []

            # Use vector storage to search for similar relations
            similar_vectors = self.vector_storage.search_similar_vectors(
                query_embedding, top_k + 10, similarity_threshold  # Get more, then filter
            )

            # Filter relation vectors, ensure relations exist in the graph
            filtered_results = []
            for vector_id, similarity in similar_vectors:
                if vector_id.startswith("relation_"):
                    relation_id = vector_id[len("relation_") :]  # Remove prefix
                    # Ensure relation exists in the graph
                    if self.graph and relation_id in self.graph.relations:
                        relation = self.graph.get_relation(relation_id)
                        if relation is not None:
                            filtered_results.append(
                                RetrievalRelation(
                                    relation=relation,
                                    score=similarity,
                                )
                            )
                        if len(filtered_results) >= top_k:
                            break

            # Track search statistics
            self._track_search("relation_search", query, len(filtered_results), True)

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching relations: {e}")
            self._track_search("relation_search", query, 0, False, str(e))
            return []

    async def search_text_chunks(
        self, query: str, top_k: int = 10, similarity_threshold: float = 0.5
    ) -> List[RetrievalTextChunk]:
        """Search for similar text chunks based on embedding.

        Args:
            query: Query text.
            top_k: Number of results to return.
            similarity_threshold: Similarity threshold.

        Returns:
            List[RetrievalTextChunk]: List of text chunks with similarity scores.
        """
        if not self.vector_storage:
            logger.warning("No vector storage configured for search")
            return []

        try:
            # Use text embedding for similarity search
            if self.graph_embedding:
                query_embedding = await self.graph_embedding.embed_text(query)
            else:
                logger.warning("No graph embedding available, cannot embed query text")
                return []

            if query_embedding is None:
                return []

            # Use vector storage to search for similar text chunks
            similar_vectors = self.vector_storage.search_similar_vectors(
                query_embedding, top_k + 10, similarity_threshold  # Get more, then filter
            )
            logger.info(f"search similar vectors: {similar_vectors}")

            # Filter text chunk vectors, ensure text chunks exist in the graph
            filtered_results = []
            for vector_id, similarity in similar_vectors:
                if vector_id.startswith("text_chunk_"):
                    chunk_id = vector_id[len("text_chunk_") :]  # Remove prefix
                    # Ensure text chunk exists in the graph
                    if self.graph and chunk_id in self.graph.text_chunks:
                        text_chunk = self.graph.get_text_chunk(chunk_id)
                        if text_chunk is not None:
                            filtered_results.append(
                                RetrievalTextChunk(
                                    text_chunk=text_chunk,
                                    score=similarity,
                                )
                            )
                        if len(filtered_results) >= top_k:
                            break

            # Track search statistics
            self._track_search("text_chunk_search", query, len(filtered_results), True)

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching text chunks: {e}")
            self._track_search("text_chunk_search", query, 0, False, str(e))
            return []

    async def search_knowledge(
        self,
        query: str,
        top_k_entities: int = 5,
        top_k_relations: int = 5,
        top_k_text_chunks: int = 5,
        similarity_threshold: float = 0.5,
    ) -> RetrievalResult:
        """Comprehensive knowledge graph search.

        Args:
            query: Query text.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            top_k_text_chunks: Number of text chunks to return.
            similarity_threshold: Similarity threshold.

        Returns:
            RetrievalResult: Search results containing entities, relations, and text chunks.
        """
        result = RetrievalResult()
        if not self.vector_storage:
            logger.warning("No vector storage configured for search")
            return result

        try:
            # Search entities, relations, and text chunks in parallel
            entity_task = self.search_entities(query, top_k_entities, similarity_threshold)
            relation_task = self.search_relations(query, top_k_relations, similarity_threshold)
            text_chunk_task = self.search_text_chunks(query, top_k_text_chunks, similarity_threshold)

            entities, relations, text_chunks = await asyncio.gather(entity_task, relation_task, text_chunk_task)

            result.entities = entities
            result.relations = relations
            result.text_chunks = text_chunks
            result.success = True

            # Track comprehensive search statistics
            total_results = len(entities) + len(relations) + len(text_chunks)
            self._track_search("knowledge_search", query, total_results, True)

            return result

        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            self._track_search("knowledge_search", query, 0, False, str(e))
            result.success = False
            result.msg = str(e)
            return result

    async def search_knowledge_enhanced(
        self,
        query: str,
        top_k_entities: int = 5,
        top_k_relations: int = 5,
        top_k_text_chunks: int = 5,
        similarity_threshold: float = 0.5,
        include_connected_chunks: bool = True,
    ) -> RetrievalResult:
        """Enhanced comprehensive knowledge graph search that includes connected text chunks.

        Args:
            query: Query text.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            top_k_text_chunks: Number of text chunks to return.
            similarity_threshold: Similarity threshold.
            include_connected_chunks: Whether to include text chunks connected to found entities/relations.

        Returns:
            RetrievalResult: Enhanced search results with connected information.
        """
        result = RetrievalResult()
        if not self.vector_storage:
            logger.warning("No vector storage configured for search")
            return result

        try:
            # First, perform basic knowledge search
            basic_result = await self.search_knowledge(
                query, top_k_entities, top_k_relations, top_k_text_chunks, similarity_threshold
            )

            if not basic_result.success:
                return basic_result

            result.entities = basic_result.entities
            result.relations = basic_result.relations
            result.text_chunks = basic_result.text_chunks

            if include_connected_chunks:
                # Find additional text chunks connected to retrieved entities and relations
                connected_chunks = set()

                # Get text chunks connected to retrieved entities
                for entity_retrieval in result.entities:
                    entity_chunks = self.get_text_chunks_by_entity(entity_retrieval.entity.id)
                    for chunk in entity_chunks:
                        connected_chunks.add(chunk)

                # Get text chunks connected to retrieved relations
                for relation_retrieval in result.relations:
                    relation_chunks = self.get_text_chunks_by_relation(relation_retrieval.relation.id)
                    for chunk in relation_chunks:
                        connected_chunks.add(chunk)

                # Add connected chunks that aren't already in results
                existing_chunk_ids = {chunk.text_chunk.id for chunk in result.text_chunks}
                additional_chunks = []

                for chunk in connected_chunks:
                    if chunk.id not in existing_chunk_ids:
                        # Calculate relevance score based on connection strength
                        connection_score = self._calculate_chunk_relevance_score(
                            chunk, result.entities, result.relations
                        )
                        additional_chunks.append(RetrievalTextChunk(text_chunk=chunk, score=connection_score))

                # Sort additional chunks by relevance and add top ones
                additional_chunks.sort(key=lambda x: x.score, reverse=True)
                max_additional = max(0, top_k_text_chunks - len(result.text_chunks))
                result.text_chunks.extend(additional_chunks[:max_additional])

            result.success = True
            total_results = len(result.entities) + len(result.relations) + len(result.text_chunks)
            self._track_search("enhanced_knowledge_search", query, total_results, True)

            return result

        except Exception as e:
            logger.error(f"Error in enhanced knowledge search: {e}")
            self._track_search("enhanced_knowledge_search", query, 0, False, str(e))
            result.success = False
            result.msg = str(e)
            return result

    def _calculate_chunk_relevance_score(
        self, text_chunk: TextChunk, entities: List[RetrievalEntity], relations: List[RetrievalRelation]
    ) -> float:
        """Calculate relevance score for a text chunk based on connected entities and relations.

        Args:
            text_chunk: The text chunk to score.
            entities: List of retrieved entities.
            relations: List of retrieved relations.

        Returns:
            float: Relevance score between 0.0 and 1.0.
        """
        try:
            entity_connections = 0
            relation_connections = 0
            total_entity_score = 0.0
            total_relation_score = 0.0

            # Count connections to retrieved entities
            for entity_retrieval in entities:
                if text_chunk.has_entity(entity_retrieval.entity.id):
                    entity_connections += 1
                    total_entity_score += entity_retrieval.score

            # Count connections to retrieved relations
            for relation_retrieval in relations:
                if text_chunk.has_relation(relation_retrieval.relation.id):
                    relation_connections += 1
                    total_relation_score += relation_retrieval.score

            # Calculate weighted score
            if entity_connections + relation_connections == 0:
                return 0.0

            # Average scores weighted by connection counts
            avg_entity_score = total_entity_score / entity_connections if entity_connections > 0 else 0.0
            avg_relation_score = total_relation_score / relation_connections if relation_connections > 0 else 0.0

            # Combine scores with weights favoring more connections
            entity_weight = entity_connections / (entity_connections + relation_connections)
            relation_weight = relation_connections / (entity_connections + relation_connections)

            final_score = avg_entity_score * entity_weight + avg_relation_score * relation_weight

            # Boost score based on number of connections
            connection_boost = min(1.0, (entity_connections + relation_connections) / 5.0)

            return min(1.0, final_score * (0.5 + 0.5 * connection_boost))

        except Exception as e:
            logger.error(f"Error calculating chunk relevance score: {e}")
            return 0.0

    def search_entities_by_type(self, entity_type: str, top_k: int = 10) -> List[Tuple[str, Any]]:
        """Search entities by type.

        Args:
            entity_type: Entity type to search for.
            top_k: Number of results to return.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing entity ID and entity object.
        """
        try:
            results = []
            for entity_id, entity in self.graph.entities.items():
                if get_type_value(entity.entity_type) == entity_type:
                    results.append((entity_id, entity))
                    if len(results) >= top_k:
                        break

            self._track_search("type_search", f"type:{entity_type}", len(results), True)
            return results

        except Exception as e:
            logger.error(f"Error searching entities by type: {e}")
            self._track_search("type_search", f"type:{entity_type}", 0, False, str(e))
            return []

    def search_relations_by_type(self, relation_type: str, top_k: int = 10) -> List[Tuple[str, Any]]:
        """Search relations by type.

        Args:
            relation_type: Relation type to search for.
            top_k: Number of results to return.

        Returns:
            List[Tuple[str, Any]]: List of tuples containing relation ID and relation object.
        """
        try:
            results = []
            for relation_id, relation in self.graph.relations.items():
                if get_type_value(relation.relation_type) == relation_type:
                    results.append((relation_id, relation))
                    if len(results) >= top_k:
                        break

            self._track_search("relation_type_search", f"type:{relation_type}", len(results), True)
            return results

        except Exception as e:
            logger.error(f"Error searching relations by type: {e}")
            self._track_search("relation_type_search", f"type:{relation_type}", 0, False, str(e))
            return []

    def search_text_chunks_by_type(self, chunk_type: str, top_k: int = 10) -> List[Tuple[str, TextChunk]]:
        """Search text chunks by type.

        Args:
            chunk_type: Text chunk type to search for.
            top_k: Number of results to return.

        Returns:
            List[Tuple[str, TextChunk]]: List of tuples containing chunk ID and chunk object.
        """
        try:
            results = []
            for chunk_id, text_chunk in self.graph.text_chunks.items():
                if text_chunk.chunk_type == chunk_type:
                    results.append((chunk_id, text_chunk))
                    if len(results) >= top_k:
                        break

            self._track_search("text_chunk_type_search", f"type:{chunk_type}", len(results), True)
            return results

        except Exception as e:
            logger.error(f"Error searching text chunks by type: {e}")
            self._track_search("text_chunk_type_search", f"type:{chunk_type}", 0, False, str(e))
            return []

    def search_text_chunks_by_source(self, source: str, top_k: int = 10) -> List[Tuple[str, TextChunk]]:
        """Search text chunks by source.

        Args:
            source: Source to search for.
            top_k: Number of results to return.

        Returns:
            List[Tuple[str, TextChunk]]: List of tuples containing chunk ID and chunk object.
        """
        try:
            results = []
            for chunk_id, text_chunk in self.graph.text_chunks.items():
                if text_chunk.source == source:
                    results.append((chunk_id, text_chunk))
                    if len(results) >= top_k:
                        break

            self._track_search("text_chunk_source_search", f"source:{source}", len(results), True)
            return results

        except Exception as e:
            logger.error(f"Error searching text chunks by source: {e}")
            self._track_search("text_chunk_source_search", f"source:{source}", 0, False, str(e))
            return []

    def get_text_chunks_by_entity(self, entity_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific entity.

        Args:
            entity_id: Entity ID to find connected text chunks for.

        Returns:
            List[TextChunk]: List of text chunks connected to the entity.
        """
        try:
            if entity_id not in self.graph.entities:
                return []

            connected_chunks = []
            for text_chunk in self.graph.text_chunks.values():
                if text_chunk.has_entity(entity_id):
                    connected_chunks.append(text_chunk)

            self._track_search("entity_chunks_search", f"entity:{entity_id}", len(connected_chunks), True)
            return connected_chunks

        except Exception as e:
            logger.error(f"Error getting text chunks by entity: {e}")
            self._track_search("entity_chunks_search", f"entity:{entity_id}", 0, False, str(e))
            return []

    def get_text_chunks_by_relation(self, relation_id: str) -> List[TextChunk]:
        """Get all text chunks connected to a specific relation.

        Args:
            relation_id: Relation ID to find connected text chunks for.

        Returns:
            List[TextChunk]: List of text chunks connected to the relation.
        """
        try:
            if relation_id not in self.graph.relations:
                return []

            connected_chunks = []
            for text_chunk in self.graph.text_chunks.values():
                if text_chunk.has_relation(relation_id):
                    connected_chunks.append(text_chunk)

            self._track_search("relation_chunks_search", f"relation:{relation_id}", len(connected_chunks), True)
            return connected_chunks

        except Exception as e:
            logger.error(f"Error getting text chunks by relation: {e}")
            self._track_search("relation_chunks_search", f"relation:{relation_id}", 0, False, str(e))
            return []

    def get_entity_neighbors(self, entity_id: str, max_hops: int = 1) -> Dict[str, List[str]]:
        """Get neighbor nodes of an entity.

        Args:
            entity_id: Entity ID to find neighbors for.
            max_hops: Maximum number of hops to traverse.

        Returns:
            Dict[str, List[str]]: Neighbor entities grouped by hop distance.
        """
        try:
            if entity_id not in self.graph.entities:
                return {}

            neighbors: Dict[str, List[str]] = {f"hop_{i}": [] for i in range(1, max_hops + 1)}
            visited = {entity_id}
            current_level = {entity_id}

            for hop in range(1, max_hops + 1):
                next_level = set()

                for current_entity_id in current_level:
                    # Find neighbors connected through relations
                    for relation in self.graph.relations.values():
                        neighbor_id = None

                        if (
                            relation.head_entity
                            and relation.head_entity.id == current_entity_id
                            and relation.tail_entity
                        ):
                            neighbor_id = relation.tail_entity.id
                        elif (
                            relation.tail_entity
                            and relation.tail_entity.id == current_entity_id
                            and relation.head_entity
                        ):
                            neighbor_id = relation.head_entity.id

                        if neighbor_id and neighbor_id not in visited:
                            neighbors[f"hop_{hop}"].append(neighbor_id)
                            next_level.add(neighbor_id)
                            visited.add(neighbor_id)

                current_level = next_level
                if not current_level:
                    break

            self._track_search("neighbor_search", f"entity:{entity_id}", sum(len(v) for v in neighbors.values()), True)
            return neighbors

        except Exception as e:
            logger.error(f"Error getting entity neighbors: {e}")
            self._track_search("neighbor_search", f"entity:{entity_id}", 0, False, str(e))
            return {}

    def get_shortest_path(self, start_entity_id: str, end_entity_id: str, max_length: int = 5) -> Optional[List[str]]:
        """Find the shortest path between two entities.

        Args:
            start_entity_id: Starting entity ID.
            end_entity_id: Target entity ID.
            max_length: Maximum path length to consider.

        Returns:
            Optional[List[str]]: List of entity IDs in the path, or None if no path exists.
        """
        try:
            if start_entity_id not in self.graph.entities or end_entity_id not in self.graph.entities:
                return None

            if start_entity_id == end_entity_id:
                return [start_entity_id]

            # BFS to find shortest path
            from collections import deque

            queue = deque([(start_entity_id, [start_entity_id])])
            visited = {start_entity_id}

            while queue:
                current_id, path = queue.popleft()

                if len(path) > max_length:
                    continue

                # Get neighbors of current entity
                neighbors = self.get_entity_neighbors(current_id, max_hops=1)
                for neighbor_id in neighbors.get("hop_1", []):
                    if neighbor_id == end_entity_id:
                        result_path = path + [neighbor_id]
                        self._track_search("path_search", f"{start_entity_id}->{end_entity_id}", len(result_path), True)
                        return result_path

                    if neighbor_id not in visited:
                        visited.add(neighbor_id)
                        queue.append((neighbor_id, path + [neighbor_id]))

            self._track_search("path_search", f"{start_entity_id}->{end_entity_id}", 0, True)
            return None

        except Exception as e:
            logger.error(f"Error finding shortest path: {e}")
            self._track_search("path_search", f"{start_entity_id}->{end_entity_id}", 0, False, str(e))
            return None

    def update_graph(self, new_graph: KnowledgeGraph) -> None:
        """Update the underlying knowledge graph.

        Args:
            new_graph: New knowledge graph instance.
        """
        self.graph = new_graph
        self.retrieval_stats["last_updated"] = datetime.now()
        logger.info("Knowledge retriever updated with new graph")

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """Get retrieval statistics.

        Returns:
            Dict[str, Any]: Retrieval statistics information.
        """
        return self.retrieval_stats.copy()

    def reset_statistics(self) -> None:
        """Reset retrieval statistics."""
        self.retrieval_stats.update(
            {
                "total_searches": 0,
                "entity_searches": 0,
                "relation_searches": 0,
                "text_chunk_searches": 0,
                "knowledge_searches": 0,
                "errors": 0,
                "search_history": [],
                "last_updated": datetime.now(),
            }
        )
        logger.info("Retrieval statistics reset")

    def _track_search(
        self, search_type: str, query: str, result_count: int, success: bool, error_msg: Optional[str] = None
    ) -> None:
        """Track search statistics.

        Args:
            search_type: Type of search operation.
            query: Query content.
            result_count: Number of results returned.
            success: Whether the search was successful.
            error_msg: Error message if any.
        """
        self.retrieval_stats["total_searches"] += 1

        if success:
            if search_type == "entity_search":
                self.retrieval_stats["entity_searches"] += 1
            elif search_type == "relation_search":
                self.retrieval_stats["relation_searches"] += 1
            elif search_type == "text_chunk_search":
                self.retrieval_stats["text_chunk_searches"] += 1
            elif search_type == "knowledge_search":
                self.retrieval_stats["knowledge_searches"] += 1
        else:
            self.retrieval_stats["errors"] += 1

        # Record search history
        search_record = {
            "timestamp": datetime.now(),
            "search_type": search_type,
            "query": query,
            "result_count": result_count,
            "success": success,
            "error_msg": error_msg,
        }
        self.retrieval_stats["search_history"].append(search_record)

        # Limit history record count
        if len(self.retrieval_stats["search_history"]) > 1000:
            self.retrieval_stats["search_history"] = self.retrieval_stats["search_history"][-1000:]


class ChatKnowledgeRetriever(KnowledgeRetriever):
    """Knowledge-based chat retriever for conversational interfaces."""

    llm = AsyncOpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE)

    async def chat_llm(
        self,
        query: str,
        entities: List[RetrievalEntity],
        relations: List[RetrievalRelation],
        text_chunks: List[RetrievalTextChunk],
        history_messages: list,
    ) -> str | None:
        """Generate chat response using LLM.

        Args:
            query: User query.
            entities: List of retrieved entities.
            relations: List of retrieved relations.
            text_chunks: List of retrieved text chunks.
            history_messages: Chat history message list.

        Returns:
            str | None: LLM generated response, or None if generation fails.
        """
        # Prepare context information
        kg_context = {
            "entities": [entity.entity.to_dict() for entity in entities],
            "relations": [relation.relation.to_dict() for relation in relations],
            "text_chunks": [chunk.text_chunk.to_dict() for chunk in text_chunks],
        }

        # Extract text content for better context
        text_content = "\n".join(
            [f"文本片段 {i+1}: {chunk.text_chunk.content[:200]}..." for i, chunk in enumerate(text_chunks)]
        )

        # Enhanced system prompt with text chunk information
        enhanced_prompt = settings.RAG_SYS_PROMPT.format(
            history=history_messages,
            kg_context=json.dumps(kg_context, ensure_ascii=False),
            response_type="text",
        )

        if text_chunks:
            enhanced_prompt += f"\n\n相关文本内容:\n{text_content}"

        result = await self.llm.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": enhanced_prompt,
                },
                {"role": "user", "content": query},
            ],
            model=settings.LLM_MODEL,
        )
        result_txt = result.choices[0].message.content
        return result_txt if isinstance(result_txt, str) else None

    async def achat(
        self,
        query: str,
        top_k_entities: int = 5,
        top_k_relations: int = 5,
        top_k_text_chunks: int = 5,
        history_messages: list = [],
    ) -> Any:
        """Knowledge-based chat (async version).

        Args:
            query: User query.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            top_k_text_chunks: Number of text chunks to return.
            history_messages: Chat history message list.

        Returns:
            Dict[str, Any]: Search results containing entities, relations, text chunks and answer.
        """
        try:
            # Perform knowledge retrieval
            results = await self.search_knowledge(query, top_k_entities, top_k_relations, top_k_text_chunks)
            logger.info(
                f"retrieved knowledge successfully: {len(results.entities)} entities, "
                f"{len(results.relations)} relations, {len(results.text_chunks)} text chunks"
            )

            answer = await self.chat_llm(
                query,
                entities=results.entities,
                relations=results.relations,
                text_chunks=results.text_chunks,
                history_messages=history_messages,
            )

            # Generate chat response
            response = {
                "query": query,
                "entities": results.entities,
                "relations": results.relations,
                "text_chunks": results.text_chunks,
                "timestamp": datetime.now().isoformat(),
                "answer": answer,
            }

            # Track chat statistics
            total_results = len(results.entities) + len(results.relations) + len(results.text_chunks)
            self._track_search("chat", query, total_results, True)

            return response

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            self._track_search("chat", query, 0, False, str(e))
            return {"error": str(e)}

    def chat(
        self,
        query: str,
        top_k_entities: int = 5,
        top_k_relations: int = 5,
        top_k_text_chunks: int = 5,
        history_messages: list = [],
    ) -> Dict[str, Any]:
        """Synchronous chat method.

        Args:
            query: User query.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            top_k_text_chunks: Number of text chunks to return.
            history_messages: Chat history message list.

        Returns:
            Dict[str, Any]: Search results containing entities, relations, text chunks and answer.
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.achat(query, top_k_entities, top_k_relations, top_k_text_chunks, history_messages)
            )
        except Exception as e:
            logger.warning(f"No event loop running, using asyncio.run(), {e}")
            # If no event loop exists, use asyncio.run()
            return asyncio.run(self.achat(query, top_k_entities, top_k_relations, top_k_text_chunks, history_messages))

    async def achat_enhanced(
        self,
        query: str,
        top_k_entities: int = 5,
        top_k_relations: int = 5,
        top_k_text_chunks: int = 5,
        history_messages: list = [],
        include_connected_chunks: bool = True,
    ) -> Any:
        """Enhanced knowledge-based chat using enhanced retrieval.

        Args:
            query: User query.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            top_k_text_chunks: Number of text chunks to return.
            history_messages: Chat history message list.
            include_connected_chunks: Whether to include connected text chunks.

        Returns:
            Dict[str, Any]: Enhanced search results with comprehensive context.
        """
        try:
            # Perform enhanced knowledge retrieval
            results = await self.search_knowledge_enhanced(
                query,
                top_k_entities,
                top_k_relations,
                top_k_text_chunks,
                include_connected_chunks=include_connected_chunks,
            )
            logger.info(
                f"enhanced retrieval completed: {len(results.entities)} entities, "
                f"{len(results.relations)} relations, {len(results.text_chunks)} text chunks"
            )

            answer = await self.chat_llm(
                query,
                entities=results.entities,
                relations=results.relations,
                text_chunks=results.text_chunks,
                history_messages=history_messages,
            )

            # Generate enhanced chat response
            response = {
                "query": query,
                "entities": results.entities,
                "relations": results.relations,
                "text_chunks": results.text_chunks,
                "timestamp": datetime.now().isoformat(),
                "answer": answer,
                "enhanced_search": True,
                "total_context_items": len(results.entities) + len(results.relations) + len(results.text_chunks),
            }

            # Track enhanced chat statistics
            total_results = len(results.entities) + len(results.relations) + len(results.text_chunks)
            self._track_search("enhanced_chat", query, total_results, True)

            return response

        except Exception as e:
            logger.error(f"Error in enhanced chat: {e}")
            self._track_search("enhanced_chat", query, 0, False, str(e))
            return {"error": str(e)}

    def chat_enhanced(
        self,
        query: str,
        top_k_entities: int = 5,
        top_k_relations: int = 5,
        top_k_text_chunks: int = 5,
        history_messages: list = [],
        include_connected_chunks: bool = True,
    ) -> Dict[str, Any]:
        """Synchronous enhanced chat method.

        Args:
            query: User query.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            top_k_text_chunks: Number of text chunks to return.
            history_messages: Chat history message list.
            include_connected_chunks: Whether to include connected text chunks.

        Returns:
            Dict[str, Any]: Enhanced search results with comprehensive context.
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.achat_enhanced(
                    query,
                    top_k_entities,
                    top_k_relations,
                    top_k_text_chunks,
                    history_messages,
                    include_connected_chunks,
                )
            )
        except Exception as e:
            logger.warning(f"No event loop running, using asyncio.run(), {e}")
            # If no event loop exists, use asyncio.run()
            return asyncio.run(
                self.achat_enhanced(
                    query,
                    top_k_entities,
                    top_k_relations,
                    top_k_text_chunks,
                    history_messages,
                    include_connected_chunks,
                )
            )
