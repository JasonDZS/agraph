"""Knowledge retriever module.

Dedicated to knowledge graph retrieval functionality, separated from the building process.
"""

import asyncio
import json
import logging
import os.path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from openai import AsyncOpenAI

from ..config import settings
from ..embeddings import GraphEmbedding, OpenAIEmbedding
from ..graph import KnowledgeGraph
from ..storage import JsonVectorStorage, VectorStorage
from ..utils import get_type_value
from .base import RetrievalEntity, RetrievalRelation, RetrievalResult

logger = logging.getLogger(__name__)


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
            logger.info("search similar vectors: %s", similar_vectors)

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

    async def search_knowledge(
        self, query: str, top_k_entities: int = 5, top_k_relations: int = 5, similarity_threshold: float = 0.5
    ) -> RetrievalResult:
        """Comprehensive knowledge graph search.

        Args:
            query: Query text.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            similarity_threshold: Similarity threshold.

        Returns:
            RetrievalResult: Search results containing entities and relations.
        """
        result = RetrievalResult()
        if not self.vector_storage:
            logger.warning("No vector storage configured for search")
            return result

        try:
            # Search entities and relations in parallel
            entity_task = self.search_entities(query, top_k_entities, similarity_threshold)
            relation_task = self.search_relations(query, top_k_relations, similarity_threshold)

            entities, relations = await asyncio.gather(entity_task, relation_task)

            # result = {"entities": entities, "relations": relations}
            result.entities = entities
            result.relations = relations
            result.success = True

            # Track comprehensive search statistics
            total_results = len(entities) + len(relations)
            self._track_search("knowledge_search", query, total_results, True)

            return result

        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            self._track_search("knowledge_search", query, 0, False, str(e))
            result.success = False
            result.msg = str(e)
            return result

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
        self, query: str, entities: List[RetrievalEntity], relations: List[RetrievalRelation], history_messages: list
    ) -> str | None:
        """Generate chat response using LLM.

        Args:
            query: User query.
            entities: List of retrieved entities.
            relations: List of retrieved relations.
            history_messages: Chat history message list.

        Returns:
            str | None: LLM generated response, or None if generation fails.
        """
        # This can call actual LLM API for chat response generation
        # For example, using OpenAI's API
        # This is just an example, actual implementation needs to be adjusted according to specific LLM API
        result = await self.llm.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": settings.RAG_SYS_PROMPT.format(
                        history=history_messages,
                        kg_context=json.dumps(
                            {
                                "entities": [entity.entity.to_dict() for entity in entities],
                                "relations": [relation.relation.to_dict() for relation in relations],
                            },
                            ensure_ascii=False,
                        ),
                        response_type="text",
                    ),
                },
                {"role": "user", "content": query},
            ],
            model=settings.LLM_MODEL,
        )
        result_txt = result.choices[0].message.content
        return result_txt if isinstance(result_txt, str) else None

    async def achat(
        self, query: str, top_k_entities: int = 5, top_k_relations: int = 5, history_messages: list = []
    ) -> Any:
        """Knowledge-based chat (async version).

        Args:
            query: User query.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            history_messages: Chat history message list.

        Returns:
            Dict[str, Any]: Search results containing entities and relations.
        """
        try:
            # Perform knowledge retrieval
            results = await self.search_knowledge(query, top_k_entities, top_k_relations)
            logger.info(
                "retrieved entities and relations successfully, %s entities, %s relations",
                len(results.entities),
                len(results.relations),
            )
            answer = await self.chat_llm(
                query, entities=results.entities, relations=results.relations, history_messages=history_messages
            )

            # Generate chat response
            response = {
                "query": query,
                "entities": results.entities,
                "relations": results.relations,
                "timestamp": datetime.now().isoformat(),
                "answer": answer,
            }

            # Track chat statistics
            self._track_search("chat", query, len(results.entities) + len(results.relations), True)

            return response

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            self._track_search("chat", query, 0, False, str(e))
            return {"error": str(e)}

    def chat(
        self, query: str, top_k_entities: int = 5, top_k_relations: int = 5, history_messages: list = []
    ) -> Dict[str, Any]:
        """Synchronous chat method.

        Args:
            query: User query.
            top_k_entities: Number of entities to return.
            top_k_relations: Number of relations to return.
            history_messages: Chat history message list.

        Returns:
            Dict[str, Any]: Search results containing entities and relations.
        """
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.achat(query, top_k_entities, top_k_relations, history_messages))
        except Exception as e:
            logger.warning("No event loop running, using asyncio.run(), %s", e)
            # If no event loop exists, use asyncio.run()
            return asyncio.run(self.achat(query, top_k_entities, top_k_relations, history_messages))
