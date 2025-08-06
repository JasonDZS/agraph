"""
知识检索器模块

专门负责知识图谱的检索功能，与构建过程分离
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from ..embeddings import GraphEmbedding
from ..graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """知识检索器 - 专门负责知识图谱的检索功能"""

    def __init__(
        self,
        graph: KnowledgeGraph,
        graph_embedding: Optional[GraphEmbedding] = None,
    ):
        """
        初始化知识检索器

        Args:
            graph: 知识图谱
            graph_embedding: 图嵌入实例，用于向量检索
        """
        self.graph = graph
        self.graph_embedding = graph_embedding

        # 初始化检索统计
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
    ) -> List[Tuple[str, float]]:
        """
        基于embedding搜索相似实体

        Args:
            query: 查询文本
            top_k: 返回数量
            similarity_threshold: 相似度阈值

        Returns:
            List[Tuple[str, float]]: 实体ID和相似度列表
        """
        if not self.graph_embedding:
            logger.warning("No graph embedding configured for search")
            return []

        try:
            # 使用文本嵌入进行相似度搜索
            query_embedding = await self.graph_embedding.embed_text(query)
            if query_embedding is None:
                return []

            # 使用向量存储搜索相似实体
            similar_vectors = self.graph_embedding.vector_storage.search_similar_vectors(
                query_embedding, top_k + 10, similarity_threshold  # 多取一些，再过滤
            )

            # 过滤出实体向量，确保实体存在于图中
            filtered_results = []
            for vector_id, similarity in similar_vectors:
                if vector_id.startswith("entity_"):
                    entity_id = vector_id[7:]  # 移除"entity_"前缀
                    if entity_id in self.graph.entities:
                        filtered_results.append((entity_id, similarity))
                        if len(filtered_results) >= top_k:
                            break

            # 跟踪搜索统计
            self._track_search("entity_search", query, len(filtered_results), True)

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching entities: {e}")
            self._track_search("entity_search", query, 0, False, str(e))
            return []

    async def search_relations(
        self, query: str, top_k: int = 10, similarity_threshold: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        基于embedding搜索相似关系

        Args:
            query: 查询文本
            top_k: 返回数量
            similarity_threshold: 相似度阈值

        Returns:
            List[Tuple[str, float]]: 关系ID和相似度列表
        """
        if not self.graph_embedding:
            logger.warning("No graph embedding configured for search")
            return []

        try:
            # 使用文本嵌入进行相似度搜索
            query_embedding = await self.graph_embedding.embed_text(query)
            if query_embedding is None:
                return []

            # 使用向量存储搜索相似关系
            similar_vectors = self.graph_embedding.vector_storage.search_similar_vectors(
                query_embedding, top_k + 10, similarity_threshold  # 多取一些，再过滤
            )

            # 过滤出关系向量，确保关系存在于图中
            filtered_results = []
            for vector_id, similarity in similar_vectors:
                if vector_id.startswith("relation_"):
                    relation_id = vector_id[9:]  # 移除"relation_"前缀
                    if relation_id in self.graph.relations:
                        filtered_results.append((relation_id, similarity))
                        if len(filtered_results) >= top_k:
                            break

            # 跟踪搜索统计
            self._track_search("relation_search", query, len(filtered_results), True)

            return filtered_results

        except Exception as e:
            logger.error(f"Error searching relations: {e}")
            self._track_search("relation_search", query, 0, False, str(e))
            return []

    async def search_knowledge(
        self, query: str, top_k_entities: int = 5, top_k_relations: int = 5, similarity_threshold: float = 0.5
    ) -> Dict[str, List[Tuple[str, float]]]:
        """
        综合搜索知识图谱

        Args:
            query: 查询文本
            top_k_entities: 返回实体数量
            top_k_relations: 返回关系数量
            similarity_threshold: 相似度阈值

        Returns:
            Dict: 包含entities和relations的搜索结果
        """
        if not self.graph_embedding:
            logger.warning("No graph embedding configured for search")
            return {"entities": [], "relations": []}

        try:
            # 并行搜索实体和关系
            entity_task = self.search_entities(query, top_k_entities, similarity_threshold)
            relation_task = self.search_relations(query, top_k_relations, similarity_threshold)

            entities, relations = await asyncio.gather(entity_task, relation_task)

            result = {"entities": entities, "relations": relations}

            # 跟踪综合搜索统计
            total_results = len(entities) + len(relations)
            self._track_search("knowledge_search", query, total_results, True)

            return result

        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            self._track_search("knowledge_search", query, 0, False, str(e))
            return {"entities": [], "relations": []}

    def search_entities_by_type(self, entity_type: str, top_k: int = 10) -> List[Tuple[str, Any]]:
        """
        按类型搜索实体

        Args:
            entity_type: 实体类型
            top_k: 返回数量

        Returns:
            List[Tuple[str, Any]]: 实体ID和实体对象列表
        """
        try:
            results = []
            for entity_id, entity in self.graph.entities.items():
                if entity.entity_type.value == entity_type:
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
        """
        按类型搜索关系

        Args:
            relation_type: 关系类型
            top_k: 返回数量

        Returns:
            List[Tuple[str, Any]]: 关系ID和关系对象列表
        """
        try:
            results = []
            for relation_id, relation in self.graph.relations.items():
                if relation.relation_type.value == relation_type:
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
        """
        获取实体的邻居节点

        Args:
            entity_id: 实体ID
            max_hops: 最大跳数

        Returns:
            Dict[str, List[str]]: 邻居实体按跳数分组
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
                    # 查找通过关系连接的邻居
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
        """
        查找两个实体之间的最短路径

        Args:
            start_entity_id: 起始实体ID
            end_entity_id: 终止实体ID
            max_length: 最大路径长度

        Returns:
            Optional[List[str]]: 路径上的实体ID列表，如果没有路径则返回None
        """
        try:
            if start_entity_id not in self.graph.entities or end_entity_id not in self.graph.entities:
                return None

            if start_entity_id == end_entity_id:
                return [start_entity_id]

            # BFS查找最短路径
            from collections import deque

            queue = deque([(start_entity_id, [start_entity_id])])
            visited = {start_entity_id}

            while queue:
                current_id, path = queue.popleft()

                if len(path) > max_length:
                    continue

                # 获取当前实体的邻居
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
        """
        更新底层知识图谱

        Args:
            new_graph: 新的知识图谱
        """
        self.graph = new_graph
        self.retrieval_stats["last_updated"] = datetime.now()
        logger.info("Knowledge retriever updated with new graph")

    def get_retrieval_statistics(self) -> Dict[str, Any]:
        """
        获取检索统计信息

        Returns:
            Dict[str, Any]: 检索统计信息
        """
        return self.retrieval_stats.copy()

    def reset_statistics(self) -> None:
        """重置检索统计信息"""
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
        """
        跟踪搜索统计

        Args:
            search_type: 搜索类型
            query: 查询内容
            result_count: 结果数量
            success: 是否成功
            error_msg: 错误信息
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

        # 记录搜索历史
        search_record = {
            "timestamp": datetime.now(),
            "search_type": search_type,
            "query": query,
            "result_count": result_count,
            "success": success,
            "error_msg": error_msg,
        }
        self.retrieval_stats["search_history"].append(search_record)

        # 限制历史记录数量
        if len(self.retrieval_stats["search_history"]) > 1000:
            self.retrieval_stats["search_history"] = self.retrieval_stats["search_history"][-1000:]
