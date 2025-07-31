"""
图存储基类
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..entities import Entity
from ..graph import KnowledgeGraph
from ..relations import Relation
from ..types import RelationType

logger = logging.getLogger(__name__)


class GraphStorage(ABC):
    """图存储基类"""

    def __init__(self) -> None:
        self.connection = None
        self.is_connected = False

    @abstractmethod
    def connect(self) -> bool:
        """
        连接到存储后端

        Returns:
            bool: 连接是否成功
        """

    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""

    @abstractmethod
    def save_graph(self, graph: KnowledgeGraph) -> bool:
        """
        保存知识图谱

        Args:
            graph: 知识图谱

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def load_graph(self, graph_id: str) -> Optional[KnowledgeGraph]:
        """
        加载知识图谱

        Args:
            graph_id: 图谱ID

        Returns:
            KnowledgeGraph: 加载的知识图谱，如果不存在则返回None
        """

    @abstractmethod
    def delete_graph(self, graph_id: str) -> bool:
        """
        删除知识图谱

        Args:
            graph_id: 图谱ID

        Returns:
            bool: 删除是否成功
        """

    @abstractmethod
    def list_graphs(self) -> List[Dict[str, Any]]:
        """
        列出所有图谱

        Returns:
            List[Dict[str, Any]]: 图谱信息列表
        """

    @abstractmethod
    def query_entities(self, conditions: Dict[str, Any]) -> List[Entity]:
        """
        查询实体

        Args:
            conditions: 查询条件

        Returns:
            List[Entity]: 查询结果
        """

    @abstractmethod
    def query_relations(
        self,
        head_entity: Optional[str] = None,
        tail_entity: Optional[str] = None,
        relation_type: Optional[RelationType] = None,
    ) -> List[Relation]:
        """
        查询关系

        Args:
            head_entity: 头实体ID
            tail_entity: 尾实体ID
            relation_type: 关系类型

        Returns:
            List[Relation]: 查询结果
        """

    @abstractmethod
    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        添加实体

        Args:
            graph_id: 图谱ID
            entity: 实体对象

        Returns:
            bool: 添加是否成功
        """

    @abstractmethod
    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        添加关系

        Args:
            graph_id: 图谱ID
            relation: 关系对象

        Returns:
            bool: 添加是否成功
        """

    @abstractmethod
    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        """
        更新实体

        Args:
            graph_id: 图谱ID
            entity: 实体对象

        Returns:
            bool: 更新是否成功
        """

    @abstractmethod
    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        """
        更新关系

        Args:
            graph_id: 图谱ID
            relation: 关系对象

        Returns:
            bool: 更新是否成功
        """

    @abstractmethod
    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        """
        删除实体

        Args:
            graph_id: 图谱ID
            entity_id: 实体ID

        Returns:
            bool: 删除是否成功
        """

    @abstractmethod
    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        """
        删除关系

        Args:
            graph_id: 图谱ID
            relation_id: 关系ID

        Returns:
            bool: 删除是否成功
        """

    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        获取图谱统计信息

        Args:
            graph_id: 图谱ID

        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            graph = self.load_graph(graph_id)
            if graph:
                return graph.get_statistics()
            return {}
        except Exception as e:
            logger.error("Error getting graph statistics: %s", e)
            return {}

    def backup_graph(self, graph_id: str, backup_path: str) -> bool:
        """
        备份图谱

        Args:
            graph_id: 图谱ID
            backup_path: 备份路径

        Returns:
            bool: 备份是否成功
        """
        try:
            graph = self.load_graph(graph_id)
            if not graph:
                return False

            # 子类可以重写此方法实现具体的备份逻辑
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
        从备份恢复图谱

        Args:
            backup_path: 备份文件路径

        Returns:
            str: 恢复的图谱ID，失败返回None
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
        导出图谱数据

        Args:
            graph_id: 图谱ID
            outformat: 导出格式 ('json', 'csv', 'graphml', etc.)

        Returns:
            Dict[str, Any]: 导出的数据
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
        """导出为CSV格式"""
        entities_data = []
        for entity in graph.entities.values():
            entities_data.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "type": entity.entity_type.value,
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
                    "relation_type": relation.relation_type.value,
                    "confidence": relation.confidence,
                    "source": relation.source,
                }
            )

        return {"entities": entities_data, "relations": relations_data}

    def _export_to_graphml_format(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """导出为GraphML格式"""
        # 简化的GraphML格式
        nodes = []
        edges = []

        for entity in graph.entities.values():
            nodes.append(
                {
                    "id": entity.id,
                    "label": entity.name,
                    "type": entity.entity_type.value,
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
                    "label": relation.relation_type.value,
                    "confidence": relation.confidence,
                }
            )

        return {"graph": {"nodes": nodes, "edges": edges}}
