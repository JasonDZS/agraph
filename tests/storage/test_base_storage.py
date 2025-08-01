import json
import os
import tempfile
import pytest
from unittest.mock import patch, mock_open, MagicMock
from datetime import datetime

from agraph.storage.base_storage import GraphStorage
from agraph.entities import Entity
from agraph.relations import Relation
from agraph.graph import KnowledgeGraph
from agraph.types import EntityType, RelationType


class MockGraphStorage(GraphStorage):
    """用于测试的模拟存储实现"""

    def __init__(self):
        super().__init__()
        self.graphs = {}
        self.entities = {}
        self.relations = {}

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> None:
        self.is_connected = False

    def save_graph(self, graph: KnowledgeGraph) -> bool:
        if not self.is_connected:
            return False
        self.graphs[graph.id] = graph
        return True

    def load_graph(self, graph_id: str) -> KnowledgeGraph:
        if not self.is_connected:
            return None
        return self.graphs.get(graph_id)

    def delete_graph(self, graph_id: str) -> bool:
        if not self.is_connected:
            return False
        if graph_id in self.graphs:
            del self.graphs[graph_id]
            return True
        return False

    def list_graphs(self) -> list:
        if not self.is_connected:
            return []
        return [{"id": gid, "name": graph.name} for gid, graph in self.graphs.items()]

    def query_entities(self, conditions: dict) -> list:
        if not self.is_connected:
            return []
        # 简化实现
        return list(self.entities.values())

    def query_relations(self, head_entity=None, tail_entity=None, relation_type=None) -> list:
        if not self.is_connected:
            return []
        # 简化实现
        return list(self.relations.values())

    def add_entity(self, graph_id: str, entity: Entity) -> bool:
        if not self.is_connected:
            return False
        self.entities[entity.id] = entity
        return True

    def add_relation(self, graph_id: str, relation: Relation) -> bool:
        if not self.is_connected:
            return False
        self.relations[relation.id] = relation
        return True

    def update_entity(self, graph_id: str, entity: Entity) -> bool:
        if not self.is_connected:
            return False
        self.entities[entity.id] = entity
        return True

    def update_relation(self, graph_id: str, relation: Relation) -> bool:
        if not self.is_connected:
            return False
        self.relations[relation.id] = relation
        return True

    def remove_entity(self, graph_id: str, entity_id: str) -> bool:
        if not self.is_connected:
            return False
        if entity_id in self.entities:
            del self.entities[entity_id]
            return True
        return False

    def remove_relation(self, graph_id: str, relation_id: str) -> bool:
        if not self.is_connected:
            return False
        if relation_id in self.relations:
            del self.relations[relation_id]
            return True
        return False


class TestGraphStorage:
    """测试基础存储类"""

    def setup_method(self):
        """每个测试方法前的设置"""
        self.storage = MockGraphStorage()

    def test_init(self):
        """测试初始化"""
        # 使用MockGraphStorage来测试基类的初始化
        storage = MockGraphStorage()
        assert storage.connection is None
        assert storage.is_connected is False

    def test_abstract_methods(self):
        """测试抽象方法"""
        # 直接实例化抽象类应该失败
        with pytest.raises(TypeError):
            GraphStorage()

    def test_connection_management(self):
        """测试连接管理"""
        # 初始状态
        assert not self.storage.is_connected

        # 连接
        result = self.storage.connect()
        assert result is True
        assert self.storage.is_connected

        # 断开连接
        self.storage.disconnect()
        assert not self.storage.is_connected

    def test_graph_operations(self):
        """测试图谱操作"""
        self.storage.connect()

        # 创建测试图谱
        graph = KnowledgeGraph(id="test_graph", name="Test Graph")

        # 保存图谱
        result = self.storage.save_graph(graph)
        assert result is True

        # 加载图谱
        loaded_graph = self.storage.load_graph("test_graph")
        assert loaded_graph is not None
        assert loaded_graph.id == "test_graph"
        assert loaded_graph.name == "Test Graph"

        # 列出图谱
        graphs = self.storage.list_graphs()
        assert len(graphs) == 1
        assert graphs[0]["id"] == "test_graph"

        # 删除图谱
        result = self.storage.delete_graph("test_graph")
        assert result is True

        # 验证删除
        loaded_graph = self.storage.load_graph("test_graph")
        assert loaded_graph is None

    def test_entity_operations(self):
        """测试实体操作"""
        self.storage.connect()

        # 创建测试实体
        entity = Entity(
            id="entity_1",
            name="Test Entity",
            entity_type=EntityType.PERSON,
            description="A test person"
        )

        # 添加实体
        result = self.storage.add_entity("test_graph", entity)
        assert result is True

        # 查询实体
        entities = self.storage.query_entities({"graph_id": "test_graph"})
        assert len(entities) >= 1

        # 更新实体
        entity.description = "Updated description"
        result = self.storage.update_entity("test_graph", entity)
        assert result is True

        # 删除实体
        result = self.storage.remove_entity("test_graph", "entity_1")
        assert result is True

    def test_relation_operations(self):
        """测试关系操作"""
        self.storage.connect()

        # 创建测试实体
        entity1 = Entity(id="entity_1", name="Entity 1", entity_type=EntityType.PERSON)
        entity2 = Entity(id="entity_2", name="Entity 2", entity_type=EntityType.ORGANIZATION)

        # 创建测试关系
        relation = Relation(
            id="relation_1",
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.BELONGS_TO
        )

        # 添加关系
        result = self.storage.add_relation("test_graph", relation)
        assert result is True

        # 查询关系
        relations = self.storage.query_relations(head_entity="entity_1")
        assert len(relations) >= 1

        # 更新关系
        relation.confidence = 0.8
        result = self.storage.update_relation("test_graph", relation)
        assert result is True

        # 删除关系
        result = self.storage.remove_relation("test_graph", "relation_1")
        assert result is True

    def test_operations_without_connection(self):
        """测试未连接时的操作"""
        # 未连接状态下所有操作都应该失败
        graph = KnowledgeGraph(id="test", name="test")
        entity = Entity(id="test", name="test")
        relation = Relation(id="test")

        assert self.storage.save_graph(graph) is False
        assert self.storage.load_graph("test") is None
        assert self.storage.delete_graph("test") is False
        assert self.storage.list_graphs() == []
        assert self.storage.query_entities({}) == []
        assert self.storage.query_relations() == []
        assert self.storage.add_entity("test", entity) is False
        assert self.storage.add_relation("test", relation) is False
        assert self.storage.update_entity("test", entity) is False
        assert self.storage.update_relation("test", relation) is False
        assert self.storage.remove_entity("test", "test") is False
        assert self.storage.remove_relation("test", "test") is False

    def test_get_graph_statistics(self):
        """测试获取图谱统计信息"""
        self.storage.connect()

        # 创建带有实体和关系的图谱
        graph = KnowledgeGraph(id="test_graph", name="Test Graph")

        # 添加实体
        entity1 = Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON)
        entity2 = Entity(id="e2", name="Entity 2", entity_type=EntityType.ORGANIZATION)
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        # 添加关系
        relation = Relation(
            id="r1",
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.BELONGS_TO
        )
        graph.add_relation(relation)

        # 保存图谱
        self.storage.save_graph(graph)

        # 获取统计信息
        stats = self.storage.get_graph_statistics("test_graph")
        assert stats is not None
        assert stats["total_entities"] == 2
        assert stats["total_relations"] == 1
        assert "entity_types" in stats
        assert "relation_types" in stats

    def test_get_graph_statistics_nonexistent(self):
        """测试获取不存在图谱的统计信息"""
        self.storage.connect()
        stats = self.storage.get_graph_statistics("nonexistent")
        assert stats == {}

    def test_backup_graph(self):
        """测试图谱备份"""
        self.storage.connect()

        # 创建测试图谱
        graph = KnowledgeGraph(id="test_graph", name="Test Graph")
        entity = Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON)
        graph.add_entity(entity)
        self.storage.save_graph(graph)

        # 备份图谱
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            backup_path = tmp_file.name

        try:
            result = self.storage.backup_graph("test_graph", backup_path)
            assert result is True

            # 验证备份文件存在且内容正确
            assert os.path.exists(backup_path)
            with open(backup_path, 'r', encoding='utf-8') as f:
                backup_data = json.load(f)
            assert backup_data["id"] == "test_graph"
            assert backup_data["name"] == "Test Graph"
            assert len(backup_data["entities"]) == 1

        finally:
            # 清理临时文件
            if os.path.exists(backup_path):
                os.unlink(backup_path)

    def test_backup_nonexistent_graph(self):
        """测试备份不存在的图谱"""
        self.storage.connect()

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            backup_path = tmp_file.name

        try:
            result = self.storage.backup_graph("nonexistent", backup_path)
            assert result is False

        finally:
            if os.path.exists(backup_path):
                os.unlink(backup_path)

    def test_restore_graph(self):
        """测试从备份恢复图谱"""
        self.storage.connect()

        # 创建备份数据
        backup_data = {
            "id": "restored_graph",
            "name": "Restored Graph",
            "entities": {
                "e1": {
                    "id": "e1",
                    "name": "Entity 1",
                    "entity_type": "person",
                    "description": "",
                    "properties": {},
                    "aliases": [],
                    "confidence": 1.0,
                    "source": "",
                    "created_at": datetime.now().isoformat()
                }
            },
            "relations": {},
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }

        # 写入临时备份文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            json.dump(backup_data, tmp_file, ensure_ascii=False, indent=2)
            backup_path = tmp_file.name

        try:
            # 恢复图谱
            restored_id = self.storage.restore_graph(backup_path)
            assert restored_id == "restored_graph"

            # 验证恢复的图谱
            assert "restored_graph" in self.storage.graphs
            restored_graph = self.storage.graphs["restored_graph"]
            assert restored_graph.name == "Restored Graph"
            assert len(restored_graph.entities) == 1

        finally:
            # 清理临时文件
            if os.path.exists(backup_path):
                os.unlink(backup_path)

    def test_restore_invalid_backup(self):
        """测试恢复无效备份文件"""
        self.storage.connect()

        # 创建无效的备份文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_file.write("invalid json content")
            backup_path = tmp_file.name

        try:
            result = self.storage.restore_graph(backup_path)
            assert result is None

        finally:
            if os.path.exists(backup_path):
                os.unlink(backup_path)

    def test_export_graph_json(self):
        """测试导出图谱为JSON格式"""
        self.storage.connect()

        # 创建测试图谱
        graph = KnowledgeGraph(id="test_graph", name="Test Graph")
        entity = Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON)
        graph.add_entity(entity)
        self.storage.save_graph(graph)

        # 导出为JSON
        exported_data = self.storage.export_graph("test_graph", "json")
        assert exported_data is not None
        assert exported_data["id"] == "test_graph"
        assert exported_data["name"] == "Test Graph"
        assert len(exported_data["entities"]) == 1

    def test_export_graph_csv(self):
        """测试导出图谱为CSV格式"""
        self.storage.connect()

        # 创建测试图谱
        graph = KnowledgeGraph(id="test_graph", name="Test Graph")
        entity1 = Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON)
        entity2 = Entity(id="e2", name="Entity 2", entity_type=EntityType.ORGANIZATION)
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        relation = Relation(
            id="r1",
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.BELONGS_TO
        )
        graph.add_relation(relation)
        self.storage.save_graph(graph)

        # 导出为CSV
        exported_data = self.storage.export_graph("test_graph", "csv")
        assert exported_data is not None
        assert "entities" in exported_data
        assert "relations" in exported_data
        assert len(exported_data["entities"]) == 2
        assert len(exported_data["relations"]) == 1

        # 验证实体数据格式
        entity_data = exported_data["entities"][0]
        assert "id" in entity_data
        assert "name" in entity_data
        assert "type" in entity_data
        assert "confidence" in entity_data

        # 验证关系数据格式
        relation_data = exported_data["relations"][0]
        assert "id" in relation_data
        assert "head_entity" in relation_data
        assert "tail_entity" in relation_data
        assert "relation_type" in relation_data

    def test_export_graph_graphml(self):
        """测试导出图谱为GraphML格式"""
        self.storage.connect()

        # 创建测试图谱
        graph = KnowledgeGraph(id="test_graph", name="Test Graph")
        entity1 = Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON)
        entity2 = Entity(id="e2", name="Entity 2", entity_type=EntityType.ORGANIZATION)
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        relation = Relation(
            id="r1",
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.BELONGS_TO
        )
        graph.add_relation(relation)
        self.storage.save_graph(graph)

        # 导出为GraphML
        exported_data = self.storage.export_graph("test_graph", "graphml")
        assert exported_data is not None
        assert "graph" in exported_data
        assert "nodes" in exported_data["graph"]
        assert "edges" in exported_data["graph"]
        assert len(exported_data["graph"]["nodes"]) == 2
        assert len(exported_data["graph"]["edges"]) == 1

        # 验证节点数据格式
        node_data = exported_data["graph"]["nodes"][0]
        assert "id" in node_data
        assert "label" in node_data
        assert "type" in node_data

        # 验证边数据格式
        edge_data = exported_data["graph"]["edges"][0]
        assert "id" in edge_data
        assert "source" in edge_data
        assert "target" in edge_data
        assert "label" in edge_data

    def test_export_unsupported_format(self):
        """测试导出不支持的格式"""
        self.storage.connect()

        graph = KnowledgeGraph(id="test_graph", name="Test Graph")
        self.storage.save_graph(graph)

        exported_data = self.storage.export_graph("test_graph", "unsupported")
        assert exported_data is None

    def test_export_nonexistent_graph(self):
        """测试导出不存在的图谱"""
        self.storage.connect()

        exported_data = self.storage.export_graph("nonexistent", "json")
        assert exported_data is None

    def test_export_to_csv_format_empty_relations(self):
        """测试导出包含无效关系的图谱到CSV格式"""
        self.storage.connect()

        # 创建测试图谱
        graph = KnowledgeGraph(id="test_graph", name="Test Graph")
        entity = Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON)
        graph.add_entity(entity)

        # 添加无效关系（没有head或tail实体）
        invalid_relation = Relation(id="r1", relation_type=RelationType.BELONGS_TO)
        graph.relations["r1"] = invalid_relation  # 直接添加避免验证

        self.storage.save_graph(graph)

        # 导出应该跳过无效关系
        exported_data = self.storage.export_graph("test_graph", "csv")
        assert exported_data is not None
        assert len(exported_data["entities"]) == 1
        assert len(exported_data["relations"]) == 0  # 无效关系被跳过

    def test_export_to_graphml_format_empty_relations(self):
        """测试导出包含无效关系的图谱到GraphML格式"""
        self.storage.connect()

        # 创建测试图谱
        graph = KnowledgeGraph(id="test_graph", name="Test Graph")
        entity = Entity(id="e1", name="Entity 1", entity_type=EntityType.PERSON)
        graph.add_entity(entity)

        # 添加无效关系
        invalid_relation = Relation(id="r1", relation_type=RelationType.BELONGS_TO)
        graph.relations["r1"] = invalid_relation

        self.storage.save_graph(graph)

        # 导出应该跳过无效关系
        exported_data = self.storage.export_graph("test_graph", "graphml")
        assert exported_data is not None
        assert len(exported_data["graph"]["nodes"]) == 1
        assert len(exported_data["graph"]["edges"]) == 0  # 无效关系被跳过


def test_base_storage():
    """基础测试用例"""
    assert True  # hello world test case
