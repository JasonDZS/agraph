"""
JSON存储测试用例
"""

import json
import os
import tempfile
import shutil
from datetime import datetime
from typing import Dict, Any

import pytest

from agraph.entities import Entity
from agraph.graph import KnowledgeGraph
from agraph.relations import Relation
from agraph.storage.json_storage import JsonStorage
from agraph.types import EntityType, RelationType


@pytest.fixture
def temp_storage_dir():
    """创建临时存储目录"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def json_storage(temp_storage_dir):
    """创建JSON存储实例"""
    storage = JsonStorage(temp_storage_dir)
    storage.connect()
    yield storage
    storage.disconnect()


@pytest.fixture
def sample_entities():
    """创建示例实体"""
    entity1 = Entity(
        id="entity_1",
        name="张三",
        entity_type=EntityType.PERSON,
        description="一个人",
        confidence=0.9
    )

    entity2 = Entity(
        id="entity_2",
        name="北京",
        entity_type=EntityType.LOCATION,
        description="中国首都",
        confidence=0.95
    )

    entity3 = Entity(
        id="entity_3",
        name="清华大学",
        entity_type=EntityType.ORGANIZATION,
        description="中国知名大学",
        confidence=0.88
    )

    return [entity1, entity2, entity3]


@pytest.fixture
def sample_relations(sample_entities):
    """创建示例关系"""
    entity1, entity2, entity3 = sample_entities

    relation1 = Relation(
        id="relation_1",
        head_entity=entity1,
        tail_entity=entity2,
        relation_type=RelationType.BELONGS_TO,
        confidence=0.8
    )

    relation2 = Relation(
        id="relation_2",
        head_entity=entity1,
        tail_entity=entity3,
        relation_type=RelationType.REFERENCES,
        confidence=0.7
    )

    return [relation1, relation2]


@pytest.fixture
def sample_graph(sample_entities, sample_relations):
    """创建示例知识图谱"""
    graph = KnowledgeGraph(id="test_graph", name="测试图谱")

    # 添加实体
    for entity in sample_entities:
        graph.add_entity(entity)

    # 添加关系
    for relation in sample_relations:
        graph.add_relation(relation)

    return graph


class TestJsonStorage:
    """JSON存储测试类"""

    def test_storage_initialization(self, temp_storage_dir):
        """测试存储初始化"""
        storage = JsonStorage(temp_storage_dir)
        assert storage.storage_dir == temp_storage_dir
        assert not storage.is_connected
        assert storage.graphs_file == os.path.join(temp_storage_dir, "graphs.json")

    def test_storage_connection(self, json_storage, temp_storage_dir):
        """测试存储连接"""
        assert json_storage.is_connected
        assert os.path.exists(temp_storage_dir)

        # 测试断开连接
        json_storage.disconnect()
        assert not json_storage.is_connected

        # 测试重新连接
        assert json_storage.connect()
        assert json_storage.is_connected

    def test_save_and_load_graph(self, json_storage, sample_graph):
        """测试保存和加载图谱"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 验证文件存在
        graph_file = os.path.join(json_storage.storage_dir, f"{sample_graph.id}.json")
        assert os.path.exists(graph_file)

        # 加载图谱
        loaded_graph = json_storage.load_graph(sample_graph.id)
        assert loaded_graph is not None
        assert loaded_graph.id == sample_graph.id
        assert loaded_graph.name == sample_graph.name
        assert len(loaded_graph.entities) == len(sample_graph.entities)
        assert len(loaded_graph.relations) == len(sample_graph.relations)

        # 验证实体
        for entity_id, entity in sample_graph.entities.items():
            loaded_entity = loaded_graph.entities[entity_id]
            assert loaded_entity.name == entity.name
            assert loaded_entity.entity_type == entity.entity_type
            assert loaded_entity.description == entity.description

        # 验证关系
        for relation_id, relation in sample_graph.relations.items():
            loaded_relation = loaded_graph.relations[relation_id]
            assert loaded_relation.relation_type == relation.relation_type
            assert loaded_relation.head_entity.id == relation.head_entity.id
            assert loaded_relation.tail_entity.id == relation.tail_entity.id

    def test_load_nonexistent_graph(self, json_storage):
        """测试加载不存在的图谱"""
        result = json_storage.load_graph("nonexistent_graph")
        assert result is None

    def test_delete_graph(self, json_storage, sample_graph):
        """测试删除图谱"""
        # 先保存图谱
        assert json_storage.save_graph(sample_graph)

        # 验证图谱存在
        graph_file = os.path.join(json_storage.storage_dir, f"{sample_graph.id}.json")
        assert os.path.exists(graph_file)

        # 删除图谱
        assert json_storage.delete_graph(sample_graph.id)

        # 验证文件被删除
        assert not os.path.exists(graph_file)

        # 验证无法加载已删除的图谱
        assert json_storage.load_graph(sample_graph.id) is None

    def test_list_graphs(self, json_storage, sample_graph):
        """测试列出图谱"""
        # 初始状态应该为空
        graphs = json_storage.list_graphs()
        assert len(graphs) == 0

        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 列出图谱
        graphs = json_storage.list_graphs()
        assert len(graphs) == 1

        graph_info = graphs[0]
        assert graph_info["id"] == sample_graph.id
        assert graph_info["name"] == sample_graph.name
        assert graph_info["entity_count"] == len(sample_graph.entities)
        assert graph_info["relation_count"] == len(sample_graph.relations)
        assert "created_at" in graph_info
        assert "updated_at" in graph_info

    def test_query_entities(self, json_storage, sample_graph):
        """测试查询实体"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 查询所有实体
        entities = json_storage.query_entities({"graph_id": sample_graph.id})
        assert len(entities) == 3

        # 按类型查询
        person_entities = json_storage.query_entities({
            "graph_id": sample_graph.id,
            "entity_type": EntityType.PERSON.value
        })
        assert len(person_entities) == 1
        assert person_entities[0].name == "张三"

        # 按名称查询
        entities_with_name = json_storage.query_entities({
            "graph_id": sample_graph.id,
            "name": "北京"
        })
        assert len(entities_with_name) == 1
        assert entities_with_name[0].name == "北京"

        # 按置信度查询
        high_confidence_entities = json_storage.query_entities({
            "graph_id": sample_graph.id,
            "min_confidence": 0.9
        })
        assert len(high_confidence_entities) == 2  # 张三和北京

        # 限制返回数量
        limited_entities = json_storage.query_entities({
            "graph_id": sample_graph.id,
            "limit": 1
        })
        assert len(limited_entities) == 1

    def test_query_relations(self, json_storage, sample_graph):
        """测试查询关系"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 查询所有关系
        relations = json_storage.query_relations(graph_id=sample_graph.id)
        assert len(relations) == 2

        # 按头实体查询
        entity1_relations = json_storage.query_relations(
            head_entity="entity_1",
            graph_id=sample_graph.id
        )
        assert len(entity1_relations) == 2

        # 按尾实体查询
        entity2_relations = json_storage.query_relations(
            tail_entity="entity_2",
            graph_id=sample_graph.id
        )
        assert len(entity2_relations) == 1

        # 按关系类型查询
        belongs_to_relations = json_storage.query_relations(
            relation_type=RelationType.BELONGS_TO,
            graph_id=sample_graph.id
        )
        assert len(belongs_to_relations) == 1
        assert belongs_to_relations[0].relation_type == RelationType.BELONGS_TO

    def test_add_entity(self, json_storage, sample_graph):
        """测试添加实体"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 创建新实体
        new_entity = Entity(
            id="entity_4",
            name="上海",
            entity_type=EntityType.LOCATION,
            description="中国直辖市"
        )

        # 添加实体
        assert json_storage.add_entity(sample_graph.id, new_entity)

        # 验证实体被添加
        loaded_graph = json_storage.load_graph(sample_graph.id)
        assert len(loaded_graph.entities) == 4
        assert "entity_4" in loaded_graph.entities
        assert loaded_graph.entities["entity_4"].name == "上海"

    def test_add_relation(self, json_storage, sample_graph):
        """测试添加关系"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 创建新关系
        new_relation = Relation(
            id="relation_3",
            head_entity=sample_graph.entities["entity_2"],
            tail_entity=sample_graph.entities["entity_3"],
            relation_type=RelationType.CONTAINS
        )

        # 添加关系
        assert json_storage.add_relation(sample_graph.id, new_relation)

        # 验证关系被添加
        loaded_graph = json_storage.load_graph(sample_graph.id)
        assert len(loaded_graph.relations) == 3
        assert "relation_3" in loaded_graph.relations

    def test_update_entity(self, json_storage, sample_graph):
        """测试更新实体"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 更新实体
        entity = sample_graph.entities["entity_1"]
        entity.description = "更新后的描述"
        entity.confidence = 0.99

        assert json_storage.update_entity(sample_graph.id, entity)

        # 验证更新
        loaded_graph = json_storage.load_graph(sample_graph.id)
        updated_entity = loaded_graph.entities["entity_1"]
        assert updated_entity.description == "更新后的描述"
        assert updated_entity.confidence == 0.99

    def test_update_relation(self, json_storage, sample_graph):
        """测试更新关系"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 更新关系
        relation = sample_graph.relations["relation_1"]
        relation.confidence = 0.95

        assert json_storage.update_relation(sample_graph.id, relation)

        # 验证更新
        loaded_graph = json_storage.load_graph(sample_graph.id)
        updated_relation = loaded_graph.relations["relation_1"]
        assert updated_relation.confidence == 0.95

    def test_remove_entity(self, json_storage, sample_graph):
        """测试删除实体"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 删除实体
        assert json_storage.remove_entity(sample_graph.id, "entity_1")

        # 验证实体被删除
        loaded_graph = json_storage.load_graph(sample_graph.id)
        assert len(loaded_graph.entities) == 2
        assert "entity_1" not in loaded_graph.entities

        # 验证相关关系也被删除
        assert len(loaded_graph.relations) == 0  # entity_1的两个关系都应该被删除

    def test_remove_relation(self, json_storage, sample_graph):
        """测试删除关系"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 删除关系
        assert json_storage.remove_relation(sample_graph.id, "relation_1")

        # 验证关系被删除
        loaded_graph = json_storage.load_graph(sample_graph.id)
        assert len(loaded_graph.relations) == 1
        assert "relation_1" not in loaded_graph.relations

    def test_graph_index_operations(self, json_storage, sample_graph):
        """测试图谱索引操作"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 验证索引文件存在
        assert os.path.exists(json_storage.graphs_file)

        # 读取索引文件
        with open(json_storage.graphs_file, "r", encoding="utf-8") as f:
            index_data = json.load(f)

        assert "graphs" in index_data
        assert len(index_data["graphs"]) == 1

        graph_info = index_data["graphs"][0]
        assert graph_info["id"] == sample_graph.id
        assert graph_info["entity_count"] == len(sample_graph.entities)
        assert graph_info["relation_count"] == len(sample_graph.relations)

    def test_compact_storage(self, json_storage, sample_graph):
        """测试存储压缩"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 手动删除图谱文件但保留索引
        graph_file = os.path.join(json_storage.storage_dir, f"{sample_graph.id}.json")
        os.remove(graph_file)

        # 执行压缩
        json_storage.compact_storage()

        # 验证索引被清理
        graphs = json_storage.list_graphs()
        assert len(graphs) == 0

    def test_get_storage_info(self, json_storage, sample_graph):
        """测试获取存储信息"""
        # 保存图谱
        assert json_storage.save_graph(sample_graph)

        # 获取存储信息
        info = json_storage.get_storage_info()

        assert "storage_dir" in info
        assert "total_size_bytes" in info
        assert "total_size_mb" in info
        assert "file_count" in info
        assert "graphs_count" in info
        assert "is_connected" in info

        assert info["storage_dir"] == json_storage.storage_dir
        assert info["file_count"] >= 1  # 至少有一个图谱文件
        assert info["graphs_count"] == 1
        assert info["is_connected"] is True

    def test_error_handling(self, temp_storage_dir):
        """测试错误处理"""
        storage = JsonStorage(temp_storage_dir)

        # 未连接时的操作应该失败
        assert not storage.save_graph(KnowledgeGraph())
        assert storage.load_graph("test") is None
        assert not storage.delete_graph("test")
        assert len(storage.list_graphs()) == 0
        assert len(storage.query_entities({})) == 0
        assert len(storage.query_relations()) == 0

    def test_invalid_operations(self, json_storage):
        """测试无效操作"""
        # 测试操作不存在的图谱
        new_entity = Entity(name="测试实体")
        assert not json_storage.add_entity("nonexistent_graph", new_entity)

        new_relation = Relation()
        assert not json_storage.add_relation("nonexistent_graph", new_relation)

        assert not json_storage.update_entity("nonexistent_graph", new_entity)
        assert not json_storage.update_relation("nonexistent_graph", new_relation)
        assert not json_storage.remove_entity("nonexistent_graph", "entity_id")
        assert not json_storage.remove_relation("nonexistent_graph", "relation_id")

    def test_multiple_graphs(self, json_storage):
        """测试多个图谱管理"""
        # 创建多个图谱
        graph1 = KnowledgeGraph(id="graph_1", name="图谱1")
        graph2 = KnowledgeGraph(id="graph_2", name="图谱2")

        # 保存图谱
        assert json_storage.save_graph(graph1)
        assert json_storage.save_graph(graph2)

        # 列出所有图谱
        graphs = json_storage.list_graphs()
        assert len(graphs) == 2

        # 验证可以分别加载
        loaded_graph1 = json_storage.load_graph("graph_1")
        loaded_graph2 = json_storage.load_graph("graph_2")

        assert loaded_graph1.name == "图谱1"
        assert loaded_graph2.name == "图谱2"

        # 删除一个图谱
        assert json_storage.delete_graph("graph_1")

        # 验证只剩一个图谱
        graphs = json_storage.list_graphs()
        assert len(graphs) == 1
        assert graphs[0]["id"] == "graph_2"


def test_json_storage_basic():
    """基础功能测试"""
    assert True  # 保留原有的简单测试
