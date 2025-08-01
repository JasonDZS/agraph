"""
Neo4j存储测试用例
"""

import unittest
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from agraph.entities import Entity
from agraph.graph import KnowledgeGraph
from agraph.relations import Relation
from agraph.storage.neo4j_storage import Neo4jStorage
from agraph.types import EntityType, RelationType


class TestNeo4jStorage:
    """Neo4j存储测试类"""

    @pytest.fixture
    def mock_driver(self):
        """模拟 Neo4j 驱动"""
        driver = Mock()
        session = Mock()
        tx = Mock()

        # 配置mock的返回值
        driver.session.return_value = session
        session.__enter__ = Mock(return_value=session)
        session.__exit__ = Mock(return_value=False)
        session.begin_transaction.return_value = tx
        tx.__enter__ = Mock(return_value=tx)
        tx.__exit__ = Mock(return_value=False)
        session.run.return_value = []
        tx.run.return_value = None
        tx.commit.return_value = None

        return driver

    @pytest.fixture
    def neo4j_storage(self, mock_driver):
        """创建Neo4j存储实例"""
        storage = Neo4jStorage(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="test_db"
        )
        storage.driver = mock_driver
        storage.is_connected = True
        return storage

    @pytest.fixture
    def sample_entities(self):
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
    def sample_relations(self, sample_entities):
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
    def sample_graph(self, sample_entities, sample_relations):
        """创建示例知识图谱"""
        graph = KnowledgeGraph(id="test_graph", name="测试图谱")

        # 添加实体
        for entity in sample_entities:
            graph.add_entity(entity)

        # 添加关系
        for relation in sample_relations:
            graph.add_relation(relation)

        return graph

    def test_storage_initialization(self):
        """测试存储初始化"""
        storage = Neo4jStorage(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="test_db"
        )

        assert storage.uri == "bolt://localhost:7687"
        assert storage.username == "neo4j"
        assert storage.password == "password"
        assert storage.database == "test_db"
        assert storage.driver is None
        assert not storage.is_connected

    @patch('neo4j.GraphDatabase')
    def test_connection_success(self, mock_graph_db):
        """测试成功连接"""
        # 配置mock
        mock_driver = Mock()
        mock_session = Mock()
        mock_graph_db.driver.return_value = mock_driver
        mock_driver.session.return_value = mock_session
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=False)
        mock_session.run.return_value = None

        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        # 测试连接
        result = storage.connect()

        assert result is True
        assert storage.is_connected is True
        assert storage.driver == mock_driver

        # 验证调用了创建索引
        mock_session.run.assert_called()

    @patch('neo4j.GraphDatabase')
    def test_connection_failure(self, mock_graph_db):
        """测试连接失败"""
        # 配置mock抛出异常
        mock_graph_db.driver.side_effect = Exception("Connection failed")

        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        # 测试连接失败
        result = storage.connect()

        assert result is False
        assert storage.is_connected is False
        assert storage.driver is None

    @patch('builtins.__import__', side_effect=ImportError("neo4j not installed"))
    def test_neo4j_not_installed(self, mock_import):
        """测试neo4j包未安装的情况"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        # 当neo4j包未安装时，connect应该返回False
        result = storage.connect()

        assert result is False
        assert storage.is_connected is False

    def test_disconnect(self, neo4j_storage):
        """测试断开连接"""
        # 测试断开连接
        neo4j_storage.disconnect()

        assert neo4j_storage.is_connected is False
        assert neo4j_storage.driver is None

    def test_save_graph(self, neo4j_storage, sample_graph):
        """测试保存图谱"""
        # 测试保存图谱
        result = neo4j_storage.save_graph(sample_graph)

        assert result is True

        # 验证driver.session被调用
        neo4j_storage.driver.session.assert_called_with(database="test_db")

    def test_save_graph_not_connected(self, sample_graph):
        """测试未连接状态下保存图谱"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.save_graph(sample_graph)

        assert result is False

    def test_load_graph_metadata_found(self, neo4j_storage):
        """测试加载存在的图谱"""
        # 模拟返回图谱元数据
        mock_record = {
            "g": {
                "id": "test_graph",
                "name": "测试图谱",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
        }

        session = neo4j_storage.driver.session.return_value

        # 设置mock的调用返回值：第一次调用返回图谱元数据，后续调用返回空列表
        session.run.side_effect = [
            Mock(single=Mock(return_value=mock_record)),  # _load_graph_metadata
            [],  # _load_entities
            []   # _load_relations
        ]

        result = neo4j_storage.load_graph("test_graph")

        # 由于没有实体和关系数据，应该返回空图谱
        assert result is not None
        assert result.id == "test_graph"
        assert result.name == "测试图谱"

    def test_load_graph_not_found(self, neo4j_storage):
        """测试加载不存在的图谱"""
        # 模拟返回空结果
        mock_result = Mock()
        mock_result.single.return_value = None

        session = neo4j_storage.driver.session.return_value
        session.run.return_value = mock_result

        result = neo4j_storage.load_graph("nonexistent_graph")

        assert result is None

    def test_load_graph_not_connected(self):
        """测试未连接状态下加载图谱"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.load_graph("test_graph")

        assert result is None

    def test_delete_graph(self, neo4j_storage):
        """测试删除图谱"""
        result = neo4j_storage.delete_graph("test_graph")

        assert result is True

        # 验证session被调用
        neo4j_storage.driver.session.assert_called_with(database="test_db")

    def test_delete_graph_not_connected(self):
        """测试未连接状态下删除图谱"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.delete_graph("test_graph")

        assert result is False

    def test_list_graphs(self, neo4j_storage):
        """测试列出图谱"""
        # 模拟返回图谱列表
        mock_records = [
            {"id": "graph1", "name": "图谱1", "created_at": "2023-01-01T00:00:00", "updated_at": "2023-01-02T00:00:00"},
            {"id": "graph2", "name": "图谱2", "created_at": "2023-01-03T00:00:00", "updated_at": "2023-01-04T00:00:00"}
        ]

        session = neo4j_storage.driver.session.return_value
        session.run.return_value = mock_records

        result = neo4j_storage.list_graphs()

        assert len(result) == 2
        assert result[0]["id"] == "graph1"
        assert result[1]["id"] == "graph2"

    def test_list_graphs_not_connected(self):
        """测试未连接状态下列出图谱"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.list_graphs()

        assert result == []

    def test_query_entities(self, neo4j_storage):
        """测试查询实体"""
        # 模拟返回实体数据
        mock_records = [
            {
                "e": {
                    "id": "entity_1",
                    "name": "张三",
                    "entity_type": "person",
                    "description": "一个人",
                    "properties": {},
                    "aliases": [],
                    "confidence": 0.9,
                    "source": "",
                    "created_at": datetime.now().isoformat()
                }
            }
        ]

        session = neo4j_storage.driver.session.return_value
        session.run.return_value = mock_records

        result = neo4j_storage.query_entities({"graph_id": "test_graph"})

        assert len(result) == 1
        assert result[0].name == "张三"
        assert result[0].entity_type == EntityType.PERSON

    def test_query_entities_with_conditions(self, neo4j_storage):
        """测试带条件的实体查询"""
        session = neo4j_storage.driver.session.return_value
        session.run.return_value = []

        conditions = {
            "graph_id": "test_graph",
            "entity_type": "person",
            "name": "张",
            "limit": 50
        }

        result = neo4j_storage.query_entities(conditions)

        # 验证查询被调用
        session.run.assert_called_once()
        call_args = session.run.call_args

        # 验证查询中包含条件参数
        query = call_args[0][0]  # 获取查询字符串
        params = call_args[0][1]  # 获取参数

        assert "graph_id" in params
        assert "entity_type" in params
        assert "name" in params

    def test_query_entities_not_connected(self):
        """测试未连接状态下查询实体"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.query_entities({"graph_id": "test_graph"})

        assert result == []

    def test_query_relations(self, neo4j_storage, sample_entities):
        """测试查询关系"""
        entity1, entity2, _ = sample_entities

        # 模拟返回关系数据
        mock_records = [
            {
                "head": {
                    "id": "entity_1",
                    "name": "张三",
                    "entity_type": "person",
                    "description": "一个人",
                    "properties": {},
                    "aliases": [],
                    "confidence": 0.9,
                    "source": "",
                    "created_at": datetime.now().isoformat()
                },
                "tail": {
                    "id": "entity_2",
                    "name": "北京",
                    "entity_type": "location",
                    "description": "中国首都",
                    "properties": {},
                    "aliases": [],
                    "confidence": 0.95,
                    "source": "",
                    "created_at": datetime.now().isoformat()
                },
                "r": {
                    "id": "relation_1",
                    "relation_type": "belongs_to",
                    "properties": {},
                    "confidence": 0.8,
                    "source": "",
                    "created_at": datetime.now().isoformat()
                }
            }
        ]

        session = neo4j_storage.driver.session.return_value
        session.run.return_value = mock_records

        result = neo4j_storage.query_relations(
            head_entity="entity_1",
            tail_entity="entity_2",
            relation_type=RelationType.BELONGS_TO
        )

        assert len(result) == 1
        assert result[0].head_entity.name == "张三"
        assert result[0].tail_entity.name == "北京"
        assert result[0].relation_type == RelationType.BELONGS_TO

    def test_query_relations_not_connected(self):
        """测试未连接状态下查询关系"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.query_relations()

        assert result == []

    def test_add_entity(self, neo4j_storage, sample_entities):
        """测试添加实体"""
        entity = sample_entities[0]

        result = neo4j_storage.add_entity("test_graph", entity)

        assert result is True

    def test_add_entity_not_connected(self, sample_entities):
        """测试未连接状态下添加实体"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")
        entity = sample_entities[0]

        result = storage.add_entity("test_graph", entity)

        assert result is False

    def test_add_relation(self, neo4j_storage, sample_relations):
        """测试添加关系"""
        relation = sample_relations[0]

        result = neo4j_storage.add_relation("test_graph", relation)

        assert result is True

    def test_add_relation_not_connected(self, sample_relations):
        """测试未连接状态下添加关系"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")
        relation = sample_relations[0]

        result = storage.add_relation("test_graph", relation)

        assert result is False

    def test_update_entity(self, neo4j_storage, sample_entities):
        """测试更新实体"""
        entity = sample_entities[0]
        entity.description = "更新后的描述"

        result = neo4j_storage.update_entity("test_graph", entity)

        assert result is True

    def test_update_relation(self, neo4j_storage, sample_relations):
        """测试更新关系"""
        relation = sample_relations[0]
        relation.confidence = 0.95

        result = neo4j_storage.update_relation("test_graph", relation)

        assert result is True

    def test_remove_entity(self, neo4j_storage):
        """测试删除实体"""
        result = neo4j_storage.remove_entity("test_graph", "entity_1")

        assert result is True

        # 验证session被调用
        session = neo4j_storage.driver.session.return_value
        session.run.assert_called()

    def test_remove_entity_not_connected(self):
        """测试未连接状态下删除实体"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.remove_entity("test_graph", "entity_1")

        assert result is False

    def test_remove_relation(self, neo4j_storage):
        """测试删除关系"""
        result = neo4j_storage.remove_relation("test_graph", "relation_1")

        assert result is True

        # 验证session被调用
        session = neo4j_storage.driver.session.return_value
        session.run.assert_called()

    def test_remove_relation_not_connected(self):
        """测试未连接状态下删除关系"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.remove_relation("test_graph", "relation_1")

        assert result is False

    def test_execute_cypher(self, neo4j_storage):
        """测试执行Cypher查询"""
        # 模拟查询结果
        mock_records = [
            {"n.name": "张三", "n.age": 30},
            {"n.name": "李四", "n.age": 25}
        ]

        session = neo4j_storage.driver.session.return_value
        session.run.return_value = mock_records

        query = "MATCH (n:Person) RETURN n.name, n.age"
        parameters = {"limit": 10}

        result = neo4j_storage.execute_cypher(query, parameters)

        assert len(result) == 2
        assert result[0]["n.name"] == "张三"
        assert result[1]["n.name"] == "李四"

        # 验证调用参数
        session.run.assert_called_with(query, parameters)

    def test_execute_cypher_not_connected(self):
        """测试未连接状态下执行Cypher查询"""
        storage = Neo4jStorage("bolt://localhost:7687", "neo4j", "password")

        result = storage.execute_cypher("MATCH (n) RETURN n")

        assert result == []

    def test_execute_cypher_without_parameters(self, neo4j_storage):
        """测试不带参数执行Cypher查询"""
        session = neo4j_storage.driver.session.return_value
        session.run.return_value = []

        query = "MATCH (n) RETURN count(n)"

        result = neo4j_storage.execute_cypher(query)

        # 验证调用时使用了空字典作为参数
        session.run.assert_called_with(query, {})

    def test_create_indexes(self, neo4j_storage):
        """测试创建索引"""
        session = neo4j_storage.driver.session.return_value

        neo4j_storage.create_indexes()

        # 验证创建索引的查询被调用了多次
        assert session.run.call_count >= 5  # 至少5个索引

    def test_record_to_entity(self, neo4j_storage):
        """测试记录转换为实体"""
        record = {
            "id": "test_entity",
            "name": "测试实体",
            "entity_type": "person",
            "description": "测试描述",
            "properties": {"age": 30},
            "aliases": ["别名1", "别名2"],
            "confidence": 0.9,
            "source": "test_source",
            "created_at": datetime.now().isoformat()
        }

        entity = neo4j_storage._record_to_entity(record)

        assert entity.id == "test_entity"
        assert entity.name == "测试实体"
        assert entity.entity_type == EntityType.PERSON
        assert entity.description == "测试描述"
        assert entity.properties["age"] == 30
        assert entity.aliases == ["别名1", "别名2"]
        assert entity.confidence == 0.9
        assert entity.source == "test_source"

    def test_record_to_relation(self, neo4j_storage, sample_entities):
        """测试记录转换为关系"""
        head_entity, tail_entity = sample_entities[0], sample_entities[1]

        record = {
            "id": "test_relation",
            "relation_type": "belongs_to",
            "properties": {"strength": "strong"},
            "confidence": 0.8,
            "source": "test_source",
            "created_at": datetime.now().isoformat()
        }

        relation = neo4j_storage._record_to_relation(record, head_entity, tail_entity)

        assert relation.id == "test_relation"
        assert relation.head_entity == head_entity
        assert relation.tail_entity == tail_entity
        assert relation.relation_type == RelationType.BELONGS_TO
        assert relation.properties["strength"] == "strong"
        assert relation.confidence == 0.8
        assert relation.source == "test_source"

    def test_error_handling_in_operations(self, neo4j_storage):
        """测试操作中的错误处理"""
        # 模拟数据库操作异常
        session = neo4j_storage.driver.session.return_value
        session.run.side_effect = Exception("Database error")

        # 测试各种操作在出错时返回适当的值
        assert neo4j_storage.query_entities({"graph_id": "test"}) == []
        assert neo4j_storage.query_relations() == []
        assert neo4j_storage.execute_cypher("MATCH (n) RETURN n") == []
        assert neo4j_storage.add_entity("test", Entity()) is False

        # 对于add_relation，应该因为关系无效（没有实体）而返回True，但实际没有保存
        invalid_relation = Relation()  # 没有头实体和尾实体的关系
        assert neo4j_storage.add_relation("test", invalid_relation) is True  # 因为只是跳过无效关系

        assert neo4j_storage.remove_entity("test", "entity_1") is False
        assert neo4j_storage.remove_relation("test", "relation_1") is False

    def test_save_relations_with_missing_entities(self, neo4j_storage):
        """测试保存缺少实体的关系"""
        # 创建一个没有头实体或尾实体的关系
        invalid_relation = Relation(
            id="invalid_relation",
            head_entity=None,
            tail_entity=None,
            relation_type=RelationType.RELATED_TO
        )

        session = neo4j_storage.driver.session.return_value
        tx = session.begin_transaction.return_value

        # 调用_save_relations方法
        neo4j_storage._save_relations(tx, "test_graph", [invalid_relation])

        # 验证tx.run没有被调用（因为关系无效）
        tx.run.assert_not_called()


class TestNeo4jStorageClassic(unittest.TestCase):
    """使用unittest的传统测试类（保持向后兼容）"""

    def test_hello_world(self):
        """基础测试"""
        self.assertEqual("Hello, World!", "Hello, World!")

    def test_storage_basic_properties(self):
        """测试存储基本属性"""
        storage = Neo4jStorage(
            uri="bolt://localhost:7687",
            username="neo4j",
            password="password",
            database="test"
        )

        self.assertEqual(storage.uri, "bolt://localhost:7687")
        self.assertEqual(storage.username, "neo4j")
        self.assertEqual(storage.password, "password")
        self.assertEqual(storage.database, "test")
        self.assertIsNone(storage.driver)
        self.assertFalse(storage.is_connected)


if __name__ == "__main__":
    unittest.main()
