import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from agraph.builders.graph_builder import (
    BaseKnowledgeGraphBuilder,
    StandardGraphBuilder,
    MultiSourceGraphBuilder
)
from agraph.graph import KnowledgeGraph
from agraph.entities import Entity
from agraph.relations import Relation
from agraph.types import EntityType, RelationType


class MockGraphBuilder(BaseKnowledgeGraphBuilder):
    """用于测试的模拟图构建器"""

    def build_graph(self, texts=None, database_schema=None, graph_name="test_graph"):
        graph = KnowledgeGraph(name=graph_name)
        return graph

    def update_graph(self, graph, new_entities=None, new_relations=None):
        return graph


class TestBaseKnowledgeGraphBuilder(unittest.TestCase):
    """测试基础知识图谱构建器"""

    def setUp(self):
        self.builder = MockGraphBuilder()

    def test_init(self):
        """测试初始化"""
        self.assertIsNone(self.builder.entity_extractor)
        self.assertIsNone(self.builder.relation_extractor)
        self.assertIsNone(self.builder.graph_storage)
        self.assertEqual(self.builder.merge_threshold, 0.8)

    def test_merge_graphs_empty_list(self):
        """测试合并空图列表"""
        result = self.builder.merge_graphs([])
        self.assertIsInstance(result, KnowledgeGraph)
        self.assertEqual(len(result.entities), 0)

    def test_merge_graphs_single_graph(self):
        """测试合并单个图"""
        graph = KnowledgeGraph(name="test")
        entity = Entity(name="TestEntity", entity_type=EntityType.PERSON)
        graph.add_entity(entity)

        result = self.builder.merge_graphs([graph])
        self.assertEqual(result, graph)
        self.assertEqual(len(result.entities), 1)

    def test_merge_graphs_multiple_graphs(self):
        """测试合并多个图"""
        # 创建两个图
        graph1 = KnowledgeGraph(name="graph1")
        entity1 = Entity(name="Alice", entity_type=EntityType.PERSON)
        graph1.add_entity(entity1)

        graph2 = KnowledgeGraph(name="graph2")
        entity2 = Entity(name="Bob", entity_type=EntityType.PERSON)
        graph2.add_entity(entity2)

        result = self.builder.merge_graphs([graph1, graph2])
        self.assertEqual(result.name, "merged_graph")
        self.assertEqual(len(result.entities), 2)

    def test_merge_graphs_with_relations(self):
        """测试合并包含关系的图"""
        # 创建包含关系的图
        graph = KnowledgeGraph(name="test")
        entity1 = Entity(name="Alice", entity_type=EntityType.PERSON)
        entity2 = Entity(name="Bob", entity_type=EntityType.PERSON)
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        relation = Relation(
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.RELATED_TO
        )
        graph.add_relation(relation)

        result = self.builder.merge_graphs([graph])
        self.assertEqual(len(result.entities), 2)
        self.assertEqual(len(result.relations), 1)

    def test_align_entities(self):
        """测试实体对齐"""
        entity1 = Entity(name="Alice", entity_type=EntityType.PERSON, confidence=0.8)
        entity2 = Entity(name="alice", entity_type=EntityType.PERSON, confidence=0.9)  # 相同名称，不同大小写
        entity3 = Entity(name="Bob", entity_type=EntityType.PERSON)

        entities = [entity1, entity2, entity3]
        aligned = self.builder._align_entities(entities)

        # 应该合并重复的实体
        self.assertEqual(len(aligned), 2)
        # 检查置信度是否取较高值
        alice_entity = next(e for e in aligned if e.name.lower() == "alice")
        self.assertEqual(alice_entity.confidence, 0.9)

    def test_merge_entity_attributes(self):
        """测试实体属性合并"""
        target = Entity(
            name="Alice",
            entity_type=EntityType.PERSON,
            aliases=["A"],
            properties={"age": 25},
            confidence=0.7
        )
        source = Entity(
            name="Alice",
            entity_type=EntityType.PERSON,
            aliases=["Al", "A"],  # 包含重复别名
            properties={"location": "NYC", "age": 26},  # 包含冲突属性
            confidence=0.9
        )

        self.builder._merge_entity_attributes(target, source)

        # 检查别名合并和去重
        self.assertIn("Al", target.aliases)
        self.assertEqual(len(target.aliases), 2)  # A, Al

        # 检查属性合并
        self.assertEqual(target.properties["age"], 26)  # 应该被覆盖
        self.assertEqual(target.properties["location"], "NYC")

        # 检查置信度
        self.assertEqual(target.confidence, 0.9)

    def test_validate_graph_empty(self):
        """测试验证空图"""
        graph = KnowledgeGraph()
        result = self.builder.validate_graph(graph)

        self.assertTrue(result["valid"])
        self.assertEqual(len(result["issues"]), 0)
        self.assertIn("statistics", result)

    def test_validate_graph_with_invalid_relations(self):
        """测试验证包含无效关系的图"""
        graph = KnowledgeGraph()
        entity = Entity(name="TestEntity")
        graph.add_entity(entity)

        # 创建无效关系（引用不存在的实体）
        invalid_relation = Relation(
            head_entity=entity,
            tail_entity=Entity(name="NonExistentEntity"),  # 这个实体没有添加到图中
            relation_type=RelationType.RELATED_TO
        )
        # 直接添加到relations字典，绕过验证
        graph.relations[invalid_relation.id] = invalid_relation

        result = self.builder.validate_graph(graph)

        # 应该检测到完整性问题
        integrity_issues = [issue for issue in result["issues"] if "entity" in issue["type"]]
        self.assertGreater(len(integrity_issues), 0)

    def test_check_connectivity(self):
        """测试连通性检查"""
        graph = KnowledgeGraph()

        # 创建两个分离的组件
        entity1 = Entity(name="Alice")
        entity2 = Entity(name="Bob")
        entity3 = Entity(name="Charlie")
        entity4 = Entity(name="David")

        graph.add_entity(entity1)
        graph.add_entity(entity2)
        graph.add_entity(entity3)
        graph.add_entity(entity4)

        # Alice-Bob 连接
        relation1 = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.RELATED_TO)
        graph.add_relation(relation1)

        # Charlie-David 连接（分离的组件）
        relation2 = Relation(head_entity=entity3, tail_entity=entity4, relation_type=RelationType.RELATED_TO)
        graph.add_relation(relation2)

        issues = self.builder._check_connectivity(graph)

        # 应该检测到断开的组件
        connectivity_issues = [issue for issue in issues if issue["type"] == "disconnected_components"]
        self.assertEqual(len(connectivity_issues), 1)
        self.assertEqual(connectivity_issues[0]["count"], 2)

    def test_find_isolated_nodes(self):
        """测试查找孤立节点"""
        graph = KnowledgeGraph()

        entity1 = Entity(name="Connected1")
        entity2 = Entity(name="Connected2")
        entity3 = Entity(name="Isolated")

        graph.add_entity(entity1)
        graph.add_entity(entity2)
        graph.add_entity(entity3)

        # 只连接前两个实体
        relation = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.RELATED_TO)
        graph.add_relation(relation)

        isolated = self.builder._find_isolated_nodes(graph)

        self.assertEqual(len(isolated), 1)
        self.assertEqual(isolated[0], entity3.id)

    def test_detect_cycles(self):
        """测试循环检测"""
        graph = KnowledgeGraph()

        entity1 = Entity(name="A")
        entity2 = Entity(name="B")
        entity3 = Entity(name="C")

        graph.add_entity(entity1)
        graph.add_entity(entity2)
        graph.add_entity(entity3)

        # 创建循环：A -> B -> C -> A
        relation1 = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.RELATED_TO)
        relation2 = Relation(head_entity=entity2, tail_entity=entity3, relation_type=RelationType.RELATED_TO)
        relation3 = Relation(head_entity=entity3, tail_entity=entity1, relation_type=RelationType.RELATED_TO)

        graph.add_relation(relation1)
        graph.add_relation(relation2)
        graph.add_relation(relation3)

        cycles = self.builder._detect_cycles(graph)

        self.assertGreater(len(cycles), 0)

    def test_generate_recommendations(self):
        """测试生成建议"""
        graph = KnowledgeGraph()
        entity = Entity(name="TestEntity")
        graph.add_entity(entity)

        issues = [
            {"type": "isolated_nodes", "count": 1},
            {"type": "missing_head_entity", "severity": "high"}
        ]

        recommendations = self.builder._generate_recommendations(graph, issues)

        self.assertGreater(len(recommendations), 0)
        self.assertTrue(any("孤立节点" in rec for rec in recommendations))
        self.assertTrue(any("无效的关系" in rec for rec in recommendations))


class TestStandardGraphBuilder(unittest.TestCase):
    """测试标准图构建器"""

    def setUp(self):
        self.builder = StandardGraphBuilder()

    def test_init(self):
        """测试初始化"""
        self.assertIsNotNone(self.builder.text_entity_extractor)
        self.assertIsNotNone(self.builder.db_entity_extractor)
        self.assertIsNotNone(self.builder.text_relation_extractor)
        self.assertIsNotNone(self.builder.db_relation_extractor)

    @patch('agraph.builders.graph_builder.TextEntityExtractor')
    @patch('agraph.builders.graph_builder.TextRelationExtractor')
    def test_build_graph_from_texts(self, mock_relation_extractor_class, mock_entity_extractor_class):
        """测试从文本构建图"""
        # 设置模拟
        mock_entity_extractor = Mock()
        mock_relation_extractor = Mock()
        mock_entity_extractor_class.return_value = mock_entity_extractor
        mock_relation_extractor_class.return_value = mock_relation_extractor

        # 模拟实体抽取
        test_entity = Entity(name="TestEntity", entity_type=EntityType.PERSON)
        mock_entity_extractor.extract_from_text.return_value = [test_entity]
        mock_entity_extractor.deduplicate_entities.return_value = [test_entity]

        # 模拟关系抽取
        test_relation = Relation(
            head_entity=test_entity,
            tail_entity=test_entity,
            relation_type=RelationType.RELATED_TO
        )
        mock_relation_extractor.extract_from_text.return_value = [test_relation]
        mock_relation_extractor.infer_implicit_relations.return_value = []

        builder = StandardGraphBuilder()
        builder.text_entity_extractor = mock_entity_extractor
        builder.text_relation_extractor = mock_relation_extractor

        texts = ["Test text for entity extraction"]
        result = builder.build_graph(texts=texts, graph_name="test_graph")

        self.assertEqual(result.name, "test_graph")
        self.assertEqual(len(result.entities), 1)
        self.assertIn(test_entity.id, result.entities)

    @patch('agraph.builders.graph_builder.DatabaseEntityExtractor')
    @patch('agraph.builders.graph_builder.DatabaseRelationExtractor')
    def test_build_graph_from_database(self, mock_relation_extractor_class, mock_entity_extractor_class):
        """测试从数据库构建图"""
        # 设置模拟
        mock_entity_extractor = Mock()
        mock_relation_extractor = Mock()
        mock_entity_extractor_class.return_value = mock_entity_extractor
        mock_relation_extractor_class.return_value = mock_relation_extractor

        # 模拟数据库实体抽取
        test_entity = Entity(name="TestTable", entity_type=EntityType.TABLE)
        mock_entity_extractor.extract_from_database.return_value = [test_entity]

        # 模拟数据库关系抽取
        mock_relation_extractor.extract_from_database.return_value = []

        builder = StandardGraphBuilder()
        builder.db_entity_extractor = mock_entity_extractor
        builder.db_relation_extractor = mock_relation_extractor
        builder.text_entity_extractor = Mock()
        builder.text_entity_extractor.deduplicate_entities.return_value = [test_entity]
        builder.text_relation_extractor = Mock()
        builder.text_relation_extractor.infer_implicit_relations.return_value = []

        database_schema = {"tables": ["test_table"]}
        result = builder.build_graph(database_schema=database_schema)

        self.assertEqual(len(result.entities), 1)
        self.assertIn(test_entity.id, result.entities)

    def test_update_graph_new_entities(self):
        """测试更新图-添加新实体"""
        graph = KnowledgeGraph()
        existing_entity = Entity(name="Existing", entity_type=EntityType.PERSON)
        graph.add_entity(existing_entity)

        new_entity = Entity(name="New", entity_type=EntityType.PERSON)
        new_entities = [new_entity]

        updated_graph = self.builder.update_graph(graph, new_entities=new_entities)

        self.assertEqual(len(updated_graph.entities), 2)
        self.assertIn(new_entity.id, updated_graph.entities)

    def test_update_graph_merge_existing_entity(self):
        """测试更新图-合并已存在实体"""
        graph = KnowledgeGraph()
        existing_entity = Entity(
            name="Alice",
            entity_type=EntityType.PERSON,
            properties={"age": 25}
        )
        graph.add_entity(existing_entity)

        # 使用相同ID但不同属性的实体
        updated_entity = Entity(
            id=existing_entity.id,  # 相同ID
            name="Alice",
            entity_type=EntityType.PERSON,
            properties={"location": "NYC"}
        )

        updated_graph = self.builder.update_graph(graph, new_entities=[updated_entity])

        # 实体数量不变
        self.assertEqual(len(updated_graph.entities), 1)
        # 属性应该被合并
        entity = updated_graph.get_entity(existing_entity.id)
        self.assertEqual(entity.properties["location"], "NYC")

    def test_update_graph_new_relations(self):
        """测试更新图-添加新关系"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="Alice", entity_type=EntityType.PERSON)
        entity2 = Entity(name="Bob", entity_type=EntityType.PERSON)
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        new_relation = Relation(
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.RELATED_TO
        )

        updated_graph = self.builder.update_graph(graph, new_relations=[new_relation])

        self.assertEqual(len(updated_graph.relations), 1)
        self.assertIn(new_relation.id, updated_graph.relations)


class TestMultiSourceGraphBuilder(unittest.TestCase):
    """测试多源图构建器"""

    def setUp(self):
        self.builder = MultiSourceGraphBuilder()

    def test_init(self):
        """测试初始化"""
        self.assertIn("database_schema", self.builder.source_weights)
        self.assertIn("text_extraction", self.builder.source_weights)
        self.assertEqual(self.builder.source_weights["database_schema"], 1.0)

    @patch.object(MultiSourceGraphBuilder, 'build_graph')
    def test_build_graph_from_multiple_sources(self, mock_build_graph):
        """测试从多个数据源构建图"""
        # 模拟不同数据源的子图
        text_graph = KnowledgeGraph(name="text_graph")
        text_entity = Entity(name="TextEntity", entity_type=EntityType.PERSON)
        text_graph.add_entity(text_entity)

        db_graph = KnowledgeGraph(name="db_graph")
        db_entity = Entity(name="DbEntity", entity_type=EntityType.TABLE)
        db_graph.add_entity(db_entity)

        # 设置mock返回值
        mock_build_graph.side_effect = [text_graph, db_graph]

        sources = [
            {
                "type": "text",
                "data": ["Sample text"],
                "name": "text_source",
                "weight": 1.0
            },
            {
                "type": "database",
                "data": {"tables": ["test_table"]},
                "name": "db_source",
                "weight": 0.8
            }
        ]

        result = self.builder.build_graph_from_multiple_sources(sources, "multi_graph")

        self.assertEqual(result.name, "multi_graph")
        # 应该合并两个子图的实体
        self.assertEqual(len(result.entities), 2)

    @patch.object(MultiSourceGraphBuilder, 'build_graph')
    def test_build_graph_from_mixed_source(self, mock_build_graph):
        """测试从混合数据源构建图"""
        mixed_graph = KnowledgeGraph(name="mixed_graph")
        entity = Entity(name="MixedEntity", entity_type=EntityType.CONCEPT)
        mixed_graph.add_entity(entity)

        mock_build_graph.return_value = mixed_graph

        sources = [
            {
                "type": "mixed",
                "data": {
                    "texts": ["Sample text"],
                    "database_schema": {"tables": ["test_table"]}
                },
                "name": "mixed_source"
            }
        ]

        result = self.builder.build_graph_from_multiple_sources(sources, "mixed_test_graph")

        self.assertEqual(result.name, "mixed_test_graph")
        mock_build_graph.assert_called_once()

    def test_build_graph_from_unknown_source_type(self):
        """测试处理未知数据源类型"""
        sources = [
            {
                "type": "unknown_type",
                "data": "some data",
                "name": "unknown_source"
            }
        ]

        with patch('agraph.builders.graph_builder.logger') as mock_logger:
            result = self.builder.build_graph_from_multiple_sources(sources, "test_graph")

            # 应该记录警告
            mock_logger.warning.assert_called_once()
            # 应该返回空图
            self.assertEqual(len(result.entities), 0)

    def test_apply_source_weights(self):
        """测试应用数据源权重"""
        graph = KnowledgeGraph()
        entity1 = Entity(
            name="TestEntity1",
            entity_type=EntityType.PERSON,
            confidence=1.0,
            source="text_extraction"
        )
        entity2 = Entity(
            name="TestEntity2",
            entity_type=EntityType.PERSON,
            confidence=1.0,
            source="text_extraction"
        )
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        relation = Relation(
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.RELATED_TO,
            confidence=1.0,
            source="text_extraction"
        )
        graph.add_relation(relation)

        # 应用权重
        weight = 0.5
        self.builder._apply_source_weights(graph, weight)

        # 检查置信度是否被正确调整
        expected_confidence = weight * self.builder.source_weights["text_extraction"]
        self.assertEqual(entity1.confidence, expected_confidence)
        self.assertEqual(entity2.confidence, expected_confidence)
        self.assertEqual(relation.confidence, expected_confidence)

    def test_empty_sources_list(self):
        """测试空数据源列表"""
        result = self.builder.build_graph_from_multiple_sources([], "empty_graph")

        self.assertEqual(result.name, "empty_graph")
        self.assertEqual(len(result.entities), 0)
        self.assertEqual(len(result.relations), 0)


class TestGraphBuilderErrorHandling(unittest.TestCase):
    """测试图构建器错误处理"""

    def setUp(self):
        self.builder = StandardGraphBuilder()

    @patch('agraph.builders.graph_builder.TextEntityExtractor')
    def test_build_graph_extraction_error(self, mock_extractor_class):
        """测试实体抽取时的错误处理"""
        mock_extractor = Mock()
        mock_extractor.extract_from_text.side_effect = Exception("Extraction failed")
        mock_extractor_class.return_value = mock_extractor

        builder = StandardGraphBuilder()
        builder.text_entity_extractor = mock_extractor

        with self.assertRaises(Exception):
            builder.build_graph(texts=["test text"])

    @patch('agraph.builders.graph_builder.logger')
    def test_update_graph_error_logging(self, mock_logger):
        """测试更新图时的错误日志"""
        graph = KnowledgeGraph()

        # 创建无效的新关系（没有实体）
        invalid_relation = Relation(
            head_entity=None,
            tail_entity=None,
            relation_type=RelationType.RELATED_TO
        )

        # 这不应该抛出异常，而是静默处理
        result = self.builder.update_graph(graph, new_relations=[invalid_relation])

        # 图应该保持不变
        self.assertEqual(len(result.relations), 0)


if __name__ == "__main__":
    unittest.main()
