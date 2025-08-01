import pytest
import unittest
from agraph.extractors.entity_extractor import (
    BaseEntityExtractor,
    TextEntityExtractor,
    DatabaseEntityExtractor,
)
from agraph.entities import Entity
from agraph.types import EntityType


class TestBaseEntityExtractor(unittest.TestCase):
    """测试实体提取器基类"""

    def setUp(self):
        """测试前的设置"""
        self.extractor = TextEntityExtractor()

    def test_extractor_initialization(self):
        """测试实体提取器基本功能"""
        self.assertIsNotNone(self.extractor)
        self.assertIsInstance(self.extractor, BaseEntityExtractor)
        self.assertTrue(hasattr(self.extractor, 'extract_from_text'))
        self.assertTrue(hasattr(self.extractor, 'extract_from_database'))

    def test_confidence_calculation(self):
        """测试置信度计算"""
        # 测试不同类型的实体名称
        confidence_short = self.extractor._calculate_entity_confidence("a")
        confidence_normal = self.extractor._calculate_entity_confidence("John")
        confidence_long = self.extractor._calculate_entity_confidence("John Smith")
        confidence_stopword = self.extractor._calculate_entity_confidence("the")

        # 短名称应该有最低置信度
        self.assertLess(confidence_short, confidence_normal)

        # 验证具体的置信度值
        self.assertGreaterEqual(confidence_normal, 0.6)  # 4个字符以上，首字母大写
        self.assertGreaterEqual(confidence_long, 0.6)   # 同样适用

        # 停用词应该有很低的置信度
        self.assertLess(confidence_stopword, 0.5)

    def test_entity_normalization(self):
        """测试实体标准化功能"""
        entity = Entity(
            name="  JOHN SMITH  ",
            aliases=["  Johnny  ", "J. Smith", "  Johnny  ", ""],  # 包含重复和空值
            entity_type=EntityType.PERSON
        )

        normalized = self.extractor.normalize_entity(entity)

        # 名称应该被标准化
        self.assertEqual(normalized.name, "john smith")

        # 别名应该去重和标准化
        self.assertIn("johnny", normalized.aliases)
        self.assertIn("j. smith", normalized.aliases)
        self.assertNotIn("", normalized.aliases)
        self.assertEqual(len(normalized.aliases), 2)  # 去重后只有2个

    def test_entity_deduplication(self):
        """测试实体去重功能"""
        # 创建重复实体
        entities = [
            Entity(name="John Smith", entity_type=EntityType.PERSON, confidence=0.8),
            Entity(name="john smith", entity_type=EntityType.PERSON, confidence=0.9),  # 相同但大小写不同
            Entity(name="Mary Johnson", entity_type=EntityType.PERSON, confidence=0.7),
        ]

        deduplicated = self.extractor.deduplicate_entities(entities)

        # 应该去重相同的实体
        self.assertEqual(len(deduplicated), 2)

        # 检查去重后的实体名称
        names = [e.name.lower() for e in deduplicated]
        self.assertIn("john smith", names)
        self.assertIn("mary johnson", names)

        # 验证john smith实体保留了更高置信度
        john_entity = next(e for e in deduplicated if e.name.lower() == "john smith")
        self.assertEqual(john_entity.confidence, 0.9)


class TestTextEntityExtractor(unittest.TestCase):
    """测试文本实体提取器"""

    def setUp(self):
        """测试前的设置"""
        self.extractor = TextEntityExtractor()

        """测试文本实体提取器 - 人名提取"""
        # 测试常见人名模式
        text = "John Smith and Mary Johnson are working on the project."
        entities = self.extractor.extract_from_text(text)

        person_entities = [e for e in entities if e.entity_type == EntityType.PERSON]
        self.assertGreaterEqual(len(person_entities), 1)

        # 验证提取的人名
        person_names = [e.name for e in person_entities]
        self.assertTrue(any("John Smith" in name for name in person_names))

    def test_organization_extraction(self):
        """测试文本实体提取器 - 组织提取"""
        # 测试组织名称模式
        text = "Apple Inc and Microsoft Corp are leading technology companies."
        entities = self.extractor.extract_from_text(text)

        org_entities = [e for e in entities if e.entity_type == EntityType.ORGANIZATION]
        self.assertGreaterEqual(len(org_entities), 1)

        # 验证提取的组织名
        org_names = [e.name for e in org_entities]
        self.assertTrue(any("Apple Inc" in name for name in org_names))

    def test_location_extraction(self):
        """测试文本实体提取器 - 地点提取"""
        # 测试地点模式
        text = "The meeting will be held in New York City and Los Angeles."
        entities = self.extractor.extract_from_text(text)

        location_entities = [e for e in entities if e.entity_type == EntityType.LOCATION]
        self.assertGreaterEqual(len(location_entities), 1)

    def test_concept_extraction(self):
        """测试文本实体提取器 - 概念提取"""
        # 测试概念词汇
        text = "Machine learning concept and artificial intelligence theory are important."
        entities = self.extractor.extract_from_text(text)

        concept_entities = [e for e in entities if e.entity_type == EntityType.CONCEPT]
        self.assertGreaterEqual(len(concept_entities), 1)

    def test_confidence_threshold_filtering(self):
        """测试置信度阈值过滤"""
        self.extractor.confidence_threshold = 0.8  # 设置较高阈值

        text = "a b c"  # 短词，低置信度
        entities = self.extractor.extract_from_text(text)

        # 应该过滤掉低置信度的实体
        self.assertEqual(len(entities), 0)

    def test_empty_text_handling(self):
        """测试空文本处理"""
        entities = self.extractor.extract_from_text("")
        self.assertEqual(entities, [])

        entities = self.extractor.extract_from_text("   ")
        self.assertEqual(entities, [])

    def test_stopwords_filtering(self):
        """测试停用词过滤"""
        # 停用词应该被过滤
        text = "the and or but"
        entities = self.extractor.extract_from_text(text)

        # 停用词不应该被提取为实体
        entity_names = [e.name.lower() for e in entities]
        self.assertNotIn("the", entity_names)
        self.assertNotIn("and", entity_names)

    def test_entity_properties_preservation(self):
        """测试实体属性保存"""
        text = "Dr. Smith works at Apple Inc."
        entities = self.extractor.extract_from_text(text)

        # 检查实体是否包含位置和上下文信息
        for entity in entities:
            self.assertIn("position", entity.properties)
            self.assertIn("context", entity.properties)
            self.assertEqual(entity.source, "text_extraction")

    def test_concept_keyword_extraction(self):
        """测试概念关键词提取"""
        text = "Machine learning algorithms are used in artificial intelligence systems. " \
               "Machine learning is a subset of artificial intelligence."

        keywords = self.extractor._extract_concept_keywords(text)

        # 应该提取高频关键词
        self.assertGreater(len(keywords), 0)
        self.assertTrue("machine" in keywords or "learning" in keywords)
        self.assertTrue("artificial" in keywords or "intelligence" in keywords)

    def test_multiple_entity_types_in_text(self):
        """测试混合实体类型提取"""
        text = """
        Dr. John Smith from Stanford University is researching machine learning concepts.
        He works with Microsoft Corp in Silicon Valley.
        """

        entities = self.extractor.extract_from_text(text)

        # 应该提取多种类型的实体
        entity_types = {e.entity_type for e in entities}
        self.assertGreater(len(entity_types), 1)  # 至少有2种不同类型的实体

        # 验证每种类型都有对应的实体
        type_counts = {}
        for entity in entities:
            type_counts[entity.entity_type] = type_counts.get(entity.entity_type, 0) + 1

        self.assertGreater(len(type_counts), 0)

    def test_error_handling(self):
        """测试错误处理"""
        # 测试异常情况不会崩溃
        entities = self.extractor.extract_from_text(None)  # 可能会引发异常
        # 应该返回空列表而不是崩溃
        self.assertIsInstance(entities, list)


class TestDatabaseEntityExtractor(unittest.TestCase):
    """测试数据库实体提取器"""

    def setUp(self):
        """测试前的设置"""
        self.extractor = DatabaseEntityExtractor()


    def test_basic_functionality(self):
        """测试数据库实体提取器基本功能"""
        # 数据库提取器不应该处理文本
        entities = self.extractor.extract_from_text("some text")
        self.assertEqual(entities, [])

    def test_schema_extraction(self):
        """测试数据库模式实体提取"""
        schema = {
            "database_name": "test_db",
            "description": "测试数据库",
            "tables": [
                {
                    "name": "users",
                    "comment": "用户表",
                    "schema": "public",
                    "row_count": 1000,
                    "columns": [
                        {
                            "name": "id",
                            "type": "integer",
                            "primary_key": True,
                            "nullable": False
                        },
                        {
                            "name": "username",
                            "type": "varchar(100)",
                            "unique": True,
                            "nullable": False
                        },
                        {
                            "name": "email",
                            "type": "varchar(255)",
                            "nullable": False
                        }
                    ]
                }
            ]
        }

        entities = self.extractor.extract_from_database(schema)

        # 应该提取数据库、表和列实体
        db_entities = [e for e in entities if e.entity_type == EntityType.DATABASE]
        table_entities = [e for e in entities if e.entity_type == EntityType.TABLE]
        column_entities = [e for e in entities if e.entity_type == EntityType.COLUMN]

        self.assertEqual(len(db_entities), 1)
        self.assertEqual(db_entities[0].name, "test_db")

        self.assertEqual(len(table_entities), 1)
        self.assertEqual(table_entities[0].name, "users")

        # 应该排除通用列（如id），只提取业务相关列
        self.assertGreaterEqual(len(column_entities), 2)
        column_names = [e.name for e in column_entities]
        self.assertTrue(any("username" in name for name in column_names))
        self.assertTrue(any("email" in name for name in column_names))

    def test_business_concepts_extraction(self):
        """测试业务概念提取"""
        schema = {
            "database_name": "ecommerce_db",
            "tables": [
                {"name": "users", "comment": "用户表"},
                {"name": "orders", "comment": "订单表"},
                {"name": "products", "comment": "产品表"},
                {"name": "categories", "comment": "分类表"}
            ]
        }

        entities = self.extractor.extract_from_database(schema)

        # 应该提取业务概念
        concept_entities = [e for e in entities if e.entity_type == EntityType.CONCEPT]
        self.assertGreater(len(concept_entities), 0)

        concept_names = [e.name for e in concept_entities]
        # 基于表名应该推断出相关业务概念
        expected_concepts = ["User Management", "Order Management", "Product Management"]
        for concept in expected_concepts:
            self.assertIn(concept, concept_names)

    def test_table_name_cleaning(self):
        """测试表名清理功能"""
        # 测试各种表名前缀
        self.assertEqual(self.extractor._clean_table_name("tbl_users"), "users")
        self.assertEqual(self.extractor._clean_table_name("tb_products"), "products")
        self.assertEqual(self.extractor._clean_table_name("t_orders"), "orders")
        self.assertEqual(self.extractor._clean_table_name("customers"), "customers")  # 无前缀

    def test_error_handling(self):
        """测试错误处理"""
        # 测试无效的数据库模式
        entities = self.extractor.extract_from_database({})
        self.assertIsInstance(entities, list)


if __name__ == '__main__':
    unittest.main()
