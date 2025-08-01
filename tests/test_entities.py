import uuid
import unittest
from datetime import datetime
from agraph.entities import Entity
from agraph.types import EntityType


class TestEntity(unittest.TestCase):

    def test_entity_creation_default(self):
        """测试实体的默认创建"""
        entity = Entity()

        self.assertIsNotNone(entity.id)
        self.assertIsInstance(entity.id, str)
        self.assertEqual(entity.name, "")
        self.assertEqual(entity.entity_type, EntityType.UNKNOWN)
        self.assertEqual(entity.description, "")
        self.assertEqual(entity.properties, {})
        self.assertEqual(entity.aliases, [])
        self.assertEqual(entity.confidence, 1.0)
        self.assertEqual(entity.source, "")
        self.assertIsInstance(entity.created_at, datetime)

    def test_entity_creation_with_params(self):
        """测试带参数的实体创建"""
        test_id = str(uuid.uuid4())
        test_name = "测试实体"
        test_type = EntityType.PERSON
        test_description = "这是一个测试实体"
        test_properties = {"age": 30, "city": "北京"}
        test_aliases = ["别名1", "别名2"]
        test_confidence = 0.8
        test_source = "测试源"

        entity = Entity(
            id=test_id,
            name=test_name,
            entity_type=test_type,
            description=test_description,
            properties=test_properties,
            aliases=test_aliases,
            confidence=test_confidence,
            source=test_source
        )

        self.assertEqual(entity.id, test_id)
        self.assertEqual(entity.name, test_name)
        self.assertEqual(entity.entity_type, test_type)
        self.assertEqual(entity.description, test_description)
        self.assertEqual(entity.properties, test_properties)
        self.assertEqual(entity.aliases, test_aliases)
        self.assertEqual(entity.confidence, test_confidence)
        self.assertEqual(entity.source, test_source)

    def test_entity_hash(self):
        """测试实体的哈希功能"""
        entity1 = Entity(id="test-id-1", name="实体1")
        entity2 = Entity(id="test-id-2", name="实体2")
        entity3 = Entity(id="test-id-1", name="实体3")  # 相同ID，不同name

        # 相同ID的实体应该有相同的哈希值
        self.assertEqual(hash(entity1), hash(entity3))
        # 不同ID的实体应该有不同的哈希值
        self.assertNotEqual(hash(entity1), hash(entity2))

    def test_entity_equality(self):
        """测试实体的相等性比较"""
        entity1 = Entity(id="test-id-1", name="实体1")
        entity2 = Entity(id="test-id-2", name="实体2")
        entity3 = Entity(id="test-id-1", name="实体3")  # 相同ID，不同name

        # 相同ID的实体应该相等
        self.assertEqual(entity1, entity3)
        # 不同ID的实体应该不相等
        self.assertNotEqual(entity1, entity2)
        # 与非Entity对象比较应该返回NotImplemented
        self.assertNotEqual(entity1, "not an entity")

    def test_add_alias(self):
        """测试添加别名功能"""
        entity = Entity(name="原始名称")

        # 添加新别名
        entity.add_alias("别名1")
        self.assertIn("别名1", entity.aliases)
        self.assertEqual(len(entity.aliases), 1)

        # 添加另一个别名
        entity.add_alias("别名2")
        self.assertIn("别名2", entity.aliases)
        self.assertEqual(len(entity.aliases), 2)

        # 尝试添加重复别名
        entity.add_alias("别名1")
        self.assertEqual(len(entity.aliases), 2)  # 不应该增加

        # 尝试添加空别名
        entity.add_alias("")
        entity.add_alias(None)
        self.assertEqual(len(entity.aliases), 2)  # 不应该增加

    def test_add_property(self):
        """测试添加属性功能"""
        entity = Entity()

        entity.add_property("age", 25)
        entity.add_property("city", "上海")
        entity.add_property("active", True)

        self.assertEqual(entity.properties["age"], 25)
        self.assertEqual(entity.properties["city"], "上海")
        self.assertTrue(entity.properties["active"])
        self.assertEqual(len(entity.properties), 3)

    def test_get_property(self):
        """测试获取属性功能"""
        entity = Entity()
        entity.add_property("age", 25)
        entity.add_property("city", "广州")

        # 获取存在的属性
        self.assertEqual(entity.get_property("age"), 25)
        self.assertEqual(entity.get_property("city"), "广州")

        # 获取不存在的属性（默认值为None）
        self.assertIsNone(entity.get_property("unknown"))

        # 获取不存在的属性（自定义默认值）
        self.assertEqual(entity.get_property("unknown", "default"), "default")

    def test_to_dict(self):
        """测试转换为字典功能"""
        test_id = "test-id"
        test_name = "测试实体"
        test_type = EntityType.ORGANIZATION
        test_description = "测试描述"
        test_properties = {"location": "深圳", "size": "large"}
        test_aliases = ["公司", "企业"]
        test_confidence = 0.9
        test_source = "测试数据源"

        entity = Entity(
            id=test_id,
            name=test_name,
            entity_type=test_type,
            description=test_description,
            properties=test_properties,
            aliases=test_aliases,
            confidence=test_confidence,
            source=test_source
        )

        result_dict = entity.to_dict()

        self.assertEqual(result_dict["id"], test_id)
        self.assertEqual(result_dict["name"], test_name)
        self.assertEqual(result_dict["entity_type"], test_type.value)
        self.assertEqual(result_dict["description"], test_description)
        self.assertEqual(result_dict["properties"], test_properties)
        self.assertEqual(result_dict["aliases"], test_aliases)
        self.assertEqual(result_dict["confidence"], test_confidence)
        self.assertEqual(result_dict["source"], test_source)
        self.assertIn("created_at", result_dict)
        self.assertIsInstance(result_dict["created_at"], str)

    def test_from_dict(self):
        """测试从字典创建实体功能"""
        test_data = {
            "id": "test-id",
            "name": "从字典创建的实体",
            "entity_type": "person",
            "description": "测试描述",
            "properties": {"hobby": "reading", "score": 95},
            "aliases": ["张三", "小张"],
            "confidence": 0.85,
            "source": "字典数据源",
            "created_at": "2023-01-01T12:00:00"
        }

        entity = Entity.from_dict(test_data)

        self.assertEqual(entity.id, "test-id")
        self.assertEqual(entity.name, "从字典创建的实体")
        self.assertEqual(entity.entity_type, EntityType.PERSON)
        self.assertEqual(entity.description, "测试描述")
        self.assertEqual(entity.properties, {"hobby": "reading", "score": 95})
        self.assertEqual(entity.aliases, ["张三", "小张"])
        self.assertEqual(entity.confidence, 0.85)
        self.assertEqual(entity.source, "字典数据源")
        self.assertEqual(entity.created_at, datetime.fromisoformat("2023-01-01T12:00:00"))

    def test_from_dict_minimal(self):
        """测试从最小字典创建实体"""
        minimal_data = {"name": "最小实体"}

        entity = Entity.from_dict(minimal_data)

        self.assertEqual(entity.name, "最小实体")
        self.assertEqual(entity.entity_type, EntityType.UNKNOWN)
        self.assertEqual(entity.description, "")
        self.assertEqual(entity.properties, {})
        self.assertEqual(entity.aliases, [])
        self.assertEqual(entity.confidence, 1.0)
        self.assertEqual(entity.source, "")
        self.assertIsInstance(entity.id, str)
        self.assertIsInstance(entity.created_at, datetime)

    def test_from_dict_empty(self):
        """测试从空字典创建实体"""
        empty_data = {}

        entity = Entity.from_dict(empty_data)

        self.assertEqual(entity.name, "")
        self.assertEqual(entity.entity_type, EntityType.UNKNOWN)
        self.assertEqual(entity.description, "")
        self.assertEqual(entity.properties, {})
        self.assertEqual(entity.aliases, [])
        self.assertEqual(entity.confidence, 1.0)
        self.assertEqual(entity.source, "")
        self.assertIsInstance(entity.id, str)
        self.assertIsInstance(entity.created_at, datetime)

    def test_entity_round_trip(self):
        """测试实体的序列化和反序列化往返"""
        original_entity = Entity(
            name="往返测试实体",
            entity_type=EntityType.CONCEPT,
            description="用于测试往返转换",
            properties={"key1": "value1", "key2": 42},
            aliases=["别名A", "别名B"],
            confidence=0.75,
            source="往返测试源"
        )

        # 转换为字典再转换回实体
        entity_dict = original_entity.to_dict()
        restored_entity = Entity.from_dict(entity_dict)

        # 验证所有字段都正确保持
        self.assertEqual(restored_entity.id, original_entity.id)
        self.assertEqual(restored_entity.name, original_entity.name)
        self.assertEqual(restored_entity.entity_type, original_entity.entity_type)
        self.assertEqual(restored_entity.description, original_entity.description)
        self.assertEqual(restored_entity.properties, original_entity.properties)
        self.assertEqual(restored_entity.aliases, original_entity.aliases)
        self.assertEqual(restored_entity.confidence, original_entity.confidence)
        self.assertEqual(restored_entity.source, original_entity.source)
        self.assertEqual(restored_entity.created_at, original_entity.created_at)

    def test_all_entity_types(self):
        """测试所有实体类型"""
        for entity_type in EntityType:
            entity = Entity(
                name=f"测试{entity_type.value}实体",
                entity_type=entity_type
            )
            self.assertEqual(entity.entity_type, entity_type)

            # 测试序列化和反序列化
            entity_dict = entity.to_dict()
            restored_entity = Entity.from_dict(entity_dict)
            self.assertEqual(restored_entity.entity_type, entity_type)


if __name__ == '__main__':
    unittest.main()
