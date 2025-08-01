import uuid
import unittest
from datetime import datetime
from agraph.entities import Entity
from agraph.relations import Relation
from agraph.types import EntityType, RelationType


class TestRelation(unittest.TestCase):

    def test_relation_creation_default(self):
        """测试关系的默认创建"""
        relation = Relation()

        self.assertIsNotNone(relation.id)
        self.assertIsInstance(relation.id, str)
        self.assertIsNone(relation.head_entity)
        self.assertIsNone(relation.tail_entity)
        self.assertEqual(relation.relation_type, RelationType.RELATED_TO)
        self.assertEqual(relation.properties, {})
        self.assertEqual(relation.confidence, 1.0)
        self.assertEqual(relation.source, "")
        self.assertIsInstance(relation.created_at, datetime)

    def test_relation_creation_with_entities(self):
        """测试带实体的关系创建"""
        head_entity = Entity(name="实体1", entity_type=EntityType.PERSON)
        tail_entity = Entity(name="实体2", entity_type=EntityType.ORGANIZATION)

        relation = Relation(
            head_entity=head_entity,
            tail_entity=tail_entity,
            relation_type=RelationType.BELONGS_TO
        )

        self.assertEqual(relation.head_entity, head_entity)
        self.assertEqual(relation.tail_entity, tail_entity)
        self.assertEqual(relation.relation_type, RelationType.BELONGS_TO)

    def test_relation_creation_with_params(self):
        """测试带完整参数的关系创建"""
        test_id = str(uuid.uuid4())
        head_entity = Entity(name="张三", entity_type=EntityType.PERSON)
        tail_entity = Entity(name="某公司", entity_type=EntityType.ORGANIZATION)
        test_properties = {"position": "员工", "start_date": "2023-01-01"}
        test_confidence = 0.9
        test_source = "测试源"

        relation = Relation(
            id=test_id,
            head_entity=head_entity,
            tail_entity=tail_entity,
            relation_type=RelationType.BELONGS_TO,
            properties=test_properties,
            confidence=test_confidence,
            source=test_source
        )

        self.assertEqual(relation.id, test_id)
        self.assertEqual(relation.head_entity, head_entity)
        self.assertEqual(relation.tail_entity, tail_entity)
        self.assertEqual(relation.relation_type, RelationType.BELONGS_TO)
        self.assertEqual(relation.properties, test_properties)
        self.assertEqual(relation.confidence, test_confidence)
        self.assertEqual(relation.source, test_source)

    def test_relation_hash(self):
        """测试关系的哈希功能"""
        relation1 = Relation(id="test-id-1")
        relation2 = Relation(id="test-id-2")
        relation3 = Relation(id="test-id-1")  # 相同ID

        # 相同ID的关系应该有相同的哈希值
        self.assertEqual(hash(relation1), hash(relation3))
        # 不同ID的关系应该有不同的哈希值
        self.assertNotEqual(hash(relation1), hash(relation2))

    def test_relation_equality(self):
        """测试关系的相等性比较"""
        relation1 = Relation(id="test-id-1")
        relation2 = Relation(id="test-id-2")
        relation3 = Relation(id="test-id-1")  # 相同ID

        # 相同ID的关系应该相等
        self.assertEqual(relation1, relation3)
        # 不同ID的关系应该不相等
        self.assertNotEqual(relation1, relation2)
        # 与非Relation对象比较应该返回False
        self.assertNotEqual(relation1, "not a relation")

    def test_add_property(self):
        """测试添加属性功能"""
        relation = Relation()

        relation.add_property("weight", 0.8)
        relation.add_property("type", "strong")
        relation.add_property("verified", True)

        self.assertEqual(relation.properties["weight"], 0.8)
        self.assertEqual(relation.properties["type"], "strong")
        self.assertTrue(relation.properties["verified"])
        self.assertEqual(len(relation.properties), 3)

    def test_get_property(self):
        """测试获取属性功能"""
        relation = Relation()
        relation.add_property("importance", 5)
        relation.add_property("category", "business")

        # 获取存在的属性
        self.assertEqual(relation.get_property("importance"), 5)
        self.assertEqual(relation.get_property("category"), "business")

        # 获取不存在的属性（默认值为None）
        self.assertIsNone(relation.get_property("unknown"))

        # 获取不存在的属性（自定义默认值）
        self.assertEqual(relation.get_property("unknown", "default"), "default")

    def test_is_valid(self):
        """测试关系有效性验证"""
        entity1 = Entity(name="实体1")
        entity2 = Entity(name="实体2")

        # 无实体的关系无效
        relation1 = Relation()
        self.assertFalse(relation1.is_valid())

        # 只有头实体的关系无效
        relation2 = Relation(head_entity=entity1)
        self.assertFalse(relation2.is_valid())

        # 只有尾实体的关系无效
        relation3 = Relation(tail_entity=entity2)
        self.assertFalse(relation3.is_valid())

        # 自环关系无效
        relation4 = Relation(head_entity=entity1, tail_entity=entity1)
        self.assertFalse(relation4.is_valid())

        # 有效关系
        relation5 = Relation(head_entity=entity1, tail_entity=entity2)
        self.assertTrue(relation5.is_valid())

    def test_reverse_relation(self):
        """测试反向关系创建"""
        head_entity = Entity(name="部门", entity_type=EntityType.ORGANIZATION)
        tail_entity = Entity(name="员工", entity_type=EntityType.PERSON)

        original_relation = Relation(
            head_entity=head_entity,
            tail_entity=tail_entity,
            relation_type=RelationType.CONTAINS,
            properties={"role": "manager"},
            confidence=0.95,
            source="HR系统"
        )

        reversed_relation = original_relation.reverse()

        # 验证实体位置互换
        self.assertEqual(reversed_relation.head_entity, tail_entity)
        self.assertEqual(reversed_relation.tail_entity, head_entity)

        # 验证关系类型正确反转
        self.assertEqual(reversed_relation.relation_type, RelationType.BELONGS_TO)

        # 验证其他属性保持不变
        self.assertEqual(reversed_relation.properties, original_relation.properties)
        self.assertEqual(reversed_relation.confidence, original_relation.confidence)
        self.assertEqual(reversed_relation.source, original_relation.source)

        # 验证是新的关系对象
        self.assertNotEqual(reversed_relation.id, original_relation.id)

    def test_reverse_relation_types(self):
        """测试各种关系类型的反转"""
        entity1 = Entity(name="实体1")
        entity2 = Entity(name="实体2")

        # 测试对称关系类型（反转后保持不变）
        symmetric_types = [
            RelationType.REFERENCES,
            RelationType.SIMILAR_TO,
            RelationType.SYNONYMS,
            RelationType.RELATED_TO,  # 默认情况
            RelationType.MENTIONS,    # 不在反转映射中的类型
        ]

        for relation_type in symmetric_types:
            relation = Relation(
                head_entity=entity1,
                tail_entity=entity2,
                relation_type=relation_type
            )
            reversed_relation = relation.reverse()

            if relation_type in [RelationType.REFERENCES, RelationType.SIMILAR_TO, RelationType.SYNONYMS]:
                self.assertEqual(reversed_relation.relation_type, relation_type)
            else:
                # 不在映射中的类型应该保持原样
                self.assertEqual(reversed_relation.relation_type, relation_type)

        # 测试非对称关系类型
        contains_relation = Relation(
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.CONTAINS
        )
        belongs_to_relation = contains_relation.reverse()
        self.assertEqual(belongs_to_relation.relation_type, RelationType.BELONGS_TO)

        belongs_to_relation_orig = Relation(
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.BELONGS_TO
        )
        contains_relation_rev = belongs_to_relation_orig.reverse()
        self.assertEqual(contains_relation_rev.relation_type, RelationType.CONTAINS)

    def test_to_dict(self):
        """测试转换为字典功能"""
        head_entity = Entity(id="head-id", name="头实体")
        tail_entity = Entity(id="tail-id", name="尾实体")
        test_id = "test-relation-id"
        test_properties = {"strength": 0.8, "type": "direct"}
        test_confidence = 0.85
        test_source = "测试数据源"

        relation = Relation(
            id=test_id,
            head_entity=head_entity,
            tail_entity=tail_entity,
            relation_type=RelationType.DEPENDS_ON,
            properties=test_properties,
            confidence=test_confidence,
            source=test_source
        )

        result_dict = relation.to_dict()

        self.assertEqual(result_dict["id"], test_id)
        self.assertEqual(result_dict["head_entity_id"], "head-id")
        self.assertEqual(result_dict["tail_entity_id"], "tail-id")
        self.assertEqual(result_dict["relation_type"], RelationType.DEPENDS_ON.value)
        self.assertEqual(result_dict["properties"], test_properties)
        self.assertEqual(result_dict["confidence"], test_confidence)
        self.assertEqual(result_dict["source"], test_source)
        self.assertIn("created_at", result_dict)
        self.assertIsInstance(result_dict["created_at"], str)

    def test_to_dict_with_none_entities(self):
        """测试含空实体的关系转换为字典"""
        relation = Relation(
            relation_type=RelationType.MENTIONS,
            confidence=0.5
        )

        result_dict = relation.to_dict()

        self.assertIsNone(result_dict["head_entity_id"])
        self.assertIsNone(result_dict["tail_entity_id"])
        self.assertEqual(result_dict["relation_type"], RelationType.MENTIONS.value)
        self.assertEqual(result_dict["confidence"], 0.5)

    def test_from_dict(self):
        """测试从字典创建关系功能"""
        head_entity = Entity(id="head-id", name="头实体")
        tail_entity = Entity(id="tail-id", name="尾实体")
        entities_map = {
            "head-id": head_entity,
            "tail-id": tail_entity
        }

        test_data = {
            "id": "test-relation-id",
            "head_entity_id": "head-id",
            "tail_entity_id": "tail-id",
            "relation_type": "foreign_key",
            "properties": {"table": "users", "column": "department_id"},
            "confidence": 0.9,
            "source": "数据库分析",
            "created_at": "2023-06-15T10:30:00"
        }

        relation = Relation.from_dict(test_data, entities_map)

        self.assertEqual(relation.id, "test-relation-id")
        self.assertEqual(relation.head_entity, head_entity)
        self.assertEqual(relation.tail_entity, tail_entity)
        self.assertEqual(relation.relation_type, RelationType.FOREIGN_KEY)
        self.assertEqual(relation.properties, {"table": "users", "column": "department_id"})
        self.assertEqual(relation.confidence, 0.9)
        self.assertEqual(relation.source, "数据库分析")
        self.assertEqual(relation.created_at, datetime.fromisoformat("2023-06-15T10:30:00"))

    def test_from_dict_minimal(self):
        """测试从最小字典创建关系"""
        entities_map = {}
        minimal_data = {"relation_type": "similar_to"}

        relation = Relation.from_dict(minimal_data, entities_map)

        self.assertIsNone(relation.head_entity)
        self.assertIsNone(relation.tail_entity)
        self.assertEqual(relation.relation_type, RelationType.SIMILAR_TO)
        self.assertEqual(relation.properties, {})
        self.assertEqual(relation.confidence, 1.0)
        self.assertEqual(relation.source, "")
        self.assertIsInstance(relation.id, str)
        self.assertIsInstance(relation.created_at, datetime)

    def test_from_dict_missing_entities(self):
        """测试从字典创建关系时实体不存在的情况"""
        entities_map = {}  # 空的实体映射

        test_data = {
            "head_entity_id": "missing-head",
            "tail_entity_id": "missing-tail",
            "relation_type": "mentions"
        }

        relation = Relation.from_dict(test_data, entities_map)

        self.assertIsNone(relation.head_entity)
        self.assertIsNone(relation.tail_entity)
        self.assertEqual(relation.relation_type, RelationType.MENTIONS)

    def test_from_dict_empty(self):
        """测试从空字典创建关系"""
        entities_map = {}
        empty_data = {}

        relation = Relation.from_dict(empty_data, entities_map)

        self.assertIsNone(relation.head_entity)
        self.assertIsNone(relation.tail_entity)
        self.assertEqual(relation.relation_type, RelationType.RELATED_TO)
        self.assertEqual(relation.properties, {})
        self.assertEqual(relation.confidence, 1.0)
        self.assertEqual(relation.source, "")
        self.assertIsInstance(relation.id, str)
        self.assertIsInstance(relation.created_at, datetime)

    def test_relation_round_trip(self):
        """测试关系的序列化和反序列化往返"""
        head_entity = Entity(id="head-123", name="往返头实体")
        tail_entity = Entity(id="tail-456", name="往返尾实体")
        entities_map = {
            "head-123": head_entity,
            "tail-456": tail_entity
        }

        original_relation = Relation(
            head_entity=head_entity,
            tail_entity=tail_entity,
            relation_type=RelationType.DESCRIBES,
            properties={"detail": "详细描述", "level": 3},
            confidence=0.88,
            source="往返测试源"
        )

        # 转换为字典再转换回关系
        relation_dict = original_relation.to_dict()
        restored_relation = Relation.from_dict(relation_dict, entities_map)

        # 验证所有字段都正确保持
        self.assertEqual(restored_relation.id, original_relation.id)
        self.assertEqual(restored_relation.head_entity, original_relation.head_entity)
        self.assertEqual(restored_relation.tail_entity, original_relation.tail_entity)
        self.assertEqual(restored_relation.relation_type, original_relation.relation_type)
        self.assertEqual(restored_relation.properties, original_relation.properties)
        self.assertEqual(restored_relation.confidence, original_relation.confidence)
        self.assertEqual(restored_relation.source, original_relation.source)
        self.assertEqual(restored_relation.created_at, original_relation.created_at)

    def test_all_relation_types(self):
        """测试所有关系类型"""
        entity1 = Entity(name="实体1")
        entity2 = Entity(name="实体2")
        entities_map = {entity1.id: entity1, entity2.id: entity2}

        for relation_type in RelationType:
            relation = Relation(
                head_entity=entity1,
                tail_entity=entity2,
                relation_type=relation_type
            )
            self.assertEqual(relation.relation_type, relation_type)

            # 测试序列化和反序列化
            relation_dict = relation.to_dict()
            restored_relation = Relation.from_dict(relation_dict, entities_map)
            self.assertEqual(restored_relation.relation_type, relation_type)


if __name__ == '__main__':
    unittest.main()
