import uuid
import unittest
from datetime import datetime
from agraph.entities import Entity
from agraph.relations import Relation
from agraph.graph import KnowledgeGraph
from agraph.types import EntityType, RelationType


class TestKnowledgeGraph(unittest.TestCase):

    def test_knowledge_graph_creation(self):
        """测试知识图谱的创建"""
        graph = KnowledgeGraph()

        self.assertIsNotNone(graph.id)
        self.assertIsInstance(graph.id, str)
        self.assertEqual(graph.name, "")
        self.assertEqual(graph.entities, {})
        self.assertEqual(graph.relations, {})
        self.assertEqual(graph.entity_index, {})
        self.assertEqual(graph.relation_index, {})
        self.assertIsInstance(graph.created_at, datetime)
        self.assertIsInstance(graph.updated_at, datetime)

    def test_knowledge_graph_creation_with_params(self):
        """测试带参数的知识图谱创建"""
        test_id = str(uuid.uuid4())
        test_name = "测试知识图谱"

        graph = KnowledgeGraph(id=test_id, name=test_name)

        self.assertEqual(graph.id, test_id)
        self.assertEqual(graph.name, test_name)

    def test_add_entity(self):
        """测试添加实体"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="实体1", entity_type=EntityType.PERSON)
        entity2 = Entity(name="实体2", entity_type=EntityType.ORGANIZATION)

        # 添加第一个实体
        result1 = graph.add_entity(entity1)
        self.assertTrue(result1)
        self.assertIn(entity1.id, graph.entities)
        self.assertEqual(graph.entities[entity1.id], entity1)
        self.assertIn(EntityType.PERSON.value, graph.entity_index)
        self.assertIn(entity1.id, graph.entity_index[EntityType.PERSON.value])

        # 添加第二个实体
        result2 = graph.add_entity(entity2)
        self.assertTrue(result2)
        self.assertIn(entity2.id, graph.entities)
        self.assertEqual(len(graph.entities), 2)

        # 尝试添加重复实体
        result3 = graph.add_entity(entity1)
        self.assertFalse(result3)
        self.assertEqual(len(graph.entities), 2)

    def test_add_relation(self):
        """测试添加关系"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="张三", entity_type=EntityType.PERSON)
        entity2 = Entity(name="某公司", entity_type=EntityType.ORGANIZATION)

        # 先添加实体
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        # 创建有效关系
        relation = Relation(
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.BELONGS_TO
        )

        # 添加关系
        result = graph.add_relation(relation)
        self.assertTrue(result)
        self.assertIn(relation.id, graph.relations)
        self.assertEqual(graph.relations[relation.id], relation)
        self.assertIn(RelationType.BELONGS_TO.value, graph.relation_index)
        self.assertIn(relation.id, graph.relation_index[RelationType.BELONGS_TO.value])

    def test_add_relation_invalid(self):
        """测试添加无效关系"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="实体1")
        entity2 = Entity(name="实体2")

        # 实体不在图中的关系
        relation1 = Relation(head_entity=entity1, tail_entity=entity2)
        result1 = graph.add_relation(relation1)
        self.assertFalse(result1)

        # 添加实体
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        # 无效关系（无实体）
        relation2 = Relation()
        result2 = graph.add_relation(relation2)
        self.assertFalse(result2)

        # 自环关系
        relation3 = Relation(head_entity=entity1, tail_entity=entity1)
        result3 = graph.add_relation(relation3)
        self.assertFalse(result3)

        # 重复关系
        valid_relation = Relation(head_entity=entity1, tail_entity=entity2)
        graph.add_relation(valid_relation)
        result4 = graph.add_relation(valid_relation)
        self.assertFalse(result4)

    def test_remove_entity(self):
        """测试删除实体"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="实体1", entity_type=EntityType.PERSON)
        entity2 = Entity(name="实体2", entity_type=EntityType.ORGANIZATION)
        entity3 = Entity(name="实体3", entity_type=EntityType.CONCEPT)

        # 添加实体和关系
        graph.add_entity(entity1)
        graph.add_entity(entity2)
        graph.add_entity(entity3)

        relation1 = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.BELONGS_TO)
        relation2 = Relation(head_entity=entity2, tail_entity=entity3, relation_type=RelationType.CONTAINS)
        relation3 = Relation(head_entity=entity1, tail_entity=entity3, relation_type=RelationType.RELATED_TO)

        graph.add_relation(relation1)
        graph.add_relation(relation2)
        graph.add_relation(relation3)

        # 删除实体1，应该同时删除相关关系
        initial_relations_count = len(graph.relations)
        result = graph.remove_entity(entity1.id)

        self.assertTrue(result)
        self.assertNotIn(entity1.id, graph.entities)
        self.assertEqual(len(graph.entities), 2)
        # 关系1和关系3应该被删除
        self.assertNotIn(relation1.id, graph.relations)
        self.assertNotIn(relation3.id, graph.relations)
        # 关系2应该保留
        self.assertIn(relation2.id, graph.relations)
        self.assertEqual(len(graph.relations), 1)

        # 尝试删除不存在的实体
        result2 = graph.remove_entity("non-existent")
        self.assertFalse(result2)

    def test_remove_relation(self):
        """测试删除关系"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="实体1")
        entity2 = Entity(name="实体2")

        graph.add_entity(entity1)
        graph.add_entity(entity2)

        relation = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.SIMILAR_TO)
        graph.add_relation(relation)

        # 删除关系
        result = graph.remove_relation(relation.id)
        self.assertTrue(result)
        self.assertNotIn(relation.id, graph.relations)
        self.assertEqual(len(graph.relations), 0)

        # 尝试删除不存在的关系
        result2 = graph.remove_relation("non-existent")
        self.assertFalse(result2)

    def test_get_entity_and_relation(self):
        """测试获取实体和关系"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="测试实体1")
        entity2 = Entity(name="测试实体2")

        graph.add_entity(entity1)
        graph.add_entity(entity2)

        # 创建有效关系
        relation = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.RELATED_TO)
        graph.add_relation(relation)

        # 获取存在的实体和关系
        self.assertEqual(graph.get_entity(entity1.id), entity1)
        self.assertEqual(graph.get_relation(relation.id), relation)

        # 获取不存在的实体和关系
        self.assertIsNone(graph.get_entity("non-existent"))
        self.assertIsNone(graph.get_relation("non-existent"))

    def test_get_entities_by_type(self):
        """测试按类型获取实体"""
        graph = KnowledgeGraph()

        person1 = Entity(name="张三", entity_type=EntityType.PERSON)
        person2 = Entity(name="李四", entity_type=EntityType.PERSON)
        org1 = Entity(name="公司A", entity_type=EntityType.ORGANIZATION)
        concept1 = Entity(name="概念1", entity_type=EntityType.CONCEPT)

        graph.add_entity(person1)
        graph.add_entity(person2)
        graph.add_entity(org1)
        graph.add_entity(concept1)

        # 按类型获取实体
        persons = graph.get_entities_by_type(EntityType.PERSON)
        self.assertEqual(len(persons), 2)
        self.assertIn(person1, persons)
        self.assertIn(person2, persons)

        orgs = graph.get_entities_by_type(EntityType.ORGANIZATION)
        self.assertEqual(len(orgs), 1)
        self.assertIn(org1, orgs)

        # 获取不存在的类型
        locations = graph.get_entities_by_type(EntityType.LOCATION)
        self.assertEqual(len(locations), 0)

    def test_get_relations_by_type(self):
        """测试按类型获取关系"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="实体1")
        entity2 = Entity(name="实体2")
        entity3 = Entity(name="实体3")

        graph.add_entity(entity1)
        graph.add_entity(entity2)
        graph.add_entity(entity3)

        relation1 = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.BELONGS_TO)
        relation2 = Relation(head_entity=entity2, tail_entity=entity3, relation_type=RelationType.BELONGS_TO)
        relation3 = Relation(head_entity=entity1, tail_entity=entity3, relation_type=RelationType.SIMILAR_TO)

        graph.add_relation(relation1)
        graph.add_relation(relation2)
        graph.add_relation(relation3)

        # 按类型获取关系
        belongs_to_relations = graph.get_relations_by_type(RelationType.BELONGS_TO)
        self.assertEqual(len(belongs_to_relations), 2)
        self.assertIn(relation1, belongs_to_relations)
        self.assertIn(relation2, belongs_to_relations)

        similar_to_relations = graph.get_relations_by_type(RelationType.SIMILAR_TO)
        self.assertEqual(len(similar_to_relations), 1)
        self.assertIn(relation3, similar_to_relations)

        # 获取不存在的类型
        contains_relations = graph.get_relations_by_type(RelationType.CONTAINS)
        self.assertEqual(len(contains_relations), 0)

    def test_get_entity_relations(self):
        """测试获取实体的关系"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="中心实体")
        entity2 = Entity(name="实体2")
        entity3 = Entity(name="实体3")
        entity4 = Entity(name="实体4")

        graph.add_entity(entity1)
        graph.add_entity(entity2)
        graph.add_entity(entity3)
        graph.add_entity(entity4)

        # 创建不同方向的关系
        out_relation1 = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.BELONGS_TO)
        out_relation2 = Relation(head_entity=entity1, tail_entity=entity3, relation_type=RelationType.CONTAINS)
        in_relation = Relation(head_entity=entity4, tail_entity=entity1, relation_type=RelationType.BELONGS_TO)

        graph.add_relation(out_relation1)
        graph.add_relation(out_relation2)
        graph.add_relation(in_relation)

        # 获取所有关系
        all_relations = graph.get_entity_relations(entity1.id, direction="both")
        self.assertEqual(len(all_relations), 3)

        # 获取出向关系
        out_relations = graph.get_entity_relations(entity1.id, direction="out")
        self.assertEqual(len(out_relations), 2)
        self.assertIn(out_relation1, out_relations)
        self.assertIn(out_relation2, out_relations)

        # 获取入向关系
        in_relations = graph.get_entity_relations(entity1.id, direction="in")
        self.assertEqual(len(in_relations), 1)
        self.assertIn(in_relation, in_relations)

        # 按关系类型过滤
        belongs_to_relations = graph.get_entity_relations(entity1.id, relation_type=RelationType.BELONGS_TO)
        self.assertEqual(len(belongs_to_relations), 2)
        self.assertIn(out_relation1, belongs_to_relations)
        self.assertIn(in_relation, belongs_to_relations)

        # 获取不存在实体的关系
        empty_relations = graph.get_entity_relations("non-existent")
        self.assertEqual(len(empty_relations), 0)

    def test_get_neighbors(self):
        """测试获取邻居实体"""
        graph = KnowledgeGraph()
        entity1 = Entity(name="中心实体")
        entity2 = Entity(name="邻居1")
        entity3 = Entity(name="邻居2")
        entity4 = Entity(name="邻居3")

        graph.add_entity(entity1)
        graph.add_entity(entity2)
        graph.add_entity(entity3)
        graph.add_entity(entity4)

        # 创建关系
        relation1 = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.BELONGS_TO)
        relation2 = Relation(head_entity=entity1, tail_entity=entity3, relation_type=RelationType.CONTAINS)
        relation3 = Relation(head_entity=entity4, tail_entity=entity1, relation_type=RelationType.BELONGS_TO)

        graph.add_relation(relation1)
        graph.add_relation(relation2)
        graph.add_relation(relation3)

        # 获取所有邻居
        all_neighbors = graph.get_neighbors(entity1.id)
        self.assertEqual(len(all_neighbors), 3)
        self.assertIn(entity2, all_neighbors)
        self.assertIn(entity3, all_neighbors)
        self.assertIn(entity4, all_neighbors)

        # 获取出向邻居
        out_neighbors = graph.get_neighbors(entity1.id, direction="out")
        self.assertEqual(len(out_neighbors), 2)
        self.assertIn(entity2, out_neighbors)
        self.assertIn(entity3, out_neighbors)

        # 获取入向邻居
        in_neighbors = graph.get_neighbors(entity1.id, direction="in")
        self.assertEqual(len(in_neighbors), 1)
        self.assertIn(entity4, in_neighbors)

        # 按关系类型过滤
        belongs_to_neighbors = graph.get_neighbors(entity1.id, relation_type=RelationType.BELONGS_TO)
        self.assertEqual(len(belongs_to_neighbors), 2)
        self.assertIn(entity2, belongs_to_neighbors)
        self.assertIn(entity4, belongs_to_neighbors)

    def test_find_path(self):
        """测试查找路径"""
        graph = KnowledgeGraph()

        # 创建一个简单的路径：A -> B -> C -> D
        entity_a = Entity(name="A")
        entity_b = Entity(name="B")
        entity_c = Entity(name="C")
        entity_d = Entity(name="D")
        entity_e = Entity(name="E")  # 孤立节点

        graph.add_entity(entity_a)
        graph.add_entity(entity_b)
        graph.add_entity(entity_c)
        graph.add_entity(entity_d)
        graph.add_entity(entity_e)

        relation_ab = Relation(head_entity=entity_a, tail_entity=entity_b, relation_type=RelationType.RELATED_TO)
        relation_bc = Relation(head_entity=entity_b, tail_entity=entity_c, relation_type=RelationType.RELATED_TO)
        relation_cd = Relation(head_entity=entity_c, tail_entity=entity_d, relation_type=RelationType.RELATED_TO)

        graph.add_relation(relation_ab)
        graph.add_relation(relation_bc)
        graph.add_relation(relation_cd)

        # 查找A到D的路径
        paths_ad = graph.find_path(entity_a.id, entity_d.id, max_depth=5)
        self.assertEqual(len(paths_ad), 1)
        self.assertEqual(len(paths_ad[0]), 3)
        self.assertEqual(paths_ad[0], [relation_ab, relation_bc, relation_cd])

        # 查找A到C的路径
        paths_ac = graph.find_path(entity_a.id, entity_c.id, max_depth=5)
        self.assertEqual(len(paths_ac), 1)
        self.assertEqual(len(paths_ac[0]), 2)
        self.assertEqual(paths_ac[0], [relation_ab, relation_bc])

        # 查找A到E的路径（无路径）
        paths_ae = graph.find_path(entity_a.id, entity_e.id, max_depth=5)
        self.assertEqual(len(paths_ae), 0)

        # 查找不存在实体的路径
        paths_invalid = graph.find_path("non-existent", entity_d.id)
        self.assertEqual(len(paths_invalid), 0)

    def test_merge_entity(self):
        """测试合并实体"""
        graph = KnowledgeGraph()

        target_entity = Entity(
            name="目标实体",
            entity_type=EntityType.PERSON,
            properties={"age": 30},
            aliases=["目标"]
        )
        source_entity = Entity(
            name="源实体",
            entity_type=EntityType.PERSON,
            properties={"city": "北京"},
            aliases=["源", "目标"]  # 有重复别名
        )
        other_entity = Entity(name="其他实体")

        graph.add_entity(target_entity)
        graph.add_entity(source_entity)
        graph.add_entity(other_entity)

        # 创建涉及源实体的关系
        relation1 = Relation(head_entity=source_entity, tail_entity=other_entity, relation_type=RelationType.RELATED_TO)
        relation2 = Relation(head_entity=other_entity, tail_entity=source_entity, relation_type=RelationType.MENTIONS)

        graph.add_relation(relation1)
        graph.add_relation(relation2)

        # 合并实体
        result = graph.merge_entity(target_entity, source_entity)
        self.assertTrue(result)

        # 验证源实体已被删除
        self.assertNotIn(source_entity.id, graph.entities)
        self.assertIn(target_entity.id, graph.entities)

        # 验证属性和别名已合并
        self.assertEqual(target_entity.properties["age"], 30)
        self.assertEqual(target_entity.properties["city"], "北京")
        self.assertIn("目标", target_entity.aliases)
        self.assertIn("源", target_entity.aliases)
        self.assertEqual(len(set(target_entity.aliases)), len(target_entity.aliases))  # 确保去重

        # 验证关系中的实体引用已更新
        for relation in graph.relations.values():
            if relation.head_entity and relation.head_entity.id == source_entity.id:
                self.fail("关系中仍然引用源实体")
            if relation.tail_entity and relation.tail_entity.id == source_entity.id:
                self.fail("关系中仍然引用源实体")

        # 验证关系现在指向目标实体
        relations = list(graph.relations.values())
        self.assertEqual(len(relations), 2)
        self.assertTrue(relations[0].head_entity == target_entity or relations[0].tail_entity == target_entity)
        self.assertTrue(relations[1].head_entity == target_entity or relations[1].tail_entity == target_entity)

    def test_merge_entity_invalid(self):
        """测试无效的实体合并"""
        graph = KnowledgeGraph()

        entity1 = Entity(name="实体1")
        entity2 = Entity(name="实体2")
        entity3 = Entity(name="实体3")

        graph.add_entity(entity1)
        graph.add_entity(entity2)
        # entity3 不在图中

        # 合并不存在的实体
        result1 = graph.merge_entity(entity1, entity3)
        self.assertFalse(result1)

        result2 = graph.merge_entity(entity3, entity1)
        self.assertFalse(result2)

    def test_get_statistics(self):
        """测试获取统计信息"""
        graph = KnowledgeGraph(name="测试图谱")

        # 添加不同类型的实体
        person1 = Entity(name="张三", entity_type=EntityType.PERSON)
        person2 = Entity(name="李四", entity_type=EntityType.PERSON)
        org1 = Entity(name="公司A", entity_type=EntityType.ORGANIZATION)
        concept1 = Entity(name="概念1", entity_type=EntityType.CONCEPT)

        graph.add_entity(person1)
        graph.add_entity(person2)
        graph.add_entity(org1)
        graph.add_entity(concept1)

        # 添加不同类型的关系
        relation1 = Relation(head_entity=person1, tail_entity=org1, relation_type=RelationType.BELONGS_TO)
        relation2 = Relation(head_entity=person2, tail_entity=org1, relation_type=RelationType.BELONGS_TO)
        relation3 = Relation(head_entity=person1, tail_entity=person2, relation_type=RelationType.SIMILAR_TO)
        relation4 = Relation(head_entity=org1, tail_entity=concept1, relation_type=RelationType.RELATED_TO)

        graph.add_relation(relation1)
        graph.add_relation(relation2)
        graph.add_relation(relation3)
        graph.add_relation(relation4)

        # 获取统计信息
        stats = graph.get_statistics()

        self.assertEqual(stats["total_entities"], 4)
        self.assertEqual(stats["total_relations"], 4)
        self.assertEqual(stats["entity_types"]["person"], 2)
        self.assertEqual(stats["entity_types"]["organization"], 1)
        self.assertEqual(stats["entity_types"]["concept"], 1)
        self.assertEqual(stats["relation_types"]["belongs_to"], 2)
        self.assertEqual(stats["relation_types"]["similar_to"], 1)
        self.assertEqual(stats["relation_types"]["related_to"], 1)
        self.assertIn("created_at", stats)
        self.assertIn("updated_at", stats)

    def test_to_dict(self):
        """测试转换为字典"""
        graph = KnowledgeGraph(name="测试图谱")

        entity1 = Entity(name="实体1", entity_type=EntityType.PERSON)
        entity2 = Entity(name="实体2", entity_type=EntityType.ORGANIZATION)

        graph.add_entity(entity1)
        graph.add_entity(entity2)

        relation = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.BELONGS_TO)
        graph.add_relation(relation)

        # 转换为字典
        graph_dict = graph.to_dict()

        self.assertEqual(graph_dict["id"], graph.id)
        self.assertEqual(graph_dict["name"], "测试图谱")
        self.assertIn("entities", graph_dict)
        self.assertIn("relations", graph_dict)
        self.assertIn(entity1.id, graph_dict["entities"])
        self.assertIn(entity2.id, graph_dict["entities"])
        self.assertIn(relation.id, graph_dict["relations"])
        self.assertIn("created_at", graph_dict)
        self.assertIn("updated_at", graph_dict)
        self.assertIsInstance(graph_dict["created_at"], str)
        self.assertIsInstance(graph_dict["updated_at"], str)

    def test_from_dict(self):
        """测试从字典创建知识图谱"""
        # 创建原始图谱
        original_graph = KnowledgeGraph(name="原始图谱")

        entity1 = Entity(name="实体1", entity_type=EntityType.PERSON)
        entity2 = Entity(name="实体2", entity_type=EntityType.ORGANIZATION)

        original_graph.add_entity(entity1)
        original_graph.add_entity(entity2)

        relation = Relation(head_entity=entity1, tail_entity=entity2, relation_type=RelationType.BELONGS_TO)
        original_graph.add_relation(relation)

        # 转换为字典
        graph_dict = original_graph.to_dict()

        # 从字典重建图谱
        restored_graph = KnowledgeGraph.from_dict(graph_dict)

        # 验证基本属性
        self.assertEqual(restored_graph.id, original_graph.id)
        self.assertEqual(restored_graph.name, original_graph.name)
        self.assertEqual(len(restored_graph.entities), len(original_graph.entities))
        self.assertEqual(len(restored_graph.relations), len(original_graph.relations))

        # 验证实体
        for entity_id, original_entity in original_graph.entities.items():
            restored_entity = restored_graph.entities[entity_id]
            self.assertEqual(restored_entity.name, original_entity.name)
            self.assertEqual(restored_entity.entity_type, original_entity.entity_type)

        # 验证关系
        for relation_id, original_relation in original_graph.relations.items():
            restored_relation = restored_graph.relations[relation_id]
            self.assertEqual(restored_relation.relation_type, original_relation.relation_type)
            self.assertEqual(restored_relation.head_entity.id, original_relation.head_entity.id)
            self.assertEqual(restored_relation.tail_entity.id, original_relation.tail_entity.id)

    def test_graph_round_trip(self):
        """测试图谱的序列化和反序列化往返"""
        original_graph = KnowledgeGraph(name="往返测试图谱")

        # 创建复杂的图结构
        entities = []
        for i in range(5):
            entity = Entity(
                name=f"实体{i}",
                entity_type=EntityType.PERSON if i % 2 == 0 else EntityType.ORGANIZATION,
                properties={"index": i, "type": "test"}
            )
            entities.append(entity)
            original_graph.add_entity(entity)

        # 创建多个关系
        for i in range(len(entities) - 1):
            relation = Relation(
                head_entity=entities[i],
                tail_entity=entities[i + 1],
                relation_type=RelationType.RELATED_TO,
                properties={"weight": 0.5 + i * 0.1}
            )
            original_graph.add_relation(relation)

        # 序列化和反序列化
        graph_dict = original_graph.to_dict()
        restored_graph = KnowledgeGraph.from_dict(graph_dict)

        # 验证完整性
        self.assertEqual(restored_graph.id, original_graph.id)
        self.assertEqual(restored_graph.name, original_graph.name)
        self.assertEqual(len(restored_graph.entities), len(original_graph.entities))
        self.assertEqual(len(restored_graph.relations), len(original_graph.relations))

        # 验证统计信息一致
        original_stats = original_graph.get_statistics()
        restored_stats = restored_graph.get_statistics()
        self.assertEqual(original_stats["total_entities"], restored_stats["total_entities"])
        self.assertEqual(original_stats["total_relations"], restored_stats["total_relations"])
        self.assertEqual(original_stats["entity_types"], restored_stats["entity_types"])
        self.assertEqual(original_stats["relation_types"], restored_stats["relation_types"])

    def test_empty_graph_operations(self):
        """测试空图的操作"""
        graph = KnowledgeGraph()

        # 空图的统计信息
        stats = graph.get_statistics()
        self.assertEqual(stats["total_entities"], 0)
        self.assertEqual(stats["total_relations"], 0)
        self.assertEqual(stats["entity_types"], {})
        self.assertEqual(stats["relation_types"], {})

        # 空图的查询操作
        self.assertIsNone(graph.get_entity("non-existent"))
        self.assertIsNone(graph.get_relation("non-existent"))
        self.assertEqual(len(graph.get_entities_by_type(EntityType.PERSON)), 0)
        self.assertEqual(len(graph.get_relations_by_type(RelationType.BELONGS_TO)), 0)
        self.assertEqual(len(graph.get_entity_relations("non-existent")), 0)
        self.assertEqual(len(graph.get_neighbors("non-existent")), 0)
        self.assertEqual(len(graph.find_path("non-existent1", "non-existent2")), 0)

        # 空图的删除操作
        self.assertFalse(graph.remove_entity("non-existent"))
        self.assertFalse(graph.remove_relation("non-existent"))


if __name__ == '__main__':
    unittest.main()
