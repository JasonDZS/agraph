"""
agraph.utils模块的单元测试

测试覆盖的功能：
1. 图谱导出功能（Cytoscape、D3）
2. 路径查找功能
3. 图谱指标计算
4. 实体相似度计算和合并
5. 图谱一致性验证
6. 图谱摘要生成
7. 内部辅助函数
"""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

from agraph.entities import Entity, EntityType
from agraph.graph import KnowledgeGraph
from agraph.relations import Relation, RelationType
from agraph.utils import (
    export_graph_to_cytoscape,
    export_graph_to_d3,
    find_shortest_path,
    calculate_graph_metrics,
    merge_similar_entities,
    validate_graph_consistency,
    create_graph_summary,
    _calculate_name_similarity,
    _find_connected_components,
    _dfs_component,
    _calculate_type_distribution,
)


@pytest.fixture
def sample_graph():
    """创建一个示例知识图谱用于测试"""
    graph = KnowledgeGraph("test_graph")

    # 创建实体
    entity1 = Entity(
        id="entity1",
        name="张三",
        entity_type=EntityType.PERSON,
        description="一个人",
        confidence=0.9
    )
    entity2 = Entity(
        id="entity2",
        name="北京大学",
        entity_type=EntityType.ORGANIZATION,
        description="一所大学",
        confidence=0.8
    )
    entity3 = Entity(
        id="entity3",
        name="计算机科学",
        entity_type=EntityType.CONCEPT,
        description="一个学科",
        confidence=0.7
    )

    # 添加实体到图谱
    graph.add_entity(entity1)
    graph.add_entity(entity2)
    graph.add_entity(entity3)

    # 创建关系
    relation1 = Relation(
        id="rel1",
        head_entity=entity1,
        tail_entity=entity2,
        relation_type=RelationType.BELONGS_TO,
        confidence=0.9
    )
    relation2 = Relation(
        id="rel2",
        head_entity=entity2,
        tail_entity=entity3,
        relation_type=RelationType.RELATED_TO,
        confidence=0.8
    )

    # 添加关系到图谱
    graph.add_relation(relation1)
    graph.add_relation(relation2)

    return graph


@pytest.fixture
def empty_graph():
    """创建一个空的知识图谱"""
    return KnowledgeGraph("empty_graph")


class TestExportFunctions:
    """测试导出功能"""

    def test_export_graph_to_cytoscape_success(self, sample_graph):
        """测试成功导出到Cytoscape格式"""
        result = export_graph_to_cytoscape(sample_graph)

        # 验证返回结构
        assert isinstance(result, dict)
        assert "elements" in result
        assert "graph_info" in result
        assert isinstance(result["elements"], dict)
        assert "nodes" in result["elements"]
        assert "edges" in result["elements"]

        # 验证节点数量和结构
        nodes = result["elements"]["nodes"]
        assert len(nodes) == 3

        for node in nodes:
            assert "data" in node
            assert "classes" in node
            node_data = node["data"]
            required_fields = ["id", "label", "type", "description", "confidence", "source"]
            for field in required_fields:
                assert field in node_data
            # 验证类型字段是字符串值
            assert isinstance(node_data["type"], str)
            assert node["classes"] == node_data["type"]

        # 验证边数量和结构
        edges = result["elements"]["edges"]
        assert len(edges) == 2

        for edge in edges:
            assert "data" in edge
            assert "classes" in edge
            edge_data = edge["data"]
            required_fields = ["id", "source", "target", "label", "type", "confidence", "source_info"]
            for field in required_fields:
                assert field in edge_data

        # 验证图谱信息
        graph_info = result["graph_info"]
        required_info = ["id", "name", "created_at", "updated_at", "statistics"]
        for field in required_info:
            assert field in graph_info

    def test_export_graph_to_cytoscape_empty_graph(self, empty_graph):
        """测试导出空图谱到Cytoscape格式"""
        result = export_graph_to_cytoscape(empty_graph)

        assert len(result["elements"]["nodes"]) == 0
        assert len(result["elements"]["edges"]) == 0
        assert "graph_info" in result

    def test_export_graph_to_cytoscape_skip_invalid_relations(self, sample_graph):
        """测试跳过无效关系的导出"""
        # 添加无效关系（head_entity为None）
        invalid_relation = Relation(
            id="invalid_rel",
            head_entity=None,
            tail_entity=sample_graph.entities["entity1"],
            relation_type=RelationType.RELATED_TO
        )
        sample_graph.relations["invalid_rel"] = invalid_relation

        result = export_graph_to_cytoscape(sample_graph)

        # 应该只有原来的2个有效关系
        assert len(result["elements"]["edges"]) == 2

    def test_export_graph_to_d3_success(self, sample_graph):
        """测试成功导出到D3格式"""
        result = export_graph_to_d3(sample_graph)

        # 验证返回结构
        assert isinstance(result, dict)
        required_keys = ["nodes", "links", "graph_info"]
        for key in required_keys:
            assert key in result

        # 验证节点
        nodes = result["nodes"]
        assert len(nodes) == 3

        for i, node in enumerate(nodes):
            required_fields = ["id", "entity_id", "name", "type", "description", "confidence", "group", "size"]
            for field in required_fields:
                assert field in node
            assert node["id"] == i  # D3节点ID应该是数字索引
            assert 5 <= node["size"] <= 20  # 大小应该在合理范围内
            assert node["group"] == node["type"]

        # 验证链接
        links = result["links"]
        assert len(links) == 2

        for link in links:
            required_fields = ["source", "target", "relation_id", "type", "confidence", "value"]
            for field in required_fields:
                assert field in link
            # source和target应该是节点索引
            assert isinstance(link["source"], int)
            assert isinstance(link["target"], int)
            assert 0 <= link["source"] < 3
            assert 0 <= link["target"] < 3
            assert link["value"] == link["confidence"]

        # 验证图谱信息
        graph_info = result["graph_info"]
        assert graph_info["node_count"] == len(nodes)
        assert graph_info["link_count"] == len(links)

    def test_export_graph_to_d3_empty_graph(self, empty_graph):
        """测试导出空图谱到D3格式"""
        result = export_graph_to_d3(empty_graph)

        assert len(result["nodes"]) == 0
        assert len(result["links"]) == 0
        assert result["graph_info"]["node_count"] == 0
        assert result["graph_info"]["link_count"] == 0

    def test_export_graph_to_d3_skip_invalid_relations(self, sample_graph):
        """测试D3导出时跳过无效关系"""
        # 添加关系引用不存在的实体
        nonexistent_entity = Entity(id="nonexistent", name="不存在", entity_type=EntityType.PERSON)
        invalid_relation = Relation(
            id="invalid_rel",
            head_entity=nonexistent_entity,
            tail_entity=sample_graph.entities["entity1"],
            relation_type=RelationType.RELATED_TO
        )
        sample_graph.relations["invalid_rel"] = invalid_relation

        result = export_graph_to_d3(sample_graph)

        # 应该只有原来的2个有效关系
        assert len(result["links"]) == 2


class TestPathFinding:
    """测试路径查找功能"""

    def test_find_shortest_path_direct_connection(self, sample_graph):
        """测试直接连接的最短路径"""
        path = find_shortest_path(sample_graph, "entity1", "entity2")

        assert path is not None
        assert len(path) == 1
        assert path[0].id == "rel1"
        assert path[0].head_entity.id == "entity1"
        assert path[0].tail_entity.id == "entity2"

    def test_find_shortest_path_two_hops(self, sample_graph):
        """测试两跳连接的最短路径"""
        path = find_shortest_path(sample_graph, "entity1", "entity3")

        assert path is not None
        assert len(path) == 2
        assert path[0].id == "rel1"  # entity1 -> entity2
        assert path[1].id == "rel2"  # entity2 -> entity3

    def test_find_shortest_path_same_entity(self, sample_graph):
        """测试相同实体的路径查找"""
        path = find_shortest_path(sample_graph, "entity1", "entity1")

        assert path == []

    def test_find_shortest_path_no_connection(self, sample_graph):
        """测试不存在连接的路径查找"""
        # 添加一个孤立的实体
        isolated_entity = Entity(
            id="isolated",
            name="孤立实体",
            entity_type=EntityType.CONCEPT,
            description="孤立的实体",
            confidence=0.5
        )
        sample_graph.add_entity(isolated_entity)

        path = find_shortest_path(sample_graph, "entity1", "isolated")
        assert path is None

    def test_find_shortest_path_invalid_start_entity(self, sample_graph):
        """测试起始实体不存在的情况"""
        path = find_shortest_path(sample_graph, "nonexistent_start", "entity1")
        assert path is None

    def test_find_shortest_path_invalid_end_entity(self, sample_graph):
        """测试目标实体不存在的情况"""
        path = find_shortest_path(sample_graph, "entity1", "nonexistent_end")
        assert path is None

    def test_find_shortest_path_both_entities_invalid(self, sample_graph):
        """测试起始和目标实体都不存在的情况"""
        path = find_shortest_path(sample_graph, "nonexistent1", "nonexistent2")
        assert path is None

    def test_find_shortest_path_bfs_correctness(self, sample_graph):
        """测试BFS算法的正确性（通过结果验证）"""
        # 验证BFS找到的是最短路径
        path = find_shortest_path(sample_graph, "entity1", "entity3")

        assert path is not None
        assert len(path) == 2  # 最短路径应该是2跳

        # 验证路径的正确性
        assert path[0].head_entity.id == "entity1"
        assert path[0].tail_entity.id == "entity2"
        assert path[1].head_entity.id == "entity2"
        assert path[1].tail_entity.id == "entity3"

    def test_find_shortest_path_complex_graph(self):
        """测试复杂图谱中的最短路径"""
        # 创建更复杂的图谱
        graph = KnowledgeGraph("complex_graph")

        # 创建多个实体
        entities = []
        for i in range(5):
            entity = Entity(
                id=f"entity_{i}",
                name=f"实体{i}",
                entity_type=EntityType.CONCEPT,
                confidence=0.8
            )
            entities.append(entity)
            graph.add_entity(entity)

        # 创建路径: 0->1->2, 0->3->4->2（后者路径更长）
        relations = [
            Relation("rel_0_1", entities[0], entities[1], RelationType.RELATED_TO),
            Relation("rel_1_2", entities[1], entities[2], RelationType.RELATED_TO),
            Relation("rel_0_3", entities[0], entities[3], RelationType.RELATED_TO),
            Relation("rel_3_4", entities[3], entities[4], RelationType.RELATED_TO),
            Relation("rel_4_2", entities[4], entities[2], RelationType.RELATED_TO),
        ]

        for rel in relations:
            graph.add_relation(rel)

        # 查找最短路径，应该返回较短的路径
        path = find_shortest_path(graph, "entity_0", "entity_2")

        assert path is not None
        assert len(path) == 2  # 应该选择更短的路径
        assert path[0].id == "rel_0_1"
        assert path[1].id == "rel_1_2"


class TestGraphMetrics:
    """测试图谱指标计算"""

    def test_calculate_graph_metrics_complete(self, sample_graph):
        """测试完整的图谱指标计算"""
        metrics = calculate_graph_metrics(sample_graph)

        # 验证返回结构
        assert isinstance(metrics, dict)
        required_sections = ["basic_stats", "centrality", "connectivity", "type_distribution"]
        for section in required_sections:
            assert section in metrics

        # 验证基本统计
        basic_stats = metrics["basic_stats"]
        assert basic_stats["node_count"] == 3
        assert basic_stats["edge_count"] == 2
        assert isinstance(basic_stats["density"], float)
        assert 0 <= basic_stats["density"] <= 1
        # 检查平均度数，允许小的浮点误差
        assert abs(basic_stats["avg_degree"] - 4/3) < 0.01  # 总度数为4，3个节点
        assert basic_stats["max_degree"] == 2  # entity2有最高度数
        assert basic_stats["min_degree"] == 1  # entity1和entity3各有度数1

        # 验证中心性分析
        centrality = metrics["centrality"]
        assert "top_central_nodes" in centrality
        top_nodes = centrality["top_central_nodes"]
        assert len(top_nodes) <= 5
        assert len(top_nodes) == 3  # 只有3个节点

        # 验证最高中心性节点
        assert top_nodes[0]["entity_id"] == "entity2"  # entity2应该是中心节点
        assert top_nodes[0]["degree"] == 2
        assert top_nodes[0]["entity_name"] == "北京大学"

        # 验证连通性
        connectivity = metrics["connectivity"]
        assert connectivity["connected_components"] == 1
        assert connectivity["largest_component_size"] == 3
        assert connectivity["is_connected"] is True

        # 验证类型分布
        type_dist = metrics["type_distribution"]
        assert "entity_types" in type_dist
        assert "relation_types" in type_dist

    def test_calculate_graph_metrics_empty_graph(self, empty_graph):
        """测试空图谱的指标计算"""
        metrics = calculate_graph_metrics(empty_graph)
        assert metrics == {}

    def test_calculate_graph_metrics_single_node(self):
        """测试单节点图谱的指标"""
        graph = KnowledgeGraph("single_node")
        entity = Entity(
            id="single",
            name="单一实体",
            entity_type=EntityType.PERSON,
            confidence=0.9
        )
        graph.add_entity(entity)

        metrics = calculate_graph_metrics(graph)

        basic_stats = metrics["basic_stats"]
        assert basic_stats["node_count"] == 1
        assert basic_stats["edge_count"] == 0
        assert basic_stats["density"] == 0
        assert basic_stats["avg_degree"] == 0
        assert basic_stats["max_degree"] == 0
        assert basic_stats["min_degree"] == 0

        connectivity = metrics["connectivity"]
        assert connectivity["connected_components"] == 1
        assert connectivity["largest_component_size"] == 1
        assert connectivity["is_connected"] is True

    def test_calculate_graph_metrics_disconnected_graph(self, sample_graph):
        """测试非连通图谱的指标"""
        # 添加两个孤立的实体
        isolated1 = Entity(id="isolated1", name="孤立1", entity_type=EntityType.CONCEPT)
        isolated2 = Entity(id="isolated2", name="孤立2", entity_type=EntityType.CONCEPT)
        sample_graph.add_entity(isolated1)
        sample_graph.add_entity(isolated2)

        # 连接两个孤立实体
        relation = Relation(
            id="isolated_rel",
            head_entity=isolated1,
            tail_entity=isolated2,
            relation_type=RelationType.RELATED_TO
        )
        sample_graph.add_relation(relation)

        metrics = calculate_graph_metrics(sample_graph)

        # 现在应该有两个连通分量
        connectivity = metrics["connectivity"]
        assert connectivity["connected_components"] == 2
        assert connectivity["largest_component_size"] == 3  # 原始的连通分量
        assert connectivity["is_connected"] is False

    def test_calculate_type_distribution_detailed(self, sample_graph):
        """测试详细的类型分布计算"""
        distribution = _calculate_type_distribution(sample_graph)

        assert "entity_types" in distribution
        assert "relation_types" in distribution

        entity_types = distribution["entity_types"]
        expected_entity_types = {
            EntityType.PERSON.value: 1,
            EntityType.ORGANIZATION.value: 1,
            EntityType.CONCEPT.value: 1
        }
        assert entity_types == expected_entity_types

        relation_types = distribution["relation_types"]
        expected_relation_types = {
            RelationType.BELONGS_TO.value: 1,
            RelationType.RELATED_TO.value: 1
        }
        assert relation_types == expected_relation_types

    def test_calculate_type_distribution_empty_graph(self, empty_graph):
        """测试空图谱的类型分布"""
        distribution = _calculate_type_distribution(empty_graph)

        assert distribution["entity_types"] == {}
        assert distribution["relation_types"] == {}

    @patch('agraph.graph.KnowledgeGraph.get_entity_relations')
    def test_calculate_graph_metrics_mock_relations(self, mock_get_relations, sample_graph):
        """测试使用Mock验证关系获取调用"""
        # 模拟返回值
        mock_get_relations.return_value = []

        metrics = calculate_graph_metrics(sample_graph)

        # 验证每个实体都调用了get_entity_relations
        assert mock_get_relations.call_count >= 3  # 至少调用3次（每个实体）

        # 验证结果反映了模拟的空关系
        basic_stats = metrics["basic_stats"]
        assert basic_stats["avg_degree"] == 0


class TestConnectedComponents:
    """测试连通分量分析"""

    def test_find_connected_components_single_component(self, sample_graph):
        """测试单个连通分量的识别"""
        components = _find_connected_components(sample_graph)

        assert len(components) == 1
        assert len(components[0]) == 3
        assert set(components[0]) == {"entity1", "entity2", "entity3"}

    def test_find_connected_components_multiple_components(self, sample_graph):
        """测试多个连通分量的识别"""
        # 添加两个孤立实体并连接它们
        isolated1 = Entity(id="isolated1", name="孤立1", entity_type=EntityType.CONCEPT)
        isolated2 = Entity(id="isolated2", name="孤立2", entity_type=EntityType.CONCEPT)
        sample_graph.add_entity(isolated1)
        sample_graph.add_entity(isolated2)

        isolated_relation = Relation(
            id="isolated_rel",
            head_entity=isolated1,
            tail_entity=isolated2,
            relation_type=RelationType.RELATED_TO
        )
        sample_graph.add_relation(isolated_relation)

        components = _find_connected_components(sample_graph)

        assert len(components) == 2
        # 检查分量大小
        component_sizes = sorted([len(comp) for comp in components])
        assert component_sizes == [2, 3]

        # 验证原始分量包含预期实体
        original_component = max(components, key=len)
        assert set(original_component) == {"entity1", "entity2", "entity3"}

        # 验证孤立分量包含预期实体
        isolated_component = min(components, key=len)
        assert set(isolated_component) == {"isolated1", "isolated2"}

    def test_find_connected_components_all_isolated(self):
        """测试所有实体都孤立的情况"""
        graph = KnowledgeGraph("isolated_graph")

        # 添加3个孤立实体
        for i in range(3):
            entity = Entity(
                id=f"isolated_{i}",
                name=f"孤立实体{i}",
                entity_type=EntityType.CONCEPT
            )
            graph.add_entity(entity)

        components = _find_connected_components(graph)

        assert len(components) == 3
        for component in components:
            assert len(component) == 1

    def test_find_connected_components_empty_graph(self, empty_graph):
        """测试空图谱的连通分量"""
        components = _find_connected_components(empty_graph)
        assert components == []

    def test_dfs_component_complete_traversal(self, sample_graph):
        """测试DFS连通分量完整遍历"""
        visited = set()
        component = _dfs_component(sample_graph, "entity1", visited)

        # 应该访问所有连通的实体
        assert len(component) == 3
        expected_entities = {"entity1", "entity2", "entity3"}
        assert set(component) == expected_entities

        # 验证visited集合被正确更新
        assert visited == expected_entities

    def test_dfs_component_partial_graph(self, sample_graph):
        """测试DFS在部分连通图中的表现"""
        # 添加孤立实体
        isolated = Entity(id="isolated", name="孤立", entity_type=EntityType.CONCEPT)
        sample_graph.add_entity(isolated)

        visited = set()
        component = _dfs_component(sample_graph, "entity1", visited)

        # 应该只访问连通的实体，不包括孤立实体
        assert len(component) == 3
        assert "isolated" not in component
        assert "isolated" not in visited

    def test_dfs_component_already_visited(self, sample_graph):
        """测试DFS处理已访问节点的情况"""
        visited = {"entity1", "entity2"}  # 预先标记为已访问
        component = _dfs_component(sample_graph, "entity1", visited)

        # 如果起始节点已访问，应该返回空列表
        assert component == []

        # 尝试从未访问的节点开始
        component = _dfs_component(sample_graph, "entity3", visited)
        assert len(component) == 1
        assert component == ["entity3"]

    @patch('agraph.graph.KnowledgeGraph.get_neighbors')
    def test_dfs_component_mock_neighbors(self, mock_get_neighbors, sample_graph):
        """测试DFS使用模拟邻居"""
        # 模拟get_neighbors返回空列表
        mock_get_neighbors.return_value = []

        visited = set()
        component = _dfs_component(sample_graph, "entity1", visited)

        # 应该只包含起始节点
        assert component == ["entity1"]
        assert visited == {"entity1"}

        # 验证get_neighbors被调用
        mock_get_neighbors.assert_called_with("entity1")

    def test_find_connected_components_star_topology(self):
        """测试星形拓扑的连通分量"""
        graph = KnowledgeGraph("star_graph")

        # 创建中心节点
        center = Entity(id="center", name="中心", entity_type=EntityType.CONCEPT)
        graph.add_entity(center)

        # 创建周围节点并连接到中心
        for i in range(4):
            node = Entity(id=f"node_{i}", name=f"节点{i}", entity_type=EntityType.CONCEPT)
            graph.add_entity(node)

            relation = Relation(
                id=f"rel_{i}",
                head_entity=center,
                tail_entity=node,
                relation_type=RelationType.RELATED_TO
            )
            graph.add_relation(relation)

        components = _find_connected_components(graph)

        assert len(components) == 1
        assert len(components[0]) == 5  # 中心节点 + 4个周围节点


class TestSimilarityFunctions:
    """测试相似度功能"""

    def test_calculate_name_similarity_identical_names(self):
        """测试完全相同名称的相似度"""
        similarity = _calculate_name_similarity("张三", "张三")
        assert similarity == 1.0

    def test_calculate_name_similarity_completely_different(self):
        """测试完全不同名称的相似度"""
        similarity = _calculate_name_similarity("张三", "李四")
        assert similarity == 0.0

    def test_calculate_name_similarity_partial_match_english(self):
        """测试英文名称的部分匹配"""
        similarity = _calculate_name_similarity("Beijing University", "Tsinghua University")
        # "Beijing University" -> {"beijing", "university"}
        # "Tsinghua University" -> {"tsinghua", "university"}
        # 交集: {"university"} (1个), 并集: {"beijing", "tsinghua", "university"} (3个)
        # 相似度 = 1/3 ≈ 0.333
        expected_similarity = 1/3
        assert abs(similarity - expected_similarity) < 0.01

    def test_calculate_name_similarity_full_overlap_english(self):
        """测试英文名称完全重叠"""
        similarity = _calculate_name_similarity("Computer Science", "Computer Science")
        assert similarity == 1.0

    def test_calculate_name_similarity_subset_english(self):
        """测试英文名称子集关系"""
        similarity = _calculate_name_similarity("University", "Beijing University")
        # "University" -> {"university"}
        # "Beijing University" -> {"beijing", "university"}
        # 交集: {"university"} (1个), 并集: {"beijing", "university"} (2个)
        # 相似度 = 1/2 = 0.5
        assert similarity == 0.5

    def test_calculate_name_similarity_chinese_identical(self):
        """测试中文名称相同"""
        similarity = _calculate_name_similarity("北京大学", "北京大学")
        assert similarity == 1.0

    def test_calculate_name_similarity_chinese_different(self):
        """测试中文名称不同（无空格分词）"""
        # 中文字符串不包含空格时被视为单个词
        similarity = _calculate_name_similarity("北京大学", "清华大学")
        assert similarity == 0.0

    def test_calculate_name_similarity_chinese_with_spaces(self):
        """测试带空格的中文名称"""
        similarity = _calculate_name_similarity("北京 大学", "清华 大学")
        # "北京 大学" -> {"北京", "大学"}
        # "清华 大学" -> {"清华", "大学"}
        # 交集: {"大学"} (1个), 并集: {"北京", "清华", "大学"} (3个)
        expected_similarity = 1/3
        assert abs(similarity - expected_similarity) < 0.01

    def test_calculate_name_similarity_empty_strings(self):
        """测试空字符串的相似度"""
        similarity = _calculate_name_similarity("", "")
        assert similarity == 0.0

    def test_calculate_name_similarity_one_empty(self):
        """测试一个空字符串的相似度"""
        similarity = _calculate_name_similarity("", "非空字符串")
        assert similarity == 0.0

        similarity = _calculate_name_similarity("非空字符串", "")
        assert similarity == 0.0

    def test_calculate_name_similarity_case_insensitive(self):
        """测试大小写不敏感"""
        similarity = _calculate_name_similarity("Computer Science", "COMPUTER SCIENCE")
        assert similarity == 1.0

        similarity = _calculate_name_similarity("Computer Science", "computer science")
        assert similarity == 1.0

    def test_calculate_name_similarity_whitespace_normalization(self):
        """测试空白字符规范化"""
        similarity = _calculate_name_similarity("  Computer   Science  ", "Computer Science")
        assert similarity == 1.0

    def test_merge_similar_entities_successful_merge(self, sample_graph):
        """测试成功合并相似实体"""
        # 添加一个相似的实体
        similar_entity = Entity(
            id="entity_similar",
            name="张三",  # 与entity1同名
            entity_type=EntityType.PERSON,
            description="另一个张三",
            confidence=0.8
        )
        sample_graph.add_entity(similar_entity)

        # 模拟merge_entity方法返回成功
        with patch.object(sample_graph, 'merge_entity', return_value=True) as mock_merge:
            merged_count = merge_similar_entities(sample_graph, similarity_threshold=0.8)

            assert merged_count >= 1
            mock_merge.assert_called()

    def test_merge_similar_entities_no_merge_needed(self, sample_graph):
        """测试不需要合并的情况"""
        # 所有实体名称都不同，相似度为0
        merged_count = merge_similar_entities(sample_graph, similarity_threshold=0.8)
        assert merged_count == 0

    def test_merge_similar_entities_high_threshold(self, sample_graph):
        """测试高相似度阈值"""
        # 添加部分相似的实体
        partial_similar = Entity(
            id="partial_similar",
            name="张 三",  # 与"张三"部分相似
            entity_type=EntityType.PERSON,
            confidence=0.8
        )
        sample_graph.add_entity(partial_similar)

        # 使用高阈值，应该不会合并
        merged_count = merge_similar_entities(sample_graph, similarity_threshold=0.9)
        assert merged_count == 0

    def test_merge_similar_entities_failed_merge(self, sample_graph):
        """测试合并失败的情况"""
        similar_entity = Entity(
            id="entity_similar",
            name="张三",
            entity_type=EntityType.PERSON,
            confidence=0.8
        )
        sample_graph.add_entity(similar_entity)

        # 模拟merge_entity方法返回失败
        with patch.object(sample_graph, 'merge_entity', return_value=False):
            merged_count = merge_similar_entities(sample_graph, similarity_threshold=0.8)
            assert merged_count == 0

    @patch('agraph.utils.logger')
    def test_merge_similar_entities_logging(self, mock_logger, sample_graph):
        """测试合并过程中的日志记录"""
        similar_entity = Entity(
            id="entity_similar",
            name="张三",
            entity_type=EntityType.PERSON,
            confidence=0.8
        )
        sample_graph.add_entity(similar_entity)

        with patch.object(sample_graph, 'merge_entity', return_value=True):
            merge_similar_entities(sample_graph, similarity_threshold=0.8)

            # 验证日志被调用
            mock_logger.info.assert_called()
            call_args = mock_logger.info.call_args[0]
            assert "Merged entities" in call_args[0]

    def test_merge_similar_entities_empty_graph(self, empty_graph):
        """测试空图谱的实体合并"""
        merged_count = merge_similar_entities(empty_graph, similarity_threshold=0.8)
        assert merged_count == 0

    def test_merge_similar_entities_single_entity(self):
        """测试单个实体的图谱"""
        graph = KnowledgeGraph("single_entity")
        entity = Entity(id="single", name="单个实体", entity_type=EntityType.PERSON)
        graph.add_entity(entity)

        merged_count = merge_similar_entities(graph, similarity_threshold=0.8)
        assert merged_count == 0


class TestValidation:
    """测试图谱一致性验证功能"""

    def test_validate_graph_consistency_valid_graph(self, sample_graph):
        """测试有效图谱的一致性验证"""
        issues = validate_graph_consistency(sample_graph)

        assert isinstance(issues, list)
        # 对于有效图谱，问题列表应该为空
        assert len(issues) == 0

    def test_validate_graph_consistency_missing_head_entity(self, sample_graph):
        """测试缺失头实体的一致性验证"""
        # 创建一个引用不存在实体的关系
        nonexistent_entity = Entity(id="nonexistent", name="不存在", entity_type=EntityType.PERSON)
        invalid_relation = Relation(
            id="invalid_rel",
            head_entity=nonexistent_entity,
            tail_entity=sample_graph.entities["entity1"],
            relation_type=RelationType.RELATED_TO
        )
        sample_graph.relations["invalid_rel"] = invalid_relation

        issues = validate_graph_consistency(sample_graph)

        # 应该检测到缺失头实体的问题
        missing_head_issues = [issue for issue in issues if issue["type"] == "missing_head_entity"]
        assert len(missing_head_issues) == 1

        issue = missing_head_issues[0]
        assert issue["relation_id"] == "invalid_rel"
        assert "头实体不存在" in issue["description"]

    def test_validate_graph_consistency_missing_tail_entity(self, sample_graph):
        """测试缺失尾实体的一致性验证"""
        nonexistent_entity = Entity(id="nonexistent", name="不存在", entity_type=EntityType.PERSON)
        invalid_relation = Relation(
            id="invalid_rel",
            head_entity=sample_graph.entities["entity1"],
            tail_entity=nonexistent_entity,
            relation_type=RelationType.RELATED_TO
        )
        sample_graph.relations["invalid_rel"] = invalid_relation

        issues = validate_graph_consistency(sample_graph)

        missing_tail_issues = [issue for issue in issues if issue["type"] == "missing_tail_entity"]
        assert len(missing_tail_issues) == 1

        issue = missing_tail_issues[0]
        assert issue["relation_id"] == "invalid_rel"
        assert "尾实体不存在" in issue["description"]

    def test_validate_graph_consistency_null_entities(self, sample_graph):
        """测试空实体引用的验证"""
        # 创建头实体为None的关系
        invalid_relation1 = Relation(
            id="null_head",
            head_entity=None,
            tail_entity=sample_graph.entities["entity1"],
            relation_type=RelationType.RELATED_TO
        )

        # 创建尾实体为None的关系
        invalid_relation2 = Relation(
            id="null_tail",
            head_entity=sample_graph.entities["entity1"],
            tail_entity=None,
            relation_type=RelationType.RELATED_TO
        )

        sample_graph.relations["null_head"] = invalid_relation1
        sample_graph.relations["null_tail"] = invalid_relation2

        issues = validate_graph_consistency(sample_graph)

        assert len(issues) == 2
        issue_types = [issue["type"] for issue in issues]
        assert "missing_head_entity" in issue_types
        assert "missing_tail_entity" in issue_types

    def test_validate_graph_consistency_self_loop(self, sample_graph):
        """测试自环关系的检测"""
        # 创建自环关系
        self_loop_relation = Relation(
            id="self_loop",
            head_entity=sample_graph.entities["entity1"],
            tail_entity=sample_graph.entities["entity1"],
            relation_type=RelationType.RELATED_TO
        )
        sample_graph.relations["self_loop"] = self_loop_relation

        issues = validate_graph_consistency(sample_graph)

        # 应该检测到自环问题
        self_loop_issues = [issue for issue in issues if issue["type"] == "self_loop"]
        assert len(self_loop_issues) == 1

        issue = self_loop_issues[0]
        assert issue["relation_id"] == "self_loop"
        assert issue["entity_id"] == "entity1"
        assert "自环关系" in issue["description"]

    def test_validate_graph_consistency_duplicate_relations(self, sample_graph):
        """测试重复关系的检测"""
        # 创建与现有关系相同的重复关系
        duplicate_relation = Relation(
            id="duplicate_rel",
            head_entity=sample_graph.entities["entity1"],
            tail_entity=sample_graph.entities["entity2"],
            relation_type=RelationType.BELONGS_TO,  # 与rel1相同
            confidence=0.5
        )
        sample_graph.relations["duplicate_rel"] = duplicate_relation

        issues = validate_graph_consistency(sample_graph)

        # 应该检测到重复关系问题
        duplicate_issues = [issue for issue in issues if issue["type"] == "duplicate_relation"]
        assert len(duplicate_issues) == 1

        issue = duplicate_issues[0]
        assert issue["relation_id"] == "duplicate_rel"
        assert "重复关系" in issue["description"]

    def test_validate_graph_consistency_multiple_issues(self, sample_graph):
        """测试检测多个问题的情况"""
        # 添加多种类型的问题

        # 1. 缺失实体的关系
        nonexistent = Entity(id="nonexistent", name="不存在", entity_type=EntityType.PERSON)
        missing_rel = Relation(
            id="missing_rel",
            head_entity=nonexistent,
            tail_entity=sample_graph.entities["entity1"],
            relation_type=RelationType.RELATED_TO
        )

        # 2. 自环关系
        self_loop_rel = Relation(
            id="self_loop_rel",
            head_entity=sample_graph.entities["entity2"],
            tail_entity=sample_graph.entities["entity2"],
            relation_type=RelationType.RELATED_TO
        )

        # 3. 重复关系
        duplicate_rel = Relation(
            id="duplicate_rel",
            head_entity=sample_graph.entities["entity1"],
            tail_entity=sample_graph.entities["entity2"],
            relation_type=RelationType.BELONGS_TO
        )

        sample_graph.relations.update({
            "missing_rel": missing_rel,
            "self_loop_rel": self_loop_rel,
            "duplicate_rel": duplicate_rel
        })

        issues = validate_graph_consistency(sample_graph)

        # 应该检测到所有类型的问题
        issue_types = [issue["type"] for issue in issues]
        assert "missing_head_entity" in issue_types
        assert "self_loop" in issue_types
        assert "duplicate_relation" in issue_types
        assert len(issues) >= 3

    def test_validate_graph_consistency_empty_graph(self, empty_graph):
        """测试空图谱的一致性验证"""
        issues = validate_graph_consistency(empty_graph)
        assert issues == []

    def test_validate_graph_consistency_no_relations(self):
        """测试只有实体没有关系的图谱"""
        graph = KnowledgeGraph("entities_only")
        entity = Entity(id="entity", name="实体", entity_type=EntityType.PERSON)
        graph.add_entity(entity)

        issues = validate_graph_consistency(graph)
        assert issues == []

    def test_validate_graph_consistency_issue_structure(self, sample_graph):
        """测试问题报告的结构"""
        # 添加一个自环关系来产生问题
        self_loop = Relation(
            id="test_loop",
            head_entity=sample_graph.entities["entity1"],
            tail_entity=sample_graph.entities["entity1"],
            relation_type=RelationType.RELATED_TO
        )
        sample_graph.relations["test_loop"] = self_loop

        issues = validate_graph_consistency(sample_graph)

        assert len(issues) == 1
        issue = issues[0]

        # 验证问题报告包含必要字段
        required_fields = ["type", "relation_id", "description"]
        for field in required_fields:
            assert field in issue

        # 对于自环问题，还应该包含entity_id
        if issue["type"] == "self_loop":
            assert "entity_id" in issue


class TestSummary:
    """测试图谱摘要生成功能"""

    def test_create_graph_summary_complete(self, sample_graph):
        """测试创建完整的图谱摘要"""
        summary = create_graph_summary(sample_graph)

        assert isinstance(summary, str)
        assert len(summary) > 0

        # 验证摘要包含所有必要部分
        required_sections = [
            "知识图谱摘要报告",
            "基本信息",
            "统计信息",
            "实体类型分布",
            "关系类型分布",
            "中心节点",
            "连通性"
        ]

        for section in required_sections:
            assert section in summary, f"摘要中缺少 '{section}' 部分"

        # 验证基本信息
        assert sample_graph.name in summary
        assert sample_graph.id in summary

        # 验证统计信息
        assert "实体数量: 3" in summary
        assert "关系数量: 2" in summary

        # 验证实体类型分布 (枚举值是小写的)
        assert "person: 1个" in summary
        assert "organization: 1个" in summary
        assert "concept: 1个" in summary

        # 验证关系类型分布 (枚举值是小写的)
        assert "belongs_to: 1个" in summary
        assert "related_to: 1个" in summary

        # 验证中心节点信息
        assert "北京大学" in summary  # 应该是中心节点

        # 验证连通性信息
        assert "连通分量数: 1" in summary
        assert "最大连通分量大小: 3" in summary
        assert "是否连通: 是" in summary

    def test_create_graph_summary_empty_graph(self, empty_graph):
        """测试空图谱的摘要生成"""
        summary = create_graph_summary(empty_graph)

        assert isinstance(summary, str)
        assert "知识图谱摘要报告" in summary
        assert "实体数量: 0" in summary
        assert "关系数量: 0" in summary
        assert "连通分量数: 0" in summary
        assert "是否连通: 否" in summary

    def test_create_graph_summary_single_entity(self):
        """测试单实体图谱的摘要"""
        graph = KnowledgeGraph("single_entity_graph")
        entity = Entity(
            id="single",
            name="单个实体",
            entity_type=EntityType.PERSON,
            confidence=0.9
        )
        graph.add_entity(entity)

        summary = create_graph_summary(graph)

        assert "实体数量: 1" in summary
        assert "关系数量: 0" in summary
        assert "person: 1个" in summary
        assert "连通分量数: 1" in summary
        assert "最大连通分量大小: 1" in summary
        assert "是否连通: 是" in summary

    def test_create_graph_summary_disconnected_graph(self, sample_graph):
        """测试非连通图谱的摘要"""
        # 添加孤立实体
        isolated = Entity(
            id="isolated",
            name="孤立实体",
            entity_type=EntityType.CONCEPT,
            confidence=0.7
        )
        sample_graph.add_entity(isolated)

        summary = create_graph_summary(sample_graph)

        assert "实体数量: 4" in summary
        assert "关系数量: 2" in summary
        assert "连通分量数: 2" in summary
        assert "最大连通分量大小: 3" in summary
        assert "是否连通: 否" in summary

    def test_create_graph_summary_date_formatting(self, sample_graph):
        """测试日期格式化"""
        # 设置特定的创建和更新时间
        test_time = datetime(2024, 1, 15, 10, 30, 45)
        sample_graph.created_at = test_time
        sample_graph.updated_at = test_time

        summary = create_graph_summary(sample_graph)

        # 验证日期格式
        assert "2024-01-15 10:30:45" in summary

    def test_create_graph_summary_central_nodes_limit(self):
        """测试中心节点数量限制"""
        graph = KnowledgeGraph("hub_graph")

        # 创建多个实体
        entities = []
        for i in range(10):
            entity = Entity(
                id=f"entity_{i}",
                name=f"实体{i}",
                entity_type=EntityType.CONCEPT,
                confidence=0.8
            )
            entities.append(entity)
            graph.add_entity(entity)

        # 创建星形结构，entity_0为中心
        for i in range(1, 10):
            relation = Relation(
                id=f"rel_{i}",
                head_entity=entities[0],
                tail_entity=entities[i],
                relation_type=RelationType.RELATED_TO
            )
            graph.add_relation(relation)

        summary = create_graph_summary(graph)

        # 验证只显示前3个中心节点（注意：搜索条件更精确）
        lines = summary.split('\n')
        central_node_section_started = False
        central_node_lines = []

        for line in lines:
            if "中心节点" in line:
                central_node_section_started = True
                continue
            elif central_node_section_started:
                if "连通性:" in line:  # 到达下一节时停止
                    break
                if "度数:" in line and line.strip().startswith('-'):
                    central_node_lines.append(line)

        assert len(central_node_lines) <= 3

    def test_create_graph_summary_complex_types(self):
        """测试复杂类型分布的摘要"""
        graph = KnowledgeGraph("complex_types")

        # 创建不同类型的实体（使用实际存在的类型）
        entity_types = [
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            EntityType.LOCATION,
            EntityType.CONCEPT,
            EntityType.DOCUMENT
        ]

        entities = []
        for i, etype in enumerate(entity_types):
            # 每种类型创建2个实体
            for j in range(2):
                entity = Entity(
                    id=f"entity_{etype.value}_{j}",
                    name=f"{etype.value}实体{j}",
                    entity_type=etype,
                    confidence=0.8
                )
                entities.append(entity)
                graph.add_entity(entity)

        # 创建不同类型的关系（使用实际存在的类型）
        relation_types = [
            RelationType.BELONGS_TO,
            RelationType.RELATED_TO,
            RelationType.CONTAINS,
            RelationType.REFERENCES
        ]

        for i, rtype in enumerate(relation_types):
            if i + 1 < len(entities):
                relation = Relation(
                    id=f"rel_{rtype.value}_{i}",
                    head_entity=entities[i],
                    tail_entity=entities[i + 1],
                    relation_type=rtype
                )
                graph.add_relation(relation)

        summary = create_graph_summary(graph)

        # 验证所有类型都在摘要中
        for etype in entity_types:
            assert f"{etype.value}: 2个" in summary

        for rtype in relation_types:
            assert f"{rtype.value}: 1个" in summary

    @patch('agraph.utils.calculate_graph_metrics')
    def test_create_graph_summary_metrics_integration(self, mock_calculate_metrics, sample_graph):
        """测试摘要与指标计算的集成"""
        # 模拟指标计算结果
        mock_metrics = {
            "basic_stats": {
                "density": 0.6667,
                "avg_degree": 1.33
            },
            "centrality": {
                "top_central_nodes": [
                    {"entity_name": "测试实体", "degree": 5}
                ]
            },
            "connectivity": {
                "connected_components": 1,
                "largest_component_size": 3,
                "is_connected": True
            }
        }
        mock_calculate_metrics.return_value = mock_metrics

        summary = create_graph_summary(sample_graph)

        # 验证使用了模拟的指标
        mock_calculate_metrics.assert_called_once_with(sample_graph)
        assert "图密度: 0.6667" in summary
        assert "平均度数: 1.33" in summary
        assert "测试实体" in summary


class TestUtilsModuleIntegration:
    """测试utils模块的集成测试"""

    def test_utils_module_imports(self):
        """测试模块导入的完整性"""
        # 确保所有导出的函数都可以正常导入
        from agraph.utils import (
            export_graph_to_cytoscape,
            export_graph_to_d3,
            find_shortest_path,
            calculate_graph_metrics,
            merge_similar_entities,
            validate_graph_consistency,
            create_graph_summary,
        )

        # 检查函数是否可调用
        functions_to_test = [
            export_graph_to_cytoscape,
            export_graph_to_d3,
            find_shortest_path,
            calculate_graph_metrics,
            merge_similar_entities,
            validate_graph_consistency,
            create_graph_summary,
        ]

        for func in functions_to_test:
            assert callable(func), f"函数 {func.__name__} 不可调用"

    def test_internal_functions_import(self):
        """测试内部函数的导入"""
        from agraph.utils import (
            _calculate_name_similarity,
            _find_connected_components,
            _dfs_component,
            _calculate_type_distribution,
        )

        internal_functions = [
            _calculate_name_similarity,
            _find_connected_components,
            _dfs_component,
            _calculate_type_distribution,
        ]

        for func in internal_functions:
            assert callable(func), f"内部函数 {func.__name__} 不可调用"

    def test_full_workflow_integration(self, sample_graph):
        """测试完整工作流的集成"""
        # 1. 验证图谱一致性
        issues = validate_graph_consistency(sample_graph)
        assert len(issues) == 0, "图谱应该是一致的"

        # 2. 计算图谱指标
        metrics = calculate_graph_metrics(sample_graph)
        assert "basic_stats" in metrics

        # 3. 查找路径
        path = find_shortest_path(sample_graph, "entity1", "entity3")
        assert path is not None
        assert len(path) == 2

        # 4. 导出数据
        cytoscape_data = export_graph_to_cytoscape(sample_graph)
        d3_data = export_graph_to_d3(sample_graph)
        assert len(cytoscape_data["elements"]["nodes"]) == len(d3_data["nodes"])

        # 5. 生成摘要
        summary = create_graph_summary(sample_graph)
        assert "知识图谱摘要报告" in summary

        # 6. 合并相似实体（应该不会合并任何实体，因为名称都不同）
        merged_count = merge_similar_entities(sample_graph, similarity_threshold=0.8)
        assert merged_count == 0

    def test_error_handling_integration(self):
        """测试错误处理的集成"""
        # 测试空图谱的处理
        empty_graph = KnowledgeGraph("empty")

        # 所有函数都应该能处理空图谱
        assert export_graph_to_cytoscape(empty_graph)["elements"]["nodes"] == []
        assert export_graph_to_d3(empty_graph)["nodes"] == []
        assert find_shortest_path(empty_graph, "nonexistent1", "nonexistent2") is None
        assert calculate_graph_metrics(empty_graph) == {}
        assert validate_graph_consistency(empty_graph) == []
        assert merge_similar_entities(empty_graph) == 0

        summary = create_graph_summary(empty_graph)
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_data_consistency_across_exports(self, sample_graph):
        """测试不同导出格式的数据一致性"""
        cytoscape_data = export_graph_to_cytoscape(sample_graph)
        d3_data = export_graph_to_d3(sample_graph)

        # 节点数量应该一致
        assert len(cytoscape_data["elements"]["nodes"]) == len(d3_data["nodes"])

        # 边数量应该一致
        assert len(cytoscape_data["elements"]["edges"]) == len(d3_data["links"])

        # 图谱信息应该一致
        cytoscape_info = cytoscape_data["graph_info"]
        d3_info = d3_data["graph_info"]

        assert cytoscape_info["id"] == d3_info["id"]
        assert cytoscape_info["name"] == d3_info["name"]

    def test_performance_with_larger_graph(self):
        """测试较大图谱的性能"""
        # 创建一个较大的图谱
        large_graph = KnowledgeGraph("large_graph")

        # 添加100个实体
        entities = []
        for i in range(100):
            entity = Entity(
                id=f"entity_{i}",
                name=f"实体{i}",
                entity_type=EntityType.CONCEPT,
                confidence=0.8
            )
            entities.append(entity)
            large_graph.add_entity(entity)

        # 添加一些关系（创建链式结构）
        for i in range(99):
            relation = Relation(
                id=f"rel_{i}",
                head_entity=entities[i],
                tail_entity=entities[i + 1],
                relation_type=RelationType.RELATED_TO
            )
            large_graph.add_relation(relation)

        # 所有功能都应该能在合理时间内完成
        import time

        start_time = time.time()
        metrics = calculate_graph_metrics(large_graph)
        metrics_time = time.time() - start_time
        assert metrics_time < 5.0, "指标计算应该在5秒内完成"

        start_time = time.time()
        cytoscape_data = export_graph_to_cytoscape(large_graph)
        export_time = time.time() - start_time
        assert export_time < 2.0, "导出应该在2秒内完成"

        # 验证结果正确性
        assert metrics["basic_stats"]["node_count"] == 100
        assert metrics["basic_stats"]["edge_count"] == 99
        assert len(cytoscape_data["elements"]["nodes"]) == 100
        assert len(cytoscape_data["elements"]["edges"]) == 99


# 模块级别的测试函数保持独立
def test_module_level_function_access():
    """测试模块级别的函数访问"""
    import agraph.utils as utils_module

    # 验证可以通过模块直接访问函数
    assert hasattr(utils_module, 'export_graph_to_cytoscape')
    assert hasattr(utils_module, 'export_graph_to_d3')
    assert hasattr(utils_module, 'find_shortest_path')
    assert hasattr(utils_module, 'calculate_graph_metrics')
    assert hasattr(utils_module, 'merge_similar_entities')
    assert hasattr(utils_module, 'validate_graph_consistency')
    assert hasattr(utils_module, 'create_graph_summary')


def test_utils_constants_and_configuration():
    """测试utils模块的常量和配置"""
    import agraph.utils as utils_module

    # 验证模块有正确的logger配置
    assert hasattr(utils_module, 'logger')
    assert utils_module.logger.name == 'agraph.utils'


@pytest.mark.parametrize("similarity_threshold", [0.0, 0.5, 0.8, 1.0])
def test_merge_similarity_thresholds(similarity_threshold):
    """测试不同相似度阈值的合并行为"""
    graph = KnowledgeGraph("threshold_test")

    # 添加两个部分相似的实体
    entity1 = Entity(
        id="entity1",
        name="Computer Science",
        entity_type=EntityType.CONCEPT,
        confidence=0.9
    )
    entity2 = Entity(
        id="entity2",
        name="Computer Engineering",  # 部分相似
        entity_type=EntityType.CONCEPT,
        confidence=0.8
    )

    graph.add_entity(entity1)
    graph.add_entity(entity2)

    with patch.object(graph, 'merge_entity', return_value=True) as mock_merge:
        merged_count = merge_similar_entities(graph, similarity_threshold=similarity_threshold)

        # 计算实际相似度
        actual_similarity = _calculate_name_similarity("Computer Science", "Computer Engineering")

        if actual_similarity >= similarity_threshold:
            assert merged_count > 0
            mock_merge.assert_called()
        else:
            assert merged_count == 0
            mock_merge.assert_not_called()
