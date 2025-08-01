import unittest
import asyncio
import tempfile
import shutil
import logging
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from pathlib import Path
import xml.etree.ElementTree as ET
from datetime import datetime

from agraph.builders.lightrag_builder import LightRAGGraphBuilder
from agraph.graph import KnowledgeGraph
from agraph.entities import Entity
from agraph.relations import Relation
from agraph.types import EntityType, RelationType


def async_test(func):
    """装饰器，用于运行异步测试"""
    def wrapper(self):
        return asyncio.run(func(self))
    return wrapper


class TestLightRAGGraphBuilder(unittest.TestCase):
    """测试 LightRAG 图构建器"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.builder = LightRAGGraphBuilder(working_dir=self.temp_dir)

    def tearDown(self):
        """清理测试环境"""
        # 清理临时目录
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

        # 清理日志处理器以避免关闭错误
        try:
            logger = logging.getLogger('agraph.builders.lightrag_builder')
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        except Exception:
            pass

    def test_init(self):
        """测试初始化"""
        builder = LightRAGGraphBuilder("test_dir")
        self.assertEqual(builder.working_dir, Path("test_dir"))
        self.assertIsNone(builder.rag_instance)

    def test_working_directory_creation(self):
        """测试工作目录设置"""
        test_dir = Path(self.temp_dir) / "new_lightrag_dir"
        builder = LightRAGGraphBuilder(working_dir=str(test_dir))

        # 工作目录路径应该被正确设置
        self.assertEqual(builder.working_dir, test_dir)
        # 目录在构造函数中不会立即创建，而是在初始化时创建

    @async_test
    @patch('agraph.builders.lightrag_builder.Settings')
    async def test_initialize_lightrag_success(self, mock_settings):
        """测试成功初始化 LightRAG"""
        # 设置模拟配置
        mock_settings.LLM_MODEL = "test-model"
        mock_settings.OPENAI_API_BASE = "http://test.api"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.EMBEDDING_MODEL = "test-embedding"
        mock_settings.EMBEDDING_DIM = 1024
        mock_settings.EMBEDDING_MAX_TOKEN_SIZE = 8192

        # 模拟 LightRAG 导入
        mock_lightrag = Mock()
        mock_rag_instance = Mock()
        mock_rag_instance.initialize_storages = AsyncMock()
        mock_lightrag.return_value = mock_rag_instance

        mock_embedding_func = Mock()
        mock_initialize_pipeline = AsyncMock()

        with patch.dict('sys.modules', {
            'lightrag': Mock(LightRAG=mock_lightrag),
            'lightrag.kg.shared_storage': Mock(initialize_pipeline_status=mock_initialize_pipeline),
            'lightrag.llm.openai': Mock(
                openai_complete_if_cache=AsyncMock(return_value="test response"),
                openai_embed=AsyncMock(return_value=[[0.1, 0.2, 0.3]])
            ),
            'lightrag.utils': Mock(EmbeddingFunc=mock_embedding_func)
        }):
            result = await self.builder.initialize_lightrag()

            self.assertEqual(result, mock_rag_instance)
            self.assertEqual(self.builder.rag_instance, mock_rag_instance)
            mock_rag_instance.initialize_storages.assert_called_once()
            mock_initialize_pipeline.assert_called_once()

    @async_test
    async def test_initialize_lightrag_import_error(self):
        """测试 LightRAG 导入错误"""
        with patch.dict('sys.modules', {'lightrag': None}, clear=False):
            with self.assertRaises(ImportError):
                await self.builder.initialize_lightrag()

    @async_test
    async def test_initialize_lightrag_general_error(self):
        """测试 LightRAG 初始化一般错误"""
        mock_lightrag = Mock(side_effect=Exception("Test error"))

        with patch.dict('sys.modules', {
            'lightrag': Mock(LightRAG=mock_lightrag)
        }):
            with self.assertRaises(Exception):
                await self.builder.initialize_lightrag()

    @async_test
    async def test_initialize_lightrag_cached(self):
        """测试 LightRAG 实例缓存"""
        mock_instance = Mock()
        self.builder.rag_instance = mock_instance

        result = await self.builder.initialize_lightrag()
        self.assertEqual(result, mock_instance)

    def test_build_graph_sync(self):
        """测试同步构建图"""
        mock_graph = KnowledgeGraph(name="test_graph")

        # 模拟异步方法，确保正确返回协程
        async def mock_build_graph_async(*args, **kwargs):
            return mock_graph

        with patch.object(self.builder, '_build_graph_async', side_effect=mock_build_graph_async):
            result = self.builder.build_graph(
                texts=["test text"],
                graph_name="test_graph"
            )

            self.assertEqual(result.name, "test_graph")

    # def test_build_graph_new_event_loop(self):
    #     """测试在没有事件循环时创建新循环"""
    #     with patch('asyncio.get_event_loop', side_effect=RuntimeError("No event loop")):
    #         with patch('asyncio.new_event_loop') as mock_new_loop:
    #             with patch('asyncio.set_event_loop') as mock_set_loop:
    #                 mock_loop = Mock()
    #                 mock_loop.run_until_complete.return_value = KnowledgeGraph()
    #                 mock_new_loop.return_value = mock_loop
    #
    #                 result = self.builder.build_graph()
    #
    #                 mock_new_loop.assert_called_once()
    #                 mock_set_loop.assert_called_once_with(mock_loop)
    #                 self.assertIsInstance(result, KnowledgeGraph)

    @async_test
    async def test_build_graph_async_with_texts(self):
        """测试异步构建图（包含文本）"""
        # 模拟 LightRAG 实例
        mock_rag = Mock()
        mock_rag.ainsert = AsyncMock()
        self.builder.rag_instance = mock_rag

        # 创建模拟 GraphML 文件
        graphml_content = self._create_mock_graphml()
        graphml_file = Path(self.temp_dir) / "graph_chunk_entity_relation.graphml"
        graphml_file.write_text(graphml_content, encoding='utf-8')

        with patch.object(self.builder, 'initialize_lightrag', return_value=mock_rag):
            result = await self.builder._build_graph_async(
                texts=["test text 1", "test text 2"],
                graph_name="test_graph"
            )

            # 验证文本插入调用
            self.assertEqual(mock_rag.ainsert.call_count, 2)
            mock_rag.ainsert.assert_any_call("test text 1")
            mock_rag.ainsert.assert_any_call("test text 2")

            # 验证图构建结果
            self.assertEqual(result.name, "test_graph")
            self.assertGreater(len(result.entities), 0)

    @async_test
    async def test_build_graph_async_no_graphml(self):
        """测试异步构建图（无 GraphML 文件）"""
        mock_rag = Mock()
        self.builder.rag_instance = mock_rag

        with patch.object(self.builder, 'initialize_lightrag', return_value=mock_rag):
            result = await self.builder._build_graph_async(graph_name="empty_graph")

            self.assertEqual(result.name, "empty_graph")
            self.assertEqual(len(result.entities), 0)

    @async_test
    async def test_build_graph_async_rag_not_initialized(self):
        """测试异步构建图（RAG 未初始化）"""
        with patch.object(self.builder, 'initialize_lightrag', return_value=None):
            with self.assertRaises(RuntimeError):
                await self.builder._build_graph_async()

    @async_test
    async def test_build_graph_async_insert_error(self):
        """测试异步构建图（插入错误）"""
        mock_rag = Mock()
        mock_rag.ainsert = AsyncMock(side_effect=Exception("Insert failed"))

        with patch.object(self.builder, 'initialize_lightrag', return_value=mock_rag):
            with self.assertRaises(Exception):
                await self.builder._build_graph_async(texts=["test text"])

    def test_update_graph(self):
        """测试更新图"""
        graph = KnowledgeGraph(name="test_graph")
        entity = Entity(name="TestEntity")
        relation = Relation(head_entity=entity, tail_entity=entity, relation_type=RelationType.RELATED_TO)

        # 更新图应该返回原图（LightRAG 不支持直接更新）
        result = self.builder.update_graph(graph, [entity], [relation])
        self.assertEqual(result, graph)

    def test_add_documents_sync(self):
        """测试同步添加文档"""
        mock_graph = KnowledgeGraph(name="updated_graph")

        # 模拟异步方法，确保正确返回协程
        async def mock_add_documents_async(*args, **kwargs):
            return mock_graph

        with patch.object(self.builder, '_add_documents_async', side_effect=mock_add_documents_async):
            result = self.builder.add_documents(["doc1", "doc2"], "updated_graph")

            self.assertEqual(result.name, "updated_graph")

    @async_test
    async def test_add_documents_async(self):
        """测试异步添加文档"""
        mock_rag = Mock()
        mock_rag.ainsert = AsyncMock()

        # 创建模拟 GraphML 文件
        graphml_content = self._create_mock_graphml()
        graphml_file = Path(self.temp_dir) / "graph_chunk_entity_relation.graphml"
        graphml_file.write_text(graphml_content, encoding='utf-8')

        # 直接设置 rag_instance 而不是模拟 initialize_lightrag
        self.builder.rag_instance = mock_rag

        result = await self.builder._add_documents_async(
            ["document 1", "document 2"],
            "updated_graph"
        )

        # 验证文档插入
        self.assertEqual(mock_rag.ainsert.call_count, 2)
        self.assertEqual(result.name, "updated_graph")

    @async_test
    async def test_add_documents_async_no_graphml(self):
        """测试异步添加文档（无 GraphML 文件）"""
        mock_rag = Mock()
        mock_rag.ainsert = AsyncMock()

        # 直接设置 rag_instance
        self.builder.rag_instance = mock_rag

        result = await self.builder._add_documents_async(["doc"], "test_graph")

        self.assertEqual(result.name, "test_graph")
        self.assertEqual(len(result.entities), 0)

    def test_search_graph_sync(self):
        """测试同步搜索图"""
        mock_result = {"query": "test", "result": "search result"}

        # 模拟异步方法，确保正确返回协程
        async def mock_search_graph_async(*args, **kwargs):
            return mock_result

        with patch.object(self.builder, '_search_graph_async', side_effect=mock_search_graph_async):
            result = self.builder.search_graph("test query", "hybrid")

            self.assertEqual(result["query"], "test")

    @async_test
    async def test_search_graph_async(self):
        """测试异步搜索图"""
        mock_rag = Mock()
        mock_rag.aquery = AsyncMock(return_value="search result")

        # 直接设置 rag_instance
        self.builder.rag_instance = mock_rag

        # 模拟 QueryParam
        mock_query_param = Mock()
        with patch.dict('sys.modules', {
            'lightrag': Mock(QueryParam=mock_query_param)
        }):
            result = await self.builder._search_graph_async("test query", "local")

            self.assertEqual(result["query"], "test query")
            self.assertEqual(result["search_type"], "local")
            self.assertEqual(result["result"], "search result")
            self.assertIn("timestamp", result)

    @async_test
    async def test_search_graph_async_rag_not_initialized(self):
        """测试异步搜索图（RAG 未初始化）"""
        with patch.object(self.builder, 'initialize_lightrag', return_value=None):
            with self.assertRaises(RuntimeError):
                await self.builder._search_graph_async("test")

    def test_cleanup_sync(self):
        """测试同步清理"""
        # 模拟异步方法，确保正确返回协程
        async def mock_cleanup_async():
            return None

        with patch.object(self.builder, '_cleanup_async', side_effect=mock_cleanup_async):
            self.builder.cleanup()

    @async_test
    async def test_cleanup_async(self):
        """测试异步清理"""
        mock_rag = Mock()
        mock_rag.finalize_storages = AsyncMock()
        self.builder.rag_instance = mock_rag

        await self.builder._cleanup_async()

        mock_rag.finalize_storages.assert_called_once()
        self.assertIsNone(self.builder.rag_instance)

    @async_test
    async def test_cleanup_async_error(self):
        """测试异步清理错误"""
        mock_rag = Mock()
        mock_rag.finalize_storages = AsyncMock(side_effect=Exception("Cleanup failed"))
        self.builder.rag_instance = mock_rag

        # 不应该抛出异常
        await self.builder._cleanup_async()
        self.assertIsNone(self.builder.rag_instance)

    def test_load_graph_from_graphml(self):
        """测试从 GraphML 加载图"""
        # 创建模拟 GraphML 文件
        graphml_content = self._create_mock_graphml()
        graphml_file = Path(self.temp_dir) / "test.graphml"
        graphml_file.write_text(graphml_content, encoding='utf-8')

        graph = self.builder._load_graph_from_graphml(str(graphml_file), "loaded_graph")

        self.assertEqual(graph.name, "loaded_graph")
        self.assertGreater(len(graph.entities), 0)
        # 检查实体属性
        entity = list(graph.entities.values())[0]
        self.assertEqual(entity.source, "lightrag")

    def test_load_graph_from_graphml_file_not_found(self):
        """测试加载不存在的 GraphML 文件"""
        with self.assertRaises(Exception):
            self.builder._load_graph_from_graphml("nonexistent.graphml", "test")

    def test_parse_graphml_node(self):
        """测试解析 GraphML 节点"""
        # 创建模拟节点 XML
        node_xml = '''
        <node id="entity1" xmlns="http://graphml.graphdrawing.org/xmlns">
            <data key="d0">Test Entity</data>
            <data key="d1">person</data>
            <data key="d2">A test person entity</data>
            <data key="d3">source123</data>
        </node>
        '''
        node = ET.fromstring(node_xml)
        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

        entity = self.builder._parse_graphml_node(node, ns)

        self.assertIsNotNone(entity)
        self.assertEqual(entity.id, "entity1")
        self.assertEqual(entity.name, "Test Entity")
        self.assertEqual(entity.entity_type, EntityType.PERSON)
        self.assertEqual(entity.description, "A test person entity")
        self.assertEqual(entity.source, "lightrag")

    def test_parse_graphml_node_invalid(self):
        """测试解析无效的 GraphML 节点"""
        # 创建无效节点
        node_xml = '<invalid></invalid>'
        node = ET.fromstring(node_xml)
        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

        entity = self.builder._parse_graphml_node(node, ns)
        # 即使是无效节点，方法也会创建一个实体，但 ID 和 name 可能为 None
        self.assertIsNotNone(entity)
        self.assertIsNone(entity.id)
        self.assertIsNone(entity.name)

    def test_parse_graphml_edge(self):
        """测试解析 GraphML 边"""
        # 创建测试实体
        entity1 = Entity(id="e1", name="Entity1")
        entity2 = Entity(id="e2", name="Entity2")
        entities_map = {"e1": entity1, "e2": entity2}

        # 创建模拟边 XML
        edge_xml = '''
        <edge source="e1" target="e2" xmlns="http://graphml.graphdrawing.org/xmlns">
            <data key="d6">0.8</data>
            <data key="d7">Related entities</data>
            <data key="d8">keyword1,keyword2</data>
        </edge>
        '''
        edge = ET.fromstring(edge_xml)
        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

        relation = self.builder._parse_graphml_edge(edge, ns, entities_map)

        self.assertIsNotNone(relation)
        self.assertEqual(relation.head_entity, entity1)
        self.assertEqual(relation.tail_entity, entity2)
        self.assertEqual(relation.relation_type, RelationType.RELATED_TO)
        self.assertEqual(relation.confidence, 0.8)
        self.assertEqual(relation.source, "lightrag")

    def test_parse_graphml_edge_missing_entities(self):
        """测试解析缺少实体的 GraphML 边"""
        entities_map = {}

        edge_xml = '<edge source="e1" target="e2" xmlns="http://graphml.graphdrawing.org/xmlns"></edge>'
        edge = ET.fromstring(edge_xml)
        ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

        relation = self.builder._parse_graphml_edge(edge, ns, entities_map)
        self.assertIsNone(relation)

    def test_map_entity_type(self):
        """测试实体类型映射"""
        test_cases = [
            ("person", EntityType.PERSON),
            ("organization", EntityType.ORGANIZATION),
            ("location", EntityType.LOCATION),
            ("concept", EntityType.CONCEPT),
            ("unknown_type", EntityType.UNKNOWN),
            ("", EntityType.UNKNOWN),
        ]

        for input_type, expected_type in test_cases:
            result = self.builder._map_entity_type(input_type)
            self.assertEqual(result, expected_type)

    def test_export_to_graphml(self):
        """测试导出为 GraphML"""
        # 创建测试图
        graph = KnowledgeGraph(name="test_export")
        entity1 = Entity(name="Entity1", entity_type=EntityType.PERSON, description="Test person")
        entity2 = Entity(name="Entity2", entity_type=EntityType.ORGANIZATION)
        graph.add_entity(entity1)
        graph.add_entity(entity2)

        relation = Relation(
            head_entity=entity1,
            tail_entity=entity2,
            relation_type=RelationType.RELATED_TO,
            confidence=0.9
        )
        graph.add_relation(relation)

        output_path = Path(self.temp_dir) / "export_test.graphml"

        result = self.builder.export_to_graphml(graph, str(output_path))

        self.assertTrue(result)
        self.assertTrue(output_path.exists())

        # 验证导出的 XML 是否有效
        tree = ET.parse(str(output_path))
        root = tree.getroot()
        # XML 标签可能包含命名空间
        self.assertTrue(root.tag.endswith("graphml"))

    def test_export_to_graphml_error(self):
        """测试导出 GraphML 错误"""
        graph = KnowledgeGraph()

        # 使用无效路径
        result = self.builder.export_to_graphml(graph, "/invalid/path/test.graphml")
        self.assertFalse(result)

    def test_get_graph_statistics_with_file(self):
        """测试获取图统计信息（存在文件）"""
        # 创建模拟 GraphML 文件
        graphml_content = self._create_mock_graphml()
        graphml_file = Path(self.temp_dir) / "graph_chunk_entity_relation.graphml"
        graphml_file.write_text(graphml_content, encoding='utf-8')

        stats = self.builder.get_graph_statistics()

        self.assertEqual(stats["status"], "ready")
        self.assertGreater(stats["entities_count"], 0)
        self.assertIn("last_modified", stats)
        self.assertIn("graphml_file", stats)

    def test_get_graph_statistics_no_file(self):
        """测试获取图统计信息（无文件）"""
        stats = self.builder.get_graph_statistics()

        self.assertEqual(stats["status"], "no_graph")
        self.assertEqual(stats["entities_count"], 0)
        self.assertEqual(stats["relations_count"], 0)

    def test_get_graph_statistics_error(self):
        """测试获取图统计信息错误"""
        # 创建无效的 GraphML 文件
        graphml_file = Path(self.temp_dir) / "graph_chunk_entity_relation.graphml"
        graphml_file.write_text("invalid xml content", encoding='utf-8')

        stats = self.builder.get_graph_statistics()

        self.assertEqual(stats["status"], "error")
        self.assertIn("error", stats)

    def _create_mock_graphml(self) -> str:
        """创建模拟 GraphML 内容"""
        return '''<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
    <key id="d0" for="node" attr.name="entity_id" attr.type="string"/>
    <key id="d1" for="node" attr.name="entity_type" attr.type="string"/>
    <key id="d2" for="node" attr.name="description" attr.type="string"/>
    <key id="d6" for="edge" attr.name="weight" attr.type="double"/>
    <key id="d7" for="edge" attr.name="description" attr.type="string"/>

    <graph edgedefault="undirected">
        <node id="n1">
            <data key="d0">Test Entity 1</data>
            <data key="d1">person</data>
            <data key="d2">A test person</data>
        </node>
        <node id="n2">
            <data key="d0">Test Entity 2</data>
            <data key="d1">organization</data>
            <data key="d2">A test organization</data>
        </node>
        <edge source="n1" target="n2">
            <data key="d6">0.8</data>
            <data key="d7">Works for</data>
        </edge>
    </graph>
</graphml>'''


class TestLightRAGGraphBuilderIntegration(unittest.TestCase):
    """LightRAG 图构建器集成测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.builder = LightRAGGraphBuilder(working_dir=self.temp_dir)

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

        # 清理日志处理器以避免关闭错误
        try:
            logger = logging.getLogger('agraph.builders.lightrag_builder')
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        except Exception:
            pass

    @patch('agraph.builders.lightrag_builder.Settings')
    def test_full_workflow_mock(self, mock_settings):
        """测试完整工作流程（模拟）"""
        # 设置配置
        mock_settings.LLM_MODEL = "test-model"
        mock_settings.OPENAI_API_BASE = "http://test.api"
        mock_settings.OPENAI_API_KEY = "test-key"
        mock_settings.EMBEDDING_MODEL = "test-embedding"
        mock_settings.EMBEDDING_DIM = 1024
        mock_settings.EMBEDDING_MAX_TOKEN_SIZE = 8192

        # 模拟 LightRAG
        mock_rag = Mock()
        mock_rag.ainsert = AsyncMock()
        mock_rag.aquery = AsyncMock(return_value="search result")
        mock_rag.initialize_storages = AsyncMock()
        mock_rag.finalize_storages = AsyncMock()

        mock_lightrag_class = Mock(return_value=mock_rag)
        mock_initialize_pipeline = AsyncMock()

        with patch.dict('sys.modules', {
            'lightrag': Mock(LightRAG=mock_lightrag_class, QueryParam=Mock()),
            'lightrag.kg.shared_storage': Mock(initialize_pipeline_status=mock_initialize_pipeline),
            'lightrag.llm.openai': Mock(
                openai_complete_if_cache=AsyncMock(return_value="response"),
                openai_embed=AsyncMock(return_value=[[0.1, 0.2]])
            ),
            'lightrag.utils': Mock(EmbeddingFunc=Mock())
        }):
            # 创建模拟 GraphML 文件
            graphml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
    <graph edgedefault="undirected">
        <node id="n1"></node>
    </graph>
</graphml>'''
            graphml_file = Path(self.temp_dir) / "graph_chunk_entity_relation.graphml"
            graphml_file.write_text(graphml_content, encoding='utf-8')

            # 测试构建图
            graph = self.builder.build_graph(
                texts=["Sample document for testing"],
                graph_name="integration_test"
            )

            self.assertEqual(graph.name, "integration_test")

            # 测试搜索
            search_result = self.builder.search_graph("test query")
            self.assertIn("result", search_result)

            # 测试清理
            self.builder.cleanup()


class TestLightRAGGraphBuilderErrorHandling(unittest.TestCase):
    """LightRAG 图构建器错误处理测试"""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.builder = LightRAGGraphBuilder(working_dir=self.temp_dir)

    def tearDown(self):
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

        # 清理日志处理器以避免关闭错误
        try:
            logger = logging.getLogger('agraph.builders.lightrag_builder')
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)
        except Exception:
            pass

    # def test_build_graph_with_database_schema_warning(self):
    #     """测试使用数据库模式构建图的警告"""
    #     with patch('agraph.builders.lightrag_builder.logger') as mock_logger:
    #         # 数据库模式应该被忽略，但不应该导致错误
    #         graph = self.builder.build_graph(
    #             database_schema={"tables": ["test_table"]},
    #             graph_name="db_test"
    #         )
    #
    #         self.assertEqual(graph.name, "db_test")

    # def test_xml_parsing_robustness(self):
    #     """测试 XML 解析的鲁棒性"""
    #     # 测试格式错误的 XML
    #     with self.assertRaises(ET.ParseError):
    #         self.builder._load_graph_from_graphml("nonexistent", "test")
    #
    @async_test
    async def test_async_operations_exception_handling(self):
        """测试异步操作的异常处理"""
        # 测试初始化失败时的异常处理
        with patch.object(self.builder, 'initialize_lightrag', side_effect=Exception("Init failed")):
            with self.assertRaises(Exception):
                await self.builder._build_graph_async()


if __name__ == "__main__":
    unittest.main()
