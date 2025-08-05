"""
Improved LightRAG builders following Interface Segregation Principle

These builders implement only the interfaces they need, demonstrating
proper ISP compliance through composition and focused responsibilities.

Based on LightRAG framework for knowledge graph construction with automatic
entity and relation extraction from documents, generating GraphML format files.
"""

import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TypeVar

from ..config import settings
from ..entities import Entity
from ..graph import KnowledgeGraph
from ..relations import Relation
from ..types import EntityType, RelationType
from .interfaces import (
    BasicGraphBuilder,
    BatchGraphBuilder,
    FullFeaturedGraphBuilder,
    GraphExporter,
    StreamingGraphBuilder,
    UpdatableGraphBuilder,
)
from .mixins import (
    GraphExporterMixin,
    GraphMergerMixin,
    GraphStatisticsMixin,
    GraphValidatorMixin,
    IncrementalBuilderMixin,
)

logger = logging.getLogger(__name__)

# 定义类型变量
T = TypeVar("T")


class LightRAGCore:
    """LightRAG核心功能类 - 单一职责：管理LightRAG实例和GraphML解析"""

    def __init__(self, working_dir: str = "lightrag_storage"):
        self.working_dir = Path(working_dir)
        self.rag_instance: Optional[Any] = None
        self._initialized: bool = False

    def __del__(self) -> None:
        """确保资源被正确清理"""
        if self._initialized and self.rag_instance:
            try:
                self.cleanup()
            except Exception:
                pass

    async def initialize_lightrag(self) -> Any:
        """初始化LightRAG实例"""
        if self._initialized and self.rag_instance is not None:
            return self.rag_instance

        try:
            # pylint: disable=import-outside-toplevel
            # 延迟导入LightRAG以避免依赖问题
            from lightrag import LightRAG
            from lightrag.kg.shared_storage import initialize_pipeline_status
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc

            # 确保工作目录存在
            self.working_dir.mkdir(parents=True, exist_ok=True)

            # 创建自定义LLM函数
            async def custom_llm_complete(
                prompt: str,
                system_prompt: Optional[str] = None,
                history_messages: Optional[List[Any]] = None,
                **kwargs: Any,
            ) -> str:
                result = await openai_complete_if_cache(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages or [],
                    model=settings.LLM_MODEL,
                    base_url=settings.OPENAI_API_BASE,
                    api_key=settings.OPENAI_API_KEY,
                    **kwargs,
                )
                return str(result)

            # 创建自定义嵌入函数
            async def custom_embed(texts: List[str]) -> List[List[float]]:
                result = await openai_embed(
                    texts,
                    model=settings.EMBEDDING_MODEL,
                    base_url=settings.OPENAI_API_BASE,
                    api_key=settings.OPENAI_API_KEY,
                )
                return list(result)

            # 初始化LightRAG实例
            self.rag_instance = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=custom_llm_complete,
                embedding_func=EmbeddingFunc(
                    embedding_dim=settings.EMBEDDING_DIM,
                    max_token_size=settings.EMBEDDING_MAX_TOKEN_SIZE,
                    func=custom_embed,
                ),
            )

            # 重要：两个初始化调用都是必需的！
            if self.rag_instance:
                await self.rag_instance.initialize_storages()  # 初始化存储后端
            await initialize_pipeline_status()  # 初始化处理管道

            self._initialized = True
            logger.info("LightRAG initialized with working directory: %s", self.working_dir)
            return self.rag_instance

        except ImportError:
            logger.error("LightRAG not installed. Please install with: pip install lightrag")
            self.rag_instance = None
            raise
        except Exception as e:
            logger.error("Failed to initialize LightRAG: %s", e)
            self.rag_instance = None
            self._initialized = False
            raise

    async def build_graph_async(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "lightrag_graph",
    ) -> KnowledgeGraph:
        """异步构建知识图谱的核心实现"""
        return await self.abuild_graph(texts, database_schema, graph_name)

    async def abuild_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "lightrag_graph",
    ) -> KnowledgeGraph:
        """
        异步构建知识图谱的内部实现
        """
        # 初始化RAG实例
        await self.initialize_lightrag()

        # 确保实例已初始化
        if self.rag_instance is None:
            raise RuntimeError("Failed to initialize LightRAG instance")

        try:
            logger.info("Building knowledge graph with LightRAG: %s", graph_name)

            # 处理文本输入
            if texts:
                for i, text in enumerate(texts):
                    logger.info("Inserting text document %d/%d", i + 1, len(texts))
                    # 使用异步版本插入文档
                    await self.rag_instance.ainsert(text)

            # 等待LightRAG处理完成并生成GraphML文件
            graphml_file = self.working_dir / "graph_chunk_entity_relation.graphml"

            if graphml_file.exists():
                # 从GraphML文件加载知识图谱
                graph = self._load_graph_from_graphml(str(graphml_file), graph_name)
                logger.info(
                    "Successfully built graph: %d entities, %d relations",
                    len(graph.entities),
                    len(graph.relations),
                )
                return graph

            logger.warning("GraphML file not generated by LightRAG, creating empty graph")
            return KnowledgeGraph(name=graph_name)

        except Exception as e:
            logger.error("Error building graph with LightRAG: %s", e)
            raise

    async def update_graph_with_texts_async(
        self,
        texts: List[str],
        graph_name: Optional[str] = None,
    ) -> KnowledgeGraph:
        """使用新文本更新图谱的核心实现"""
        return await self.aadd_documents(texts, graph_name)

    async def aadd_documents(self, documents: List[str], graph_name: Optional[str] = None) -> KnowledgeGraph:
        """
        异步添加文档的内部实现
        """
        # 初始化RAG实例
        await self.initialize_lightrag()

        # 确保实例已初始化
        if self.rag_instance is None:
            raise RuntimeError("Failed to initialize LightRAG instance")

        try:
            logger.info("Adding %d documents to knowledge graph", len(documents))

            for i, doc in enumerate(documents):
                logger.info("Adding document %d/%d", i + 1, len(documents))
                await self.rag_instance.ainsert(doc)

            # 重新加载图谱
            graphml_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            if graphml_file.exists():
                graph = self._load_graph_from_graphml(str(graphml_file), graph_name or "lightrag_graph")
                logger.info(
                    "Updated graph: %d entities, %d relations",
                    len(graph.entities),
                    len(graph.relations),
                )
                return graph
            logger.warning("GraphML file not found after adding documents")
            return KnowledgeGraph(name=graph_name or "lightrag_graph")

        except Exception as e:
            logger.error("Error adding documents: %s", e)
            raise

    async def asearch_graph(
        self, query: str, search_type: Literal["naive", "local", "global", "hybrid"] = "hybrid"
    ) -> Dict[str, Any]:
        """
        异步搜索图谱的内部实现
        """
        # 初始化RAG实例
        await self.initialize_lightrag()

        # 确保实例已初始化
        if self.rag_instance is None:
            raise RuntimeError("Failed to initialize LightRAG instance")

        try:
            logger.info("Searching graph with query: %s, type: %s", query, search_type)

            # 根据搜索类型调用相应的查询方法
            from lightrag import QueryParam  # pylint: disable=import-outside-toplevel

            param = QueryParam(mode=search_type)
            result = await self.rag_instance.aquery(query, param=param)

            return {
                "query": query,
                "search_type": search_type,
                "result": result,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error("Error searching graph: %s", e)
            raise

    def cleanup(self) -> None:
        """
        清理LightRAG资源 - 使用同步方式避免事件循环问题
        """
        if self.rag_instance:
            try:
                # 尝试同步清理，避免创建新的事件循环
                # 直接设置为None，让Python垃圾回收处理
                logger.info("LightRAG resources cleaned up (sync mode)")
                self.rag_instance = None
                self._initialized = False
            except Exception as e:
                logger.error("Error during cleanup: %s", e)
                self.rag_instance = None
                self._initialized = False

    def _load_graph_from_graphml(self, graphml_file: str, graph_name: str) -> KnowledgeGraph:
        """
        从GraphML文件加载知识图谱

        Args:
            graphml_file: GraphML文件路径
            graph_name: 图谱名称

        Returns:
            KnowledgeGraph: 加载的知识图谱
        """
        try:
            # 解析GraphML文件
            tree = ET.parse(graphml_file)
            root = tree.getroot()

            # GraphML命名空间
            ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

            # 创建知识图谱
            graph = KnowledgeGraph(name=graph_name)

            # 解析节点（实体）
            entities_map: Dict[str, Entity] = {}
            for node in root.findall(".//graphml:node", ns):
                node_id = node.get("id")
                if node_id is None:
                    continue
                entity = self._parse_graphml_node(node, ns)
                if entity:
                    entities_map[node_id] = entity
                    graph.add_entity(entity)

            # 解析边（关系）
            for edge in root.findall(".//graphml:edge", ns):
                relation = self._parse_graphml_edge(edge, ns, entities_map)
                if relation:
                    graph.add_relation(relation)

            logger.info(
                "Loaded graph from GraphML: %d entities, %d relations",
                len(graph.entities),
                len(graph.relations),
            )
            return graph

        except Exception as e:
            logger.error("Error loading graph from GraphML file %s: %s", graphml_file, e)
            raise

    def _parse_graphml_node(self, node: Any, ns: Dict[str, str]) -> Optional[Entity]:
        """
        解析GraphML节点为实体

        Args:
            node: XML节点
            ns: 命名空间

        Returns:
            Entity: 解析的实体
        """
        try:
            node_id = node.get("id")
            entity_data = {}

            # 提取节点属性
            for data in node.findall("graphml:data", ns):
                key = data.get("key")
                value = data.text
                if key and value:
                    # 根据LightRAG的GraphML格式映射属性
                    if key == "d0":  # entity_id
                        entity_data["entity_id"] = value
                    elif key == "d1":  # entity_type
                        entity_data["entity_type"] = value
                    elif key == "d2":  # description
                        entity_data["description"] = value
                    elif key == "d3":  # source_id
                        entity_data["source_id"] = value
                    elif key == "d4":  # file_path
                        entity_data["file_path"] = value
                    elif key == "d5":  # created_at
                        entity_data["created_at"] = value

            # 映射实体类型
            entity_type_str = entity_data.get("entity_type", "unknown").lower()
            entity_type = self._map_entity_type(entity_type_str)

            # 创建实体
            entity = Entity(
                id=node_id,
                name=entity_data.get("entity_id", node_id),
                entity_type=entity_type,
                description=entity_data.get("description", ""),
                source="lightrag",
                properties={
                    "source_id": entity_data.get("source_id", ""),
                    "file_path": entity_data.get("file_path", ""),
                    "created_at": entity_data.get("created_at", ""),
                },
            )

            return entity

        except Exception as e:
            logger.error("Error parsing GraphML node: %s", e)
            return None

    def _parse_graphml_edge(self, edge: Any, ns: Dict[str, str], entities_map: Dict[str, Entity]) -> Optional[Relation]:
        """
        解析GraphML边为关系

        Args:
            edge: XML边
            ns: 命名空间
            entities_map: 实体映射

        Returns:
            Relation: 解析的关系
        """
        try:
            source_id = edge.get("source")
            target_id = edge.get("target")

            if source_id not in entities_map or target_id not in entities_map:
                return None

            edge_data = {}
            # 提取边属性
            for data in edge.findall("graphml:data", ns):
                key = data.get("key")
                value = data.text
                if key and value:
                    if key == "d6":  # weight
                        edge_data["weight"] = float(value)
                    elif key == "d7":  # description
                        edge_data["description"] = value
                    elif key == "d8":  # keywords
                        edge_data["keywords"] = value
                    elif key == "d9":  # source_id
                        edge_data["source_id"] = value
                    elif key == "d10":  # file_path
                        edge_data["file_path"] = value
                    elif key == "d11":  # created_at
                        edge_data["created_at"] = value

            # 创建关系
            relation = Relation(
                head_entity=entities_map[source_id],
                tail_entity=entities_map[target_id],
                relation_type=RelationType.RELATED_TO,  # LightRAG通常不区分关系类型
                confidence=edge_data.get("weight", 1.0),
                source="lightrag",
                properties={
                    "description": edge_data.get("description", ""),
                    "keywords": edge_data.get("keywords", ""),
                    "source_id": edge_data.get("source_id", ""),
                    "file_path": edge_data.get("file_path", ""),
                    "created_at": edge_data.get("created_at", ""),
                },
            )

            return relation

        except Exception as e:
            logger.error("Error parsing GraphML edge: %s", e)
            return None

    def _map_entity_type(self, entity_type_str: str) -> EntityType:
        """
        映射实体类型字符串到EntityType枚举

        Args:
            entity_type_str: 实体类型字符串

        Returns:
            EntityType: 映射的实体类型
        """
        type_mapping = {
            "person": EntityType.PERSON,
            "organization": EntityType.ORGANIZATION,
            "location": EntityType.LOCATION,
            "concept": EntityType.CONCEPT,
            "document": EntityType.DOCUMENT,
            "keyword": EntityType.KEYWORD,
            "table": EntityType.TABLE,
            "column": EntityType.COLUMN,
            "database": EntityType.DATABASE,
        }

        return type_mapping.get(entity_type_str.lower(), EntityType.UNKNOWN)

    def export_to_graphml(self, graph: KnowledgeGraph, output_path: str) -> bool:
        """
        将知识图谱导出为GraphML格式

        Args:
            graph: 知识图谱
            output_path: 输出文件路径

        Returns:
            bool: 导出是否成功
        """
        try:
            from xml.dom import minidom  # pylint: disable=import-outside-toplevel

            # 创建GraphML根元素
            root = ET.Element("graphml")
            root.set("xmlns", "http://graphml.graphdrawing.org/xmlns")
            root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
            root.set(
                "xsi:schemaLocation",
                "http://graphml.graphdrawing.org/xmlns http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd",
            )

            # 定义属性键
            keys = [
                ("d0", "node", "entity_id", "string"),
                ("d1", "node", "entity_type", "string"),
                ("d2", "node", "description", "string"),
                ("d3", "node", "source_id", "string"),
                ("d4", "node", "file_path", "string"),
                ("d5", "node", "created_at", "long"),
                ("d6", "edge", "weight", "double"),
                ("d7", "edge", "description", "string"),
                ("d8", "edge", "keywords", "string"),
                ("d9", "edge", "source_id", "string"),
                ("d10", "edge", "file_path", "string"),
                ("d11", "edge", "created_at", "long"),
            ]

            for key_id, for_type, attr_name, attr_type in keys:
                key_elem = ET.SubElement(root, "key")
                key_elem.set("id", key_id)
                key_elem.set("for", for_type)
                key_elem.set("attr.name", attr_name)
                key_elem.set("attr.type", attr_type)

            # 创建图元素
            graph_elem = ET.SubElement(root, "graph")
            graph_elem.set("edgedefault", "undirected")

            # 添加节点
            for entity in graph.entities.values():
                node_elem = ET.SubElement(graph_elem, "node")
                node_elem.set("id", entity.id)

                # 添加节点属性
                data_attrs = [
                    ("d0", entity.name),
                    ("d1", entity.entity_type.value),
                    ("d2", entity.description),
                    ("d3", entity.properties.get("source_id", "")),
                    ("d4", entity.properties.get("file_path", "")),
                    ("d5", str(int(entity.created_at.timestamp()))),
                ]

                for key, value in data_attrs:
                    if value:
                        data_elem = ET.SubElement(node_elem, "data")
                        data_elem.set("key", key)
                        data_elem.text = str(value)

            # 添加边
            for relation in graph.relations.values():
                if relation.head_entity and relation.tail_entity:
                    edge_elem = ET.SubElement(graph_elem, "edge")
                    edge_elem.set("source", relation.head_entity.id)
                    edge_elem.set("target", relation.tail_entity.id)

                    # 添加边属性
                    data_attrs = [
                        ("d6", str(relation.confidence)),
                        ("d7", relation.properties.get("description", "")),
                        ("d8", relation.properties.get("keywords", "")),
                        ("d9", relation.properties.get("source_id", "")),
                        ("d10", relation.properties.get("file_path", "")),
                        ("d11", str(int(relation.created_at.timestamp()))),
                    ]

                    for key, value in data_attrs:
                        if value:
                            data_elem = ET.SubElement(edge_elem, "data")
                            data_elem.set("key", key)
                            data_elem.text = str(value)

            # 写入文件
            rough_string = ET.tostring(root, "unicode")
            reparsed = minidom.parseString(rough_string)

            with open(output_path, "w", encoding="utf-8") as f:
                reparsed.writexml(f, indent="  ", addindent="  ", newl="\n", encoding="utf-8")

            logger.info("Graph exported to GraphML: %s", output_path)
            return True

        except Exception as e:
            logger.error("Error exporting graph to GraphML: %s", e)
            return False

    def get_basic_statistics(self) -> Dict[str, Any]:
        """
        获取图谱统计信息

        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            graphml_file = self.working_dir / "graph_chunk_entity_relation.graphml"
            if not graphml_file.exists():
                return {"entities_count": 0, "relations_count": 0, "status": "no_graph"}

            tree = ET.parse(str(graphml_file))
            root = tree.getroot()

            ns = {"graphml": "http://graphml.graphdrawing.org/xmlns"}

            entities_count = len(root.findall(".//graphml:node", ns))
            relations_count = len(root.findall(".//graphml:edge", ns))

            return {
                "entities_count": entities_count,
                "relations_count": relations_count,
                "graphml_file": str(graphml_file),
                "last_modified": datetime.fromtimestamp(graphml_file.stat().st_mtime).isoformat(),
                "status": "ready",
            }

        except Exception as e:
            logger.error("Error getting graph statistics: %s", e)
            return {"entities_count": 0, "relations_count": 0, "status": "error", "error": str(e)}


# ============================================================================
# ISP-compliant LightRAG Builder Implementations
# ============================================================================


class MinimalLightRAGBuilder(BasicGraphBuilder):
    """
    最小化的LightRAG图构建器 - 只实现核心构建功能

    适用于只需要基本图构建而不需要更新、合并或其他高级功能的客户端。
    """

    def __init__(self, working_dir: str = "minimal_lightrag_storage"):
        """初始化最小化LightRAG图构建器"""
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "minimal_lightrag_graph",
    ) -> KnowledgeGraph:
        """构建知识图谱 - 异步版本"""
        return await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


class FlexibleLightRAGBuilder(UpdatableGraphBuilder):
    """
    灵活的LightRAG图构建器 - 支持构建和更新

    适用于需要更新图谱但不需要合并、验证等高级功能的客户端。
    """

    def __init__(self, working_dir: str = "flexible_lightrag_storage"):
        """初始化灵活LightRAG图构建器"""
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "flexible_lightrag_graph",
    ) -> KnowledgeGraph:
        """异步构建知识图谱"""
        return await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)

    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """更新现有图谱"""
        logger.warning("LightRAG does not support direct entity/relation updates")
        logger.info("To update the graph, please add new documents using update_graph_with_texts")
        return graph

    async def update_graph_with_texts(
        self,
        texts: List[str],
        graph_name: Optional[str] = None,
    ) -> KnowledgeGraph:
        """使用新文本更新图谱"""
        return await self.lightrag_core.update_graph_with_texts_async(texts, graph_name)

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


class LightRAGBuilder(
    GraphMergerMixin,
    GraphValidatorMixin,
    GraphExporterMixin,
    GraphStatisticsMixin,
    FullFeaturedGraphBuilder,
):
    """
    全功能LightRAG图构建器 - 包含所有功能

    只有需要所有功能的客户端才应该使用这个类。
    大多数客户端应该使用更专注的接口。
    """

    def __init__(self, working_dir: str = "comprehensive_lightrag_storage"):
        """初始化全功能LightRAG图构建器"""
        super().__init__()
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "comprehensive_lightrag_graph",
    ) -> KnowledgeGraph:
        """构建全功能图谱"""
        # 委托给核心构建器
        graph = await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)

        # 执行验证
        validation_result = await self.validate_graph(graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return graph

    async def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """更新图谱并验证"""
        logger.warning("LightRAG does not support direct entity/relation updates")
        logger.info("To update the graph, please add new documents using update_graph_with_texts")

        # 执行验证
        validation_result = await self.validate_graph(graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return graph

    async def update_graph_with_texts(
        self,
        texts: List[str],
        graph_name: Optional[str] = None,
    ) -> KnowledgeGraph:
        """使用新文本更新图谱并验证"""
        graph = await self.lightrag_core.update_graph_with_texts_async(texts, graph_name)

        # 执行验证
        validation_result = await self.validate_graph(graph)
        if not validation_result.get("valid", True):
            logger.warning(f"Graph validation issues: {validation_result.get('issues', [])}")

        return graph

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.lightrag_core.get_basic_statistics()

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


class StreamingLightRAGBuilder(StreamingGraphBuilder, IncrementalBuilderMixin):
    """
    流式LightRAG图构建器 - 适用于实时增量更新

    专为需要实时处理文档流而不需要合并或验证功能的应用设计。
    """

    def __init__(self, working_dir: str = "streaming_lightrag_storage"):
        """初始化流式LightRAG图构建器"""
        super().__init__()
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "streaming_lightrag_graph",
    ) -> KnowledgeGraph:
        """构建初始图谱用于流式处理"""
        graph = await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)
        self._current_graph = graph
        return graph

    async def add_documents(self, documents: List[str], document_ids: Optional[List[str]] = None) -> KnowledgeGraph:
        """添加新文档到流式图谱"""
        if not documents:
            return self._current_graph or KnowledgeGraph()

        logger.info(f"Adding {len(documents)} documents to streaming LightRAG graph")

        # 直接使用LightRAG的增量更新功能
        graph = await self.lightrag_core.update_graph_with_texts_async(documents)
        self._current_graph = graph

        # 记录文档（简化版，基于时间戳）
        doc_timestamp = datetime.now().timestamp()
        for i, _ in enumerate(documents):
            doc_id = f"lightrag_doc_{doc_timestamp}_{i}"
            self._document_registry[doc_id] = [f"lightrag_entities_{doc_timestamp}_{i}"]

        return graph

    async def remove_documents(self, document_ids: List[str]) -> KnowledgeGraph:
        """移除文档（LightRAG不直接支持，返回当前图谱）"""
        logger.warning("LightRAG does not support document removal")
        logger.info("Consider rebuilding the graph from scratch if document removal is required")

        # 清理注册表
        for doc_id in document_ids:
            if doc_id in self._document_registry:
                del self._document_registry[doc_id]

        return self._current_graph or KnowledgeGraph()

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


class BatchLightRAGBuilder(GraphMergerMixin, BatchGraphBuilder):
    """
    批量LightRAG图构建器 - 优化批量处理多个数据源

    适用于需要处理多个数据源并合并它们，但不需要增量更新或验证的场景。
    """

    def __init__(self, working_dir: str = "batch_lightrag_storage"):
        """初始化批量LightRAG图构建器"""
        super().__init__()
        self.lightrag_core = LightRAGCore(working_dir)

    async def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "batch_lightrag_graph",
    ) -> KnowledgeGraph:
        """构建批量图谱"""
        return await self.lightrag_core.build_graph_async(texts, database_schema, graph_name)

    async def build_from_multiple_sources(
        self, sources: List[Dict[str, Any]], graph_name: str = "multi_source_lightrag_graph"
    ) -> KnowledgeGraph:
        """从多个异构数据源构建图谱"""
        # LightRAG处理多个数据源的策略：顺序处理并合并到同一个工作目录
        all_texts = []

        for source in sources:
            source_type = source.get("type")
            source_data = source.get("data")

            if source_type == "text":
                texts = source_data if isinstance(source_data, list) else [source_data]
                all_texts.extend(texts)
            elif source_type == "mixed":
                if source_data is not None:
                    texts = source_data.get("texts", [])
                    all_texts.extend(texts)
            else:
                logger.warning(f"Unknown source type: {source_type}")

        if not all_texts:
            return KnowledgeGraph(name=graph_name)

        # 使用单个LightRAG实例处理所有文本
        logger.info(f"Building LightRAG graph from {len(all_texts)} texts across {len(sources)} sources")
        graph = await self.lightrag_core.build_graph_async(all_texts, None, graph_name)

        return graph

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


class LightRAGSearchBuilder(GraphExporter):
    """
    LightRAG搜索构建器 - 专门用于搜索和导出功能

    遵循ISP：只实现搜索和导出相关的接口，不包含构建功能。
    """

    def __init__(self, working_dir: str = "search_lightrag_storage"):
        """初始化搜索构建器"""
        self.lightrag_core = LightRAGCore(working_dir)

    async def search_graph(
        self, query: str, search_type: Literal["naive", "local", "global", "hybrid"] = "hybrid"
    ) -> Dict[str, Any]:
        """搜索图谱"""
        return await self.lightrag_core.asearch_graph(query, search_type)

    async def export_to_format(self, graph: KnowledgeGraph, format: str) -> Dict[str, Any]:
        """导出图谱到指定格式"""
        format_lower = format.lower()

        if format_lower == "graphml":
            # LightRAG原生支持GraphML
            success = self.lightrag_core.export_to_graphml(graph, f"{graph.name}.graphml")
            return {"success": success, "format": "graphml", "message": "Exported to GraphML"}
        elif format_lower == "json":
            return graph.to_dict()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.lightrag_core.get_basic_statistics()

    def cleanup(self) -> None:
        """清理资源"""
        self.lightrag_core.cleanup()


# 导出所有公共类
__all__ = [
    "LightRAGCore",
    "MinimalLightRAGBuilder",
    "FlexibleLightRAGBuilder",
    "LightRAGBuilder",
    "StreamingLightRAGBuilder",
    "BatchLightRAGBuilder",
    "LightRAGSearchBuilder",
]
