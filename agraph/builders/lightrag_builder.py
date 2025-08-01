"""
LightRAG知识图谱构建器

基于LightRAG框架构建知识图谱，支持从文档中自动抽取实体和关系，
并生成GraphML格式的知识图谱文件。
"""

import asyncio
import concurrent.futures
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Literal, Optional, TypeVar

from ..config import Settings
from ..entities import Entity
from ..graph import KnowledgeGraph
from ..relations import Relation
from ..types import EntityType, RelationType
from .graph_builder import BaseKnowledgeGraphBuilder

logger = logging.getLogger(__name__)

# 定义类型变量
T = TypeVar("T")


class LightRAGGraphBuilder(BaseKnowledgeGraphBuilder):
    """LightRAG知识图谱构建器"""

    def __init__(self, working_dir: str = "lightrag_storage"):
        super().__init__()
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

    def _run_async(self, coro: Coroutine[Any, Any, T]) -> T:
        """运行异步协程的辅助方法，处理事件循环"""
        try:
            # 尝试获取当前运行的事件循环
            loop = asyncio.get_running_loop()
            # 如果已经在事件循环中，创建任务在当前循环中运行
            task = loop.create_task(coro)
            # 使用简单的方式等待任务完成

            while not task.done():
                time.sleep(0.01)
            return task.result()
        except RuntimeError:
            # 没有正在运行的事件循环，使用asyncio.run
            pass

        # 使用独立的线程运行协程，每次都创建全新的环境
        def run_in_separate_thread() -> T:
            # 确保线程中没有事件循环
            asyncio.set_event_loop(None)
            # 创建包装协程来处理清理

            async def wrapped_coro() -> T:
                try:
                    return await coro
                finally:
                    # 清理残留任务
                    await asyncio.sleep(0.05)  # 给其他任务时间完成

            # 使用asyncio.run，它会自动处理事件循环的创建和清理
            try:
                return asyncio.run(wrapped_coro())
            except Exception as e:
                # 如果asyncio.run失败，回退到手动管理
                logger.error("error %s", e)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(wrapped_coro())
                finally:
                    try:
                        # 强制关闭所有剩余任务
                        pending = asyncio.all_tasks(loop)
                        if pending:
                            for task in pending:
                                task.cancel()
                            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                    except Exception as e:
                        logger.error(e)
                    finally:
                        loop.close()
                        asyncio.set_event_loop(None)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_in_separate_thread)
            return future.result()

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
                    model=Settings.LLM_MODEL,
                    base_url=Settings.OPENAI_API_BASE,
                    api_key=Settings.OPENAI_API_KEY,
                    **kwargs,
                )
                return str(result)

            # 创建自定义嵌入函数
            async def custom_embed(texts: List[str]) -> List[List[float]]:
                result = await openai_embed(
                    texts,
                    model=Settings.EMBEDDING_MODEL,
                    base_url=Settings.OPENAI_API_BASE,
                    api_key=Settings.OPENAI_API_KEY,
                )
                return list(result)

            # 初始化LightRAG实例
            self.rag_instance = LightRAG(
                working_dir=str(self.working_dir),
                llm_model_func=custom_llm_complete,
                embedding_func=EmbeddingFunc(
                    embedding_dim=Settings.EMBEDDING_DIM,
                    max_token_size=Settings.EMBEDDING_MAX_TOKEN_SIZE,
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

    def build_graph(
        self,
        texts: Optional[List[str]] = None,
        database_schema: Optional[Dict[str, Any]] = None,
        graph_name: str = "lightrag_graph",
    ) -> KnowledgeGraph:
        """
        使用LightRAG构建知识图谱

        Args:
            texts: 文本列表
            database_schema: 数据库模式（LightRAG不直接支持）
            graph_name: 图谱名称

        Returns:
            KnowledgeGraph: 构建的知识图谱
        """
        return self._run_async(self.abuild_graph(texts, database_schema, graph_name))

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

    def update_graph(
        self,
        graph: KnowledgeGraph,
        new_entities: Optional[List[Entity]] = None,
        new_relations: Optional[List[Relation]] = None,
    ) -> KnowledgeGraph:
        """
        更新知识图谱（LightRAG方式）

        Args:
            graph: 现有知识图谱
            new_entities: 新增实体（LightRAG不直接支持）
            new_relations: 新增关系（LightRAG不直接支持）

        Returns:
            KnowledgeGraph: 更新后的知识图谱
        """
        logger.warning("LightRAG does not support direct entity/relation updates")
        logger.info("To update the graph, please add new documents using build_graph with texts")
        return graph

    def add_documents(self, documents: List[str], graph_name: Optional[str] = None) -> KnowledgeGraph:
        """
        添加新文档到现有知识图谱

        Args:
            documents: 文档文本列表
            graph_name: 图谱名称

        Returns:
            KnowledgeGraph: 更新后的知识图谱
        """
        return self._run_async(self.aadd_documents(documents, graph_name))

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

    def search_graph(
        self, query: str, search_type: Literal["naive", "local", "global", "hybrid"] = "hybrid"
    ) -> Dict[str, Any]:
        """
        在知识图谱中搜索

        Args:
            query: 查询字符串
            search_type: 搜索类型 ("naive", "local", "global", "hybrid")

        Returns:
            Dict[str, Any]: 搜索结果
        """
        return self._run_async(self.asearch_graph(query, search_type))

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

    async def _cleanup_async(self) -> None:
        """
        异步清理LightRAG资源
        """
        if self.rag_instance:
            try:
                # 调用 LightRAG 的异步清理方法
                await self.rag_instance.finalize_storages()
                logger.info("LightRAG async cleanup completed")
            except Exception as e:
                logger.error("Error during async cleanup: %s", e)
            finally:
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

    def get_graph_statistics(self) -> Dict[str, Any]:
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
