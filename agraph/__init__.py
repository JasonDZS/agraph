"""
知识图谱模块

本模块提供完整的知识图谱构建、存储和查询功能，包括：
- 实体和关系的数据结构
- 从文本和数据库抽取实体和关系
- 知识图谱的构建和管理
- 多种存储后端支持 (JSON, Neo4j)
- 图嵌入和相似度计算
"""

from .builders import (
    BatchBuilder,
    ComprehensiveGraphBuilder,
    FlexibleGraphBuilder,
    MinimalGraphBuilder,
    StreamingBuilder,
)

# 嵌入
from .embeddings import GraphEmbedding
from .entities import Entity

# 抽取器
from .extractors import (
    BaseEntityExtractor,
    BaseRelationExtractor,
    DatabaseEntityExtractor,
    DatabaseRelationExtractor,
    TextEntityExtractor,
    TextRelationExtractor,
)
from .graph import KnowledgeGraph
from .relations import Relation

# 检索
from .retrieval import ChatKnowledgeRetriever, KnowledgeRetriever

# 服务
from .services import EntityMerger, GraphAnalyzer, GraphPathFinder, GraphValidator

# 存储
from .storage import GraphStorage, JsonStorage
from .storage import JsonVectorStorage as StorageJsonVectorStorage  # 区别于embeddings的版本
from .storage import Neo4jStorage
from .storage.interfaces import BasicGraphStorage, FullGraphStorage, QueryableGraphStorage, VectorStorage

# 核心数据结构
from .types import EntityType, RelationType

__version__ = "1.0.0"

__all__ = [
    # 核心类型和数据结构
    "EntityType",
    "RelationType",
    "Entity",
    "Relation",
    "KnowledgeGraph",
    # 抽取器
    "BaseEntityExtractor",
    "TextEntityExtractor",
    "DatabaseEntityExtractor",
    "BaseRelationExtractor",
    "TextRelationExtractor",
    "DatabaseRelationExtractor",
    # 图构建器
    "MinimalGraphBuilder",
    "FlexibleGraphBuilder",
    "StreamingBuilder",
    "BatchBuilder",
    "ComprehensiveGraphBuilder",
    # 存储
    "GraphStorage",
    "Neo4jStorage",
    "JsonStorage",
    "StorageJsonVectorStorage",
    # 嵌入
    "GraphEmbedding",
    # 检索
    "KnowledgeRetriever",
    "ChatKnowledgeRetriever",
    # 服务
    "GraphAnalyzer",
    "GraphPathFinder",
    "EntityMerger",
    "GraphValidator",
    # 存储接口
    "BasicGraphStorage",
    "QueryableGraphStorage",
    "FullGraphStorage",
    "VectorStorage",
]
