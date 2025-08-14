# 自定义向量数据库实现指南

本指南将教您如何创建自己的向量数据库实现，包括接口设计、核心功能实现和集成到 agraph 系统中。

## 目录

1. [架构概述](#架构概述)
2. [接口定义](#接口定义)
3. [实现步骤](#实现步骤)
4. [完整示例](#完整示例)
5. [注册和使用](#注册和使用)
6. [测试和验证](#测试和验证)
7. [性能优化](#性能优化)
8. [最佳实践](#最佳实践)

## 架构概述

agraph 的向量数据库系统采用模块化设计，支持多种存储后端：

```shell
VectorStore (接口)
├── VectorStoreCore (核心功能)
├── EntityStore (实体存储)
├── RelationStore (关系存储)
├── ClusterStore (集群存储)
└── TextChunkStore (文本块存储)
```

### 设计原则

1. **接口隔离** - 只实现需要的功能接口
2. **可插拔架构** - 轻松替换存储后端
3. **异步支持** - 所有操作都是异步的
4. **类型安全** - 完整的类型注解
5. **错误处理** - 统一的异常管理

## 接口定义

### 核心接口

```python
from agraph.vectordb.interfaces import VectorStore
from agraph.vectordb.exceptions import VectorStoreError

class CustomVectorStore(VectorStore):
    """自定义向量存储实现"""

    def __init__(self, collection_name: str = "knowledge_graph", **kwargs):
        super().__init__(collection_name, **kwargs)
        # 初始化自定义参数

    async def initialize(self) -> None:
        """初始化存储连接和配置"""
        # 实现初始化逻辑
        self._is_initialized = True

    async def close(self) -> None:
        """关闭存储连接"""
        # 实现清理逻辑
        self._is_initialized = False
```

### 必需方法

每个向量存储实现必须提供以下方法：

#### 实体操作

- `add_entity()` - 添加实体
- `update_entity()` - 更新实体
- `delete_entity()` - 删除实体
- `get_entity()` - 获取实体
- `search_entities()` - 搜索实体
- `batch_add_entities()` - 批量添加实体

#### 关系操作

- `add_relation()` - 添加关系
- `update_relation()` - 更新关系
- `delete_relation()` - 删除关系
- `get_relation()` - 获取关系
- `search_relations()` - 搜索关系
- `batch_add_relations()` - 批量添加关系

#### 工具方法

- `get_stats()` - 获取统计信息
- `clear_all()` - 清空所有数据

## 实现步骤

### 步骤1：设计数据模型

首先确定如何存储实体、关系和向量：

```python
class CustomVectorStore(VectorStore):
    def __init__(self, storage_path: str, **kwargs):
        super().__init__(**kwargs)
        self.storage_path = storage_path
        self._entities = {}  # 内存存储示例
        self._embeddings = {}  # 向量存储
```

### 步骤2：实现向量操作

实现向量计算和相似度搜索：

```python
import numpy as np

def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
    """计算余弦相似度"""
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)

    if np.linalg.norm(arr1) == 0 or np.linalg.norm(arr2) == 0:
        return 0.0

    similarity = np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2))
    return float(similarity)

def _generate_embedding(self, text: str) -> List[float]:
    """生成文本嵌入（简单实现）"""
    # 这里可以集成任何嵌入模型
    # 示例：简单字符频率向量
    char_counts = [0.0] * 128
    for char in text.lower():
        if ord(char) < 128:
            char_counts[ord(char)] += 1.0

    # 归一化
    total = sum(char_counts)
    if total > 0:
        char_counts = [c / total for c in char_counts]

    return char_counts
```

### 步骤3：实现核心方法

```python
async def add_entity(
    self, entity: Entity, embedding: Optional[List[float]] = None
) -> bool:
    """添加实体到存储"""
    try:
        if not self._is_initialized:
            raise VectorStoreError("Store not initialized")

        # 生成嵌入向量（如果未提供）
        if embedding is None:
            text = f"{entity.name} {entity.description}"
            embedding = self._generate_embedding(text)

        # 验证嵌入向量
        if not self._validate_embedding(embedding):
            raise VectorStoreError(f"Invalid embedding for entity {entity.id}")

        # 存储实体和向量
        self._entities[entity.id] = entity
        self._embeddings[entity.id] = embedding

        return True

    except Exception as e:
        raise VectorStoreError(f"Failed to add entity {entity.id}: {e}") from e

async def search_entities(
    self,
    query: Union[str, List[float]],
    top_k: int = 10,
    filter_dict: Optional[Dict[str, Any]] = None
) -> List[Tuple[Entity, float]]:
    """搜索相似实体"""
    try:
        # 生成查询向量
        if isinstance(query, str):
            query_vector = self._generate_embedding(query)
        else:
            query_vector = query

        results = []

        for entity_id, entity in self._entities.items():
            # 应用过滤条件
            if filter_dict and not self._apply_filters(entity, filter_dict):
                continue

            # 计算相似度
            if entity_id in self._embeddings:
                entity_vector = self._embeddings[entity_id]
                similarity = self._calculate_similarity(query_vector, entity_vector)
                results.append((entity, similarity))

        # 排序并返回前 top_k 个结果
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    except Exception as e:
        raise VectorStoreError(f"Failed to search entities: {e}") from e

def _apply_filters(self, entity: Entity, filter_dict: Dict[str, Any]) -> bool:
    """应用过滤条件"""
    for key, value in filter_dict.items():
        if hasattr(entity, key):
            if getattr(entity, key) != value:
                return False
    return True
```

## 完整示例

以下是一个基于 SQLite 的完整向量存储实现示例文件，您可以参考实现：

> 📄 **示例文件**: [sqlite_vectorstore.py](./examples/sqlite_vectorstore.py)

该示例包含：

- 完整的 SQLite 数据库表结构
- 向量序列化和反序列化
- 实体的完整 CRUD 操作
- 相似度计算和搜索
- 批量操作支持

## 注册和使用

### 注册自定义存储

```python
from agraph.vectordb import VectorStoreFactory, VectorStoreType

# 方式1：直接注册
from my_package import SQLiteVectorStore
VectorStoreFactory.register_store_class("sqlite", SQLiteVectorStore)

# 方式2：使用装饰器
@VectorStoreFactory.register("sqlite")
class SQLiteVectorStore(VectorStore):
    # 实现...
    pass

# 使用自定义存储
store = VectorStoreFactory.create_store(
    "sqlite",
    db_path="./my_vectors.db",
    embedding_dimension=256
)
```

### 创建便捷函数

```python
def create_sqlite_store(db_path: str = "vectorstore.db", **kwargs) -> SQLiteVectorStore:
    """便捷创建函数"""
    return SQLiteVectorStore(db_path=db_path, **kwargs)

# 导出到 __init__.py
__all__ = ["create_sqlite_store", "SQLiteVectorStore"]

# 使用
from my_vectordb import create_sqlite_store
store = create_sqlite_store("./knowledge_base.db")
```

## 测试和验证

### 单元测试框架

```python
import unittest
from agraph.base import Entity, EntityType

class TestCustomVectorStore(unittest.TestCase):
    async def setUp(self):
        self.store = CustomVectorStore()
        await self.store.initialize()

    async def test_add_and_get_entity(self):
        entity = Entity(
            name="Python",
            entity_type=EntityType.CONCEPT,
            description="编程语言"
        )

        # 测试添加
        result = await self.store.add_entity(entity)
        self.assertTrue(result)

        # 测试获取
        retrieved = await self.store.get_entity(entity.id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "Python")

    async def test_search_functionality(self):
        # 添加测试数据
        entities = [
            Entity(name="Python", entity_type=EntityType.CONCEPT),
            Entity(name="Java", entity_type=EntityType.CONCEPT),
            Entity(name="JavaScript", entity_type=EntityType.CONCEPT)
        ]

        for entity in entities:
            await self.store.add_entity(entity)

        # 搜索测试
        results = await self.store.search_entities("编程", top_k=3)
        self.assertGreater(len(results), 0)

        # 验证结果格式
        for entity, score in results:
            self.assertIsInstance(entity, Entity)
            self.assertIsInstance(score, float)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    async def tearDown(self):
        await self.store.close()
```

### 性能基准测试

```python
import time
import asyncio
from typing import List

async def benchmark_operations():
    """性能基准测试"""
    store = CustomVectorStore()
    await store.initialize()

    # 测试批量添加性能
    entities = [
        Entity(name=f"Entity_{i}", entity_type=EntityType.CONCEPT)
        for i in range(1000)
    ]

    start_time = time.time()
    results = await store.batch_add_entities(entities)
    add_time = time.time() - start_time

    print(f"批量添加 1000 个实体:")
    print(f"  耗时: {add_time:.2f} 秒")
    print(f"  成功率: {sum(results) / len(results) * 100:.1f}%")
    print(f"  平均速度: {len(entities) / add_time:.0f} 个/秒")

    # 测试搜索性能
    search_queries = ["test", "概念", "实体", "数据", "信息"]

    start_time = time.time()
    for query in search_queries:
        await store.search_entities(query, top_k=10)
    search_time = time.time() - start_time

    print(f"\n搜索性能 ({len(search_queries)} 次查询):")
    print(f"  总耗时: {search_time:.3f} 秒")
    print(f"  平均耗时: {search_time / len(search_queries) * 1000:.1f} 毫秒/查询")

    await store.close()

# 运行基准测试
asyncio.run(benchmark_operations())
```

## 性能优化

### 1. 向量索引优化

```python
# 使用 Faiss 进行高效向量搜索
import faiss
import numpy as np

class IndexedVectorStore(CustomVectorStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._index: Optional[faiss.Index] = None
        self._id_mapping: Dict[int, str] = {}
        self._embedding_dimension = 128

    async def _build_index(self):
        """构建向量索引"""
        if not self._embeddings:
            return

        vectors = list(self._embeddings.values())
        ids = list(self._embeddings.keys())

        # 创建 Faiss 索引
        self._index = faiss.IndexFlatIP(self._embedding_dimension)

        # 添加向量
        vector_array = np.array(vectors, dtype=np.float32)
        self._index.add(vector_array)

        # 建立 ID 映射
        self._id_mapping = {i: entity_id for i, entity_id in enumerate(ids)}

    async def search_entities(self, query, top_k=10, filter_dict=None):
        """使用索引进行快速搜索"""
        if self._index is None:
            await self._build_index()

        # 生成查询向量
        if isinstance(query, str):
            query_vector = self._generate_embedding(query)
        else:
            query_vector = query

        # 搜索
        query_array = np.array([query_vector], dtype=np.float32)
        scores, indices = self._index.search(query_array, top_k)

        results = []
        for i, score in zip(indices[0], scores[0]):
            if i != -1:  # 有效索引
                entity_id = self._id_mapping[i]
                entity = self._entities[entity_id]

                # 应用过滤
                if filter_dict and not self._apply_filters(entity, filter_dict):
                    continue

                results.append((entity, float(score)))

        return results
```

### 2. 批量操作优化

```python
async def optimized_batch_add_entities(
    self,
    entities: List[Entity],
    embeddings: Optional[List[List[float]]] = None
) -> List[bool]:
    """优化的批量添加"""
    try:
        # 准备批量数据
        batch_entities = {}
        batch_embeddings = {}

        for i, entity in enumerate(entities):
            embedding = embeddings[i] if embeddings else None
            if embedding is None:
                text = f"{entity.name} {entity.description}"
                embedding = self._generate_embedding(text)

            if self._validate_embedding(embedding):
                batch_entities[entity.id] = entity
                batch_embeddings[entity.id] = embedding

        # 批量存储（具体实现取决于后端）
        self._entities.update(batch_entities)
        self._embeddings.update(batch_embeddings)

        # 重建索引
        if hasattr(self, '_build_index'):
            await self._build_index()

        return [True] * len(entities)

    except Exception as e:
        raise VectorStoreError(f"Batch add failed: {e}") from e
```

### 3. 内存管理和缓存

```python
from collections import OrderedDict

class CachedVectorStore(CustomVectorStore):
    def __init__(self, cache_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.cache_size = cache_size
        self._entity_cache: OrderedDict[str, Entity] = OrderedDict()

    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """带缓存的实体获取"""
        # 检查缓存
        if entity_id in self._entity_cache:
            # 移动到末尾（LRU）
            self._entity_cache.move_to_end(entity_id)
            return self._entity_cache[entity_id]

        # 从存储加载
        entity = await self._load_entity_from_storage(entity_id)

        if entity:
            # 添加到缓存
            self._add_to_cache(entity_id, entity)

        return entity

    def _add_to_cache(self, entity_id: str, entity: Entity):
        """添加到缓存，管理缓存大小"""
        if len(self._entity_cache) >= self.cache_size:
            # 移除最旧的条目
            self._entity_cache.popitem(last=False)

        self._entity_cache[entity_id] = entity

    async def _load_entity_from_storage(self, entity_id: str) -> Optional[Entity]:
        """从存储后端加载实体"""
        # 具体实现取决于存储后端
        return self._entities.get(entity_id)
```

## 最佳实践

### 1. 错误处理和恢复

```python
from agraph.vectordb.exceptions import (
    VectorStoreError,
    VectorStoreNotInitializedError,
    VectorStoreOperationError
)

async def robust_add_entity(self, entity: Entity,
                          embedding: Optional[List[float]] = None,
                          max_retries: int = 3) -> bool:
    """带重试的健壮添加方法"""
    if not self._is_initialized:
        raise VectorStoreNotInitializedError("Store not initialized")

    for attempt in range(max_retries):
        try:
            # 验证输入
            if not entity or not entity.id:
                raise VectorStoreOperationError("Invalid entity: missing ID")

            # 生成嵌入
            if embedding is None:
                if not entity.name and not entity.description:
                    raise VectorStoreOperationError(
                        "Cannot generate embedding: missing name and description"
                    )

                text = f"{entity.name} {entity.description}"
                embedding = self._generate_embedding(text)

            # 验证嵌入
            if not self._validate_embedding(embedding):
                raise VectorStoreOperationError(f"Invalid embedding for entity {entity.id}")

            # 执行存储操作
            result = await self._store_entity(entity, embedding)

            if result:
                logger.info(f"Successfully added entity {entity.id}")
                return True

        except VectorStoreError:
            # 重新抛出已知错误（不重试）
            raise
        except Exception as e:
            logger.warning(
                f"Attempt {attempt + 1} failed for entity {entity.id}: {e}"
            )

            if attempt == max_retries - 1:
                # 最后一次尝试失败
                raise VectorStoreOperationError(
                    f"Failed to add entity {entity.id} after {max_retries} attempts: {e}"
                ) from e

            # 等待后重试
            await asyncio.sleep(0.1 * (2 ** attempt))  # 指数退避

    return False
```

### 2. 监控和指标

```python
import time
from dataclasses import dataclass, field
from typing import Dict, Any

@dataclass
class VectorStoreMetrics:
    """向量存储性能指标"""
    operations_count: int = 0
    errors_count: int = 0
    total_search_time: float = 0.0
    total_add_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    @property
    def error_rate(self) -> float:
        return self.errors_count / max(1, self.operations_count)

    @property
    def avg_search_time(self) -> float:
        return self.total_search_time / max(1, self.operations_count)

class MonitoredVectorStore(CustomVectorStore):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metrics = VectorStoreMetrics()

    async def add_entity(self, entity: Entity, embedding: Optional[List[float]] = None) -> bool:
        """带监控的添加方法"""
        start_time = time.time()

        try:
            result = await super().add_entity(entity, embedding)

            # 记录成功指标
            self.metrics.operations_count += 1
            self.metrics.total_add_time += time.time() - start_time

            return result

        except Exception as e:
            # 记录错误指标
            self.metrics.errors_count += 1
            raise

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            "operations": self.metrics.operations_count,
            "errors": self.metrics.errors_count,
            "error_rate": f"{self.metrics.error_rate:.2%}",
            "avg_search_time": f"{self.metrics.avg_search_time:.3f}s",
            "cache_hit_rate": f"{self.metrics.cache_hits / max(1, self.metrics.cache_hits + self.metrics.cache_misses):.2%}"
        }
```

### 3. 配置和环境管理

```python
from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class VectorStoreConfig:
    """向量存储配置"""
    storage_path: str = "vectorstore.db"
    embedding_dimension: int = 128
    cache_size: int = 1000
    batch_size: int = 100
    max_retries: int = 3
    enable_monitoring: bool = True

    @classmethod
    def from_env(cls) -> "VectorStoreConfig":
        """从环境变量创建配置"""
        return cls(
            storage_path=os.getenv("VECTOR_STORE_PATH", "vectorstore.db"),
            embedding_dimension=int(os.getenv("EMBEDDING_DIM", "128")),
            cache_size=int(os.getenv("CACHE_SIZE", "1000")),
            batch_size=int(os.getenv("BATCH_SIZE", "100")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            enable_monitoring=os.getenv("ENABLE_MONITORING", "true").lower() == "true"
        )

# 使用配置
config = VectorStoreConfig.from_env()
store = CustomVectorStore(
    storage_path=config.storage_path,
    embedding_dimension=config.embedding_dimension,
    cache_size=config.cache_size
)
```

## 总结

创建自定义向量数据库实现的关键步骤：

1. **🏗️ 架构设计** - 基于 agraph 的接口系统设计
2. **⚡ 核心实现** - 实现向量存储、搜索和管理功能
3. **🔧 性能优化** - 索引、缓存和批量操作优化
4. **🧪 测试验证** - 完善的单元测试和性能测试
5. **📊 监控运维** - 指标收集、错误处理和日志管理
6. **🚀 集成部署** - 注册到工厂系统，便于使用

通过遵循这些指南和最佳实践，您可以创建高效、可靠的自定义向量数据库实现，满足特定的业务需求。

## 相关链接

- [向量数据库使用教程](./vectordb_tutorial.md)
- [API 参考文档](../source/agraph.vectordb.rst)
- [完整示例代码](./examples/)
