# 向量数据库使用教程

本教程将指导您如何使用 agraph 的向量数据库功能来存储和检索知识图谱数据。

## 目录

1. [快速开始](#快速开始)
2. [配置向量数据库](#配置向量数据库)
3. [基本操作](#基本操作)
4. [搜索和检索](#搜索和检索)
5. [批量操作](#批量操作)
6. [高级功能](#高级功能)
7. [性能优化](#性能优化)
8. [故障排除](#故障排除)

## 快速开始

### 创建内存向量存储

最简单的方式是使用内存向量存储进行测试和小规模应用：

```python
import asyncio
from agraph.vectordb import create_memory_store
from agraph.base import Entity, Relation, EntityType, RelationType

# 创建内存向量存储
async def quick_start():
    store = create_memory_store()
    await store.initialize()

    # 创建实体
    entity = Entity(
        name="Python",
        entity_type=EntityType.CONCEPT,
        description="一种高级编程语言"
    )

    # 添加到向量存储
    success = await store.add_entity(entity)
    print(f"实体添加成功: {success}")

    # 搜索相似实体
    results = await store.search_entities("编程语言", top_k=5)
    for entity, score in results:
        print(f"实体: {entity.name}, 相似度: {score:.3f}")

    await store.close()

# 运行示例
asyncio.run(quick_start())
```

### 使用 ChromaDB 持久化存储

对于生产环境，推荐使用 ChromaDB：

```python
import asyncio
from agraph.base import Entity, Relation, EntityType, RelationType
from agraph.vectordb import create_chroma_store

async def chroma_example():
    # 创建持久化 ChromaDB 存储
    store = create_chroma_store(
        collection_name="my_knowledge_graph",
        persist_directory="./vector_data",
        use_openai_embeddings=True
    )

    await store.initialize()

    # 创建实体
    entity = Entity(
        name="Python",
        entity_type=EntityType.CONCEPT,
        description="一种高级编程语言"
    )

    # 添加到向量存储
    success = await store.add_entity(entity)
    print(f"实体添加成功: {success}")

    # 搜索相似实体
    results = await store.search_entities("编程语言", top_k=5)
    for entity, score in results:
        print(f"实体: {entity.name}, 相似度: {score:.3f}")

    await store.close()

asyncio.run(chroma_example())
```

## 配置向量数据库

### 使用工厂模式

```python
from agraph.vectordb import VectorStoreFactory, VectorStoreType

# 方式1: 使用字符串类型
store = VectorStoreFactory.create_store("memory", collection_name="test")

# 方式2: 使用枚举类型
store = VectorStoreFactory.create_store(
    VectorStoreType.CHROMA,
    persist_directory="./data",
    use_openai_embeddings=True
)
```

### 使用配置类

```python
from agraph.vectordb.config import ConfigBuilder

# 创建嵌入配置
embedding_config = ConfigBuilder.embedding_config(
    model="text-embedding-3-small",
    api_key="your-openai-api-key",
    use_cache=True,
    cache_size=1000
)

# 创建 ChromaDB 配置
chroma_config = ConfigBuilder.chroma_store_config(
    collection_name="knowledge_graph",
    persist_directory="./vector_storage",
    use_openai_embeddings=True,
    embedding_config=embedding_config
)

# 使用配置创建存储
config_dict = chroma_config.to_dict()
store = VectorStoreFactory.create_store(VectorStoreType.CHROMA, config_dict)
```

## 基本操作

### 实体操作

```python
from agraph.base import Entity, EntityType
from datetime import datetime

async def entity_operations(store):
    # 创建实体
    entity = Entity(
        id="python_lang",
        name="Python",
        entity_type=EntityType.CONCEPT,
        description="一种解释型、高级编程语言",
        properties={"version": "3.11", "paradigm": "multi-paradigm"},
        aliases=["Python语言", "蟒蛇语言"],
        confidence=0.95
    )

    # 添加实体
    await store.add_entity(entity)

    # 获取实体
    retrieved = await store.get_entity("python_lang")
    print(f"获取的实体: {retrieved.name}")

    # 更新实体
    entity.description = "更新后的描述"
    await store.update_entity(entity)

    # 删除实体
    await store.delete_entity("python_lang")
```

### 关系操作

```python
from agraph.base import Relation, RelationType

async def relation_operations(store):
    # 创建关系
    relation = Relation(
        id="python_used_for_ai",
        head_entity=python_entity,
        tail_entity=ai_entity,
        relation_type=RelationType.USED_FOR,
        description="Python常用于人工智能开发",
        confidence=0.9
    )

    # 添加关系
    await store.add_relation(relation)

    # 搜索关系
    results = await store.search_relations("用于开发", top_k=10)
    for rel, score in results:
        print(f"关系: {rel.relation_type}, 相似度: {score}")
```

### 文本块操作

```python
from agraph.base import TextChunk

async def text_chunk_operations(store):
    # 创建文本块
    chunk = TextChunk(
        id="doc_chunk_1",
        content="Python是一种广泛使用的编程语言...",
        title="Python编程语言介绍",
        source="programming_guide.txt",
        start_index=0,
        end_index=100,
        chunk_type="paragraph"
    )

    # 添加文本块
    await store.add_text_chunk(chunk)

    # 搜索文本块
    results = await store.search_text_chunks("编程语言", top_k=5)
```

## 搜索和检索

### 基本搜索

```python
async def basic_search(store):
    # 文本搜索
    entities = await store.search_entities("机器学习", top_k=10)

    # 使用向量搜索
    embedding = [0.1, 0.2, 0.3, ...]  # 预计算的向量
    entities = await store.search_entities(embedding, top_k=10)

    # 带过滤条件的搜索
    filter_dict = {"entity_type": "CONCEPT"}
    entities = await store.search_entities(
        "人工智能",
        top_k=5,
        filter_dict=filter_dict
    )
```

### 混合搜索

```python
async def hybrid_search(store):
    # 跨多种数据类型搜索
    search_types = {"entity", "relation", "text_chunk"}
    results = await store.hybrid_search(
        query="深度学习",
        search_types=search_types,
        top_k=5
    )

    # 处理结果
    for data_type, items in results.items():
        print(f"\n{data_type} 搜索结果:")
        for item, score in items:
            print(f"  {item.name if hasattr(item, 'name') else item.id}: {score:.3f}")
```

## 批量操作

### 批量添加数据

```python
async def batch_operations(store):
    # 批量添加实体
    entities = [
        Entity(name="TensorFlow", entity_type=EntityType.TOOL),
        Entity(name="PyTorch", entity_type=EntityType.TOOL),
        Entity(name="Scikit-learn", entity_type=EntityType.TOOL)
    ]

    results = await store.batch_add_entities(entities)
    print(f"批量添加结果: {results}")

    # 批量添加关系
    relations = [...]  # 关系列表
    results = await store.batch_add_relations(relations)

    # 批量添加时提供预计算向量
    embeddings = [
        [0.1, 0.2, 0.3, ...],  # TensorFlow 向量
        [0.2, 0.3, 0.4, ...],  # PyTorch 向量
        [0.3, 0.4, 0.5, ...]   # Scikit-learn 向量
    ]

    results = await store.batch_add_entities(entities, embeddings)
```

## 高级功能

### 自定义嵌入函数

```python
from agraph.vectordb.embeddings import create_openai_embedding_function

# 创建自定义嵌入函数
embedding_func = create_openai_embedding_function(
    model="text-embedding-3-large",
    api_key="your-api-key",
    batch_size=50,
    max_concurrency=5,
    use_cache=True,
    cache_size=2000
)

# 在 ChromaDB 中使用
store = create_chroma_store(
    embedding_function=embedding_func
)
```

### 使用上下文管理器

```python
async def context_manager_example():
    async with create_chroma_store(persist_directory="./data") as store:
        # 自动初始化
        await store.add_entity(entity)
        # 自动清理
```

### 统计信息

```python
async def statistics_example(store):
    # 获取存储统计
    stats = await store.get_stats()
    print(f"实体数量: {stats['entities']}")
    print(f"关系数量: {stats['relations']}")
    print(f"文本块数量: {stats['text_chunks']}")

    # 获取嵌入统计（如果使用OpenAI）
    embedding_stats = store.get_embedding_stats()
    if embedding_stats:
        print(f"总请求数: {embedding_stats['total_requests']}")
        print(f"缓存命中率: {embedding_stats.get('hit_rate', 0):.2%}")
```

## 性能优化

### 1. 批量操作优化

```python
# 好的做法：使用批量操作
entities = [...]  # 大量实体
await store.batch_add_entities(entities)

# 避免：循环单个添加
for entity in entities:
    await store.add_entity(entity)  # 低效
```

### 2. 嵌入缓存

```python
# 启用缓存减少重复计算
store = create_chroma_store(
    use_openai_embeddings=True,
    openai_embedding_config={
        "use_cache": True,
        "cache_size": 5000
    }
)
```

### 3. 预计算向量

```python
# 如果有预计算的向量，直接使用
pre_computed_embedding = [0.1, 0.2, ...]
await store.add_entity(entity, embedding=pre_computed_embedding)
```

### 4. 合理设置批处理大小

```python
embedding_config = {
    "batch_size": 100,        # 根据API限制调整
    "max_concurrency": 10,    # 控制并发请求数
    "timeout": 30.0           # 设置合理超时
}
```

## 故障排除

### 常见问题

1. **ChromaDB 初始化失败**

```python
try:
    store = create_chroma_store(persist_directory="./data")
    await store.initialize()
except VectorStoreError as e:
    print(f"初始化失败: {e}")
    # 检查目录权限和依赖安装
```

1. **OpenAI API 限制**

```python
# 降低并发和批次大小
embedding_config = {
    "batch_size": 20,
    "max_concurrency": 3,
    "max_retries": 5
}
```

1. **内存不足**

```python
# 使用较小的缓存和批次
store = create_memory_store()
# 或者使用持久化存储
store = create_chroma_store(persist_directory="./data")
```

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查向量存储状态
print(f"是否已初始化: {store.is_initialized()}")

# 验证嵌入向量
if store._validate_embedding([0.1, 0.2, 0.3]):
    print("向量格式正确")
```

## 最佳实践

1. **始终使用异步操作**
2. **合理选择存储后端**（内存用于测试，ChromaDB用于生产）
3. **启用嵌入缓存**减少API调用
4. **使用批量操作**提高性能
5. **设置合理的超时和重试**
6. **定期备份持久化数据**
7. **监控API使用量和成本**

## 下一步

- 学习[自定义向量数据库实现](./custom_vectordb_guide.md)
- 查看[API参考文档](./api_reference.md)
- 探索[高级用例](./advanced_examples.md)
