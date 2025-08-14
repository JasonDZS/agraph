# AGraph: 统一知识图谱系统

AGraph是一个统一的知识图谱构建、向量存储和对话系统，集成了知识图谱构建、向量检索和基于知识库的问答功能。

## 核心特性

### 🏗️ 知识图谱构建

- **多源输入支持**: 从文档文件或文本列表构建知识图谱
- **实体和关系提取**: 自动提取实体、关系和聚类
- **增量缓存**: 支持缓存机制，提高构建效率
- **灵活配置**: 可自定义置信度阈值、聚类算法等参数

### 🗂️ 向量存储

- **多后端支持**: 支持ChromaDB、内存存储等多种向量数据库
- **批量操作**: 高效的批量数据存储和检索
- **自动持久化**: 知识图谱自动保存到向量存储
- **灵活检索**: 支持实体、关系、文本块的语义检索

### 💬 智能对话

- **上下文检索**: 基于用户问题检索相关实体、关系和文档
- **LLM集成**: 支持OpenAI兼容API的大语言模型
- **对话历史**: 支持多轮对话上下文管理
- **结构化回答**: 提供带引用的结构化回答

## 快速开始

### 安装依赖

```python
# 基本功能
pip install agraph

# 完整功能（包括ChromaDB支持）
pip install "agraph[chroma]"
```

### 基本使用

```python
import asyncio
from agraph import AGraph

async def main():
    # 1. 创建AGraph实例
    agraph = AGraph(
        collection_name="my_knowledge_base",
        persist_directory="./vectordb",
        vector_store_type="chroma",  # 或 "memory"
        use_openai_embeddings=True
    )

    # 2. 使用上下文管理器
    async with agraph:
        # 3. 从文本构建知识图谱
        texts = [
            "苹果公司是一家美国科技公司。",
            "史蒂夫·乔布斯是苹果公司的联合创始人。",
            "iPhone是苹果公司的智能手机产品。"
        ]

        kg = agraph.build_from_texts(
            texts=texts,
            graph_name="科技知识图谱",
            save_to_vector_store=True
        )

        print(f"构建完成: {len(kg.entities)} 实体, {len(kg.relations)} 关系")

        # 4. 搜索实体
        entities = await agraph.search_entities("苹果公司", top_k=5)
        print(f"找到 {len(entities)} 个相关实体")

        # 5. 智能对话
        response = await agraph.chat(
            question="苹果公司的创始人是谁？",
            response_type="简洁回答"
        )
        print(f"回答: {response['answer']}")

# 运行示例
asyncio.run(main())
```

### 从文档构建

```python
async def build_from_documents():
    async with AGraph() as agraph:
        # 从文档文件构建知识图谱
        documents = ["./doc1.txt", "./doc2.pdf", "./doc3.docx"]

        kg = agraph.build_from_documents(
            documents=documents,
            graph_name="文档知识图谱",
            graph_description="基于公司文档构建的知识图谱",
            use_cache=True
        )

        return kg
```

## 配置选项

### 初始化参数

```python
agraph = AGraph(
    collection_name="agraph_knowledge",     # 集合名称
    persist_directory="./agraph_vectordb",  # 持久化目录
    vector_store_type="chroma",             # 向量存储类型
    config=BuilderConfig(...),             # 构建器配置
    use_openai_embeddings=True              # 是否使用OpenAI嵌入
)
```

### BuilderConfig配置

```python
from agraph import BuilderConfig

config = BuilderConfig(
    chunk_size=1000,                        # 文本块大小
    chunk_overlap=200,                      # 文本块重叠
    entity_confidence_threshold=0.7,        # 实体置信度阈值
    relation_confidence_threshold=0.6,      # 关系置信度阈值
    llm_provider="openai",                  # LLM提供商
    llm_model="gpt-3.5-turbo",             # LLM模型
    cluster_algorithm="community_detection" # 聚类算法
)
```

## API 参考

### 主要方法

#### 构建方法

- `build_from_documents(documents, graph_name, ...)` - 从文档构建知识图谱
- `build_from_texts(texts, graph_name, ...)` - 从文本构建知识图谱
- `save_knowledge_graph()` - 保存知识图谱到向量存储

#### 检索方法

- `search_entities(query, top_k, filter_dict)` - 搜索实体
- `search_relations(query, top_k, filter_dict)` - 搜索关系
- `search_text_chunks(query, top_k, filter_dict)` - 搜索文本块

#### 对话方法

- `chat(question, conversation_history, ...)` - 基于知识库的智能对话

#### 管理方法

- `get_stats()` - 获取系统统计信息
- `clear_all()` - 清除所有数据
- `close()` - 关闭系统

### 属性

- `is_initialized` - 检查是否已初始化
- `has_knowledge_graph` - 检查是否有知识图谱
- `knowledge_graph` - 当前知识图谱实例
- `vector_store` - 向量存储实例

## 环境变量配置

```bash
# OpenAI配置
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.openai.com/v1

# LLM配置
LLM_MODEL=gpt-3.5-turbo
LLM_PROVIDER=openai
LLM_MAX_TOKENS=4096

# 嵌入配置
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_PROVIDER=openai
```

## 高级功能

### 多轮对话

```python
conversation_history = []

async def chat_loop():
    while True:
        question = input("用户: ")
        if question.lower() in ['quit', 'exit']:
            break

        response = await agraph.chat(
            question=question,
            conversation_history=conversation_history,
            entity_top_k=5,
            text_chunk_top_k=3
        )

        print(f"助手: {response['answer']}")

        # 更新对话历史
        conversation_history.append({
            "user": question,
            "assistant": response['answer']
        })
```

### 自定义检索和过滤

```python
# 使用过滤条件检索实体
entities = await agraph.search_entities(
    query="公司",
    top_k=10,
    filter_dict={"entity_type": "organization"}
)

# 检索特定类型的关系
relations = await agraph.search_relations(
    query="创立",
    top_k=5,
    filter_dict={"relation_type": "founded_by"}
)
```

### 系统监控

```python
# 获取详细统计信息
stats = await agraph.get_stats()
print(f"向量存储统计: {stats['vector_store']}")
print(f"知识图谱统计: {stats['knowledge_graph']}")
print(f"构建器统计: {stats['builder']}")
```

## 最佳实践

### 1. 资源管理

- 始终使用异步上下文管理器 `async with AGraph() as agraph:`
- 确保在完成后正确关闭连接

### 2. 缓存策略

- 开启缓存以提高重复构建的效率
- 定期清理缓存以释放磁盘空间

### 3. 性能优化

- 对于大型文档集合，考虑分批处理
- 调整chunk_size和置信度阈值以平衡质量和性能

### 4. 错误处理

```python
try:
    kg = agraph.build_from_texts(texts)
except Exception as e:
    logger.error(f"构建失败: {e}")
    # 处理错误或重试
```

## 故障排除

### 常见问题

**问题**: "向量存储未初始化"
**解决**: 确保调用 `await agraph.initialize()` 或使用上下文管理器

**问题**: "LLM调用失败"
**解决**: 检查API密钥和网络连接，确保环境变量配置正确

**问题**: "ChromaDB不可用"
**解决**: 安装ChromaDB依赖 `pip install chromadb`

### 日志调试

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 示例项目

完整示例请参考:

- `examples/agraph_example.py` - 基本功能演示
- `examples/end_to_end_example.py` - 端到端示例

## 贡献指南

欢迎提交Issue和Pull Request来改进AGraph！

## 许可证

MIT License - 详见LICENSE文件
