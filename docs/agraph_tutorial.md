# AGraph 使用教程

本教程将引导你通过实际示例学习AGraph知识图谱工具包的核心功能。

## 目录

1. [环境准备](#环境准备)
2. [快速开始](#快速开始)
3. [核心功能详解](#核心功能详解)
4. [高级用法](#高级用法)
5. [最佳实践](#最佳实践)
6. [故障排除](#故障排除)

## 环境准备

### 系统要求

- Python 3.7+
- OpenAI API密钥（用于LLM和embedding服务）

### 安装依赖

```bash
# 开发安装
make install-dev

# 或者基础安装
pip install -e .
```

### 环境配置

设置环境变量或创建`.env`文件：

```bash
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # 可选，默认值
OPENAI_MODEL=gpt-3.5-turbo  # 可选，默认值
```

## 快速开始

### 1. 基本设置

```python
import asyncio
from pathlib import Path
from agraph import AGraph, get_settings

# 配置工作目录
settings = get_settings()
settings.workdir = str(Path("workdir/my_project"))
```

### 2. 初始化AGraph实例

```python
async def main():
    async with AGraph(
        collection_name="my_knowledge_graph",
        persist_directory=settings.workdir,
        vector_store_type="chroma",
        use_openai_embeddings=True
    ) as agraph:
        await agraph.initialize()
        print("✅ AGraph初始化成功")
```

### 3. 准备文档数据

AGraph支持多种文档格式：

```python
# 方式1: 直接提供文本列表
sample_texts = [
    "我们公司是一家专注于人工智能技术的科技企业。",
    "团队由50名工程师组成，主要研发机器学习和深度学习产品。",
    "公司总部位于北京，在上海设有研发中心。"
]

# 方式2: 从文件读取
documents_dir = Path("documents")
sample_texts = []

for file_path in documents_dir.glob("*.txt"):
    with open(file_path, 'r', encoding='utf-8') as f:
        sample_texts.append(f.read())
```

### 4. 构建知识图谱

```python
# 从文本构建知识图谱
knowledge_graph = await agraph.build_from_texts(
    texts=sample_texts,
    graph_name="企业知识图谱",
    graph_description="基于企业文档构建的知识图谱",
    use_cache=True,  # 启用缓存加速
    save_to_vector_store=True  # 保存到向量存储
)

print(f"📊 构建完成: {len(knowledge_graph.entities)} 个实体, {len(knowledge_graph.relations)} 个关系")
```

## 核心功能详解

### 语义搜索

#### 搜索实体

```python
# 按名称搜索实体
entities = await agraph.search_entities("公司", top_k=5)
for entity, score in entities:
    print(f"实体: {entity.name} ({entity.entity_type}) - 相似度: {score:.3f}")
```

#### 搜索文本块

```python
# 按内容搜索文本块
text_chunks = await agraph.search_text_chunks("人工智能技术", top_k=3)
for chunk, score in text_chunks:
    preview = chunk.content[:100] + "..." if len(chunk.content) > 100 else chunk.content
    print(f"文本: {preview} - 相似度: {score:.3f}")
```

### 智能问答

#### 基础问答

```python
# 简单问答
question = "公司的主要业务是什么？"
response = await agraph.chat(question)
print(f"回答: {response}")
```

#### 流式问答

```python
# 流式响应，实时显示生成过程
async for chunk_data in await agraph.chat(question, stream=True):
    if chunk_data["chunk"]:
        print(chunk_data["chunk"], end="", flush=True)
    if chunk_data["finished"]:
        print(f"\n✅ 完整回答: {chunk_data['answer']}")

        # 显示检索上下文信息
        context = chunk_data['context']
        entities_used = len(context.get('entities', []))
        chunks_used = len(context.get('text_chunks', []))
        print(f"📊 使用了 {entities_used} 个实体, {chunks_used} 个文档片段")
        break
```

### 图谱分析

#### 查看统计信息

```python
# 获取系统统计
stats = await agraph.get_stats()

if 'vector_store' in stats:
    vs_stats = stats['vector_store']
    print("向量存储统计:")
    print(f"  - 实体数: {vs_stats.get('entities', 0)}")
    print(f"  - 关系数: {vs_stats.get('relations', 0)}")
    print(f"  - 文本块数: {vs_stats.get('text_chunks', 0)}")
```

#### 实体关系探索

```python
# 获取特定实体的关系
entity_name = "公司"
entities = await agraph.search_entities(entity_name, top_k=1)
if entities:
    entity = entities[0][0]
    print(f"实体: {entity.name}")
    print(f"类型: {entity.entity_type}")
    print(f"属性: {entity.properties}")
    print(f"别名: {entity.aliases}")
```

## 高级用法

### 缓存机制

AGraph提供智能缓存来提高性能：

```python
# 启用缓存构建
knowledge_graph = await agraph.build_from_texts(
    texts=sample_texts,
    graph_name="cached_graph",
    use_cache=True,  # 第一次构建后会缓存结果
    cache_ttl=3600   # 缓存1小时
)

# 后续相同文本的构建将直接使用缓存
```

### 持久化存储

```python
# 指定持久化目录
async with AGraph(
    collection_name="persistent_graph",
    persist_directory="/path/to/storage",
    vector_store_type="chroma"
) as agraph:
    # 数据会自动保存到指定目录
    # 下次启动时会自动加载
    pass
```

### 自定义配置

```python
from agraph import get_settings

settings = get_settings()
# 自定义LLM模型
settings.openai_model = "gpt-4"
# 自定义embedding模型
settings.embedding_model = "text-embedding-ada-002"
# 自定义文本分块大小
settings.chunk_size = 1000
settings.chunk_overlap = 200
```

## 最佳实践

### 1. 文档预处理

```python
def preprocess_texts(texts):
    """文本预处理最佳实践"""
    processed = []
    for text in texts:
        # 清理空白字符
        text = text.strip()
        # 过滤过短的文本
        if len(text) < 50:
            continue
        # 规范化编码
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        processed.append(text)
    return processed

sample_texts = preprocess_texts(raw_texts)
```

### 2. 错误处理

```python
async def robust_build():
    try:
        knowledge_graph = await agraph.build_from_texts(
            texts=sample_texts,
            graph_name="robust_graph",
            use_cache=True
        )
        return knowledge_graph
    except Exception as e:
        print(f"构建失败: {e}")
        # 降级处理或重试逻辑
        return None
```

### 3. 性能优化

```python
# 批量处理大量文档
async def process_large_dataset(texts, batch_size=10):
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        graph = await agraph.build_from_texts(
            texts=batch,
            graph_name=f"batch_{i//batch_size}",
            use_cache=True
        )
        results.append(graph)
    return results
```

### 4. 质量监控

```python
async def monitor_quality():
    stats = await agraph.get_stats()

    # 检查实体数量是否合理
    entity_count = stats['vector_store'].get('entities', 0)
    text_chunks = stats['vector_store'].get('text_chunks', 0)

    if entity_count == 0:
        print("⚠️ 警告: 没有提取到实体")
    elif entity_count / text_chunks < 0.1:
        print("⚠️ 警告: 实体密度过低，可能需要调整提取参数")
```

## 故障排除

### 常见问题

#### 1. 初始化失败

```python
# 检查API密钥
import os
if not os.getenv('OPENAI_API_KEY'):
    print("❌ 请设置OPENAI_API_KEY环境变量")

# 检查网络连接
try:
    await agraph.initialize()
except Exception as e:
    print(f"初始化失败: {e}")
    # 可能是网络问题或API限额
```

#### 2. 构建速度慢

```python
# 启用缓存
knowledge_graph = await agraph.build_from_texts(
    texts=sample_texts,
    use_cache=True  # 重要！
)

# 减少文本量
if len(sample_texts) > 100:
    sample_texts = sample_texts[:100]  # 先用小数据集测试
```

#### 3. 内存不足

```python
# 分批处理
batch_size = 5  # 根据系统内存调整
for i in range(0, len(texts), batch_size):
    batch = texts[i:i+batch_size]
    # 处理批次
```

### 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查向量存储状态
stats = await agraph.get_stats()
print(f"调试信息: {stats}")

# 测试简单查询
simple_entities = await agraph.search_entities("测试", top_k=1)
print(f"测试查询结果: {simple_entities}")
```

## 完整示例

这里是一个完整的工作示例：

```python
#!/usr/bin/env python3
import asyncio
from pathlib import Path
from agraph import AGraph, get_settings

async def complete_example():
    # 配置
    settings = get_settings()
    settings.workdir = str(Path("workdir/tutorial"))

    # 示例文档
    sample_texts = [
        "TechCorp是一家成立于2018年的人工智能公司。公司专注于自然语言处理和计算机视觉技术。",
        "公司总部位于北京中关村，员工总数120人，其中研发人员占80%。",
        "TechCorp的主要产品包括智能客服系统、文档分析平台和图像识别API。",
        "公司在2023年完成了B轮融资，融资金额5000万美元，由红杉资本领投。"
    ]

    # 初始化并构建
    async with AGraph(
        collection_name="techcorp_knowledge",
        persist_directory=settings.workdir,
        vector_store_type="chroma",
        use_openai_embeddings=True
    ) as agraph:
        await agraph.initialize()

        # 构建知识图谱
        knowledge_graph = await agraph.build_from_texts(
            texts=sample_texts,
            graph_name="TechCorp知识图谱",
            graph_description="关于TechCorp公司的知识图谱",
            use_cache=True,
            save_to_vector_store=True
        )

        print(f"✅ 构建完成: {len(knowledge_graph.entities)} 实体, {len(knowledge_graph.relations)} 关系")

        # 语义搜索
        entities = await agraph.search_entities("公司", top_k=3)
        print("\n🔍 实体搜索结果:")
        for entity, score in entities:
            print(f"  - {entity.name} ({entity.entity_type})")

        # 智能问答
        questions = [
            "TechCorp是什么时候成立的？",
            "公司有多少员工？",
            "主要产品有哪些？"
        ]

        print("\n💬 问答演示:")
        for question in questions:
            print(f"\n❓ {question}")
            response = await agraph.chat(question)
            print(f"🤖 {response}")

        # 系统统计
        stats = await agraph.get_stats()
        print(f"\n📊 系统统计: {stats}")

if __name__ == "__main__":
    asyncio.run(complete_example())
```

运行这个示例：

```bash
python complete_example.py
```

## 总结

AGraph提供了一个简单而强大的接口来构建和查询知识图谱。通过本教程，你已经学会了：

- ✅ 环境设置和初始化
- ✅ 从文本构建知识图谱
- ✅ 语义搜索和智能问答
- ✅ 性能优化和错误处理
- ✅ 最佳实践和调试技巧

现在你可以开始在自己的项目中使用AGraph了！

## 更多资源

- [API参考文档](../source/modules.rst)
- [向量数据库教程](vectordb_tutorial.md)
- [导入导出功能](import_export_tutorial.md)
- [自定义向量数据库指南](custom_vectordb_guide.md)
