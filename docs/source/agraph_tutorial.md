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

### 知识图谱构建控制

AGraph支持控制是否启用知识图谱构建。当禁用时，系统只进行文本分块和向量存储，跳过实体、关系和聚类提取：

#### 启用知识图谱（默认行为）

```python
# 完整的知识图谱构建，包括实体、关系、聚类提取
agraph = AGraph(
    collection_name="full_knowledge_graph",
    enable_knowledge_graph=True  # 默认值，可省略
)

# 构建完整的知识图谱
kg = await agraph.build_from_texts(texts)
print(f"实体数: {len(kg.entities)}")      # > 0
print(f"关系数: {len(kg.relations)}")     # > 0
print(f"聚类数: {len(kg.clusters)}")      # > 0
print(f"文本块数: {len(kg.text_chunks)}")  # > 0
```

#### 禁用知识图谱（仅文本模式）

```python
# 仅进行文本处理和向量存储，适合纯文档检索场景
agraph = AGraph(
    collection_name="text_only_mode",
    enable_knowledge_graph=False  # 关键设置
)

# 只会创建文本块，跳过实体/关系/聚类提取
kg = await agraph.build_from_texts(texts)
print(f"实体数: {len(kg.entities)}")      # 0
print(f"关系数: {len(kg.relations)}")     # 0
print(f"聚类数: {len(kg.clusters)}")      # 0
print(f"文本块数: {len(kg.text_chunks)}")  # > 0，保留文本分块功能

# 仍然支持基于文本块的搜索和问答
results = await agraph.search_text_chunks("查询内容")
response = await agraph.chat("用户问题")
```

#### 使用场景对比

| 功能特性 | 启用知识图谱 | 禁用知识图谱 |
|---------|-------------|-------------|
| 文本分块 | ✅ | ✅ |
| 实体提取 | ✅ | ❌ |
| 关系提取 | ✅ | ❌ |
| 聚类分析 | ✅ | ❌ |
| 语义搜索 | ✅ | ✅（仅文本块）|
| 智能问答 | ✅ | ✅（基于文本块）|
| 构建速度 | 较慢 | 快速 |
| 存储开销 | 较大 | 较小 |

#### 性能和资源对比

```python
# 测试两种模式的性能差异
import time

# 知识图谱模式
start = time.time()
agraph_kg = AGraph(enable_knowledge_graph=True)
kg_full = await agraph_kg.build_from_texts(large_texts)
kg_time = time.time() - start
print(f"知识图谱模式耗时: {kg_time:.2f}s")

# 仅文本模式
start = time.time()
agraph_text = AGraph(enable_knowledge_graph=False)
kg_text = await agraph_text.build_from_texts(large_texts)
text_time = time.time() - start
print(f"仅文本模式耗时: {text_time:.2f}s")
print(f"速度提升: {kg_time/text_time:.1f}x")
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

### 1. 选择合适的构建模式

#### 何时启用知识图谱

```python
# 适合以下场景启用知识图谱构建：
scenarios_for_kg = [
    "需要分析实体间复杂关系",
    "要进行知识图谱可视化",
    "需要基于实体和关系的精确搜索",
    "要进行知识推理和路径查找",
    "文档包含丰富的结构化信息",
    "需要进行聚类分析和主题发现"
]

# 示例：分析公司组织架构文档
agraph = AGraph(
    collection_name="company_structure",
    enable_knowledge_graph=True  # 需要提取人员、部门、职位关系
)
```

#### 何时禁用知识图谱

```python
# 适合以下场景禁用知识图谱构建：
scenarios_for_text_only = [
    "纯文档检索和相似性搜索",
    "大规模文档快速索引",
    "资源受限的环境部署",
    "主要进行关键词搜索",
    "文档结构相对简单",
    "需要快速原型验证"
]

# 示例：构建FAQ知识库
agraph = AGraph(
    collection_name="faq_database",
    enable_knowledge_graph=False  # 只需要问答匹配，不需要实体关系
)
```

#### 动态模式选择

```python
def choose_mode_by_content(texts):
    """根据文档内容动态选择构建模式"""

    # 简单的启发式规则
    total_length = sum(len(text) for text in texts)
    avg_length = total_length / len(texts) if texts else 0

    # 检测是否包含结构化信息
    structured_indicators = ["公司", "部门", "负责人", "项目", "产品", "客户"]
    structured_score = sum(
        1 for text in texts
        for indicator in structured_indicators
        if indicator in text
    ) / len(texts)

    if avg_length > 1000 and structured_score > 2:
        return True  # 启用知识图谱
    else:
        return False  # 仅文本模式

# 使用示例
enable_kg = choose_mode_by_content(document_texts)
agraph = AGraph(
    collection_name="adaptive_mode",
    enable_knowledge_graph=enable_kg
)
```

### 2. 文档预处理

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

#### 4. 知识图谱模式问题

```python
# 问题：知识图谱模式下没有提取到实体
async def debug_kg_mode():
    agraph = AGraph(enable_knowledge_graph=True)
    kg = await agraph.build_from_texts(texts)

    if len(kg.entities) == 0:
        print("⚠️ 未提取到实体，可能的原因：")
        print("1. LLM API配置问题")
        print("2. 文本内容缺乏结构化信息")
        print("3. 实体置信度阈值过高")

        # 解决方案：降低阈值或切换到文本模式
        agraph_text = AGraph(enable_knowledge_graph=False)
        kg_text = await agraph_text.build_from_texts(texts)
        print(f"文本模式创建了 {len(kg_text.text_chunks)} 个文本块")

# 问题：向量数据库为空
async def debug_empty_vectordb():
    agraph = AGraph(enable_knowledge_graph=False)
    kg = await agraph.build_from_texts(texts)

    # 检查构建结果
    stats = await agraph.get_stats()
    print(f"向量存储统计: {stats}")

    if stats.get('vector_store', {}).get('text_chunks', 0) == 0:
        print("❌ 向量数据库为空，检查：")
        print("1. texts 是否为空或内容过短")
        print("2. save_to_vector_store 是否设置为True")
        print("3. 文本分块是否正常工作")

# 问题：模式切换后数据不一致
async def handle_mode_switching():
    # 清理旧数据
    agraph = AGraph(collection_name="test", enable_knowledge_graph=True)
    await agraph.clear_all()

    # 重新构建
    kg = await agraph.build_from_texts(texts)
    print(f"模式切换后重建完成")
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

### 基础示例

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

### 知识图谱模式对比示例

这个示例展示了启用和禁用知识图谱的区别：

```python
#!/usr/bin/env python3
import asyncio
import time
from pathlib import Path
from agraph import AGraph, get_settings

async def compare_modes_example():
    """对比知识图谱模式和仅文本模式的差异"""

    # 准备测试文档
    documents = [
        "苹果公司(Apple Inc.)是一家美国跨国科技企业，由史蒂夫·乔布斯、史蒂夫·沃兹尼亚克和罗纳德·韦恩于1976年4月1日创立。",
        "微软公司(Microsoft Corporation)是一家美国跨国科技企业，由比尔·盖茨和保罗·艾伦于1975年4月4日创立。",
        "谷歌公司(Google LLC)是一家美国跨国科技企业，由拉里·佩奇和谢尔盖·布林于1998年9月4日创立。",
        "特斯拉公司(Tesla, Inc.)是一家美国电动汽车及能源公司，由马丁·艾伯哈德和马克·塔彭宁于2003年创立。"
    ]

    print("🔄 开始对比不同构建模式的表现")
    print("=" * 60)

    # 模式1：完整知识图谱构建
    print("\n📊 模式1: 启用知识图谱构建")
    start_time = time.time()

    async with AGraph(
        collection_name="kg_mode_demo",
        enable_knowledge_graph=True,
        persist_directory="./demo_kg"
    ) as agraph_kg:
        await agraph_kg.initialize()

        kg_full = await agraph_kg.build_from_texts(
            texts=documents,
            graph_name="科技企业知识图谱",
            save_to_vector_store=True
        )

        kg_build_time = time.time() - start_time

        print(f"⏱️  构建耗时: {kg_build_time:.2f}秒")
        print(f"📈 构建结果:")
        print(f"   • 实体数: {len(kg_full.entities)}")
        print(f"   • 关系数: {len(kg_full.relations)}")
        print(f"   • 聚类数: {len(kg_full.clusters)}")
        print(f"   • 文本块数: {len(kg_full.text_chunks)}")

        # 测试实体搜索
        entities = await agraph_kg.search_entities("公司", top_k=3)
        print(f"\n🔍 实体搜索 '公司' 找到 {len(entities)} 个结果:")
        for entity, score in entities[:2]:
            print(f"   • {entity.name} ({entity.entity_type}) - 相似度: {score:.3f}")

        # 测试智能问答
        question = "苹果公司是什么时候成立的？"
        response = await agraph_kg.chat(question)
        print(f"\n💬 问答测试: {question}")
        print(f"🤖 回答: {response['answer'][:100]}...")

        stats_kg = await agraph_kg.get_stats()

    # 模式2：仅文本模式
    print("\n" + "=" * 60)
    print("📄 模式2: 禁用知识图谱（仅文本模式）")
    start_time = time.time()

    async with AGraph(
        collection_name="text_mode_demo",
        enable_knowledge_graph=False,
        persist_directory="./demo_text"
    ) as agraph_text:
        await agraph_text.initialize()

        kg_text = await agraph_text.build_from_texts(
            texts=documents,
            graph_name="科技企业文档库",
            save_to_vector_store=True
        )

        text_build_time = time.time() - start_time

        print(f"⏱️  构建耗时: {text_build_time:.2f}秒")
        print(f"📈 构建结果:")
        print(f"   • 实体数: {len(kg_text.entities)}")
        print(f"   • 关系数: {len(kg_text.relations)}")
        print(f"   • 聚类数: {len(kg_text.clusters)}")
        print(f"   • 文本块数: {len(kg_text.text_chunks)}")

        # 测试文本块搜索
        chunks = await agraph_text.search_text_chunks("苹果公司", top_k=3)
        print(f"\n🔍 文本搜索 '苹果公司' 找到 {len(chunks)} 个结果:")
        for chunk, score in chunks[:2]:
            preview = chunk.content[:50] + "..." if len(chunk.content) > 50 else chunk.content
            print(f"   • {preview} - 相似度: {score:.3f}")

        # 测试智能问答（基于文本块）
        question = "苹果公司是什么时候成立的？"
        response = await agraph_text.chat(question)
        print(f"\n💬 问答测试: {question}")
        print(f"🤖 回答: {response['answer'][:100]}...")

        stats_text = await agraph_text.get_stats()

    # 性能对比总结
    print("\n" + "=" * 60)
    print("📊 性能对比总结")
    print(f"构建速度提升: {kg_build_time/text_build_time:.1f}x (文本模式更快)")
    print(f"知识图谱模式: {kg_build_time:.2f}s")
    print(f"仅文本模式: {text_build_time:.2f}s")

    print(f"\n💾 存储对比:")
    if 'vector_store' in stats_kg and 'vector_store' in stats_text:
        kg_total = sum(stats_kg['vector_store'].values())
        text_total = sum(stats_text['vector_store'].values())
        print(f"知识图谱模式存储项: {kg_total}")
        print(f"仅文本模式存储项: {text_total}")

    print(f"\n✨ 功能对比:")
    print("知识图谱模式: 支持实体搜索、关系分析、聚类发现")
    print("仅文本模式: 支持文档检索、相似性搜索、快速问答")

    print(f"\n🎯 推荐使用场景:")
    print("• 知识图谱模式 → 需要深度分析实体关系的复杂应用")
    print("• 仅文本模式 → 快速文档检索和问答的轻量级应用")

if __name__ == "__main__":
    asyncio.run(compare_modes_example())
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
- ✅ **知识图谱构建控制** - 新功能亮点！
- ✅ 选择合适的构建模式（完整KG vs 仅文本）
- ✅ 性能优化和错误处理
- ✅ 最佳实践和调试技巧

### 🎯 关键新功能：知识图谱构建可选项

本教程重点介绍了 AGraph v0.2+ 的重要新功能：

- **灵活模式选择**：通过 `enable_knowledge_graph` 参数控制构建行为
- **仅文本模式**：快速文档索引，跳过复杂的实体关系提取
- **完整KG模式**：深度分析，提取实体、关系、聚类
- **无缝切换**：相同的API，不同的处理逻辑
- **性能优化**：根据需求选择最适合的模式

### 📈 使用建议

| 应用场景 | 推荐模式 | 优势 |
|---------|---------|------|
| 文档检索系统 | 仅文本模式 | 快速、轻量 |
| FAQ知识库 | 仅文本模式 | 简单高效 |
| 企业知识管理 | 完整KG模式 | 深度分析 |
| 科研文献分析 | 完整KG模式 | 关系挖掘 |
| 快速原型验证 | 仅文本模式 | 开发迅速 |

现在你可以根据具体需求，在自己的项目中灵活使用AGraph了！

## 更多资源

- [API参考文档](../source/modules.rst)
- [向量数据库教程](vectordb_tutorial.md)
- [导入导出功能](import_export_tutorial.md)
- [自定义向量数据库指南](custom_vectordb_guide.md)
