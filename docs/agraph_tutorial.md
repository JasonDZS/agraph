# AGraph 使用教程

AGraph 是一个强大的知识图谱构建和查询工具，支持从文本构建知识图谱、语义搜索和智能问答。本教程将指导您完成从安装到实际使用的全过程。

## 目录

1. [环境准备](#环境准备)
2. [基本配置](#基本配置)
3. [创建您的第一个知识图谱](#创建您的第一个知识图谱)
4. [语义搜索](#语义搜索)
5. [智能问答](#智能问答)
6. [高级功能](#高级功能)
7. [常见问题](#常见问题)

## 环境准备

### 系统要求
- Python 3.10 或更高版本
- 足够的磁盘空间用于存储向量数据库

### 安装 AGraph

```bash
pip install agraph
```

### 设置 OpenAI API Key

AGraph 使用 OpenAI 的 API 进行文本处理和嵌入生成。请设置您的 API Key：

```bash
export OPENAI_API_KEY="your-openai-api-key"
```

或者在代码中配置：

```python
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"
```

## 基本配置

### 1. 导入必要的模块

```python
import asyncio
import sys
from pathlib import Path
from agraph import AGraph, get_settings
from agraph.config import update_settings, save_config_to_workdir
```

### 2. 配置工作目录

```python
# 设置工作目录
project_root = Path(__file__).parent
workdir = str(project_root / "workdir" / "my_agraph_cache")
update_settings({"workdir": workdir})

# 保存配置到工作目录
try:
    config_path = save_config_to_workdir()
    print(f"✅ 配置已保存到: {config_path}")
except Exception as e:
    print(f"⚠️ 配置保存失败: {e}")

settings = get_settings()
```

## 创建您的第一个知识图谱

### 1. 准备文档数据

首先，您需要准备一些文本数据。AGraph 支持多种文件格式：

```python
# 从文件目录读取文档
documents_dir = Path("your_documents_directory")
sample_texts = []

if documents_dir.exists():
    print(f"📂 从 {documents_dir} 读取文档...")
    supported_extensions = {'.txt', '.md', '.json', '.csv'}

    for file_path in documents_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        sample_texts.append(content)
                        print(f"   📄 读取: {file_path.name} ({len(content)} 字符)")
            except UnicodeDecodeError:
                # 尝试 GBK 编码
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        content = f.read()
                        if content.strip():
                            sample_texts.append(content)
                            print(f"   📄 读取: {file_path.name} ({len(content)} 字符, GBK编码)")
                except Exception as e:
                    print(f"   ⚠️ 跳过文件 {file_path.name}: {e}")
            except Exception as e:
                print(f"   ⚠️ 读取文件失败 {file_path.name}: {e}")
```

### 2. 创建 AGraph 实例

```python
async def create_knowledge_graph():
    # 创建 AGraph 实例
    async with AGraph(
        collection_name="my_knowledge_graph",
        persist_directory=settings.workdir,  # 使用配置的工作目录
        vector_store_type="chroma",  # 使用 Chroma 向量存储
        use_openai_embeddings=True,  # 使用 OpenAI 嵌入
        enable_knowledge_graph=True,  # 启用知识图谱功能
    ) as agraph:
        # 初始化 AGraph
        await agraph.initialize()
        print("✅ AGraph初始化成功")

        # 从文本构建知识图谱
        print("🏗️ 构建知识图谱...")
        knowledge_graph = await agraph.build_from_texts(
            texts=sample_texts,
            graph_name="我的知识图谱",
            graph_description="基于文档构建的知识图谱",
            use_cache=True,  # 启用缓存
            save_to_vector_store=True,  # 保存到向量存储
        )

        print("✅ 知识图谱构建成功!")
        print(f"   📊 实体: {len(knowledge_graph.entities)} 个")
        print(f"   🔗 关系: {len(knowledge_graph.relations)} 个")
        print(f"   📄 文本块: {len(knowledge_graph.text_chunks)} 个")

        return agraph
```

## 语义搜索

AGraph 提供强大的语义搜索功能，可以搜索实体、关系和文本内容：

### 1. 搜索实体

```python
async def search_entities_example(agraph):
    """搜索实体示例"""
    search_term = "公司"
    print(f"🔍 搜索实体 '{search_term}':")

    entities = await agraph.search_entities(search_term, top_k=5)
    for i, (entity, score) in enumerate(entities):
        print(f"   {i+1}. {entity.name} ({entity.entity_type}) - 相似度: {score:.3f}")
        if entity.description:
            print(f"      描述: {entity.description[:100]}...")
```

### 2. 搜索文本内容

```python
async def search_text_example(agraph):
    """搜索文本示例"""
    search_term = "技术"
    print(f"🔍 搜索文本 '{search_term}':")

    text_chunks = await agraph.search_text_chunks(search_term, top_k=3)
    for i, (chunk, score) in enumerate(text_chunks):
        preview = chunk.content[:80] + "..." if len(chunk.content) > 80 else chunk.content
        print(f"   {i+1}. {preview} - 相似度: {score:.3f}")
```

### 3. 搜索关系

```python
async def search_relations_example(agraph):
    """搜索关系示例"""
    search_term = "管理"
    print(f"🔍 搜索关系 '{search_term}':")

    relations = await agraph.search_relations(search_term, top_k=3)
    for i, (relation, score) in enumerate(relations):
        print(f"   {i+1}. {relation.source} -> {relation.target}")
        print(f"      关系类型: {relation.relation_type}")
        print(f"      相似度: {score:.3f}")
```

## 智能问答

AGraph 的智能问答功能可以基于知识图谱回答用户问题：

### 1. 基本问答

```python
async def basic_qa_example(agraph):
    """基本问答示例"""
    question = "公司的主要业务是什么？"
    print(f"❓ 问题: {question}")

    # 获取回答
    response = await agraph.chat(question)
    print(f"🤖 回答: {response['answer']}")

    # 显示上下文信息
    context = response['context']
    entity_count = len(context.get('entities', []))
    chunk_count = len(context.get('text_chunks', []))
    print(f"   📊 使用了 {entity_count} 个实体, {chunk_count} 个文档片段")
```

### 2. 流式问答

```python
async def streaming_qa_example(agraph):
    """流式问答示例"""
    question = "公司的核心技术有哪些？"
    print(f"❓ 问题: {question}")
    print("🤖 回答: ", end="", flush=True)

    # 流式获取回答
    async for chunk_data in await agraph.chat(question, stream=True):
        if chunk_data["chunk"]:
            print(chunk_data["chunk"], end="", flush=True)
        if chunk_data["finished"]:
            print(f"\n✅ 完整回答: {chunk_data['answer']}")

            # 显示检索统计
            context = chunk_data['context']
            entity_count = len(context.get('entities', []))
            chunk_count = len(context.get('text_chunks', []))
            print(f"   📊 检索了 {entity_count} 个实体, {chunk_count} 个文档")
            break
```

## 高级功能

### 1. 获取系统统计信息

```python
async def get_stats_example(agraph):
    """获取系统统计信息"""
    print("📊 系统统计信息:")
    stats = await agraph.get_stats()

    if 'vector_store' in stats:
        vs_stats = stats['vector_store']
        print("向量存储:")
        print(f"   - 实体: {vs_stats.get('entities', 0)}")
        print(f"   - 关系: {vs_stats.get('relations', 0)}")
        print(f"   - 文本块: {vs_stats.get('text_chunks', 0)}")

    if 'knowledge_graph' in stats:
        kg_stats = stats['knowledge_graph']
        print("知识图谱:")
        print(f"   - 总实体数: {kg_stats.get('total_entities', 0)}")
        print(f"   - 总关系数: {kg_stats.get('total_relations', 0)}")
```

### 2. 批量处理

```python
async def batch_processing_example(agraph):
    """批量处理示例"""
    questions = [
        "公司的主要业务是什么？",
        "团队规模如何？",
        "有哪些核心技术？"
    ]

    print("🔄 批量问答处理:")
    results = []

    for i, question in enumerate(questions):
        print(f"\n处理问题 {i+1}/{len(questions)}: {question}")
        try:
            response = await agraph.chat(question)
            results.append({
                'question': question,
                'answer': response['answer'],
                'context_size': len(response['context'].get('entities', []))
            })
        except Exception as e:
            print(f"   ⚠️ 处理失败: {e}")
            results.append({
                'question': question,
                'answer': None,
                'error': str(e)
            })

    return results
```

## 完整示例

以下是一个完整的使用示例：

```python
#!/usr/bin/env python3
"""
AGraph 完整使用示例
"""

import asyncio
import sys
from pathlib import Path
from agraph import AGraph, get_settings
from agraph.config import update_settings, save_config_to_workdir

async def main():
    """主函数"""
    print("🚀 AGraph 完整使用示例")
    print("=" * 50)

    # 1. 配置设置
    project_root = Path(__file__).parent
    workdir = str(project_root / "workdir" / "tutorial_cache")
    update_settings({"workdir": workdir})

    # 保存配置
    try:
        config_path = save_config_to_workdir()
        print(f"✅ 配置已保存到: {config_path}")
    except Exception as e:
        print(f"⚠️ 配置保存失败: {e}")

    settings = get_settings()

    # 2. 准备示例数据
    sample_texts = [
        """
        ABC科技公司是一家专注于人工智能技术的创新企业。
        公司成立于2020年，总部位于北京，拥有50名技术专家。
        主要业务包括自然语言处理、计算机视觉和机器学习算法开发。
        """,
        """
        公司的核心技术团队由张博士领导，拥有丰富的AI研发经验。
        主要产品包括智能客服系统、图像识别平台和推荐算法引擎。
        客户覆盖金融、教育、医疗等多个行业。
        """,
        """
        公司采用敏捷开发模式，重视技术创新和人才培养。
        每年投入营收的30%用于研发，已获得15项技术专利。
        未来计划扩展到深度学习和强化学习领域。
        """
    ]

    # 3. 创建和使用 AGraph
    async with AGraph(
        collection_name="tutorial_demo",
        persist_directory=settings.workdir,
        vector_store_type="chroma",
        use_openai_embeddings=True,
        enable_knowledge_graph=True,
    ) as agraph:
        # 初始化
        await agraph.initialize()
        print("✅ AGraph 初始化成功")

        # 构建知识图谱
        print("\n🏗️ 构建知识图谱...")
        knowledge_graph = await agraph.build_from_texts(
            texts=sample_texts,
            graph_name="ABC科技公司知识图谱",
            graph_description="关于ABC科技公司的综合信息",
            use_cache=True,
            save_to_vector_store=True,
        )

        print("✅ 知识图谱构建完成!")
        print(f"   📊 实体: {len(knowledge_graph.entities)} 个")
        print(f"   🔗 关系: {len(knowledge_graph.relations)} 个")
        print(f"   📄 文本块: {len(knowledge_graph.text_chunks)} 个")

        # 语义搜索演示
        print("\n🔍 语义搜索演示")
        print("-" * 30)

        # 搜索实体
        entities = await agraph.search_entities("公司", top_k=3)
        print("搜索实体 '公司':")
        for i, (entity, score) in enumerate(entities):
            print(f"   {i+1}. {entity.name} ({entity.entity_type})")

        # 搜索文本
        text_chunks = await agraph.search_text_chunks("技术", top_k=2)
        print("\n搜索文本 '技术':")
        for i, (chunk, score) in enumerate(text_chunks):
            preview = chunk.content[:60] + "..." if len(chunk.content) > 60 else chunk.content
            print(f"   {i+1}. {preview}")

        # 智能问答演示
        print("\n💬 智能问答演示")
        print("-" * 30)

        questions = [
            "ABC科技公司的主要业务是什么？",
            "公司有多少员工？",
            "公司的核心技术有哪些？"
        ]

        for question in questions:
            print(f"\n❓ 问题: {question}")
            try:
                response = await agraph.chat(question)
                print(f"🤖 回答: {response['answer']}")

                # 显示上下文统计
                context = response['context']
                entity_count = len(context.get('entities', []))
                chunk_count = len(context.get('text_chunks', []))
                print(f"   📊 使用了 {entity_count} 个实体, {chunk_count} 个文档片段")

            except Exception as e:
                print(f"🤖 回答: 抱歉，无法回答这个问题: {e}")

        # 系统统计
        print("\n📊 系统统计信息")
        print("-" * 30)
        stats = await agraph.get_stats()

        if 'vector_store' in stats:
            vs_stats = stats['vector_store']
            print("向量存储:")
            print(f"   - 实体: {vs_stats.get('entities', 0)}")
            print(f"   - 关系: {vs_stats.get('relations', 0)}")
            print(f"   - 文本块: {vs_stats.get('text_chunks', 0)}")

        print(f"\n系统状态: {agraph}")

    print("\n✅ 教程演示完成!")

if __name__ == "__main__":
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("❌ 需要Python 3.7+版本")
        sys.exit(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⏹️ 演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        print("💡 提示: 请确保已正确安装agraph包并配置OpenAI API Key")
```

## 常见问题

### Q1: 如何处理不同编码的文件？
A: AGraph 会自动尝试 UTF-8 和 GBK 编码。如果需要其他编码，可以在读取文件时指定编码格式。

### Q2: 如何优化知识图谱的构建速度？
A:
- 启用缓存功能 (`use_cache=True`)
- 使用合适的文本分块大小
- 考虑使用更快的向量存储后端

### Q3: 如何处理大量文档？
A:
- 分批处理文档
- 使用持久化存储避免重复构建
- 监控内存使用情况

### Q4: 搜索结果不准确怎么办？
A:
- 调整 `top_k` 参数
- 检查文档质量和相关性
- 考虑使用更精确的搜索词

### Q5: 如何自定义实体和关系类型？
A: AGraph 支持自定义实体和关系类型，详见 API 文档中的类型定义部分。

## 下一步

现在您已经掌握了 AGraph 的基本用法，可以：

1. 尝试使用自己的文档数据
2. 探索更多的配置选项
3. 集成到您的应用程序中
4. 查看 API 文档了解更多高级功能

更多信息和示例，请查看项目的 `examples/` 目录。
