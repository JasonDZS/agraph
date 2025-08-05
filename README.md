# 知识图谱模块
## 🚀 快速开始

### 1. 基本知识图谱构建

```python
import asyncio
from agraph.builders import MinimalLLMGraphBuilder
from agraph.embeddings import JsonVectorStorage

async def build_knowledge_graph():
    # 创建图构建器
    builder = MinimalLLMGraphBuilder(
        openai_api_key="your-openai-api-key",
        llm_model="gpt-4o-mini",  # 指定LLM模型
        temperature=0.1
    )

    # 从文本构建知识图谱
    texts = [
        "苹果公司是由史蒂夫·乔布斯创立的科技公司。",
        "iPhone是苹果公司的旗舰智能手机产品。",
        "史蒂夫·乔布斯在2011年之前担任苹果公司CEO。"
    ]

    graph = await builder.build_graph(texts=texts, graph_name="科技公司")

    print(f"构建了包含 {len(graph.entities)} 个实体和 {len(graph.relations)} 个关系的知识图谱")
    return graph, builder

# 运行示例
if __name__ == "__main__":
    graph, builder = asyncio.run(build_knowledge_graph())
```

### 2. 知识问答功能

```python
import asyncio
from agraph import KnowledgeRetriever
from agraph.builders import FlexibleLLMGraphBuilder
from agraph.embeddings import JsonVectorStorage

async def question_answering():
    # 首先创建带搜索功能的构建器
    builder = FlexibleLLMGraphBuilder(
        openai_api_key="your-openai-api-key",
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        vector_storage=JsonVectorStorage("./vectors.json")
    )

    # 构建知识图谱
    texts = [
        "苹果公司是由史蒂夫·乔布斯创立的科技公司。",
        "iPhone是苹果公司的旗舰智能手机产品。",
        "史蒂夫·乔布斯在2011年之前担任苹果公司CEO。"
    ]

    graph = await builder.build_graph(texts=texts, graph_name="科技公司")

    # 创建知识检索器（与构建器分离）
    retriever = KnowledgeRetriever(
        graph=graph,
        graph_embedding=builder.graph_embedding
    )

    # 对知识图谱进行问答
    questions = [
        "谁创立了苹果公司？",
        "苹果公司生产什么产品？",
        "史蒂夫·乔布斯什么时候离开苹果？"
    ]

    for question in questions:
        # 搜索相关实体
        entities = await retriever.search_entities(question, top_k=3)
        print(f"问题: {question}")
        print(f"相关实体: {[entity['entity_name'] for entity in entities]}")
        print()

# 运行问答示例
if __name__ == "__main__":
    asyncio.run(question_answering())
```

### 3. 文档处理示例

```python
import asyncio
from agraph.builders import FlexibleLLMGraphBuilder
from agraph.processer.factory import DocumentProcessorFactory
from agraph.embeddings import JsonVectorStorage

async def process_documents():
    # 创建文档处理构建器
    builder = FlexibleLLMGraphBuilder(
        openai_api_key="your-openai-api-key",
        llm_model="gpt-4o-mini",
        embedding_model="text-embedding-3-small",
        vector_storage=JsonVectorStorage("./doc_vectors.json")
    )

    # 处理不同类型的文档
    document_paths = [
        "./examples/documents/company_info.txt",
        "./examples/documents/products.json",
        "./examples/documents/team.html"
    ]

    texts = []
    processor_factory = DocumentProcessorFactory()

    for doc_path in document_paths:
        processor = processor_factory.get_processor(doc_path)
        content = processor.process(doc_path)
        texts.append(f"文档: {doc_path}\n{content}")

    # 从处理后的文档构建图谱
    graph = await builder.build_graph(texts=texts, graph_name="文档知识库")

    print(f"处理了 {len(document_paths)} 个文档")
    print(f"构建了包含 {len(graph.entities)} 个实体的图谱")

if __name__ == "__main__":
    asyncio.run(process_documents())
```

## 📚 主要功能特性

### 🏗️ 知识图谱构建
- **智能实体识别**: 基于LLM自动抽取实体和关系
- **多格式支持**: PDF、Word、HTML、JSON、CSV等文档类型
- **增量更新**: 支持动态添加新文档到现有图谱
- **向量化存储**: 支持语义相似度搜索
- **多种构建器**: 提供MinimalLLMGraphBuilder、FlexibleLLMGraphBuilder等不同功能的构建器

### 🔍 知识问答检索
- **语义搜索**: 基于向量相似度的智能搜索
- **实体查询**: 查找相关实体和它们的属性
- **关系探索**: 发现实体间的复杂关系
- **智能问答**: 专门的KnowledgeRetriever提供问答功能
- **多种搜索模式**: 支持实体搜索、关系搜索和综合搜索

### 💾 灵活存储方案
- **JSON存储**: 轻量级文件存储，适合小规模应用
- **Neo4j存储**: 企业级图数据库，支持复杂查询
- **向量存储**: JsonVectorStorage支持高效的相似度搜索
- **LightRAG集成**: 支持GraphML格式和LightRAG工作目录结构

## 🔧 环境配置

### 安装依赖

```bash
# 开发安装（推荐）
make install-dev

# 或者直接安装
pip install -e .

# 可选依赖（根据需要安装）
pip install beautifulsoup4  # HTML处理
pip install pypdf          # PDF处理
pip install python-docx    # Word文档处理
pip install pandas         # Excel/CSV处理
pip install pillow          # 图像处理
pip install pytesseract     # OCR功能
```

### API密钥设置

```bash
# 设置OpenAI API密钥（必需）
export OPENAI_API_KEY="your-openai-api-key"

# 可选：自定义API地址
export OPENAI_API_BASE="https://api.openai.com/v1"
```

## 📖 更多示例

查看 `examples/` 目录获取更多完整示例：

- **基础功能**: `llm_builder_example.py` - 展示多种LLM构建器的使用
- **LightRAG集成**: `lightrag_example.py` - LightRAG构建器使用示例
- **文档处理**: `llm_builder_folder.py` - 批量文档处理示例
- **示例文档**: `documents/` - 包含各种格式的示例文档

## ⚡ 核心优势

- **🤖 智能化**: 基于LLM的自动实体关系抽取，无需手工规则
- **🔍 语义化**: 支持向量相似度搜索，理解语义而非仅匹配关键词
- **📄 多格式**: 自动处理PDF、Word、HTML等多种文档格式
- **⚡ 高性能**: 支持增量更新和批量处理，适合大规模应用
- **🔧 易扩展**: 模块化设计，支持自定义构建器和存储后端
- **🏗️ SOLID设计**: 严格遵循SOLID原则，提供专门的构建器和检索器
- **🔌 LightRAG集成**: 深度集成LightRAG框架，支持高级知识图谱功能

## 📝 注意事项

1. **API费用**: 使用OpenAI API会产生费用，建议先用小数据集测试
2. **网络连接**: 构建图谱时需要稳定的网络连接访问LLM服务
3. **内存使用**: 大规模文档可能需要较多内存，建议分批处理
4. **异步编程**: 所有构建和搜索操作都是异步的，需要使用`asyncio.run()`

## 🆘 快速问题解决

- **安装问题**: 运行 `make install-dev` 或 `pip install -e .`
- **API密钥**: 确保设置了有效的`OPENAI_API_KEY`
- **文档处理失败**: 安装相应的可选依赖包
- **内存不足**: 减少单次处理的文档数量或文档大小
- **测试运行**: 使用 `make test` 运行所有测试
- **代码检查**: 使用 `make check` 进行代码质量检查

---

🚀 **开始构建你的知识图谱吧！** 从简单的文本开始，逐步探索更多高级功能。
