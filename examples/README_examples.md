# AGraph 示例文件说明

本目录包含了AGraph的各种使用示例，帮助您快速了解和使用AGraph的功能。

## 📚 示例文件列表

### 🚀 入门示例

#### `agraph_quickstart.py` - 快速开始

- **适合**: 初学者，快速了解AGraph基本功能
- **功能**: 基本的知识图谱构建、搜索和问答
- **运行时间**: 1-2分钟
- **依赖**: 无外部API依赖

```bash
python examples/agraph_quickstart.py
```

**演示内容:**

- AGraph系统初始化
- 从文本构建知识图谱
- 实体和文档搜索
- 智能问答对话

### 📖 完整功能示例

#### `agraph_complete_demo.py` - 完整功能演示

- **适合**: 了解AGraph所有核心功能
- **功能**: 全面展示AGraph的各项能力
- **运行时间**: 3-5分钟
- **依赖**: 无外部API依赖

```bash
python examples/agraph_complete_demo.py
```

**演示内容:**

- 系统初始化和配置
- 知识图谱构建和优化
- 多种检索方式对比
- 多轮智能对话
- 系统监控和管理

#### `agraph_documents_demo.py` - 真实文档处理

- **适合**: 了解如何处理企业文档
- **功能**: 从documents/目录读取文档并构建知识图谱
- **运行时间**: 2-4分钟（取决于文档数量）
- **依赖**: documents/目录中的文档文件

```bash
python examples/agraph_documents_demo.py
```

**演示内容:**

- 多格式文档处理（.txt, .md, .json, .csv）
- 企业级知识图谱构建
- 基于实际文档的问答
- 文档内容检索和分析

### ⚖️ 对比示例

#### `agraph_vs_traditional.py` - 方法对比

- **适合**: 了解AGraph的优势和应用场景
- **功能**: 对比传统文本检索与AGraph知识图谱方法
- **运行时间**: 3-4分钟
- **依赖**: 无外部API依赖

```bash
python examples/agraph_vs_traditional.py
```

**演示内容:**

- 传统关键词搜索 vs 语义搜索
- 简单问答 vs 智能推理
- 性能和准确性对比
- 使用场景建议

### 📊 已有示例

#### `end_to_end_example.py` - 端到端流程

- **适合**: 了解完整的知识图谱构建流程
- **功能**: 使用KnowledgeGraphBuilder构建知识图谱
- **特点**: 展示原始的构建器功能

#### `agraph_example.py` - AGraph基础示例

- **适合**: AGraph类的基本使用
- **功能**: 展示AGraph的基础API

## 🚀 快速运行指南

### 环境要求

```bash
# Python版本要求
Python 3.7+

# 基本依赖
pip install -e .  # 从项目根目录安装

# 可选依赖（用于ChromaDB）
pip install chromadb
```

### 推荐运行顺序

1. **首次使用**: 从 `agraph_quickstart.py` 开始
2. **深入了解**: 运行 `agraph_complete_demo.py`
3. **实际应用**: 尝试 `agraph_documents_demo.py`
4. **方案选择**: 查看 `agraph_vs_traditional.py`

### 命令行运行

```bash
# 从项目根目录运行
cd /path/to/agraph

# 快速开始
python examples/agraph_quickstart.py

# 完整演示
python examples/agraph_complete_demo.py

# 文档处理
python examples/agraph_documents_demo.py

# 方法对比
python examples/agraph_vs_traditional.py
```

## 📁 documents/ 目录

示例文档目录包含各种格式的测试文件：

- `company_info.txt` - 公司介绍文本
- `technology_stack.md` - 技术栈说明
- `products.json` - 产品信息（JSON格式）
- `research_papers.csv` - 研究论文数据
- `team.html` - 团队信息（HTML格式）
- `财务费用报账指南（poc素材）.docx` - Word文档示例

## 🔧 配置说明

### 环境变量设置

创建 `.env` 文件（可选，用于LLM功能）:

```bash
# OpenAI配置（用于完整LLM功能）
OPENAI_API_KEY=your-api-key
OPENAI_API_BASE=https://api.openai.com/v1

# LLM配置
LLM_MODEL=gpt-3.5-turbo
LLM_PROVIDER=openai

# 嵌入配置
EMBEDDING_MODEL=text-embedding-ada-002
EMBEDDING_PROVIDER=openai
```

### 工作目录

示例运行时会在以下目录创建工作文件：

- `workdir/` - 各种示例的工作目录
- `*.vectordb/` - 向量数据库持久化目录
- `demo_data/` - 临时演示数据

## 📋 故障排除

### 常见问题

**问题1**: "ModuleNotFoundError: No module named 'agraph'"
**解决**: 从项目根目录运行 `pip install -e .`

**问题2**: 知识图谱构建失败
**解决**: 示例包含备用数据机制，会自动创建演示数据

**问题3**: "向量存储未初始化"
**解决**: 确保使用了 `async with AGraph() as agraph:` 语法

**问题4**: ChromaDB相关错误
**解决**: 使用 `vector_store_type="memory"` 避免ChromaDB依赖

### 性能优化

- 使用内存存储 (`vector_store_type="memory"`) 进行快速演示
- 设置较小的 `chunk_size` 和 `top_k` 值可提高响应速度
- 生产环境建议使用ChromaDB以获得持久化能力

## 💡 自定义示例

### 创建您自己的示例

```python
#!/usr/bin/env python3
import asyncio
from agraph import AGraph

async def my_custom_demo():
    async with AGraph(
        collection_name="my_demo",
        vector_store_type="memory"
    ) as agraph:

        # 您的自定义代码
        texts = ["您的文本数据..."]
        kg = agraph.build_from_texts(texts)

        response = await agraph.chat("您的问题")
        print(response['answer'])

if __name__ == "__main__":
    asyncio.run(my_custom_demo())
```

## 🤝 贡献

欢迎提交新的示例！请确保：

- 代码清晰易懂
- 包含适当的注释
- 提供运行说明
- 处理异常情况

## 📖 更多资源

- [AGraph完整文档](../docs/AGraph_README.md)
- [API参考](../docs/)
- [GitHub仓库](https://github.com/your-repo/agraph)

---

🎯 **开始您的AGraph之旅**: 运行 `python examples/agraph_quickstart.py`
