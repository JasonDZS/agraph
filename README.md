# AGraph

AGraph is a modern knowledge graph toolkit for building, managing, and querying knowledge graphs.
It provides a clean API for handling entities and relations, with integrated vector database support
for semantic search and intelligent Q&A.

## Features

- **Pipeline Architecture**: Modern pipeline-based processing with 83% complexity reduction
- **Easy to Use**: Intuitive API design for quick adoption
- **Smart Construction**: Automatically extract entities and relations from text
- **Intelligent Caching**: Advanced caching system with error recovery capabilities
- **Performance Monitoring**: Detailed metrics and performance tracking
- **Semantic Search**: Vector similarity-based intelligent search
- **Smart Q&A**: RAG-based conversational system using knowledge graphs
- **Streaming Support**: Real-time streaming conversations and incremental updates
- **Type Safe**: Complete type annotations and validation
- **Multiple Storage**: Support for ChromaDB and other vector databases
- **Configuration Management**: Persistent configuration with workdir support

## Installation

### Basic Installation

```bash
# Basic installation with core functionality
pip install agraph

# Or with uv (recommended)
uv add agraph
```

### Optional Dependencies

AGraph provides optional dependency groups for specific use cases:

```bash
# API Server Dependencies
uv add "agraph[api]"        # FastAPI web server
uv add "agraph[server]"     # Production server with Gunicorn
uv add "agraph[all]"        # All optional dependencies

# Individual components
uv add "agraph[vectordb]"   # Enhanced vector database support
uv add "agraph[jupyter]"    # Jupyter notebook support
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/JasonDZS/agraph.git
cd agraph

# Install with development dependencies
make install-dev
# or
uv add -e ".[dev]"
```

### Installation Options Summary

| Option | Dependencies | Use Case |
|--------|-------------|----------|
| `agraph` | Core only | Basic knowledge graph functionality |
| `agraph[api]` | + FastAPI, Uvicorn | REST API development |
| `agraph[server]` | + API + Gunicorn | Production deployment |
| `agraph[vectordb]` | + ChromaDB | Enhanced vector operations |
| `agraph[jupyter]` | + Jupyter support | Notebook development |
| `agraph[all]` | All above | Complete installation |
| `agraph[dev]` | All + dev tools | Development environment |

## Quick Start

### Basic Usage

```python
#!/usr/bin/env python3
"""
AGraph 快速开始示例 (Pipeline架构版本)
"""

import asyncio
import sys
import time
from pathlib import Path
from agraph import AGraph, get_settings
from agraph.config import update_settings, save_config_to_workdir

# 设置工作目录并保存配置
workdir = "./workdir/agraph_quickstart-cache"
update_settings({"workdir": workdir})

# 保存配置到工作目录
try:
    config_path = save_config_to_workdir()
    print(f"✅ 配置已保存到: {config_path}")
except Exception as e:
    print(f"⚠️  配置保存失败: {e}")

settings = get_settings()

async def main():
    print("🚀 AGraph 快速开始示例")
    print("=" * 40)

    # 示例文本数据
    sample_texts = [
        "Apple Inc. is an American multinational technology company.",
        "Apple Inc. is headquartered in Cupertino, California.",
        "Apple Inc. develops iPhone and iPad products.",
        "The company was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne.",
        "Apple is known for its innovative design and user experience."
    ]

    # 1. 创建AGraph实例并初始化 (Pipeline架构)
    print("\n📦 1. 初始化AGraph (Pipeline架构)...")
    print("   🏗️ 使用新的Pipeline架构 (83%复杂度降低)")
    print("   ⚡ 智能缓存和错误恢复")
    print("   📊 详细的性能监控和指标")

    async with AGraph(
        collection_name="quickstart_demo",
        persist_directory=settings.workdir,
        vector_store_type="chroma",
        use_openai_embeddings=True,
        enable_knowledge_graph=True,  # 启用知识图谱功能
    ) as agraph:
        await agraph.initialize()
        print("✅ AGraph初始化成功 (内部使用Pipeline架构)")

        # 2. 从文本构建知识图谱 (使用Pipeline架构)
        print("\n🏗️ 2. 构建知识图谱 (Pipeline架构)...")
        print("   📋 Pipeline步骤: 文本分块 → 实体提取 → 关系提取 → 聚类 → 组装")

        start_time = time.time()
        knowledge_graph = await agraph.build_from_texts(
            texts=sample_texts,
            graph_name="科技公司知识图谱",
            graph_description="关于科技公司的基础知识图谱",
            use_cache=True,  # 启用缓存以加快后续构建速度
            save_to_vector_store=True,  # 保存到向量存储
        )
        build_time = time.time() - start_time

        print("✅ 知识图谱构建成功!")
        print(f"   ⏱️ 构建时间: {build_time:.2f}秒 (Pipeline优化)")
        print(f"   📊 实体: {len(knowledge_graph.entities)} 个")
        print(f"   🔗 关系: {len(knowledge_graph.relations)} 个")
        print(f"   📄 文本块: {len(knowledge_graph.text_chunks)} 个")

        # 3. 语义搜索演示
        print("\n🔍 3. 语义搜索演示...")

        # 搜索实体
        print("搜索实体 'technology company':")
        entities = await agraph.search_entities("technology company", top_k=3)
        for i, (entity, score) in enumerate(entities):
            print(f"   {i+1}. {entity.name} ({entity.entity_type})")

        # 搜索文本
        print("\n搜索文本 'headquarters':")
        text_chunks = await agraph.search_text_chunks("headquarters", top_k=2)
        for i, (chunk, score) in enumerate(text_chunks):
            preview = chunk.content[:60] + "..." if len(chunk.content) > 60 else chunk.content
            print(f"   {i+1}. {preview}")

        # 4. 智能问答演示
        print("\n💬 4. 智能问答演示...")
        questions = [
            "Where is Apple Inc. headquartered?",
            "Who founded Apple?",
            "What products does Apple develop?"
        ]

        for i, question in enumerate(questions):
            print(f"\n❓ 问题 {i+1}: {question}")
            try:
                # 流式调用
                async for chunk_data in await agraph.chat(question, stream=True):
                    if chunk_data["chunk"]:
                        print(chunk_data["chunk"], end="", flush=True)
                    if chunk_data["finished"]:
                        print(f"\n   📊 检索了 {len(chunk_data['context'].get('entities', []))} 个实体")
                        break
            except Exception as e:
                print(f"🤖 回答: 抱歉，无法回答这个问题: {e}")

        # 5. 系统信息
        print("\n📊 5. 系统信息...")
        stats = await agraph.get_stats()
        if 'vector_store' in stats:
            vs_stats = stats['vector_store']
            print("向量存储:")
            print(f"   - 实体: {vs_stats.get('entities', 0)}")
            print(f"   - 关系: {vs_stats.get('relations', 0)}")
            print(f"   - 文本块: {vs_stats.get('text_chunks', 0)}")

    print("\n✅ 快速开始演示完成!")

if __name__ == "__main__":
    asyncio.run(main())
```

### REST API Server

AGraph provides a comprehensive REST API server for web applications:

```bash
# Install API dependencies
uv add "agraph[api]"

# Start development server
agraph-server --host 0.0.0.0 --port 8000 --reload

# Start production server
agraph-server --host 0.0.0.0 --port 8000 --workers 4
```

#### API Server Features

- **FastAPI-based**: Modern, fast, and auto-documented REST API
- **Modular Architecture**: Router-based design for maintainability
- **Auto-documentation**: Swagger UI at `/docs` and ReDoc at `/redoc`
- **Streaming Support**: Real-time streaming chat responses
- **Cache Inspection**: View cached entities, relations, and clusters
- **Production Ready**: Gunicorn support for production deployment

#### API Endpoints

- **Configuration**: `GET/POST /config` - Manage system configuration
- **Document Processing**: `POST /documents/upload`, `POST /documents/from-text`
- **Knowledge Graph**: `POST /knowledge-graph/build` - Build graphs from data
- **Search**: `POST /search` - Search entities, relations, and text chunks
- **Chat**: `POST /chat`, `POST /chat/stream` - RAG-based conversations
- **Cache**: `GET /cache/{type}` - View cached data with pagination
- **System**: `GET /system/stats`, `POST /system/clear-all` - System management

#### Server Options

```bash
agraph-server --help                    # Show all options
agraph-server --host 0.0.0.0           # Bind to all interfaces
agraph-server --port 8080              # Custom port
agraph-server --reload                 # Auto-reload for development
agraph-server --workers 4              # Multiple workers for production
agraph-server --log-level debug        # Set logging level
agraph-server --env-file .env          # Load environment variables
```

#### Production Deployment

```bash
# Install production dependencies
uv add "agraph[server]"

# Run with Gunicorn (production)
gunicorn -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 agraph.api.app:app

# Or use the built-in production command
python -c "from agraph.api.server import run_with_gunicorn; run_with_gunicorn()"
```

### Environment Configuration

Create a `.env` file to configure OpenAI API and other settings:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo

# Workdir Configuration (optional)
AGRAPH_WORKDIR=./workdir/agraph-cache
```

### Configuration Management

AGraph provides persistent configuration management:

```python
from agraph.config import update_settings, save_config_to_workdir, get_settings

# Update settings programmatically
update_settings({"workdir": "./custom/workdir"})

# Save configuration to workdir for persistence
config_path = save_config_to_workdir()
print(f"配置已保存到: {config_path}")

# Get current settings
settings = get_settings()
print(f"当前工作目录: {settings.workdir}")
```

## Detailed Example

See `examples/agraph_quickstart.py` for a comprehensive example featuring the new Pipeline architecture:

### Pipeline Architecture Features
- **83% Complexity Reduction**: Streamlined processing pipeline
- **Intelligent Caching**: Smart caching with error recovery
- **Performance Monitoring**: Detailed metrics and timing information
- **Enhanced Error Handling**: Robust error recovery mechanisms

### Example Features Demonstrated
- Reading multiple document formats (`.txt`, `.md`, `.json`, `.csv`)
- Pipeline-based knowledge graph construction
- Semantic search for entities and text chunks
- Streaming Q&A conversations with context retrieval
- System statistics and performance monitoring
- Configuration management with persistent settings

### Pipeline Processing Steps
1. **Text Chunking** (文本分块): Intelligent text segmentation
2. **Entity Extraction** (实体提取): Advanced entity recognition
3. **Relation Extraction** (关系提取): Relationship identification
4. **Clustering** (聚类): Entity and relation clustering
5. **Graph Assembly** (组装): Final knowledge graph construction

Run the example:

```bash
# Ensure there are text files in examples/documents/ directory
python examples/agraph_quickstart.py
```

### Sample Output
```
🚀 AGraph 快速开始示例
========================================
📦 1. 初始化AGraph (Pipeline架构)...
   🏗️ 使用新的Pipeline架构 (83%复杂度降低)
   ⚡ 智能缓存和错误恢复
   📊 详细的性能监控和指标
✅ AGraph初始化成功 (内部使用Pipeline架构)

🏗️ 2. 构建知识图谱 (Pipeline架构)...
   📋 Pipeline步骤: 文本分块 → 实体提取 → 关系提取 → 聚类 → 组装
✅ 知识图谱构建成功!
   ⏱️ 构建时间: 2.34秒 (Pipeline优化)
   📊 实体: 15 个
   🔗 关系: 8 个
   📄 文本块: 5 个
```

## Development

### Development Commands

```bash
# Run tests
make test

# Format code
make format

# Lint code
make lint

# Build package
make build

# Clean build files
make clean
```

### Code Quality

The project follows strict code quality standards:

- **Black** code formatting (line length 100)
- **isort** import sorting
- **mypy** type checking
- **flake8** style checking
- **pylint** code quality analysis

## Dependencies

### Core Dependencies

- `chromadb`: Vector database
- `openai`: OpenAI API client
- `pydantic`: Data validation and settings management
- `networkx`: Graph data structures
- `loguru`: Logging management
- `tiktoken`: Text tokenization

### Optional Dependency Groups

- `vectordb`: ChromaDB vector database support
- `jupyter`: Jupyter environment support
- `dev`: Development tools and testing framework
- `docs`: Documentation generation tools

## License

MIT License

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

- [Documentation](https://agraph.readthedocs.io)
- [Issue Tracker](https://github.com/JasonDZS/agraph/issues)
- [Discussions](https://github.com/JasonDZS/agraph/discussions)
