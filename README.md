# Knowledge Graph Module (AGraph)

> 🤖 Intelligent Knowledge Graph Construction and Q&A Tool Based on LLM

AGraph is a powerful Python toolkit specifically designed for automatically building knowledge graphs from various documents and providing intelligent Q&A functionality. It supports multiple document formats, integrates the LightRAG framework, and provides advanced features such as semantic search and vector storage.

## ✨ Feature Overview

- 🧠 **Intelligent Construction**: Automatic entity and relation extraction based on LLM
- 📚 **Multi-format Support**: PDF, Word, HTML, JSON, images, etc.
- 🔍 **Semantic Search**: Intelligent retrieval based on vector similarity
- 💾 **Flexible Storage**: Support for JSON, Neo4j, and other storage solutions
- 🚀 **High Performance**: Support for incremental updates and batch processing
- 🔌 **LightRAG Integration**: Deep integration with advanced knowledge graph framework

## 📋 System Requirements

- **Python**: 3.10+
- **Operating System**: Windows, macOS, Linux

## 📖 Documentation

Comprehensive documentation is available at: **https://jasondizs.github.io/agraph/**

- [Getting Started Guide](https://jasondizs.github.io/agraph/getting_started.html)
- [API Reference](https://jasondizs.github.io/agraph/api/modules.html)
- [Examples](https://jasondizs.github.io/agraph/examples.html)
- [Changelog](https://jasondizs.github.io/agraph/changelog.html)

### Building Documentation Locally

```bash
# Install documentation dependencies
uv sync --extra docs

# Build documentation
make docs

# View documentation
open docs/build/index.html
```

## 🛠️ Quick Installation

### Basic Installation

```bash
# Install from source (recommended for development)
git clone <repository-url>
cd agraph
uv sync --python 3.12

# Or install directly
pip install -e .
```

### Environment Variable Setup

```bash
cp .env.example .env
```
Edit the `.env` file to set your model address and API key

### Optional Dependencies Installation

Install corresponding dependencies based on the document types you need to process:

```bash
# Document processing
pip install beautifulsoup4  # HTML processing
pip install pypdf          # PDF processing
pip install python-docx    # Word documents
pip install pandas         # Excel/CSV processing

# Image processing and OCR
pip install pillow pytesseract
```
## 🚀 Quick Start

### 1. Basic Knowledge Graph Construction

```python
import os
import asyncio
from agraph.builders import LLMGraphBuilder
from agraph.storage import JsonVectorStorage
from agraph.config import settings
from agraph import ChatKnowledgeRetriever

settings.workdir = "./workdir/llm_builder_example"  # Set working directory
os.makedirs(settings.workdir, exist_ok=True)  # Ensure working directory exists

async def build_knowledge_graph():
    # Create graph builder
    builder = LLMGraphBuilder(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        llm_model=settings.LLM_MODEL,  # Specify LLM model
        embedding_model=settings.EMBEDDING_MODEL,  # Specify embedding model
        vector_storage=JsonVectorStorage(),  # Use JSON vector storage
        temperature=0.1
    )

    # Build knowledge graph from text
    texts = [
        "Apple Inc. is a technology company founded by Steve Jobs.",
        "iPhone is Apple's flagship smartphone product.",
        "Steve Jobs served as CEO of Apple until 2011."
    ]

    graph = await builder.build_graph(texts=texts, graph_name="Technology Company")

    print(f"Built knowledge graph with {len(graph.entities)} entities and {len(graph.relations)} relations")

    # Print entity information for debugging
    print("\n=== Built Entities ===")
    for entity_id, entity in graph.entities.items():
        print(f"- {entity.name} ({entity.entity_type.value}): {entity.description}")

    # Print relation information for debugging
    print(f"\n=== Built Relations ({len(graph.relations)}) ===")
    for relation_id, relation in graph.relations.items():
        if relation.head_entity and relation.tail_entity:
            print(f"- {relation.head_entity.name} --{relation.relation_type.value}--> {relation.tail_entity.name}: {relation.description}")
    return graph, builder

# Run example
if __name__ == "__main__":
    graph, builder = asyncio.run(build_knowledge_graph())
    # Use the built knowledge graph and builder
    retriever = ChatKnowledgeRetriever()
    result = retriever.chat("Who founded Apple Inc.?")

    print("\n=== Retrieval Results Debug ===")
    print(f"Retrieved entities count: {len(result.get('entities', []))}")
    print(f"Retrieved relations count: {len(result.get('relations', []))}")
    if result.get('entities'):
        print("Retrieved entities:")
        for entity in result['entities']:
            if hasattr(entity, 'entity'):
                print(f"  - {entity.entity.name}: score={entity.score}")

    print(f"\n=== Final Result ===")
    print(result)
```

### 2. Document Processing Example

```python
import os
import asyncio
from agraph.builders import LLMGraphBuilder
from agraph.storage import JsonVectorStorage
from agraph.config import settings
from agraph.processer.factory import DocumentProcessorFactory
from agraph.retrieval import ChatKnowledgeRetriever

settings.workdir = "./workdir/llm_builder_folder"  # Set working directory
os.makedirs(settings.workdir, exist_ok=True)  # Ensure working directory exists

async def build_knowledge_graph():
    # Create graph builder
    builder = LLMGraphBuilder(
        openai_api_key=settings.OPENAI_API_KEY,
        openai_api_base=settings.OPENAI_API_BASE,
        llm_model=settings.LLM_MODEL,  # Specify LLM model
        embedding_model=settings.EMBEDDING_MODEL,  # Specify embedding model
        vector_storage=JsonVectorStorage(),  # Use JSON vector storage
        temperature=0.1
    )

    document_paths = [
        "./examples/documents/company_info.txt",
        "./examples/documents/products.json",
        "./examples/documents/team.html",
        "./examples/documents/research_papers.csv",
        "./examples/documents/technology_stack.md"
    ]

    texts = []
    processor_factory = DocumentProcessorFactory()

    for doc_path in document_paths:
        processor = processor_factory.get_processor(doc_path)
        content = processor.process(doc_path)
        texts.append(f"Document: {doc_path}\n{content}")

    graph = await builder.build_graph(texts=texts, graph_name="Technology Company")

    print(f"Built knowledge graph with {len(graph.entities)} entities and {len(graph.relations)} relations")
    return graph, builder

# Run example
if __name__ == "__main__":
    graph, builder = asyncio.run(build_knowledge_graph())
    # Use the built knowledge graph and builder
    retriever = ChatKnowledgeRetriever()
    result = retriever.chat("Company headquarters location")
    print(result)
```

## 📚 Main Features

### 🏗️ Knowledge Graph Construction
- **Intelligent Entity Recognition**: Automatic entity and relation extraction based on LLM
- **Multi-format Support**: PDF, Word, HTML, JSON, CSV and other document types
- **Incremental Updates**: Support for dynamically adding new documents to existing graphs
- **Vector Storage**: Support for semantic similarity search
- **Multiple Builders**: Provides different functional builders like MinimalLLMGraphBuilder, FlexibleLLMGraphBuilder

### 🔍 Knowledge Q&A Retrieval
- **Semantic Search**: Intelligent search based on vector similarity
- **Entity Queries**: Find related entities and their properties
- **Relation Exploration**: Discover complex relationships between entities
- **Intelligent Q&A**: Dedicated KnowledgeRetriever provides Q&A functionality
- **Multiple Search Modes**: Support for entity search, relation search, and comprehensive search

### 💾 Flexible Storage Solutions
- **JSON Storage**: Lightweight file storage, suitable for small-scale applications
- **Neo4j Storage**: Enterprise-level graph database, supports complex queries
- **Vector Storage**: JsonVectorStorage supports efficient similarity search
- **LightRAG Integration**: Support for GraphML format and LightRAG working directory structure

## 🔧 Environment Configuration

### Install Dependencies

```bash
# Development installation (recommended)
make install-dev

# Or install directly
pip install -e .

# Optional dependencies (install as needed)
pip install beautifulsoup4  # HTML processing
pip install pypdf          # PDF processing
pip install python-docx    # Word document processing
pip install pandas         # Excel/CSV processing
pip install pillow          # Image processing
pip install pytesseract     # OCR functionality
```

### API Key Setup

```bash
# Set OpenAI API key (required)
export OPENAI_API_KEY="your-openai-api-key"

# Optional: Custom API address
export OPENAI_API_BASE="https://api.openai.com/v1"
```

## 📖 More Examples

Check the `examples/` directory for more complete examples:

- **Basic Functionality**: `llm_builder_example.py` - Demonstrates usage of various LLM builders
- **LightRAG Integration**: `lightrag_example.py` - LightRAG builder usage example
- **Document Processing**: `llm_builder_folder.py` - Batch document processing example
- **Sample Documents**: `documents/` - Contains sample documents in various formats

## ⚡ Core Advantages

- **🤖 Intelligence**: LLM-based automatic entity-relation extraction, no manual rules needed
- **🔍 Semantics**: Support for vector similarity search, understanding semantics beyond keyword matching
- **📄 Multi-format**: Automatic processing of PDF, Word, HTML and other document formats
- **⚡ High Performance**: Support for incremental updates and batch processing, suitable for large-scale applications
- **🔧 Extensible**: Modular design, support for custom builders and storage backends
- **🏗️ SOLID Design**: Strictly follows SOLID principles, provides dedicated builders and retrievers
- **🔌 LightRAG Integration**: Deep integration with LightRAG framework, supports advanced knowledge graph features

## 📝 Important Notes

1. **API Costs**: Using OpenAI API will incur costs, recommend testing with small datasets first
2. **Network Connection**: Building graphs requires stable network connection to access LLM services
3. **Memory Usage**: Large-scale documents may require significant memory, recommend batch processing
4. **Asynchronous Programming**: All building and search operations are asynchronous, need to use `asyncio.run()`

## 🆘 Quick Troubleshooting

- **Installation Issues**: Run `make install-dev` or `pip install -e .`
- **API Key**: Ensure you have set a valid `OPENAI_API_KEY`
- **Document Processing Failure**: Install corresponding optional dependency packages
- **Memory Insufficient**: Reduce the number or size of documents processed at once
- **Run Tests**: Use `make test` to run all tests
- **Code Check**: Use `make check` for code quality checking

---

🚀 **Start building your knowledge graph!** Begin with simple text and gradually explore more advanced features.
