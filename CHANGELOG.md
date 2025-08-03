# v0.1.0

**Release Date:** 2025-01-XX

Initial release of AGraph - A comprehensive toolkit for building, storing and querying knowledge graphs.

## 🌟 Features

### Core Components
- **Entity System** - Robust entity data structures with type definitions and operations
- **Relations Framework** - Comprehensive relationship modeling and management
- **Knowledge Graph Core** - Central graph data structure for organizing and querying knowledge

### Document Processing
- **Multi-format Support** - Process PDF, Word, Text, HTML, Spreadsheet, JSON, and Image files
- **Factory Pattern Architecture** - Extensible processor factory for adding new document types
- **Batch Processing** - Efficient handling of multiple documents and folders

### Graph Builders
- **StandardGraphBuilder** - Rule-based entity and relation extraction
- **LightRAGGraphBuilder** - LLM-powered graph building using the LightRAG framework
- **MultiSourceGraphBuilder** - Integration of multiple data sources into unified graphs

### Storage Backends
- **JsonStorage** - File-based JSON storage for lightweight deployments
- **Neo4jStorage** - Neo4j graph database integration for enterprise applications
- **Abstract Storage Interface** - Easy extension for custom storage backends

### Extractors
- **Entity Extractors** - Extract entities from text and database sources
- **Relation Extractors** - Identify and extract relationships between entities
- **Extensible Architecture** - Base interfaces for implementing custom extractors

### Embeddings
- **Graph Embeddings** - Node2Vec and TransE algorithms for graph representation learning
- **Entity Similarity** - Compute semantic similarity between entities
- **Vector Operations** - Support for downstream ML tasks

### LightRAG Integration
- **LLM-powered Extraction** - Automatic entity/relation extraction using language models
- **GraphML Export** - Standard format support for visualization tools
- **Multiple Search Modes** - Naive, local, global, and hybrid search capabilities
- **Incremental Updates** - Add documents to existing graphs without rebuilding

## 🔧 Developer Experience

### API Design
- **Convenience Functions** - One-line graph creation with `quick_build_graph()`
- **Factory Methods** - Simple creation of builders and storage instances
- **Type Safety** - Full type annotations for better IDE support

### Testing & Quality
- **Comprehensive Test Suite** - 95%+ code coverage across all modules
- **Quality Tools** - Black, isort, mypy, flake8, bandit integration
- **CI/CD Ready** - Pre-commit hooks and automated quality checks

### Documentation
- **Usage Examples** - Complete examples for common workflows
- **API Documentation** - Detailed documentation for all public APIs
- **Architecture Guide** - Clear explanation of design patterns and extensibility

## 📦 Installation

```bash
pip install agraph
```

### Optional Dependencies

```bash
# Development tools
pip install agraph[dev]

# Documentation tools
pip install agraph[docs]

# All dependencies
pip install agraph[all]
```

## 🚀 Quick Start

```python
from agraph import quick_build_graph, create_json_storage

# Build a knowledge graph
graph = quick_build_graph(
    texts=["Knowledge graphs represent structured information"],
    graph_name="my_graph"
)

# Save to JSON storage
storage = create_json_storage("./graphs")
storage.save_graph(graph)
```

## 🏗️ Architecture

- **Factory Pattern** - Consistent object creation across the framework
- **Strategy Pattern** - Multiple algorithms for graph building and processing
- **Plugin Architecture** - Easy extension with custom components
- **Abstract Base Classes** - Well-defined interfaces for all major components

## 📋 Requirements

- **Python** - 3.10+ required
- **Core Dependencies** - BeautifulSoup4, LightRAG, OpenAI, PyPDF, Neo4j driver
- **Optional Dependencies** - Development and documentation tools

## 🛡️ Stability

This is the initial release (v0.1.0) of AGraph. While the core functionality is stable and well-tested, the API may evolve based on user feedback. We recommend pinning to specific versions in production environments.
