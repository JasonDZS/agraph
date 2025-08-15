# AGraph

AGraph is a modern knowledge graph toolkit for building, managing, and querying knowledge graphs.
It provides a clean API for handling entities and relations, with integrated vector database support
for semantic search and intelligent Q&A.

## Features

- **Easy to Use**: Intuitive API design for quick adoption
- **Smart Construction**: Automatically extract entities and relations from text
- **Semantic Search**: Vector similarity-based intelligent search
- **Smart Q&A**: RAG-based conversational system using knowledge graphs
- **Type Safe**: Complete type annotations and validation
- **Streaming**: Support for streaming conversations and incremental updates
- **Multiple Storage**: Support for ChromaDB and other vector databases

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
import asyncio
from agraph import AGraph, get_settings

settings = get_settings()
settings.workdir = "./workdir/demo"  # Set working directory for storage

async def main():
    # Create AGraph instance
    async with AGraph(
        collection_name="my_knowledge_graph",
        vector_store_type="chroma",
        persist_directory = settings.workdir,
        use_openai_embeddings=True
    ) as agraph:
        # Initialize
        await agraph.initialize()

        # Build knowledge graph from texts
        texts = [
            "Apple Inc. is an American multinational technology company.",
            "Apple Inc. is headquartered in Cupertino, California.",
            "Apple Inc. develops iPhone and iPad products."
        ]

        knowledge_graph = await agraph.build_from_texts(
            texts=texts,
            graph_name="Tech Company Knowledge Graph",
            graph_description="Basic information about technology companies"
        )

        print(f"Built: {len(knowledge_graph.entities)} entities, {len(knowledge_graph.relations)} relations")

        # Semantic search
        entities = await agraph.search_entities("technology company", top_k=5)
        for entity, score in entities:
            print(f"Entity: {entity.name} ({entity.entity_type})")

        # Smart Q&A
        async for chunk_data in await agraph.chat("Where is Apple Inc. headquartered?", stream=True):
            if chunk_data["chunk"]:
                print(chunk_data["chunk"], end="", flush=True)
            if chunk_data["finished"]:
                break

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

Create a `.env` file to configure OpenAI API:

```env
OPENAI_API_KEY=your_openai_api_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
```

## Detailed Example

See `examples/agraph_quickstart.py` for a complete quick start example, including:

- Reading text files from documents directory
- Building knowledge graphs
- Semantic search demonstration
- Smart Q&A conversations
- System statistics

Run the example:

```bash
# Ensure there are text files in examples/documents/ directory
python examples/agraph_quickstart.py
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
