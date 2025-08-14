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
pip install agraph
```

### Development Installation

```bash
# Clone repository
git clone https://github.com/JasonDZS/agraph.git
cd agraph

# Install with development dependencies
make install-dev
# or
pip install -e .[dev,docs]
```

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

### Optional Dependencies

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
