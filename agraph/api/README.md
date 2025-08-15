# AGraph FastAPI

AGraph FastAPI provides a comprehensive REST API for the AGraph knowledge graph construction
and RAG (Retrieval-Augmented Generation) system.

## Features

- **Configuration Management**: Get and update system configuration
- **Document Processing**: Upload and process various document types
- **Knowledge Graph Construction**: Build knowledge graphs from documents or texts
- **Vector Storage**: Store and retrieve embeddings
- **Search Capabilities**: Search entities, relations, and text chunks
- **RAG Chat System**: Interactive chat with knowledge base
- **Streaming Responses**: Real-time streaming chat responses
- **System Management**: Monitor stats, cache, and build status

## Quick Start

### 1. Install Dependencies

```bash
# Install with uv (recommended)
uv add "agraph[api]"        # For API server
uv add "agraph[server]"     # For production deployment
uv add "agraph[all]"        # For all features

# Or with pip
pip install "agraph[api]"   # For API server
pip install "agraph[server]" # For production deployment
```

### 2. Set Environment Variables

Create a `.env` file in your project root:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=https://api.openai.com/v1

# LLM Configuration
LLM_MODEL=gpt-3.5-turbo
LLM_PROVIDER=openai
LLM_MAX_TOKENS=4096

# Text Processing
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=100

# Embedding Configuration
EMBEDDING_MODEL=Pro/BAAI/bge-m3
EMBEDDING_PROVIDER=openai
EMBEDDING_DIM=1024
EMBEDDING_MAX_TOKENS=8192
EMBEDDING_BATCH_SIZE=32
```

### 3. Start the Server

```bash
# Method 1: Use the agraph-server command (recommended)
agraph-server --host 0.0.0.0 --port 8000 --reload

# Method 2: Use uvicorn directly
uvicorn agraph.api.app:app --reload --host 0.0.0.0 --port 8000

# Method 3: Production with multiple workers
agraph-server --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Access the API

- **API Documentation**: <http://localhost:8000/docs>
- **Alternative Documentation**: <http://localhost:8000/redoc>
- **Health Check**: <http://localhost:8000/health>

## API Architecture

The API is built using a modular router-based architecture following FastAPI best practices:

```text
agraph/api/
├── app.py              # Main FastAPI application
├── dependencies.py     # Shared dependencies and AGraph instance management
├── models.py           # Pydantic models and schemas
├── routers/           # Modular router implementations
│   ├── __init__.py
│   ├── config.py      # Configuration management
│   ├── documents.py   # Document upload and processing
│   ├── knowledge_graph.py  # Knowledge graph construction
│   ├── search.py      # Search functionality
│   ├── chat.py        # Chat and RAG functionality
│   ├── cache.py       # Cache data viewing
│   └── system.py      # System management
├── requirements.txt   # API dependencies
├── example.py         # Example usage and tests
└── README.md          # This documentation
```

## API Endpoints

### Health & Configuration

- `GET /health` - Health check
- `GET /config` - Get current configuration
- `POST /config` - Update configuration

### Document Processing

- `POST /documents/upload` - Upload and process files
- `POST /documents/from-text` - Process text directly

### Knowledge Graph

- `POST /knowledge-graph/build` - Build knowledge graph from documents or texts

### Search

- `POST /search` - Search entities, relations, or text chunks

### Chat (RAG)

- `POST /chat` - Chat with knowledge base (non-streaming)
- `POST /chat/stream` - Chat with knowledge base (streaming)

### Cache Data Viewing

- `GET /cache/text-chunks` - View cached text chunks with pagination
- `GET /cache/entities` - View cached entities with pagination
- `GET /cache/relations` - View cached relations with pagination
- `GET /cache/clusters` - View cached clusters with pagination

### System Management

- `GET /system/stats` - Get system statistics
- `GET /system/build-status` - Get build status
- `GET /system/cache-info` - Get cache information
- `POST /system/clear-cache` - Clear cache
- `POST /system/clear-all` - Clear all system data

## Usage Examples

### 1. Upload Documents

```python
import httpx

async with httpx.AsyncClient() as client:
    with open("document.pdf", "rb") as f:
        files = {"files": ("document.pdf", f, "application/pdf")}
        data = {
            "graph_name": "My Knowledge Graph",
            "graph_description": "Knowledge from my documents",
            "use_cache": "true",
            "save_to_vector_store": "true"
        }

        response = await client.post(
            "http://localhost:8000/documents/upload",
            files=files,
            data=data
        )
        print(response.json())
```

### 2. Process Texts

```python
import httpx

async with httpx.AsyncClient() as client:
    data = {
        "texts": [
            "Apple Inc. is a technology company.",
            "Microsoft was founded by Bill Gates."
        ],
        "graph_name": "Tech Companies",
        "use_cache": True,
        "save_to_vector_store": True
    }

    response = await client.post(
        "http://localhost:8000/documents/from-text",
        json=data
    )
    print(response.json())
```

### 3. Search Knowledge Base

```python
import httpx

async with httpx.AsyncClient() as client:
    data = {
        "query": "technology companies",
        "top_k": 5,
        "search_type": "entities"  # or "relations", "text_chunks"
    }

    response = await client.post(
        "http://localhost:8000/search",
        json=data
    )
    print(response.json())
```

### 4. Chat with Knowledge Base

```python
import httpx

async with httpx.AsyncClient() as client:
    data = {
        "question": "Tell me about technology companies",
        "entity_top_k": 5,
        "relation_top_k": 5,
        "text_chunk_top_k": 5,
        "response_type": "详细回答",
        "stream": False
    }

    response = await client.post(
        "http://localhost:8000/chat",
        json=data
    )
    print(response.json())
```

### 5. Streaming Chat

```python
import httpx
import json

async with httpx.AsyncClient() as client:
    data = {
        "question": "Explain artificial intelligence",
        "stream": True
    }

    async with client.stream(
        "POST",
        "http://localhost:8000/chat/stream",
        json=data
    ) as response:
        async for chunk in response.aiter_text():
            if chunk.strip() and chunk.startswith("data: "):
                data = json.loads(chunk[6:])
                print(data.get("chunk", ""), end="", flush=True)
                if data.get("finished"):
                    break
```

### 6. View Cached Data

```python
import httpx

async with httpx.AsyncClient() as client:
    # View text chunks with pagination and filtering
    response = await client.get(
        "http://localhost:8000/cache/text-chunks?page=1&page_size=5&filter_by=technology"
    )
    result = response.json()
    print(f"Found {result['data']['total_count']} text chunks")
    for chunk in result['data']['text_chunks']:
        print(f"Chunk {chunk['id']}: {chunk['content'][:100]}...")

    # View entities
    response = await client.get(
        "http://localhost:8000/cache/entities?page=1&page_size=10"
    )
    result = response.json()
    for entity in result['data']['entities']:
        print(f"Entity: {entity['name']} ({entity['entity_type']})")
        print(f"  Confidence: {entity['confidence']}")
        print(f"  Description: {entity['description'] or 'No description'}")

    # View relations
    response = await client.get(
        "http://localhost:8000/cache/relations?page=1&page_size=10"
    )
    result = response.json()
    for relation in result['data']['relations']:
        head_name = relation['head_entity']['name'] if relation['head_entity'] else 'Unknown'
        tail_name = relation['tail_entity']['name'] if relation['tail_entity'] else 'Unknown'
        print(f"Relation: {head_name} --[{relation['relation_type']}]--> {tail_name}")

    # View clusters
    response = await client.get(
        "http://localhost:8000/cache/clusters?page=1&page_size=5"
    )
    result = response.json()
    for cluster in result['data']['clusters']:
        print(f"Cluster: {cluster['name']}")
        print(f"  Entities: {cluster['entity_count']}, Relations: {cluster['relation_count']}")
```

## Testing

Run the example test script:

```bash
python agraph/api/example.py
```

This will test various API endpoints and demonstrate usage patterns.

## Configuration Options

The API supports various configuration options through environment variables:

- **OpenAI Settings**: API key, base URL, model selection
- **Text Processing**: Chunk size, overlap settings
- **Knowledge Graph**: Entity/relation confidence thresholds
- **Vector Storage**: Embedding model, dimensions, batch size
- **Cache**: TTL settings, cleanup options

## Error Handling

The API provides comprehensive error handling with structured error responses:

```json
{
  "status": "error",
  "message": "Error description",
  "timestamp": "2025-01-14T10:30:00",
  "error_code": "400",
  "error_details": {
    "exception": "Detailed error information"
  }
}
```

## Modular Architecture Benefits

The router-based architecture provides several advantages:

- **Separation of Concerns**: Each router handles a specific domain
- **Maintainability**: Easy to maintain and extend individual modules
- **Testing**: Easier to unit test individual routers
- **Scalability**: Simple to add new functionality or modify existing ones
- **Code Organization**: Clean and logical code structure
- **Dependency Injection**: Shared dependencies through FastAPI's DI system

## Performance Considerations

- **Caching**: Enable caching for better performance with repeated operations
- **Batch Processing**: Use batch operations for multiple documents
- **Streaming**: Use streaming chat for better user experience with long responses
- **Background Tasks**: Long-running operations are handled asynchronously
- **Router Isolation**: Each router can be optimized independently

## Security Notes

- Ensure proper API key management
- Use HTTPS in production
- Implement authentication/authorization as needed
- Validate file uploads and size limits
- Consider rate limiting for production use
- Router-level middleware can be added for specific security requirements

## Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["uvicorn", "agraph.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t agraph-api .
docker run -p 8000:8000 -e OPENAI_API_KEY=your_key agraph-api
```
