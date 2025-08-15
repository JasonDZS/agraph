# API Separation Guide

## Overview

The AGraph API has been restructured to separate document management from knowledge graph
construction, providing better modularity and flexibility.

## Architecture Changes

### Before: Coupled Approach

```text
Upload Files → Process Immediately → Build Knowledge Graph
```

### After: Separated Approach

```text
Upload Documents → Store in Document Manager
                ↓
Retrieve Document IDs → Build Knowledge Graph
```

## API Endpoints

### Document Management (`/documents`)

#### Upload Files

```http
POST /documents/upload
Content-Type: multipart/form-data

files: [file1.txt, file2.txt]
metadata: {"source": "example"}
tags: ["important", "research"]
```

#### Upload Text Directly

```http
POST /documents/from-text
Content-Type: application/json

{
  "texts": ["Text content 1", "Text content 2"],
  "metadata": {"source": "direct"},
  "tags": ["text", "direct"]
}
```

#### List Documents

```http
GET /documents/list?page=1&page_size=10&tag_filter=["research"]&search_query=python
```

#### Get Specific Document

```http
GET /documents/{doc_id}
```

#### Delete Documents

```http
DELETE /documents/delete
Content-Type: application/json

{
  "document_ids": ["doc_123", "doc_456"]
}
```

#### Get Storage Statistics

```http
GET /documents/stats/summary
```

### Knowledge Graph Construction (`/knowledge-graph`)

#### Build from Stored Documents

```http
POST /knowledge-graph/build
Content-Type: application/json

{
  "document_ids": ["doc_123", "doc_456"],
  "graph_name": "My Knowledge Graph",
  "graph_description": "Built from stored documents",
  "use_cache": true,
  "save_to_vector_store": true
}
```

#### Build from Direct Text

```http
POST /knowledge-graph/build
Content-Type: application/json

{
  "texts": ["Direct text content"],
  "graph_name": "Direct Text Graph",
  "use_cache": true
}
```

#### Update Existing Knowledge Graph

```http
POST /knowledge-graph/update
Content-Type: application/json

{
  "additional_document_ids": ["doc_789"],
  "additional_texts": ["New content"],
  "use_cache": true
}
```

#### Get Knowledge Graph Status

```http
GET /knowledge-graph/status
```

## Benefits of Separation

### 1. **Reusability**

- Documents are stored once and can be used in multiple knowledge graphs
- No need to re-upload documents for different graph configurations

### 2. **Flexibility**

- Build knowledge graphs from any combination of stored documents
- Support both stored documents and direct text in the same request

### 3. **Scalability**

- Document storage is independent of knowledge graph operations
- Better memory management and resource utilization

### 4. **Maintainability**

- Clear separation of concerns
- Easier to test and debug individual components

## Workflow Examples

### Basic Workflow

```python
import httpx

async with httpx.AsyncClient() as client:
    # 1. Upload documents
    upload_response = await client.post("/documents/from-text", json={
        "texts": ["Sample text content"],
        "tags": ["example"]
    })
    doc_ids = [doc["id"] for doc in upload_response.json()["data"]["uploaded_documents"]]

    # 2. Build knowledge graph
    kg_response = await client.post("/knowledge-graph/build", json={
        "document_ids": doc_ids,
        "graph_name": "Example Graph"
    })
```

### Advanced Workflow

```python
# 1. Upload multiple document types
file_upload = await client.post("/documents/upload", files=[
    ("files", ("doc1.txt", open("doc1.txt", "rb"), "text/plain")),
    ("files", ("doc2.txt", open("doc2.txt", "rb"), "text/plain"))
])

text_upload = await client.post("/documents/from-text", json={
    "texts": ["Additional direct text"],
    "metadata": {"source": "direct"}
})

# 2. Get all document IDs
all_docs = await client.get("/documents/list?page_size=100")
doc_ids = [doc["id"] for doc in all_docs.json()["data"]["documents"]]

# 3. Build comprehensive knowledge graph
kg_build = await client.post("/knowledge-graph/build", json={
    "document_ids": doc_ids,
    "graph_name": "Comprehensive Graph",
    "use_cache": True
})

# 4. Later, add more documents and update
new_upload = await client.post("/documents/from-text", json={
    "texts": ["New information to add"]
})

new_doc_ids = [doc["id"] for doc in new_upload.json()["data"]["uploaded_documents"]]

kg_update = await client.post("/knowledge-graph/update", json={
    "additional_document_ids": new_doc_ids
})
```

## Migration Guide

### From Coupled API

**Old approach:**

```python
# Everything in one request
response = await client.post("/build-knowledge-graph", files=[...])
```

**New approach:**

```python
# Step 1: Upload documents
upload_response = await client.post("/documents/upload", files=[...])
doc_ids = [doc["id"] for doc in upload_response.json()["data"]["uploaded_documents"]]

# Step 2: Build knowledge graph
kg_response = await client.post("/knowledge-graph/build", json={
    "document_ids": doc_ids
})
```

## Document Storage

Documents are stored in the local filesystem with the following structure:

```text
./document_storage/
├── documents/           # Actual document content
│   ├── doc_123.txt
│   └── doc_456.txt
├── metadata/           # Document metadata
│   ├── doc_123.json
│   └── doc_456.json
└── index.json         # Document index for quick lookup
```

## Error Handling

### Document Errors

- `400`: Invalid file format or missing content
- `404`: Document not found
- `500`: Storage system error

### Knowledge Graph Errors

- `400`: No documents provided or invalid parameters
- `404`: Referenced documents not found
- `500`: Knowledge graph construction error

## Performance Considerations

1. **Document Storage**: O(1) for retrieval by ID
2. **Pagination**: Efficient for large document collections
3. **Filtering**: Index-based filtering for tags and metadata
4. **Caching**: Knowledge graph construction benefits from document reuse

## Security Notes

- Documents are stored locally and not exposed directly through API
- Document IDs are generated securely and are unpredictable
- File uploads are validated for content type and size
