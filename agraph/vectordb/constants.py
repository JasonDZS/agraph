"""
Constants and configuration values for vector database operations.

This module centralizes all magic numbers and configuration values used
throughout the vector database implementation.
"""

from typing import Any, Dict

# Embedding Configuration
DEFAULT_EMBEDDING_DIMENSION = 128
OPENAI_EMBEDDING_DIMENSION = 1536
MAX_TEXT_LENGTH_FOR_EMBEDDING = 8192

# Batch Processing
DEFAULT_BATCH_SIZE = 100
MAX_BATCH_SIZE = 1000
DEFAULT_MAX_CONCURRENCY = 10
MAX_CONCURRENCY_LIMIT = 50

# Search Configuration
DEFAULT_TOP_K = 10
MAX_TOP_K = 1000
MIN_SIMILARITY_THRESHOLD = 0.0
MAX_SIMILARITY_THRESHOLD = 1.0

# Cache Configuration
DEFAULT_CACHE_SIZE = 1000
MAX_CACHE_SIZE = 10000

# Timeout Configuration (in seconds)
DEFAULT_REQUEST_TIMEOUT = 30.0
MAX_REQUEST_TIMEOUT = 300.0
DEFAULT_RETRY_COUNT = 3
MAX_RETRY_COUNT = 10

# ChromaDB Configuration
DEFAULT_CHROMA_HOST = "localhost"
DEFAULT_CHROMA_PORT = 8000
CHROMA_COLLECTION_NAME_PREFIX = "agraph"

# Content Length Limits (for metadata storage)
MAX_CONTENT_LENGTH_IN_METADATA = 1000
MAX_DESCRIPTION_LENGTH = 500
MAX_NAME_LENGTH = 100

# Collection Name Suffixes
COLLECTION_SUFFIXES: Dict[str, str] = {
    "entity": "_entities",
    "relation": "_relations",
    "cluster": "_clusters",
    "text_chunk": "_text_chunks",
}

# Error Messages
ERROR_MESSAGES: Dict[str, str] = {
    "not_initialized": "VectorStore is not initialized",
    "unknown_data_type": "Unknown data type: {data_type}",
    "invalid_embedding": "Invalid embedding for {object_type} {object_id}",
    "openai_not_available": (
        "OpenAI is not installed. " "Please install it with: pip install openai>=1.99.9"
    ),
    "chromadb_not_available": (
        "ChromaDB is not installed. Please install it with: pip install '.[vectordb]' "
        "or pip install chromadb>=0.5.0"
    ),
    "openai_api_key_required": "OpenAI API key is required",
    "chromadb_init_failed": "Failed to initialize ChromaDB: {error}",
    "openai_embedding_failed": "Failed to create OpenAI embedding function: {error}",
}

# Default Settings for different vector store types
MEMORY_STORE_DEFAULTS: Dict[str, Any] = {
    "collection_name": "knowledge_graph",
    "use_openai_embeddings": False,
}

CHROMA_STORE_DEFAULTS: Dict[str, Any] = {
    "collection_name": "knowledge_graph",
    "persist_directory": None,
    "host": None,
    "port": None,
    "use_openai_embeddings": False,
}

# Embedding Function Defaults
OPENAI_EMBEDDING_DEFAULTS: Dict[str, Any] = {
    "batch_size": DEFAULT_BATCH_SIZE,
    "max_concurrency": DEFAULT_MAX_CONCURRENCY,
    "max_retries": DEFAULT_RETRY_COUNT,
    "timeout": DEFAULT_REQUEST_TIMEOUT,
}

CACHED_EMBEDDING_DEFAULTS: Dict[str, Any] = {
    "cache_size": DEFAULT_CACHE_SIZE,
    **OPENAI_EMBEDDING_DEFAULTS,
}
