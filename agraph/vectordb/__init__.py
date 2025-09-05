"""
VectorDB module for knowledge graph vector storage.

This module provides vector storage implementations for storing and retrieving
entities, relations, clusters, and text chunks in vector databases.
"""

from .exceptions import VectorStoreError
from .factory import VectorStoreFactory, VectorStoreType, create_chroma_store, create_memory_store
from .interfaces import ClusterStore, EntityStore, RelationStore, TextChunkStore, VectorStore, VectorStoreCore
from .memory import MemoryVectorStore

# Embedding functions
try:
    from .embeddings import (
        CachedOpenAIEmbeddingFunction,
        ChromaEmbeddingFunction,
        EmbeddingFunctionError,
        OpenAIEmbeddingFunction,
        create_openai_embedding_function,
    )

    _EMBEDDINGS_AVAILABLE = True
except ImportError:
    _EMBEDDINGS_AVAILABLE = False
    OpenAIEmbeddingFunction = None  # type: ignore
    CachedOpenAIEmbeddingFunction = None  # type: ignore
    create_openai_embedding_function = None  # type: ignore
    ChromaEmbeddingFunction = None  # type: ignore
    EmbeddingFunctionError = None  # type: ignore

# ChromaDB is optional dependency
try:
    from .chroma import ChromaVectorStore

    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    ChromaVectorStore = None  # type: ignore

__all__ = [
    # Exceptions
    "VectorStoreError",
    # Interfaces
    "VectorStore",
    "VectorStoreCore",
    "EntityStore",
    "RelationStore",
    "ClusterStore",
    "TextChunkStore",
    # Implementations
    "MemoryVectorStore",
    # Factory
    "VectorStoreFactory",
    "VectorStoreType",
    "create_memory_store",
    "create_chroma_store",
]

# Add embeddings if available
if _EMBEDDINGS_AVAILABLE:
    __all__.extend(
        [
            "OpenAIEmbeddingFunction",
            "CachedOpenAIEmbeddingFunction",
            "create_openai_embedding_function",
            "ChromaEmbeddingFunction",
            "EmbeddingFunctionError",
        ]
    )

# Only export ChromaVectorStore if available
if _CHROMA_AVAILABLE:
    __all__.append("ChromaVectorStore")
