"""
Vector database exceptions.

This module defines exceptions used throughout the vector database implementation.
"""


class VectorStoreError(Exception):
    """Vector storage related exceptions."""


class VectorStoreNotInitializedError(VectorStoreError):
    """Raised when attempting operations on an uninitialized vector store."""


class VectorStoreConfigurationError(VectorStoreError):
    """Raised when vector store configuration is invalid."""


class VectorStoreOperationError(VectorStoreError):
    """Raised when a vector store operation fails."""


class EmbeddingError(VectorStoreError):
    """Raised when embedding operations fail."""


class QueryError(VectorStoreError):
    """Raised when query operations fail."""
