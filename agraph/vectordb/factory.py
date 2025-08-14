"""
Vector store factory for creating different vector database implementations.

This module provides a factory pattern for creating vector store instances
with appropriate configurations based on the backend type.
"""

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Type, Union

from .constants import CHROMA_STORE_DEFAULTS, ERROR_MESSAGES, MEMORY_STORE_DEFAULTS
from .exceptions import VectorStoreError
from .interfaces import VectorStore

if TYPE_CHECKING:
    from .chroma import ChromaVectorStore
    from .memory import MemoryVectorStore
else:
    try:
        from .memory import MemoryVectorStore
    except ImportError:
        MemoryVectorStore = None  # type: ignore

    try:
        from .chroma import ChromaVectorStore
    except ImportError:
        ChromaVectorStore = None  # type: ignore


class VectorStoreType(Enum):
    """Supported vector store types."""

    MEMORY = "memory"
    CHROMA = "chroma"


class VectorStoreFactory:
    """Factory for creating vector store instances."""

    _store_classes: Dict[VectorStoreType, Type[VectorStore]] = {}

    @classmethod
    def register_store_class(
        cls, store_type: VectorStoreType, store_class: Type[VectorStore]
    ) -> None:
        """Register a vector store class for a specific type.

        Args:
            store_type: The vector store type
            store_class: The vector store class
        """
        cls._store_classes[store_type] = store_class

    @classmethod
    def create_store(
        cls,
        store_type: Union[str, VectorStoreType],
        config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> VectorStore:
        """Create a vector store instance.

        Args:
            store_type: Type of vector store to create
            config: Configuration dictionary
            **kwargs: Additional keyword arguments

        Returns:
            Vector store instance

        Raises:
            VectorStoreError: If store type is not supported or creation fails
        """
        # Convert string to enum if needed
        if isinstance(store_type, str):
            try:
                store_type = VectorStoreType(store_type.lower())
            except ValueError as exc:
                raise VectorStoreError(f"Unsupported vector store type: {store_type}") from exc

        # Get store class
        store_class = cls._store_classes.get(store_type)
        if not store_class:
            raise VectorStoreError(
                f"No implementation registered for store type: {store_type.value}"
            )

        # Merge configuration with defaults
        merged_config = cls._get_default_config(store_type)
        if config:
            merged_config.update(config)
        merged_config.update(kwargs)

        try:
            return store_class(**merged_config)
        except Exception as e:
            raise VectorStoreError(f"Failed to create {store_type.value} vector store: {e}") from e

    @classmethod
    def _get_default_config(cls, store_type: VectorStoreType) -> Dict[str, Any]:
        """Get default configuration for a store type.

        Args:
            store_type: The vector store type

        Returns:
            Default configuration dictionary
        """
        if store_type == VectorStoreType.MEMORY:
            return MEMORY_STORE_DEFAULTS.copy()
        if store_type == VectorStoreType.CHROMA:
            return CHROMA_STORE_DEFAULTS.copy()
        return {}

    @classmethod
    def get_supported_types(cls) -> list[str]:
        """Get list of supported vector store types.

        Returns:
            List of supported store type names
        """
        return [store_type.value for store_type in cls._store_classes]

    @classmethod
    def is_store_available(cls, store_type: VectorStoreType) -> bool:
        """Check if a store type is available.

        Args:
            store_type: The store type to check

        Returns:
            True if the store type is available, False otherwise
        """
        return store_type in cls._store_classes


# Auto-register available implementations
def _auto_register_implementations() -> None:
    """Automatically register available vector store implementations."""
    # Register MemoryVectorStore
    if MemoryVectorStore is not None:
        VectorStoreFactory.register_store_class(VectorStoreType.MEMORY, MemoryVectorStore)

    # Register ChromaVectorStore (optional dependency)
    if ChromaVectorStore is not None:
        VectorStoreFactory.register_store_class(VectorStoreType.CHROMA, ChromaVectorStore)


# Register implementations on module import
_auto_register_implementations()


def create_memory_store(
    collection_name: str = "knowledge_graph",
    use_openai_embeddings: bool = False,
    **kwargs: Any,
) -> VectorStore:
    """Convenience function to create a memory vector store.

    Args:
        collection_name: Collection name
        use_openai_embeddings: Whether to use OpenAI embeddings
        **kwargs: Additional arguments

    Returns:
        Memory vector store instance
    """
    return VectorStoreFactory.create_store(
        VectorStoreType.MEMORY,
        collection_name=collection_name,
        use_openai_embeddings=use_openai_embeddings,
        **kwargs,
    )


def create_chroma_store(
    collection_name: str = "knowledge_graph",
    persist_directory: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    use_openai_embeddings: bool = False,
    **kwargs: Any,
) -> VectorStore:
    """Convenience function to create a ChromaDB vector store.

    Args:
        collection_name: Collection name
        persist_directory: Directory for persistent storage
        host: ChromaDB server host
        port: ChromaDB server port
        use_openai_embeddings: Whether to use OpenAI embeddings
        **kwargs: Additional arguments

    Returns:
        ChromaDB vector store instance

    Raises:
        VectorStoreError: If ChromaDB is not available
    """
    if not VectorStoreFactory.is_store_available(VectorStoreType.CHROMA):
        raise VectorStoreError(ERROR_MESSAGES["chromadb_not_available"])

    config = {
        "collection_name": collection_name,
        "persist_directory": persist_directory,
        "host": host,
        "port": port,
        "use_openai_embeddings": use_openai_embeddings,
        **kwargs,
    }
    return VectorStoreFactory.create_store(VectorStoreType.CHROMA, **config)
