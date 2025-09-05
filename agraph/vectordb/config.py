"""
Configuration classes for vector database operations.

This module provides configuration classes and builders for embedding functions
and vector store setups, improving the configuration management and reusability.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .constants import CACHED_EMBEDDING_DEFAULTS, DEFAULT_CHROMA_HOST, DEFAULT_CHROMA_PORT, OPENAI_EMBEDDING_DEFAULTS


@dataclass
class EmbeddingConfig:
    """Configuration for embedding functions."""

    model: str = "text-embedding-3-small"
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    batch_size: int = field(default_factory=lambda: OPENAI_EMBEDDING_DEFAULTS["batch_size"])
    max_concurrency: int = field(default_factory=lambda: OPENAI_EMBEDDING_DEFAULTS["max_concurrency"])
    max_retries: int = field(default_factory=lambda: OPENAI_EMBEDDING_DEFAULTS["max_retries"])
    timeout: float = field(default_factory=lambda: OPENAI_EMBEDDING_DEFAULTS["timeout"])
    use_cache: bool = True
    cache_size: int = field(default_factory=lambda: CACHED_EMBEDDING_DEFAULTS["cache_size"])

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": self.model,
            "api_key": self.api_key,
            "api_base": self.api_base,
            "batch_size": self.batch_size,
            "max_concurrency": self.max_concurrency,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "cache_size": self.cache_size if self.use_cache else None,
        }


@dataclass
class MemoryStoreConfig:
    """Configuration for memory vector store."""

    collection_name: str = "knowledge_graph"
    use_openai_embeddings: bool = False
    embedding_config: Optional[EmbeddingConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config = {
            "collection_name": self.collection_name,
            "use_openai_embeddings": self.use_openai_embeddings,
        }

        if self.embedding_config and self.use_openai_embeddings:
            config["openai_embedding_config"] = self.embedding_config.to_dict()

        return config


@dataclass
class ChromaStoreConfig:
    """Configuration for ChromaDB vector store."""

    collection_name: str = "knowledge_graph"
    persist_directory: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    use_openai_embeddings: bool = False
    embedding_config: Optional[EmbeddingConfig] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config: Dict[str, Any] = {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "host": self.host,
            "port": self.port,
            "use_openai_embeddings": self.use_openai_embeddings,
        }

        if self.embedding_config and self.use_openai_embeddings:
            config["openai_embedding_config"] = self.embedding_config.to_dict()

        return config

    @property
    def is_remote(self) -> bool:
        """Check if this is a remote ChromaDB configuration."""
        return self.host is not None and self.port is not None

    @property
    def is_persistent(self) -> bool:
        """Check if this uses persistent storage."""
        return self.persist_directory is not None


class ConfigBuilder:
    """Builder for creating vector store configurations."""

    @staticmethod
    def embedding_config(
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ) -> EmbeddingConfig:
        """Create embedding configuration.

        Args:
            model: Embedding model name
            api_key: OpenAI API key
            api_base: OpenAI API base URL
            use_cache: Whether to use caching
            **kwargs: Additional configuration options

        Returns:
            Embedding configuration
        """
        config = EmbeddingConfig(
            model=model,
            api_key=api_key,
            api_base=api_base,
            use_cache=use_cache,
        )

        # Update with additional _
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

        return config

    @staticmethod
    def memory_store_config(
        collection_name: str = "knowledge_graph",
        use_openai_embeddings: bool = False,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> MemoryStoreConfig:
        """Create memory store configuration.

        Args:
            collection_name: Collection name
            use_openai_embeddings: Whether to use OpenAI embeddings
            embedding_config: Embedding configuration

        Returns:
            Memory store configuration
        """
        return MemoryStoreConfig(
            collection_name=collection_name,
            use_openai_embeddings=use_openai_embeddings,
            embedding_config=embedding_config,
        )

    @staticmethod
    def chroma_store_config(
        collection_name: str = "knowledge_graph",
        persist_directory: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        use_openai_embeddings: bool = False,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> ChromaStoreConfig:
        """Create ChromaDB store configuration.

        Args:
            collection_name: Collection name
            persist_directory: Directory for persistent storage
            host: ChromaDB server host
            port: ChromaDB server port
            use_openai_embeddings: Whether to use OpenAI embeddings
            embedding_config: Embedding configuration

        Returns:
            ChromaDB store configuration
        """
        return ChromaStoreConfig(
            collection_name=collection_name,
            persist_directory=persist_directory,
            host=host,
            port=port,
            use_openai_embeddings=use_openai_embeddings,
            embedding_config=embedding_config,
        )

    @staticmethod
    def remote_chroma_config(
        host: str = DEFAULT_CHROMA_HOST,
        port: int = DEFAULT_CHROMA_PORT,
        collection_name: str = "knowledge_graph",
        use_openai_embeddings: bool = False,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> ChromaStoreConfig:
        """Create remote ChromaDB configuration.

        Args:
            host: ChromaDB server host
            port: ChromaDB server port
            collection_name: Collection name
            use_openai_embeddings: Whether to use OpenAI embeddings
            embedding_config: Embedding configuration

        Returns:
            Remote ChromaDB configuration
        """
        return ChromaStoreConfig(
            collection_name=collection_name,
            host=host,
            port=port,
            use_openai_embeddings=use_openai_embeddings,
            embedding_config=embedding_config,
        )

    @staticmethod
    def persistent_chroma_config(
        persist_directory: str,
        collection_name: str = "knowledge_graph",
        use_openai_embeddings: bool = False,
        embedding_config: Optional[EmbeddingConfig] = None,
    ) -> ChromaStoreConfig:
        """Create persistent ChromaDB configuration.

        Args:
            persist_directory: Directory for persistent storage
            collection_name: Collection name
            use_openai_embeddings: Whether to use OpenAI embeddings
            embedding_config: Embedding configuration

        Returns:
            Persistent ChromaDB configuration
        """
        return ChromaStoreConfig(
            collection_name=collection_name,
            persist_directory=persist_directory,
            use_openai_embeddings=use_openai_embeddings,
            embedding_config=embedding_config,
        )
