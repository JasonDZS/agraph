"""
Builder-related configuration classes.

This module contains configuration classes specifically for the knowledge graph
building process, including caching, processing status, and build management.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import EmbeddingConfig, GraphConfig, LLMConfig, OpenAIConfig


@dataclass
class BuilderConfig:
    """Configuration for KnowledgeGraph Builder."""

    # Cache configuration
    enable_cache: bool = True
    cache_dir: str = "./cache"
    cache_ttl: int = 86400  # Cache TTL in seconds
    auto_cleanup: bool = True

    # Chunking configuration
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Entity extraction configuration
    entity_confidence_threshold: float = 0.7
    entity_types: List[str] = field(default_factory=list)

    # Relation extraction configuration
    relation_confidence_threshold: float = 0.6
    relation_types: List[str] = field(default_factory=list)

    # Clustering configuration
    cluster_algorithm: str = "community_detection"
    min_cluster_size: int = 2

    # User interaction configuration
    enable_user_interaction: bool = True
    auto_save_edits: bool = True

    # Integrated configuration instances
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    openai_config: OpenAIConfig = field(default_factory=OpenAIConfig)

    def __post_init__(self) -> None:
        """Post-initialization to handle configuration integration."""
        # Initialize default entity and relation types if empty
        if not self.entity_types:
            self.entity_types = GraphConfig().entity_types
        if not self.relation_types:
            self.relation_types = GraphConfig().relation_types

    # Convenience properties for backward compatibility
    @property
    def llm_provider(self) -> str:
        """Get LLM provider from integrated config."""
        return self.llm_config.provider

    @property
    def llm_model(self) -> str:
        """Get LLM model from integrated config."""
        return self.llm_config.model

    @property
    def llm_temperature(self) -> float:
        """Get LLM temperature from integrated config."""
        return self.llm_config.temperature

    @property
    def llm_max_tokens(self) -> int:
        """Get LLM max tokens from integrated config."""
        return self.llm_config.max_tokens

    @property
    def embedding_provider(self) -> str:
        """Get embedding provider from integrated config."""
        return self.embedding_config.provider

    @property
    def embedding_model(self) -> str:
        """Get embedding model from integrated config."""
        return self.embedding_config.model

    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension from integrated config."""
        return self.embedding_config.dimension

    @property
    def embedding_max_token_size(self) -> int:
        """Get embedding max token size from integrated config."""
        return self.embedding_config.max_token_size

    @property
    def embedding_batch_size(self) -> int:
        """Get embedding batch size from integrated config."""
        return self.embedding_config.batch_size

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "enable_cache": self.enable_cache,
            "cache_dir": self.cache_dir,
            "cache_ttl": self.cache_ttl,
            "auto_cleanup": self.auto_cleanup,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "entity_confidence_threshold": self.entity_confidence_threshold,
            "entity_types": [str(et) for et in self.entity_types],
            "relation_confidence_threshold": self.relation_confidence_threshold,
            "relation_types": [str(rt) for rt in self.relation_types],
            "cluster_algorithm": self.cluster_algorithm,
            "min_cluster_size": self.min_cluster_size,
            "enable_user_interaction": self.enable_user_interaction,
            "auto_save_edits": self.auto_save_edits,
            "llm_config": self.llm_config.model_dump(),
            "embedding_config": self.embedding_config.model_dump(),
            "openai_config": self.openai_config.model_dump(),
        }

    def to_dict_safe(self) -> Dict[str, Any]:
        """Convert config to dictionary with masked sensitive data for API responses."""
        return {
            "enable_cache": self.enable_cache,
            "cache_dir": self.cache_dir,
            "cache_ttl": self.cache_ttl,
            "auto_cleanup": self.auto_cleanup,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "entity_confidence_threshold": self.entity_confidence_threshold,
            "entity_types": [str(et) for et in self.entity_types],
            "relation_confidence_threshold": self.relation_confidence_threshold,
            "relation_types": [str(rt) for rt in self.relation_types],
            "cluster_algorithm": self.cluster_algorithm,
            "min_cluster_size": self.min_cluster_size,
            "enable_user_interaction": self.enable_user_interaction,
            "auto_save_edits": self.auto_save_edits,
            "llm_config": self.llm_config.model_dump(),
            "embedding_config": self.embedding_config.model_dump(),
            "openai_config": self.openai_config.mask_sensitive_data(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuilderConfig":
        """Create config from dictionary with validation."""
        # Extract nested config data
        llm_config_data = data.pop("llm_config", {})
        embedding_config_data = data.pop("embedding_config", {})
        openai_config_data = data.pop("openai_config", {})

        # Create nested config instances
        llm_config = LLMConfig(**llm_config_data) if llm_config_data else LLMConfig()
        embedding_config = EmbeddingConfig(**embedding_config_data) if embedding_config_data else EmbeddingConfig()
        openai_config = OpenAIConfig(**openai_config_data) if openai_config_data else OpenAIConfig()

        # Create base config
        config = cls(llm_config=llm_config, embedding_config=embedding_config, openai_config=openai_config)

        # Set remaining fields
        for key, value in data.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config


@dataclass
class CacheMetadata:
    """Metadata for cached results."""

    step_name: str
    timestamp: datetime
    input_hash: str
    config_hash: str
    version: str = "1.0"
    file_hashes: Optional[Dict[str, str]] = None  # File path -> hash mapping
    processing_time: Optional[float] = None  # Processing time in seconds

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "step_name": self.step_name,
            "timestamp": self.timestamp.isoformat(),
            "input_hash": self.input_hash,
            "config_hash": self.config_hash,
            "version": self.version,
            "file_hashes": self.file_hashes,
            "processing_time": self.processing_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CacheMetadata":
        """Create metadata from dictionary."""
        return cls(
            step_name=data["step_name"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            input_hash=data["input_hash"],
            config_hash=data["config_hash"],
            version=data.get("version", "1.0"),
            file_hashes=data.get("file_hashes"),
            processing_time=data.get("processing_time"),
        )


@dataclass
class DocumentProcessingStatus:
    """Document processing status for incremental updates."""

    file_path: str
    file_hash: str
    last_modified: datetime
    processing_status: str  # "pending", "processing", "completed", "failed"
    extracted_text_hash: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "file_path": self.file_path,
            "file_hash": self.file_hash,
            "last_modified": self.last_modified.isoformat(),
            "processing_status": self.processing_status,
            "extracted_text_hash": self.extracted_text_hash,
            "processing_time": self.processing_time,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentProcessingStatus":
        """Create status from dictionary."""
        return cls(
            file_path=data["file_path"],
            file_hash=data["file_hash"],
            last_modified=datetime.fromisoformat(data["last_modified"]),
            processing_status=data["processing_status"],
            extracted_text_hash=data.get("extracted_text_hash"),
            processing_time=data.get("processing_time"),
            error_message=data.get("error_message"),
        )


@dataclass
class BuildStatus:
    """Build status tracking."""

    current_step: Optional[str] = None
    last_completed_step: Optional[str] = None
    total_steps: int = 6
    completed_steps: int = 0
    started_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @property
    def progress(self) -> float:
        """Calculate build progress as percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100.0

    @property
    def is_completed(self) -> bool:
        """Check if build is completed."""
        return self.completed_steps == self.total_steps

    @property
    def has_error(self) -> bool:
        """Check if build has error."""
        return self.error_message is not None

    def to_dict(self) -> Dict[str, Any]:
        """Convert status to dictionary."""
        return {
            "current_step": self.current_step,
            "last_completed_step": self.last_completed_step,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "progress": self.progress,
            "is_completed": self.is_completed,
            "has_error": self.has_error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuildStatus":
        """Create status from dictionary."""
        return cls(
            current_step=data.get("current_step"),
            last_completed_step=data.get("last_completed_step"),
            total_steps=data.get("total_steps", 6),
            completed_steps=data.get("completed_steps", 0),
            started_at=(datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None),
            updated_at=(datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None),
            error_message=data.get("error_message"),
        )


# Step names constants
class BuildSteps:
    """Constants for build step names."""

    DOCUMENT_PROCESSING = "document_processing"
    TEXT_CHUNKING = "text_chunking"
    ENTITY_EXTRACTION = "entity_extraction"
    RELATION_EXTRACTION = "relation_extraction"
    CLUSTER_FORMATION = "cluster_formation"
    GRAPH_ASSEMBLY = "graph_assembly"

    ALL_STEPS = [
        DOCUMENT_PROCESSING,
        TEXT_CHUNKING,
        ENTITY_EXTRACTION,
        RELATION_EXTRACTION,
        CLUSTER_FORMATION,
        GRAPH_ASSEMBLY,
    ]

    STEP_DEPENDENCIES = {
        TEXT_CHUNKING: [DOCUMENT_PROCESSING],
        ENTITY_EXTRACTION: [TEXT_CHUNKING],
        RELATION_EXTRACTION: [TEXT_CHUNKING, ENTITY_EXTRACTION],
        CLUSTER_FORMATION: [ENTITY_EXTRACTION, RELATION_EXTRACTION],
        GRAPH_ASSEMBLY: [TEXT_CHUNKING, ENTITY_EXTRACTION, RELATION_EXTRACTION, CLUSTER_FORMATION],
    }

    @classmethod
    def get_step_index(cls, step_name: str) -> int:
        """Get step index by name."""
        try:
            return cls.ALL_STEPS.index(step_name)
        except ValueError:
            return -1

    @classmethod
    def get_dependent_steps(cls, step_name: str) -> List[str]:
        """Get all steps that depend on the given step."""
        dependent_steps = []
        for step, dependencies in cls.STEP_DEPENDENCIES.items():
            if step_name in dependencies:
                dependent_steps.append(step)
                # Recursively find steps that depend on this step
                dependent_steps.extend(cls.get_dependent_steps(step))
        return list(set(dependent_steps))  # Remove duplicates
