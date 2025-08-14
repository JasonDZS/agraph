import os
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(".env")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    api_base: str = Field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    )


class LLMConfig(BaseModel):
    """Language model configuration."""

    model: str = Field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-3.5-turbo"))
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096")))
    provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "Pro/BAAI/bge-m3"))
    provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "openai"))
    dimension: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024")))
    max_token_size: int = Field(
        default_factory=lambda: int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))
    )
    batch_size: int = Field(default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32")))


class GraphConfig(BaseModel):
    """Knowledge graph configuration."""

    entity_types: List[str] = Field(
        default_factory=lambda: [
            "person",
            "organization",
            "location",
            "concept",
            "event",
            "other",
            "table",
            "column",
            "database",
            "document",
            "keyword",
            "product",
            "software",
            "unknown",
        ]
    )

    relation_types: List[str] = Field(
        default_factory=lambda: [
            "contains",
            "belongs_to",
            "located_in",
            "works_for",
            "causes",
            "part_of",
            "is_a",
            "references",
            "similar_to",
            "related_to",
            "depends_on",
            "foreign_key",
            "mentions",
            "describes",
            "synonyms",
            "develops",
            "creates",
            "founded_by",
            "other",
        ]
    )


class TextConfig(BaseModel):
    """Text processing configuration."""

    max_chunk_size: int = Field(default_factory=lambda: int(os.getenv("MAX_CHUNK_SIZE", "512")))
    chunk_overlap: int = Field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "100")))


class RAGConfig(BaseModel):
    """RAG system configuration."""

    system_prompt: str = Field(
        default="""---Role---

You are a helpful assistant responding to user query about Data Sources provided below.


---Goal---

Generate a concise response based on Data Sources and follow Response Rules, considering both
the conversation history and the current query. Data sources contain two parts: Knowledge
Graph(KG) and Document Chunks(DC). Summarize all information in the provided Data Sources,
and incorporating general knowledge relevant to the Data Sources. Do not include information
not provided by Data Sources.

When handling information with timestamps:
1. Each piece of information (both relationships and content) has a "created_at" timestamp
   indicating when we acquired this knowledge
2. When encountering conflicting information, consider both the content/relationship
   and the timestamp
3. Don't automatically prefer the most recent information - use judgment based on
   the context
4. For time-specific queries, prioritize temporal information in the content before
   considering creation timestamps

---Conversation History---
{history}

---Data Sources---

1. From Knowledge Graph(KG):
{kg_context}

---Response Rules---

- Target format and length: {response_type}
- Use markdown formatting with appropriate section headings
- Please respond in the same language as the user's question.
- Ensure the response maintains continuity with the conversation history.
- Organize answer in sesctions focusing on one main point or aspect of the answer
- Use clear and descriptive section titles that reflect the content
- List up to 5 most important reference sources at the end under "References" sesction.
  Clearly indicating whether each source is from Knowledge Graph (KG) or Vector Data (DC),
  in the following format: [KG/DC] Source content
- If you don't know the answer, just say so. Do not make anything up.
- Do not include information not provided by the Data Sources."""
    )


class Settings(BaseModel):
    """Main application settings."""

    workdir: str = Field(default="workdir")
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Builder configuration classes (migrated from builder/config.py)


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

    # LLM configuration
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"

    # User interaction configuration
    enable_user_interaction: bool = True
    auto_save_edits: bool = True

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
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model,
            "enable_user_interaction": self.enable_user_interaction,
            "auto_save_edits": self.auto_save_edits,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuilderConfig":
        """Create config from dictionary."""
        config = cls()
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
            started_at=(
                datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None
            ),
            updated_at=(
                datetime.fromisoformat(data["updated_at"]) if data.get("updated_at") else None
            ),
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


# Load settings at module level
settings = get_settings()
