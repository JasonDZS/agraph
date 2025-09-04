import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
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
        default="""# System Role

You are an expert knowledge assistant specializing in information retrieval and synthesis from structured knowledge graphs and document collections.

## Objective

Provide comprehensive, well-structured responses to user queries by synthesizing information from the provided data sources. Your responses must be grounded exclusively in the given data sources while maintaining accuracy, clarity, and proper attribution.

## Data Sources Available

**Knowledge Graph (KG)**: Structured entities, relationships, and semantic connections
**Document Chunks (DC)**: Relevant text segments from documents with contextual information

### Temporal Information Handling
- Each data point includes a `created_at` timestamp indicating knowledge acquisition time
- For conflicting information, evaluate both content relevance and temporal context
- Prioritize content-based temporal information over creation timestamps
- Apply contextual judgment rather than defaulting to most recent information

---

## Conversation Context
{history}

## Available Knowledge Sources
{kg_context}

---

## Response Requirements

### Format and Structure
- **Response Type**: {response_type}
- **Language**: Respond in the same language as the user's question
- **Formatting**: Use markdown with clear section headers and proper structure
- **Continuity**: Maintain coherence with conversation history

### Content Organization
- Structure responses with focused sections addressing distinct aspects
- Use descriptive section headers that clearly indicate content focus
- Present information in logical, easily digestible segments

### Citation System
- **Inline Citations**: Use the format `[ID:reference_number]` immediately after each statement or claim that references data sources
  - Example: `The system processes over 10,000 queries daily [ID:1].`
  - Example: `According to the latest research findings [ID:2], performance improved significantly.`
  - Place citations at the end of sentences or clauses, before punctuation

- **References Section**: Always conclude with a "# References" section containing:
  - Format: `ID:number - [Source_Type] Brief description of the source content`
  - Source type indicators: `[KG]` for Knowledge Graph, `[DC]` for Document Chunks
  - Maximum 5 most relevant references

### Reference Format Template
```
### References
- ID:1 [KG] Entity relationship describing system performance metrics
- ID:2 [DC] Research document excerpt about performance improvements
- ID:3 [KG] Semantic connection between entities showing growth trends
```

### Quality Standards
- **Accuracy**: Base all claims exclusively on provided data sources
- **Transparency**: Clearly distinguish between different source types
- **Completeness**: Address all relevant aspects found in the data sources
- **Honesty**: State limitations clearly when information is insufficient
- **No Fabrication**: Never generate information not present in the provided sources

If the available data sources are insufficient to answer the query, explicitly state this limitation and describe what additional information would be needed."""
    )


class Settings(BaseModel):
    """Main application settings."""

    workdir: str = Field(default_factory=lambda: os.getenv("AGRAPH_WORKDIR", "workdir"))
    current_project: Optional[str] = Field(default=None)  # Current active project
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    text: TextConfig = Field(default_factory=TextConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    max_current: int = Field(
        default_factory=lambda: int(os.getenv("MAX_CURRENT", "5")),
        description="Maximum number of concurrent operations",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "workdir": self.workdir,
            "current_project": self.current_project,
            "max_current": self.max_current,
            "openai": self.openai.model_dump(),
            "llm": self.llm.model_dump(),
            "embedding": self.embedding.model_dump(),
            "graph": self.graph.model_dump(),
            "text": self.text.model_dump(),
            "rag": self.rag.model_dump(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create settings from dictionary."""
        return cls(
            workdir=data.get("workdir", "workdir"),
            current_project=data.get("current_project"),
            max_current=data.get("max_current", 5),
            openai=OpenAIConfig(**data.get("openai", {})),
            llm=LLMConfig(**data.get("llm", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            graph=GraphConfig(**data.get("graph", {})),
            text=TextConfig(**data.get("text", {})),
            rag=RAGConfig(**data.get("rag", {})),
        )

    def save_to_file(self, file_path: str) -> None:
        """Save settings to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> "Settings":
        """Load settings from JSON file."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def update_from_dict(self, updates: Dict[str, Any]) -> "Settings":
        """Update settings with new values from dictionary."""
        current_dict = self.to_dict()

        # Deep merge the updates
        def deep_merge(target: Dict, source: Dict) -> Dict:
            for key, value in source.items():
                if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                    target[key] = deep_merge(target[key], value)
                else:
                    target[key] = value
            return target

        updated_dict = deep_merge(current_dict, updates)
        return self.from_dict(updated_dict)


# Global settings instance
_settings_instance: Optional[Settings] = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get application settings instance."""
    global _settings_instance  # pylint: disable=global-statement
    if _settings_instance is None:
        _settings_instance = Settings()
        # Try to auto-load config from workdir if it exists
        _try_auto_load_workdir_config()
    return _settings_instance


def _try_auto_load_workdir_config() -> None:
    """Internal function to try loading config from workdir."""
    global _settings_instance  # pylint: disable=global-statement
    if _settings_instance is None:
        return
    config_path = Path(_settings_instance.workdir) / "config.json"
    if config_path.exists():
        try:
            new_settings = Settings.load_from_file(str(config_path))
            _settings_instance = new_settings
        except Exception:
            # If loading fails, keep the default settings
            pass


def set_settings(new_settings: Settings) -> None:
    """Set new application settings instance."""
    global _settings_instance  # pylint: disable=global-statement
    _settings_instance = new_settings
    # Clear the cache to force reload
    get_settings.cache_clear()


def update_settings(updates: Dict[str, Any]) -> Settings:
    """Update current settings with new values."""
    current_settings = get_settings()
    new_settings = current_settings.update_from_dict(updates)
    set_settings(new_settings)

    # Invalidate cached instances to ensure new settings are used
    try:
        from .base.infrastructure.instances import (  # pylint: disable=import-outside-toplevel
            reset_instances,
        )

        reset_instances()  # Reset all instances for global settings update
    except ImportError:
        # Instance management module not available, skip cache invalidation
        pass

    return new_settings


def reset_settings() -> Settings:
    """Reset settings to default values."""
    global _settings_instance  # pylint: disable=global-statement
    _settings_instance = None
    get_settings.cache_clear()
    return get_settings()


def save_config_to_workdir(config_name: str = "config.json") -> str:
    """Save current configuration to workdir.

    Args:
        config_name: Name of the config file

    Returns:
        Path to the saved config file
    """
    settings = get_settings()
    config_path = Path(settings.workdir) / config_name

    # Ensure directory exists
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Save settings to file
    settings.save_to_file(str(config_path))

    return str(config_path)


def load_config_from_workdir(config_name: str = "config.json") -> Optional[Settings]:
    """Load configuration from workdir.

    Args:
        config_name: Name of the config file

    Returns:
        Loaded settings or None if file doesn't exist
    """
    settings = get_settings()
    config_path = Path(settings.workdir) / config_name

    if not config_path.exists():
        return None

    # Load settings from file
    new_settings = Settings.load_from_file(str(config_path))
    set_settings(new_settings)

    return new_settings


def auto_load_workdir_config(config_name: str = "config.json") -> bool:
    """Automatically load configuration from workdir if exists.

    Args:
        config_name: Name of the config file

    Returns:
        True if config was loaded, False otherwise
    """
    loaded_settings = load_config_from_workdir(config_name)
    return loaded_settings is not None


def get_workdir_config_path() -> str:
    """Get the path to the config file in workdir.

    Returns:
        Full path to the config file
    """
    settings = get_settings()
    return str(Path(settings.workdir) / "config.json")


def has_workdir_config() -> bool:
    """Check if config file exists in workdir.

    Returns:
        True if config file exists
    """
    config_path = Path(get_workdir_config_path())
    return config_path.exists()


def get_config_file_path(workdir: Optional[str] = None, project_name: Optional[str] = None) -> str:
    """Get the path to the configuration file."""
    if workdir is None:
        workdir = get_settings().workdir

    if project_name:
        return str(Path(workdir) / "projects" / project_name / "config.json")

    return str(Path(workdir) / "config.json")


def get_project_dir(project_name: str, workdir: Optional[str] = None) -> str:
    """Get the path to a project directory."""
    if workdir is None:
        workdir = get_settings().workdir
    return str(Path(workdir) / "projects" / project_name)


def get_project_paths(project_name: str, workdir: Optional[str] = None) -> Dict[str, str]:
    """Get all relevant paths for a project."""
    project_dir = get_project_dir(project_name, workdir)

    return {
        "project_dir": project_dir,
        "config_file": str(Path(project_dir) / "config.json"),
        "document_storage": str(Path(project_dir) / "document_storage"),
        "vector_db": str(Path(project_dir) / "agraph_vectordb"),
        "cache": str(Path(project_dir) / "cache"),
        "logs": str(Path(project_dir) / "logs"),
    }


def list_projects(workdir: Optional[str] = None) -> List[str]:
    """List all available projects."""
    if workdir is None:
        workdir = get_settings().workdir

    projects_dir = Path(workdir) / "projects"
    if not projects_dir.exists():
        return []

    projects = []
    for item in projects_dir.iterdir():
        if item.is_dir() and (item / "config.json").exists():
            projects.append(item.name)

    return sorted(projects)


def create_project(
    project_name: str, description: Optional[str] = None, workdir: Optional[str] = None
) -> Dict[str, Any]:
    """Create a new project with its directory structure."""
    if not project_name or not project_name.strip():
        raise ValueError("Project name cannot be empty")

    # Validate project name
    if not re.match(r"^[a-zA-Z0-9_-]+$", project_name):
        raise ValueError("Project name can only contain letters, numbers, underscores, and hyphens")

    paths = get_project_paths(project_name, workdir)
    project_dir = Path(paths["project_dir"])

    # Check if project already exists
    if project_dir.exists():
        raise ValueError(f"Project '{project_name}' already exists")

    # Create project directories
    project_dir.mkdir(parents=True, exist_ok=True)
    Path(paths["document_storage"]).mkdir(parents=True, exist_ok=True)
    Path(paths["vector_db"]).mkdir(parents=True, exist_ok=True)
    Path(paths["cache"]).mkdir(parents=True, exist_ok=True)
    Path(paths["logs"]).mkdir(parents=True, exist_ok=True)

    # Create project-specific config with default settings
    base_settings = get_settings()
    project_settings = base_settings.model_copy()
    project_settings.current_project = project_name

    project_config = {
        "project_name": project_name,
        "description": description or f"AGraph project: {project_name}",
        "created_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "paths": paths,
        "settings": project_settings.to_dict(),
    }

    # Save project config
    with open(paths["config_file"], "w", encoding="utf-8") as f:
        json.dump(project_config, f, indent=2, ensure_ascii=False)

    return project_config


def delete_project(project_name: str, workdir: Optional[str] = None) -> bool:
    """Delete a project and all its data."""
    project_dir = Path(get_project_dir(project_name, workdir))
    if not project_dir.exists():
        return False

    # Remove the entire project directory
    shutil.rmtree(project_dir)

    # If this was the current project, reset current_project
    settings = get_settings()
    if settings.current_project == project_name:
        update_settings({"current_project": None})

    return True


def set_current_project(project_name: Optional[str]) -> Settings:
    """Set the current active project."""
    if project_name and project_name not in list_projects():
        raise ValueError(f"Project '{project_name}' does not exist")

    return update_settings({"current_project": project_name})


def get_current_project() -> Optional[str]:
    """Get the current active project."""
    return get_settings().current_project


def save_settings_to_file(
    file_path: Optional[str] = None, project_name: Optional[str] = None
) -> str:
    """Save current settings to file."""
    if file_path is None:
        file_path = get_config_file_path(project_name=project_name)

    # Ensure directory exists
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    settings = get_settings()

    if project_name:
        # For project-specific configs, save as part of project config
        project_config_path = get_config_file_path(project_name=project_name)
        if Path(project_config_path).exists():
            with open(project_config_path, "r", encoding="utf-8") as f:
                project_config = json.load(f)
            project_config["settings"] = settings.to_dict()
            with open(project_config_path, "w", encoding="utf-8") as f:
                json.dump(project_config, f, indent=2, ensure_ascii=False)
            return project_config_path

    # Default behavior for global settings
    settings.save_to_file(file_path)
    return file_path


def load_settings_from_file(
    file_path: Optional[str] = None, project_name: Optional[str] = None
) -> Settings:
    """Load settings from file and set as current settings."""
    if file_path is None:
        file_path = get_config_file_path(project_name=project_name)

    if project_name:
        # Load from project-specific config
        return load_project_settings(project_name)

    # Default behavior for global settings
    new_settings = Settings.load_from_file(file_path)
    set_settings(new_settings)
    return new_settings


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

    # User interaction configuration
    enable_user_interaction: bool = True
    auto_save_edits: bool = True

    # Integrated configuration instances
    llm_config: LLMConfig = field(default_factory=LLMConfig)
    embedding_config: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    openai_config: OpenAIConfig = field(default_factory=OpenAIConfig)

    def __post_init__(self):
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

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BuilderConfig":
        """Create config from dictionary."""
        # Extract nested config data
        llm_config_data = data.pop("llm_config", {})
        embedding_config_data = data.pop("embedding_config", {})
        openai_config_data = data.pop("openai_config", {})
        
        # Create nested config instances
        llm_config = LLMConfig(**llm_config_data) if llm_config_data else LLMConfig()
        embedding_config = EmbeddingConfig(**embedding_config_data) if embedding_config_data else EmbeddingConfig()
        openai_config = OpenAIConfig(**openai_config_data) if openai_config_data else OpenAIConfig()
        
        # Create base config
        config = cls(
            llm_config=llm_config,
            embedding_config=embedding_config,
            openai_config=openai_config
        )
        
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


def load_project_settings(project_name: str, workdir: Optional[str] = None) -> Settings:
    """Load settings for a specific project."""
    project_config_path = get_config_file_path(workdir, project_name)

    if not Path(project_config_path).exists():
        raise FileNotFoundError(f"Project configuration not found: {project_config_path}")

    with open(project_config_path, "r", encoding="utf-8") as f:
        project_config = json.load(f)

    # Extract settings from project config
    settings_data = project_config.get("settings", {})
    project_settings = Settings.from_dict(settings_data)
    project_settings.current_project = project_name

    return project_settings


def save_project_settings(
    project_name: str, settings: Settings, workdir: Optional[str] = None
) -> str:
    """Save settings for a specific project."""
    project_config_path = get_config_file_path(workdir, project_name)

    if not Path(project_config_path).exists():
        raise FileNotFoundError(f"Project configuration not found: {project_config_path}")

    # Load existing project config
    with open(project_config_path, "r", encoding="utf-8") as f:
        project_config = json.load(f)

    # Update settings in project config
    project_config["settings"] = settings.to_dict()
    project_config["settings"]["current_project"] = project_name

    # Save updated project config
    with open(project_config_path, "w", encoding="utf-8") as f:
        json.dump(project_config, f, indent=2, ensure_ascii=False)

    return project_config_path


def update_project_settings(
    project_name: str, updates: Dict[str, Any], workdir: Optional[str] = None
) -> Settings:
    """Update settings for a specific project."""
    # Load current project settings
    current_settings = load_project_settings(project_name, workdir)

    # Apply updates
    updated_settings = current_settings.update_from_dict(updates)
    updated_settings.current_project = project_name

    # Save updated settings
    save_project_settings(project_name, updated_settings, workdir)

    # Invalidate cached instances for this project to ensure new settings are used
    try:
        from .base.infrastructure.instances import (  # pylint: disable=import-outside-toplevel
            reset_instances,
        )

        reset_instances(project_name)
    except ImportError:
        # Instance management module not available, skip cache invalidation
        pass

    return updated_settings


def get_project_info(project_name: str, workdir: Optional[str] = None) -> Dict[str, Any]:
    """Get project information including settings."""
    project_config_path = get_config_file_path(workdir, project_name)

    if not Path(project_config_path).exists():
        raise FileNotFoundError(f"Project configuration not found: {project_config_path}")

    with open(project_config_path, "r", encoding="utf-8") as f:
        project_config: Dict[str, Any] = json.load(f)

    return project_config


def copy_project_settings(
    source_project: str, target_project: str, workdir: Optional[str] = None
) -> Settings:
    """Copy settings from one project to another."""
    # Load source project settings
    source_settings = load_project_settings(source_project, workdir)

    # Update project name
    target_settings = source_settings.model_copy()
    target_settings.current_project = target_project

    # Save to target project
    save_project_settings(target_project, target_settings, workdir)

    return target_settings


def reset_project_settings(project_name: str, workdir: Optional[str] = None) -> Settings:
    """Reset project settings to default values."""
    # Create new default settings
    default_settings = Settings()
    default_settings.current_project = project_name

    # Save to project
    save_project_settings(project_name, default_settings, workdir)

    return default_settings


# Settings will be loaded on first access via get_settings()
# Removed module-level initialization to avoid circular imports
