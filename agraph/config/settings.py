"""
Unified Settings class and configuration manager.

This module contains the main Settings class that serves as the unified
entry point for all configurations, along with the settings manager
for instance lifecycle management.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from .base import EmbeddingConfig, GraphConfig, LLMConfig, OpenAIConfig, RAGConfig, TextConfig
from .builder import BuilderConfig


class Settings(BaseModel):
    """Main application settings - unified entry point for all configurations."""

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

    # Builder configuration - integrated into Settings
    builder_config: Optional[BuilderConfig] = Field(default=None, exclude=True)

    @property
    def builder(self) -> BuilderConfig:
        """Get builder configuration. Creates default if not exists."""
        if self.builder_config is None:
            self.builder_config = BuilderConfig(
                llm_config=self.llm, embedding_config=self.embedding, openai_config=self.openai
            )
        return self.builder_config

    @builder.setter
    def builder(self, config: BuilderConfig) -> None:
        """Set builder configuration."""
        self.builder_config = config

    # Unified access properties for all configuration types
    @property
    def cache_config(self) -> Dict[str, Any]:
        """Get cache configuration from builder settings."""
        return {
            "enable_cache": self.builder.enable_cache,
            "cache_dir": self.builder.cache_dir,
            "cache_ttl": self.builder.cache_ttl,
            "auto_cleanup": self.builder.auto_cleanup,
        }

    @property
    def processing_config(self) -> Dict[str, Any]:
        """Get processing configuration combining text and builder settings."""
        return {
            "chunk_size": self.builder.chunk_size,
            "chunk_overlap": self.builder.chunk_overlap,
            "max_chunk_size": self.text.max_chunk_size,
            "text_chunk_overlap": self.text.chunk_overlap,
        }

    @property
    def extraction_config(self) -> Dict[str, Any]:
        """Get entity and relation extraction configuration."""
        return {
            "entity_confidence_threshold": self.builder.entity_confidence_threshold,
            "entity_types": self.builder.entity_types or self.graph.entity_types,
            "relation_confidence_threshold": self.builder.relation_confidence_threshold,
            "relation_types": self.builder.relation_types or self.graph.relation_types,
        }

    @property
    def clustering_config(self) -> Dict[str, Any]:
        """Get clustering configuration from builder settings."""
        return {
            "cluster_algorithm": self.builder.cluster_algorithm,
            "min_cluster_size": self.builder.min_cluster_size,
        }

    @property
    def interaction_config(self) -> Dict[str, Any]:
        """Get user interaction configuration from builder settings."""
        return {
            "enable_user_interaction": self.builder.enable_user_interaction,
            "auto_save_edits": self.builder.auto_save_edits,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary including all configurations."""
        result = {
            "workdir": self.workdir,
            "current_project": self.current_project,
            "max_current": self.max_current,
            "openai": self.openai.model_dump(),
            "llm": self.llm.model_dump(),
            "embedding": self.embedding.model_dump(),
            "graph": self.graph.model_dump(),
            "text": self.text.model_dump(),
            "rag": self.rag.model_dump(),
            "builder": self.builder.to_dict(),  # Always include builder config
        }

        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """Create settings from dictionary with validation."""
        # Extract builder config data if present
        builder_data = data.pop("builder", None)

        # Create base settings instance
        settings = cls(
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

        # Set builder config if provided
        if builder_data:
            settings.builder = BuilderConfig.from_dict(builder_data)

        return settings

    def save_to_file(self, file_path: str) -> None:
        """Save settings to JSON file."""
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_file(cls, file_path: str) -> "Settings":
        """Load settings from JSON file with validation."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {file_path}: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to read configuration file {file_path}: {e}") from e

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

    # Configuration access by category
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations organized by category."""
        return {
            "core": {
                "workdir": self.workdir,
                "current_project": self.current_project,
                "max_current": self.max_current,
            },
            "openai": self.openai.model_dump(),
            "llm": self.llm.model_dump(),
            "embedding": self.embedding.model_dump(),
            "graph": self.graph.model_dump(),
            "text": self.text.model_dump(),
            "rag": self.rag.model_dump(),
            "builder": self.builder.to_dict() if self.builder_config else None,
            "unified_views": {
                "cache": self.cache_config,
                "processing": self.processing_config,
                "extraction": self.extraction_config,
                "clustering": self.clustering_config,
                "interaction": self.interaction_config,
            },
        }


# Global settings management
class _SettingsManager:
    """Centralized settings instance manager."""

    def __init__(self) -> None:
        self._instance: Optional[Settings] = None
        self._instance_lock = False  # Simple lock to prevent circular loading

    def get_instance(self) -> Settings:
        """Get or create settings instance with unified loading logic."""
        if self._instance is None and not self._instance_lock:
            self._instance_lock = True
            try:
                # Create default instance
                self._instance = Settings()

                # Try to auto-load from workdir
                self._try_auto_load()

            finally:
                self._instance_lock = False

        return self._instance or Settings()  # Fallback if somehow still None

    def _try_auto_load(self) -> None:
        """Try to auto-load configuration from workdir."""
        if self._instance:
            config_path = Path(self._instance.workdir) / "config.json"
            if config_path.exists():
                try:
                    auto_loaded = Settings.load_from_file(str(config_path))
                    self._instance = auto_loaded
                except Exception:
                    # If auto-loading fails, keep default instance
                    pass

    def set_instance(self, new_instance: Settings) -> None:
        """Set new settings instance."""
        self._instance = new_instance

    def reset_instance(self) -> None:
        """Reset settings instance to None."""
        self._instance = None

    def has_instance(self) -> bool:
        """Check if instance exists."""
        return self._instance is not None


# Global settings manager
_settings_manager = _SettingsManager()


def get_settings() -> Settings:
    """Get application settings instance through unified manager."""
    return _settings_manager.get_instance()


def set_settings(new_settings: Settings) -> None:
    """Set new application settings instance through unified manager."""
    _settings_manager.set_instance(new_settings)


def reset_settings() -> Settings:
    """Reset settings to default values through unified manager."""
    _settings_manager.reset_instance()
    return get_settings()


def has_settings_instance() -> bool:
    """Check if settings instance exists (useful for debugging)."""
    return _settings_manager.has_instance()


def update_settings(updates: Dict[str, Any]) -> Settings:
    """Update current settings with new values through unified manager."""
    current_settings = get_settings()
    new_settings = current_settings.update_from_dict(updates)
    set_settings(new_settings)
    return new_settings
