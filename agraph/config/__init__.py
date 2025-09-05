"""
AGraph Configuration Package.

This package provides a modular configuration system for AGraph, with
separate modules for different aspects of configuration management.

The package is organized as follows:
- base: Basic configuration classes (OpenAIConfig, LLMConfig, etc.)
- settings: Unified Settings class and manager
- builder: Builder-related configurations
- project: Project management utilities
- utils: Utility functions and helpers

For backward compatibility, all main classes and functions are re-exported
from this module, so existing code can continue to import directly from
agraph.config.
"""

# Import environment variables
from dotenv import load_dotenv

# Core configuration classes
from .base import EmbeddingConfig, GraphConfig, LLMConfig, OpenAIConfig, RAGConfig, TextConfig

# Builder configurations
from .builder import BuilderConfig, BuildStatus, BuildSteps, CacheMetadata, DocumentProcessingStatus

# Project management
from .project import (
    copy_project_settings,
    create_project,
    delete_project,
    get_config_file_path,
    get_current_project,
    get_project_dir,
    get_project_info,
    get_project_paths,
    list_projects,
    load_project_settings,
    reset_project_settings,
    save_project_settings,
    set_current_project,
)

# Settings management
from .settings import Settings, get_settings, has_settings_instance, reset_settings, set_settings, update_settings

# Utility functions
from .utils import (
    auto_load_workdir_config,
    export_config_template,
    get_config_by_category,
    get_unified_config,
    get_workdir_config_path,
    has_workdir_config,
    health_check_config,
    load_config_from_workdir,
    save_config_to_workdir,
    validate_current_settings,
)

# Load environment variables after imports
load_dotenv(".env")


# For backward compatibility - functions that were in the original config.py
def save_settings_to_file(file_path: str | None = None, project_name: str | None = None) -> str:
    """Save current settings to file (backward compatibility wrapper)."""
    if project_name:
        settings = get_settings()
        return save_project_settings(project_name, settings)
    file_path = file_path or get_workdir_config_path()
    settings = get_settings()
    settings.save_to_file(file_path)
    return file_path


def load_settings_from_file(file_path: str | None = None, project_name: str | None = None) -> Settings:
    """Load settings from file (backward compatibility wrapper)."""
    if project_name:
        return load_project_settings(project_name)
    file_path = file_path or get_workdir_config_path()
    new_settings = Settings.load_from_file(file_path)
    set_settings(new_settings)
    return new_settings


def update_project_settings(project_name: str, updates: dict, workdir: str | None = None) -> Settings:
    """Update settings for a specific project (backward compatibility wrapper)."""
    current_settings = load_project_settings(project_name, workdir)
    updated_settings = current_settings.update_from_dict(updates)
    updated_settings.current_project = project_name
    save_project_settings(project_name, updated_settings, workdir)
    return updated_settings


def update_config_by_category(category: str, updates: dict) -> Settings:
    """Update configuration for a specific category (backward compatibility wrapper)."""
    if category == "unified_views":
        # Handle unified view updates by delegating to appropriate update methods
        settings = get_settings()
        for view_name, view_updates in updates.items():
            if view_name == "cache" and isinstance(view_updates, dict):
                # Update cache config through builder
                for key, value in view_updates.items():
                    if hasattr(settings.builder, key):
                        setattr(settings.builder, key, value)
            # Add other view handlers as needed
        set_settings(settings)
        return settings
    # Regular category update
    return update_settings({category: updates})


# Export all symbols for backward compatibility
__all__ = [
    # Base configurations
    "OpenAIConfig",
    "LLMConfig",
    "EmbeddingConfig",
    "GraphConfig",
    "TextConfig",
    "RAGConfig",
    # Settings management
    "Settings",
    "get_settings",
    "set_settings",
    "reset_settings",
    "has_settings_instance",
    "update_settings",
    # Builder configurations
    "BuilderConfig",
    "CacheMetadata",
    "DocumentProcessingStatus",
    "BuildStatus",
    "BuildSteps",
    # Project management
    "get_config_file_path",
    "get_project_dir",
    "get_project_paths",
    "list_projects",
    "create_project",
    "delete_project",
    "set_current_project",
    "get_current_project",
    "load_project_settings",
    "save_project_settings",
    "get_project_info",
    "copy_project_settings",
    "reset_project_settings",
    "update_project_settings",
    # Utility functions
    "get_workdir_config_path",
    "has_workdir_config",
    "save_config_to_workdir",
    "load_config_from_workdir",
    "auto_load_workdir_config",
    "get_config_by_category",
    "get_unified_config",
    "export_config_template",
    "validate_current_settings",
    "health_check_config",
    # Backward compatibility
    "save_settings_to_file",
    "load_settings_from_file",
    "update_config_by_category",
]
