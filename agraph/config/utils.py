"""
Configuration utility functions and helpers.

This module provides utility functions for configuration management,
validation, and common operations.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .project import list_projects
from .settings import Settings, get_settings, set_settings


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


def get_config_by_category(category: str) -> Dict[str, Any]:
    """Get configuration for a specific category.

    Args:
        category: Configuration category ('core', 'llm', 'embedding', 'graph',
                 'text', 'rag', 'builder', 'openai', 'unified_views')

    Returns:
        Configuration dictionary for the specified category

    Raises:
        KeyError: If category does not exist
    """
    settings = get_settings()
    all_configs = settings.get_all_configs()
    if category not in all_configs:
        available = list(all_configs.keys())
        raise KeyError(f"Category '{category}' not found. Available categories: {available}")
    return dict(all_configs[category])


def get_unified_config() -> Dict[str, Any]:
    """Get all configurations in a unified dictionary format.

    This is a convenience method that returns all configuration data
    organized by category through the unified Settings interface.

    Returns:
        Dictionary containing all configurations organized by category
    """
    settings = get_settings()
    return settings.get_all_configs()


def export_config_template(template_type: str = "default") -> Dict[str, Any]:
    """Export a configuration template for different use cases.

    Args:
        template_type: Type of template ('default', 'research', 'production', 'development')

    Returns:
        Configuration template dictionary
    """
    base_template: Dict[str, Any] = {
        "workdir": "./workdir",
        "current_project": None,
        "max_current": 5,
        "openai": {"api_key": "${OPENAI_API_KEY}", "api_base": "https://api.openai.com/v1"},
        "llm": {
            "model": "gpt-3.5-turbo",
            "temperature": 0.0,
            "max_tokens": 4096,
            "provider": "openai",
        },
        "embedding": {
            "model": "text-embedding-ada-002",
            "provider": "openai",
            "dimension": 1536,
            "max_token_size": 8192,
            "batch_size": 32,
        },
        "graph": {
            "entity_types": [
                "person",
                "organization",
                "location",
                "concept",
                "event",
                "other",
                "document",
                "keyword",
            ],
            "relation_types": [
                "contains",
                "belongs_to",
                "located_in",
                "works_for",
                "causes",
                "part_of",
                "is_a",
                "references",
                "related_to",
            ],
        },
        "text": {"max_chunk_size": 512, "chunk_overlap": 100},
    }

    if template_type == "research":
        base_template.update(
            {
                "llm": {
                    **dict(base_template["llm"]),
                    "model": "gpt-4",
                    "temperature": 0.1,
                    "max_tokens": 8192,
                },
                "text": {"max_chunk_size": 1000, "chunk_overlap": 200},
            }
        )
    elif template_type == "production":
        base_template.update(
            {
                "llm": {
                    **dict(base_template["llm"]),
                    "model": "gpt-4",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                },
                "embedding": {**dict(base_template["embedding"]), "batch_size": 64},
                "max_current": 10,
            }
        )
    elif template_type == "development":
        base_template.update(
            {
                "llm": {**dict(base_template["llm"]), "temperature": 0.7},
                "embedding": {**dict(base_template["embedding"]), "batch_size": 16},
                "max_current": 3,
            }
        )

    return base_template


def validate_current_settings() -> Tuple[bool, list]:
    """Validate current settings and return any issues.

    Returns:
        Tuple of (is_valid, issues_list)
    """
    settings = get_settings()
    issues = []

    # Basic validation checks
    if not settings.openai.api_key:
        issues.append("OpenAI API key is not configured")

    if not Path(settings.workdir).exists():
        issues.append(f"Working directory does not exist: {settings.workdir}")

    if settings.max_current <= 0:
        issues.append("max_current must be positive")

    if settings.llm.max_tokens <= 0:
        issues.append("LLM max_tokens must be positive")

    return len(issues) == 0, issues


def health_check_config() -> Dict[str, Any]:
    """Perform a comprehensive health check of the current configuration.

    Returns:
        Dictionary with health check results
    """
    settings = get_settings()
    issues = []
    warnings = []

    # Validate settings
    _, validation_issues = validate_current_settings()
    issues.extend(validation_issues)

    # Check workdir accessibility
    workdir_path = Path(settings.workdir)
    if workdir_path.exists():
        if not workdir_path.is_dir():
            issues.append(f"Working directory path is not a directory: {workdir_path}")
        try:
            # Test write access
            test_file = workdir_path / ".agraph_test"
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            issues.append(f"Working directory is not writable: {e}")
    else:
        warnings.append(f"Working directory does not exist: {workdir_path}")

    # Check current project if set
    if settings.current_project:
        available_projects = list_projects()
        if settings.current_project not in available_projects:
            issues.append(f"Current project '{settings.current_project}' does not exist")

    return {
        "healthy": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "settings_summary": {
            "workdir": settings.workdir,
            "current_project": settings.current_project,
            "llm_model": settings.llm.model,
            "embedding_model": settings.embedding.model,
        },
    }
