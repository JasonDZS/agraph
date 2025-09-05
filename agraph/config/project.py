"""
Project management utilities for AGraph configurations.

This module provides functions for creating, managing, and manipulating
project-specific configurations and directory structures.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .settings import Settings, get_settings, update_settings


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
    # Validate project name
    if not project_name or not project_name.strip():
        raise ValueError("Project name cannot be empty")

    if not project_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("Project name must contain only alphanumeric characters, hyphens, and underscores")

    paths = get_project_paths(project_name, workdir)
    project_dir = Path(paths["project_dir"])

    # Check if project already exists
    if project_dir.exists():
        raise ValueError(f"Project '{project_name}' already exists")

    # Create project directories
    try:
        project_dir.mkdir(parents=True, exist_ok=True)
        Path(paths["document_storage"]).mkdir(parents=True, exist_ok=True)
        Path(paths["vector_db"]).mkdir(parents=True, exist_ok=True)
        Path(paths["cache"]).mkdir(parents=True, exist_ok=True)
        Path(paths["logs"]).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Failed to create project directories: {e}") from e

    # Create project-specific config with default settings
    base_settings = get_settings()
    project_settings = base_settings.model_copy()
    project_settings.current_project = project_name

    # Save Settings directly to config.json (pure Settings format)
    try:
        with open(paths["config_file"], "w", encoding="utf-8") as f:
            json.dump(project_settings.to_dict(), f, indent=2, ensure_ascii=False)
    except Exception as e:
        # Clean up created directories if config save fails
        try:
            shutil.rmtree(project_dir)
        except Exception:  # More specific exception handling
            pass
        raise RuntimeError(f"Failed to save project configuration: {e}") from e

    # Return minimal project info (not saved to config.json)
    project_config = {
        "project_name": project_name,
        "description": description or f"AGraph project: {project_name}",
        "created_at": datetime.now().isoformat(),
        "version": "1.0.0",
        "paths": paths,
    }

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


def load_project_settings(project_name: str, workdir: Optional[str] = None) -> Settings:
    """Load settings for a specific project."""
    project_config_path = get_config_file_path(workdir, project_name)

    if not Path(project_config_path).exists():
        raise FileNotFoundError(f"Project configuration not found: {project_config_path}")

    with open(project_config_path, "r", encoding="utf-8") as f:
        settings_data = json.load(f)

    # config.json now directly contains Settings data
    project_settings = Settings.from_dict(settings_data)
    project_settings.current_project = project_name

    return project_settings


def save_project_settings(project_name: str, settings: Settings, workdir: Optional[str] = None) -> str:
    """Save settings for a specific project."""
    project_config_path = get_config_file_path(workdir, project_name)

    if not Path(project_config_path).exists():
        raise FileNotFoundError(f"Project configuration not found: {project_config_path}")

    # Save Settings directly to config.json (simplified structure)
    settings.current_project = project_name

    with open(project_config_path, "w", encoding="utf-8") as f:
        json.dump(settings.to_dict(), f, indent=2, ensure_ascii=False)

    return project_config_path


def get_project_info(project_name: str, workdir: Optional[str] = None) -> Dict[str, Any]:
    """Get project information including settings."""
    project_config_path = get_config_file_path(workdir, project_name)

    if not Path(project_config_path).exists():
        raise FileNotFoundError(f"Project configuration not found: {project_config_path}")

    with open(project_config_path, "r", encoding="utf-8") as f:
        project_config: Dict[str, Any] = json.load(f)

    return project_config


def copy_project_settings(source_project: str, target_project: str, workdir: Optional[str] = None) -> Settings:
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
