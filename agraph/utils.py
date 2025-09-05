"""
Utility functions for the agraph package.

This module provides common utility functions used across the package,
including path management, project utilities, validation helpers, and more.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .base.core.types import ClusterType, EntityType, RelationType
from .config import get_settings


def get_type_value(type_obj: Union[EntityType, RelationType, ClusterType, str]) -> str:
    """
    Get the string value from a type object.

    Args:
        type_obj: The type object to convert to string

    Returns:
        String representation of the type
    """
    if isinstance(type_obj, (EntityType, RelationType, ClusterType)):
        return str(type_obj.value)
    return str(type_obj)


# Path and Project Management Utilities


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


def get_workdir_config_path(workdir: Optional[str] = None) -> str:
    """Get the path to the config file in workdir.

    Returns:
        Full path to the config file
    """
    if workdir is None:
        workdir = get_settings().workdir
    return str(Path(workdir) / "config.json")


def has_workdir_config(workdir: Optional[str] = None) -> bool:
    """Check if config file exists in workdir.

    Returns:
        True if config file exists
    """
    config_path = Path(get_workdir_config_path(workdir))
    return config_path.exists()


# Project Information and Management


def get_project_info(project_name: str, workdir: Optional[str] = None) -> Dict[str, Any]:
    """Get project information including settings."""
    project_config_path = get_config_file_path(workdir, project_name)

    if not Path(project_config_path).exists():
        raise FileNotFoundError(f"Project configuration not found: {project_config_path}")

    with open(project_config_path, "r", encoding="utf-8") as f:
        project_config: Dict[str, Any] = json.load(f)

    return project_config


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


# Build Steps Constants and Utilities


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


# Helper Functions for Deep Merging


def deep_merge_dict(target: Dict, source: Dict) -> Dict:
    """Deep merge two dictionaries.

    Args:
        target: Target dictionary to merge into
        source: Source dictionary to merge from

    Returns:
        Merged dictionary
    """
    for key, value in source.items():
        if key in target and isinstance(target[key], dict) and isinstance(value, dict):
            target[key] = deep_merge_dict(target[key], value)
        else:
            target[key] = value
    return target
