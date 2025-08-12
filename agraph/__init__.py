"""
agraph: A knowledge graph toolkit for entity and relation management.

This package provides tools for creating, managing, and analyzing knowledge graphs
with focus on entities, relations, and text processing capabilities.
"""

from . import utils
from .config import Settings, get_settings as _get_settings
from .entities import Entity
from .relations import Relation
from .types import EntityType, RelationType

__version__ = "0.1.0"
__author__ = "JasonDZS"
__email__ = "dizhensheng@sz.tsinghua.edu.cn"

__all__ = [
    "Entity",
    "Relation",
    "EntityType",
    "RelationType",
    "Settings",
    "utils",
]


def get_version() -> str:
    """Return the version of the agraph package."""
    return __version__


# Package-level configuration access
def get_settings() -> Settings:
    """Get application settings instance."""
    return _get_settings()
