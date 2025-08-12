"""
agraph: A knowledge graph toolkit for entity and relation management.

This package provides tools for creating, managing, and analyzing knowledge graphs
with focus on entities, relations, and text processing capabilities.
"""

from agraph.base.clusters import Cluster
from agraph.base.entities import Entity
from agraph.base.graph import KnowledgeGraph
from agraph.base.relations import Relation
from agraph.base.text import TextChunk
from agraph.base.types import ClusterType, EntityType, RelationType

from . import utils
from .config import Settings, get_settings as _get_settings

__version__ = "0.1.0"
__author__ = "JasonDZS"
__email__ = "dizhensheng@sz.tsinghua.edu.cn"

__all__ = [
    "Entity",
    "Relation",
    "Cluster",
    "KnowledgeGraph",
    "TextChunk",
    "EntityType",
    "RelationType",
    "ClusterType",
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
