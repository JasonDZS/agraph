"""
: A knowledge graph toolkit for entity and relation management.

This package provides tools for creating, managing, and analyzing knowledge graphs
with focus on entities, relations, and text processing capabilities.
"""

from .agraph import AGraph
from .base.core.types import ClusterType, EntityType, RelationType
from .base.graphs.optimized import KnowledgeGraph
from .base.models.clusters import Cluster
from .base.models.entities import Entity
from .base.models.relations import Relation
from .base.models.text import TextChunk
from .builder.builder import KnowledgeGraphBuilder
from .config import BuilderConfig
from .chunker import SimpleTokenChunker, TokenChunker
from .config import Settings, get_settings

__version__ = "0.2.1"
__author__ = "JasonDZS"
__email__ = "dizhensheng@sz.tsinghua.edu.cn"

__all__ = [
    "AGraph",
    "Entity",
    "Relation",
    "Cluster",
    "KnowledgeGraph",
    "TextChunk",
    "EntityType",
    "RelationType",
    "ClusterType",
    "Settings",
    "KnowledgeGraphBuilder",
    "BuilderConfig",
    "TokenChunker",
    "SimpleTokenChunker",
    "get_settings",
]


def get_version() -> str:
    """Return the version of the  package."""
    return __version__


# Rebuild Pydantic models to resolve forward references
Entity.model_rebuild()
Relation.model_rebuild()
Cluster.model_rebuild()
TextChunk.model_rebuild()
KnowledgeGraph.model_rebuild()
