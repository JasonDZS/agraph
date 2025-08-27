"""
: A knowledge graph toolkit for entity and relation management.

This package provides tools for creating, managing, and analyzing knowledge graphs
with focus on entities, relations, and text processing capabilities.
"""

from .agraph import AGraph
from .base.clusters import Cluster
from .base.entities import Entity
from .base.graph import KnowledgeGraph  # DEPRECATED: Use OptimizedKnowledgeGraph instead
from .base.optimized_graph import OptimizedKnowledgeGraph  # RECOMMENDED
from .base.relations import Relation
from .base.text import TextChunk
from .base.types import ClusterType, EntityType, RelationType
from .builder import BuilderConfig, KnowledgeGraphBuilder
from .chunker import SimpleTokenChunker, TokenChunker
from .config import Settings, get_settings

__version__ = "0.1.0"
__author__ = "JasonDZS"
__email__ = "dizhensheng@sz.tsinghua.edu.cn"

__all__ = [
    "AGraph",
    "Entity",
    "Relation",
    "Cluster",
    "KnowledgeGraph",  # DEPRECATED
    "OptimizedKnowledgeGraph",  # RECOMMENDED
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
