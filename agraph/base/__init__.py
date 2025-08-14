"""
Base module for agraph knowledge graph components.

This module provides the core data structures and functionality for building
knowledge graphs, including entities, relations, clusters, text chunks, and
the main knowledge graph container.
"""

from .base import GraphNodeBase, TextChunkMixin
from .clusters import Cluster
from .entities import Entity
from .graph import KnowledgeGraph
from .managers import ClusterManager, EntityManager, RelationManager, TextChunkManager
from .mixins import PropertyMixin, SerializableMixin, TimestampMixin
from .relations import Relation
from .text import TextChunk
from .types import ClusterType, EntityType, RelationType

__all__ = [
    # Base classes
    "GraphNodeBase",
    "TextChunkMixin",
    # Core data structures
    "Entity",
    "Relation",
    "Cluster",
    "TextChunk",
    "KnowledgeGraph",
    # Managers
    "EntityManager",
    "RelationManager",
    "ClusterManager",
    "TextChunkManager",
    # Mixins
    "SerializableMixin",
    "PropertyMixin",
    "TimestampMixin",
    # Types
    "EntityType",
    "RelationType",
    "ClusterType",
]
