"""
Data models module containing core knowledge graph entities.

This module provides the primary data structures for the knowledge graph:
- Entity: Knowledge graph entities with types, properties, and relationships
- Relation: Connections between entities with typed relationships
- Cluster: Groups of related entities for organizational purposes
- TextChunk: Text segments with associated entities and relations
- Position: Entity positioning system for precise location tracking
"""

from .clusters import Cluster
from .entities import Entity
from .positioning import AlignmentStatus, CharInterval, Position, PositionMixin, TokenInterval
from .relations import Relation
from .text import TextChunk

__all__ = [
    "Entity",
    "Relation",
    "Cluster",
    "TextChunk",
    "Position",
    "PositionMixin",
    "CharInterval",
    "TokenInterval",
    "AlignmentStatus",
]
