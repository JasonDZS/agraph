"""
Core module containing foundational classes and types.

This module provides the fundamental building blocks for the knowledge graph system:
- Type definitions and enums
- Result handling and error management
- Base classes and mixins
- Common abstractions
"""

from .base import GraphNodeBase, TextChunkMixin
from .mixins import PropertyMixin, SerializableMixin, TimestampMixin
from .result import ErrorCode, ErrorDetail, Result, ResultUtils
from .types import ClusterType, EntityType, RelationType

__all__ = [
    # Base classes
    "GraphNodeBase",
    "TextChunkMixin",
    # Mixins
    "SerializableMixin",
    "PropertyMixin",
    "TimestampMixin",
    # Result handling
    "Result",
    "ErrorCode",
    "ErrorDetail",
    "ResultUtils",
    # Types
    "EntityType",
    "RelationType",
    "ClusterType",
]
