"""
Graph implementations module.

This module contains different knowledge graph implementations:
- Legacy: Original KnowledgeGraph implementation (deprecated)
- Optimized: Modern implementation with indexing and caching
"""

from .legacy import KnowledgeGraph
from .optimized import OptimizedKnowledgeGraph

__all__ = [
    "KnowledgeGraph",  # Legacy (deprecated)
    "OptimizedKnowledgeGraph",  # Current recommended implementation
]
