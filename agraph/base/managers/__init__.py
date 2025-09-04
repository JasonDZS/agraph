"""
Manager module containing all entity and relation management implementations.

This module provides different manager implementations:
- Interfaces: Abstract base classes for all managers
- Legacy: Original manager implementations (backward compatibility)
- Unified: Modern managers using DAO pattern
- Optimized: Performance-enhanced managers with indexing and caching
- Factory: Factory patterns for manager instantiation
"""

# Factory
from .factory import DefaultManagerFactory, OptimizedManagerFactory

# Interfaces
from .interfaces import (
    BatchOperationManager,
    ClusterManager as ClusterManagerInterface,
    EntityManager as EntityManagerInterface,
    Manager,
    ManagerFactory,
    RelationManager as RelationManagerInterface,
    TextChunkManager as TextChunkManagerInterface,
)

# Legacy managers (for backward compatibility)
from .legacy import ClusterManager, EntityManager, RelationManager, TextChunkManager

# Optimized implementations
from .optimized import OptimizedEntityManager, OptimizedRelationManager

# Modern implementations
from .unified import (
    UnifiedClusterManager,
    UnifiedEntityManager,
    UnifiedRelationManager,
    UnifiedTextChunkManager,
)

__all__ = [
    # Interfaces
    "Manager",
    "EntityManagerInterface",
    "RelationManagerInterface",
    "ClusterManagerInterface",
    "TextChunkManagerInterface",
    "BatchOperationManager",
    "ManagerFactory",
    # Legacy managers
    "EntityManager",
    "RelationManager",
    "ClusterManager",
    "TextChunkManager",
    # Unified managers
    "UnifiedEntityManager",
    "UnifiedRelationManager",
    "UnifiedClusterManager",
    "UnifiedTextChunkManager",
    # Optimized managers
    "OptimizedEntityManager",
    "OptimizedRelationManager",
    # Factory
    "DefaultManagerFactory",
    "OptimizedManagerFactory",
]
