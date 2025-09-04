"""
Base module for agraph knowledge graph components.

This module provides the core data structures and functionality for building
knowledge graphs, including entities, relations, clusters, text chunks, and
the main knowledge graph container.

The module is now organized into logical subdirectories:
- core/: Foundational classes, types, and result handling
- models/: Data structures (Entity, Relation, Cluster, TextChunk)
- infrastructure/: Caching, indexing, and data access layers
- events/: Event system for decoupled architecture
- managers/: Entity and relation management implementations
- transactions/: Batch operations and ACID transaction support
- graphs/: Knowledge graph implementations
- utils/: Utility functions and deprecation management
"""

# Import from reorganized modules to maintain backward compatibility
from .core import (
    ClusterType,
    EntityType,
    ErrorCode,
    ErrorDetail,
    GraphNodeBase,
    PropertyMixin,
    RelationType,
    Result,
    ResultUtils,
    SerializableMixin,
    TextChunkMixin,
    TimestampMixin,
)
from .events import (
    CacheInvalidationListener,
    EventListener,
    EventManager,
    EventPersistenceBackend,
    EventPersistenceListener,
    EventPriority,
    EventType,
    GraphEvent,
    IndexUpdateListener,
    IntegrityCheckListener,
    JSONFileBackend,
)
from .graphs import KnowledgeGraph, OptimizedKnowledgeGraph
from .infrastructure import (
    CacheManager,
    CacheStrategy,
    DataAccessLayer,
    IndexManager,
    IndexType,
    MemoryDataAccessLayer,
    TransactionContext,
    cached,
)
from .managers import (  # Interfaces; Legacy managers (for backward compatibility); Modern implementations; Factory
    BatchOperationManager,
    ClusterManager,
    ClusterManagerInterface,
    DefaultManagerFactory,
    EntityManager,
    EntityManagerInterface,
    Manager,
    ManagerFactory,
    OptimizedEntityManager,
    OptimizedManagerFactory,
    OptimizedRelationManager,
    RelationManager,
    RelationManagerInterface,
    TextChunkManager,
    TextChunkManagerInterface,
    UnifiedClusterManager,
    UnifiedEntityManager,
    UnifiedRelationManager,
    UnifiedTextChunkManager,
)
from .models import Cluster, Entity, Relation, TextChunk
from .transactions import (
    BatchContext,
    BatchOperation,
    BatchOperationContext,
    IsolationLevel,
    Transaction,
    TransactionAwareBatchContext,
    TransactionInfo,
    TransactionManager,
    TransactionStatus,
)
from .utils import deprecated

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
    "OptimizedKnowledgeGraph",
    # Managers (Legacy - for backward compatibility)
    "EntityManager",
    "RelationManager",
    "ClusterManager",
    "TextChunkManager",
    # Manager interfaces
    "Manager",
    "EntityManagerInterface",
    "RelationManagerInterface",
    "ClusterManagerInterface",
    "TextChunkManagerInterface",
    "BatchOperationManager",
    "ManagerFactory",
    # Modern manager implementations
    "UnifiedEntityManager",
    "UnifiedRelationManager",
    "UnifiedClusterManager",
    "UnifiedTextChunkManager",
    "OptimizedEntityManager",
    "OptimizedRelationManager",
    # Manager factory
    "DefaultManagerFactory",
    "OptimizedManagerFactory",
    # Mixins
    "SerializableMixin",
    "PropertyMixin",
    "TimestampMixin",
    # Types
    "EntityType",
    "RelationType",
    "ClusterType",
    # Result handling
    "Result",
    "ErrorCode",
    "ErrorDetail",
    "ResultUtils",
    # Infrastructure
    "CacheManager",
    "CacheStrategy",
    "cached",
    "DataAccessLayer",
    "MemoryDataAccessLayer",
    "TransactionContext",
    "IndexManager",
    "IndexType",
    # Events
    "EventManager",
    "EventType",
    "GraphEvent",
    "EventListener",
    "EventPriority",
    "CacheInvalidationListener",
    "IndexUpdateListener",
    "IntegrityCheckListener",
    "EventPersistenceBackend",
    "EventPersistenceListener",
    "JSONFileBackend",
    # Transactions
    "BatchOperation",
    "BatchContext",
    "BatchOperationContext",
    "TransactionAwareBatchContext",
    "TransactionManager",
    "TransactionStatus",
    "Transaction",
    "TransactionInfo",
    "IsolationLevel",
    # Utils
    "deprecated",
]
