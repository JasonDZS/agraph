"""
Infrastructure module providing core system services.

This module contains the foundational infrastructure components:
- Cache: Intelligent caching with LRU+TTL strategies
- Indexes: Multi-index system for O(1) query performance
- DAO: Data access layer abstraction
- Instance management utilities
"""

from .cache import CacheManager, CacheStrategy, cached
from .dao import DataAccessLayer, MemoryDataAccessLayer, TransactionContext
from .indexes import IndexManager, IndexType
from .instances import register_reset_callback

__all__ = [
    # Caching
    "CacheManager",
    "CacheStrategy",
    "cached",
    # Data access
    "DataAccessLayer",
    "MemoryDataAccessLayer",
    "TransactionContext",
    # Indexing
    "IndexManager",
    "IndexType",
    # Instance management
    "register_reset_callback",
]
