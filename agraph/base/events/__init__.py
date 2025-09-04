"""
Event system module for decoupled architecture.

This module provides a comprehensive event system for monitoring and reacting
to changes in the knowledge graph:
- Event management and dispatching
- Event listeners for system integration
- Event persistence for audit and analytics
"""

from .events import EventListener, EventManager, EventPriority, EventType, GraphEvent
from .listeners import CacheInvalidationListener, IndexUpdateListener, IntegrityCheckListener
from .persistence import EventPersistenceBackend, EventPersistenceListener, JSONFileBackend

__all__ = [
    # Core event system
    "EventManager",
    "EventType",
    "GraphEvent",
    "EventListener",
    "EventPriority",
    # Event listeners
    "CacheInvalidationListener",
    "IndexUpdateListener",
    "IntegrityCheckListener",
    # Event persistence
    "EventPersistenceBackend",
    "EventPersistenceListener",
    "JSONFileBackend",
]
