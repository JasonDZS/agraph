"""
Event system for AGraph knowledge graph operations.

This module provides a comprehensive event system that allows components to
subscribe to and emit events for various knowledge graph operations. This
enables decoupled architecture and supports features like:
- Index updates on entity/relation changes
- Cache invalidation on data modifications
- Integrity checks on data changes
- Audit logging
- Real-time notifications

Integrates with the existing DAO layer and transaction system.
"""

import asyncio
import threading
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from ..core.result import ErrorCode, Result


class EventType(Enum):
    """Types of events that can occur in the knowledge graph."""

    # Entity events
    ENTITY_ADDED = "entity_added"
    ENTITY_REMOVED = "entity_removed"
    ENTITY_UPDATED = "entity_updated"
    ENTITY_LOADED = "entity_loaded"  # For cache warming

    # Relation events
    RELATION_ADDED = "relation_added"
    RELATION_REMOVED = "relation_removed"
    RELATION_UPDATED = "relation_updated"
    RELATION_LOADED = "relation_loaded"

    # Cluster events
    CLUSTER_ADDED = "cluster_added"
    CLUSTER_REMOVED = "cluster_removed"
    CLUSTER_UPDATED = "cluster_updated"
    CLUSTER_ENTITY_ADDED = "cluster_entity_added"
    CLUSTER_ENTITY_REMOVED = "cluster_entity_removed"

    # Text chunk events
    TEXT_CHUNK_ADDED = "text_chunk_added"
    TEXT_CHUNK_REMOVED = "text_chunk_removed"
    TEXT_CHUNK_UPDATED = "text_chunk_updated"

    # Batch and transaction events
    BATCH_STARTED = "batch_started"
    BATCH_COMMITTED = "batch_committed"
    BATCH_ROLLED_BACK = "batch_rolled_back"
    TRANSACTION_STARTED = "transaction_started"
    TRANSACTION_COMMITTED = "transaction_committed"
    TRANSACTION_ROLLED_BACK = "transaction_rolled_back"

    # System events
    GRAPH_CLEARED = "graph_cleared"
    GRAPH_LOADED = "graph_loaded"
    GRAPH_SAVED = "graph_saved"
    CACHE_CLEARED = "cache_cleared"
    INDEX_REBUILT = "index_rebuilt"

    # Error events
    OPERATION_FAILED = "operation_failed"
    INTEGRITY_VIOLATION = "integrity_violation"
    DEADLOCK_DETECTED = "deadlock_detected"


class EventPriority(Enum):
    """Priority levels for event processing."""

    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class GraphEvent:
    """
    Represents an event in the knowledge graph system.

    Attributes:
        event_type: Type of event
        event_id: Unique identifier for the event
        timestamp: When the event occurred
        source: Source component that generated the event
        target_type: Type of target object (entity, relation, etc.)
        target_id: ID of the target object
        data: Event-specific data
        metadata: Additional metadata
        priority: Event priority for processing order
        transaction_id: Associated transaction ID if any
        thread_id: Thread that generated the event
    """

    event_type: EventType
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source: Optional[str] = None
    target_type: Optional[str] = None
    target_id: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    priority: EventPriority = EventPriority.NORMAL
    transaction_id: Optional[str] = None
    thread_id: Optional[int] = field(default_factory=lambda: threading.current_thread().ident)


class EventListener(ABC):
    """
    Abstract base class for event listeners.

    Event listeners process specific types of events and can be registered
    with the EventManager to receive notifications.
    """

    @abstractmethod
    def handle_event(self, event: GraphEvent) -> Result[bool]:
        """
        Handle an event.

        Args:
            event: The event to handle

        Returns:
            Result indicating success or failure
        """

    @abstractmethod
    def get_event_types(self) -> Set[EventType]:
        """
        Get the types of events this listener is interested in.

        Returns:
            Set of event types this listener handles
        """

    def get_priority(self) -> EventPriority:
        """Get the priority of this listener.

        Higher priority listeners are called first.

        Returns:
            Priority level
        """
        return EventPriority.NORMAL

    def should_handle_event(self, event: GraphEvent) -> bool:
        """
        Determine if this listener should handle a specific event.

        Args:
            event: Event to check

        Returns:
            True if this listener should handle the event
        """
        return event.event_type in self.get_event_types()


class AsyncEventListener(EventListener):
    """
    Base class for asynchronous event listeners.

    Useful for listeners that need to perform I/O operations or
    other async work in response to events.
    """

    @abstractmethod
    async def handle_event_async(self, event: GraphEvent) -> Result[bool]:
        """
        Handle an event asynchronously.

        Args:
            event: The event to handle

        Returns:
            Result indicating success or failure
        """

    def handle_event(self, event: GraphEvent) -> Result[bool]:
        """
        Synchronous wrapper for async event handling.

        Args:
            event: The event to handle

        Returns:
            Result indicating success or failure
        """
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, schedule the coroutine
                asyncio.ensure_future(self.handle_event_async(event))
                # For now, return success - the actual result will be handled asynchronously
                return Result.ok(True)
            # Run in the event loop
            return loop.run_until_complete(self.handle_event_async(event))
        except RuntimeError:
            # No event loop, create one
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(self.handle_event_async(event))
            finally:
                loop.close()


@dataclass
class EventSubscription:
    """
    Represents a subscription to events.

    Attributes:
        subscription_id: Unique identifier for the subscription
        event_types: Types of events to subscribe to
        listener: The listener to notify
        priority: Priority of this subscription
        filter_func: Optional filter function for events
        created_at: When the subscription was created
        active: Whether the subscription is active
    """

    subscription_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_types: Set[EventType] = field(default_factory=set)
    listener: Optional[EventListener] = None
    priority: EventPriority = EventPriority.NORMAL
    filter_func: Optional[Callable[[GraphEvent], bool]] = None
    created_at: float = field(default_factory=time.time)
    active: bool = True


class EventManager:
    """
    Central event manager for the knowledge graph system.

    Manages event subscriptions, dispatching, and processing. Supports both
    synchronous and asynchronous event processing with priority handling.

    Features:
    - Priority-based event processing
    - Synchronous and asynchronous listeners
    - Event filtering
    - Thread-safe operation
    - Event queuing and batching
    - Error handling and recovery
    """

    def __init__(self, max_queue_size: int = 10000, enable_async: bool = True):
        """
        Initialize the event manager.

        Args:
            max_queue_size: Maximum number of events to queue
            enable_async: Whether to enable asynchronous event processing
        """
        self.max_queue_size = max_queue_size
        self.enable_async = enable_async

        # Event subscriptions: event_type -> list of subscriptions
        self._subscriptions: Dict[EventType, List[EventSubscription]] = {}

        # All subscriptions by ID
        self._subscription_by_id: Dict[str, EventSubscription] = {}

        # Event queue for asynchronous processing
        self._event_queue: List[GraphEvent] = []

        # Statistics
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "listeners_count": 0,
            "queue_size": 0,
            "processing_time_total": 0.0,
        }

        # Thread safety
        self._lock = threading.RLock()
        self._processing_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        # Start async processing thread if enabled
        if self.enable_async:
            self._start_processing_thread()

    def subscribe(
        self,
        listener: EventListener,
        event_types: Optional[Set[EventType]] = None,
        priority: Optional[EventPriority] = None,
        filter_func: Optional[Callable[[GraphEvent], bool]] = None,
    ) -> Result[str]:
        """
        Subscribe a listener to events.

        Args:
            listener: The event listener
            event_types: Types of events to subscribe to (defaults to listener's types)
            priority: Priority for processing (defaults to listener's priority)
            filter_func: Optional filter function for events

        Returns:
            Result containing the subscription ID
        """
        with self._lock:
            if event_types is None:
                event_types = listener.get_event_types()

            if priority is None:
                priority = listener.get_priority()

            subscription = EventSubscription(
                event_types=event_types,
                listener=listener,
                priority=priority,
                filter_func=filter_func,
            )

            # Add to subscriptions by event type
            for event_type in event_types:
                if event_type not in self._subscriptions:
                    self._subscriptions[event_type] = []

                self._subscriptions[event_type].append(subscription)

                # Sort by priority (highest first)
                self._subscriptions[event_type].sort(key=lambda s: s.priority.value, reverse=True)

            # Add to subscription lookup
            self._subscription_by_id[subscription.subscription_id] = subscription

            self._stats["listeners_count"] = len(self._subscription_by_id)

            return Result.ok(
                subscription.subscription_id,
                metadata={
                    "event_types": [et.value for et in event_types],
                    "priority": priority.value,
                },
            )

    def unsubscribe(self, subscription_id: str) -> Result[bool]:
        """
        Unsubscribe a listener.

        Args:
            subscription_id: ID of the subscription to remove

        Returns:
            Result indicating success or failure
        """
        with self._lock:
            if subscription_id not in self._subscription_by_id:
                return Result.not_found("Subscription", subscription_id)

            subscription = self._subscription_by_id[subscription_id]
            subscription.active = False

            # Remove from event type subscriptions
            for event_type in subscription.event_types:
                if event_type in self._subscriptions:
                    self._subscriptions[event_type] = [
                        s for s in self._subscriptions[event_type] if s.subscription_id != subscription_id
                    ]

                    # Clean up empty event type entries
                    if not self._subscriptions[event_type]:
                        del self._subscriptions[event_type]

            # Remove from subscription lookup
            del self._subscription_by_id[subscription_id]

            self._stats["listeners_count"] = len(self._subscription_by_id)

            return Result.ok(True)

    def publish(self, event: GraphEvent, synchronous: bool = False) -> Result[bool]:
        """
        Publish an event to all interested listeners.

        Args:
            event: Event to publish
            synchronous: Whether to process the event synchronously

        Returns:
            Result indicating success or failure
        """
        with self._lock:
            self._stats["events_published"] += 1

            if synchronous or not self.enable_async:
                return self._process_event_sync(event)
            # Add to queue for asynchronous processing
            if len(self._event_queue) >= self.max_queue_size:
                return Result.fail(
                    ErrorCode.RESOURCE_EXHAUSTED,
                    f"Event queue is full (max size: {self.max_queue_size})",
                )

            self._event_queue.append(event)
            self._stats["queue_size"] = len(self._event_queue)

            return Result.ok(True)

    def _process_event_sync(self, event: GraphEvent) -> Result[bool]:
        """
        Process an event synchronously.

        Args:
            event: Event to process

        Returns:
            Result indicating success or failure
        """
        start_time = time.time()

        try:
            # Get subscriptions for this event type
            subscriptions = self._subscriptions.get(event.event_type, [])

            processed_count = 0
            failed_count = 0

            for subscription in subscriptions:
                if not subscription.active:
                    continue

                # Apply filter if present
                if subscription.filter_func and not subscription.filter_func(event):
                    continue

                # Check if listener should handle this event
                if subscription.listener and subscription.listener.should_handle_event(event):
                    try:
                        result = subscription.listener.handle_event(event)
                        if result.is_ok():
                            processed_count += 1
                        else:
                            failed_count += 1
                    except Exception:
                        failed_count += 1
                        # Log error but continue processing other listeners

            self._stats["events_processed"] += processed_count
            self._stats["events_failed"] += failed_count
            self._stats["processing_time_total"] += time.time() - start_time

            return Result.ok(
                True,
                metadata={
                    "processed_listeners": processed_count,
                    "failed_listeners": failed_count,
                    "processing_time": time.time() - start_time,
                },
            )

        except Exception as e:
            self._stats["events_failed"] += 1
            return Result.internal_error(e)

    def _start_processing_thread(self) -> None:
        """Start the background event processing thread."""
        if self._processing_thread and self._processing_thread.is_alive():
            return

        self._processing_thread = threading.Thread(
            target=self._process_events_async, name="EventManager-Processor", daemon=True
        )
        self._processing_thread.start()

    def _process_events_async(self) -> None:
        """Background thread for processing events asynchronously."""
        while not self._shutdown_event.is_set():
            try:
                events_to_process = []

                with self._lock:
                    if self._event_queue:
                        # Process events in batches
                        batch_size = min(100, len(self._event_queue))
                        events_to_process = self._event_queue[:batch_size]
                        self._event_queue = self._event_queue[batch_size:]
                        self._stats["queue_size"] = len(self._event_queue)

                # Process events outside the lock
                for event in events_to_process:
                    self._process_event_sync(event)

                # Sleep briefly if no events to process
                if not events_to_process:
                    time.sleep(0.01)

            except Exception:
                # Continue processing even if individual events fail
                time.sleep(0.1)

    def flush_events(self, timeout: float = 5.0) -> Result[int]:
        """
        Flush all queued events synchronously.

        Args:
            timeout: Maximum time to wait for processing

        Returns:
            Result containing number of events processed
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            with self._lock:
                if not self._event_queue:
                    return Result.ok(0)

                events_processed = 0
                events_to_process = self._event_queue.copy()
                self._event_queue.clear()
                self._stats["queue_size"] = 0

            # Process all queued events
            for event in events_to_process:
                result = self._process_event_sync(event)
                if result.is_ok():
                    events_processed += 1

            return Result.ok(events_processed)

        return Result.fail(ErrorCode.TIMEOUT, f"Failed to flush events within {timeout} seconds")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get event manager statistics.

        Returns:
            Dictionary containing statistics
        """
        with self._lock:
            stats = self._stats.copy()
            stats["active_subscriptions"] = len(self._subscription_by_id)
            stats["event_types_registered"] = len(self._subscriptions)
            stats["average_processing_time"] = stats["processing_time_total"] / max(1, stats["events_processed"])
            return stats

    def shutdown(self, timeout: float = 5.0) -> Result[bool]:
        """
        Shutdown the event manager.

        Args:
            timeout: Maximum time to wait for shutdown

        Returns:
            Result indicating success or failure
        """
        # Flush remaining events
        flush_result = self.flush_events(timeout / 2)

        # Signal shutdown
        self._shutdown_event.set()

        # Wait for processing thread to stop
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout / 2)

            if self._processing_thread.is_alive():
                return Result.fail(ErrorCode.TIMEOUT, "Event processing thread did not shutdown within timeout")

        return Result.ok(True, metadata={"events_flushed": flush_result.data or 0})

    @contextmanager
    def event_context(self, source: str) -> Any:
        """
        Context manager for event publishing with automatic source tagging.

        Args:
            source: Source component name

        Yields:
            Event publisher function
        """

        def publish_event(
            event_type: EventType,
            target_type: Optional[str] = None,
            target_id: Optional[str] = None,
            data: Optional[Dict[str, Any]] = None,
            metadata: Optional[Dict[str, Any]] = None,
            priority: EventPriority = EventPriority.NORMAL,
            transaction_id: Optional[str] = None,
        ) -> Result[bool]:
            event = GraphEvent(
                event_type=event_type,
                source=source,
                target_type=target_type,
                target_id=target_id,
                data=data,
                metadata=metadata,
                priority=priority,
                transaction_id=transaction_id,
            )
            return self.publish(event)

        yield publish_event


# Utility functions for creating common events


def create_entity_event(
    event_type: EventType,
    entity_id: str,
    entity_data: Optional[Dict[str, Any]] = None,
    source: str = "unknown",
    transaction_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GraphEvent:
    """Create an entity-related event."""
    return GraphEvent(
        event_type=event_type,
        source=source,
        target_type="entity",
        target_id=entity_id,
        data=entity_data,
        metadata=metadata,
        transaction_id=transaction_id,
    )


def create_relation_event(
    event_type: EventType,
    relation_id: str,
    relation_data: Optional[Dict[str, Any]] = None,
    source: str = "unknown",
    transaction_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> GraphEvent:
    """Create a relation-related event."""
    return GraphEvent(
        event_type=event_type,
        source=source,
        target_type="relation",
        target_id=relation_id,
        data=relation_data,
        metadata=metadata,
        transaction_id=transaction_id,
    )


def create_system_event(
    event_type: EventType,
    source: str = "system",
    data: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    priority: EventPriority = EventPriority.NORMAL,
) -> GraphEvent:
    """Create a system-related event."""
    return GraphEvent(event_type=event_type, source=source, data=data, metadata=metadata, priority=priority)
