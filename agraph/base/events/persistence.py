"""
Event persistence mechanisms for AGraph.

This module provides optional event persistence capabilities for:
- Audit logging
- Event replay
- System debugging
- Compliance requirements
- Long-term analytics

Events can be persisted to various backends including files, databases, or message queues.
"""

import json
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from ..core.result import ErrorCode, Result
from .events import EventListener, EventPriority, EventType, GraphEvent

if TYPE_CHECKING:
    from .events import EventManager


class EventPersistenceBackend(ABC):
    """
    Abstract base class for event persistence backends.

    Different backends can be implemented for various storage systems:
    - File system (JSON, CSV, etc.)
    - Databases (SQLite, PostgreSQL, etc.)
    - Message queues (Redis, RabbitMQ, etc.)
    - Cloud services (AWS S3, Google Cloud Storage, etc.)
    """

    @abstractmethod
    def persist_event(self, event: GraphEvent) -> Result[bool]:
        """
        Persist a single event.

        Args:
            event: The event to persist

        Returns:
            Result indicating success or failure
        """

    @abstractmethod
    def persist_events_batch(self, events: List[GraphEvent]) -> Result[int]:
        """
        Persist a batch of events.

        Args:
            events: List of events to persist

        Returns:
            Result containing the number of successfully persisted events
        """

    @abstractmethod
    def retrieve_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        limit: Optional[int] = None,
    ) -> Result[List[GraphEvent]]:
        """
        Retrieve events based on criteria.

        Args:
            start_time: Start timestamp (inclusive)
            end_time: End timestamp (exclusive)
            event_types: Filter by event types
            limit: Maximum number of events to return

        Returns:
            Result containing list of events
        """

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get persistence backend statistics.

        Returns:
            Dictionary containing backend statistics
        """

    @abstractmethod
    def cleanup(self, older_than_seconds: Optional[float] = None) -> Result[int]:
        """
        Clean up old events.

        Args:
            older_than_seconds: Remove events older than this many seconds

        Returns:
            Result containing number of cleaned up events
        """


class JSONFileBackend(EventPersistenceBackend):
    """
    File-based event persistence using JSON format.

    Events are stored in JSON files with configurable rotation policies.
    This backend is suitable for:
    - Development and testing
    - Small to medium event volumes
    - Simple audit requirements
    """

    def __init__(
        self,
        storage_dir: Union[str, Path],
        max_file_size_mb: int = 100,
        max_files: int = 10,
        compress_old_files: bool = True,
    ):
        """
        Initialize JSON file backend.

        Args:
            storage_dir: Directory to store event files
            max_file_size_mb: Maximum size per file in MB
            max_files: Maximum number of files to keep
            compress_old_files: Whether to compress rotated files
        """
        self.storage_dir = Path(storage_dir)
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.max_files = max_files
        self.compress_old_files = compress_old_files

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Current file info
        self.current_file_path = self._get_current_file_path()
        self.current_file_size = self._get_file_size(self.current_file_path)

        # Thread safety
        self._lock = threading.RLock()

        # Statistics
        self.stats: Dict[str, int] = {
            "events_written": 0,
            "files_created": 0,
            "files_rotated": 0,
            "write_errors": 0,
            "total_bytes_written": 0,
        }

    def _get_current_file_path(self) -> Path:
        """Get the path for the current event file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.storage_dir / f"events_{timestamp}.jsonl"

    def _get_file_size(self, file_path: Path) -> int:
        """Get the size of a file in bytes."""
        try:
            return file_path.stat().st_size if file_path.exists() else 0
        except OSError:
            return 0

    def _should_rotate_file(self) -> bool:
        """Check if the current file should be rotated."""
        return self.current_file_size >= self.max_file_size_bytes

    def _rotate_file(self) -> None:
        """Rotate the current file and create a new one."""
        if self.compress_old_files and self.current_file_path.exists():
            # In a real implementation, you'd compress the file here
            pass

        # Create new current file
        self.current_file_path = self._get_current_file_path()
        self.current_file_size = 0
        self.stats["files_rotated"] += 1

        # Clean up old files
        self._cleanup_old_files()

    def _cleanup_old_files(self) -> None:
        """Remove old event files beyond the retention limit."""
        event_files = sorted(
            list(self.storage_dir.glob("events_*.jsonl*")),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        # Keep only max_files most recent files
        for old_file in event_files[self.max_files :]:
            try:
                old_file.unlink()
            except OSError:
                pass  # Ignore errors during cleanup

    def persist_event(self, event: GraphEvent) -> Result[bool]:
        """Persist a single event to JSON file."""
        with self._lock:
            try:
                # Check if file rotation is needed
                if self._should_rotate_file():
                    self._rotate_file()

                # Convert event to dictionary
                event_dict = asdict(event)

                # Add ISO timestamp for better readability
                event_dict["timestamp_iso"] = datetime.fromtimestamp(event.timestamp, tz=timezone.utc).isoformat()

                # Convert enum values to strings
                event_dict["event_type"] = event.event_type.value
                event_dict["priority"] = event.priority.value

                # Write to file
                event_json = json.dumps(event_dict, separators=(",", ":"))

                with open(self.current_file_path, "a", encoding="utf-8") as f:
                    f.write(event_json + "\n")

                # Update statistics
                bytes_written = len(event_json.encode("utf-8")) + 1  # +1 for newline
                self.current_file_size += bytes_written
                self.stats["events_written"] += 1
                self.stats["total_bytes_written"] += bytes_written

                return Result.ok(True)

            except Exception as e:
                self.stats["write_errors"] += 1
                return Result.internal_error(e)

    def persist_events_batch(self, events: List[GraphEvent]) -> Result[int]:
        """Persist a batch of events."""
        successful_count = 0

        for event in events:
            result = self.persist_event(event)
            if result.is_ok():
                successful_count += 1

        return Result.ok(successful_count)

    def retrieve_events(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_types: Optional[List[EventType]] = None,
        limit: Optional[int] = None,
    ) -> Result[List[GraphEvent]]:
        """Retrieve events from JSON files."""
        with self._lock:
            try:
                events: List[GraphEvent] = []
                event_files = sorted(
                    list(self.storage_dir.glob("events_*.jsonl*")),
                    key=lambda f: f.stat().st_mtime,
                )

                event_type_values = [et.value for et in event_types] if event_types else None

                for file_path in event_files:
                    if limit and len(events) >= limit:
                        break

                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            for line in f:
                                if limit and len(events) >= limit:
                                    break

                                line = line.strip()
                                if not line:
                                    continue

                                try:
                                    event_dict = json.loads(line)

                                    # Apply filters
                                    event_timestamp = event_dict.get("timestamp", 0)
                                    if start_time and event_timestamp < start_time:
                                        continue
                                    if end_time and event_timestamp >= end_time:
                                        continue
                                    if event_type_values and event_dict.get("event_type") not in event_type_values:
                                        continue

                                    # Convert back to GraphEvent
                                    event = self._dict_to_event(event_dict)
                                    if event:
                                        events.append(event)

                                except (json.JSONDecodeError, KeyError, ValueError):
                                    continue  # Skip malformed events

                    except OSError:
                        continue  # Skip files that can't be opened

                return Result.ok(events)

            except Exception as e:
                return Result.internal_error(e)

    def _dict_to_event(self, event_dict: Dict[str, Any]) -> Optional[GraphEvent]:
        """Convert a dictionary back to a GraphEvent."""
        try:
            # Convert enum values back
            event_type = EventType(event_dict["event_type"])
            priority = EventPriority(event_dict.get("priority", EventPriority.NORMAL.value))

            # Remove ISO timestamp (we use the numeric one)
            event_dict.pop("timestamp_iso", None)

            return GraphEvent(
                event_type=event_type,
                event_id=event_dict.get("event_id", ""),
                timestamp=event_dict.get("timestamp", time.time()),
                source=event_dict.get("source"),
                target_type=event_dict.get("target_type"),
                target_id=event_dict.get("target_id"),
                data=event_dict.get("data"),
                metadata=event_dict.get("metadata"),
                priority=priority,
                transaction_id=event_dict.get("transaction_id"),
                thread_id=event_dict.get("thread_id"),
            )
        except (KeyError, ValueError, TypeError):
            return None

    def get_statistics(self) -> Dict[str, Any]:
        """Get backend statistics."""
        with self._lock:
            stats_copy: Dict[str, Any] = dict(self.stats)
            stats_copy.update(
                {
                    "storage_directory": str(self.storage_dir),
                    "current_file": str(self.current_file_path),
                    "current_file_size_bytes": float(self.current_file_size),
                    "total_files": len(list(self.storage_dir.glob("events_*.jsonl*"))),
                    "average_event_size_bytes": (
                        self.stats["total_bytes_written"] / max(1.0, float(self.stats["events_written"]))
                    ),
                }
            )
            return stats_copy

    def cleanup(self, older_than_seconds: Optional[float] = None) -> Result[int]:
        """Clean up old event files."""
        with self._lock:
            try:
                if older_than_seconds is None:
                    older_than_seconds = 30 * 24 * 3600  # 30 days default

                cutoff_time = time.time() - older_than_seconds
                cleaned_count = 0

                event_files = list(self.storage_dir.glob("events_*.jsonl*"))
                for file_path in event_files:
                    try:
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            cleaned_count += 1
                    except OSError:
                        continue

                return Result.ok(cleaned_count)

            except Exception as e:
                return Result.internal_error(e)


class EventPersistenceListener(EventListener):
    """
    Event listener that persists events to a backend.

    This listener can be configured to persist all events or only
    specific event types based on filters.
    """

    def __init__(
        self,
        backend: EventPersistenceBackend,
        event_types: Optional[List[EventType]] = None,
        min_priority: EventPriority = EventPriority.LOW,
        batch_size: int = 100,
        batch_timeout_seconds: float = 10.0,
    ):
        """
        Initialize the persistence listener.

        Args:
            backend: The persistence backend to use
            event_types: Event types to persist (None = all types)
            min_priority: Minimum priority to persist
            batch_size: Number of events to batch before writing
            batch_timeout_seconds: Maximum time to wait before flushing batch
        """
        self.backend = backend
        self.event_types = set(event_types) if event_types else None
        self.min_priority = min_priority
        self.batch_size = batch_size
        self.batch_timeout_seconds = batch_timeout_seconds

        # Batching
        self.event_batch: List[GraphEvent] = []
        self.last_flush_time = time.time()
        self._lock = threading.RLock()

    def get_event_types(self) -> set[EventType]:
        """Get event types this listener handles."""
        if self.event_types:
            return self.event_types
        # Return all event types if none specified
        return set(EventType)

    def should_handle_event(self, event: GraphEvent) -> bool:
        """Check if this event should be persisted."""
        # Check priority
        if event.priority.value < self.min_priority.value:
            return False

        # Check event type
        if self.event_types and event.event_type not in self.event_types:
            return False

        return True

    def handle_event(self, event: GraphEvent) -> Result[bool]:
        """Handle an event by adding it to the persistence batch."""
        if not self.should_handle_event(event):
            return Result.ok(True)

        with self._lock:
            self.event_batch.append(event)

            # Check if we should flush the batch
            should_flush = (
                len(self.event_batch) >= self.batch_size
                or time.time() - self.last_flush_time >= self.batch_timeout_seconds
            )

            if should_flush:
                return self._flush_batch()

            return Result.ok(True)

    def _flush_batch(self) -> Result[bool]:
        """Flush the current batch to the backend."""
        if not self.event_batch:
            return Result.ok(True)

        events_to_persist = self.event_batch.copy()
        self.event_batch.clear()
        self.last_flush_time = time.time()

        # Persist the batch
        result = self.backend.persist_events_batch(events_to_persist)

        if result.is_ok():
            return Result.ok(True, metadata={"persisted_count": result.data})
        # Put events back in batch if persistence failed
        self.event_batch.extend(events_to_persist)
        return Result.fail(
            result.error_code or ErrorCode.INTERNAL_ERROR,
            result.error_message or "Batch persistence failed",
        )

    def flush(self) -> Result[bool]:
        """Manually flush the current batch."""
        with self._lock:
            return self._flush_batch()

    @contextmanager
    def auto_flush_context(self) -> Any:
        """Context manager that ensures batch is flushed on exit."""
        try:
            yield self
        finally:
            self.flush()


# Factory functions


def create_json_file_backend(
    storage_dir: Union[str, Path], max_file_size_mb: int = 100, max_files: int = 10
) -> JSONFileBackend:
    """Create a JSON file persistence backend."""
    return JSONFileBackend(storage_dir, max_file_size_mb, max_files)


def create_persistence_listener(
    backend: EventPersistenceBackend,
    event_types: Optional[List[EventType]] = None,
    min_priority: EventPriority = EventPriority.LOW,
) -> EventPersistenceListener:
    """Create an event persistence listener."""
    return EventPersistenceListener(backend, event_types, min_priority)


def setup_event_persistence(
    event_manager: "EventManager",
    storage_dir: Union[str, Path],
    event_types: Optional[List[EventType]] = None,
    min_priority: EventPriority = EventPriority.LOW,
) -> Result[Dict[str, Any]]:
    """
    Set up event persistence with default JSON file backend.

    Args:
        event_manager: The event manager to add persistence to
        storage_dir: Directory to store event files
        event_types: Event types to persist (None = all)
        min_priority: Minimum priority to persist

    Returns:
        Result containing setup information
    """
    try:
        # Create backend and listener
        backend = create_json_file_backend(storage_dir)
        listener = create_persistence_listener(backend, event_types, min_priority)

        # Subscribe to events
        subscription_result = event_manager.subscribe(listener)

        if subscription_result.is_ok():
            return Result.ok(
                {
                    "subscription_id": subscription_result.data,
                    "backend_type": "json_file",
                    "storage_directory": str(storage_dir),
                    "event_types": [et.value for et in event_types] if event_types else "all",
                    "min_priority": min_priority.value,
                }
            )
        return Result.fail(
            subscription_result.error_code or ErrorCode.INTERNAL_ERROR,
            subscription_result.error_message or "Subscription failed",
        )

    except Exception as e:
        return Result.internal_error(e)


# Utility functions for event analysis


def analyze_persisted_events(
    backend: EventPersistenceBackend,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Result[Dict[str, Any]]:
    """
    Analyze persisted events and generate statistics.

    Args:
        backend: The persistence backend to analyze
        start_time: Start time for analysis
        end_time: End time for analysis

    Returns:
        Result containing analysis statistics
    """
    try:
        # Retrieve events
        events_result = backend.retrieve_events(start_time, end_time)
        if not events_result.is_ok():
            return Result.fail(
                events_result.error_code or ErrorCode.INTERNAL_ERROR,
                events_result.error_message or "Failed to retrieve events",
            )

        events = events_result.data

        if not events:
            return Result.ok({"total_events": 0})

        # Analyze events
        event_type_counts: Dict[str, int] = {}
        priority_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        hourly_counts: Dict[str, int] = {}

        for event in events:
            # Event type counts
            event_type = event.event_type.value
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

            # Priority counts
            priority = str(event.priority.value)
            priority_counts[priority] = priority_counts.get(priority, 0) + 1

            # Source counts
            source = event.source or "unknown"
            source_counts[source] = source_counts.get(source, 0) + 1

            # Hourly counts
            hour = datetime.fromtimestamp(event.timestamp).strftime("%Y-%m-%d %H:00")
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1

        analysis = {
            "total_events": len(events),
            "time_range": {
                "start": events[0].timestamp if events else None,
                "end": events[-1].timestamp if events else None,
                "duration_seconds": ((events[-1].timestamp - events[0].timestamp) if len(events) > 1 else 0),
            },
            "event_type_distribution": event_type_counts,
            "priority_distribution": priority_counts,
            "source_distribution": source_counts,
            "hourly_distribution": hourly_counts,
            "most_common_event_type": (
                max(event_type_counts, key=lambda x: event_type_counts[x]) if event_type_counts else None
            ),
            "most_active_source": (max(source_counts, key=lambda x: source_counts[x]) if source_counts else None),
        }

        return Result.ok(analysis)

    except Exception as e:
        return Result.internal_error(e)
