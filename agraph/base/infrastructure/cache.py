"""
Cache management system for AGraph knowledge graph.

This module provides caching capabilities to optimize expensive operations
like graph statistics calculation and complex queries.
"""

import threading
import time
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, Set, TypeVar, cast

# Type variable for generic cache values
T = TypeVar("T")
F = TypeVar("F", bound=Callable[..., Any])


class CacheStrategy(Enum):
    """Cache eviction strategies."""

    LRU = "lru"  # Least Recently Used
    TTL = "ttl"  # Time To Live
    LRU_TTL = "lru_ttl"  # Combined LRU and TTL


@dataclass
class CacheEntry:
    """Represents a single cache entry with metadata."""

    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    ttl: Optional[float] = None
    tags: Optional[Set[str]] = None

    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl is None:
            return False
        return (time.time() - self.created_at) > self.ttl

    def touch(self) -> None:
        """Update the last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class CacheManager:
    """
    Thread-safe cache manager with multiple eviction strategies.

    Supports LRU (Least Recently Used), TTL (Time To Live), and combined strategies
    for optimizing expensive operations in the knowledge graph.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl: Optional[float] = None,
        strategy: CacheStrategy = CacheStrategy.LRU_TTL,
    ):
        """
        Initialize the CacheManager.

        Args:
            max_size: Maximum number of entries in the cache
            default_ttl: Default TTL in seconds for cache entries
            strategy: Cache eviction strategy
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.strategy = strategy

        # Cache storage
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: list[str] = []  # For LRU tracking

        # Thread safety
        self._lock = threading.RWLock() if hasattr(threading, "RWLock") else threading.Lock()

        # Cache statistics
        self._stats = {"hits": 0, "misses": 0, "evictions": 0, "expired_entries": 0}

        # Dirty flags for invalidation
        self._dirty_tags: Set[str] = set()

    def _with_write_lock(self, func: F) -> F:
        """Decorator to ensure write operations are thread-safe."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if hasattr(self._lock, "writer"):
                with self._lock.writer():
                    return func(*args, **kwargs)
            else:
                with self._lock:
                    return func(*args, **kwargs)

        return cast(F, wrapper)

    def _with_read_lock(self, func: F) -> F:
        """Decorator to ensure read operations are thread-safe."""

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if hasattr(self._lock, "reader"):
                with self._lock.reader():
                    return func(*args, **kwargs)
            else:
                with self._lock:
                    return func(*args, **kwargs)

        return cast(F, wrapper)

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""

        @self._with_read_lock
        def _get() -> Optional[Any]:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            # Check if expired
            if entry.is_expired():
                # Remove expired entry (will be done in write operation)
                self._remove_expired_entry(key)
                self._stats["misses"] += 1
                self._stats["expired_entries"] += 1
                return None

            # Update access information
            entry.touch()

            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

            self._stats["hits"] += 1
            return entry.value

        return _get()

    def put(self, key: str, value: Any, ttl: Optional[float] = None, tags: Optional[Set[str]] = None) -> None:
        """Put a value into the cache."""

        @self._with_write_lock
        def _put() -> None:
            current_time = time.time()
            entry_ttl = ttl if ttl is not None else self.default_ttl

            # Create new cache entry
            entry = CacheEntry(
                value=value,
                created_at=current_time,
                last_accessed=current_time,
                access_count=1,
                ttl=entry_ttl,
            )

            # Store tags with the entry for invalidation
            if tags:
                entry.tags = tags

            # Check if we need to evict entries
            if key not in self._cache and len(self._cache) >= self.max_size:
                self._evict_entries()

            # Add/update entry
            self._cache[key] = entry

            # Update LRU order
            if key in self._access_order:
                self._access_order.remove(key)
            self._access_order.append(key)

        _put()

    def _remove_expired_entry(self, key: str) -> None:
        """Remove an expired entry from the cache."""

        @self._with_write_lock
        def _remove() -> None:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

        _remove()

    def _evict_entries(self) -> None:
        """Evict entries based on the configured strategy."""
        if self.strategy in [CacheStrategy.LRU, CacheStrategy.LRU_TTL]:
            # Remove least recently used entries
            entries_to_remove = max(1, len(self._cache) // 10)  # Remove 10% of entries

            for _ in range(entries_to_remove):
                if self._access_order:
                    lru_key = self._access_order.pop(0)
                    if lru_key in self._cache:
                        del self._cache[lru_key]
                        self._stats["evictions"] += 1

        elif self.strategy == CacheStrategy.TTL:
            # Remove expired entries first
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                self._stats["evictions"] += 1
                self._stats["expired_entries"] += 1

    def invalidate(self, key: str) -> bool:
        """Invalidate a specific cache entry."""

        @self._with_write_lock
        def _invalidate() -> bool:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
                return True
            return False

        return _invalidate()

    def invalidate_by_tags(self, tags: Set[str]) -> int:
        """Invalidate cache entries by tags."""

        @self._with_write_lock
        def _invalidate_by_tags() -> int:
            keys_to_remove = []

            for key, entry in self._cache.items():
                if hasattr(entry, "tags") and entry.tags:
                    if any(tag in tags for tag in entry.tags):
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

            return len(keys_to_remove)

        return _invalidate_by_tags()

    def clear(self) -> None:
        """Clear all cache entries."""

        @self._with_write_lock
        def _clear() -> None:
            self._cache.clear()
            self._access_order.clear()
            self._dirty_tags.clear()

        _clear()

    def cleanup_expired(self) -> int:
        """Remove all expired entries and return the count."""

        @self._with_write_lock
        def _cleanup() -> int:
            expired_keys = []

            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)

            self._stats["expired_entries"] += len(expired_keys)
            return len(expired_keys)

        return _cleanup()

    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""

        @self._with_read_lock
        def _get_stats() -> Dict[str, Any]:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_ratio = self._stats["hits"] / total_requests if total_requests > 0 else 0.0

            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_ratio": hit_ratio,
                "evictions": self._stats["evictions"],
                "expired_entries": self._stats["expired_entries"],
                "strategy": self.strategy.value,
                "default_ttl": self.default_ttl,
            }

        return _get_stats()


def cached(
    cache_manager: CacheManager,
    key_func: Optional[Callable] = None,
    ttl: Optional[float] = None,
    tags: Optional[Set[str]] = None,
) -> Callable:
    """
    Decorator to cache function results.

    Args:
        cache_manager: The cache manager to use
        key_func: Function to generate cache key from function args
        ttl: Time to live for the cached result
        tags: Tags for cache invalidation

    Returns:
        Decorated function that uses caching
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation - handle unhashable objects like self
                try:
                    # Try to hash args and kwargs
                    cache_key = f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
                except TypeError:
                    # If args contain unhashable types, create key from hashable parts
                    hashable_args = []
                    for arg in args:
                        if hasattr(arg, "id") and hasattr(arg, "__class__"):
                            # Use object type and id for instances
                            hashable_args.append(f"{arg.__class__.__name__}_{getattr(arg, 'id', id(arg))}")
                        else:
                            try:
                                hashable_args.append(str(hash(arg)))
                            except TypeError:
                                hashable_args.append(str(type(arg).__name__))

                    hashable_kwargs = [(k, str(v)) for k, v in sorted(kwargs.items())]
                    cache_key = f"{func.__name__}:{hash((tuple(hashable_args), tuple(hashable_kwargs)))}"

            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.put(cache_key, result, ttl=ttl, tags=tags)
            return result

        # Add cache management methods to the function
        setattr(wrapper, "cache_manager", cache_manager)
        setattr(
            wrapper,
            "invalidate",
            lambda *args, **kwargs: cache_manager.invalidate(
                key_func(*args, **kwargs)
                if key_func
                else f"{func.__name__}:{hash((args, tuple(sorted(kwargs.items()))))}"
            ),
        )

        return wrapper

    return decorator


# Global cache manager instance
_global_cache_manager: Optional[CacheManager] = None


def get_global_cache_manager() -> CacheManager:
    """Get the global CacheManager instance."""
    # pylint: disable=global-statement
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager()
    return _global_cache_manager


def set_global_cache_manager(cache_manager: CacheManager) -> None:
    """Set the global CacheManager instance."""
    # pylint: disable=global-statement
    global _global_cache_manager
    _global_cache_manager = cache_manager
