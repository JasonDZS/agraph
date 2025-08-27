"""
Base cache backend interface.
"""

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, TypeVar

if TYPE_CHECKING:
    from ...config import CacheMetadata

T = TypeVar("T")


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    def __init__(self, cache_dir: str):
        """Initialize cache backend.

        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir

    @abstractmethod
    def get(self, key: str, expected_type: Type[T]) -> Optional[T]:
        """Get cached value by key.

        Args:
            key: Cache key
            expected_type: Expected type of cached value

        Returns:
            Cached value or None if not found
        """

    @abstractmethod
    def set(self, key: str, value: Any, metadata: Optional["CacheMetadata"] = None) -> None:
        """Set cached value.

        Args:
            key: Cache key
            value: Value to cache
            metadata: Optional cache metadata
        """

    @abstractmethod
    def has(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists
        """

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete cached value.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """

    @abstractmethod
    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache.

        Args:
            pattern: Optional pattern to match keys

        Returns:
            Number of deleted keys
        """

    @abstractmethod
    def get_metadata(self, key: str) -> Optional["CacheMetadata"]:
        """Get cache metadata.

        Args:
            key: Cache key

        Returns:
            Cache metadata or None if not found
        """

    @abstractmethod
    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List cache keys.

        Args:
            pattern: Optional pattern to match keys

        Returns:
            List of cache keys
        """

    @abstractmethod
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information.

        Returns:
            Dictionary with cache statistics
        """

    def generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments.

        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Generated cache key
        """
        # Create a deterministic string from arguments
        key_parts = []

        for arg in args:
            if hasattr(arg, "__dict__"):
                # For objects, use their dict representation
                key_parts.append(json.dumps(arg.__dict__, sort_keys=True, default=str))
            elif isinstance(arg, (list, tuple)):
                # For sequences, convert to string
                key_parts.append(json.dumps(arg, sort_keys=True, default=str))
            elif isinstance(arg, dict):
                # For dicts, sort keys
                key_parts.append(json.dumps(arg, sort_keys=True, default=str))
            else:
                key_parts.append(str(arg))

        for key, value in sorted(kwargs.items()):
            key_parts.append(f"{key}:{value}")

        # Create hash from combined parts
        combined = "|".join(key_parts)
        return hashlib.sha256(combined.encode()).hexdigest()

    def is_expired(self, metadata: "CacheMetadata", ttl: int) -> bool:
        """Check if cache entry is expired.

        Args:
            metadata: Cache metadata
            ttl: Time to live in seconds

        Returns:
            True if expired
        """
        if ttl <= 0:
            return False

        now = datetime.now()
        age = (now - metadata.timestamp).total_seconds()
        return age > ttl
