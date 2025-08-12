"""
Common mixins for the agraph package.

This module provides reusable mixins for common functionality across the package.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict


class SerializableMixin(ABC):
    """Mixin for objects that can be serialized to/from dictionaries."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert object to dictionary representation."""

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs: Any) -> "SerializableMixin":
        """Create object from dictionary representation."""


class TimestampMixin:
    """Mixin for objects that track creation and update timestamps."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        now = datetime.now()
        if not hasattr(self, "created_at"):
            self.created_at = now
        if not hasattr(self, "updated_at"):
            self.updated_at = now

    def touch(self) -> None:
        """Update the timestamp to current time."""
        self.updated_at = datetime.now()


class PropertyMixin:
    """Mixin for objects that support dynamic properties."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if not hasattr(self, "properties"):
            self.properties: Dict[str, Any] = {}

    def set_property(self, key: str, value: Any) -> None:
        """Set a property value."""
        self.properties[key] = value
        if hasattr(self, "touch"):
            self.touch()

    def get_property(self, key: str, default: Any = None) -> Any:
        """Get a property value."""
        return self.properties.get(key, default)

    def has_property(self, key: str) -> bool:
        """Check if property exists."""
        return key in self.properties

    def remove_property(self, key: str) -> None:
        """Remove a property."""
        if key in self.properties:
            del self.properties[key]
            if hasattr(self, "touch"):
                self.touch()
