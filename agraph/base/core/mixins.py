"""
Common mixins for the agraph package.

This module provides reusable mixins for common functionality across the package.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Union


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


class ImportExportMixin(ABC):
    """Mixin for objects that support import/export to various formats.

    This mixin provides a base interface for importing and exporting data
    in different formats. Currently supports JSON format with extensibility
    for future formats like GraphML.
    """

    # JSON Format Support
    def export_to_json(self, file_path: Union[str, Path], **kwargs: Any) -> None:
        """Export the object to a JSON file.

        Args:
            file_path: Path where the JSON file will be saved
            **kwargs: Additional arguments passed to json.dump()
        """
        file_path = Path(file_path)

        # Ensure directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Set default JSON export options
        json_kwargs = {"indent": 2, "ensure_ascii": False, "sort_keys": True}
        json_kwargs.update(kwargs)

        data = self.to_dict() if hasattr(self, "to_dict") else self._export_data()

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, **json_kwargs)  # type: ignore[arg-type]

    @classmethod
    def import_from_json(cls, file_path: Union[str, Path], **kwargs: Any) -> "ImportExportMixin":
        """Import an object from a JSON file.

        Args:
            file_path: Path to the JSON file to import
            **kwargs: Additional arguments for object creation

        Returns:
            Instance of the class created from the JSON data
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSON file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if hasattr(cls, "from_dict"):
            return cls.from_dict(data, **kwargs)  # type: ignore[no-any-return]
        return cls._import_data(data, **kwargs)

    def export_to_json_string(self, **kwargs: Any) -> str:
        """Export the object to a JSON string.

        Args:
            **kwargs: Additional arguments passed to json.dumps()

        Returns:
            JSON string representation of the object
        """
        json_kwargs = {"indent": 2, "ensure_ascii": False, "sort_keys": True}
        json_kwargs.update(kwargs)

        data = self.to_dict() if hasattr(self, "to_dict") else self._export_data()
        return json.dumps(data, **json_kwargs)  # type: ignore[arg-type]

    @classmethod
    def import_from_json_string(cls, json_string: str, **kwargs: Any) -> "ImportExportMixin":
        """Import an object from a JSON string.

        Args:
            json_string: JSON string to import
            **kwargs: Additional arguments for object creation

        Returns:
            Instance of the class created from the JSON data
        """
        data = json.loads(json_string)

        if hasattr(cls, "from_dict"):
            return cls.from_dict(data, **kwargs)  # type: ignore[no-any-return]
        return cls._import_data(data, **kwargs)

    # Extension Interface for Future Formats
    @abstractmethod
    def _export_data(self) -> Dict[str, Any]:
        """Export data to dictionary format.

        This method should be implemented by subclasses that don't have
        a to_dict() method from SerializableMixin.

        Returns:
            Dictionary representation of the object
        """

    @classmethod
    @abstractmethod
    def _import_data(cls, data: Dict[str, Any], **kwargs: Any) -> "ImportExportMixin":
        """Import data from dictionary format.

        This method should be implemented by subclasses that don't have
        a from_dict() class method from SerializableMixin.

        Args:
            data: Dictionary containing the object data
            **kwargs: Additional arguments for object creation

        Returns:
            Instance of the class created from the dictionary data
        """

    # Future format support methods (to be implemented in subclasses)
    def export_to_graphml(self, file_path: Union[str, Path], **kwargs: Any) -> None:
        """Export the object to GraphML format.

        This method should be implemented by subclasses that support GraphML export.

        Args:
            file_path: Path where the GraphML file will be saved
            **kwargs: Additional arguments for GraphML export
        """
        _ = file_path, kwargs  # Suppress unused variable warnings
        raise NotImplementedError("GraphML export not implemented for this class")

    @classmethod
    def import_from_graphml(cls, file_path: Union[str, Path], **kwargs: Any) -> "ImportExportMixin":
        """Import an object from GraphML format.

        This method should be implemented by subclasses that support GraphML import.

        Args:
            file_path: Path to the GraphML file to import
            **kwargs: Additional arguments for object creation

        Returns:
            Instance of the class created from the GraphML data
        """
        _ = file_path, kwargs  # Suppress unused variable warnings
        raise NotImplementedError("GraphML import not implemented for this class")
