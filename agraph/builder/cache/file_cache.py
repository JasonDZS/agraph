"""
File-based cache backend implementation.
"""

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

from ...base.clusters import Cluster
from ...base.entities import Entity
from ...base.graph import KnowledgeGraph
from ...base.relations import Relation
from ...base.text import TextChunk
from ...config import CacheMetadata
from .base import CacheBackend

T = TypeVar("T")


class FileCacheBackend(CacheBackend):
    """File-based cache backend with grouped storage for similar data types."""

    def __init__(self, cache_dir: str):
        """Initialize file cache backend.

        Args:
            cache_dir: Directory to store cache files
        """
        super().__init__(cache_dir)
        self.cache_path = Path(cache_dir)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.steps_dir = self.cache_path / "steps"
        self.metadata_dir = self.cache_path / "metadata"
        self.user_edits_dir = self.cache_path / "user_edits"
        self.grouped_dir = self.cache_path / "grouped"

        for directory in [self.steps_dir, self.metadata_dir, self.user_edits_dir, self.grouped_dir]:
            directory.mkdir(exist_ok=True)

        # Define data types that should be grouped together
        self.grouped_types = {
            "TextChunk": "text_chunks",
            "Entity": "entities",
            "Relation": "relations",
            "Cluster": "clusters",
            "DocumentText": "document_texts",
        }

    def get(self, key: str, expected_type: Type[T]) -> Optional[T]:
        """Get cached value by key."""
        # Check if this is a grouped data type
        if self._should_use_grouped_storage(key):
            return self._get_from_grouped_cache(key, expected_type)

        # Fall back to individual file storage
        cache_file = self._get_cache_file(key)
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            return self._deserialize_data(data, expected_type)
        except Exception:
            # If deserialization fails, remove invalid cache
            cache_file.unlink(missing_ok=True)
            return None

    def set(self, key: str, value: Any, metadata: Optional[CacheMetadata] = None) -> None:
        """Set cached value."""
        # Check if this should use grouped storage
        if self._should_use_grouped_storage(key):
            self._set_in_grouped_cache(key, value, metadata)
            return

        # Fall back to individual file storage
        cache_file = self._get_cache_file(key)
        cache_file.parent.mkdir(parents=True, exist_ok=True)

        # Serialize data
        serialized_data = self._serialize_data(value)

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(serialized_data, f, indent=2, ensure_ascii=False)

        # Save metadata if provided
        if metadata:
            metadata_file = self._get_metadata_file(key)
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)

    def has(self, key: str) -> bool:
        """Check if key exists in cache."""
        if self._should_use_grouped_storage(key):
            return self._has_in_grouped_cache(key)
        return self._get_cache_file(key).exists()

    def delete(self, key: str) -> bool:
        """Delete cached value."""
        if self._should_use_grouped_storage(key):
            return self._delete_from_grouped_cache(key)

        cache_file = self._get_cache_file(key)
        metadata_file = self._get_metadata_file(key)

        deleted = False

        if cache_file.exists():
            cache_file.unlink()
            deleted = True

        if metadata_file.exists():
            metadata_file.unlink()

        return deleted

    def clear(self, pattern: Optional[str] = None) -> int:
        """Clear cache."""
        if pattern is None:
            # Clear entire cache
            if self.cache_path.exists():
                shutil.rmtree(self.cache_path)
                FileCacheBackend.__init__(self, str(self.cache_path))
            return -1  # Unknown count

        # Clear files matching pattern
        deleted_count = 0

        # Clear individual files
        for cache_file in self.cache_path.rglob("*.json"):
            if cache_file.parent.name == "grouped":
                continue  # Handle grouped files separately

            if pattern in cache_file.name:
                cache_file.unlink()
                deleted_count += 1

        # Clear items from grouped files
        if self.grouped_dir.exists():
            for grouped_file in self.grouped_dir.glob("*.json"):
                try:
                    with open(grouped_file, "r", encoding="utf-8") as f:
                        grouped_data = json.load(f)

                    # Remove items matching pattern
                    keys_to_remove = [key for key in grouped_data.keys() if pattern in key]
                    for key in keys_to_remove:
                        del grouped_data[key]
                        deleted_count += 1

                        # Delete metadata
                        metadata_file = self._get_metadata_file(key)
                        if metadata_file.exists():
                            metadata_file.unlink()

                    # Update grouped file or remove if empty
                    if grouped_data:
                        with open(grouped_file, "w", encoding="utf-8") as f:
                            json.dump(grouped_data, f, indent=2, ensure_ascii=False)
                    else:
                        grouped_file.unlink()

                except Exception:
                    continue

        return deleted_count

    def get_metadata(self, key: str) -> Optional[CacheMetadata]:
        """Get cache metadata."""
        metadata_file = self._get_metadata_file(key)
        if not metadata_file.exists():
            return None

        try:
            with open(metadata_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return CacheMetadata.from_dict(data)
        except Exception:
            return None

    def list_keys(self, pattern: Optional[str] = None) -> List[str]:
        """List cache keys."""
        keys = []

        # Get keys from individual files
        for cache_file in self.cache_path.rglob("*.json"):
            if cache_file.name.endswith(".metadata.json"):
                continue

            # Skip grouped files, handle them separately
            if cache_file.parent.name == "grouped":
                continue

            # Extract key from file path
            relative_path = cache_file.relative_to(self.cache_path)
            key = str(relative_path).replace(".json", "").replace("/", "_")

            if pattern is None or pattern in key:
                keys.append(key)

        # Get keys from grouped files
        if self.grouped_dir.exists():
            for grouped_file in self.grouped_dir.glob("*.json"):
                try:
                    with open(grouped_file, "r", encoding="utf-8") as f:
                        grouped_data = json.load(f)

                    for key in grouped_data.keys():
                        if pattern is None or pattern in key:
                            keys.append(key)
                except Exception:
                    continue

        return keys

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        total_size = 0
        file_count = 0
        step_counts: Dict[str, int] = {}

        for cache_file in self.cache_path.rglob("*.json"):
            total_size += cache_file.stat().st_size
            file_count += 1

            # Count files per step
            if cache_file.parent.name != self.cache_path.name:
                step_name = cache_file.parent.name
                step_counts[step_name] = step_counts.get(step_name, 0) + 1

        return {
            "total_size": total_size,
            "total_files": file_count,
            "steps": step_counts,
            "cache_dir": str(self.cache_path),
        }

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Organize cache files by step if key contains step info
        if "_" in key:
            parts = key.split("_", 1)
            if parts[0] in [
                "document_processing",
                "text_chunking",
                "entity_extraction",
                "relation_extraction",
                "cluster_formation",
                "graph_assembly",
            ]:
                step_dir = self.steps_dir / parts[0]
                step_dir.mkdir(exist_ok=True)
                return step_dir / f"{parts[1]}.json"

        return self.cache_path / f"{key}.json"

    def _get_metadata_file(self, key: str) -> Path:
        """Get metadata file path for key."""
        cache_file = self._get_cache_file(key)
        return cache_file.with_suffix(".metadata.json")

    def _serialize_data(self, data: Any) -> Any:
        """Serialize data for JSON storage."""
        if isinstance(data, (Entity, Relation, Cluster, TextChunk, KnowledgeGraph)):
            return {"_type": data.__class__.__name__, "_data": data.to_dict()}
        if isinstance(data, list):
            return [self._serialize_data(item) for item in data]
        if isinstance(data, dict):
            return {key: self._serialize_data(value) for key, value in data.items()}
        return data

    def _deserialize_data(
        self, data: Any, expected_type: Type[T], entities_map: Optional[Dict[str, Entity]] = None
    ) -> T:
        """Deserialize data from JSON storage."""
        # pylint: disable=too-many-return-statements
        if isinstance(data, dict) and "_type" in data and "_data" in data:
            obj_type = data["_type"]
            obj_data = data["_data"]

            if obj_type == "Entity":
                return Entity.from_dict(obj_data)  # type: ignore
            if obj_type == "Relation":
                return Relation.from_dict(obj_data, entities_map=entities_map)  # type: ignore
            if obj_type == "Cluster":
                return Cluster.from_dict(obj_data)  # type: ignore
            if obj_type == "TextChunk":
                return TextChunk.from_dict(obj_data)  # type: ignore
            if obj_type == "KnowledgeGraph":
                return KnowledgeGraph.from_dict(obj_data)  # type: ignore

        if isinstance(data, list) and expected_type in [list, List]:
            # For lists, we need to handle entity resolution differently
            result = []
            # Try to collect entities from the same data context first
            if entities_map is None:
                entities_map = self._collect_entities_from_context(data)

            for item in data:
                result.append(self._deserialize_data(item, Any, entities_map))
            return result  # type: ignore

        return data  # type: ignore

    def _collect_entities_from_context(self, data: List[Any]) -> Optional[Dict[str, Entity]]:
        """Collect entities from a cached list that might contain mixed types."""
        entities_map = {}

        for item in data:
            if isinstance(item, dict) and item.get("_type") == "Entity":
                entity = Entity.from_dict(item["_data"])
                entities_map[entity.id] = entity

        return entities_map if entities_map else None

    def get_relations_with_context(
        self, relations_data: Any, entities_context: List[Entity], expected_type: Type[T]
    ) -> T:
        """Deserialize relations with entities context for proper entity resolution."""
        # Build entities map from context
        entities_map = {}
        for entity in entities_context:
            if hasattr(entity, "id"):
                entities_map[entity.id] = entity
            elif isinstance(entity, dict) and entity.get("_type") == "Entity":
                # Handle serialized entity
                entity_obj = Entity.from_dict(entity["_data"])
                entities_map[entity_obj.id] = entity_obj

        # Deserialize relations with entities context
        return self._deserialize_data(relations_data, expected_type, entities_map)

    def _should_use_grouped_storage(self, key: str) -> bool:
        """Check if key should use grouped storage based on data type."""
        # Define patterns that should use grouped storage
        grouped_patterns = [
            "text_chunk",
            "entity_extraction",
            "relation_extraction",
            "cluster_formation",
            "document_text_",
        ]

        # Check if key contains any grouped patterns
        for pattern in grouped_patterns:
            if pattern in key:
                return True
        return False

    def _get_data_type_from_key(self, key: str) -> Optional[str]:
        """Extract data type from cache key."""
        if "text_chunk" in key or "chunk" in key:
            return "TextChunk"
        if "entity" in key:
            return "Entity"
        if "relation" in key:
            return "Relation"
        if "cluster" in key:
            return "Cluster"
        if "document_text_" in key:
            return "DocumentText"
        return None

    def _get_grouped_file_path(self, data_type: str) -> Path:
        """Get file path for grouped data storage."""
        filename = self.grouped_types.get(data_type, f"{data_type.lower()}s")
        return self.grouped_dir / f"{filename}.json"

    def _get_from_grouped_cache(self, key: str, expected_type: Type[T]) -> Optional[T]:
        """Get item from grouped cache file."""
        data_type = self._get_data_type_from_key(key)
        if not data_type:
            return None

        grouped_file = self._get_grouped_file_path(data_type)
        if not grouped_file.exists():
            return None

        try:
            with open(grouped_file, "r", encoding="utf-8") as f:
                grouped_data = json.load(f)

            if key in grouped_data:
                return self._deserialize_data(grouped_data[key], expected_type)
        except Exception:
            # If deserialization fails, remove invalid cache
            grouped_file.unlink(missing_ok=True)

        return None

    def _set_in_grouped_cache(
        self, key: str, value: Any, metadata: Optional[CacheMetadata] = None
    ) -> None:
        """Set item in grouped cache file."""
        data_type = self._get_data_type_from_key(key)
        if not data_type:
            # Fall back to individual file storage
            cache_file = self._get_cache_file(key)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            serialized_data = self._serialize_data(value)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(serialized_data, f, indent=2, ensure_ascii=False)
            return

        grouped_file = self._get_grouped_file_path(data_type)

        # Load existing data or create new
        grouped_data = {}
        if grouped_file.exists():
            try:
                with open(grouped_file, "r", encoding="utf-8") as f:
                    grouped_data = json.load(f)
            except Exception:
                grouped_data = {}

        # Add/update the item
        grouped_data[key] = self._serialize_data(value)

        # Save updated data
        with open(grouped_file, "w", encoding="utf-8") as f:
            json.dump(grouped_data, f, indent=2, ensure_ascii=False)

        # Save metadata if provided
        if metadata:
            metadata_file = self._get_metadata_file(key)
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata.to_dict(), f, indent=2)

    def _has_in_grouped_cache(self, key: str) -> bool:
        """Check if key exists in grouped cache."""
        data_type = self._get_data_type_from_key(key)
        if not data_type:
            return False

        grouped_file = self._get_grouped_file_path(data_type)
        if not grouped_file.exists():
            return False

        try:
            with open(grouped_file, "r", encoding="utf-8") as f:
                grouped_data = json.load(f)
            return key in grouped_data
        except Exception:
            return False

    def _delete_from_grouped_cache(self, key: str) -> bool:
        """Delete item from grouped cache."""
        data_type = self._get_data_type_from_key(key)
        if not data_type:
            return False

        grouped_file = self._get_grouped_file_path(data_type)
        if not grouped_file.exists():
            return False

        try:
            with open(grouped_file, "r", encoding="utf-8") as f:
                grouped_data = json.load(f)

            if key not in grouped_data:
                return False

            del grouped_data[key]

            # Save updated data
            if grouped_data:
                with open(grouped_file, "w", encoding="utf-8") as f:
                    json.dump(grouped_data, f, indent=2, ensure_ascii=False)
            else:
                # Remove empty file
                grouped_file.unlink()

            # Delete metadata
            metadata_file = self._get_metadata_file(key)
            if metadata_file.exists():
                metadata_file.unlink()

            return True
        except Exception:
            return False
