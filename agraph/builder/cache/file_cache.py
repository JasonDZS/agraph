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
    """File-based cache backend."""

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

        for directory in [self.steps_dir, self.metadata_dir, self.user_edits_dir]:
            directory.mkdir(exist_ok=True)

    def get(self, key: str, expected_type: Type[T]) -> Optional[T]:
        """Get cached value by key."""
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
        return self._get_cache_file(key).exists()

    def delete(self, key: str) -> bool:
        """Delete cached value."""
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
        for cache_file in self.cache_path.rglob("*.json"):
            if pattern in cache_file.name:
                cache_file.unlink()
                deleted_count += 1

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

        for cache_file in self.cache_path.rglob("*.json"):
            if cache_file.name.endswith(".metadata.json"):
                continue

            # Extract key from file path
            relative_path = cache_file.relative_to(self.cache_path)
            key = str(relative_path).replace(".json", "").replace("/", "_")

            if pattern is None or pattern in key:
                keys.append(key)

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
