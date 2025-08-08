"""
Vector storage implementation module.

Provides JSON file-based vector storage implementation with similarity search.
"""

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..config import settings
from ..logger import logger
from .interfaces import VectorStorage


class JsonVectorStorage(VectorStorage):
    """JSON file-based vector storage implementation."""

    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize JSON vector storage.

        Args:
            file_path: Path to the vector storage file. If None, uses default path.
        """
        if file_path is None:
            file_path = os.path.join(settings.workdir, "vectors.json")
        self.file_path = file_path
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict = {}
        self._is_connected = False
        self._ensure_file_exists()

    def _ensure_file_exists(self) -> None:
        """Ensure storage file exists."""
        if not os.path.exists(self.file_path):
            dir_path = os.path.dirname(self.file_path)
            if dir_path:  # Only create directory if there is a directory part
                os.makedirs(dir_path, exist_ok=True)
            self._save_to_file({}, {})
        else:
            # If file exists, try to load data
            self.vectors, self.metadata = self._load_from_file()

    def _save_to_file(self, vectors: Dict[str, np.ndarray], metadata: Dict) -> None:
        """Save vectors to file."""
        try:
            data = {"vectors": {k: v.tolist() for k, v in vectors.items()}, "metadata": metadata}
            with open(self.file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error("Error saving vectors to file: %s", e)

    def _load_from_file(self) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Load vectors from file."""
        try:
            if not os.path.exists(self.file_path):
                return {}, {}

            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            vectors = {k: np.array(v, dtype=np.float32) for k, v in data.get("vectors", {}).items()}
            metadata = data.get("metadata", {})
            return vectors, metadata
        except Exception as e:
            logger.error("Error loading vectors from file: %s", e)
            return {}, {}

    # VectorStorageConnection methods

    def connect(self) -> bool:
        """Connect to vector storage."""
        try:
            self._ensure_file_exists()
            # Load existing data
            self.vectors, self.metadata = self._load_from_file()
            self._is_connected = True
            logger.info("Connected to vector storage at %s", self.file_path)
            return True
        except Exception as e:
            logger.error("Failed to connect to vector storage: %s", e)
            self._is_connected = False
            return False

    def disconnect(self) -> None:
        """Disconnect from vector storage."""
        # Save current state
        if self._is_connected:
            self.save()
        self._is_connected = False
        logger.info("Disconnected from vector storage")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._is_connected

    # VectorStorageCRUD methods

    def add_vector(self, vector_id: str, vector: Any, metadata: Optional[Dict] = None) -> bool:
        """Add vector to storage."""
        try:
            if isinstance(vector, (list, tuple)):
                vector = np.array(vector, dtype=np.float32)
            elif isinstance(vector, np.ndarray):
                vector = vector.astype(np.float32)
            else:
                logger.error("Invalid vector type: %s", type(vector))
                return False

            self.vectors[vector_id] = vector
            if metadata:
                if "vector_metadata" not in self.metadata:
                    self.metadata["vector_metadata"] = {}
                self.metadata["vector_metadata"][vector_id] = metadata

            # 自动保存
            if self._is_connected:
                self.save()
            return True
        except Exception as e:
            logger.error("Error adding vector %s: %s", vector_id, e)
            return False

    def get_vector(self, vector_id: str) -> Optional[Any]:
        """Get vector from storage."""
        return self.vectors.get(vector_id)

    def delete_vector(self, vector_id: str) -> bool:
        """Delete vector from storage."""
        try:
            if vector_id in self.vectors:
                del self.vectors[vector_id]
                if "vector_metadata" in self.metadata and vector_id in self.metadata["vector_metadata"]:
                    del self.metadata["vector_metadata"][vector_id]

                # Auto-save
                if self._is_connected:
                    self.save()
                return True
            return False
        except Exception as e:
            logger.error("Error deleting vector %s: %s", vector_id, e)
            return False

    def save_vectors(self, vectors: Dict[str, Any], metadata: Optional[Dict] = None) -> bool:
        """Batch save vectors."""
        try:
            # Update vectors in memory
            for vector_id, vector in vectors.items():
                if isinstance(vector, (list, tuple)):
                    vector = np.array(vector, dtype=np.float32)
                elif isinstance(vector, np.ndarray):
                    vector = vector.astype(np.float32)

                self.vectors[vector_id] = vector

            # Update metadata
            if metadata:
                self.metadata.update(metadata)

            # Save to file
            if self._is_connected:
                self.save()
            return True
        except Exception as e:
            logger.error("Error saving vectors: %s", e)
            return False

    def load_vectors(self) -> Tuple[Dict[str, Any], Dict]:
        """Load all vectors."""
        try:
            self.vectors, self.metadata = self._load_from_file()
            return dict(self.vectors), self.metadata
        except Exception as e:
            logger.error("Error loading vectors: %s", e)
            return {}, {}

    def clear(self) -> bool:
        """Clear all vectors."""
        try:
            self.vectors.clear()
            self.metadata.clear()
            if self._is_connected:
                self.save()
            return True
        except Exception as e:
            logger.error("Error clearing vectors: %s", e)
            return False

    # VectorStorageQuery methods

    def search_similar_vectors(
        self, query_vector: Any, top_k: int = 10, threshold: float = 0.0
    ) -> List[Tuple[str, float]]:
        """Search for similar vectors."""
        try:
            if isinstance(query_vector, (list, tuple)):
                query_vector = np.array(query_vector, dtype=np.float32)
            elif not isinstance(query_vector, np.ndarray):
                logger.error("Invalid query vector type: %s", type(query_vector))
                return []

            similarities = []
            for vector_id, vector in self.vectors.items():
                similarity = self.compute_similarity(query_vector, vector)
                if similarity >= threshold:
                    similarities.append((vector_id, similarity))

            # 按相似度降序排序
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:top_k]
        except Exception as e:
            logger.error("Error searching similar vectors: %s", e)
            return []

    def compute_similarity(self, vector1: Any, vector2: Any) -> float:
        """Compute cosine similarity between two vectors."""
        try:
            if isinstance(vector1, (list, tuple)):
                vector1 = np.array(vector1, dtype=np.float32)
            if isinstance(vector2, (list, tuple)):
                vector2 = np.array(vector2, dtype=np.float32)

            dot_product = np.dot(vector1, vector2)
            norm_a = np.linalg.norm(vector1)
            norm_b = np.linalg.norm(vector2)

            if norm_a == 0 or norm_b == 0:
                return 0.0

            return float(dot_product / (norm_a * norm_b))
        except Exception as e:
            logger.error("Error computing similarity: %s", e)
            return 0.0

    # Additional utility methods

    def save(self) -> None:
        """Save current state to file."""
        self._save_to_file(self.vectors, self.metadata)

    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information and statistics."""
        try:
            file_size = 0
            if os.path.exists(self.file_path):
                file_size = os.path.getsize(self.file_path)

            return {
                "file_path": self.file_path,
                "vector_count": len(self.vectors),
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "is_connected": self.is_connected(),
            }
        except Exception as e:
            logger.error("Error getting storage info: %s", e)
            return {}
