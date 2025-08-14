"""
Query builder utilities for vector database operations.

This module provides common functionality for building queries and handling
results across different vector store implementations.
"""

from typing import Any, Dict, List, Optional, Union

from .constants import DEFAULT_TOP_K


class QueryBuilder:
    """Builder for constructing vector database queries."""

    def __init__(self) -> None:
        self.query_params: Dict[str, Any] = {}

    def with_query(self, query: Union[str, List[float]]) -> "QueryBuilder":
        """Set the query (text or embedding vector)."""
        if isinstance(query, str):
            self.query_params["query_texts"] = [query]
        else:
            self.query_params["query_embeddings"] = [query]
        return self

    def with_top_k(self, top_k: int = DEFAULT_TOP_K) -> "QueryBuilder":
        """Set the number of results to return."""
        self.query_params["n_results"] = top_k
        return self

    def with_filters(self, filter_dict: Optional[Dict[str, Any]] = None) -> "QueryBuilder":
        """Add filter conditions for ChromaDB."""
        if filter_dict:
            where_clause = {}
            for key, value in filter_dict.items():
                where_clause[key] = {"$eq": value}
            self.query_params["where"] = where_clause
        return self

    def with_includes(self, includes: List[str]) -> "QueryBuilder":
        """Set what to include in the results."""
        self.query_params["include"] = includes
        return self

    def build(self) -> Dict[str, Any]:
        """Build and return the query parameters."""
        return self.query_params.copy()


class ResultProcessor:
    """Processor for handling vector database query results."""

    @staticmethod
    def calculate_similarity(distance: float) -> float:
        """Convert distance to similarity score.

        Args:
            distance: Distance value from vector database

        Returns:
            Similarity score (higher is more similar)
        """
        return max(0.0, 1.0 - distance)

    @staticmethod
    def apply_filters(obj: Any, filter_dict: Optional[Dict[str, Any]] = None) -> bool:
        """Apply filter conditions to an object.

        Args:
            obj: Object to filter
            filter_dict: Filter conditions

        Returns:
            True if object passes all filters, False otherwise
        """
        if not filter_dict:
            return True

        for key, value in filter_dict.items():
            if hasattr(obj, key):
                obj_value = getattr(obj, key)
                if obj_value != value:
                    return False
            else:
                # If object doesn't have the attribute, it doesn't match
                return False
        return True

    @staticmethod
    def sort_by_similarity(results: List[tuple], descending: bool = True) -> List[tuple]:
        """Sort results by similarity score.

        Args:
            results: List of (object, similarity) tuples
            descending: Whether to sort in descending order (highest similarity first)

        Returns:
            Sorted list of results
        """
        return sorted(results, key=lambda x: x[1], reverse=descending)


class EmbeddingGenerator:
    """Helper for generating embeddings from text."""

    @staticmethod
    def generate_simple_embedding(text: str, dimension: int = 128) -> List[float]:
        """Generate simple character frequency embedding.

        This is a fallback implementation for when advanced embedding models
        are not available. Should not be used in production.

        Args:
            text: Input text
            dimension: Desired embedding dimension

        Returns:
            Simple embedding vector
        """
        # Create character frequency vector
        char_freq = [0.0] * dimension

        for char in text.lower():
            if ord(char) < dimension:
                char_freq[ord(char)] += 1.0

        # Normalize
        total = sum(char_freq)
        if total > 0:
            char_freq = [f / total for f in char_freq]

        return char_freq

    @staticmethod
    def prepare_text_for_embedding(name: str, description: str, max_length: int = 8192) -> str:
        """Prepare text for embedding generation.

        Args:
            name: Object name
            description: Object description
            max_length: Maximum text length

        Returns:
            Prepared text for embedding
        """
        text = f"{name} {description}".strip()
        if len(text) > max_length:
            text = text[:max_length]
        return text
