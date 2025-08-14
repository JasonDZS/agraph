"""
Vector store mixins providing common functionality.

This module provides mixin classes that contain common code shared
between different vector store implementations to reduce code duplication.
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union


class EmbeddingStatsMixin:
    """Mixin providing embedding statistics functionality."""

    def get_embedding_stats(self) -> Optional[Dict[str, Union[int, float]]]:
        """Get embedding function statistics.

        Returns:
            Embedding statistics, or None if no OpenAI embedding function is used
        """
        if hasattr(self, "_openai_embedding") and self._openai_embedding:
            stats = self._openai_embedding.get_stats()
            if hasattr(self._openai_embedding, "get_cache_stats"):
                cache_stats = self._openai_embedding.get_cache_stats()
                if cache_stats:
                    stats.update(cache_stats)
            return stats  # type: ignore
        return None


class HybridSearchMixin:
    """Mixin providing hybrid search functionality."""

    async def hybrid_search(
        self,
        query: Union[str, List[float]],
        search_types: Set[str],
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Tuple[Any, float]]]:
        """Hybrid search for multiple types of objects."""
        results = {}

        if "entity" in search_types:
            if hasattr(self, "search_entities"):
                results["entity"] = await self.search_entities(query, top_k, filter_dict)

        if "relation" in search_types:
            if hasattr(self, "search_relations"):
                results["relation"] = await self.search_relations(query, top_k, filter_dict)

        if "cluster" in search_types:
            if hasattr(self, "search_clusters"):
                results["cluster"] = await self.search_clusters(query, top_k, filter_dict)

        if "text_chunk" in search_types:
            if hasattr(self, "search_text_chunks"):
                results["text_chunk"] = await self.search_text_chunks(query, top_k, filter_dict)

        return results
