"""
Embedding function implementations.

Provides OpenAI-Compatible API embedding functionality with async concurrent request optimization.
"""

import asyncio
import concurrent.futures
import time
from typing import Any, Dict, List, Optional, Union

try:
    import nest_asyncio

    NEST_ASYNCIO_AVAILABLE = True
except ImportError:
    NEST_ASYNCIO_AVAILABLE = False

try:
    from openai import AsyncOpenAI

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None  # type: ignore

from ..config import EmbeddingConfig, OpenAIConfig, get_settings
from ..logger import logger
from .exceptions import EmbeddingError


class EmbeddingFunctionError(EmbeddingError):
    """Embedding function related exception."""


class OpenAIEmbeddingFunction:
    """OpenAI-Compatible API embedding function.

    Supports async concurrent requests with configurable batch size and concurrency limits.
    Compatible with OpenAI and other compatible API services (e.g., Azure OpenAI, local deployment).
    """

    # pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: Optional[str] = None,
        batch_size: int = 100,
        max_concurrency: int = 10,
        max_retries: int = 3,
        timeout: float = 30.0,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI embedding function.

        Args:
            api_key: API key, read from config if None
            api_base: API base URL, read from config if None
            model: Embedding model name, read from config if None
            batch_size: Batch size, number of texts per request
            max_concurrency: Maximum number of concurrent requests
            max_retries: Maximum number of retries
            timeout: Request timeout in seconds
            **kwargs: Additional parameters passed to OpenAI client
        """
        if not OPENAI_AVAILABLE:
            raise EmbeddingFunctionError(
                "OpenAI is not installed. Please install it with: pip install openai>=1.99.9"
            )

        # Get configuration
        settings = get_settings()

        self.api_key = api_key or settings.openai.api_key
        self.api_base = api_base or settings.openai.api_base
        self.model = model or settings.embedding.model

        if not self.api_key:
            raise EmbeddingFunctionError("OpenAI API key is required")

        self.batch_size = batch_size
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.timeout = timeout

        # Create async client
        self._client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs,
        )

        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrency)

        # Statistics
        self._stats = {
            "total_requests": 0,
            "total_texts": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "error_count": 0,
        }

    def __call__(self, texts: List[str]) -> List[List[float]]:
        """Synchronous interface for ChromaDB calls.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (like in Jupyter/PyCharm),
                # use nest_asyncio or run in thread
                if NEST_ASYNCIO_AVAILABLE:
                    nest_asyncio.apply()
                    return loop.run_until_complete(self.aembed_texts(texts))
                # If nest_asyncio is not available, run in a separate thread
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self.aembed_texts(texts))
                    return future.result()
            return loop.run_until_complete(self.aembed_texts(texts))
        except RuntimeError:
            # No event loop exists, create a new one
            return asyncio.run(self.aembed_texts(texts))

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Split texts by batch size
        batches = [texts[i : i + self.batch_size] for i in range(0, len(texts), self.batch_size)]

        # Process all batches concurrently
        tasks = [self._embed_batch(batch) for batch in batches]
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle results and exceptions
        embeddings: List[List[float]] = []
        for result in batch_results:
            if isinstance(result, Exception):
                raise EmbeddingFunctionError(f"Batch embedding failed: {result}")
            if isinstance(result, list):
                embeddings.extend(result)

        return embeddings

    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts.

        Args:
            texts: Text batch

        Returns:
            List of embedding vectors
        """
        async with self._semaphore:
            start_time = time.time()

            try:
                response = await self._client.embeddings.create(
                    model=self.model, input=texts, encoding_format="float"
                )

                # Extract embedding vectors
                embeddings = [data.embedding for data in response.data]

                # Update statistics
                elapsed_time = time.time() - start_time
                self._update_stats(len(texts), response.usage.total_tokens, elapsed_time)

                return embeddings

            except Exception as e:
                self._stats["error_count"] += 1
                raise EmbeddingFunctionError(f"API request failed: {e}") from e

    async def embed_single(self, text: str) -> List[float]:
        """Embed a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        embeddings = await self.aembed_texts([text])
        return embeddings[0]

    def _update_stats(self, text_count: int, token_count: int, response_time: float) -> None:
        """Update statistics.

        Args:
            text_count: Number of texts processed
            token_count: Number of tokens used
            response_time: Response time
        """
        self._stats["total_requests"] += 1
        self._stats["total_texts"] += text_count
        self._stats["total_tokens"] += token_count

        # Calculate average response time
        total_time = (
            self._stats["avg_response_time"] * (self._stats["total_requests"] - 1) + response_time
        )
        self._stats["avg_response_time"] = total_time / self._stats["total_requests"]

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get statistics.

        Returns:
            Dictionary containing statistics
        """
        return self._stats.copy()

    def reset_stats(self) -> None:
        """Reset statistics."""
        self._stats = {
            "total_requests": 0,
            "total_texts": 0,
            "total_tokens": 0,
            "avg_response_time": 0.0,
            "error_count": 0,
        }

    async def close(self) -> None:
        """Close client connection."""
        if hasattr(self._client, "close"):
            try:
                await self._client.close()
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    # Event loop is already closed, client cleanup is not needed
                    logger.warning("Event loop is closed, client cleanup skipped.")
                else:
                    raise EmbeddingFunctionError(f"Failed to close client: {e}") from e


class CachedOpenAIEmbeddingFunction(OpenAIEmbeddingFunction):
    """Cached OpenAI embedding function.

    Caches embedding results in memory to avoid duplicate API calls.
    """

    def __init__(self, cache_size: int = 1000, **kwargs: Any) -> None:
        """Initialize cached embedding function.

        Args:
            cache_size: Cache size
            **kwargs: Parameters passed to parent class
        """
        super().__init__(**kwargs)
        self.cache_size = cache_size
        self._cache: Dict[str, List[float]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    async def aembed_texts(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously embed text list (with caching)."""
        # Separate cache hits and misses
        cached_results = {}
        uncached_texts = []
        uncached_indices = []

        for i, text in enumerate(texts):
            if text in self._cache:
                cached_results[i] = self._cache[text]
                self._cache_hits += 1
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
                self._cache_misses += 1

        # Get embeddings for uncached texts
        if uncached_texts:
            uncached_embeddings = await super().aembed_texts(uncached_texts)

            # Update cache
            for text, embedding in zip(uncached_texts, uncached_embeddings):
                self._cache[text] = embedding

                # Limit cache size
                if len(self._cache) > self.cache_size:
                    # Remove oldest entry (simple FIFO strategy)
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

            # Place results in correct positions
            for i, embedding in zip(uncached_indices, uncached_embeddings):
                cached_results[i] = embedding

        # Rebuild results in original order
        return [cached_results[i] for i in range(len(texts))]

    def get_cache_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics.

        Returns:
            Cache statistics dictionary
        """
        total_queries = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_queries if total_queries > 0 else 0.0

        return {
            "cache_size": len(self._cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate,
            "max_cache_size": self.cache_size,
        }

    def clear_cache(self) -> None:
        """Clear cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


def create_openai_embedding_function(
    embedding_config: Optional[EmbeddingConfig] = None,
    openai_config: Optional[OpenAIConfig] = None,
    use_cache: bool = True,
    **kwargs: Any,
) -> Union[OpenAIEmbeddingFunction, CachedOpenAIEmbeddingFunction]:
    """Convenience function to create OpenAI embedding function.

    Args:
        embedding_config: Embedding configuration, get from global settings if None
        openai_config: OpenAI configuration, get from global settings if None
        use_cache: Whether to use cache
        **kwargs: Additional parameters

    Returns:
        Embedding function instance
    """
    settings = get_settings()

    embedding_cfg = embedding_config or settings.embedding
    openai_cfg = openai_config or settings.openai

    function_kwargs = {
        "api_key": openai_cfg.api_key,
        "api_base": openai_cfg.api_base,
        "model": embedding_cfg.model,
        **kwargs,
    }

    if use_cache:
        return CachedOpenAIEmbeddingFunction(**function_kwargs)

    return OpenAIEmbeddingFunction(**function_kwargs)


# Wrapper for ChromaDB
class ChromaEmbeddingFunction:
    """ChromaDB embedding function wrapper.

    Wraps async OpenAI embedding function as ChromaDB compatible synchronous interface.
    """

    def __init__(self, embedding_function: OpenAIEmbeddingFunction) -> None:
        """Initialize wrapper.

        Args:
            embedding_function: OpenAI embedding function instance
        """
        self.embedding_function = embedding_function

    def __call__(self, input: List[str]) -> List[List[float]]:  # pylint: disable=redefined-builtin
        """Chromadb compatible call interface.

        Args:
            input: Input text list

        Returns:
            List of embedding vectors
        """
        return self.embedding_function(input)
