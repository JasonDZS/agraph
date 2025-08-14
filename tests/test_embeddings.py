"""
Test cases for embedding functions.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

try:
    from agraph.vectordb.embeddings import (
        CachedOpenAIEmbeddingFunction,
        ChromaEmbeddingFunction,
        EmbeddingFunctionError,
        OpenAIEmbeddingFunction,
        create_openai_embedding_function,
    )

    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@unittest.skipUnless(EMBEDDINGS_AVAILABLE, "Embedding functions not available")
class TestEmbeddingFunctions(unittest.TestCase):
    """Test cases for embedding functions."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock OpenAI client to avoid actual API calls
        self.mock_response = MagicMock()
        self.mock_response.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5]),
            MagicMock(embedding=[0.2, 0.3, 0.4, 0.5, 0.6]),
        ]
        self.mock_response.usage.total_tokens = 10

    @patch("agraph.vectordb.embeddings.AsyncOpenAI")
    def test_openai_embedding_function_init(self, mock_client_class) -> None:
        """Test OpenAI embedding function initialization."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        embedding_fn = OpenAIEmbeddingFunction(
            api_key="test-key", api_base="https://api.openai.com/v1", model="text-embedding-3-small"
        )

        self.assertEqual(embedding_fn.api_key, "test-key")
        self.assertEqual(embedding_fn.api_base, "https://api.openai.com/v1")
        self.assertEqual(embedding_fn.model, "text-embedding-3-small")
        self.assertEqual(embedding_fn.batch_size, 100)
        self.assertEqual(embedding_fn.max_concurrency, 10)

    @patch("agraph.vectordb.embeddings.AsyncOpenAI")
    def test_openai_embedding_function_embed_batch(self, mock_client_class) -> None:
        """Test batch embedding functionality."""
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = self.mock_response
        mock_client_class.return_value = mock_client

        embedding_fn = OpenAIEmbeddingFunction(api_key="test-key", model="text-embedding-3-small")

        async def test():
            texts = ["Hello world", "Test text"]
            embeddings = await embedding_fn._embed_batch(texts)

            # Verify API was called correctly
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small", input=texts, encoding_format="float"
            )

            # Verify embeddings returned
            self.assertEqual(len(embeddings), 2)
            self.assertEqual(embeddings[0], [0.1, 0.2, 0.3, 0.4, 0.5])
            self.assertEqual(embeddings[1], [0.2, 0.3, 0.4, 0.5, 0.6])

            # Check stats updated
            stats = embedding_fn.get_stats()
            self.assertEqual(stats["total_requests"], 1)
            self.assertEqual(stats["total_texts"], 2)
            self.assertEqual(stats["total_tokens"], 10)

        asyncio.run(test())

    @patch("agraph.vectordb.embeddings.AsyncOpenAI")
    def test_openai_embedding_function_sync_call(self, mock_client_class) -> None:
        """Test synchronous call interface."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = self.mock_response
        mock_client_class.return_value = mock_client

        embedding_fn = OpenAIEmbeddingFunction(api_key="test-key", model="text-embedding-3-small")

        texts = ["Hello world", "Test text"]
        embeddings = embedding_fn(texts)

        self.assertEqual(len(embeddings), 2)
        self.assertEqual(embeddings[0], [0.1, 0.2, 0.3, 0.4, 0.5])

    @patch("agraph.vectordb.embeddings.AsyncOpenAI")
    def test_openai_embedding_function_large_batch(self, mock_client_class) -> None:
        """Test large batch processing with batching."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = self.mock_response
        mock_client_class.return_value = mock_client

        embedding_fn = OpenAIEmbeddingFunction(
            api_key="test-key",
            model="text-embedding-3-small",
            batch_size=2,  # Small batch size for testing
        )

        async def test():
            texts = ["Text 1", "Text 2", "Text 3", "Text 4", "Text 5"]
            embeddings = await embedding_fn.aembed_texts(texts)

            # Should make 3 API calls (2+2+1)
            self.assertEqual(mock_client.embeddings.create.call_count, 3)

            # Should return 5 embeddings (2 per batch * 3 batches = 6, but last batch only has 1)
            # Actually it will be 2*3 = 6 embeddings because our mock returns 2 embeddings per call
            self.assertEqual(len(embeddings), 6)

        asyncio.run(test())

    @patch("agraph.vectordb.embeddings.AsyncOpenAI")
    def test_openai_embedding_function_error_handling(self, mock_client_class) -> None:
        """Test error handling."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.side_effect = Exception("API Error")
        mock_client_class.return_value = mock_client

        embedding_fn = OpenAIEmbeddingFunction(api_key="test-key", model="text-embedding-3-small")

        async def test():
            texts = ["Hello world"]

            with self.assertRaises(EmbeddingFunctionError):
                await embedding_fn.aembed_texts(texts)

            # Check error stats
            stats = embedding_fn.get_stats()
            self.assertEqual(stats["error_count"], 1)

        asyncio.run(test())

    @patch("agraph.vectordb.embeddings.get_settings")
    def test_openai_embedding_function_no_api_key(self, mock_get_settings) -> None:
        """Test error when no API key provided."""
        # Mock settings to return empty API key
        mock_settings = MagicMock()
        mock_settings.openai.api_key = ""
        mock_settings.openai.api_base = None
        mock_settings.embedding.model = "text-embedding-3-small"
        mock_get_settings.return_value = mock_settings

        with self.assertRaises(EmbeddingFunctionError):
            OpenAIEmbeddingFunction()

    @patch("agraph.vectordb.embeddings.AsyncOpenAI")
    def test_cached_embedding_function(self, mock_client_class) -> None:
        """Test cached embedding function."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = self.mock_response
        mock_client_class.return_value = mock_client

        embedding_fn = CachedOpenAIEmbeddingFunction(
            api_key="test-key", model="text-embedding-3-small", cache_size=10
        )

        async def test():
            # First call - cache miss
            texts = ["Hello world", "Test text"]
            embeddings1 = await embedding_fn.aembed_texts(texts)

            # Second call with same texts - cache hit
            embeddings2 = await embedding_fn.aembed_texts(texts)

            # Should only make one API call
            self.assertEqual(mock_client.embeddings.create.call_count, 1)

            # Results should be the same
            self.assertEqual(embeddings1, embeddings2)

            # Check cache stats
            cache_stats = embedding_fn.get_cache_stats()
            self.assertEqual(cache_stats["cache_hits"], 2)
            self.assertEqual(cache_stats["cache_misses"], 2)  # Initial misses
            self.assertEqual(cache_stats["cache_size"], 2)

        asyncio.run(test())

    def test_chroma_embedding_function_wrapper(self) -> None:
        """Test ChromaEmbeddingFunction wrapper."""
        # Create a mock OpenAI embedding function
        mock_openai_fn = MagicMock()
        mock_openai_fn.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

        wrapper = ChromaEmbeddingFunction(mock_openai_fn)

        texts = ["Hello", "World"]
        result = wrapper(texts)

        # Should call the underlying function
        mock_openai_fn.assert_called_once_with(texts)
        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    @patch("agraph.vectordb.embeddings.get_settings")
    def test_create_openai_embedding_function(self, mock_get_settings) -> None:
        """Test the convenience function for creating embedding functions."""
        # Mock settings
        mock_settings = MagicMock()
        mock_settings.embedding.model = "test-model"
        mock_settings.openai.api_key = "test-key"
        mock_settings.openai.api_base = "https://api.test.com"
        mock_get_settings.return_value = mock_settings

        with patch("agraph.vectordb.embeddings.CachedOpenAIEmbeddingFunction") as mock_cached:
            # Test with cache
            create_openai_embedding_function(use_cache=True, custom_param="value")

            mock_cached.assert_called_once_with(
                api_key="test-key",
                api_base="https://api.test.com",
                model="test-model",
                custom_param="value",
            )

        with patch("agraph.vectordb.embeddings.OpenAIEmbeddingFunction") as mock_regular:
            # Test without cache
            create_openai_embedding_function(use_cache=False)

            mock_regular.assert_called_once_with(
                api_key="test-key", api_base="https://api.test.com", model="test-model"
            )

    @patch("agraph.vectordb.embeddings.AsyncOpenAI")
    def test_embedding_function_stats(self, mock_client_class) -> None:
        """Test embedding function statistics."""
        mock_client = AsyncMock()
        mock_client.embeddings.create.return_value = self.mock_response
        mock_client_class.return_value = mock_client

        embedding_fn = OpenAIEmbeddingFunction(api_key="test-key", model="text-embedding-3-small")

        async def test():
            # Initial stats should be empty
            stats = embedding_fn.get_stats()
            self.assertEqual(stats["total_requests"], 0)
            self.assertEqual(stats["total_texts"], 0)
            self.assertEqual(stats["total_tokens"], 0)
            self.assertEqual(stats["avg_response_time"], 0.0)
            self.assertEqual(stats["error_count"], 0)

            # Make some requests
            await embedding_fn.aembed_texts(["Text 1", "Text 2"])
            await embedding_fn.aembed_texts(["Text 3"])

            # Check updated stats
            stats = embedding_fn.get_stats()
            self.assertEqual(stats["total_requests"], 2)
            self.assertEqual(stats["total_texts"], 3)
            self.assertEqual(stats["total_tokens"], 20)  # 10 tokens per request
            self.assertGreater(stats["avg_response_time"], 0.0)

            # Reset stats
            embedding_fn.reset_stats()
            stats = embedding_fn.get_stats()
            self.assertEqual(stats["total_requests"], 0)

        asyncio.run(test())


@unittest.skipUnless(EMBEDDINGS_AVAILABLE, "Embedding functions not available")
class TestEmbeddingFunctionWithoutOpenAI(unittest.TestCase):
    """Test embedding function behavior when OpenAI is not available."""

    @patch("agraph.vectordb.embeddings.OPENAI_AVAILABLE", False)
    def test_embedding_function_without_openai(self) -> None:
        """Test that appropriate error is raised when OpenAI is not available."""
        with self.assertRaises(EmbeddingFunctionError):
            OpenAIEmbeddingFunction(api_key="test-key")


if __name__ == "__main__":
    if not EMBEDDINGS_AVAILABLE:
        print("Embedding functions not available. Skipping tests.")
        print("This is expected if OpenAI is not installed.")

    unittest.main()
