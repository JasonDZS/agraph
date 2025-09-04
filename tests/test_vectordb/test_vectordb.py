"""
Test cases for vector database module.
"""

import asyncio
import unittest

from agraph.base.models.clusters import Cluster
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation
from agraph.base.models.text import TextChunk
from agraph.vectordb import MemoryVectorStore


class TestMemoryVectorStore(unittest.TestCase):
    """Test cases for MemoryVectorStore."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.vector_store = MemoryVectorStore()

    async def async_test_helper(self, coro):
        """Helper to run async tests."""
        return await coro

    def test_initialization(self) -> None:
        """Test vector store initialization."""
        self.assertEqual(self.vector_store.collection_name, "knowledge_graph")
        self.assertFalse(self.vector_store.is_initialized())

    def test_initialize_and_close(self) -> None:
        """Test initialize and close methods."""

        async def test():
            await self.vector_store.initialize()
            self.assertTrue(self.vector_store.is_initialized())

            await self.vector_store.close()
            self.assertFalse(self.vector_store.is_initialized())

        asyncio.run(test())

    def test_context_manager(self) -> None:
        """Test async context manager functionality."""

        async def test():
            async with self.vector_store as store:
                self.assertTrue(store.is_initialized())
            self.assertFalse(self.vector_store.is_initialized())

        asyncio.run(test())

    def test_entity_operations(self) -> None:
        """Test entity CRUD operations."""

        async def test():
            await self.vector_store.initialize()

            # Create test entity
            entity = Entity(name="Test Entity", description="Test description")

            # Add entity
            result = await self.vector_store.add_entity(entity)
            self.assertTrue(result)

            # Get entity
            retrieved = await self.vector_store.get_entity(entity.id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.name, entity.name)

            # Update entity
            entity.description = "Updated description"
            result = await self.vector_store.update_entity(entity)
            self.assertTrue(result)

            # Delete entity
            result = await self.vector_store.delete_entity(entity.id)
            self.assertTrue(result)

            # Verify deletion
            retrieved = await self.vector_store.get_entity(entity.id)
            self.assertIsNone(retrieved)

        asyncio.run(test())

    def test_search_entities(self) -> None:
        """Test entity search functionality."""

        async def test():
            await self.vector_store.initialize()

            # Create test entities
            entity1 = Entity(name="Python Programming", description="Programming language")
            entity2 = Entity(name="Java Programming", description="Another language")

            await self.vector_store.add_entity(entity1)
            await self.vector_store.add_entity(entity2)

            # Search entities
            results = await self.vector_store.search_entities("programming", top_k=5)
            self.assertGreater(len(results), 0)

            # Check result format
            for entity, score in results:
                self.assertIsInstance(entity, Entity)
                self.assertIsInstance(score, float)

        asyncio.run(test())

    def test_relation_operations(self) -> None:
        """Test relation CRUD operations."""

        async def test():
            await self.vector_store.initialize()

            # Create test entities
            entity1 = Entity(name="Entity 1")
            entity2 = Entity(name="Entity 2")

            # Create test relation
            relation = Relation(
                head_entity=entity1, tail_entity=entity2, description="Test relation"
            )

            # Add relation
            result = await self.vector_store.add_relation(relation)
            self.assertTrue(result)

            # Get relation
            retrieved = await self.vector_store.get_relation(relation.id)
            self.assertIsNotNone(retrieved)

            # Delete relation
            result = await self.vector_store.delete_relation(relation.id)
            self.assertTrue(result)

        asyncio.run(test())

    def test_cluster_operations(self) -> None:
        """Test cluster CRUD operations."""

        async def test():
            await self.vector_store.initialize()

            # Create test cluster
            cluster = Cluster(name="Test Cluster", description="Test cluster description")

            # Add cluster
            result = await self.vector_store.add_cluster(cluster)
            self.assertTrue(result)

            # Get cluster
            retrieved = await self.vector_store.get_cluster(cluster.id)
            self.assertIsNotNone(retrieved)

            # Delete cluster
            result = await self.vector_store.delete_cluster(cluster.id)
            self.assertTrue(result)

        asyncio.run(test())

    def test_text_chunk_operations(self) -> None:
        """Test text chunk CRUD operations."""

        async def test():
            await self.vector_store.initialize()

            # Create test text chunk
            chunk = TextChunk(content="This is test content", title="Test Title")

            # Add text chunk
            result = await self.vector_store.add_text_chunk(chunk)
            self.assertTrue(result)

            # Get text chunk
            retrieved = await self.vector_store.get_text_chunk(chunk.id)
            self.assertIsNotNone(retrieved)

            # Delete text chunk
            result = await self.vector_store.delete_text_chunk(chunk.id)
            self.assertTrue(result)

        asyncio.run(test())

    def test_batch_operations(self) -> None:
        """Test batch operations."""

        async def test():
            await self.vector_store.initialize()

            # Create test entities
            entities = [Entity(name="Entity 1"), Entity(name="Entity 2"), Entity(name="Entity 3")]

            # Batch add entities
            results = await self.vector_store.batch_add_entities(entities)
            self.assertEqual(len(results), 3)
            self.assertTrue(all(results))

        asyncio.run(test())

    def test_hybrid_search(self) -> None:
        """Test hybrid search functionality."""

        async def test():
            await self.vector_store.initialize()

            # Add test data
            entity = Entity(name="Test Entity")
            chunk = TextChunk(content="Test content")

            await self.vector_store.add_entity(entity)
            await self.vector_store.add_text_chunk(chunk)

            # Hybrid search
            results = await self.vector_store.hybrid_search(
                query="test", search_types={"entity", "text_chunk"}, top_k=5
            )

            self.assertIn("entity", results)
            self.assertIn("text_chunk", results)

        asyncio.run(test())

    def test_stats(self) -> None:
        """Test statistics functionality."""

        async def test():
            await self.vector_store.initialize()

            # Add test data
            entity = Entity(name="Test Entity")
            await self.vector_store.add_entity(entity)

            # Get stats
            stats = await self.vector_store.get_stats()
            self.assertEqual(stats["entities"], 1)
            self.assertEqual(stats["relations"], 0)
            self.assertEqual(stats["clusters"], 0)
            self.assertEqual(stats["text_chunks"], 0)

        asyncio.run(test())

    def test_clear_all(self) -> None:
        """Test clear all functionality."""

        async def test():
            await self.vector_store.initialize()

            # Add test data
            entity = Entity(name="Test Entity")
            await self.vector_store.add_entity(entity)

            # Clear all data
            result = await self.vector_store.clear_all()
            self.assertTrue(result)

            # Verify data is cleared
            stats = await self.vector_store.get_stats()
            self.assertEqual(stats["entities"], 0)

        asyncio.run(test())

    def test_embedding_validation(self) -> None:
        """Test embedding validation."""
        # Valid embedding
        self.assertTrue(self.vector_store._validate_embedding([1.0, 2.0, 3.0]))
        self.assertTrue(self.vector_store._validate_embedding(None))

        # Invalid embedding
        self.assertFalse(self.vector_store._validate_embedding("invalid"))
        self.assertFalse(self.vector_store._validate_embedding([1.0, "invalid"]))


if __name__ == "__main__":
    unittest.main()
