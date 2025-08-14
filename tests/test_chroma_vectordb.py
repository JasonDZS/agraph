"""
Test cases for ChromaDB vector database implementation.

Note: These tests require ChromaDB to be installed.
Install with: pip install 'agraph[vectordb]' or pip install chromadb>=0.5.0
"""

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from agraph.vectordb import VectorStoreError
    from agraph.vectordb.chroma import ChromaVectorStore

    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

from agraph.base.clusters import Cluster
from agraph.base.entities import Entity
from agraph.base.relations import Relation
from agraph.base.text import TextChunk


@unittest.skipUnless(CHROMADB_AVAILABLE, "ChromaDB not available")
class TestChromaVectorStore(unittest.TestCase):
    """Test cases for ChromaVectorStore."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # 使用临时目录进行测试
        self.temp_dir = tempfile.mkdtemp()
        self.vector_store = ChromaVectorStore(
            collection_name="test_kg", persist_directory=self.temp_dir
        )

    def tearDown(self) -> None:
        """Clean up test fixtures."""
        # 清理临时目录
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self) -> None:
        """Test vector store initialization."""
        self.assertEqual(self.vector_store.collection_name, "test_kg")
        self.assertFalse(self.vector_store.is_initialized())
        self.assertEqual(self.vector_store.persist_directory, self.temp_dir)

    def test_initialize_and_close(self) -> None:
        """Test initialize and close methods."""

        async def test():
            await self.vector_store.initialize()
            self.assertTrue(self.vector_store.is_initialized())

            await self.vector_store.close()
            self.assertFalse(self.vector_store.is_initialized())

        asyncio.run(test())

    def test_memory_mode_initialization(self) -> None:
        """Test initialization in memory mode."""

        async def test():
            memory_store = ChromaVectorStore(collection_name="test_memory")
            await memory_store.initialize()
            self.assertTrue(memory_store.is_initialized())
            await memory_store.close()

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
            self.assertEqual(retrieved.description, entity.description)

            # Update entity
            entity.description = "Updated description"
            result = await self.vector_store.update_entity(entity)
            self.assertTrue(result)

            # Verify update
            updated = await self.vector_store.get_entity(entity.id)
            self.assertEqual(updated.description, "Updated description")

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
                self.assertGreaterEqual(score, 0.0)
                self.assertLessEqual(score, 1.0)

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
            self.assertEqual(retrieved.description, relation.description)

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
            self.assertEqual(retrieved.name, cluster.name)

            # Delete cluster
            result = await self.vector_store.delete_cluster(cluster.id)
            self.assertTrue(result)

        asyncio.run(test())

    def test_text_chunk_operations(self) -> None:
        """Test text chunk CRUD operations."""

        async def test():
            await self.vector_store.initialize()

            # Create test text chunk
            chunk = TextChunk(content="This is test content for ChromaDB", title="Test Title")

            # Add text chunk
            result = await self.vector_store.add_text_chunk(chunk)
            self.assertTrue(result)

            # Get text chunk
            retrieved = await self.vector_store.get_text_chunk(chunk.id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.title, chunk.title)

            # Delete text chunk
            result = await self.vector_store.delete_text_chunk(chunk.id)
            self.assertTrue(result)

        asyncio.run(test())

    def test_batch_operations(self) -> None:
        """Test batch operations."""

        async def test():
            await self.vector_store.initialize()

            # Create test entities
            entities = [
                Entity(name="Entity 1", description="First entity"),
                Entity(name="Entity 2", description="Second entity"),
                Entity(name="Entity 3", description="Third entity"),
            ]

            # Batch add entities
            results = await self.vector_store.batch_add_entities(entities)
            self.assertEqual(len(results), 3)
            self.assertTrue(all(results))

            # Verify entities were added
            for entity in entities:
                retrieved = await self.vector_store.get_entity(entity.id)
                self.assertIsNotNone(retrieved)

        asyncio.run(test())

    def test_search_with_filters(self) -> None:
        """Test search with filter conditions."""

        async def test():
            await self.vector_store.initialize()

            # Create entities with different types
            entity1 = Entity(name="Test Person", entity_type="PERSON")
            entity2 = Entity(name="Test Organization", entity_type="ORGANIZATION")

            await self.vector_store.add_entity(entity1)
            await self.vector_store.add_entity(entity2)

            # Search with filter
            results = await self.vector_store.search_entities(
                query="test", top_k=5, filter_dict={"entity_type": "PERSON"}
            )

            # Should only return the PERSON entity
            self.assertGreater(len(results), 0)
            for entity, score in results:
                self.assertEqual(entity.entity_type, "PERSON")

        asyncio.run(test())

    def test_hybrid_search(self) -> None:
        """Test hybrid search functionality."""

        async def test():
            await self.vector_store.initialize()

            # Add test data
            entity = Entity(name="Test Entity", description="Entity for testing")
            chunk = TextChunk(content="Test content for hybrid search")

            await self.vector_store.add_entity(entity)
            await self.vector_store.add_text_chunk(chunk)

            # Hybrid search
            results = await self.vector_store.hybrid_search(
                query="test", search_types={"entity", "text_chunk"}, top_k=5
            )

            self.assertIn("entity", results)
            self.assertIn("text_chunk", results)
            self.assertGreater(len(results["entity"]), 0)
            self.assertGreater(len(results["text_chunk"]), 0)

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
            self.assertIn("entities", stats)
            self.assertEqual(stats["entities"], 1)

        asyncio.run(test())

    def test_clear_all(self) -> None:
        """Test clear all functionality."""

        async def test():
            await self.vector_store.initialize()

            # Add test data
            entity = Entity(name="Test Entity")
            await self.vector_store.add_entity(entity)

            # Verify data exists
            stats = await self.vector_store.get_stats()
            self.assertEqual(stats["entities"], 1)

            # Clear all data
            result = await self.vector_store.clear_all()
            self.assertTrue(result)

            # Verify data is cleared
            stats = await self.vector_store.get_stats()
            self.assertEqual(stats["entities"], 0)

        asyncio.run(test())

    def test_error_handling_not_initialized(self) -> None:
        """Test error handling when vector store is not initialized."""

        async def test():
            entity = Entity(name="Test Entity")

            with self.assertRaises(VectorStoreError):
                await self.vector_store.add_entity(entity)

        asyncio.run(test())

    def test_collection_names_mapping(self) -> None:
        """Test collection names are properly mapped."""
        expected_names = {
            "entity": "test_kg_entities",
            "relation": "test_kg_relations",
            "cluster": "test_kg_clusters",
            "text_chunk": "test_kg_text_chunks",
        }

        self.assertEqual(self.vector_store._collection_names, expected_names)

    def test_data_preparation_methods(self) -> None:
        """Test data preparation methods."""
        # Test entity data preparation
        entity = Entity(name="Test Entity", description="Test description")
        entity_id, document, metadata = self.vector_store._prepare_entity_data(entity)

        self.assertEqual(entity_id, entity.id)
        self.assertIn("Test Entity", document)
        self.assertIn("Test description", document)
        self.assertEqual(metadata["name"], "Test Entity")

    def test_reconstruction_methods(self) -> None:
        """Test object reconstruction from metadata."""
        # Test entity reconstruction
        entity_id = "test-id"
        metadata = {
            "name": "Test Entity",
            "entity_type": "PERSON",
            "description": "Test description",
            "properties": "{}",
            "aliases": "[]",
            "confidence": 1.0,
            "source": "test",
            "text_chunks": "[]",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
        }

        entity = self.vector_store._reconstruct_entity_from_metadata(entity_id, metadata)

        self.assertEqual(entity.id, entity_id)
        self.assertEqual(entity.name, "Test Entity")
        self.assertEqual(entity.entity_type, "PERSON")

    def test_with_custom_embedding_function(self) -> None:
        """Test with custom embedding function."""

        async def test():
            # Create a proper mock embedding function with correct signature
            class MockEmbeddingFunction:
                def __call__(self, input):  # pylint: disable=redefined-builtin
                    # Return mock embeddings for each input text
                    return [[0.1, 0.2, 0.3] for _ in input]

            mock_embedding_fn = MockEmbeddingFunction()

            custom_store = ChromaVectorStore(
                collection_name="test_custom",
                persist_directory=self.temp_dir,
                embedding_function=mock_embedding_fn,
            )

            await custom_store.initialize()
            self.assertTrue(custom_store.is_initialized())
            await custom_store.close()

        asyncio.run(test())


@unittest.skipUnless(CHROMADB_AVAILABLE, "ChromaDB not available")
class TestChromaVectorStoreWithoutChromaDB(unittest.TestCase):
    """Test ChromaVectorStore import error handling."""

    @patch("agraph.vectordb.chroma.chromadb", None)
    def test_import_error_handling(self) -> None:
        """Test that appropriate error is raised when ChromaDB is not available."""
        # This test would fail to import the module, but we can't easily test
        # the import error scenario in the current setup
        pass


if __name__ == "__main__":
    if not CHROMADB_AVAILABLE:
        print("ChromaDB not available. Skipping tests.")
        print("To run these tests, install ChromaDB with: pip install chromadb>=0.5.0")

    unittest.main()
