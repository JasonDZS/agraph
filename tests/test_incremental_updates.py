"""
Test incremental updates functionality.
"""

import tempfile
import time
from pathlib import Path
from unittest import TestCase, mock
from unittest.mock import MagicMock

from agraph.builder import KnowledgeGraphBuilder
from agraph.builder.cache import CacheManager
from agraph.config import BuilderConfig, DocumentProcessingStatus


class TestIncrementalUpdates(TestCase):
    """Test incremental updates functionality."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_dir = Path(self.temp_dir) / "cache"

        # Create test config
        self.config = BuilderConfig()
        self.config.cache_dir = str(self.cache_dir)
        self.config.enable_cache = True

        # Create test documents
        self.doc1_path = Path(self.temp_dir) / "doc1.txt"
        self.doc2_path = Path(self.temp_dir) / "doc2.txt"

        self.doc1_path.write_text("This is document 1 content.")
        self.doc2_path.write_text("This is document 2 content.")

    def test_cache_manager_document_status(self):
        """Test document status tracking in cache manager."""
        cache_manager = CacheManager(self.config)

        # Test file hash calculation
        hash1 = cache_manager.get_file_hash(self.doc1_path)
        self.assertIsInstance(hash1, str)
        self.assertTrue(len(hash1) > 0)

        # Test document is not processed initially
        self.assertFalse(cache_manager.is_document_processed(self.doc1_path))

        # Test document status creation and retrieval
        from datetime import datetime

        status = DocumentProcessingStatus(
            file_path=str(self.doc1_path),
            file_hash=hash1,
            last_modified=datetime.fromtimestamp(self.doc1_path.stat().st_mtime),
            processing_status="completed",
            extracted_text_hash="test_hash",
            processing_time=1.5,
        )

        cache_manager.update_document_status(self.doc1_path, status)
        retrieved_status = cache_manager.get_document_status(self.doc1_path)

        self.assertIsNotNone(retrieved_status)
        self.assertEqual(retrieved_status.file_path, str(self.doc1_path))
        self.assertEqual(retrieved_status.processing_status, "completed")
        self.assertEqual(retrieved_status.processing_time, 1.5)

    def test_processed_vs_unprocessed_separation(self):
        """Test separation of processed and unprocessed documents."""
        cache_manager = CacheManager(self.config)

        # Save processing result for doc1
        cache_manager.save_document_processing_result(self.doc1_path, "extracted content 1", 1.0)

        # Test separation
        documents = [self.doc1_path, self.doc2_path]
        processed, unprocessed = cache_manager.get_processed_documents(documents)

        self.assertEqual(len(processed), 1)
        self.assertEqual(len(unprocessed), 1)
        self.assertIn(self.doc1_path, processed)
        self.assertIn(self.doc2_path, unprocessed)

    def test_cached_document_results_retrieval(self):
        """Test retrieval of cached document processing results."""
        cache_manager = CacheManager(self.config)

        # Save processing results
        text1 = "extracted content from doc1"
        text2 = "extracted content from doc2"

        cache_manager.save_document_processing_result(self.doc1_path, text1, 1.0)
        cache_manager.save_document_processing_result(self.doc2_path, text2, 1.5)

        # Retrieve cached results
        cached_results = cache_manager.get_cached_document_results([self.doc1_path, self.doc2_path])

        self.assertEqual(len(cached_results), 2)
        self.assertEqual(cached_results[str(self.doc1_path)], text1)
        self.assertEqual(cached_results[str(self.doc2_path)], text2)

    @mock.patch("agraph.processor.factory.DocumentProcessorFactory")
    def test_incremental_document_processing(self, mock_factory):
        """Test incremental document processing with KnowledgeGraphBuilder."""
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process.side_effect = ["processed content 1", "processed content 2"]
        mock_factory_instance = MagicMock()
        mock_factory_instance.get_processor.return_value = mock_processor
        mock_factory.return_value = mock_factory_instance

        # Create builder
        builder = KnowledgeGraphBuilder(config=self.config)
        builder.processor_factory = mock_factory_instance

        documents = [self.doc1_path, self.doc2_path]

        # First processing run - should process both documents
        texts1 = builder.document_processor.process_documents(documents)

        self.assertEqual(len(texts1), 2)
        self.assertEqual(texts1[0], "processed content 1")
        self.assertEqual(texts1[1], "processed content 2")

        # Verify both documents were processed
        self.assertEqual(mock_processor.process.call_count, 2)

        # Reset mock
        mock_processor.reset_mock()

        # Second processing run - should use cache (no new processing)
        texts2 = builder.process_documents(documents)

        self.assertEqual(len(texts2), 2)
        self.assertEqual(texts2[0], "processed content 1")
        self.assertEqual(texts2[1], "processed content 2")

        # Verify no new processing happened
        self.assertEqual(mock_processor.process.call_count, 0)

    @mock.patch("agraph.processor.factory.DocumentProcessorFactory")
    def test_modified_document_reprocessing(self, mock_factory):
        """Test that modified documents are reprocessed."""
        # Mock processor
        mock_processor = MagicMock()
        mock_processor.process.side_effect = ["original content", "modified content"]
        mock_factory_instance = MagicMock()
        mock_factory_instance.get_processor.return_value = mock_processor
        mock_factory.return_value = mock_factory_instance

        builder = KnowledgeGraphBuilder(config=self.config)
        builder.processor_factory = mock_factory_instance

        # First processing
        texts1 = builder.document_processor.process_documents([self.doc1_path])
        self.assertEqual(texts1[0], "original content")
        self.assertEqual(mock_processor.process.call_count, 1)

        # Modify document
        time.sleep(0.1)  # Ensure different timestamp
        self.doc1_path.write_text("Modified document 1 content.")

        # Reset mock
        mock_processor.reset_mock()

        # Second processing - should reprocess modified document
        texts2 = builder.document_processor.process_documents([self.doc1_path])
        self.assertEqual(texts2[0], "modified content")
        self.assertEqual(mock_processor.process.call_count, 1)

    def test_document_processing_summary(self):
        """Test document processing summary functionality."""
        cache_manager = CacheManager(self.config)

        # Save various processing results
        cache_manager.save_document_processing_result(self.doc1_path, "content 1", 1.0)

        # Create a failed processing status
        from datetime import datetime

        failed_status = DocumentProcessingStatus(
            file_path=str(self.doc2_path),
            file_hash=cache_manager.get_file_hash(self.doc2_path),
            last_modified=datetime.fromtimestamp(self.doc2_path.stat().st_mtime),
            processing_status="failed",
            error_message="Test error",
        )
        cache_manager.update_document_status(self.doc2_path, failed_status)

        # Get summary
        summary = cache_manager.get_document_processing_summary()

        self.assertEqual(summary["total_documents"], 2)
        self.assertEqual(summary["completed"], 1)
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(summary["processing"], 0)
        self.assertEqual(summary["pending"], 0)
        self.assertGreater(summary["total_processing_time"], 0)

    def test_clear_document_cache(self):
        """Test clearing document cache functionality."""
        cache_manager = CacheManager(self.config)

        # Save processing results
        cache_manager.save_document_processing_result(self.doc1_path, "content 1", 1.0)
        cache_manager.save_document_processing_result(self.doc2_path, "content 2", 1.5)

        # Verify documents are processed
        self.assertTrue(cache_manager.is_document_processed(self.doc1_path))
        self.assertTrue(cache_manager.is_document_processed(self.doc2_path))

        # Clear specific document cache
        cleared_count = cache_manager.clear_document_cache(self.doc1_path)
        self.assertGreater(cleared_count, 0)

        # Verify only doc1 cache was cleared
        self.assertFalse(cache_manager.is_document_processed(self.doc1_path))
        self.assertTrue(cache_manager.is_document_processed(self.doc2_path))

        # Clear all document cache
        cleared_count = cache_manager.clear_document_cache()
        self.assertGreater(cleared_count, 0)

        # Verify all cache was cleared
        self.assertFalse(cache_manager.is_document_processed(self.doc1_path))
        self.assertFalse(cache_manager.is_document_processed(self.doc2_path))

    def tearDown(self):
        """Clean up test environment."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)


if __name__ == "__main__":
    import unittest

    unittest.main()
