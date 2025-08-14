"""
Example demonstrating incremental updates functionality.
"""

import tempfile
import time
from pathlib import Path

from agraph.builder import KnowledgeGraphBuilder
from agraph.config import BuilderConfig


def main():
    """Demonstrate incremental updates functionality."""
    # Create temporary directory for the example
    temp_dir = Path(tempfile.mkdtemp())
    cache_dir = temp_dir / "cache"

    print(f"Working in temporary directory: {temp_dir}")
    print(f"Cache directory: {cache_dir}")

    # Create test documents
    doc1_path = temp_dir / "document1.txt"
    doc2_path = temp_dir / "document2.txt"
    doc3_path = temp_dir / "document3.txt"

    doc1_path.write_text("This is the first document about artificial intelligence.")
    doc2_path.write_text("This is the second document about machine learning.")
    doc3_path.write_text("This is the third document about deep learning.")

    # Create builder with caching enabled
    config = BuilderConfig()
    config.cache_dir = str(cache_dir)
    config.enable_cache = True
    config.chunk_size = 200  # Small chunks for demo
    config.chunk_overlap = 50  # Must be less than chunk_size

    builder = KnowledgeGraphBuilder(config=config)

    print("\n=== First Processing Run ===")
    print("Processing all 3 documents for the first time...")

    start_time = time.time()
    documents = [doc1_path, doc2_path, doc3_path]
    texts = builder.process_documents(documents)
    first_run_time = time.time() - start_time

    print(f"Processed {len(texts)} documents in {first_run_time:.2f} seconds")

    # Get processing status
    status = builder.get_document_processing_status()
    print(f"Document processing summary: {status}")

    print("\n=== Second Processing Run (All Cached) ===")
    print("Processing the same documents again (should use cache)...")

    start_time = time.time()
    texts = builder.process_documents(documents)
    second_run_time = time.time() - start_time

    print(f"Processed {len(texts)} documents in {second_run_time:.2f} seconds")
    print(f"Speedup: {first_run_time / second_run_time:.1f}x faster")

    print("\n=== Modifying a Document ===")
    print("Modifying document2.txt...")

    # Wait a bit to ensure different timestamp
    time.sleep(0.1)
    doc2_path.write_text("This is the MODIFIED second document about machine learning and AI.")

    start_time = time.time()
    texts = builder.process_documents(documents)
    third_run_time = time.time() - start_time

    print(f"Processed {len(texts)} documents in {third_run_time:.2f} seconds")
    print("Only the modified document should have been reprocessed.")

    # Show individual document status
    print("\n=== Individual Document Status ===")
    for doc in documents:
        status = builder.get_document_processing_status(doc)
        print(f"{doc.name}: {status}")

    print("\n=== Adding a New Document ===")
    doc4_path = temp_dir / "document4.txt"
    doc4_path.write_text("This is a NEW fourth document about neural networks.")

    extended_documents = documents + [doc4_path]

    start_time = time.time()
    texts = builder.process_documents(extended_documents)
    fourth_run_time = time.time() - start_time

    print(f"Processed {len(texts)} documents in {fourth_run_time:.2f} seconds")
    print("Only the new document should have been processed.")

    # Final status summary
    final_status = builder.get_document_processing_status()
    print(f"\n=== Final Processing Summary ===")
    print(f"Total documents: {final_status['total_documents']}")
    print(f"Completed: {final_status['completed']}")
    print(f"Failed: {final_status['failed']}")
    print(f"Total processing time: {final_status['total_processing_time']:.2f} seconds")

    # Get cache information
    cache_info = builder.get_cache_info()
    print(f"\n=== Cache Information ===")
    print(f"Backend: {cache_info['backend']}")
    print(f"Document Processing: {cache_info['document_processing']}")

    print("\n=== Force Reprocess Example ===")
    print("Forcing reprocessing of document1.txt...")

    success = builder.force_reprocess_document(doc1_path)
    print(f"Cache cleared: {success}")

    start_time = time.time()
    texts = builder.process_documents([doc1_path])
    reprocess_time = time.time() - start_time

    print(f"Reprocessed document in {reprocess_time:.2f} seconds")

    print(f"\n=== Cleanup ===")
    print(f"Temporary files created in: {temp_dir}")
    print("You can inspect the cache structure and files there.")
    print("The directory will be automatically cleaned up when you exit Python.")


if __name__ == "__main__":
    main()
