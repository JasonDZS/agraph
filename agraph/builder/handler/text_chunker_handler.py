"""
Text chunking handler for knowledge graph builder.
"""

from typing import List

from ...base.text import TextChunk
from ...builder.cache import CacheManager
from ...chunker import TokenChunker
from ...config import BuilderConfig, BuildSteps
from ...logger import logger


class TextChunkerHandler:
    """Handles text chunking with incremental processing support."""

    def __init__(self, cache_manager: CacheManager, config: BuilderConfig, chunker: TokenChunker):
        """Initialize text chunker handler.

        Args:
            cache_manager: Cache manager instance
            config: Builder configuration
            chunker: Text chunker instance
        """
        self.cache_manager = cache_manager
        self.config = config
        self.chunker = chunker

    def chunk_texts(self, texts: List[str], use_cache: bool = True) -> List[TextChunk]:
        """Chunk texts into smaller pieces with incremental processing support.

        Args:
            texts: List of texts to chunk
            use_cache: Whether to use caching

        Returns:
            List of text chunks
        """
        logger.info(f"Chunking {len(texts)} texts with chunk size {self.config.chunk_size}")

        if not use_cache:
            return self._chunk_all_texts(texts)

        # Try incremental approach: check for cached chunks per text
        all_chunks = []
        texts_to_process = []
        cached_chunks_map = {}

        for i, text in enumerate(texts):
            # Generate individual text cache key
            text_key = f"text_chunks_{self.cache_manager.backend.generate_key(text)}"
            cached_chunks = self.cache_manager.backend.get(text_key, list)

            if cached_chunks is not None:
                # Update chunk indices to maintain global order
                for chunk in cached_chunks:
                    if hasattr(chunk, "id"):
                        chunk.id = f"chunk_{i}_{chunk.id.split('_')[-1]}"
                    if hasattr(chunk, "source"):
                        chunk.source = f"document_{i}"
                all_chunks.extend(cached_chunks)
                cached_chunks_map[i] = cached_chunks
                logger.debug(f"Using cached chunks for text {i+1}: {len(cached_chunks)} chunks")
            else:
                texts_to_process.append((i, text))

        # Process uncached texts
        if texts_to_process:
            logger.info(f"Processing {len(texts_to_process)} uncached texts for chunking")
            for i, text in texts_to_process:
                try:
                    logger.debug(f"Chunking text {i+1}/{len(texts)} - length: {len(text)} chars")
                    chunks_with_positions = self.chunker.split_text_with_positions(text)
                    logger.debug(f"Text {i+1} split into {len(chunks_with_positions)} chunks")

                    text_chunks = []
                    for j, (chunk_text, start_idx, end_idx) in enumerate(chunks_with_positions):
                        chunk = TextChunk(
                            id=f"chunk_{i}_{j}",
                            content=chunk_text,
                            title=f"Document {i} Chunk {j}",
                            start_index=start_idx,
                            end_index=end_idx,
                            source=f"document_{i}",
                        )
                        text_chunks.append(chunk)
                        all_chunks.append(chunk)

                    # Cache chunks for this specific text
                    text_key = f"text_chunks_{self.cache_manager.backend.generate_key(text)}"
                    self.cache_manager.backend.set(text_key, text_chunks)
                    logger.debug(f"Cached {len(text_chunks)} chunks for text {i+1}")

                except Exception as e:
                    logger.error(f"Error chunking text {i}: {e}")
                    continue

        # Sort chunks by their original document index to maintain order
        all_chunks.sort(key=lambda x: (int(x.source.split("_")[1]), int(x.id.split("_")[2])))

        logger.info(
            f"Text chunking completed - created {len(all_chunks)} chunks from {len(texts)} texts"
            f" ({len(texts_to_process)} newly processed, {len(texts) - len(texts_to_process)} from cache)"
        )

        # Save step-level result for compatibility
        if use_cache:
            self.cache_manager.save_step_result(BuildSteps.TEXT_CHUNKING, texts, all_chunks)

        return all_chunks

    def _chunk_all_texts(self, texts: List[str]) -> List[TextChunk]:
        """Chunk all texts without caching (fallback method)."""
        all_chunks = []

        for i, text in enumerate(texts):
            try:
                logger.debug(f"Chunking text {i+1}/{len(texts)} - length: {len(text)} chars")
                chunks_with_positions = self.chunker.split_text_with_positions(text)
                logger.debug(f"Text {i+1} split into {len(chunks_with_positions)} chunks")

                for j, (chunk_text, start_idx, end_idx) in enumerate(chunks_with_positions):
                    chunk = TextChunk(
                        id=f"chunk_{i}_{j}",
                        content=chunk_text,
                        title=f"Document {i} Chunk {j}",
                        start_index=start_idx,
                        end_index=end_idx,
                        source=f"document_{i}",
                    )
                    all_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Error chunking text {i}: {e}")
                continue

        return all_chunks
