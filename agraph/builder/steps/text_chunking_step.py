"""
Text chunking step implementation.
"""

from typing import Any, List

from ...base.models.text import TextChunk
from ...config import BuildSteps
from ..handler.text_chunker_handler import TextChunkerHandler
from .base import BuildStep, StepResult
from .context import BuildContext


class TextChunkingStep(BuildStep):
    """Step for chunking text into smaller pieces."""

    def __init__(self, text_chunker_handler: TextChunkerHandler, cache_manager: Any):
        """
        Initialize text chunking step.

        Args:
            text_chunker_handler: Handler for text chunking operations
            cache_manager: Cache manager instance
        """
        super().__init__(BuildSteps.TEXT_CHUNKING, cache_manager)
        self.text_chunker_handler = text_chunker_handler

    async def _execute_step(self, context: BuildContext) -> StepResult[List[TextChunk]]:
        """
        Execute text chunking logic.

        Args:
            context: Build context containing texts to chunk

        Returns:
            StepResult containing list of text chunks
        """
        try:
            if not context.texts:
                return StepResult.failure_result("No texts provided for chunking")

            # Use documents if available (for build_from_documents), otherwise use texts
            input_data = context.documents if context.documents else context.texts

            # Execute text chunking
            chunks = self.text_chunker_handler.chunk_texts(
                input_data,
                context.use_cache,
                context.documents,  # Pass documents for proper caching
            )

            if not isinstance(chunks, list):
                return StepResult.failure_result("Text chunking returned invalid result type")

            # Validate chunks
            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, TextChunk):
                    return StepResult.failure_result(
                        f"Invalid chunk at index {i}: expected TextChunk, got {type(chunk)}"
                    )

            return StepResult.success_result(
                chunks,
                metadata={
                    "input_count": len(context.texts),
                    "output_count": len(chunks),
                    "average_chunk_size": (sum(len(chunk.content) for chunk in chunks) / len(chunks) if chunks else 0),
                },
            )

        except Exception as e:
            return StepResult.failure_result(f"Text chunking failed: {str(e)}")

    def _get_cache_input_data(self, context: BuildContext) -> Any:
        """Get input data for cache key generation."""
        # Use documents if available for consistent caching with build_from_documents
        return context.documents if context.documents else context.texts

    def _get_expected_result_type(self) -> type:
        """Get expected result type for cache deserialization."""
        return list
