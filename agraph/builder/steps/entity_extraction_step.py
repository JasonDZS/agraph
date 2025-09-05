"""
Entity extraction step implementation.
"""

from typing import Any, List

from ...base.models.entities import Entity
from ...base.models.text import TextChunk
from ...config import BuildSteps
from ..handler.entity_handler import EntityHandler
from .base import BuildStep, StepResult
from .context import BuildContext


class EntityExtractionStep(BuildStep):
    """Step for extracting entities from text chunks."""

    def __init__(self, entity_handler: EntityHandler, cache_manager: Any):
        """
        Initialize entity extraction step.

        Args:
            entity_handler: Handler for entity extraction operations
            cache_manager: Cache manager instance
        """
        super().__init__(BuildSteps.ENTITY_EXTRACTION, cache_manager)
        self.entity_handler = entity_handler

    async def _execute_step(self, context: BuildContext) -> StepResult[List[Entity]]:
        """
        Execute entity extraction logic.

        Args:
            context: Build context containing chunks to extract entities from

        Returns:
            StepResult containing list of extracted entities
        """
        # pylint: disable=too-many-return-statements
        try:
            # Get chunks from context
            chunks = context.chunks
            if not chunks:
                return StepResult.failure_result("No chunks available for entity extraction")

            # Validate chunks
            if not isinstance(chunks, list):
                return StepResult.failure_result("Invalid chunks type: expected list")

            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, TextChunk):
                    return StepResult.failure_result(
                        f"Invalid chunk at index {i}: expected TextChunk, got {type(chunk)}"
                    )

            # Execute entity extraction (async operation)
            entities = await self.entity_handler.extract_entities_from_chunks(chunks, context.use_cache)

            if not isinstance(entities, list):
                return StepResult.failure_result("Entity extraction returned invalid result type")

            # Validate entities
            for i, entity in enumerate(entities):
                if not isinstance(entity, Entity):
                    return StepResult.failure_result(
                        f"Invalid entity at index {i}: expected Entity, got {type(entity)}"
                    )

            # Calculate some metrics
            unique_entity_types = set(entity.entity_type for entity in entities)
            confidence_scores = [entity.confidence for entity in entities if entity.confidence is not None]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            return StepResult.success_result(
                entities,
                metadata={
                    "input_chunks": len(chunks),
                    "extracted_entities": len(entities),
                    "unique_entity_types": len(unique_entity_types),
                    "entity_types": list(unique_entity_types),
                    "average_confidence": avg_confidence,
                    "confidence_distribution": (
                        {
                            "high (>0.8)": len([c for c in confidence_scores if c > 0.8]),
                            "medium (0.5-0.8)": len([c for c in confidence_scores if 0.5 <= c <= 0.8]),
                            "low (<0.5)": len([c for c in confidence_scores if c < 0.5]),
                        }
                        if confidence_scores
                        else {}
                    ),
                },
            )

        except Exception as e:
            return StepResult.failure_result(f"Entity extraction failed: {str(e)}")

    def _get_cache_input_data(self, context: BuildContext) -> Any:
        """Get input data for cache key generation."""
        return context.chunks

    def _get_expected_result_type(self) -> type:
        """Get expected result type for cache deserialization."""
        return list
