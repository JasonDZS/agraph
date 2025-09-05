"""
Relation extraction step implementation.
"""

from typing import Any, Dict, List

from ...base.models.entities import Entity
from ...base.models.relations import Relation
from ...base.models.text import TextChunk
from ...config import BuildSteps
from ..handler.relation_handler import RelationHandler
from .base import BuildStep, StepResult
from .context import BuildContext


class RelationExtractionStep(BuildStep):
    """Step for extracting relations between entities from text chunks."""

    def __init__(self, relation_handler: RelationHandler, cache_manager: Any):
        """
        Initialize relation extraction step.

        Args:
            relation_handler: Handler for relation extraction operations
            cache_manager: Cache manager instance
        """
        super().__init__(BuildSteps.RELATION_EXTRACTION, cache_manager)
        self.relation_handler = relation_handler

    async def _execute_step(self, context: BuildContext) -> StepResult[List[Relation]]:
        """
        Execute relation extraction logic.

        Args:
            context: Build context containing chunks and entities for relation extraction

        Returns:
            StepResult containing list of extracted relations
        """
        # pylint: disable=too-many-return-statements
        try:
            # Get chunks and entities from context
            chunks = context.chunks
            entities = context.entities

            if not chunks:
                return StepResult.failure_result("No chunks available for relation extraction")

            if not entities:
                return StepResult.failure_result("No entities available for relation extraction")

            # Validate inputs
            if not isinstance(chunks, list):
                return StepResult.failure_result("Invalid chunks type: expected list")

            if not isinstance(entities, list):
                return StepResult.failure_result("Invalid entities type: expected list")

            for i, chunk in enumerate(chunks):
                if not isinstance(chunk, TextChunk):
                    return StepResult.failure_result(
                        f"Invalid chunk at index {i}: expected TextChunk, got {type(chunk)}"
                    )

            for i, entity in enumerate(entities):
                if not isinstance(entity, Entity):
                    return StepResult.failure_result(
                        f"Invalid entity at index {i}: expected Entity, got {type(entity)}"
                    )

            # Execute relation extraction (async operation)
            relations = await self.relation_handler.extract_relations_from_chunks(chunks, entities, context.use_cache)

            if not isinstance(relations, list):
                return StepResult.failure_result("Relation extraction returned invalid result type")

            # Validate relations
            for i, relation in enumerate(relations):
                if not isinstance(relation, Relation):
                    return StepResult.failure_result(
                        f"Invalid relation at index {i}: expected Relation, got {type(relation)}"
                    )

            # Calculate some metrics
            unique_relation_types = set(relation.relation_type for relation in relations)
            confidence_scores = [relation.confidence for relation in relations if relation.confidence is not None]
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0

            # Analyze entity connectivity
            entity_connections: Dict[str, set[str]] = {}
            for relation in relations:
                # Use head_entity and tail_entity from the Relation model
                source_id = relation.head_entity.id if relation.head_entity else None
                target_id = relation.tail_entity.id if relation.tail_entity else None

                if source_id and target_id:
                    entity_connections.setdefault(source_id, set()).add(target_id)
                    entity_connections.setdefault(target_id, set()).add(source_id)

            # Calculate network metrics
            total_connections = sum(len(connections) for connections in entity_connections.values())
            avg_connections_per_entity = total_connections / len(entity_connections) if entity_connections else 0

            return StepResult.success_result(
                relations,
                metadata={
                    "input_chunks": len(chunks),
                    "input_entities": len(entities),
                    "extracted_relations": len(relations),
                    "unique_relation_types": len(unique_relation_types),
                    "relation_types": list(unique_relation_types),
                    "average_confidence": avg_confidence,
                    "connected_entities": len(entity_connections),
                    "average_connections_per_entity": avg_connections_per_entity,
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
            return StepResult.failure_result(f"Relation extraction failed: {str(e)}")

    def _get_cache_input_data(self, context: BuildContext) -> Any:
        """Get input data for cache key generation."""
        # Use tuple of chunks and entities for cache key
        return (context.chunks, context.entities)

    def _get_expected_result_type(self) -> type:
        """Get expected result type for cache deserialization."""
        return list
