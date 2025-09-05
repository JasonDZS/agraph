"""
Pipeline factory for creating configured build pipelines.
"""

from typing import Any, Dict, Optional

from ..config import BuildSteps
from ..logger import logger
from .cache import CacheManager
from .handler.cluster_handler import ClusterHandler
from .handler.document_processor import DocumentProcessor
from .handler.entity_handler import EntityHandler
from .handler.graph_assembler import GraphAssembler
from .handler.relation_handler import RelationHandler
from .handler.text_chunker_handler import TextChunkerHandler
from .pipeline import BuildPipeline
from .steps import (
    ClusterFormationStep,
    DocumentProcessingStep,
    EntityExtractionStep,
    GraphAssemblyStep,
    RelationExtractionStep,
    TextChunkingStep,
)


class PipelineFactory:
    """Factory for creating configured build pipelines."""

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize pipeline factory.

        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager

    def create_standard_pipeline(
        self,
        text_chunker_handler: TextChunkerHandler,
        entity_handler: EntityHandler,
        relation_handler: RelationHandler,
        cluster_handler: ClusterHandler,
        graph_assembler: GraphAssembler,
        include_document_processing: bool = False,
        document_processor: Optional[DocumentProcessor] = None,
    ) -> BuildPipeline:
        """
        Create a standard knowledge graph build pipeline.

        Args:
            text_chunker_handler: Handler for text chunking
            entity_handler: Handler for entity extraction
            relation_handler: Handler for relation extraction
            cluster_handler: Handler for cluster formation
            graph_assembler: Handler for graph assembly
            include_document_processing: Whether to include document processing step
            document_processor: Document processor (required if include_document_processing=True)

        Returns:
            Configured BuildPipeline
        """
        pipeline = BuildPipeline(self.cache_manager)

        # Add document processing step if requested
        if include_document_processing:
            if not document_processor:
                raise ValueError("document_processor is required when include_document_processing=True")

            pipeline.add_step(DocumentProcessingStep(document_processor, self.cache_manager))

        # Add standard steps
        pipeline.add_step(TextChunkingStep(text_chunker_handler, self.cache_manager))
        pipeline.add_step(EntityExtractionStep(entity_handler, self.cache_manager))
        pipeline.add_step(RelationExtractionStep(relation_handler, self.cache_manager))
        pipeline.add_step(ClusterFormationStep(cluster_handler, self.cache_manager))
        pipeline.add_step(GraphAssemblyStep(graph_assembler, self.cache_manager))

        logger.debug(f"Created standard pipeline with {len(pipeline)} steps")
        return pipeline

    def create_text_only_pipeline(
        self,
        text_chunker_handler: TextChunkerHandler,
        entity_handler: EntityHandler,
        relation_handler: RelationHandler,
        cluster_handler: ClusterHandler,
        graph_assembler: GraphAssembler,
    ) -> BuildPipeline:
        """
        Create a text-only pipeline (skips document processing).

        This is optimized for build_from_text usage.

        Args:
            text_chunker_handler: Handler for text chunking
            entity_handler: Handler for entity extraction
            relation_handler: Handler for relation extraction
            cluster_handler: Handler for cluster formation
            graph_assembler: Handler for graph assembly

        Returns:
            Configured BuildPipeline
        """
        return self.create_standard_pipeline(
            text_chunker_handler=text_chunker_handler,
            entity_handler=entity_handler,
            relation_handler=relation_handler,
            cluster_handler=cluster_handler,
            graph_assembler=graph_assembler,
            include_document_processing=False,
        )

    def create_custom_pipeline(self, step_config: Dict[str, Any]) -> BuildPipeline:
        """
        Create a custom pipeline based on configuration.

        Args:
            step_config: Configuration dictionary specifying which steps to include
                       Format: {step_name: handler_instance or True/False}

        Returns:
            Configured BuildPipeline
        """
        pipeline = BuildPipeline(self.cache_manager)

        # Standard step mapping
        step_classes = {
            BuildSteps.TEXT_CHUNKING: TextChunkingStep,
            BuildSteps.ENTITY_EXTRACTION: EntityExtractionStep,
            BuildSteps.RELATION_EXTRACTION: RelationExtractionStep,
            BuildSteps.CLUSTER_FORMATION: ClusterFormationStep,
            BuildSteps.GRAPH_ASSEMBLY: GraphAssemblyStep,
        }

        # Add steps based on configuration
        for step_name, step_handler in step_config.items():
            if step_handler and step_name in step_classes:
                step_class = step_classes[step_name]
                step_instance = step_class(step_handler, self.cache_manager)
                pipeline.add_step(step_instance)
                logger.debug(f"Added {step_name} to custom pipeline")

        logger.debug(f"Created custom pipeline with {len(pipeline)} steps")
        return pipeline

    def create_minimal_pipeline(
        self, text_chunker_handler: TextChunkerHandler, graph_assembler: GraphAssembler
    ) -> BuildPipeline:
        """
        Create a minimal pipeline with only text chunking and graph assembly.

        This is useful for basic text processing without knowledge graph features.

        Args:
            text_chunker_handler: Handler for text chunking
            graph_assembler: Handler for graph assembly

        Returns:
            Configured BuildPipeline
        """
        pipeline = BuildPipeline(self.cache_manager)

        pipeline.add_step(TextChunkingStep(text_chunker_handler, self.cache_manager))
        pipeline.add_step(GraphAssemblyStep(graph_assembler, self.cache_manager))

        logger.debug("Created minimal pipeline with 2 steps")
        return pipeline

    def create_parallel_pipeline(
        self,
        text_chunker_handler: TextChunkerHandler,
        entity_handler: EntityHandler,
        relation_handler: RelationHandler,
        cluster_handler: ClusterHandler,
        graph_assembler: GraphAssembler,
    ) -> BuildPipeline:
        """
        Create a pipeline optimized for parallel execution.

        Note: This returns a standard sequential pipeline for now, but the structure
        allows for future parallel execution implementation.

        Args:
            text_chunker_handler: Handler for text chunking
            entity_handler: Handler for entity extraction
            relation_handler: Handler for relation extraction
            cluster_handler: Handler for cluster formation
            graph_assembler: Handler for graph assembly

        Returns:
            Configured BuildPipeline (sequential for now)
        """
        # For now, return standard pipeline
        # Future implementation could support parallel entity/relation extraction
        pipeline = self.create_text_only_pipeline(
            text_chunker_handler, entity_handler, relation_handler, cluster_handler, graph_assembler
        )

        logger.debug("Created parallel-ready pipeline (currently sequential)")
        return pipeline


class PipelineBuilder:
    """Builder pattern for creating pipelines with fluent interface."""

    def __init__(self, cache_manager: CacheManager):
        """Initialize pipeline builder."""
        self.cache_manager = cache_manager
        self._pipeline = BuildPipeline(cache_manager)

    def with_text_chunking(self, handler: TextChunkerHandler) -> "PipelineBuilder":
        """Add text chunking step."""
        self._pipeline.add_step(TextChunkingStep(handler, self.cache_manager))
        return self

    def with_entity_extraction(self, handler: EntityHandler) -> "PipelineBuilder":
        """Add entity extraction step."""
        self._pipeline.add_step(EntityExtractionStep(handler, self.cache_manager))
        return self

    def with_relation_extraction(self, handler: RelationHandler) -> "PipelineBuilder":
        """Add relation extraction step."""
        self._pipeline.add_step(RelationExtractionStep(handler, self.cache_manager))
        return self

    def with_cluster_formation(self, handler: ClusterHandler) -> "PipelineBuilder":
        """Add cluster formation step."""
        self._pipeline.add_step(ClusterFormationStep(handler, self.cache_manager))
        return self

    def with_graph_assembly(self, handler: GraphAssembler) -> "PipelineBuilder":
        """Add graph assembly step."""
        self._pipeline.add_step(GraphAssemblyStep(handler, self.cache_manager))
        return self

    def build(self) -> BuildPipeline:
        """Build and return the configured pipeline."""
        logger.debug(f"Built pipeline with {len(self._pipeline)} steps using builder pattern")
        return self._pipeline

    def reset(self) -> "PipelineBuilder":
        """Reset builder to create a new pipeline."""
        self._pipeline = BuildPipeline(self.cache_manager)
        return self
