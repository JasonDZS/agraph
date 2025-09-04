"""
Refactored KnowledgeGraphBuilder using pipeline architecture.

This version maintains the same public API but uses the new step-based pipeline
architecture internally for better maintainability and extensibility.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from ..base.graphs.legacy import KnowledgeGraph
from ..base.graphs.optimized import OptimizedKnowledgeGraph
from ..base.infrastructure.dao import MemoryDataAccessLayer

# Import unified architecture components
from ..base.managers.factory import create_managers
from ..base.models.clusters import Cluster
from ..base.models.entities import Entity
from ..base.models.relations import Relation
from ..base.models.text import TextChunk
from ..chunker import TokenChunker
from ..config import BuilderConfig, BuildSteps
from ..logger import logger
from ..processor.factory import DocumentProcessorFactory
from .cache import CacheManager
from .clustering import (
    ClusterAlgorithm,
    CommunityDetectionAlgorithm,
    HierarchicalClusteringAlgorithm,
)
from .extractors import EntityExtractor, LLMEntityExtractor, LLMRelationExtractor, RelationExtractor
from .handler.cluster_handler import ClusterHandler
from .handler.document_processor import DocumentProcessor
from .handler.entity_handler import EntityHandler
from .handler.graph_assembler import GraphAssembler
from .handler.relation_handler import RelationHandler
from .handler.text_chunker_handler import TextChunkerHandler
from .pipeline import BuildPipeline
from .pipeline_factory import PipelineFactory, PipelineBuilder
from .steps.context import BuildContext

T = TypeVar("T")


class KnowledgeGraphBuilderV2:
    """
    Refactored KnowledgeGraphBuilder using pipeline architecture.
    
    This version maintains backward compatibility while using the new
    step-based pipeline architecture internally.
    """

    def __init__(
        self,
        config: Optional[BuilderConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        chunker: Optional[TokenChunker] = None,
        entity_extractor: Optional[EntityExtractor] = None,
        relation_extractor: Optional[RelationExtractor] = None,
        cluster_algorithm: Optional[ClusterAlgorithm] = None,
        enable_knowledge_graph: bool = True,
    ):
        """Initialize KnowledgeGraphBuilder with pipeline architecture.

        Args:
            config: Builder configuration
            cache_dir: Cache directory (overrides config if provided)
            chunker: Text chunker instance
            entity_extractor: Entity extractor instance
            relation_extractor: Relation extractor instance
            cluster_algorithm: Clustering algorithm instance
            enable_knowledge_graph: Whether to enable knowledge graph construction
        """
        # Initialize configuration
        self.config = config or BuilderConfig()
        if cache_dir is not None:
            self.config.cache_dir = str(cache_dir)

        # Store knowledge graph toggle
        self.enable_knowledge_graph = enable_knowledge_graph

        # Initialize cache manager
        self.cache_manager = CacheManager(self.config)

        # Initialize components (same as original)
        self.chunker = chunker or TokenChunker(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )

        self.entity_extractor = entity_extractor or LLMEntityExtractor(
            config={
                "min_confidence": self.config.entity_confidence_threshold,
            },
            builder_config=self.config
        )

        self.relation_extractor = relation_extractor or LLMRelationExtractor(
            config={
                "min_confidence": self.config.relation_confidence_threshold,
            },
            builder_config=self.config
        )

        self.cluster_algorithm = cluster_algorithm or self._create_cluster_algorithm()

        # Initialize processor factory
        self.processor_factory = DocumentProcessorFactory()

        # Initialize handlers (same as original)
        self.document_processor = DocumentProcessor(self.cache_manager, self.processor_factory)
        self.text_chunker_handler = TextChunkerHandler(
            self.cache_manager, self.config, self.chunker
        )
        self.entity_handler = EntityHandler(self.cache_manager, self.entity_extractor)
        self.relation_handler = RelationHandler(self.cache_manager, self.relation_extractor)
        self.cluster_handler = ClusterHandler(self.cache_manager, self.cluster_algorithm)
        self.graph_assembler = GraphAssembler(self.cache_manager)

        # Initialize unified architecture components
        self._dao = MemoryDataAccessLayer()
        self._managers = create_managers("optimized", dao=self._dao)
        self._use_unified_architecture = True

        # Pass unified managers to graph assembler
        if hasattr(self.graph_assembler, "set_unified_managers"):
            self.graph_assembler.set_unified_managers(self._managers)

        # Initialize pipeline factory
        self.pipeline_factory = PipelineFactory(self.cache_manager)
        
        # Pipeline builder for custom pipelines
        self.pipeline_builder = PipelineBuilder(self.cache_manager)
        
        # Store last build context for cache optimization
        self.last_build_context: Optional[BuildContext] = None

        logger.info(
            f"KnowledgeGraphBuilderV2 initialized - "
            f"Enable KG: {enable_knowledge_graph}, "
            f"LLM Provider: {self.config.llm_provider}, "
            f"LLM Model: {self.config.llm_model}, "
            f"Cache Dir: {self.config.cache_dir}, "
            f"Chunk Size: {self.config.chunk_size}, "
            f"Entity Threshold: {self.config.entity_confidence_threshold}, "
            f"Relation Threshold: {self.config.relation_confidence_threshold}, "
            f"Cluster Algorithm: {self.config.cluster_algorithm}"
        )

    async def build_from_documents(
        self,
        documents: List[Union[str, Path]],
        graph_name: str = "",
        graph_description: str = "",
        use_cache: bool = True,
        from_step: Optional[str] = None,
    ) -> Union[KnowledgeGraph, OptimizedKnowledgeGraph]:
        """
        Build knowledge graph from document files using pipeline architecture.

        Args:
            documents: List of document file paths
            graph_name: Name for the knowledge graph
            graph_description: Description for the knowledge graph
            use_cache: Whether to use caching
            from_step: Step to start from (for resuming)

        Returns:
            Constructed knowledge graph
        """
        logger.info(
            f"Starting knowledge graph build from {len(documents)} documents - "
            f"Graph Name: '{graph_name}', Use Cache: {use_cache}, From Step: {from_step}"
        )

        try:
            # Create build context
            context = BuildContext(
                texts=[],  # Will be populated by document processing step
                documents=documents,
                graph_name=graph_name,
                graph_description=graph_description,
                use_cache=use_cache,
                from_step=from_step,
                enable_knowledge_graph=self.enable_knowledge_graph
            )

            # Create pipeline with document processing
            pipeline = self.pipeline_factory.create_standard_pipeline(
                text_chunker_handler=self.text_chunker_handler,
                entity_handler=self.entity_handler,
                relation_handler=self.relation_handler,
                cluster_handler=self.cluster_handler,
                graph_assembler=self.graph_assembler,
                include_document_processing=True,
                document_processor=self.document_processor
            )

            # Execute pipeline
            knowledge_graph = await pipeline.execute(context)

            # Store the build context for cache optimization
            self.last_build_context = context

            logger.info(
                f"Knowledge graph build completed successfully - "
                f"Name: '{knowledge_graph.name}', Entities: {len(knowledge_graph.entities)}, "
                f"Relations: {len(knowledge_graph.relations)}"
            )
            
            return knowledge_graph

        except Exception as e:
            logger.error(f"Knowledge graph build failed with error: {str(e)}")
            self.cache_manager.update_build_status(error_message=str(e))
            raise

    async def build_from_text(
        self,
        texts: List[str],
        graph_name: str = "",
        graph_description: str = "",
        use_cache: bool = True,
        from_step: Optional[str] = None,
    ) -> Union[KnowledgeGraph, OptimizedKnowledgeGraph]:
        """
        Build knowledge graph from text strings using pipeline architecture.

        Args:
            texts: List of text strings
            graph_name: Name for the knowledge graph
            graph_description: Description for the knowledge graph
            use_cache: Whether to use caching
            from_step: Step to start from (for resuming)

        Returns:
            Constructed knowledge graph
        """
        # Adjust from_step for text-only pipeline (skip document processing)
        actual_from_step = (
            from_step if from_step != BuildSteps.DOCUMENT_PROCESSING else BuildSteps.TEXT_CHUNKING
        )

        logger.info(
            f"Starting knowledge graph build from {len(texts)} text strings - "
            f"Graph Name: '{graph_name}', Use Cache: {use_cache}, From Step: {from_step}"
        )

        try:
            # Create build context
            context = BuildContext(
                texts=texts,
                graph_name=graph_name,
                graph_description=graph_description,
                use_cache=use_cache,
                from_step=actual_from_step,
                enable_knowledge_graph=self.enable_knowledge_graph
            )

            # Mark document processing as completed since we skip it
            if from_step is None:
                context.mark_step_completed(
                    BuildSteps.DOCUMENT_PROCESSING,
                    0.0,  # No execution time
                    texts  # Document processing "result" is the input texts
                )

            # Create text-only pipeline
            pipeline = self.pipeline_factory.create_text_only_pipeline(
                text_chunker_handler=self.text_chunker_handler,
                entity_handler=self.entity_handler,
                relation_handler=self.relation_handler,
                cluster_handler=self.cluster_handler,
                graph_assembler=self.graph_assembler
            )

            # Execute pipeline
            knowledge_graph = await pipeline.execute(context)

            # Store the build context for cache optimization
            self.last_build_context = context

            logger.info(
                f"Knowledge graph build completed successfully - "
                f"Name: '{knowledge_graph.name}', Entities: {len(knowledge_graph.entities)}, "
                f"Relations: {len(knowledge_graph.relations)}"
            )
            
            return knowledge_graph

        except Exception as e:
            logger.error(f"Knowledge graph build failed with error: {str(e)}")
            self.cache_manager.update_build_status(error_message=str(e))
            raise

    # User interaction methods (unchanged from original)
    def get_chunks_for_editing(self) -> List[TextChunk]:
        """Get text chunks for user editing."""
        return self._get_cached_step_result(BuildSteps.TEXT_CHUNKING, None, list) or []

    def update_chunks(self, chunks: List[TextChunk]) -> None:
        """Update text chunks and invalidate dependent steps."""
        # Save updated chunks
        self.cache_manager.save_step_result(BuildSteps.TEXT_CHUNKING, "user_edited", chunks)

        # Invalidate dependent steps
        self.cache_manager.invalidate_dependent_steps(BuildSteps.TEXT_CHUNKING)

    def get_entities_for_editing(self) -> List[Entity]:
        """Get entities for user editing."""
        return self._get_cached_step_result(BuildSteps.ENTITY_EXTRACTION, None, list) or []

    def update_entities(self, entities: List[Entity]) -> None:
        """Update entities and invalidate dependent steps."""
        self.cache_manager.save_step_result(BuildSteps.ENTITY_EXTRACTION, "user_edited", entities)
        self.cache_manager.invalidate_dependent_steps(BuildSteps.ENTITY_EXTRACTION)

    def get_relations_for_editing(self) -> List[Relation]:
        """Get relations for user editing."""
        return self._get_cached_step_result(BuildSteps.RELATION_EXTRACTION, None, list) or []

    def update_relations(self, relations: List[Relation]) -> None:
        """Update relations and invalidate dependent steps."""
        self.cache_manager.save_step_result(
            BuildSteps.RELATION_EXTRACTION, "user_edited", relations
        )
        self.cache_manager.invalidate_dependent_steps(BuildSteps.RELATION_EXTRACTION)

    def get_clusters_for_editing(self) -> List[Cluster]:
        """Get clusters for user editing."""
        return self._get_cached_step_result(BuildSteps.CLUSTER_FORMATION, None, list) or []

    def update_clusters(self, clusters: List[Cluster]) -> None:
        """Update clusters and invalidate dependent steps."""
        self.cache_manager.save_step_result(BuildSteps.CLUSTER_FORMATION, "user_edited", clusters)
        self.cache_manager.invalidate_dependent_steps(BuildSteps.CLUSTER_FORMATION)

    # Cache and status management methods (unchanged from original)
    def save_step_result(
        self, step_name: str, result: Any, metadata: Optional[Dict] = None
    ) -> None:
        """Save step result to cache."""
        # metadata parameter is for future extensibility
        self.cache_manager.save_step_result(step_name, "manual", result)

    def load_step_result(self, step_name: str, expected_type: Type[T]) -> Optional[T]:
        """Load step result from cache."""
        return self.cache_manager.get_step_result(step_name, "manual", expected_type)

    def has_cached_step(self, step_name: str) -> bool:
        """Check if step result is cached."""
        return self.cache_manager.backend.has(f"{step_name}_manual")

    def clear_cache(self, from_step: Optional[str] = None) -> None:
        """Clear cache from specified step onwards."""
        if from_step is None:
            self.cache_manager.clear_all_cache()
        else:
            self.cache_manager.invalidate_step(from_step)
            self.cache_manager.invalidate_dependent_steps(from_step)

    def get_build_status(self) -> Dict[str, Any]:
        """Get current build status."""
        return self.cache_manager.get_build_status().to_dict()

    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information."""
        cache_info = self.cache_manager.get_cache_info()
        # Add document processing summary
        cache_info["document_processing"] = self.cache_manager.get_document_processing_summary()
        return cache_info

    def get_document_processing_status(
        self, file_path: Optional[Union[str, Path]] = None
    ) -> Dict[str, Any]:
        """Get document processing status information."""
        if file_path:
            # Get status for specific document
            status = self.cache_manager.get_document_status(file_path)
            return status.to_dict() if status else {}

        # Get summary of all document processing
        return self.cache_manager.get_document_processing_summary()

    def clear_document_cache(self, file_path: Optional[Union[str, Path]] = None) -> int:
        """Clear document processing cache."""
        return self.cache_manager.clear_document_cache(file_path)

    def force_reprocess_document(self, file_path: Union[str, Path]) -> bool:
        """Force reprocessing of a specific document by clearing its cache."""
        cleared_count = self.clear_document_cache(file_path)
        return cleared_count > 0

    # Pipeline-specific methods (new functionality)
    def create_custom_pipeline(self, step_config: Dict[str, Any]) -> BuildPipeline:
        """
        Create a custom pipeline with specific step configuration.
        
        Args:
            step_config: Configuration dictionary specifying which steps to include
            
        Returns:
            Configured BuildPipeline
        """
        return self.pipeline_factory.create_custom_pipeline(step_config)

    def create_minimal_pipeline(self) -> BuildPipeline:
        """Create a minimal pipeline with only essential steps."""
        return self.pipeline_factory.create_minimal_pipeline(
            self.text_chunker_handler,
            self.graph_assembler
        )

    def get_pipeline_metrics(self) -> Dict[str, Any]:
        """
        Get metrics from the last executed pipeline.
        
        Returns:
            Pipeline execution metrics
        """
        # This would be enhanced to store and return actual pipeline metrics
        # For now, return basic cache manager metrics
        return {
            "cache_info": self.get_cache_info(),
            "build_status": self.get_build_status()
        }

    # Helper methods (unchanged from original)
    def _create_cluster_algorithm(self) -> ClusterAlgorithm:
        """Create clustering algorithm based on configuration."""
        algorithm_name = self.config.cluster_algorithm

        algorithm_config = {"min_cluster_size": self.config.min_cluster_size}

        if algorithm_name == "hierarchical":
            return HierarchicalClusteringAlgorithm(algorithm_config)

        return CommunityDetectionAlgorithm(algorithm_config)

    def _get_cached_step_result(
        self, step_name: str, input_data: Any, expected_type: Type[T]
    ) -> Optional[T]:
        """Get cached result for a step."""
        return self.cache_manager.get_step_result(step_name, input_data, expected_type)

    async def aclose(self) -> None:
        """Close all async resources."""
        # Close entity extractor if it has aclose method
        if hasattr(self.entity_extractor, "aclose"):
            await self.entity_extractor.aclose()

        # Close relation extractor if it has aclose method
        if hasattr(self.relation_extractor, "aclose"):
            await self.relation_extractor.aclose()

    async def __aenter__(self) -> "KnowledgeGraphBuilderV2":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.aclose()


# For backward compatibility, we can alias the new class
KnowledgeGraphBuilder = KnowledgeGraphBuilderV2