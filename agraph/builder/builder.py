"""
Main KnowledgeGraph Builder class.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union

from ..base.clusters import Cluster
from ..base.dao import MemoryDataAccessLayer
from ..base.entities import Entity
from ..base.graph import KnowledgeGraph

# Import unified architecture components
from ..base.manager_factory import create_managers
from ..base.optimized_graph import OptimizedKnowledgeGraph
from ..base.relations import Relation
from ..base.text import TextChunk
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

T = TypeVar("T")


class KnowledgeGraphBuilder:
    """Main builder class for constructing knowledge graphs from documents."""

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
        """Initialize KnowledgeGraph Builder.

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

        # Initialize components
        self.chunker = chunker or TokenChunker(
            chunk_size=self.config.chunk_size, chunk_overlap=self.config.chunk_overlap
        )

        self.entity_extractor = entity_extractor or LLMEntityExtractor(
            {
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "min_confidence": self.config.entity_confidence_threshold,
            }
        )

        self.relation_extractor = relation_extractor or LLMRelationExtractor(
            {
                "llm_provider": self.config.llm_provider,
                "llm_model": self.config.llm_model,
                "min_confidence": self.config.relation_confidence_threshold,
            }
        )

        self.cluster_algorithm = cluster_algorithm or self._create_cluster_algorithm()

        # Initialize processor factory
        self.processor_factory = DocumentProcessorFactory()

        # Initialize handlers
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

        logger.info(
            f"KnowledgeGraphBuilder initialized - "
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
        """Build knowledge graph from document files.

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
            # Reset build status if starting fresh
            if from_step is None:
                logger.info("Resetting build status for fresh build")
                self.cache_manager.reset_build_status()

            # Step 1: Process documents
            if self._should_execute_step(BuildSteps.DOCUMENT_PROCESSING, from_step):
                logger.info(f"Step 1: Processing {len(documents)} documents")
                self.cache_manager.update_build_status(current_step=BuildSteps.DOCUMENT_PROCESSING)
                texts = self.document_processor.process_documents(documents, use_cache)
                # Save step result for compatibility
                if use_cache:
                    self.cache_manager.save_step_result(
                        BuildSteps.DOCUMENT_PROCESSING, documents, texts
                    )
                logger.info(f"Document processing completed - extracted {len(texts)} texts")
                self.cache_manager.update_build_status(
                    completed_step=BuildSteps.DOCUMENT_PROCESSING
                )
            else:
                logger.info("Step 1: Using cached document processing results")
                texts = (
                    self._get_cached_step_result(BuildSteps.DOCUMENT_PROCESSING, documents, list)
                    or []
                )

            # Step 2: Chunk texts
            if self._should_execute_step(BuildSteps.TEXT_CHUNKING, from_step):
                logger.info(f"Step 2: Chunking {len(texts)} texts")
                self.cache_manager.update_build_status(current_step=BuildSteps.TEXT_CHUNKING)
                chunks = self.text_chunker_handler.chunk_texts(texts, use_cache, documents)
                logger.info(f"Text chunking completed - created {len(chunks)} chunks")
                self.cache_manager.update_build_status(completed_step=BuildSteps.TEXT_CHUNKING)
            else:
                logger.info("Step 2: Using cached text chunking results")
                chunks = self._get_cached_step_result(BuildSteps.TEXT_CHUNKING, texts, list) or []

            # Step 3: Extract entities (if knowledge graph is enabled)
            if self.enable_knowledge_graph:
                if self._should_execute_step(BuildSteps.ENTITY_EXTRACTION, from_step):
                    logger.info(f"Step 3: Extracting entities from {len(chunks)} chunks")
                    self.cache_manager.update_build_status(
                        current_step=BuildSteps.ENTITY_EXTRACTION
                    )
                    entities = await self.entity_handler.extract_entities_from_chunks(
                        chunks, use_cache
                    )
                    logger.info(f"Entity extraction completed - found {len(entities)} entities")
                    self.cache_manager.update_build_status(
                        completed_step=BuildSteps.ENTITY_EXTRACTION
                    )
                else:
                    logger.info("Step 3: Using cached entity extraction results")
                    entities = (
                        self._get_cached_step_result(BuildSteps.ENTITY_EXTRACTION, chunks, list)
                        or []
                    )
            else:
                logger.info("Step 3: Skipping entity extraction (knowledge graph disabled)")
                entities = []

            # Step 4: Extract relations (if knowledge graph is enabled)
            if self.enable_knowledge_graph:
                if self._should_execute_step(BuildSteps.RELATION_EXTRACTION, from_step):
                    logger.info(
                        f"Step 4: Extracting relations from {len(chunks)} chunks and {len(entities)} entities"
                    )
                    self.cache_manager.update_build_status(
                        current_step=BuildSteps.RELATION_EXTRACTION
                    )
                    relations = await self.relation_handler.extract_relations_from_chunks(
                        chunks, entities, use_cache
                    )
                    logger.info(f"Relation extraction completed - found {len(relations)} relations")
                    self.cache_manager.update_build_status(
                        completed_step=BuildSteps.RELATION_EXTRACTION
                    )
                else:
                    logger.info("Step 4: Using cached relation extraction results")
                    relations = (
                        self._get_cached_step_result(
                            BuildSteps.RELATION_EXTRACTION, (chunks, entities), list
                        )
                        or []
                    )
            else:
                logger.info("Step 4: Skipping relation extraction (knowledge graph disabled)")
                relations = []

            # Step 5: Form clusters (if knowledge graph is enabled)
            if self.enable_knowledge_graph:
                if self._should_execute_step(BuildSteps.CLUSTER_FORMATION, from_step):
                    logger.info(
                        f"Step 5: Forming clusters from {len(entities)} entities and {len(relations)} relations"
                    )
                    self.cache_manager.update_build_status(
                        current_step=BuildSteps.CLUSTER_FORMATION
                    )
                    clusters = self.cluster_handler.form_clusters(entities, relations, use_cache)
                    logger.info(f"Cluster formation completed - created {len(clusters)} clusters")
                    self.cache_manager.update_build_status(
                        completed_step=BuildSteps.CLUSTER_FORMATION
                    )
                else:
                    logger.info("Step 5: Using cached cluster formation results")
                    clusters = (
                        self._get_cached_step_result(
                            BuildSteps.CLUSTER_FORMATION, (entities, relations), list
                        )
                        or []
                    )
            else:
                logger.info("Step 5: Skipping cluster formation (knowledge graph disabled)")
                clusters = []

            # Step 6: Assemble knowledge graph
            if self._should_execute_step(BuildSteps.GRAPH_ASSEMBLY, from_step):
                logger.info(
                    f"Step 6: Assembling knowledge graph with {len(entities)} entities, "
                    f"{len(relations)} relations, {len(clusters)} clusters"
                )
                self.cache_manager.update_build_status(current_step=BuildSteps.GRAPH_ASSEMBLY)
                kg = self.graph_assembler.assemble_knowledge_graph(
                    entities, relations, clusters, chunks, graph_name, graph_description, use_cache
                )
                logger.info(f"Knowledge graph assembly completed - Graph: '{kg.name}'")
                self.cache_manager.update_build_status(completed_step=BuildSteps.GRAPH_ASSEMBLY)
            else:
                logger.info("Step 6: Using cached knowledge graph assembly results")
                cached_kg = self._get_cached_step_result(
                    BuildSteps.GRAPH_ASSEMBLY,
                    (entities, relations, clusters, chunks, graph_name, graph_description),
                    OptimizedKnowledgeGraph,
                )
                if cached_kg is None:
                    # If cache result is None, fall back to assembling new graph
                    logger.warning("Cached knowledge graph not found, assembling new graph")
                    kg = self.graph_assembler.assemble_knowledge_graph(
                        entities,
                        relations,
                        clusters,
                        chunks,
                        graph_name,
                        graph_description,
                        use_cache,
                    )
                else:
                    kg = cached_kg

            logger.info(
                f"Knowledge graph build completed successfully - "
                f"Name: '{kg.name}', Entities: {len(kg.entities)}, "
                f"Relations: {len(kg.relations)}, Clusters: {len(kg.clusters)}"
            )
            return kg

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
        """Build knowledge graph from text strings.

        Args:
            texts: List of text strings
            graph_name: Name for the knowledge graph
            graph_description: Description for the knowledge graph
            use_cache: Whether to use caching
            from_step: Step to start from (for resuming)

        Returns:
            Constructed knowledge graph
        """
        # Skip document processing step and start from chunking
        actual_from_step = (
            from_step if from_step != BuildSteps.DOCUMENT_PROCESSING else BuildSteps.TEXT_CHUNKING
        )

        logger.info(
            f"Starting knowledge graph build from {len(texts)} text strings - "
            f"Graph Name: '{graph_name}', Use Cache: {use_cache}, From Step: {from_step}"
        )

        try:
            # Reset build status if starting fresh
            if from_step is None:
                logger.info("Resetting build status for fresh build (skipping document processing)")
                self.cache_manager.reset_build_status()
                self.cache_manager.update_build_status(
                    completed_step=BuildSteps.DOCUMENT_PROCESSING
                )

            # Step 2: Chunk texts
            if self._should_execute_step(BuildSteps.TEXT_CHUNKING, actual_from_step):
                logger.info(f"Step 2: Chunking {len(texts)} texts")
                self.cache_manager.update_build_status(current_step=BuildSteps.TEXT_CHUNKING)
                chunks = self.text_chunker_handler.chunk_texts(texts, use_cache)
                logger.info(f"Text chunking completed - created {len(chunks)} chunks")
                self.cache_manager.update_build_status(completed_step=BuildSteps.TEXT_CHUNKING)
            else:
                logger.info("Step 2: Using cached text chunking results")
                chunks = self._get_cached_step_result(BuildSteps.TEXT_CHUNKING, texts, list) or []

            # Continue with remaining steps (same as build_from_documents)
            # Step 3: Extract entities (if knowledge graph is enabled)
            if self.enable_knowledge_graph:
                if self._should_execute_step(BuildSteps.ENTITY_EXTRACTION, actual_from_step):
                    logger.info(f"Step 3: Extracting entities from {len(chunks)} chunks")
                    self.cache_manager.update_build_status(
                        current_step=BuildSteps.ENTITY_EXTRACTION
                    )
                    entities = await self.entity_handler.extract_entities_from_chunks(
                        chunks, use_cache
                    )
                    logger.info(f"Entity extraction completed - found {len(entities)} entities")
                    self.cache_manager.update_build_status(
                        completed_step=BuildSteps.ENTITY_EXTRACTION
                    )
                else:
                    logger.info("Step 3: Using cached entity extraction results")
                    entities = (
                        self._get_cached_step_result(BuildSteps.ENTITY_EXTRACTION, chunks, list)
                        or []
                    )
            else:
                logger.info("Step 3: Skipping entity extraction (knowledge graph disabled)")
                entities = []

            # Step 4: Extract relations (if knowledge graph is enabled)
            if self.enable_knowledge_graph:
                if self._should_execute_step(BuildSteps.RELATION_EXTRACTION, actual_from_step):
                    logger.info(
                        f"Step 4: Extracting relations from {len(chunks)} chunks and {len(entities)} entities"
                    )
                    self.cache_manager.update_build_status(
                        current_step=BuildSteps.RELATION_EXTRACTION
                    )
                    relations = await self.relation_handler.extract_relations_from_chunks(
                        chunks, entities, use_cache
                    )
                    logger.info(f"Relation extraction completed - found {len(relations)} relations")
                    self.cache_manager.update_build_status(
                        completed_step=BuildSteps.RELATION_EXTRACTION
                    )
                else:
                    logger.info("Step 4: Using cached relation extraction results")
                    relations = (
                        self._get_cached_step_result(
                            BuildSteps.RELATION_EXTRACTION, (chunks, entities), list
                        )
                        or []
                    )
            else:
                logger.info("Step 4: Skipping relation extraction (knowledge graph disabled)")
                relations = []

            # Step 5: Form clusters (if knowledge graph is enabled)
            if self.enable_knowledge_graph:
                if self._should_execute_step(BuildSteps.CLUSTER_FORMATION, actual_from_step):
                    logger.info(
                        f"Step 5: Forming clusters from {len(entities)} entities and {len(relations)} relations"
                    )
                    self.cache_manager.update_build_status(
                        current_step=BuildSteps.CLUSTER_FORMATION
                    )
                    clusters = self.cluster_handler.form_clusters(entities, relations, use_cache)
                    logger.info(f"Cluster formation completed - created {len(clusters)} clusters")
                    self.cache_manager.update_build_status(
                        completed_step=BuildSteps.CLUSTER_FORMATION
                    )
                else:
                    logger.info("Step 5: Using cached cluster formation results")
                    cached_clusters = self._get_cached_step_result(
                        BuildSteps.CLUSTER_FORMATION, (entities, relations), list
                    )
                    clusters = cached_clusters or []
            else:
                logger.info("Step 5: Skipping cluster formation (knowledge graph disabled)")
                clusters = []

            # Step 6: Assemble knowledge graph
            if self._should_execute_step(BuildSteps.GRAPH_ASSEMBLY, actual_from_step):
                logger.info(
                    f"Step 6: Assembling knowledge graph with {len(entities)} entities, "
                    f"{len(relations)} relations, {len(clusters)} clusters"
                )
                self.cache_manager.update_build_status(current_step=BuildSteps.GRAPH_ASSEMBLY)
                kg = self.graph_assembler.assemble_knowledge_graph(
                    entities, relations, clusters, chunks, graph_name, graph_description, use_cache
                )
                logger.info(f"Knowledge graph assembly completed - Graph: '{kg.name}'")
                self.cache_manager.update_build_status(completed_step=BuildSteps.GRAPH_ASSEMBLY)
            else:
                logger.info("Step 6: Using cached knowledge graph assembly results")
                cached_kg = self._get_cached_step_result(
                    BuildSteps.GRAPH_ASSEMBLY,
                    (entities, relations, clusters, chunks, graph_name, graph_description),
                    OptimizedKnowledgeGraph,
                )
                if cached_kg is None:
                    # If cache result is None, fall back to assembling new graph
                    logger.warning("Cached knowledge graph not found, assembling new graph")
                    kg = self.graph_assembler.assemble_knowledge_graph(
                        entities,
                        relations,
                        clusters,
                        chunks,
                        graph_name,
                        graph_description,
                        use_cache,
                    )
                else:
                    kg = cached_kg

            logger.info(
                f"Knowledge graph build completed successfully - "
                f"Name: '{kg.name}', Entities: {len(kg.entities)}, "
                f"Relations: {len(kg.relations)}, Clusters: {len(kg.clusters)}"
            )
            return kg

        except Exception as e:
            logger.error(f"Knowledge graph build failed with error: {str(e)}")
            self.cache_manager.update_build_status(error_message=str(e))
            raise

    # User interaction methods
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

    # Cache and status management methods
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

    # Helper methods
    def _create_cluster_algorithm(self) -> ClusterAlgorithm:
        """Create clustering algorithm based on configuration."""
        algorithm_name = self.config.cluster_algorithm

        algorithm_config = {"min_cluster_size": self.config.min_cluster_size}

        if algorithm_name == "hierarchical":
            return HierarchicalClusteringAlgorithm(algorithm_config)

        return CommunityDetectionAlgorithm(algorithm_config)

    def _should_execute_step(self, step_name: str, from_step: Optional[str]) -> bool:
        """Check if step should be executed based on from_step parameter."""
        if from_step is None:
            return True

        from_step_index = BuildSteps.get_step_index(from_step)
        current_step_index = BuildSteps.get_step_index(step_name)

        return current_step_index >= from_step_index

    def _get_cached_step_result(
        self, step_name: str, input_data: Any, expected_type: Type[T]
    ) -> Optional[T]:
        """Get cached result for a step."""
        return self.cache_manager.get_step_result(step_name, input_data, expected_type)
