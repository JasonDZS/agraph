"""
Knowledge graph assembly handler for knowledge graph builder.
"""

from typing import Any, Dict, List, Optional

from ...base.graphs.optimized import KnowledgeGraph
from ...base.managers.interfaces import (
    ClusterManager,
    EntityManager,
    RelationManager,
    TextChunkManager,
)
from ...base.models.clusters import Cluster
from ...base.models.entities import Entity
from ...base.models.relations import Relation

# Import unified architecture components
from ...base.models.text import TextChunk
from ...builder.cache import CacheManager
from ...config import BuildSteps
from ...logger import logger


class GraphAssembler:
    """Handles knowledge graph assembly with caching support."""

    def __init__(self, cache_manager: CacheManager):
        """Initialize graph assembler.

        Args:
            cache_manager: Cache manager instance
        """
        self.cache_manager = cache_manager
        self._unified_managers: Optional[Dict[str, Any]] = None
        self._use_unified_architecture = False

    def set_unified_managers(self, managers: Dict[str, Any]) -> None:
        """Set unified managers for enhanced functionality.

        Args:
            managers: Dictionary containing unified managers
        """
        self._unified_managers = managers
        self._use_unified_architecture = True
        logger.info("GraphAssembler configured to use unified architecture")

    @property
    def entity_manager(self) -> Optional[EntityManager]:
        """Get entity manager from unified architecture."""
        if self._unified_managers:
            return self._unified_managers.get("entity_manager")
        return None

    @property
    def relation_manager(self) -> Optional[RelationManager]:
        """Get relation manager from unified architecture."""
        if self._unified_managers:
            return self._unified_managers.get("relation_manager")
        return None

    @property
    def cluster_manager(self) -> Optional[ClusterManager]:
        """Get cluster manager from unified architecture."""
        if self._unified_managers:
            return self._unified_managers.get("cluster_manager")
        return None

    @property
    def text_chunk_manager(self) -> Optional[TextChunkManager]:
        """Get text chunk manager from unified architecture."""
        if self._unified_managers:
            return self._unified_managers.get("text_chunk_manager")
        return None

    def assemble_knowledge_graph(
        self,
        entities: List[Entity],
        relations: List[Relation],
        clusters: List[Cluster],
        chunks: List[TextChunk],
        graph_name: str,
        graph_description: str,
        use_cache: bool = True,
    ) -> KnowledgeGraph:
        """Assemble final knowledge graph with unified architecture support.

        Args:
            entities: List of entities
            relations: List of relations
            clusters: List of clusters
            chunks: List of text chunks
            graph_name: Graph name
            graph_description: Graph description
            use_cache: Whether to use caching

        Returns:
            Assembled knowledge graph
        """
        logger.info(
            f"Assembling knowledge graph with {len(entities)} entities, {len(relations)} relations, "
            f"{len(clusters)} clusters, and {len(chunks)} chunks"
            f" (unified architecture: {self._use_unified_architecture})"
        )

        cache_input = (entities, relations, clusters, chunks, graph_name, graph_description)

        if use_cache:
            cached_result = self.cache_manager.get_step_result(
                BuildSteps.GRAPH_ASSEMBLY, cache_input, KnowledgeGraph
            )
            if cached_result is not None:
                logger.info("Using cached knowledge graph assembly results")
                return cached_result

        # Create optimized knowledge graph for better performance
        kg_name = graph_name or "Generated Knowledge Graph"
        kg_description = graph_description or "Knowledge graph generated from documents"
        logger.debug(
            f"Creating optimized knowledge graph '{kg_name}' with description: {kg_description}"
        )

        kg = KnowledgeGraph(
            name=kg_name,
            description=kg_description,
        )

        # Use unified architecture if available for enhanced performance and validation
        if self._use_unified_architecture and self._unified_managers:
            logger.debug("Using unified architecture for graph assembly")
            try:
                # Add entities using unified manager for validation and optimization
                if self.entity_manager and entities:
                    logger.debug(f"Adding {len(entities)} entities via unified manager")
                    for entity in entities:
                        # Use unified manager's add method for enhanced validation
                        result = self.entity_manager.add(entity)
                        if result.success:
                            kg.add_entity(result.unwrap())
                        else:
                            logger.warning(
                                f"Failed to add entity {entity.name}: {result.error_message}"
                            )
                            # Fall back to direct addition
                            kg.add_entity(entity)
                else:
                    # Fallback to direct addition
                    for entity in entities:
                        kg.add_entity(entity)

                # Add relations using unified manager
                if self.relation_manager and relations:
                    logger.debug(f"Adding {len(relations)} relations via unified manager")
                    for relation in relations:
                        relation_result = self.relation_manager.add(relation)
                        if relation_result.success:
                            kg.add_relation(relation_result.unwrap())
                        else:
                            logger.warning(
                                f"Failed to add relation {relation.id}: {relation_result.error_message}"
                            )
                            # Fall back to direct addition
                            kg.add_relation(relation)
                else:
                    # Fallback to direct addition
                    for relation in relations:
                        kg.add_relation(relation)

                # Add clusters using unified manager
                if self.cluster_manager and clusters:
                    logger.debug(f"Adding {len(clusters)} clusters via unified manager")
                    for cluster in clusters:
                        cluster_result = self.cluster_manager.add(cluster)
                        if cluster_result.success:
                            kg.add_cluster(cluster_result.unwrap())
                        else:
                            logger.warning(
                                f"Failed to add cluster {cluster.id}: {cluster_result.error_message}"
                            )
                            # Fall back to direct addition
                            kg.add_cluster(cluster)
                else:
                    # Fallback to direct addition
                    for cluster in clusters:
                        kg.add_cluster(cluster)

                # Add text chunks using unified manager
                if self.text_chunk_manager and chunks:
                    logger.debug(f"Adding {len(chunks)} text chunks via unified manager")
                    for chunk in chunks:
                        chunk_result = self.text_chunk_manager.add(chunk)
                        if chunk_result.success:
                            kg.add_text_chunk(chunk_result.unwrap())
                        else:
                            logger.warning(
                                f"Failed to add text chunk {chunk.id}: {chunk_result.error_message}"
                            )
                            # Fall back to direct addition
                            kg.add_text_chunk(chunk)
                else:
                    # Fallback to direct addition
                    for chunk in chunks:
                        kg.add_text_chunk(chunk)

            except Exception as e:
                logger.error_message(f"Error during unified architecture assembly: {e}")
                logger.info("Falling back to traditional assembly method")
                # Fall back to traditional method
                self._traditional_assembly(kg, entities, relations, clusters, chunks)
        else:
            # Traditional assembly method
            logger.debug("Using traditional assembly method")
            self._traditional_assembly(kg, entities, relations, clusters, chunks)

        logger.info(
            f"Knowledge graph assembly completed - Graph: '{kg.name}', "
            f"Total components: {len(kg.entities)} entities, {len(kg.relations)} relations, "
            f"{len(kg.clusters)} clusters, {len(kg.text_chunks)} chunks"
        )

        if use_cache:
            self.cache_manager.save_step_result(BuildSteps.GRAPH_ASSEMBLY, cache_input, kg)

        return kg

    def _traditional_assembly(
        self,
        kg: KnowledgeGraph,
        entities: List[Entity],
        relations: List[Relation],
        clusters: List[Cluster],
        chunks: List[TextChunk],
    ) -> None:
        """Traditional assembly method for backward compatibility."""
        # Add all components directly
        logger.debug(f"Adding {len(entities)} entities to knowledge graph")
        for entity in entities:
            kg.add_entity(entity)

        logger.debug(f"Adding {len(relations)} relations to knowledge graph")
        for relation in relations:
            kg.add_relation(relation)

        logger.debug(f"Adding {len(clusters)} clusters to knowledge graph")
        for cluster in clusters:
            kg.add_cluster(cluster)

        logger.debug(f"Adding {len(chunks)} text chunks to knowledge graph")
        for chunk in chunks:
            kg.add_text_chunk(chunk)
