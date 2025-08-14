"""
Knowledge graph assembly handler for knowledge graph builder.
"""

from typing import List

from ...base.clusters import Cluster
from ...base.entities import Entity
from ...base.graph import KnowledgeGraph
from ...base.relations import Relation
from ...base.text import TextChunk
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
        """Assemble final knowledge graph.

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
        )

        cache_input = (entities, relations, clusters, chunks, graph_name, graph_description)

        if use_cache:
            cached_result = self.cache_manager.get_step_result(
                BuildSteps.GRAPH_ASSEMBLY, cache_input, KnowledgeGraph
            )
            if cached_result is not None:
                logger.info("Using cached knowledge graph assembly results")
                return cached_result

        # Create knowledge graph
        kg_name = graph_name or "Generated Knowledge Graph"
        kg_description = graph_description or "Knowledge graph generated from documents"
        logger.debug(f"Creating knowledge graph '{kg_name}' with description: {kg_description}")

        kg = KnowledgeGraph(
            name=kg_name,
            description=kg_description,
        )

        # Add all components
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

        logger.info(
            f"Knowledge graph assembly completed - Graph: '{kg.name}', "
            f"Total components: {len(kg.entities)} entities, {len(kg.relations)} relations, "
            f"{len(kg.clusters)} clusters, {len(kg.text_chunks)} chunks"
        )

        if use_cache:
            self.cache_manager.save_step_result(BuildSteps.GRAPH_ASSEMBLY, cache_input, kg)

        return kg
