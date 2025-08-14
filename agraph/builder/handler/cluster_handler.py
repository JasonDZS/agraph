"""
Cluster formation handler for knowledge graph builder.
"""

from typing import List

from ...base.clusters import Cluster
from ...base.entities import Entity
from ...base.relations import Relation
from ...builder.cache import CacheManager
from ...builder.clustering import ClusterAlgorithm
from ...config import BuildSteps
from ...logger import logger


class ClusterHandler:
    """Handles cluster formation with caching support."""

    def __init__(self, cache_manager: CacheManager, cluster_algorithm: ClusterAlgorithm):
        """Initialize cluster handler.

        Args:
            cache_manager: Cache manager instance
            cluster_algorithm: Clustering algorithm instance
        """
        self.cache_manager = cache_manager
        self.cluster_algorithm = cluster_algorithm

    def form_clusters(
        self, entities: List[Entity], relations: List[Relation], use_cache: bool = True
    ) -> List[Cluster]:
        """Form clusters from entities and relations.

        Args:
            entities: List of entities
            relations: List of relations
            use_cache: Whether to use caching

        Returns:
            List of formed clusters
        """
        logger.info(
            f"Forming clusters from {len(entities)} entities and {len(relations)} "
            f"relations using {type(self.cluster_algorithm).__name__}"
        )

        cache_input = (entities, relations)

        if use_cache:
            cached_result = self.cache_manager.get_step_result(
                BuildSteps.CLUSTER_FORMATION, cache_input, list
            )
            if cached_result is not None:
                logger.info(
                    f"Using cached cluster formation results for {len(entities)} entities "
                    f"and {len(relations)} relations"
                )
                return cached_result

        clusters = self.cluster_algorithm.cluster(entities, relations)
        logger.info(f"Cluster formation completed - created {len(clusters)} clusters")

        if use_cache:
            self.cache_manager.save_step_result(BuildSteps.CLUSTER_FORMATION, cache_input, clusters)

        return clusters
