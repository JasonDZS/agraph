"""
Cluster formation step implementation.
"""

from typing import Any, Dict, List

from ...base.models.clusters import Cluster
from ...base.models.entities import Entity
from ...base.models.relations import Relation
from ...config import BuildSteps
from ..handler.cluster_handler import ClusterHandler
from .base import BuildStep, StepResult
from .context import BuildContext


class ClusterFormationStep(BuildStep):
    """Step for forming clusters from entities and relations."""

    def __init__(self, cluster_handler: ClusterHandler, cache_manager: Any):
        """
        Initialize cluster formation step.

        Args:
            cluster_handler: Handler for cluster formation operations
            cache_manager: Cache manager instance
        """
        super().__init__(BuildSteps.CLUSTER_FORMATION, cache_manager)
        self.cluster_handler = cluster_handler

    async def _execute_step(self, context: BuildContext) -> StepResult[List[Cluster]]:
        """
        Execute cluster formation logic.

        Args:
            context: Build context containing entities and relations for clustering

        Returns:
            StepResult containing list of formed clusters
        """
        # pylint: disable=too-many-return-statements
        try:
            # Get entities and relations from context
            entities = context.entities
            relations = context.relations

            if not entities:
                return StepResult.failure_result("No entities available for cluster formation")

            if not relations:
                return StepResult.failure_result("No relations available for cluster formation")

            # Validate inputs
            if not isinstance(entities, list):
                return StepResult.failure_result("Invalid entities type: expected list")

            if not isinstance(relations, list):
                return StepResult.failure_result("Invalid relations type: expected list")

            for i, entity in enumerate(entities):
                if not isinstance(entity, Entity):
                    return StepResult.failure_result(
                        f"Invalid entity at index {i}: expected Entity, got {type(entity)}"
                    )

            for i, relation in enumerate(relations):
                if not isinstance(relation, Relation):
                    return StepResult.failure_result(
                        f"Invalid relation at index {i}: expected Relation, got {type(relation)}"
                    )

            # Execute cluster formation (synchronous operation)
            clusters = self.cluster_handler.form_clusters(entities, relations, context.use_cache)

            if not isinstance(clusters, list):
                return StepResult.failure_result("Cluster formation returned invalid result type")

            # Validate clusters
            for i, cluster in enumerate(clusters):
                if not isinstance(cluster, Cluster):
                    return StepResult.failure_result(
                        f"Invalid cluster at index {i}: expected Cluster, got {type(cluster)}"
                    )

            # Calculate cluster metrics
            total_entities_in_clusters = sum(len(cluster.entities) for cluster in clusters)
            cluster_sizes = [len(cluster.entities) for cluster in clusters]
            avg_cluster_size = sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0

            # Check for entities not in any cluster
            entities_in_clusters = set()
            for cluster in clusters:
                entities_in_clusters.update(cluster.entities)

            all_entity_ids = {entity.id for entity in entities}
            unclustered_entities = all_entity_ids - entities_in_clusters

            # Cluster type distribution
            cluster_types = [cluster.cluster_type for cluster in clusters if cluster.cluster_type]
            cluster_type_counts: Dict[str, int] = {}
            for cluster_type in cluster_types:
                cluster_type_counts[str(cluster_type)] = cluster_type_counts.get(str(cluster_type), 0) + 1

            return StepResult.success_result(
                clusters,
                metadata={
                    "input_entities": len(entities),
                    "input_relations": len(relations),
                    "formed_clusters": len(clusters),
                    "total_entities_clustered": total_entities_in_clusters,
                    "unclustered_entities": len(unclustered_entities),
                    "clustering_coverage": (total_entities_in_clusters / len(entities) if entities else 0),
                    "average_cluster_size": avg_cluster_size,
                    "cluster_size_distribution": {
                        "small (1-5)": len([s for s in cluster_sizes if 1 <= s <= 5]),
                        "medium (6-15)": len([s for s in cluster_sizes if 6 <= s <= 15]),
                        "large (>15)": len([s for s in cluster_sizes if s > 15]),
                    },
                    "cluster_type_distribution": cluster_type_counts,
                    "largest_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
                    "smallest_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
                },
            )

        except Exception as e:
            return StepResult.failure_result(f"Cluster formation failed: {str(e)}")

    def _get_cache_input_data(self, context: BuildContext) -> Any:
        """Get input data for cache key generation."""
        # Use tuple of entities and relations for cache key
        return (context.entities, context.relations)

    def _get_expected_result_type(self) -> type:
        """Get expected result type for cache deserialization."""
        return list
