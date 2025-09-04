"""
Community detection clustering algorithm.
"""

from typing import Any, Dict, List, Optional, Set

from ...base.core.types import ClusterType
from ...base.models.clusters import Cluster
from ...base.models.entities import Entity
from ...base.models.relations import Relation
from .base import ClusterAlgorithm


class CommunityDetectionAlgorithm(ClusterAlgorithm):
    """Community detection clustering algorithm using a simplified Louvain method."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize community detection algorithm.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.resolution = self.config.get("resolution", 1.0)
        self.max_iterations = self.config.get("max_iterations", 100)
        self.min_improvement: float = self.config.get("min_improvement", 1e-6)

    def cluster(self, entities: List[Entity], relations: List[Relation]) -> List[Cluster]:
        """Perform community detection clustering.

        Args:
            entities: List of entities to cluster
            relations: List of relations between entities

        Returns:
            List of detected communities as clusters
        """
        if len(entities) < self.min_cluster_size:
            return []

        # Build adjacency matrix
        adjacency = self.build_adjacency_matrix(entities, relations)
        entities_map = {e.id: e for e in entities}

        # Find communities using simplified Louvain method
        communities = self._detect_communities(adjacency)

        # Convert communities to clusters
        clusters = []
        for community in communities:
            cluster = self.create_cluster(community, entities_map, ClusterType.TOPIC)
            if cluster:
                clusters.append(cluster)

        return self.post_process_clusters(clusters)

    def _detect_communities(self, adjacency: Dict[str, Dict[str, float]]) -> List[Set[str]]:
        """Detect communities using simplified Louvain method.

        Args:
            adjacency: Adjacency matrix

        Returns:
            List of communities (sets of entity IDs)
        """
        nodes = list(adjacency.keys())

        if not nodes:
            return []

        # Initialize each node as its own community
        communities = {node: {node} for node in nodes}

        # Calculate total weight
        total_weight = self._calculate_total_weight(adjacency)

        if total_weight == 0:
            # No connections, return single-node communities that meet min size
            return [
                community
                for community in communities.values()
                if len(community) >= self.min_cluster_size
            ]

        # Iteratively improve modularity
        improved = True
        iteration = 0

        while improved and iteration < self.max_iterations:
            improved = False
            iteration += 1

            for node in nodes:
                current_community = None
                for comm_id, community in communities.items():
                    if node in community:
                        current_community = comm_id
                        break

                if not current_community:
                    continue

                # Try moving node to neighboring communities
                best_community = current_community
                best_gain = 0.0

                # Get neighbors
                neighbors = self.get_entity_neighbors(node, adjacency)

                # Consider neighboring communities
                neighbor_communities = set()
                for neighbor in neighbors:
                    for comm_id, community in communities.items():
                        if neighbor in community:
                            neighbor_communities.add(comm_id)

                for target_comm in neighbor_communities:
                    if target_comm == current_community:
                        continue

                    # Calculate modularity gain
                    gain = self._calculate_modularity_gain(
                        node, current_community, target_comm, communities, adjacency, total_weight
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_community = target_comm

                # Move node if improvement is significant
                if best_gain > self.min_improvement and best_community != current_community:
                    communities[current_community].remove(node)
                    communities[best_community].add(node)
                    improved = True

                    # Remove empty communities
                    if not communities[current_community]:
                        del communities[current_community]

        # Return non-empty communities
        return [
            community
            for community in communities.values()
            if len(community) >= self.min_cluster_size
        ]

    def _calculate_total_weight(self, adjacency: Dict[str, Dict[str, float]]) -> float:
        """Calculate total weight of all edges.

        Args:
            adjacency: Adjacency matrix

        Returns:
            Total weight
        """
        total_weight = 0.0
        for node, neighbors in adjacency.items():
            for neighbor, weight in neighbors.items():
                if node < neighbor:  # Avoid double counting
                    total_weight += weight

        return total_weight

    def _calculate_modularity_gain(
        self,
        node: str,
        current_comm: str,
        target_comm: str,
        communities: Dict[str, Set[str]],
        adjacency: Dict[str, Dict[str, float]],
        total_weight: float,
    ) -> float:
        """Calculate modularity gain for moving a node.

        Args:
            node: Node to move
            current_comm: Current community ID
            target_comm: Target community ID
            communities: Current community assignment
            adjacency: Adjacency matrix
            total_weight: Total weight of all edges

        Returns:
            Modularity gain
        """
        if total_weight == 0:
            return 0.0

        # Calculate degree of node
        # node_degree = sum(adjacency.get(node, {}).values())

        # Calculate connections to current community (excluding self)
        current_community_nodes = communities[current_comm] - {node}
        connections_to_current = sum(
            adjacency.get(node, {}).get(other_node, 0) for other_node in current_community_nodes
        )

        # Calculate connections to target community
        target_community_nodes = communities[target_comm]
        connections_to_target = sum(
            adjacency.get(node, {}).get(other_node, 0) for other_node in target_community_nodes
        )

        # Simplified modularity calculation
        # This is an approximation of the actual Louvain modularity formula
        gain = (connections_to_target - connections_to_current) / total_weight

        return gain
