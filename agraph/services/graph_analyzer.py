"""
Graph Analysis Service

This service provides comprehensive statistical analysis and metrics computation
for knowledge graphs. It implements various graph analysis algorithms to provide
insights into graph structure, connectivity, and quality.

Key Features:
- Comprehensive graph statistics (nodes, edges, types)
- Connectivity analysis and component detection
- Node degree distribution analysis
- Graph density and centrality metrics
- Entity importance scoring
"""

from typing import Any, Dict, List, Set

from ..graph import KnowledgeGraph
from ..logger import logger
from ..types import EntityType, RelationType


class GraphAnalyzer:
    """
    Service for comprehensive graph statistical analysis and metrics computation.

    This service provides various analytical methods to understand graph structure,
    connectivity patterns, and quality metrics. It supports both basic statistics
    and advanced graph analysis algorithms.

    The analyzer can compute:
    - Basic counts and distributions
    - Connectivity metrics and component analysis
    - Node centrality and importance scores
    - Graph quality indicators
    """

    def __init__(self) -> None:
        """Initialize graph analyzer"""
        pass

    def get_statistics(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Generate comprehensive statistical analysis of the knowledge graph.

        Computes various metrics including entity/relation counts, type distributions,
        connectivity analysis, and degree statistics to provide a complete overview
        of the graph structure and characteristics.

        Args:
            graph: Knowledge graph instance to analyze

        Returns:
            Dict[str, Any]: Comprehensive statistics including:
                - total_entities: Total number of entities
                - total_relations: Total number of relations
                - entity_types: Count of entities by type
                - relation_types: Count of relations by type
                - connectivity: Component analysis and connectivity metrics
                - degree_statistics: Node degree distribution statistics
                - created_at: Graph creation timestamp
                - updated_at: Graph last update timestamp

        Note:
            Returns empty dict if analysis fails due to errors.
        """
        try:
            entity_type_counts = self._count_entities_by_type(graph)
            relation_type_counts = self._count_relations_by_type(graph)
            connectivity_stats = self._analyze_connectivity(graph)
            degree_stats = self._analyze_node_degrees(graph)

            return {
                "total_entities": len(graph.entities),
                "total_relations": len(graph.relations),
                "entity_types": entity_type_counts,
                "relation_types": relation_type_counts,
                "connectivity": connectivity_stats,
                "degree_statistics": degree_stats,
                "created_at": graph.created_at.isoformat(),
                "updated_at": graph.updated_at.isoformat(),
            }
        except Exception as e:
            logger.error(f"Error computing graph statistics: {e}")
            return {}

    def _count_entities_by_type(self, graph: KnowledgeGraph) -> Dict[str, int]:
        """
        Count entities grouped by their entity type.

        Args:
            graph: Knowledge graph to analyze

        Returns:
            Dict[str, int]: Mapping of entity type names to their counts.
                Only includes types that have at least one entity.
        """
        counts = {}
        for entity_type in EntityType:
            count = len(graph.get_entities_by_type(entity_type))
            if count > 0:
                counts[entity_type.value] = count
        return counts

    def _count_relations_by_type(self, graph: KnowledgeGraph) -> Dict[str, int]:
        """
        Count relations grouped by their relation type.

        Args:
            graph: Knowledge graph to analyze

        Returns:
            Dict[str, int]: Mapping of relation type names to their counts.
                Only includes types that have at least one relation.
        """
        counts = {}
        for relation_type in RelationType:
            count = len(graph.get_relations_by_type(relation_type))
            if count > 0:
                counts[relation_type.value] = count
        return counts

    def _analyze_connectivity(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Analyze graph connectivity"""
        if not graph.entities:
            return {"components": 0, "largest_component_size": 0, "is_connected": True}

        components = self._find_connected_components(graph)
        largest_component = max(components, key=len) if components else []

        return {
            "components": len(components),
            "largest_component_size": len(largest_component),
            "is_connected": len(components) <= 1,
            "component_sizes": [len(comp) for comp in components],
        }

    def _analyze_node_degrees(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Analyze node degree distribution"""
        degrees = []
        in_degrees = []
        out_degrees = []

        for entity_id in graph.entities:
            # Count all relations for this entity
            total_degree = 0
            in_degree = 0
            out_degree = 0

            for relation in graph.relations.values():
                if relation.head_entity and relation.head_entity.id == entity_id:
                    out_degree += 1
                    total_degree += 1
                if relation.tail_entity and relation.tail_entity.id == entity_id:
                    in_degree += 1
                    total_degree += 1

            degrees.append(total_degree)
            in_degrees.append(in_degree)
            out_degrees.append(out_degree)

        if not degrees:
            return {"average_degree": 0, "max_degree": 0, "min_degree": 0}

        return {
            "average_degree": sum(degrees) / len(degrees),
            "max_degree": max(degrees),
            "min_degree": min(degrees),
            "average_in_degree": sum(in_degrees) / len(in_degrees),
            "average_out_degree": sum(out_degrees) / len(out_degrees),
        }

    def _find_connected_components(self, graph: KnowledgeGraph) -> List[List[str]]:
        """Find connected components using DFS"""
        visited: Set[str] = set()
        components = []

        for entity_id in graph.entities:
            if entity_id not in visited:
                component = self._dfs_component(graph, entity_id, visited)
                components.append(component)

        return components

    def _dfs_component(self, graph: KnowledgeGraph, start_entity_id: str, visited: Set[str]) -> List[str]:
        """DFS traversal for connected component"""
        component = []
        stack = [start_entity_id]

        while stack:
            entity_id = stack.pop()
            if entity_id in visited:
                continue

            visited.add(entity_id)
            component.append(entity_id)

            # Find neighbors
            neighbors = graph.get_neighbors(entity_id)
            for neighbor in neighbors:
                if neighbor.id not in visited:
                    stack.append(neighbor.id)

        return component

    def find_isolated_nodes(self, graph: KnowledgeGraph) -> List[str]:
        """
        Find isolated nodes (entities with no relations)

        Args:
            graph: Knowledge graph to analyze

        Returns:
            List[str]: List of isolated entity IDs
        """
        isolated_nodes = []

        for entity_id in graph.entities:
            relations = graph.get_entity_relations(entity_id)
            if not relations:
                isolated_nodes.append(entity_id)

        return isolated_nodes

    def compute_density(self, graph: KnowledgeGraph) -> float:
        """
        Compute graph density (ratio of actual edges to possible edges)

        Args:
            graph: Knowledge graph to analyze

        Returns:
            float: Graph density between 0 and 1
        """
        num_entities = len(graph.entities)
        num_relations = len(graph.relations)

        if num_entities <= 1:
            return 0.0

        max_possible_edges = num_entities * (num_entities - 1)  # Directed graph

        return num_relations / max_possible_edges if max_possible_edges > 0 else 0.0

    def get_entity_importance_scores(self, graph: KnowledgeGraph) -> Dict[str, float]:
        """
        Calculate importance scores for entities based on degree centrality

        Args:
            graph: Knowledge graph to analyze

        Returns:
            Dict[str, float]: Entity ID to importance score mapping
        """
        importance_scores = {}

        for entity_id in graph.entities:
            # Count relations for this entity
            degree = len(graph.get_entity_relations(entity_id))

            # Normalize by maximum possible degree
            max_degree = len(graph.entities) - 1
            normalized_score = degree / max_degree if max_degree > 0 else 0.0

            importance_scores[entity_id] = normalized_score

        return importance_scores
