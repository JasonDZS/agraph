"""
Knowledge graph utility functions.

This module provides utility functions for knowledge graph operations including
export to visualization formats, graph analysis, path finding, and validation.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Union

from .graph import KnowledgeGraph
from .logger import logger
from .relations import Relation
from .types import EntityType, RelationType


def get_type_value(type_obj: Union[EntityType, RelationType, str]) -> str:
    """
    Safely get the value from a type object that could be an enum or string.

    Args:
        type_obj: The type object (EntityType, RelationType, or string)

    Returns:
        str: The string value of the type
    """
    if hasattr(type_obj, "value"):
        return str(type_obj.value)
    return str(type_obj)


def export_graph_to_cytoscape(graph: KnowledgeGraph) -> Dict[str, Any]:
    """Export knowledge graph to Cytoscape.js format.

    Args:
        graph: The knowledge graph to export.

    Returns:
        Dict[str, Any]: Data in Cytoscape.js format with nodes and edges.
    """
    nodes = []
    edges: List[Dict[str, Any]] = []

    # Convert entities to nodes
    for entity in graph.entities.values():
        node = {
            "data": {
                "id": entity.id,
                "label": entity.name,
                "type": get_type_value(entity.entity_type),
                "description": entity.description,
                "confidence": entity.confidence,
                "source": entity.source,
            },
            "classes": get_type_value(entity.entity_type),
        }
        nodes.append(node)

    # Process relations as edges
    edges = []
    for relation in graph.relations.values():
        if relation.head_entity is None or relation.tail_entity is None:
            continue
        edge = {
            "data": {
                "id": relation.id,
                "source": relation.head_entity.id,
                "target": relation.tail_entity.id,
                "label": get_type_value(relation.relation_type),
                "type": get_type_value(relation.relation_type),
                "confidence": relation.confidence,
                "source_info": relation.source,
            },
            "classes": get_type_value(relation.relation_type),
        }
        edges.append(edge)

    return {
        "elements": {"nodes": nodes, "edges": edges},
        "graph_info": {
            "id": graph.id,
            "name": graph.name,
            "created_at": graph.created_at.isoformat(),
            "updated_at": graph.updated_at.isoformat(),
            "statistics": graph.get_basic_statistics(),
        },
    }


def export_graph_to_d3(graph: KnowledgeGraph) -> Dict[str, Any]:
    """Export knowledge graph to D3.js format.

    Args:
        graph: The knowledge graph to export.

    Returns:
        Dict[str, Any]: Data in D3.js format with nodes and links.
    """
    nodes = []
    links = []

    # Create node ID mapping
    node_id_map = {entity_id: i for i, entity_id in enumerate(graph.entities.keys())}

    # Convert entities to nodes
    for i, entity in enumerate(graph.entities.values()):
        node = {
            "id": i,
            "entity_id": entity.id,
            "name": entity.name,
            "type": get_type_value(entity.entity_type),
            "description": entity.description,
            "confidence": entity.confidence,
            "group": get_type_value(entity.entity_type),
            "size": max(5, min(20, entity.confidence * 15)),  # Set size based on confidence
        }
        nodes.append(node)

    # Convert relations to links
    for relation in graph.relations.values():
        if (
            relation.head_entity is not None
            and relation.tail_entity is not None
            and relation.head_entity.id in node_id_map
            and relation.tail_entity.id in node_id_map
        ):

            link = {
                "source": node_id_map[relation.head_entity.id],
                "target": node_id_map[relation.tail_entity.id],
                "relation_id": relation.id,
                "type": get_type_value(relation.relation_type),
                "confidence": relation.confidence,
                "value": relation.confidence,  # Link weight in D3
            }
            links.append(link)

    return {
        "nodes": nodes,
        "links": links,
        "graph_info": {
            "id": graph.id,
            "name": graph.name,
            "node_count": len(nodes),
            "link_count": len(links),
        },
    }


def find_shortest_path(graph: KnowledgeGraph, start_entity_id: str, end_entity_id: str) -> Optional[List[Relation]]:
    """Find the shortest path between two entities.

    Args:
        graph: The knowledge graph to search.
        start_entity_id: The ID of the starting entity.
        end_entity_id: The ID of the target entity.

    Returns:
        Optional[List[Relation]]: List of relations on the shortest path,
                                 or None if no path exists.
    """
    if start_entity_id not in graph.entities or end_entity_id not in graph.entities:
        return None

    if start_entity_id == end_entity_id:
        return []

    # BFS to find shortest path

    queue: deque = deque([(start_entity_id, [])])
    visited = {start_entity_id}

    while queue:
        current_entity_id, path = queue.popleft()

        # Get all neighbors of current entity
        # neighbors = graph.get_neighbors(current_entity_id)
        relations = graph.get_entity_relations(current_entity_id, direction="out")

        for relation in relations:
            if relation.tail_entity is None:
                continue
            neighbor_id = relation.tail_entity.id

            if neighbor_id == end_entity_id:
                result_path: List[Relation] = path + [relation]
                return result_path

            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, path + [relation]))

    return None


def calculate_graph_metrics(graph: KnowledgeGraph) -> Dict[str, Any]:
    """Calculate network metrics for the knowledge graph.

    Args:
        graph: The knowledge graph to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing various graph metrics and statistics.
    """
    if not graph.entities:
        return {}

    # Basic statistics
    node_count = len(graph.entities)
    edge_count = len(graph.relations)

    # Calculate degree distribution
    degree_map = {}
    in_degree_map = {}
    out_degree_map = {}

    for entity_id in graph.entities:
        total_degree = len(graph.get_entity_relations(entity_id))
        in_degree = len(graph.get_entity_relations(entity_id, direction="in"))
        out_degree = len(graph.get_entity_relations(entity_id, direction="out"))

        degree_map[entity_id] = total_degree
        in_degree_map[entity_id] = in_degree
        out_degree_map[entity_id] = out_degree

    # Statistical metrics
    degrees = list(degree_map.values())
    avg_degree = sum(degrees) / len(degrees) if degrees else 0
    max_degree = max(degrees) if degrees else 0
    min_degree = min(degrees) if degrees else 0

    # Density calculation
    max_possible_edges = node_count * (node_count - 1)
    density = (2 * edge_count) / max_possible_edges if max_possible_edges > 0 else 0

    # Find nodes with highest degree (central nodes)
    central_nodes = sorted(degree_map.items(), key=lambda x: x[1], reverse=True)[:5]

    # Connectivity analysis
    components = _find_connected_components(graph)

    return {
        "basic_stats": {
            "node_count": node_count,
            "edge_count": edge_count,
            "density": round(density, 4),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max_degree,
            "min_degree": min_degree,
        },
        "centrality": {
            "top_central_nodes": [
                {
                    "entity_id": entity_id,
                    "entity_name": graph.entities[entity_id].name,
                    "degree": degree,
                }
                for entity_id, degree in central_nodes
            ]
        },
        "connectivity": {
            "connected_components": len(components),
            "largest_component_size": max(len(comp) for comp in components) if components else 0,
            "is_connected": len(components) <= 1,
        },
        "type_distribution": _calculate_type_distribution(graph),
    }


def _find_connected_components(graph: KnowledgeGraph) -> List[List[str]]:
    """Find connected components in the graph.

    Args:
        graph: The knowledge graph to analyze.

    Returns:
        List[List[str]]: List of connected components, each containing entity IDs.
    """
    visited: set[str] = set()
    components = []

    for entity_id in graph.entities:
        if entity_id not in visited:
            component = _dfs_component(graph, entity_id, visited)
            components.append(component)

    return components


def _dfs_component(graph: KnowledgeGraph, start_id: str, visited: set) -> List[str]:
    """Use DFS to find a connected component.

    Args:
        graph: The knowledge graph to search.
        start_id: The starting entity ID for DFS.
        visited: Set of already visited entity IDs.

    Returns:
        List[str]: List of entity IDs in the connected component.
    """
    component = []
    stack = [start_id]

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            component.append(current)

            # Add neighbor nodes
            neighbors = graph.get_neighbors(current)
            for neighbor in neighbors:
                if neighbor.id not in visited:
                    stack.append(neighbor.id)

    return component


def _calculate_type_distribution(graph: KnowledgeGraph) -> Dict[str, Any]:
    """Calculate type distribution for entities and relations.

    Args:
        graph: The knowledge graph to analyze.

    Returns:
        Dict[str, Any]: Dictionary containing entity and relation type distributions.
    """
    entity_types: Dict[str, int] = {}
    relation_types: Dict[str, int] = {}

    for entity in graph.entities.values():
        entity_type = get_type_value(entity.entity_type)
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1

    for relation in graph.relations.values():
        relation_type = get_type_value(relation.relation_type)
        relation_types[relation_type] = relation_types.get(relation_type, 0) + 1

    return {"entity_types": entity_types, "relation_types": relation_types}


def merge_similar_entities(graph: KnowledgeGraph, similarity_threshold: float = 0.8) -> int:
    """Merge similar entities in the knowledge graph.

    Args:
        graph: The knowledge graph to process.
        similarity_threshold: Similarity threshold for merging (0.0 to 1.0).

    Returns:
        int: Number of entities merged.
    """
    merged_count = 0
    entities_to_remove: set[str] = set()

    entity_list = list(graph.entities.values())

    for i, entity1 in enumerate(entity_list):
        if entity1.id in entities_to_remove:
            continue

        for entity2 in entity_list[i + 1 :]:
            if entity2.id in entities_to_remove:
                continue

            # Calculate name similarity
            similarity = _calculate_name_similarity(entity1.name, entity2.name)

            if similarity >= similarity_threshold:
                # Merge entities (Note: merge_entity method needs to be implemented)
                # if graph.merge_entity(entity1, entity2):
                #     entities_to_remove.add(entity2.id)
                #     merged_count += 1
                #     logger.info("Merged entities: %s <- %s", entity1.name, entity2.name)
                logger.info(
                    "Found similar entities: %s <-> %s (similarity: %.2f)", entity1.name, entity2.name, similarity
                )

    return merged_count


def _calculate_name_similarity(name1: str, name2: str) -> float:
    """Calculate name similarity using simple Jaccard similarity.

    Args:
        name1: First name to compare.
        name2: Second name to compare.

    Returns:
        float: Similarity score between 0.0 and 1.0.
    """
    set1 = set(name1.lower().split())
    set2 = set(name2.lower().split())

    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    return intersection / union if union > 0 else 0.0


def validate_graph_consistency(graph: KnowledgeGraph) -> List[Dict[str, Any]]:
    """Validate graph consistency and find issues.

    Args:
        graph: The knowledge graph to validate.

    Returns:
        List[Dict[str, Any]]: List of consistency issues found.
    """
    issues = []

    # Check entity references in relations
    for relation in graph.relations.values():
        if not relation.head_entity or relation.head_entity.id not in graph.entities:
            issues.append(
                {
                    "type": "missing_head_entity",
                    "relation_id": relation.id,
                    "description": f"Relation {relation.id} has missing head entity",
                }
            )

        if not relation.tail_entity or relation.tail_entity.id not in graph.entities:
            issues.append(
                {
                    "type": "missing_tail_entity",
                    "relation_id": relation.id,
                    "description": f"Relation {relation.id} has missing tail entity",
                }
            )

    # Check for duplicate relations
    relation_signatures: Dict[str, List[str]] = {}
    for relation in graph.relations.values():
        if relation.head_entity and relation.tail_entity:
            signature_tuple = (
                relation.head_entity.id,
                relation.tail_entity.id,
                get_type_value(relation.relation_type),
            )
            signature = str(signature_tuple)

            if signature in relation_signatures:
                relation_signatures[signature].append(relation.id)
                issues.append(
                    {
                        "type": "duplicate_relation",
                        "relation_id": relation.id,
                        "description": f"Found duplicate relation: {signature}",
                    }
                )
            else:
                relation_signatures[signature] = [relation.id]

    # Check for self-loop relations
    for relation in graph.relations.values():
        if relation.head_entity and relation.tail_entity and relation.head_entity.id == relation.tail_entity.id:
            issues.append(
                {
                    "type": "self_loop",
                    "relation_id": relation.id,
                    "entity_id": relation.head_entity.id,
                    "description": f"Entity {relation.head_entity.name} has self-loop relation",
                }
            )

    return issues


def create_graph_summary(graph: KnowledgeGraph) -> str:
    """Create a summary report of the knowledge graph.

    Args:
        graph: The knowledge graph to summarize.

    Returns:
        str: A formatted summary report of the graph.
    """
    stats = graph.get_basic_statistics()
    metrics = calculate_graph_metrics(graph)

    summary = f"""
Knowledge Graph Summary Report
==============================

Basic Information:
- Graph Name: {graph.name}
- Graph ID: {graph.id}
- Created: {graph.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- Updated: {graph.updated_at.strftime('%Y-%m-%d %H:%M:%S')}

Statistics:
- Total Entities: {stats['total_entities']}
- Total Relations: {stats['total_relations']}
- Graph Density: {metrics.get('basic_stats', {}).get('density', 0):.4f}
- Average Degree: {metrics.get('basic_stats', {}).get('avg_degree', 0):.2f}

Entity Type Distribution:
"""

    for entity_type, count in stats.get("entity_types", {}).items():
        summary += f"- {entity_type}: {count} entities\n"

    summary += "\nRelation Type Distribution:\n"
    for relation_type, count in stats.get("relation_types", {}).items():
        summary += f"- {relation_type}: {count} relations\n"

    # Add central nodes
    central_nodes = metrics.get("centrality", {}).get("top_central_nodes", [])
    if central_nodes:
        summary += "\nCentral Nodes (Highest Degree):\n"
        for node in central_nodes[:3]:
            summary += f"- {node['entity_name']} (degree: {node['degree']})\n"

    # Connectivity information
    connectivity = metrics.get("connectivity", {})
    summary += "\nConnectivity:\n"
    summary += f"- Connected Components: {connectivity.get('connected_components', 0)}\n"
    summary += f"- Largest Component Size: {connectivity.get('largest_component_size', 0)}\n"
    summary += f"- Is Connected: {'Yes' if connectivity.get('is_connected', False) else 'No'}\n"

    return summary
