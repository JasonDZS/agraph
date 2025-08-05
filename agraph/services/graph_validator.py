"""
Graph validation service - Single responsibility: graph validation and integrity checking
"""

import logging
from typing import Any, Dict, List, Set

from ..graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class GraphValidator:
    """Service for graph validation and integrity checking"""

    def __init__(self) -> None:
        """Initialize graph validator"""
        pass

    def validate_graph(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """
        Perform comprehensive graph validation

        Args:
            graph: Knowledge graph to validate

        Returns:
            Dict[str, Any]: Validation results with issues and recommendations
        """
        try:
            validation_result: Dict[str, Any] = {
                "valid": True,
                "issues": [],
                "warnings": [],
                "recommendations": [],
                "statistics": {},
            }

            # Check basic integrity
            integrity_issues = self._check_integrity(graph)
            validation_result["issues"].extend(integrity_issues)

            # Check connectivity
            connectivity_issues = self._check_connectivity(graph)
            validation_result["warnings"].extend(connectivity_issues)

            # Check for isolated nodes
            isolated_nodes = self._find_isolated_nodes(graph)
            if isolated_nodes:
                validation_result["warnings"].append(
                    {
                        "type": "isolated_nodes",
                        "count": len(isolated_nodes),
                        "nodes": isolated_nodes[:10],  # Show first 10
                        "severity": "medium",
                    }
                )

            # Check for cycles
            cycles = self._detect_cycles(graph)
            if cycles:
                validation_result["warnings"].append(
                    {
                        "type": "cycles_detected",
                        "count": len(cycles),
                        "cycles": cycles[:5],  # Show first 5
                        "severity": "low",
                    }
                )

            # Check data quality
            quality_issues = self._check_data_quality(graph)
            validation_result["warnings"].extend(quality_issues)

            # Generate recommendations
            recommendations = self._generate_recommendations(
                graph, validation_result["issues"] + validation_result["warnings"]
            )
            validation_result["recommendations"] = recommendations

            # Add statistics
            validation_result["statistics"] = self._get_validation_statistics(graph)

            # Determine overall validity
            high_severity_issues = [
                issue for issue in validation_result["issues"] if issue.get("severity", "medium") == "high"
            ]
            validation_result["valid"] = len(high_severity_issues) == 0

            return validation_result

        except Exception as e:
            logger.error(f"Error validating graph: {e}")
            return {
                "valid": False,
                "issues": [{"type": "validation_error", "message": str(e), "severity": "high"}],
                "warnings": [],
                "recommendations": [],
                "statistics": {},
            }

    def _check_integrity(self, graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Check graph integrity constraints"""
        issues = []

        # Check relation entity references
        for relation_id, relation in graph.relations.items():
            if not relation.head_entity:
                issues.append(
                    {
                        "type": "missing_head_entity",
                        "relation_id": relation_id,
                        "message": f"Relation {relation_id} has no head entity",
                        "severity": "high",
                    }
                )
            elif relation.head_entity.id not in graph.entities:
                issues.append(
                    {
                        "type": "invalid_head_entity_reference",
                        "relation_id": relation_id,
                        "entity_id": relation.head_entity.id,
                        "message": f"Relation {relation_id} references non-existent head entity",
                        "severity": "high",
                    }
                )

            if not relation.tail_entity:
                issues.append(
                    {
                        "type": "missing_tail_entity",
                        "relation_id": relation_id,
                        "message": f"Relation {relation_id} has no tail entity",
                        "severity": "high",
                    }
                )
            elif relation.tail_entity.id not in graph.entities:
                issues.append(
                    {
                        "type": "invalid_tail_entity_reference",
                        "relation_id": relation_id,
                        "entity_id": relation.tail_entity.id,
                        "message": f"Relation {relation_id} references non-existent tail entity",
                        "severity": "high",
                    }
                )

        # Check for duplicate entities with same name and type
        entity_signatures: Dict[tuple, str] = {}
        for entity_id, entity in graph.entities.items():
            signature = (entity.name.lower().strip(), entity.entity_type.value)
            if signature in entity_signatures:
                issues.append(
                    {
                        "type": "duplicate_entity",
                        "entity_ids": f"{entity_signatures[signature]},{entity_id}",
                        "signature": str(signature),
                        "message": f"Duplicate entities found: {signature}",
                        "severity": "medium",
                    }
                )
            else:
                entity_signatures[signature] = entity_id

        return issues

    def _check_connectivity(self, graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Check graph connectivity"""
        issues: List[Dict[str, Any]] = []

        if not graph.entities:
            return issues

        # Find connected components
        components = self._find_connected_components(graph)

        if len(components) > 1:
            component_sizes = [len(comp) for comp in components]
            issues.append(
                {
                    "type": "disconnected_graph",
                    "component_count": len(components),
                    "component_sizes": component_sizes,
                    "largest_component_size": max(component_sizes),
                    "message": f"Graph has {len(components)} disconnected components",
                    "severity": "medium",
                }
            )

        return issues

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

    def _find_isolated_nodes(self, graph: KnowledgeGraph) -> List[str]:
        """Find nodes with no relations"""
        isolated_nodes = []

        for entity_id in graph.entities:
            relations = graph.get_entity_relations(entity_id)
            if not relations:
                isolated_nodes.append(entity_id)

        return isolated_nodes

    def _detect_cycles(self, graph: KnowledgeGraph) -> List[List[str]]:
        """Detect cycles in the graph"""
        cycles = []
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def dfs_cycle(entity_id: str, path: List[str]) -> bool:
            if entity_id in rec_stack:
                # Found cycle
                try:
                    cycle_start = path.index(entity_id)
                    cycle = path[cycle_start:] + [entity_id]
                    cycles.append(cycle)
                    return True
                except ValueError:
                    pass

            if entity_id in visited:
                return False

            visited.add(entity_id)
            rec_stack.add(entity_id)
            path.append(entity_id)

            # Check outgoing relations
            relations = graph.get_entity_relations(entity_id, direction="out")
            for relation in relations:
                if relation.tail_entity and dfs_cycle(relation.tail_entity.id, path):
                    break

            path.pop()
            rec_stack.remove(entity_id)
            return False

        try:
            for entity_id in graph.entities:
                if entity_id not in visited:
                    dfs_cycle(entity_id, [])
        except Exception as e:
            logger.error(f"Error detecting cycles: {e}")

        return cycles

    def _check_data_quality(self, graph: KnowledgeGraph) -> List[Dict[str, Any]]:
        """Check data quality issues"""
        issues = []

        # Check for entities with empty names
        empty_name_entities = []
        low_confidence_entities = []

        for entity_id, entity in graph.entities.items():
            if not entity.name or not entity.name.strip():
                empty_name_entities.append(entity_id)

            if entity.confidence < 0.5:  # Configurable threshold
                low_confidence_entities.append({"entity_id": entity_id, "confidence": entity.confidence})

        if empty_name_entities:
            issues.append(
                {
                    "type": "empty_entity_names",
                    "count": len(empty_name_entities),
                    "entity_ids": empty_name_entities[:10],
                    "message": f"{len(empty_name_entities)} entities have empty names",
                    "severity": "medium",
                }
            )

        if low_confidence_entities:
            issues.append(
                {
                    "type": "low_confidence_entities",
                    "count": len(low_confidence_entities),
                    "entities": low_confidence_entities[:10],
                    "message": f"{len(low_confidence_entities)} entities have low confidence",
                    "severity": "low",
                }
            )

        # Check for relations with low confidence
        low_confidence_relations = []
        for relation_id, relation in graph.relations.items():
            if relation.confidence < 0.5:
                low_confidence_relations.append({"relation_id": relation_id, "confidence": relation.confidence})

        if low_confidence_relations:
            issues.append(
                {
                    "type": "low_confidence_relations",
                    "count": len(low_confidence_relations),
                    "relations": low_confidence_relations[:10],
                    "message": f"{len(low_confidence_relations)} relations have low confidence",
                    "severity": "low",
                }
            )

        return issues

    def _generate_recommendations(self, graph: KnowledgeGraph, issues: List[Dict[str, Any]]) -> List[str]:
        """Generate improvement recommendations based on issues"""
        recommendations = []

        issue_types = [issue["type"] for issue in issues]

        if "missing_head_entity" in issue_types or "missing_tail_entity" in issue_types:
            recommendations.append("Remove or fix relations with missing entity references")

        if "invalid_head_entity_reference" in issue_types or "invalid_tail_entity_reference" in issue_types:
            recommendations.append("Clean up relations that reference non-existent entities")

        if "duplicate_entity" in issue_types:
            recommendations.append("Consider merging duplicate entities with same name and type")

        if "disconnected_graph" in issue_types:
            recommendations.append("Add relations to connect isolated graph components")

        if "isolated_nodes" in issue_types:
            recommendations.append("Add relations for isolated entities or remove unused entities")

        if "empty_entity_names" in issue_types:
            recommendations.append("Provide meaningful names for entities with empty names")

        if "low_confidence_entities" in issue_types or "low_confidence_relations" in issue_types:
            recommendations.append("Review and improve low-confidence entities and relations")

        # Graph structure recommendations
        entity_count = len(graph.entities)
        relation_count = len(graph.relations)

        if relation_count == 0 and entity_count > 0:
            recommendations.append("Add relations between entities to create meaningful connections")
        elif entity_count > 0 and relation_count / entity_count < 0.1:
            recommendations.append("Consider adding more relations to increase graph connectivity")
        elif entity_count > 0 and relation_count / entity_count > 10:
            recommendations.append("Graph may be over-connected; review relation necessity")

        return recommendations

    def _get_validation_statistics(self, graph: KnowledgeGraph) -> Dict[str, Any]:
        """Get statistics for validation report"""
        try:
            entity_count = len(graph.entities)
            relation_count = len(graph.relations)

            # Calculate average confidence
            entity_confidences = [e.confidence for e in graph.entities.values()]
            relation_confidences = [r.confidence for r in graph.relations.values()]

            avg_entity_confidence = sum(entity_confidences) / len(entity_confidences) if entity_confidences else 0
            avg_relation_confidence = (
                sum(relation_confidences) / len(relation_confidences) if relation_confidences else 0
            )

            return {
                "total_entities": entity_count,
                "total_relations": relation_count,
                "density": relation_count / (entity_count * (entity_count - 1)) if entity_count > 1 else 0,
                "average_entity_confidence": avg_entity_confidence,
                "average_relation_confidence": avg_relation_confidence,
                "entities_without_relations": len(self._find_isolated_nodes(graph)),
            }
        except Exception as e:
            logger.error(f"Error computing validation statistics: {e}")
            return {}
