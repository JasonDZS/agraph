"""
Entity merger service - Single responsibility: entity merging operations
"""

import logging
from typing import Any, Dict, List, Optional

from ..entities import Entity
from ..graph import KnowledgeGraph

logger = logging.getLogger(__name__)


class EntityMerger:
    """Service for entity merging operations"""

    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize entity merger

        Args:
            similarity_threshold: Threshold for considering entities similar
        """
        self.similarity_threshold = similarity_threshold

    def merge_entity(self, graph: KnowledgeGraph, target_entity: Entity, source_entity: Entity) -> bool:
        """
        Merge source entity into target entity

        Args:
            graph: Knowledge graph
            target_entity: Target entity to merge into
            source_entity: Source entity to merge from

        Returns:
            bool: True if merge successful
        """
        if target_entity.id not in graph.entities or source_entity.id not in graph.entities:
            logger.error("One or both entities not found in graph")
            return False

        try:
            # Merge attributes
            self._merge_entity_attributes(target_entity, source_entity)

            # Update relations that reference the source entity
            self._update_entity_references(graph, source_entity.id, target_entity)

            # Remove source entity from graph
            graph.remove_entity(source_entity.id)

            logger.info(f"Successfully merged entity {source_entity.id} into {target_entity.id}")
            return True

        except Exception as e:
            logger.error(f"Error merging entities: {e}")
            return False

    def _merge_entity_attributes(self, target_entity: Entity, source_entity: Entity) -> None:
        """Merge attributes from source entity into target entity"""
        # Merge aliases
        target_entity.aliases.extend(source_entity.aliases)
        target_entity.aliases = list(set(target_entity.aliases))  # Remove duplicates

        # Merge properties
        target_entity.properties.update(source_entity.properties)

        # Use higher confidence and better description
        if source_entity.confidence > target_entity.confidence:
            target_entity.confidence = source_entity.confidence
            if source_entity.description and not target_entity.description:
                target_entity.description = source_entity.description

        # Keep track of merge source
        if "merged_from" not in target_entity.properties:
            target_entity.properties["merged_from"] = []
        target_entity.properties["merged_from"].append(source_entity.id)

    def _update_entity_references(self, graph: KnowledgeGraph, old_entity_id: str, new_entity: Entity) -> None:
        """Update all relations that reference the old entity"""
        for relation in graph.relations.values():
            if relation.head_entity and relation.head_entity.id == old_entity_id:
                relation.head_entity = new_entity
            if relation.tail_entity and relation.tail_entity.id == old_entity_id:
                relation.tail_entity = new_entity

    def find_similar_entities(
        self, graph: KnowledgeGraph, similarity_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Find pairs of similar entities that could be merged

        Args:
            graph: Knowledge graph
            similarity_threshold: Override default similarity threshold

        Returns:
            List[Dict[str, Any]]: List of similar entity pairs with similarity scores
        """
        threshold = similarity_threshold or self.similarity_threshold
        similar_pairs = []

        try:
            entities = list(graph.entities.values())

            for i, entity1 in enumerate(entities):
                for entity2 in entities[i + 1 :]:  # Avoid comparing same entity
                    similarity = self._calculate_entity_similarity(entity1, entity2)

                    if similarity >= threshold:
                        similar_pairs.append(
                            {
                                "entity1_id": entity1.id,
                                "entity1_name": entity1.name,
                                "entity2_id": entity2.id,
                                "entity2_name": entity2.name,
                                "similarity": similarity,
                                "same_type": entity1.entity_type == entity2.entity_type,
                            }
                        )

            # Sort by similarity descending
            similar_pairs.sort(key=lambda x: x["similarity"], reverse=True)

            return similar_pairs

        except Exception as e:
            logger.error(f"Error finding similar entities: {e}")
            return []

    def _calculate_entity_similarity(self, entity1: Entity, entity2: Entity) -> float:
        """
        Calculate similarity between two entities

        Args:
            entity1: First entity
            entity2: Second entity

        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            # Name similarity (Jaccard similarity of words)
            name_similarity = self._calculate_name_similarity(entity1.name, entity2.name)

            # Type similarity
            type_similarity = 1.0 if entity1.entity_type == entity2.entity_type else 0.0

            # Alias similarity
            alias_similarity = self._calculate_alias_similarity(entity1.aliases, entity2.aliases)

            # Property similarity
            property_similarity = self._calculate_property_similarity(entity1.properties, entity2.properties)

            # Weighted combination
            weights = {"name": 0.4, "type": 0.2, "alias": 0.2, "property": 0.2}

            total_similarity = (
                weights["name"] * name_similarity
                + weights["type"] * type_similarity
                + weights["alias"] * alias_similarity
                + weights["property"] * property_similarity
            )

            return total_similarity

        except Exception as e:
            logger.error(f"Error calculating entity similarity: {e}")
            return 0.0

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between entity names using Jaccard similarity"""
        if not name1 or not name2:
            return 0.0

        # Convert to lowercase and split into words
        words1 = set(name1.lower().split())
        words2 = set(name2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))

        return intersection / union if union > 0 else 0.0

    def _calculate_alias_similarity(self, aliases1: List[str], aliases2: List[str]) -> float:
        """Calculate similarity between alias lists"""
        if not aliases1 and not aliases2:
            return 1.0
        if not aliases1 or not aliases2:
            return 0.0

        set1 = set(alias.lower() for alias in aliases1)
        set2 = set(alias.lower() for alias in aliases2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _calculate_property_similarity(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
        """Calculate similarity between property dictionaries"""
        if not props1 and not props2:
            return 1.0
        if not props1 or not props2:
            return 0.0

        all_keys = set(props1.keys()).union(set(props2.keys()))
        if not all_keys:
            return 1.0

        matching_keys = 0.0
        for key in all_keys:
            val1 = props1.get(key)
            val2 = props2.get(key)

            if val1 == val2:
                matching_keys += 1
            elif val1 is not None and val2 is not None:
                # Partial similarity for different values
                if isinstance(val1, str) and isinstance(val2, str):
                    if val1.lower() == val2.lower():
                        matching_keys += 0.8
                    elif val1.lower() in val2.lower() or val2.lower() in val1.lower():
                        matching_keys += 0.5

        return matching_keys / len(all_keys)

    def auto_merge_similar_entities(
        self, graph: KnowledgeGraph, similarity_threshold: Optional[float] = None, max_merges: int = 100
    ) -> Dict[str, Any]:
        """
        Automatically merge similar entities in the graph

        Args:
            graph: Knowledge graph
            similarity_threshold: Override default similarity threshold
            max_merges: Maximum number of merges to perform

        Returns:
            Dict[str, Any]: Results of the merge operation
        """
        threshold = similarity_threshold or self.similarity_threshold
        similar_pairs = self.find_similar_entities(graph, threshold)

        merged_count = 0
        merge_results = []

        try:
            for pair in similar_pairs[:max_merges]:
                entity1_id = pair["entity1_id"]
                entity2_id = pair["entity2_id"]

                # Skip if either entity has already been merged
                if entity1_id not in graph.entities or entity2_id not in graph.entities:
                    continue

                entity1 = graph.entities[entity1_id]
                entity2 = graph.entities[entity2_id]

                # Merge entity with lower confidence into higher confidence one
                if entity1.confidence >= entity2.confidence:
                    target, source = entity1, entity2
                else:
                    target, source = entity2, entity1

                if self.merge_entity(graph, target, source):
                    merged_count += 1
                    merge_results.append(
                        {
                            "target_id": target.id,
                            "target_name": target.name,
                            "source_id": source.id,
                            "source_name": source.name,
                            "similarity": pair["similarity"],
                        }
                    )

            return {
                "total_merges": merged_count,
                "merge_details": merge_results,
                "entities_before": len(graph.entities) + merged_count,
                "entities_after": len(graph.entities),
            }

        except Exception as e:
            logger.error(f"Error in auto-merge: {e}")
            return {"total_merges": merged_count, "merge_details": merge_results, "error": str(e)}
