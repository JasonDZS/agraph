"""
Base classes for entity and relation extractors.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ...base.entities import Entity
from ...base.relations import Relation
from ...base.text import TextChunk


class EntityExtractor(ABC):
    """Abstract base class for entity extractors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize entity extractor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    async def extract(self, chunks: List[TextChunk]) -> List[Entity]:
        """Extract entities from text chunks.

        Args:
            chunks: List of text chunks to process

        Returns:
            List of extracted entities
        """

    @abstractmethod
    async def extract_from_text(self, text: str, chunk_id: Optional[str] = None) -> List[Entity]:
        """Extract entities from raw text.

        Args:
            text: Text to extract entities from
            chunk_id: Optional chunk ID for reference

        Returns:
            List of extracted entities
        """

    def validate_entity(self, entity: Entity) -> bool:
        """Validate extracted entity.

        Args:
            entity: Entity to validate

        Returns:
            True if entity is valid
        """
        if not entity.name or not entity.name.strip():
            return False

        # Check confidence threshold if configured
        min_confidence = self.config.get("min_confidence", 0.0)
        if entity.confidence < min_confidence:
            return False

        return True

    def post_process_entities(self, entities: List[Entity]) -> List[Entity]:
        """Post-process extracted entities.

        Args:
            entities: List of entities to process

        Returns:
            Processed list of entities
        """
        # Filter out invalid entities
        valid_entities = [e for e in entities if self.validate_entity(e)]

        # Remove duplicates based on name and type
        unique_entities = []
        seen = set()

        for entity in valid_entities:
            key = (entity.name.lower(), str(entity.entity_type))
            if key not in seen:
                seen.add(key)
                unique_entities.append(entity)
            else:
                # Merge with existing entity (keep higher confidence)
                existing = next(
                    e for e in unique_entities if (e.name.lower(), str(e.entity_type)) == key
                )
                if entity.confidence > existing.confidence:
                    unique_entities.remove(existing)
                    unique_entities.append(entity)

        return unique_entities


class RelationExtractor(ABC):
    """Abstract base class for relation extractors."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize relation extractor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    @abstractmethod
    async def extract(self, chunks: List[TextChunk], entities: List[Entity]) -> List[Relation]:
        """Extract relations from text chunks and entities.

        Args:
            chunks: List of text chunks to process
            entities: List of known entities

        Returns:
            List of extracted relations
        """

    @abstractmethod
    async def extract_from_text(
        self, text: str, entities: List[Entity], chunk_id: Optional[str] = None
    ) -> List[Relation]:
        """Extract relations from raw text and entities.

        Args:
            text: Text to extract relations from
            entities: List of entities to find relations between
            chunk_id: Optional chunk ID for reference

        Returns:
            List of extracted relations
        """

    def validate_relation(self, relation: Relation) -> bool:
        """Validate extracted relation.

        Args:
            relation: Relation to validate

        Returns:
            True if relation is valid
        """
        if not relation.head_entity or not relation.tail_entity:
            return False

        # Relations shouldn't be self-referential
        if relation.head_entity.id == relation.tail_entity.id:
            return False

        # Check confidence threshold if configured
        min_confidence = self.config.get("min_confidence", 0.0)
        if relation.confidence < min_confidence:
            return False

        return True

    def post_process_relations(self, relations: List[Relation]) -> List[Relation]:
        """Post-process extracted relations.

        Args:
            relations: List of relations to process

        Returns:
            Processed list of relations
        """
        # Filter out invalid relations
        valid_relations = [r for r in relations if self.validate_relation(r)]

        # Remove duplicate relations
        unique_relations = []
        seen = set()

        for relation in valid_relations:
            # Additional safety check (validate_relation should have caught this, but being extra safe)
            if relation.head_entity is None or relation.tail_entity is None:
                continue

            # Create key from head, tail, and relation type
            key = (relation.head_entity.id, relation.tail_entity.id, str(relation.relation_type))

            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # Merge with existing relation (keep higher confidence)
                existing = next(
                    r
                    for r in unique_relations
                    if (
                        r.head_entity is not None
                        and r.tail_entity is not None
                        and (r.head_entity.id, r.tail_entity.id, str(r.relation_type)) == key
                    )
                )
                if relation.confidence > existing.confidence:
                    unique_relations.remove(existing)
                    unique_relations.append(relation)

        return unique_relations

    def find_entities_in_text(self, text: str, entities: List[Entity]) -> List[Entity]:
        """Find which entities are mentioned in the text.

        Args:
            text: Text to search in
            entities: List of entities to search for

        Returns:
            List of entities found in text
        """
        found_entities = []
        text_lower = text.lower()

        for entity in entities:
            # Check if entity name is mentioned
            if entity.name.lower() in text_lower:
                found_entities.append(entity)
                continue

            # Check aliases
            for alias in entity.aliases:
                if alias.lower() in text_lower:
                    found_entities.append(entity)
                    break

        return found_entities
