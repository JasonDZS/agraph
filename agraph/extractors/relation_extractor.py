"""
Relation Extractor Module

This module provides relation extraction capabilities from various data sources.
Includes base classes and concrete implementations for text and database relation extraction.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..entities import Entity
from ..logger import logger
from ..relations import Relation
from ..types import EntityType, RelationType


class BaseRelationExtractor(ABC):
    """
    Base class for relation extractors.

    This abstract class defines the common interface for all relation extractors.
    It provides relation validation, implicit relation inference and pattern matching utilities.
    """

    def __init__(self) -> None:
        self.relation_patterns: Dict[str, List[str]] = {}
        self.dependency_parser = None
        self.confidence_threshold = 0.5

    @abstractmethod
    def extract_from_text(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Extract entity relations from text.

        Args:
            text: Input text to extract relations from
            entities: List of identified entities

        Returns:
            List[Relation]: List of extracted relations
        """

    @abstractmethod
    def extract_from_database(self, schema: Dict[str, Any], entities: List[Entity]) -> List[Relation]:
        """
        Extract relations from database schema.

        Args:
            schema: Database schema information
            entities: Database entity list

        Returns:
            List[Relation]: List of extracted relations
        """

    def validate_relation(self, relation: Relation) -> bool:
        """
        Validate relation validity.

        Args:
            relation: Relation to validate

        Returns:
            bool: True if relation is valid, False otherwise
        """
        if not relation.is_valid():
            return False

        # Check confidence threshold
        if relation.confidence < self.confidence_threshold:
            return False

        # Check relation type reasonableness
        if not self._is_relation_type_valid(relation):
            return False

        return True

    def infer_implicit_relations(self, entities: List[Entity], relations: List[Relation]) -> List[Relation]:
        """
        Infer implicit relations based on existing entities and relations.

        Args:
            entities: Entity list
            relations: Known relation list

        Returns:
            List[Relation]: List of inferred implicit relations
        """
        implicit_relations = []

        # Infer relations based on transitivity
        transitive_relations = self._infer_transitive_relations(relations)
        implicit_relations.extend(transitive_relations)

        # Infer relations based on symmetry
        symmetric_relations = self._infer_symmetric_relations(relations)
        implicit_relations.extend(symmetric_relations)

        # Infer relations based on hierarchical structure
        hierarchical_relations = self._infer_hierarchical_relations(entities, relations)
        implicit_relations.extend(hierarchical_relations)

        return implicit_relations

    def _is_relation_type_valid(self, relation: Relation) -> bool:
        """
        Check relation type reasonableness based on entity types and relation type.

        Args:
            relation: Relation to validate

        Returns:
            bool: True if relation type is reasonable, False otherwise
        """
        if relation.head_entity is None or relation.tail_entity is None:
            return False
        head_type = relation.head_entity.entity_type
        tail_type = relation.tail_entity.entity_type
        relation_type = relation.relation_type

        # Define reasonable relation type combinations
        valid_combinations = {
            (EntityType.DATABASE, EntityType.TABLE, RelationType.CONTAINS),
            (EntityType.TABLE, EntityType.COLUMN, RelationType.CONTAINS),
            (EntityType.COLUMN, EntityType.COLUMN, RelationType.FOREIGN_KEY),
            (EntityType.DOCUMENT, EntityType.CONCEPT, RelationType.MENTIONS),
            (EntityType.PERSON, EntityType.ORGANIZATION, RelationType.BELONGS_TO),
            (EntityType.PERSON, EntityType.ORGANIZATION, RelationType.FOUNDED_BY),
            (EntityType.ORGANIZATION, EntityType.PRODUCT, RelationType.DEVELOPS),
            (EntityType.ORGANIZATION, EntityType.SOFTWARE, RelationType.DEVELOPS),
            (EntityType.ORGANIZATION, EntityType.LOCATION, RelationType.BELONGS_TO),
            (EntityType.CONCEPT, EntityType.CONCEPT, RelationType.SIMILAR_TO),
            (EntityType.CONCEPT, EntityType.CONCEPT, RelationType.RELATED_TO),
            (EntityType.PRODUCT, EntityType.CONCEPT, RelationType.RELATED_TO),
            (EntityType.SOFTWARE, EntityType.CONCEPT, RelationType.RELATED_TO),
        }

        # Temporarily relax validation to support more relation type combinations
        return (head_type, tail_type, relation_type) in valid_combinations or relation_type in {
            RelationType.RELATED_TO,
            RelationType.MENTIONS,
            RelationType.DESCRIBES,
        }

    def _infer_transitive_relations(self, relations: List[Relation]) -> List[Relation]:
        """
        Infer relations based on transitivity rules.

        Args:
            relations: List of existing relations

        Returns:
            List[Relation]: List of transitive relations
        """
        transitive_relations = []

        # A contains B, B contains C => A contains C
        for r1 in relations:
            if r1.relation_type == RelationType.CONTAINS:
                for r2 in relations:
                    if (
                        r2.relation_type == RelationType.CONTAINS
                        and r1.tail_entity is not None
                        and r2.head_entity is not None
                        and r1.tail_entity.id == r2.head_entity.id
                    ):

                        # Create transitive relation
                        transitive_relation = Relation(
                            head_entity=r1.head_entity,
                            tail_entity=r2.tail_entity,
                            relation_type=RelationType.CONTAINS,
                            confidence=min(r1.confidence, r2.confidence) * 0.8,
                            source="transitive_inference",
                            properties={"inferred_from": [r1.id, r2.id]},
                        )
                        transitive_relations.append(transitive_relation)

        return transitive_relations

    def _infer_symmetric_relations(self, relations: List[Relation]) -> List[Relation]:
        """
        Infer relations based on symmetry rules.

        Args:
            relations: List of existing relations

        Returns:
            List[Relation]: List of symmetric relations
        """
        symmetric_relations = []

        symmetric_types = {RelationType.SIMILAR_TO, RelationType.SYNONYMS}

        for relation in relations:
            if relation.relation_type in symmetric_types:
                # Create reverse relation
                reverse_relation = Relation(
                    head_entity=relation.tail_entity,
                    tail_entity=relation.head_entity,
                    relation_type=relation.relation_type,
                    confidence=relation.confidence * 0.9,
                    source="symmetric_inference",
                    properties={"inferred_from": relation.id},
                )
                symmetric_relations.append(reverse_relation)

        return symmetric_relations

    def _infer_hierarchical_relations(self, entities: List[Entity], relations: List[Relation]) -> List[Relation]:
        """
        Infer relations based on hierarchical structure.

        Args:
            entities: List of entities
            relations: List of existing relations

        Returns:
            List[Relation]: List of hierarchical relations
        """
        hierarchical_relations = []

        # Infer hierarchical relations based on entity types
        type_hierarchy = {
            EntityType.DATABASE: [EntityType.TABLE],
            EntityType.TABLE: [EntityType.COLUMN],
            EntityType.ORGANIZATION: [EntityType.PERSON],
            EntityType.DOCUMENT: [EntityType.CONCEPT, EntityType.KEYWORD],
        }

        entity_by_type: dict[Any, list[Entity]] = {}
        for entity in entities:
            if entity.entity_type not in entity_by_type:
                entity_by_type[entity.entity_type] = []
            entity_by_type[entity.entity_type].append(entity)

        for parent_type, child_types in type_hierarchy.items():
            parent_entities = entity_by_type.get(parent_type, [])

            for child_type in child_types:
                child_entities = entity_by_type.get(child_type, [])

                for parent_entity in parent_entities:
                    for child_entity in child_entities:
                        # Infer hierarchical relations based on naming similarity
                        if self._is_hierarchically_related(parent_entity, child_entity):
                            hierarchical_relation = Relation(
                                head_entity=parent_entity,
                                tail_entity=child_entity,
                                relation_type=RelationType.CONTAINS,
                                confidence=0.6,
                                source="hierarchical_inference",
                            )
                            hierarchical_relations.append(hierarchical_relation)

        return hierarchical_relations

    def _is_hierarchically_related(self, parent_entity: Entity, child_entity: Entity) -> bool:
        """
        Check if entities have hierarchical relationship.

        Args:
            parent_entity: Potential parent entity
            child_entity: Potential child entity

        Returns:
            bool: True if entities are hierarchically related, False otherwise
        """
        parent_name = parent_entity.name.lower()
        child_name = child_entity.name.lower()

        # Check name containment relationship
        if parent_name in child_name:
            return True

        # Check association information in properties
        if child_entity.properties and "table" in child_entity.properties:
            return bool(child_entity.properties["table"] == parent_entity.name)

        return False


class TextRelationExtractor(BaseRelationExtractor):
    """
    Text-based relation extractor.

    Extracts relations from text using pattern matching and co-occurrence analysis.
    Supports various relation types including belongs_to, contains, similar_to, etc.
    """

    def __init__(self) -> None:
        super().__init__()
        self._init_relation_patterns()

    def _init_relation_patterns(self) -> None:
        """Initialize relation patterns for different relation types."""
        self.relation_patterns = {
            RelationType.BELONGS_TO.value: [
                r"(.+?) (?:belongs to|is part of|works for) (.+)",
                r"(.+?) of (.+)",
                r"(.+?)(?:位于|在)(.+)",  # Chinese location relations
                r"(.+?)(?:属于|隶属于)(.+)",  # Chinese belonging relations
            ],
            RelationType.CONTAINS.value: [
                r"(.+?) (?:contains|includes|has) (.+)",
                r"(.+?) with (.+)",
                r"(.+?)(?:包括|包含|有)(.+)",  # Chinese containment relations
                r"(.+?)(?:下辖|管辖)(.+)",  # Chinese governance relations
            ],
            RelationType.SIMILAR_TO.value: [
                r"(.+?) (?:is similar to|resembles|is like) (.+)",
                r"(.+?) and (.+?) are similar",
                r"(.+?)(?:类似于|相似于)(.+)",  # Chinese similarity relations
            ],
            RelationType.RELATED_TO.value: [
                r"(.+?) (?:is related to|relates to|associated with) (.+)",
                r"(.+?) and (.+?) are related",
                r"(.+?)(?:相关|关联|涉及)(.+)",  # Chinese related relations
            ],
            RelationType.DESCRIBES.value: [
                r"(.+?) (?:describes|explains|defines) (.+)",
                r"(.+?) is described by (.+)",
                r"(.+?)(?:描述|说明|定义)(.+)",  # Chinese description relations
            ],
            RelationType.DEVELOPS.value: [
                r"(.+?) (?:develops|creates|builds) (.+)",
                r"(.+?) developed by (.+)",
                r"(.+?)(?:开发|研发|创造|制造)(.+)",  # Chinese development relations
                r"(.+?)(?:由)(.+?)(?:开发|创建)",  # Chinese passive development relations
            ],
            RelationType.FOUNDED_BY.value: [
                r"(.+?) (?:founded by|established by|created by) (.+)",
                r"(.+?)(?:由)(.+?)(?:创立|成立|建立)",  # Chinese founding relations
                r"(.+?)(?:创建于|成立于)(.+)",  # Chinese time-based founding relations
            ],
        }

    def extract_from_text(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Extract entity relations from text using pattern matching and co-occurrence analysis.

        Args:
            text: Input text to extract relations from
            entities: List of identified entities

        Returns:
            List[Relation]: List of extracted relations
        """
        relations: List[Relation] = []

        try:
            # Create entity name to entity mapping
            entity_map = {entity.name.lower(): entity for entity in entities}

            # Pattern-based relation extraction
            pattern_relations = self._extract_pattern_relations(text, entity_map)
            relations.extend(pattern_relations)

            # Co-occurrence based relation extraction
            cooccurrence_relations = self._extract_cooccurrence_relations(text, entities)
            relations.extend(cooccurrence_relations)

            # Filter and validate relations
            valid_relations = [r for r in relations if self.validate_relation(r)]

            return valid_relations

        except Exception as e:
            logger.error("Error extracting relations from text: %s", e)
            return []

    def _extract_pattern_relations(self, text: str, entity_map: Dict[str, Entity]) -> List[Relation]:
        """
        Extract relations based on pattern matching.

        Args:
            text: Input text
            entity_map: Mapping from entity names to entities

        Returns:
            List[Relation]: List of pattern-extracted relations
        """
        relations: List[Relation] = []

        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    groups = match.groups()
                    if len(groups) >= 2:
                        head_name = groups[0].strip().lower()
                        tail_name = groups[1].strip().lower()

                        # Find corresponding entities
                        head_entity = self._find_entity_by_name(head_name, entity_map)
                        tail_entity = self._find_entity_by_name(tail_name, entity_map)

                        if head_entity and tail_entity:
                            relation = Relation(
                                head_entity=head_entity,
                                tail_entity=tail_entity,
                                relation_type=RelationType(relation_type),
                                confidence=0.7,
                                source="text_pattern_matching",
                                properties={
                                    "pattern": pattern,
                                    "context": text[max(0, match.start() - 50) : match.end() + 50],
                                },
                            )
                            relations.append(relation)

        return relations

    def extract_from_database(self, schema: Dict[str, Any], entities: List[Entity]) -> List[Relation]:
        """Text extractor does not process database relations."""
        return []

    def _find_entity_by_name(self, name: str, entity_map: Dict[str, Entity]) -> Optional[Entity]:
        """
        Find entity by name with exact and fuzzy matching.

        Args:
            name: Entity name to search for
            entity_map: Mapping from entity names to entities

        Returns:
            Optional[Entity]: Found entity or None
        """
        # Exact match
        if name in entity_map:
            return entity_map[name]

        # Fuzzy match
        for entity_name, entity in entity_map.items():
            if name in entity_name or entity_name in name:
                return entity

            # Check aliases
            for alias in entity.aliases:
                if name == alias.lower() or name in alias.lower():
                    return entity

        return None

    def _extract_cooccurrence_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """
        Extract relations based on entity co-occurrence in sentences.

        Args:
            text: Input text
            entities: List of entities

        Returns:
            List[Relation]: List of co-occurrence relations
        """
        relations: List[Relation] = []

        # Find co-occurring entities at sentence level
        sentences = re.split(r"[.!?]+", text)

        for sentence in sentences:
            sentence = sentence.strip().lower()
            if not sentence:
                continue

            # Find entities appearing in the same sentence
            sentence_entities = []
            for entity in entities:
                if entity.name.lower() in sentence or bool(any(alias.lower() in sentence for alias in entity.aliases)):
                    sentence_entities.append(entity)

            # Create relations for co-occurring entities
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i + 1 :]:
                    relation = Relation(
                        head_entity=entity1,
                        tail_entity=entity2,
                        relation_type=RelationType.RELATED_TO,
                        confidence=0.5,
                        source="cooccurrence",
                        properties={"sentence": sentence[:200]},
                    )
                    relations.append(relation)

        return relations


class DatabaseRelationExtractor(BaseRelationExtractor):
    """
    Database-focused relation extractor.

    Extracts relations from database schemas including foreign keys, table-column relationships,
    and semantic relationships based on naming patterns.
    """

    def extract_from_text(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Database extractor does not process text data."""
        return []

    def extract_from_database(self, schema: Dict[str, Any], entities: List[Entity]) -> List[Relation]:
        """
        Extract relations from database schema including structural and semantic relationships.

        Args:
            schema: Database schema information
            entities: Database entity list

        Returns:
            List[Relation]: List of extracted relations
        """
        relations: List[Relation] = []

        try:
            # Create entity mapping
            entity_map = {entity.name: entity for entity in entities}

            # Extract database-table relations
            db_table_relations = self._extract_database_table_relations(schema, entity_map)
            relations.extend(db_table_relations)

            # Extract table-column relations
            table_column_relations = self._extract_table_column_relations(schema, entity_map)
            relations.extend(table_column_relations)

            # Extract foreign key relations
            foreign_key_relations = self._extract_foreign_key_relations(schema, entity_map)
            relations.extend(foreign_key_relations)

            # Extract naming-based semantic relations
            semantic_relations = self._extract_semantic_relations(schema, entity_map)
            relations.extend(semantic_relations)

            return relations

        except Exception as e:
            logger.error("Error extracting database relations: %s", e)
            return []

    def _extract_database_table_relations(
        self, schema: Dict[str, Any], entity_map: Dict[str, Entity]
    ) -> List[Relation]:
        """
        Extract database-table containment relations.

        Args:
            schema: Database schema information
            entity_map: Mapping from entity names to entities

        Returns:
            List[Relation]: List of database-table relations
        """
        relations: List[Relation] = []

        database_name = schema.get("database_name")
        if not database_name or database_name not in entity_map:
            return relations

        database_entity = entity_map[database_name]
        tables = schema.get("tables", [])

        for table in tables:
            table_name = table.get("name")
            if table_name and table_name in entity_map:
                table_entity = entity_map[table_name]

                relation = Relation(
                    head_entity=database_entity,
                    tail_entity=table_entity,
                    relation_type=RelationType.CONTAINS,
                    confidence=1.0,
                    source="database_schema",
                    properties={
                        "schema_name": table.get("schema", ""),
                        "table_type": table.get("type", "table"),
                    },
                )
                relations.append(relation)

        return relations

    def _extract_table_column_relations(self, schema: Dict[str, Any], entity_map: Dict[str, Entity]) -> List[Relation]:
        """
        Extract table-column containment relations.

        Args:
            schema: Database schema information
            entity_map: Mapping from entity names to entities

        Returns:
            List[Relation]: List of table-column relations
        """
        relations: List[Relation] = []

        tables = schema.get("tables", [])
        for table in tables:
            table_name = table.get("name")
            if not table_name or table_name not in entity_map:
                continue

            table_entity = entity_map[table_name]
            columns = table.get("columns", [])

            for column in columns:
                column_name = column.get("name")
                if not column_name:
                    continue

                full_column_name = f"{table_name}.{column_name}"
                if full_column_name in entity_map:
                    column_entity = entity_map[full_column_name]

                    relation = Relation(
                        head_entity=table_entity,
                        tail_entity=column_entity,
                        relation_type=RelationType.CONTAINS,
                        confidence=1.0,
                        source="database_schema",
                        properties={
                            "column_position": column.get("position", 0),
                            "is_primary_key": column.get("primary_key", False),
                            "is_nullable": column.get("nullable", True),
                        },
                    )
                    relations.append(relation)

        return relations

    def _extract_foreign_key_relations(self, schema: Dict[str, Any], entity_map: Dict[str, Entity]) -> List[Relation]:
        """
        Extract foreign key relations between columns.

        Args:
            schema: Database schema information
            entity_map: Mapping from entity names to entities

        Returns:
            List[Relation]: List of foreign key relations
        """
        relations: List[Relation] = []

        tables = schema.get("tables", [])
        for table in tables:
            table_name = table.get("name")
            columns = table.get("columns", [])

            for column in columns:
                foreign_key = column.get("foreign_key")
                if not foreign_key:
                    continue

                source_column_name = f"{table_name}.{column.get('name')}"
                target_table = foreign_key.get("table")
                target_column = foreign_key.get("column")
                target_column_name = f"{target_table}.{target_column}"

                if source_column_name in entity_map and target_column_name in entity_map:

                    source_entity = entity_map[source_column_name]
                    target_entity = entity_map[target_column_name]

                    relation = Relation(
                        head_entity=source_entity,
                        tail_entity=target_entity,
                        relation_type=RelationType.FOREIGN_KEY,
                        confidence=1.0,
                        source="database_schema",
                        properties={
                            "constraint_name": foreign_key.get("constraint_name", ""),
                            "on_delete": foreign_key.get("on_delete", "RESTRICT"),
                            "on_update": foreign_key.get("on_update", "RESTRICT"),
                        },
                    )
                    relations.append(relation)

        return relations

    def _extract_semantic_relations(self, schema: Dict[str, Any], entity_map: Dict[str, Entity]) -> List[Relation]:
        """
        Extract semantic relations based on naming patterns.

        Args:
            schema: Database schema information
            entity_map: Mapping from entity names to entities

        Returns:
            List[Relation]: List of semantic relations
        """
        relations: List[Relation] = []

        # Table name similarity-based relations
        tables = schema.get("tables", [])
        table_entities = []

        for table in tables:
            table_name = table.get("name")
            if table_name and table_name in entity_map:
                table_entities.append(entity_map[table_name])

        # Find similar table names
        for i, table1 in enumerate(table_entities):
            for table2 in table_entities[i + 1 :]:
                similarity = self._calculate_name_similarity(table1.name, table2.name)
                if similarity > 0.6:
                    relation = Relation(
                        head_entity=table1,
                        tail_entity=table2,
                        relation_type=RelationType.SIMILAR_TO,
                        confidence=similarity,
                        source="name_similarity",
                        properties={"similarity_score": similarity},
                    )
                    relations.append(relation)

        return relations

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """
        Calculate name similarity using Jaccard similarity.

        Args:
            name1: First name to compare
            name2: Second name to compare

        Returns:
            float: Similarity score between 0.0 and 1.0
        """
        name1 = name1.lower()
        name2 = name2.lower()

        # Simple Jaccard similarity
        set1 = set(name1.split("_"))
        set2 = set(name2.split("_"))

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        if union == 0:
            return 0.0

        return intersection / union
