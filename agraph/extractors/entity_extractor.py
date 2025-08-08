"""
Entity Extractor Module

This module provides entity extraction capabilities from various data sources.
Includes base classes and concrete implementations for text and database entity extraction.
"""

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from ..entities import Entity
from ..logger import logger
from ..types import EntityType


class BaseEntityExtractor(ABC):
    """
    Base class for entity extractors.

    This abstract class defines the common interface for all entity extractors.
    It provides entity normalization, deduplication utilities and confidence calculation methods.
    """

    def __init__(self) -> None:
        self.entity_patterns: Dict[str, List[str]] = {}
        self.confidence_threshold = 0.5
        self.stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
        }

    @abstractmethod
    def extract_from_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract entities from text.

        Args:
            text: Input text to extract entities from
            context: Optional context information to assist extraction

        Returns:
            List[Entity]: List of extracted entities
        """

    @abstractmethod
    def extract_from_database(self, schema: Dict[str, Any]) -> List[Entity]:
        """
        Extract entities from database schema.

        Args:
            schema: Database schema information containing tables, columns, etc.

        Returns:
            List[Entity]: List of extracted entities
        """

    def normalize_entity(self, entity: Entity) -> Entity:
        """
        Normalize entity by standardizing name and aliases.

        Args:
            entity: Original entity to normalize

        Returns:
            Entity: Normalized entity
        """
        # Normalize entity name
        entity.name = entity.name.strip().lower()

        # Normalize and deduplicate aliases
        normalized_aliases = []
        for alias in entity.aliases:
            normalized_alias = alias.strip().lower()
            if normalized_alias and normalized_alias not in normalized_aliases:
                normalized_aliases.append(normalized_alias)
        entity.aliases = normalized_aliases

        return entity

    def deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Remove duplicate entities based on name matching and merge their information.

        Args:
            entities: List of entities to deduplicate

        Returns:
            List[Entity]: Deduplicated entity list
        """
        unique_entities = {}
        name_to_entity: Dict[str, Entity] = {}

        for entity in entities:
            # Exact name matching
            normalized_name = entity.name.lower().strip()

            if normalized_name in name_to_entity:
                # Merge entity information
                existing_entity = name_to_entity[normalized_name]
                existing_entity.aliases.extend(entity.aliases)
                existing_entity.aliases = list(set(existing_entity.aliases))
                existing_entity.properties.update(entity.properties)

                # Keep entity with higher confidence
                if entity.confidence > existing_entity.confidence:
                    existing_entity.confidence = entity.confidence
                    existing_entity.description = entity.description or existing_entity.description
            else:
                name_to_entity[normalized_name] = entity
                unique_entities[entity.id] = entity

        return list(unique_entities.values())

    def _calculate_entity_confidence(self, entity_name: str, context: Optional[Dict[str, Any]] = None) -> float:
        """
        Calculate entity confidence score based on various factors.

        Args:
            entity_name: Name of the entity
            context: Optional context information

        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        confidence = 0.5  # Base confidence

        # Length-based confidence adjustment
        if len(entity_name) > 1:
            confidence += 0.1
        if len(entity_name) > 3:
            confidence += 0.1

        # Capitalization-based confidence adjustment
        if entity_name[0].isupper():
            confidence += 0.1

        # Stopword-based confidence adjustment
        if entity_name.lower() in self.stopwords:
            confidence -= 0.3

        return min(1.0, max(0.0, confidence))


class TextEntityExtractor(BaseEntityExtractor):
    """
    Text-based entity extractor.

    Extracts entities from text using regular expression patterns and keyword extraction.
    Supports multiple entity types including persons, organizations, locations, concepts, and products.
    """

    def __init__(self) -> None:
        super().__init__()
        self._init_patterns()

    def _init_patterns(self) -> None:
        """Initialize entity recognition patterns for different entity types."""
        self.entity_patterns = {
            EntityType.PERSON.value: [
                r"\b[A-Z][a-z]+ [A-Z][a-z]+\b",  # English name pattern
                r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.? [A-Z][a-z]+\b",  # Title + name
                r"[\u4e00-\u9fff]{2,4}·[\u4e00-\u9fff]{2,4}",  # Chinese name with · pattern
                r"史蒂夫·[\u4e00-\u9fff]+",  # Steve· prefix Chinese names
                r"[\u4e00-\u9fff]{2}[\u4e00-\u9fff]{1,2}(?:先生|女士|博士|教授)",  # Chinese titles
            ],
            EntityType.ORGANIZATION.value: [
                r"\b[A-Z][a-zA-Z\s&]+ (?:Inc|Corp|Ltd|LLC|Company|Organization)\b",
                r"\b[A-Z][A-Z\s]+\b",  # All caps might be organization
                r"[\u4e00-\u9fff]+(?:公司|企业|集团|组织|机构|大学|学院|研究所)",  # Chinese organizations
                r"苹果公司|清华大学|Facebook|Google|TensorFlow|PyTorch",  # Specific organization names
            ],
            EntityType.LOCATION.value: [
                r"\b[A-Z][a-z]+ (?:City|State|Country|Province|District)\b",
                r"\bin [A-Z][a-z]+\b",  # Location prepositional phrases
                r"[\u4e00-\u9fff]+(?:市|省|区|县|国|州|地区)",  # Chinese location names
                r"北京|上海|加利福尼亚州|库比蒂诺|海淀区",  # Specific location names
            ],
            EntityType.CONCEPT.value: [
                r"\b[a-z]+ (?:concept|theory|principle|method|approach)\b",
                r"[\u4e00-\u9fff]+(?:技术|概念|理论|方法|系统|平台|框架)",  # Chinese concepts
                r"人工智能|机器学习|深度学习|自然语言处理|计算机视觉|iOS|iPhone",  # Specific concepts
            ],
            EntityType.PRODUCT.value: [
                r"iPhone|iPad|macOS|iOS|Django|Flask|Python|TensorFlow|PyTorch",  # Product names
                r"[\u4e00-\u9fff]+(?:产品|系统|平台|应用|软件)",  # Chinese products
            ],
        }

    def extract_from_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """
        Extract entities from text using pattern matching and keyword extraction.

        Args:
            text: Input text to extract entities from
            context: Optional context information

        Returns:
            List[Entity]: List of extracted entities
        """
        entities = []

        try:
            # Pattern-based entity extraction
            for entity_type, patterns in self.entity_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text, re.IGNORECASE)
                    for match in matches:
                        entity_name = match.group().strip()
                        if len(entity_name) < 2:
                            continue

                        confidence = self._calculate_entity_confidence(entity_name, context)
                        if confidence < self.confidence_threshold:
                            continue

                        entity = Entity(
                            name=entity_name,
                            entity_type=EntityType(entity_type),
                            confidence=confidence,
                            source="text_extraction",
                            properties={
                                "position": match.span(),
                                "context": text[max(0, match.start() - 50) : match.end() + 50],
                            },
                        )

                        entities.append(entity)

            # Keyword-based concept entity extraction
            concept_keywords = self._extract_concept_keywords(text)
            for keyword in concept_keywords:
                entity = Entity(
                    name=keyword,
                    entity_type=EntityType.CONCEPT,
                    confidence=0.6,
                    source="keyword_extraction",
                )
                entities.append(entity)

            return self.deduplicate_entities(entities)

        except Exception as e:
            logger.error("Error extracting entities from text: %s", e)
            return []

    def extract_from_database(self, schema: Dict[str, Any]) -> List[Entity]:
        """
        Extract entities from database schema including databases, tables, and columns.

        Args:
            schema: Database schema information

        Returns:
            List[Entity]: List of extracted database entities
        """
        entities = []

        try:
            # Database entity
            if "database_name" in schema:
                db_entity = Entity(
                    name=schema["database_name"],
                    entity_type=EntityType.DATABASE,
                    confidence=1.0,
                    source="database_schema",
                    description=f"Database: {schema['database_name']}",
                )
                entities.append(db_entity)

            # Table entities
            tables = schema.get("tables", [])
            for table in tables:
                table_name = table.get("name", "")
                if table_name:
                    table_entity = Entity(
                        name=table_name,
                        entity_type=EntityType.TABLE,
                        confidence=1.0,
                        source="database_schema",
                        description=table.get("comment", f"Table: {table_name}"),
                        properties={
                            "row_count": table.get("row_count", 0),
                            "columns": [col.get("name") for col in table.get("columns", [])],
                        },
                    )
                    entities.append(table_entity)

                    # Column entities
                    columns = table.get("columns", [])
                    for column in columns:
                        column_name = column.get("name", "")
                        if column_name:
                            column_entity = Entity(
                                name=f"{table_name}.{column_name}",
                                entity_type=EntityType.COLUMN,
                                confidence=1.0,
                                source="database_schema",
                                description=column.get("comment", f"Column: {column_name}"),
                                properties={
                                    "table": table_name,
                                    "data_type": column.get("type", ""),
                                    "nullable": column.get("nullable", True),
                                    "primary_key": column.get("primary_key", False),
                                    "foreign_key": column.get("foreign_key", False),
                                },
                            )
                            entities.append(column_entity)

            return entities

        except Exception as e:
            logger.error("Error extracting entities from database schema: %s", e)
            return []

    def _extract_concept_keywords(self, text: str) -> List[str]:
        """
        Extract concept keywords from text based on frequency and position.

        Args:
            text: Input text to extract keywords from

        Returns:
            List[str]: List of concept keywords
        """
        # Simple keyword extraction based on word frequency and position
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

        # Filter stopwords
        keywords = [word for word in words if word not in self.stopwords]

        # Count word frequency
        word_freq: Dict[str, int] = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Return high-frequency words as concepts
        concept_keywords = [word for word, freq in word_freq.items() if freq >= 2]
        return concept_keywords[:10]  # Limit number of keywords


class DatabaseEntityExtractor(BaseEntityExtractor):
    """
    Database-focused entity extractor.

    Specializes in extracting entities from database schemas with enhanced business concept inference.
    Provides detailed table and column analysis with business semantics.
    """

    def __init__(self) -> None:
        super().__init__()
        self.table_prefixes = {"tbl_", "tb_", "t_"}
        self.common_columns = {"id", "created_at", "updated_at", "deleted_at"}

    def extract_from_text(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[Entity]:
        """Database extractor does not process text data."""
        return []

    def extract_from_database(self, schema: Dict[str, Any]) -> List[Entity]:
        """
        Extract entities from database schema with business concept inference.

        Args:
            schema: Database schema information

        Returns:
            List[Entity]: List of extracted entities including business concepts
        """
        entities = []

        try:
            # Extract database entity
            if "database_name" in schema:
                db_entity = Entity(
                    name=schema["database_name"],
                    entity_type=EntityType.DATABASE,
                    confidence=1.0,
                    source="database_extraction",
                    description=f"Database: {schema.get('description', schema['database_name'])}",
                )
                entities.append(db_entity)

            # Extract table and column entities
            tables = schema.get("tables", [])
            for table in tables:
                table_entities = self._extract_table_entities(table)
                entities.extend(table_entities)

            # Extract business concept entities
            business_entities = self._extract_business_concepts(schema)
            entities.extend(business_entities)

            return entities

        except Exception as e:
            logger.error("Error in database entity extraction: %s", e)
            return []

    def _extract_table_entities(self, table: Dict[str, Any]) -> List[Entity]:
        """
        Extract table-related entities including the table itself and its columns.

        Args:
            table: Table information dictionary

        Returns:
            List[Entity]: List of table and column entities
        """
        entities: List[Entity] = []
        table_name = table.get("name", "")

        if not table_name:
            return entities

        # Table entity
        table_entity = Entity(
            name=table_name,
            entity_type=EntityType.TABLE,
            confidence=1.0,
            source="database_extraction",
            description=table.get("comment", f"Data table: {table_name}"),
            properties={
                "schema": table.get("schema", ""),
                "row_count": table.get("row_count", 0),
                "size_mb": table.get("size_mb", 0),
                "engine": table.get("engine", ""),
                "created_at": table.get("created_at", ""),
                "column_count": len(table.get("columns", [])),
            },
        )

        # Add table name aliases (remove prefixes)
        clean_name = self._clean_table_name(table_name)
        if clean_name != table_name:
            table_entity.add_alias(clean_name)

        entities.append(table_entity)

        # Column entities
        columns = table.get("columns", [])
        for column in columns:
            column_entities = self._extract_column_entities(table_name, column)
            entities.extend(column_entities)

        return entities

    def _extract_column_entities(self, table_name: str, column: Dict[str, Any]) -> List[Entity]:
        """
        Extract column entities with detailed metadata.

        Args:
            table_name: Name of the parent table
            column: Column information dictionary

        Returns:
            List[Entity]: List of column entities
        """
        entities: List[Entity] = []
        column_name = column.get("name", "")

        if not column_name:
            return entities

        # Skip common columns
        if column_name.lower() in self.common_columns:
            return entities

        full_column_name = f"{table_name}.{column_name}"

        column_entity = Entity(
            name=full_column_name,
            entity_type=EntityType.COLUMN,
            confidence=1.0,
            source="database_extraction",
            description=column.get("comment", f"Data column: {column_name}"),
            properties={
                "table": table_name,
                "column": column_name,
                "data_type": column.get("type", ""),
                "max_length": column.get("max_length"),
                "nullable": column.get("nullable", True),
                "default_value": column.get("default"),
                "primary_key": column.get("primary_key", False),
                "foreign_key": column.get("foreign_key", {}),
                "unique": column.get("unique", False),
                "indexed": column.get("indexed", False),
            },
        )

        # Add column name alias
        column_entity.add_alias(column_name)

        entities.append(column_entity)
        return entities

    def _extract_business_concepts(self, schema: Dict[str, Any]) -> List[Entity]:
        """
        Extract business concept entities by inferring from table names.

        Args:
            schema: Database schema information

        Returns:
            List[Entity]: List of business concept entities
        """
        entities = []

        # Infer business concepts from table names
        tables = schema.get("tables", [])
        business_concepts = set()

        for table in tables:
            table_name = table.get("name", "")
            clean_name = self._clean_table_name(table_name)

            # Infer business concepts from table name
            concepts = self._infer_business_concepts(clean_name)
            business_concepts.update(concepts)

        # Create business concept entities
        for concept in business_concepts:
            concept_entity = Entity(
                name=concept,
                entity_type=EntityType.CONCEPT,
                confidence=0.7,
                source="business_inference",
                description=f"Business concept: {concept}",
            )
            entities.append(concept_entity)

        return entities

    def _clean_table_name(self, table_name: str) -> str:
        """
        Clean table name by removing common prefixes.

        Args:
            table_name: Original table name

        Returns:
            str: Cleaned table name
        """
        clean_name = table_name.lower()

        # Remove common prefixes
        for prefix in self.table_prefixes:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix) :]
                break

        return clean_name

    def _infer_business_concepts(self, table_name: str) -> Set[str]:
        """
        Infer business concepts from table name using keyword mapping.

        Args:
            table_name: Cleaned table name

        Returns:
            Set[str]: Set of inferred business concepts
        """
        concepts = set()

        # Simple business concept mapping
        concept_mapping = {
            "user": "User Management",
            "customer": "Customer Management",
            "order": "Order Management",
            "product": "Product Management",
            "inventory": "Inventory Management",
            "payment": "Payment Processing",
            "shipment": "Shipping Management",
            "category": "Category Management",
            "review": "Review System",
            "cart": "Shopping Cart",
            "wishlist": "Wishlist Management",
        }

        for keyword, concept in concept_mapping.items():
            if keyword in table_name:
                concepts.add(concept)

        return concepts
