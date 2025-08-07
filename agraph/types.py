"""
Graph type definitions and enumerations.

This module provides enum definitions for entity and relation types.
"""

from enum import Enum
from typing import Union


class EntityType(Enum):
    """Entity type enumeration."""

    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    OTHER = "other"
    TABLE = "table"
    COLUMN = "column"
    DATABASE = "database"
    DOCUMENT = "document"
    KEYWORD = "keyword"
    PRODUCT = "product"
    SOFTWARE = "software"
    UNKNOWN = "unknown"


class RelationType(Enum):
    """Relation type enumeration."""

    CONTAINS = "contains"
    BELONGS_TO = "belongs_to"
    LOCATED_IN = "located_in"
    WORKS_FOR = "works_for"
    CAUSES = "causes"
    PART_OF = "part_of"
    IS_A = "is_a"
    REFERENCES = "references"
    SIMILAR_TO = "similar_to"
    RELATED_TO = "related_to"
    DEPENDS_ON = "depends_on"
    FOREIGN_KEY = "foreign_key"
    MENTIONS = "mentions"
    DESCRIBES = "describes"
    SYNONYMS = "synonyms"
    DEVELOPS = "develops"
    CREATES = "creates"
    FOUNDED_BY = "founded_by"
    OTHER = "other"


# Type aliases for type annotations
EntityTypeType = Union[EntityType, str]
RelationTypeType = Union[RelationType, str]
