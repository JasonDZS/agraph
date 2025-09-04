"""
Test cases for the relations module.
"""

import unittest
import uuid
from datetime import datetime

from agraph.base.core.types import EntityType, RelationType
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation


class TestRelation(unittest.TestCase):
    """Test cases for the Relation class."""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures."""
        # Rebuild the Relation model to resolve forward references
        # Import Entity into current module namespace for model rebuild
        # Set Entity in the module globals so model_rebuild can find it
        import sys

        from agraph.base.models.entities import Entity

        current_module = sys.modules[__name__]
        current_module.Entity = Entity
        Relation.model_rebuild(force=True)

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.head_entity = Entity(
            name="Alice", entity_type=EntityType.PERSON, description="A person entity"
        )
        self.tail_entity = Entity(
            name="Acme Corp", entity_type=EntityType.ORGANIZATION, description="A company entity"
        )

    def test_relation_creation_with_defaults(self):
        """Test creating a relation with default values."""
        relation = Relation(head_entity=self.head_entity, tail_entity=self.tail_entity)

        self.assertEqual(relation.head_entity, self.head_entity)
        self.assertEqual(relation.tail_entity, self.tail_entity)
        self.assertEqual(relation.relation_type, RelationType.RELATED_TO)
        self.assertEqual(relation.description, "")
        self.assertEqual(relation.properties, {})
        self.assertEqual(relation.text_chunks, set())
        self.assertIsInstance(relation.id, str)
        self.assertIsInstance(relation.created_at, datetime)
        self.assertIsInstance(relation.updated_at, datetime)

    def test_relation_creation_with_custom_values(self):
        """Test creating a relation with custom values."""
        custom_id = str(uuid.uuid4())
        custom_properties = {"strength": 0.9, "category": "employment"}
        custom_text_chunks = {"chunk1", "chunk2"}
        custom_description = "Alice works for Acme Corp"

        relation = Relation(
            id=custom_id,
            head_entity=self.head_entity,
            tail_entity=self.tail_entity,
            relation_type=RelationType.WORKS_FOR,
            description=custom_description,
            properties=custom_properties,
            text_chunks=custom_text_chunks,
            confidence=0.95,
            source="document1",
        )

        self.assertEqual(relation.id, custom_id)
        self.assertEqual(relation.head_entity, self.head_entity)
        self.assertEqual(relation.tail_entity, self.tail_entity)
        self.assertEqual(relation.relation_type, "works_for")
        self.assertEqual(relation.description, custom_description)
        self.assertEqual(relation.properties, custom_properties)
        self.assertEqual(relation.text_chunks, custom_text_chunks)
        self.assertEqual(relation.confidence, 0.95)
        self.assertEqual(relation.source, "document1")

    def test_relation_validation_same_entities(self):
        """Test that relation validation fails when head and tail entities are the same."""
        with self.assertRaises(ValueError) as context:
            Relation(head_entity=self.head_entity, tail_entity=self.head_entity)
        self.assertIn("Head and tail entities must be different", str(context.exception))

    def test_relation_is_valid(self):
        """Test relation validity checking."""
        # Valid relation
        valid_relation = Relation(head_entity=self.head_entity, tail_entity=self.tail_entity)
        self.assertTrue(valid_relation.is_valid())

        # Invalid relation - no head entity
        invalid_relation1 = Relation(head_entity=None, tail_entity=self.tail_entity)
        self.assertFalse(invalid_relation1.is_valid())

        # Invalid relation - no tail entity
        invalid_relation2 = Relation(head_entity=self.head_entity, tail_entity=None)
        self.assertFalse(invalid_relation2.is_valid())

        # Invalid relation - no entities
        invalid_relation3 = Relation()
        self.assertFalse(invalid_relation3.is_valid())

    def test_relation_reverse(self):
        """Test creating a reverse relation."""
        original_relation = Relation(
            head_entity=self.head_entity,
            tail_entity=self.tail_entity,
            relation_type=RelationType.WORKS_FOR,
            description="Alice works for Acme Corp",
            properties={"duration": "2 years"},
            confidence=0.9,
            source="document1",
            text_chunks={"chunk1"},
        )

        reversed_relation = original_relation.reverse()

        # Check that entities are swapped
        self.assertEqual(reversed_relation.head_entity, self.tail_entity)
        self.assertEqual(reversed_relation.tail_entity, self.head_entity)

        # Check that relation type is appropriately reversed
        self.assertEqual(reversed_relation.relation_type, "contains")

        # Check that other properties are preserved
        self.assertEqual(reversed_relation.description, original_relation.description)
        self.assertEqual(reversed_relation.properties, original_relation.properties)
        self.assertEqual(reversed_relation.confidence, original_relation.confidence)
        self.assertEqual(reversed_relation.source, original_relation.source)
        self.assertEqual(reversed_relation.text_chunks, original_relation.text_chunks)

        # Check that IDs are different
        self.assertNotEqual(reversed_relation.id, original_relation.id)

    def test_reverse_relation_type_mapping(self):
        """Test that relation types are correctly mapped when reversed."""
        test_cases = [
            (RelationType.CONTAINS, RelationType.BELONGS_TO),
            (RelationType.BELONGS_TO, RelationType.CONTAINS),
            (RelationType.LOCATED_IN, RelationType.CONTAINS),
            (RelationType.WORKS_FOR, RelationType.CONTAINS),
            (RelationType.PART_OF, RelationType.CONTAINS),
            (RelationType.FOUNDED_BY, RelationType.CREATES),
            (RelationType.CREATES, RelationType.FOUNDED_BY),
            (RelationType.DEVELOPS, RelationType.FOUNDED_BY),
            # Symmetric relations
            (RelationType.REFERENCES, RelationType.REFERENCES),
            (RelationType.SIMILAR_TO, RelationType.SIMILAR_TO),
            (RelationType.SYNONYMS, RelationType.SYNONYMS),
            (RelationType.RELATED_TO, RelationType.RELATED_TO),
        ]

        for original_type, expected_reverse_type in test_cases:
            relation = Relation(
                head_entity=self.head_entity,
                tail_entity=self.tail_entity,
                relation_type=original_type,
            )
            reversed_relation = relation.reverse()
            self.assertEqual(
                reversed_relation.relation_type,
                expected_reverse_type.value,  # Compare with string value
                f"Failed for {original_type} -> {expected_reverse_type}",
            )

    def test_reverse_relation_type_with_string(self):
        """Test reverse relation type mapping with string input."""
        relation = Relation(
            head_entity=self.head_entity, tail_entity=self.tail_entity, relation_type="works_for"
        )
        reversed_relation = relation.reverse()
        self.assertEqual(reversed_relation.relation_type, "contains")

    def test_reverse_relation_type_with_unknown_string(self):
        """Test reverse relation type mapping with unknown string."""
        relation = Relation(
            head_entity=self.head_entity,
            tail_entity=self.tail_entity,
            relation_type="unknown_relation_type",
        )
        reversed_relation = relation.reverse()
        self.assertEqual(reversed_relation.relation_type, "unknown_relation_type")

    def test_to_dict(self):
        """Test converting relation to dictionary."""
        relation = Relation(
            head_entity=self.head_entity,
            tail_entity=self.tail_entity,
            relation_type=RelationType.WORKS_FOR,
            description="Employment relationship",
            properties={"salary": "50000"},
            confidence=0.85,
            source="HR database",
            text_chunks={"chunk1", "chunk2"},
        )

        relation_dict = relation.to_dict()

        expected_keys = {
            "id",
            "head_entity_id",
            "tail_entity_id",
            "relation_type",
            "properties",
            "confidence",
            "source",
            "description",
            "text_chunks",
            "created_at",
            "updated_at",
        }
        self.assertEqual(set(relation_dict.keys()), expected_keys)

        self.assertEqual(relation_dict["id"], relation.id)
        self.assertEqual(relation_dict["head_entity_id"], self.head_entity.id)
        self.assertEqual(relation_dict["tail_entity_id"], self.tail_entity.id)
        self.assertEqual(relation_dict["relation_type"], "works_for")
        self.assertEqual(relation_dict["properties"], {"salary": "50000"})
        self.assertEqual(relation_dict["confidence"], 0.85)
        self.assertEqual(relation_dict["source"], "HR database")
        self.assertEqual(relation_dict["description"], "Employment relationship")
        self.assertEqual(set(relation_dict["text_chunks"]), {"chunk1", "chunk2"})
        self.assertIsInstance(relation_dict["created_at"], str)
        self.assertIsInstance(relation_dict["updated_at"], str)

    def test_to_dict_with_none_entities(self):
        """Test converting relation to dictionary with None entities."""
        relation = Relation(
            head_entity=None, tail_entity=None, relation_type=RelationType.RELATED_TO
        )

        relation_dict = relation.to_dict()

        self.assertIsNone(relation_dict["head_entity_id"])
        self.assertIsNone(relation_dict["tail_entity_id"])

    def test_from_dict_with_entities_map(self):
        """Test creating relation from dictionary with entities map."""
        relation_data = {
            "id": str(uuid.uuid4()),
            "head_entity_id": self.head_entity.id,
            "tail_entity_id": self.tail_entity.id,
            "relation_type": "works_for",
            "properties": {"department": "Engineering"},
            "confidence": 0.9,
            "source": "document1",
            "description": "Employment relation",
            "text_chunks": ["chunk1", "chunk2"],
            "created_at": "2023-01-01T12:00:00",
            "updated_at": "2023-01-02T12:00:00",
        }

        entities_map = {
            self.head_entity.id: self.head_entity,
            self.tail_entity.id: self.tail_entity,
        }

        relation = Relation.from_dict(relation_data, entities_map=entities_map)

        self.assertEqual(relation.id, relation_data["id"])
        self.assertEqual(relation.head_entity, self.head_entity)
        self.assertEqual(relation.tail_entity, self.tail_entity)
        self.assertEqual(relation.relation_type, "works_for")
        self.assertEqual(relation.properties, {"department": "Engineering"})
        self.assertEqual(relation.confidence, 0.9)
        self.assertEqual(relation.source, "document1")
        self.assertEqual(relation.description, "Employment relation")
        self.assertEqual(relation.text_chunks, {"chunk1", "chunk2"})
        self.assertEqual(relation.created_at, datetime.fromisoformat("2023-01-01T12:00:00"))
        self.assertEqual(relation.updated_at, datetime.fromisoformat("2023-01-02T12:00:00"))

    def test_from_dict_without_entities_map(self):
        """Test creating relation from dictionary without entities map."""
        relation_data = {
            "head_entity_id": "entity1",
            "tail_entity_id": "entity2",
            "relation_type": "contains",
        }

        relation = Relation.from_dict(relation_data)

        self.assertIsNone(relation.head_entity)
        self.assertIsNone(relation.tail_entity)
        self.assertEqual(relation.relation_type, "contains")

    def test_from_dict_with_missing_entities(self):
        """Test creating relation from dictionary with missing entities in map."""
        relation_data = {
            "head_entity_id": "missing_entity",
            "tail_entity_id": self.tail_entity.id,
            "relation_type": "related_to",
        }

        entities_map = {self.tail_entity.id: self.tail_entity}

        relation = Relation.from_dict(relation_data, entities_map=entities_map)

        self.assertIsNone(relation.head_entity)
        self.assertEqual(relation.tail_entity, self.tail_entity)

    def test_from_dict_with_invalid_relation_type(self):
        """Test creating relation from dictionary with invalid relation type."""
        relation_data = {
            "relation_type": "invalid_relation_type",
        }

        relation = Relation.from_dict(relation_data)

        self.assertEqual(relation.relation_type, "related_to")

    def test_from_dict_with_defaults(self):
        """Test creating relation from dictionary with minimal data."""
        relation_data = {}

        relation = Relation.from_dict(relation_data)

        self.assertIsNone(relation.head_entity)
        self.assertIsNone(relation.tail_entity)
        self.assertEqual(relation.relation_type, "related_to")
        self.assertEqual(relation.properties, {})
        self.assertEqual(relation.confidence, 0.8)
        self.assertEqual(relation.source, "")
        self.assertEqual(relation.description, "")
        self.assertEqual(relation.text_chunks, set())

    def test_relation_equality(self):
        """Test relation equality comparison."""
        relation1 = Relation(
            head_entity=self.head_entity,
            tail_entity=self.tail_entity,
            relation_type=RelationType.WORKS_FOR,
        )
        relation2 = Relation(
            head_entity=self.head_entity,
            tail_entity=self.tail_entity,
            relation_type=RelationType.WORKS_FOR,
        )
        relation3 = Relation(
            id=relation1.id,
            head_entity=self.head_entity,
            tail_entity=self.tail_entity,
            relation_type=RelationType.WORKS_FOR,
        )

        # Different relations should not be equal (different IDs)
        self.assertNotEqual(relation1, relation2)

        # Relations with same ID should be equal
        self.assertEqual(relation1, relation3)

    def test_relation_string_representation(self):
        """Test string representation of relation."""
        relation = Relation(
            head_entity=self.head_entity,
            tail_entity=self.tail_entity,
            relation_type=RelationType.WORKS_FOR,
            description="Employment relationship",
        )
        str_repr = str(relation)

        self.assertIn("Alice", str_repr)
        self.assertIn("Acme Corp", str_repr)
        self.assertIn("works_for", str_repr)

    def test_relation_update_timestamp(self):
        """Test that updated_at changes when relation is modified."""
        relation = Relation(head_entity=self.head_entity, tail_entity=self.tail_entity)
        original_updated = relation.updated_at

        # Simulate time passing and update
        relation.description = "Updated description"
        relation.updated_at = datetime.now()

        self.assertGreater(relation.updated_at, original_updated)


if __name__ == "__main__":
    unittest.main()
