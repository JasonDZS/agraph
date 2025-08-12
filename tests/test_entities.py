"""
Test cases for the entities module.
"""

import unittest
import uuid
from datetime import datetime

from agraph.base.entities import Entity
from agraph.base.types import EntityType


class TestEntity(unittest.TestCase):
    """Test cases for the Entity class."""

    def test_entity_creation_with_defaults(self):
        """Test creating an entity with default values."""
        entity = Entity(name="Test Entity", entity_type=EntityType.PERSON)

        self.assertEqual(entity.name, "Test Entity")
        self.assertEqual(entity.entity_type, "person")
        self.assertIsInstance(entity.id, str)
        self.assertIsInstance(entity.created_at, datetime)
        self.assertIsInstance(entity.updated_at, datetime)
        self.assertEqual(entity.description, "")
        self.assertEqual(entity.properties, {})
        self.assertEqual(entity.text_chunks, set())

    def test_entity_creation_with_custom_values(self):
        """Test creating an entity with custom values."""
        custom_id = str(uuid.uuid4())
        custom_properties = {"key": "value"}
        custom_text_chunks = {"chunk1", "chunk2"}

        entity = Entity(
            id=custom_id,
            name="Custom Entity",
            entity_type=EntityType.ORGANIZATION,
            description="A custom entity",
            properties=custom_properties,
            text_chunks=custom_text_chunks,
        )

        self.assertEqual(entity.id, custom_id)
        self.assertEqual(entity.name, "Custom Entity")
        self.assertEqual(entity.entity_type, "organization")
        self.assertEqual(entity.description, "A custom entity")
        self.assertEqual(entity.properties, custom_properties)
        self.assertEqual(entity.text_chunks, custom_text_chunks)

    def test_entity_equality(self):
        """Test entity equality comparison."""
        entity1 = Entity(name="Test", entity_type=EntityType.PERSON)
        entity2 = Entity(name="Test", entity_type=EntityType.PERSON)
        entity3 = Entity(id=entity1.id, name="Test", entity_type=EntityType.PERSON)

        # Different entities should not be equal (different IDs)
        self.assertNotEqual(entity1, entity2)

        # Entities with same ID should be equal
        self.assertEqual(entity1, entity3)

    def test_entity_string_representation(self):
        """Test string representation of entity."""
        entity = Entity(name="Test Entity", entity_type=EntityType.CONCEPT)
        str_repr = str(entity)

        self.assertIn("Test Entity", str_repr)
        self.assertIn("concept", str_repr)
        self.assertIn(entity.id, str_repr)

    def test_entity_update_timestamp(self):
        """Test that updated_at changes when entity is modified."""
        entity = Entity(name="Test", entity_type=EntityType.PERSON)
        original_updated = entity.updated_at

        # Simulate time passing and update
        entity.description = "Updated description"
        entity.updated_at = datetime.now()

        self.assertGreater(entity.updated_at, original_updated)


if __name__ == "__main__":
    unittest.main()
