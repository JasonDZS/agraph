"""
Test cases for the types module.
"""

import unittest

from agraph.base.types import EntityType, RelationType


class TestEntityType(unittest.TestCase):
    """Test cases for the EntityType enum."""

    def test_entity_type_values(self):
        """Test that EntityType enum has expected values."""
        self.assertEqual(EntityType.PERSON.value, "person")
        self.assertEqual(EntityType.ORGANIZATION.value, "organization")
        self.assertEqual(EntityType.LOCATION.value, "location")
        self.assertEqual(EntityType.CONCEPT.value, "concept")
        self.assertEqual(EntityType.EVENT.value, "event")
        self.assertEqual(EntityType.OTHER.value, "other")
        self.assertEqual(EntityType.UNKNOWN.value, "unknown")

    def test_entity_type_string_conversion(self):
        """Test string conversion of EntityType."""
        self.assertEqual(str(EntityType.PERSON), "EntityType.PERSON")

    def test_entity_type_from_string(self):
        """Test creating EntityType from string value."""
        person_type = EntityType("person")
        self.assertEqual(person_type, EntityType.PERSON)


class TestRelationType(unittest.TestCase):
    """Test cases for the RelationType enum."""

    def test_relation_type_basic_values(self):
        """Test basic RelationType enum values."""
        # Test some expected relation types
        relation_types = list(RelationType)
        self.assertGreater(len(relation_types), 0)

        # Each relation type should have a string value
        for rel_type in relation_types:
            self.assertIsInstance(rel_type.value, str)
            self.assertGreater(len(rel_type.value), 0)


if __name__ == "__main__":
    unittest.main()
