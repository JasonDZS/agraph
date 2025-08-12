"""
Utility functions for the agraph package.

This module provides common utility functions used across the package.
"""

from typing import Union

from .types import EntityType, RelationType


def get_type_value(type_obj: Union[EntityType, RelationType, str]) -> str:
    """
    Get the string value from a type object.

    Args:
        type_obj: The type object to convert to string

    Returns:
        String representation of the type
    """
    if isinstance(type_obj, (EntityType, RelationType)):
        return str(type_obj.value)
    return str(type_obj)
