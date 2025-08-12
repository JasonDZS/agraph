"""
Utility functions for the agraph package.

This module provides common utility functions used across the package.
"""

from typing import Union

from agraph.base.types import ClusterType, EntityType, RelationType


def get_type_value(type_obj: Union[EntityType, RelationType, ClusterType, str]) -> str:
    """
    Get the string value from a type object.

    Args:
        type_obj: The type object to convert to string

    Returns:
        String representation of the type
    """
    if isinstance(type_obj, (EntityType, RelationType, ClusterType)):
        return str(type_obj.value)
    return str(type_obj)
