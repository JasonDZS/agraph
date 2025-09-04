"""
Utility module containing helper functions and decorators.

This module provides common utilities:
- Deprecation management for backward compatibility
- Helper functions and decorators
"""

from .deprecation import deprecated

__all__ = [
    "deprecated",
]
