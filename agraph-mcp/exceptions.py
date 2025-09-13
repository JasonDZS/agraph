#!/usr/bin/env python3
"""
Custom exceptions for AGraph MCP Server.
"""

from typing import Optional, Dict, Any


class AGrapeMCPError(Exception):
    """Base exception for AGraph MCP Server."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or self.__class__.__name__
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary."""
        return {
            "error": self.error_code,
            "message": self.message,
            "details": self.details
        }


class ProjectNotFoundError(AGrapeMCPError):
    """Raised when a project cannot be found."""
    
    def __init__(self, project_name: str):
        super().__init__(
            f"Project '{project_name}' not found",
            error_code="PROJECT_NOT_FOUND",
            details={"project_name": project_name}
        )


class AGraphInitializationError(AGrapeMCPError):
    """Raised when AGraph initialization fails."""
    
    def __init__(self, project_name: str, reason: str):
        super().__init__(
            f"Failed to initialize AGraph for project '{project_name}': {reason}",
            error_code="AGRAPH_INIT_FAILED",
            details={"project_name": project_name, "reason": reason}
        )


class SearchError(AGrapeMCPError):
    """Raised when search operations fail."""
    
    def __init__(self, search_type: str, query: str, reason: str):
        super().__init__(
            f"Search failed for type '{search_type}' with query '{query}': {reason}",
            error_code="SEARCH_FAILED",
            details={"search_type": search_type, "query": query, "reason": reason}
        )




class ConfigurationError(AGrapeMCPError):
    """Raised when configuration is invalid."""
    
    def __init__(self, config_issue: str):
        super().__init__(
            f"Configuration error: {config_issue}",
            error_code="CONFIG_ERROR",
            details={"issue": config_issue}
        )


class ResourceLimitError(AGrapeMCPError):
    """Raised when resource limits are exceeded."""
    
    def __init__(self, resource_type: str, limit: int, requested: int):
        super().__init__(
            f"Resource limit exceeded for {resource_type}: requested {requested}, limit {limit}",
            error_code="RESOURCE_LIMIT_EXCEEDED",
            details={"resource_type": resource_type, "limit": limit, "requested": requested}
        )


def handle_agraph_error(error: Exception, operation: str, context: Dict[str, Any] = None) -> AGrapeMCPError:
    """Convert generic exceptions to AGraph MCP errors."""
    context = context or {}
    
    if isinstance(error, AGrapeMCPError):
        return error
    
    # Map common error types
    error_message = str(error)
    
    if "import" in error_message.lower() or "module" in error_message.lower():
        return ConfigurationError(f"Module import failed during {operation}: {error_message}")
    
    if "permission" in error_message.lower():
        return ConfigurationError(f"Permission denied during {operation}: {error_message}")
    
    if "timeout" in error_message.lower():
        return SearchError(operation, context.get("query", "unknown"), f"Operation timed out: {error_message}")
    
    if "memory" in error_message.lower():
        return ResourceLimitError("memory", 0, 0)  # Could be enhanced with actual limits
    
    # Handle ChromaDB specific errors
    if "chromadb" in error_message.lower() or "chroma" in error_message.lower():
        if "InvalidCollectionException" in error_message:
            return SearchError(operation, context.get("query", "unknown"), "Collection not found or invalid")
        elif "collection" in error_message.lower() and "not" in error_message.lower():
            return SearchError(operation, context.get("query", "unknown"), "Collection not found")
        else:
            return ConfigurationError(f"ChromaDB error during {operation}: {error_message}")
    
    # Handle AGraph initialization errors
    if "agraph" in error_message.lower() and ("init" in error_message.lower() or "initialize" in error_message.lower()):
        return AGraphInitializationError(context.get("project", "unknown"), error_message)
    
    # Default to generic error
    return AGrapeMCPError(
        f"Unexpected error during {operation}: {error_message}",
        error_code="UNEXPECTED_ERROR",
        details={"operation": operation, "original_error": type(error).__name__, **context}
    )