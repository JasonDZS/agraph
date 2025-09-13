"""
Custom exceptions for AGraph MCP Server
"""

class AGraphMCPError(Exception):
    """Base exception for AGraph MCP Server"""
    pass


class APIConnectionError(AGraphMCPError):
    """Raised when cannot connect to AGraph API"""
    pass


class APIRequestError(AGraphMCPError):
    """Raised when API request fails"""
    
    def __init__(self, message: str, status_code: int = None, response_text: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_text = response_text


class ConfigurationError(AGraphMCPError):
    """Raised when configuration is invalid"""
    pass


class ValidationError(AGraphMCPError):
    """Raised when request validation fails"""
    pass