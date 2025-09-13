"""
Configuration management for AGraph MCP Server
"""

import os
from dataclasses import dataclass

@dataclass
class AGraphConfig:
    """Configuration for AGraph MCP Server"""
    
    # AGraph Configuration
    workdir: str = "agraph_workspace"
    default_collection: str = "agraph_knowledge"
    persist_directory: str = "agraph_workspace"
    vector_store_type: str = "chroma"
    
    # Search Configuration
    default_entity_limit: int = 50
    default_relation_limit: int = 30
    default_text_chunk_limit: int = 20
    default_cluster_limit: int = 15
    max_content_preview_length: int = 200
    
    # Natural Language Search Configuration
    natural_search_default_limit: int = 10
    
    @classmethod
    def from_env(cls) -> 'AGraphConfig':
        """Create configuration from environment variables"""
        return cls(
            workdir=os.getenv("AGRAPH_WORKDIR", "agraph_workspace"),
            default_collection=os.getenv("AGRAPH_DEFAULT_COLLECTION", "agraph_knowledge"),
            persist_directory=os.getenv("AGRAPH_PERSIST_DIRECTORY", "agraph_workspace"),
            vector_store_type=os.getenv("AGRAPH_VECTOR_STORE_TYPE", "chroma"),
            
            default_entity_limit=int(os.getenv("AGRAPH_DEFAULT_ENTITY_LIMIT", "50")),
            default_relation_limit=int(os.getenv("AGRAPH_DEFAULT_RELATION_LIMIT", "30")),
            default_text_chunk_limit=int(os.getenv("AGRAPH_DEFAULT_TEXT_CHUNK_LIMIT", "20")),
            default_cluster_limit=int(os.getenv("AGRAPH_DEFAULT_CLUSTER_LIMIT", "15")),
            max_content_preview_length=int(os.getenv("AGRAPH_MAX_CONTENT_PREVIEW", "200")),
            
            natural_search_default_limit=int(os.getenv("AGRAPH_NATURAL_SEARCH_LIMIT", "10")),
        )
    
    def validate(self) -> None:
        """Validate configuration values"""
        if not self.workdir:
            raise ValueError("Workdir cannot be empty")
        
        if not self.default_collection:
            raise ValueError("Default collection name cannot be empty")
        
        if not self.persist_directory:
            raise ValueError("Persist directory cannot be empty")
        
        if self.default_entity_limit <= 0:
            raise ValueError("Default entity limit must be positive")
        
        if self.default_relation_limit <= 0:
            raise ValueError("Default relation limit must be positive")


# Global configuration instance
config = AGraphConfig.from_env()