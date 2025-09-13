#!/usr/bin/env python3
"""
Configuration management for AGraph MCP Server.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel


class MCPServerConfig(BaseModel):
    """Configuration for MCP Server."""
    server_name: str = "AGraph Semantic Search"
    max_concurrent_projects: int = 10
    default_top_k: int = 5
    enable_caching: bool = True
    cache_ttl: int = 3600  # 1 hour
    log_level: str = "INFO"


class AGraphConfig(BaseModel):
    """Configuration for AGraph instances."""
    default_model: str = "Qwen/Qwen2.5-32B-Instruct"
    project_dir: str = "../workdir"  # Base directory for projects
    chunk_size: int = 1000
    chunk_overlap: int = 200
    entity_confidence_threshold: float = 0.7
    relation_confidence_threshold: float = 0.6
    max_current: int = 5


@dataclass
class ServerEnvironment:
    """Environment configuration."""
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))
    openai_base_url: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_BASE_URL'))
    huggingface_api_key: Optional[str] = field(default_factory=lambda: os.getenv('HUGGINGFACE_API_KEY'))
    project_dir: str = field(default_factory=lambda: os.getenv('PROJECT_DIR', '../workdir'))
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'False').lower() == 'true')
    
    def __post_init__(self):
        """Validate environment configuration."""
        if not any([self.openai_api_key, self.huggingface_api_key]):
            print("Warning: No API keys found. Please set OPENAI_API_KEY or HUGGINGFACE_API_KEY")


class ConfigManager:
    """Manages configuration for AGraph MCP Server."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.cwd() / "agraph_mcp_config.json"
        self.mcp_config = MCPServerConfig()
        self.agraph_config = AGraphConfig()
        self.env = ServerEnvironment()
        
        # Use environment PROJECT_DIR if available
        if self.env.project_dir:
            self.agraph_config.project_dir = self.env.project_dir
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_path.exists():
            try:
                import json
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # Update configurations
                if 'mcp_server' in config_data:
                    self.mcp_config = MCPServerConfig(**config_data['mcp_server'])
                if 'agraph' in config_data:
                    self.agraph_config = AGraphConfig(**config_data['agraph'])
                    
                print(f"✅ Configuration loaded from: {self.config_path}")
            except Exception as e:
                print(f"⚠️ Error loading config: {e}. Using defaults.")
        else:
            print(f"📋 Using default configuration. To customize, create: {self.config_path}")
        
        return self.get_merged_config()
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            import json
            config_data = {
                'mcp_server': self.mcp_config.model_dump(),
                'agraph': self.agraph_config.model_dump()
            }
            
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Configuration saved to: {self.config_path}")
        except Exception as e:
            print(f"⚠️ Error saving config: {e}")
    
    def get_merged_config(self) -> Dict[str, Any]:
        """Get merged configuration from all sources."""
        return {
            'mcp_server': self.mcp_config.model_dump(),
            'agraph': self.agraph_config.model_dump(),
            'environment': {
                'has_openai_key': bool(self.env.openai_api_key),
                'has_huggingface_key': bool(self.env.huggingface_api_key),
                'openai_base_url': self.env.openai_base_url,
                'debug': self.env.debug
            }
        }
    
    def get_project_path(self, project: str) -> Path:
        """Get the full path for a project."""
        project_base = Path(self.agraph_config.project_dir)
        return project_base / project
    
    def get_agraph_settings_override(self, project: str) -> Dict[str, Any]:
        """Get AGraph settings override dictionary for a specific project."""
        project_path = self.get_project_path(project)
        project_path.mkdir(parents=True, exist_ok=True)
        
        return {
            "workdir": str(project_path),
            "llm_config": {
                "model": self.agraph_config.default_model
            },
            "processing_config": {
                "chunk_size": self.agraph_config.chunk_size,
                "chunk_overlap": self.agraph_config.chunk_overlap
            },
            "extraction_config": {
                "entity_confidence_threshold": self.agraph_config.entity_confidence_threshold,
                "relation_confidence_threshold": self.agraph_config.relation_confidence_threshold
            },
            "max_current": self.agraph_config.max_current
        }
    
    def update_mcp_config(self, **kwargs) -> None:
        """Update MCP server configuration."""
        for key, value in kwargs.items():
            if hasattr(self.mcp_config, key):
                setattr(self.mcp_config, key, value)
    
    def update_agraph_config(self, **kwargs) -> None:
        """Update AGraph configuration.""" 
        for key, value in kwargs.items():
            if hasattr(self.agraph_config, key):
                setattr(self.agraph_config, key, value)


# Global configuration manager instance
config_manager = ConfigManager()


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager."""
    return config_manager


def load_server_config() -> Dict[str, Any]:
    """Load and return server configuration."""
    return config_manager.load_config()


def get_agraph_settings(project: str) -> Dict[str, Any]:
    """Get AGraph settings override for a specific project."""
    return config_manager.get_agraph_settings_override(project)