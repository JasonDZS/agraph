#!/usr/bin/env python3
"""
AGraph Knowledge Graph MCP Server

This server provides access to AGraph knowledge graph functionalities through MCP.
It directly uses AGraph instances for local knowledge graph operations and supports
natural language queries for entities, relations, clusters, and text chunks.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mcp.server import Server
from mcp.types import Tool, TextContent

# Add the parent directory to the path to import agraph
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

try:
    from agraph import AGraph, Settings, get_settings
    from agraph.base.core.types import EntityType, RelationType, ClusterType
    from agraph.base.models.entities import Entity
    from agraph.base.models.relations import Relation
    from agraph.base.models.clusters import Cluster
    from agraph.base.models.text import TextChunk
except ImportError as e:
    logging.error(f"Failed to import AGraph: {e}")
    logging.error("Make sure you're running from the correct directory and AGraph is installed")
    sys.exit(1)

from config import config
from exceptions import AGraphMCPError, ConfigurationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agraph-mcp")

class AGraphMCPServer:
    """MCP Server for AGraph Knowledge Graph using direct AGraph instances"""
    
    def __init__(self):
        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            raise ConfigurationError(f"Invalid configuration: {e}")
        
        self.server = Server("agraph-mcp")
        self.agraph: Optional[AGraph] = None
        self._is_initialized = False
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP handlers"""
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="get_knowledge_graph_status",
                    description="Get knowledge graph status and statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": f"Collection name (default: {config.default_collection})"
                            }
                        }
                    }
                ),
                Tool(
                    name="search_entities",
                    description="Search and retrieve entities from the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": f"Collection name (default: {config.default_collection})"
                            },
                            "entity_type": {
                                "type": "string",
                                "description": "Filter by entity type (PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, OTHER)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of entities to return (default: 50)"
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination (default: 0)"
                            }
                        }
                    }
                ),
                Tool(
                    name="search_relations",
                    description="Search and retrieve relations from the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": f"Collection name (default: {config.default_collection})"
                            },
                            "relation_type": {
                                "type": "string",
                                "description": "Filter by relation type"
                            },
                            "entity_id": {
                                "type": "string",
                                "description": "Filter by entity ID (head or tail entity)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of relations to return (default: 30)"
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination (default: 0)"
                            }
                        }
                    }
                ),
                Tool(
                    name="search_text_chunks",
                    description="Search and retrieve text chunks from the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": f"Collection name (default: {config.default_collection})"
                            },
                            "search_query": {
                                "type": "string",
                                "description": "Search query for text content"
                            },
                            "entity_id": {
                                "type": "string",
                                "description": "Filter by entity ID to get text chunks containing the entity"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of text chunks to return (default: 20)"
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination (default: 0)"
                            }
                        }
                    }
                ),
                Tool(
                    name="search_clusters",
                    description="Search and retrieve clusters from the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": f"Collection name (default: {config.default_collection})"
                            },
                            "cluster_type": {
                                "type": "string",
                                "description": "Filter by cluster type"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of clusters to return (default: 15)"
                            },
                            "offset": {
                                "type": "integer",
                                "description": "Offset for pagination (default: 0)"
                            }
                        }
                    }
                ),
                Tool(
                    name="get_full_knowledge_graph",
                    description="Retrieve complete knowledge graph data",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "collection_name": {
                                "type": "string",
                                "description": f"Collection name (default: {config.default_collection})"
                            },
                            "include_text_chunks": {
                                "type": "boolean",
                                "description": "Include text chunks in response (default: false)"
                            },
                            "include_clusters": {
                                "type": "boolean",
                                "description": "Include clusters in response (default: false)"
                            },
                            "entity_limit": {
                                "type": "integer",
                                "description": "Limit number of entities returned"
                            },
                            "relation_limit": {
                                "type": "integer",
                                "description": "Limit number of relations returned"
                            }
                        }
                    }
                ),
                Tool(
                    name="natural_language_search",
                    description="Perform natural language search across the knowledge graph",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query"
                            },
                            "collection_name": {
                                "type": "string",
                                "description": f"Collection name (default: {config.default_collection})"
                            },
                            "search_type": {
                                "type": "string",
                                "enum": ["all", "entities", "relations", "text_chunks", "clusters"],
                                "description": "Type of search to perform (default: all)"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Limit number of results per type (default: 10)"
                            }
                        },
                        "required": ["query"]
                    }
                )
            ]

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Handle tool calls"""
            try:
                if name == "get_knowledge_graph_status":
                    return await self._get_knowledge_graph_status(arguments)
                elif name == "search_entities":
                    return await self._search_entities(arguments)
                elif name == "search_relations":
                    return await self._search_relations(arguments)
                elif name == "search_text_chunks":
                    return await self._search_text_chunks(arguments)
                elif name == "search_clusters":
                    return await self._search_clusters(arguments)
                elif name == "get_full_knowledge_graph":
                    return await self._get_full_knowledge_graph(arguments)
                elif name == "natural_language_search":
                    return await self._natural_language_search(arguments)
                else:
                    raise ValueError(f"Unknown tool: {name}")
            except Exception as e:
                logger.error(f"Error handling tool call {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]

    async def initialize_agraph(self, collection_name: Optional[str] = None) -> None:
        """Initialize AGraph instance"""
        if self._is_initialized and self.agraph and (
            collection_name is None or self.agraph.collection_name == collection_name
        ):
            return
        
        try:
            # Use collection_name or default
            collection = collection_name or config.default_collection
            
            # Create settings from config
            settings = get_settings()
            settings.workdir = config.workdir
            
            logger.info(f"Initializing AGraph with collection: {collection}")
            
            # Create AGraph instance
            self.agraph = AGraph(
                settings=settings,
                collection_name=collection,
                persist_directory=config.persist_directory,
                vector_store_type=config.vector_store_type,
                enable_knowledge_graph=True
            )
            
            # Initialize the AGraph instance
            await self.agraph.initialize()
            
            self._is_initialized = True
            logger.info(f"AGraph initialized successfully with collection: {collection}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AGraph: {e}")
            raise AGraphMCPError(f"AGraph initialization failed: {e}")

    async def _get_knowledge_graph_status(self, arguments: Dict) -> List[TextContent]:
        """Get knowledge graph status"""
        collection_name = arguments.get("collection_name")
        await self.initialize_agraph(collection_name)
        
        try:
            # Get statistics from AGraph
            stats = await self.agraph.get_stats()
            
            # Format the response nicely
            collection = self.agraph.collection_name
            status_text = f"# Knowledge Graph Status for '{collection}'\n\n"
            
            # Basic information
            status_text += "## Basic Information\n"
            status_text += f"- **Collection Name**: {collection}\n"
            status_text += f"- **Vector Store Type**: {self.agraph.vector_store_type}\n"
            status_text += f"- **Enable Knowledge Graph**: {self.agraph.enable_knowledge_graph}\n"
            status_text += f"- **Initialized**: {self.agraph.is_initialized}\n"
            status_text += f"- **Has Knowledge Graph**: {self.agraph.has_knowledge_graph}\n\n"
            
            # Knowledge graph statistics
            if stats.get("knowledge_graph"):
                kg_stats = stats["knowledge_graph"]
                status_text += "## Knowledge Graph Statistics\n"
                status_text += f"- **Entities**: {kg_stats.get('entities', 0)}\n"
                status_text += f"- **Relations**: {kg_stats.get('relations', 0)}\n"
                status_text += f"- **Clusters**: {kg_stats.get('clusters', 0)}\n"
                status_text += f"- **Text Chunks**: {kg_stats.get('text_chunks', 0)}\n\n"
            
            # Get entity type distribution if available
            if self.agraph.knowledge_graph and self.agraph.knowledge_graph.entities:
                entity_types = {}
                for entity in self.agraph.knowledge_graph.entities.values():
                    entity_type = entity.entity_type.value if entity.entity_type else "OTHER"
                    entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
                
                if entity_types:
                    status_text += "## Entity Types Distribution\n"
                    for entity_type, count in entity_types.items():
                        status_text += f"- **{entity_type}**: {count}\n"
                    status_text += "\n"
            
            # Get relation type distribution if available
            if self.agraph.knowledge_graph and self.agraph.knowledge_graph.relations:
                relation_types = {}
                for relation in self.agraph.knowledge_graph.relations.values():
                    relation_type = relation.relation_type.value if relation.relation_type else "OTHER"
                    relation_types[relation_type] = relation_types.get(relation_type, 0) + 1
                
                if relation_types:
                    status_text += "## Relation Types Distribution\n"
                    for relation_type, count in relation_types.items():
                        status_text += f"- **{relation_type}**: {count}\n"
                    status_text += "\n"
            
            # Vector store statistics
            if stats.get("vector_store"):
                vs_stats = stats["vector_store"]
                status_text += "## Vector Store Statistics\n"
                for key, value in vs_stats.items():
                    status_text += f"- **{key}**: {value}\n"
                status_text += "\n"
            
            return [TextContent(type="text", text=status_text)]
            
        except Exception as e:
            logger.error(f"Failed to get knowledge graph status: {e}")
            return [TextContent(type="text", text=f"Failed to get knowledge graph status: {str(e)}")]

    async def _search_entities(self, arguments: Dict) -> List[TextContent]:
        """Search entities"""
        collection_name = arguments.get("collection_name")
        await self.initialize_agraph(collection_name)
        
        try:
            # Get parameters
            entity_type = arguments.get("entity_type")
            limit = arguments.get("limit", config.default_entity_limit)
            offset = arguments.get("offset", 0)
            
            if not self.agraph.knowledge_graph or not self.agraph.knowledge_graph.entities:
                return [TextContent(type="text", text="No entities found in the knowledge graph.")]
            
            entities = list(self.agraph.knowledge_graph.entities.values())
            
            # Filter by entity type if specified
            if entity_type:
                try:
                    entity_type_enum = EntityType(entity_type.upper())
                    entities = [e for e in entities if e.entity_type == entity_type_enum]
                except ValueError:
                    return [TextContent(type="text", text=f"Invalid entity type: {entity_type}")]
            
            # Apply pagination
            total_count = len(entities)
            entities = entities[offset:offset + limit]
            
            response_text = f"# Entity Search Results\n\n"
            response_text += f"**Found**: {total_count} entities\n"
            response_text += f"**Showing**: {len(entities)} entities (offset: {offset})\n\n"
            
            for entity in entities:
                response_text += f"## {entity.name}\n"
                response_text += f"- **ID**: {entity.id}\n"
                response_text += f"- **Type**: {entity.entity_type.value if entity.entity_type else 'N/A'}\n"
                response_text += f"- **Description**: {entity.description or 'N/A'}\n"
                response_text += f"- **Confidence**: {entity.confidence}\n"
                
                if entity.aliases:
                    response_text += f"- **Aliases**: {', '.join(entity.aliases)}\n"
                
                if entity.properties:
                    response_text += f"- **Properties**: {json.dumps(entity.properties, ensure_ascii=False)}\n"
                
                response_text += "\n"
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            logger.error(f"Failed to search entities: {e}")
            return [TextContent(type="text", text=f"Failed to search entities: {str(e)}")]

    async def _search_relations(self, arguments: Dict) -> List[TextContent]:
        """Search relations"""
        collection_name = arguments.get("collection_name")
        await self.initialize_agraph(collection_name)
        
        try:
            # Get parameters
            relation_type = arguments.get("relation_type")
            entity_id = arguments.get("entity_id")
            limit = arguments.get("limit", config.default_relation_limit)
            offset = arguments.get("offset", 0)
            
            if not self.agraph.knowledge_graph or not self.agraph.knowledge_graph.relations:
                return [TextContent(type="text", text="No relations found in the knowledge graph.")]
            
            relations = list(self.agraph.knowledge_graph.relations.values())
            
            # Filter by relation type if specified
            if relation_type:
                try:
                    relation_type_enum = RelationType(relation_type.upper())
                    relations = [r for r in relations if r.relation_type == relation_type_enum]
                except ValueError:
                    # Allow string matching for custom relation types
                    relations = [r for r in relations if str(r.relation_type).upper() == relation_type.upper()]
            
            # Filter by entity ID if specified
            if entity_id:
                relations = [r for r in relations if r.head_entity.id == entity_id or r.tail_entity.id == entity_id]
            
            # Apply pagination
            total_count = len(relations)
            relations = relations[offset:offset + limit]
            
            response_text = f"# Relation Search Results\n\n"
            response_text += f"**Found**: {total_count} relations\n"
            response_text += f"**Showing**: {len(relations)} relations (offset: {offset})\n\n"
            
            for relation in relations:
                head_name = relation.head_entity.name if relation.head_entity else "Unknown"
                tail_name = relation.tail_entity.name if relation.tail_entity else "Unknown"
                
                response_text += f"## {head_name} → {tail_name}\n"
                response_text += f"- **ID**: {relation.id}\n"
                response_text += f"- **Type**: {relation.relation_type.value if relation.relation_type else 'N/A'}\n"
                response_text += f"- **Description**: {relation.description or 'N/A'}\n"
                response_text += f"- **Confidence**: {relation.confidence}\n"
                response_text += f"- **Head Entity ID**: {relation.head_entity.id if relation.head_entity else 'N/A'}\n"
                response_text += f"- **Tail Entity ID**: {relation.tail_entity.id if relation.tail_entity else 'N/A'}\n"
                
                if relation.properties:
                    response_text += f"- **Properties**: {json.dumps(relation.properties, ensure_ascii=False)}\n"
                
                response_text += "\n"
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            logger.error(f"Failed to search relations: {e}")
            return [TextContent(type="text", text=f"Failed to search relations: {str(e)}")]

    async def _search_text_chunks(self, arguments: Dict) -> List[TextContent]:
        """Search text chunks"""
        collection_name = arguments.get("collection_name")
        await self.initialize_agraph(collection_name)
        
        try:
            # Get parameters
            search_query = arguments.get("search_query")
            entity_id = arguments.get("entity_id")
            limit = arguments.get("limit", config.default_text_chunk_limit)
            offset = arguments.get("offset", 0)
            
            # Try vector search first if query is provided
            if search_query and self.agraph.vector_store:
                try:
                    results = await self.agraph.search_text_chunks(search_query, top_k=limit + offset)
                    # Apply offset manually for vector search results
                    results = results[offset:]
                    
                    response_text = f"# Text Chunk Search Results (Vector Search)\n\n"
                    response_text += f"**Query**: {search_query}\n"
                    response_text += f"**Found**: {len(results)} text chunks\n\n"
                    
                    for chunk, score in results:
                        response_text += f"## Text Chunk: {chunk.id}\n"
                        response_text += f"- **Source**: {chunk.source or 'N/A'}\n"
                        response_text += f"- **Title**: {chunk.title or 'N/A'}\n"
                        response_text += f"- **Score**: {score:.4f}\n"
                        
                        content_preview = chunk.content[:config.max_content_preview_length]
                        if len(chunk.content) > config.max_content_preview_length:
                            content_preview += "..."
                        response_text += f"- **Content**: {content_preview}\n"
                        
                        if hasattr(chunk, 'start_index') and chunk.start_index is not None:
                            response_text += f"- **Start Index**: {chunk.start_index}\n"
                        if hasattr(chunk, 'end_index') and chunk.end_index is not None:
                            response_text += f"- **End Index**: {chunk.end_index}\n"
                        
                        response_text += "\n"
                    
                    return [TextContent(type="text", text=response_text)]
                    
                except Exception as e:
                    logger.warning(f"Vector search failed, falling back to text search: {e}")
            
            # Fallback to knowledge graph text chunks
            if not self.agraph.knowledge_graph or not self.agraph.knowledge_graph.text_chunks:
                return [TextContent(type="text", text="No text chunks found in the knowledge graph.")]
            
            text_chunks = list(self.agraph.knowledge_graph.text_chunks.values())
            
            # Filter by search query if specified (simple text matching)
            if search_query:
                search_query_lower = search_query.lower()
                text_chunks = [
                    chunk for chunk in text_chunks
                    if search_query_lower in chunk.content.lower() or
                       (chunk.title and search_query_lower in chunk.title.lower()) or
                       (chunk.source and search_query_lower in chunk.source.lower())
                ]
            
            # Filter by entity ID if specified
            if entity_id:
                # This is a simplified implementation - you might want to enhance this
                # based on how entity-chunk relationships are stored in your system
                text_chunks = [
                    chunk for chunk in text_chunks
                    if entity_id in chunk.content  # Simple text-based search
                ]
            
            # Apply pagination
            total_count = len(text_chunks)
            text_chunks = text_chunks[offset:offset + limit]
            
            response_text = f"# Text Chunk Search Results\n\n"
            if search_query:
                response_text += f"**Query**: {search_query}\n"
            response_text += f"**Found**: {total_count} text chunks\n"
            response_text += f"**Showing**: {len(text_chunks)} text chunks (offset: {offset})\n\n"
            
            for chunk in text_chunks:
                response_text += f"## Text Chunk: {chunk.id}\n"
                response_text += f"- **Source**: {chunk.source or 'N/A'}\n"
                response_text += f"- **Title**: {chunk.title or 'N/A'}\n"
                
                content_preview = chunk.content[:config.max_content_preview_length]
                if len(chunk.content) > config.max_content_preview_length:
                    content_preview += "..."
                response_text += f"- **Content**: {content_preview}\n"
                
                if hasattr(chunk, 'start_index') and chunk.start_index is not None:
                    response_text += f"- **Start Index**: {chunk.start_index}\n"
                if hasattr(chunk, 'end_index') and chunk.end_index is not None:
                    response_text += f"- **End Index**: {chunk.end_index}\n"
                
                response_text += "\n"
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            logger.error(f"Failed to search text chunks: {e}")
            return [TextContent(type="text", text=f"Failed to search text chunks: {str(e)}")]

    async def _search_clusters(self, arguments: Dict) -> List[TextContent]:
        """Search clusters"""
        collection_name = arguments.get("collection_name")
        await self.initialize_agraph(collection_name)
        
        try:
            # Get parameters
            cluster_type = arguments.get("cluster_type")
            limit = arguments.get("limit", config.default_cluster_limit)
            offset = arguments.get("offset", 0)
            
            if not self.agraph.knowledge_graph or not self.agraph.knowledge_graph.clusters:
                return [TextContent(type="text", text="No clusters found in the knowledge graph.")]
            
            clusters = list(self.agraph.knowledge_graph.clusters.values())
            
            # Filter by cluster type if specified
            if cluster_type:
                try:
                    cluster_type_enum = ClusterType(cluster_type.upper())
                    clusters = [c for c in clusters if c.cluster_type == cluster_type_enum]
                except ValueError:
                    # Allow string matching for custom cluster types
                    clusters = [c for c in clusters if str(c.cluster_type).upper() == cluster_type.upper()]
            
            # Apply pagination
            total_count = len(clusters)
            clusters = clusters[offset:offset + limit]
            
            response_text = f"# Cluster Search Results\n\n"
            response_text += f"**Found**: {total_count} clusters\n"
            response_text += f"**Showing**: {len(clusters)} clusters (offset: {offset})\n\n"
            
            for cluster in clusters:
                response_text += f"## {cluster.name}\n"
                response_text += f"- **ID**: {cluster.id}\n"
                response_text += f"- **Type**: {cluster.cluster_type.value if cluster.cluster_type else 'N/A'}\n"
                response_text += f"- **Description**: {cluster.description or 'N/A'}\n"
                response_text += f"- **Confidence**: {cluster.confidence}\n"
                response_text += f"- **Entities Count**: {len(cluster.entities) if cluster.entities else 0}\n"
                response_text += f"- **Relations Count**: {len(cluster.relations) if cluster.relations else 0}\n"
                
                if cluster.properties:
                    response_text += f"- **Properties**: {json.dumps(cluster.properties, ensure_ascii=False)}\n"
                
                response_text += "\n"
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            logger.error(f"Failed to search clusters: {e}")
            return [TextContent(type="text", text=f"Failed to search clusters: {str(e)}")]

    async def _get_full_knowledge_graph(self, arguments: Dict) -> List[TextContent]:
        """Get full knowledge graph"""
        collection_name = arguments.get("collection_name")
        await self.initialize_agraph(collection_name)
        
        try:
            # Get parameters
            include_text_chunks = arguments.get("include_text_chunks", False)
            include_clusters = arguments.get("include_clusters", False)
            entity_limit = arguments.get("entity_limit")
            relation_limit = arguments.get("relation_limit")
            
            if not self.agraph.knowledge_graph:
                return [TextContent(type="text", text="No knowledge graph found.")]
            
            kg = self.agraph.knowledge_graph
            
            response_text = f"# Complete Knowledge Graph: {kg.name or 'Unnamed'}\n\n"
            response_text += f"**Description**: {kg.description or 'N/A'}\n\n"
            
            # Entities summary
            entities = list(kg.entities.values())
            if entity_limit:
                entities = entities[:entity_limit]
            
            if entities:
                response_text += f"## Entities ({len(entities)})\n\n"
                for entity in entities[:5]:  # Show first 5 entities in detail
                    response_text += f"- **{entity.name}** ({entity.entity_type.value if entity.entity_type else 'unknown'}): {entity.description or 'N/A'}\n"
                if len(entities) > 5:
                    response_text += f"- ... and {len(entities) - 5} more entities\n"
                response_text += "\n"
            
            # Relations summary
            relations = list(kg.relations.values())
            if relation_limit:
                relations = relations[:relation_limit]
            
            if relations:
                response_text += f"## Relations ({len(relations)})\n\n"
                for relation in relations[:5]:  # Show first 5 relations in detail
                    head_name = relation.head_entity.name if relation.head_entity else "Unknown"
                    tail_name = relation.tail_entity.name if relation.tail_entity else "Unknown"
                    response_text += f"- **{head_name}** --[{relation.relation_type.value if relation.relation_type else 'unknown'}]--> **{tail_name}**\n"
                if len(relations) > 5:
                    response_text += f"- ... and {len(relations) - 5} more relations\n"
                response_text += "\n"
            
            # Clusters summary (if requested)
            if include_clusters and kg.clusters:
                response_text += f"## Clusters ({len(kg.clusters)})\n\n"
                for cluster in list(kg.clusters.values())[:5]:
                    response_text += f"- **{cluster.name}**: {cluster.description or 'N/A'} ({len(cluster.entities) if cluster.entities else 0} entities)\n"
                if len(kg.clusters) > 5:
                    response_text += f"- ... and {len(kg.clusters) - 5} more clusters\n"
                response_text += "\n"
            
            # Text chunks summary (if requested)
            if include_text_chunks and kg.text_chunks:
                response_text += f"## Text Chunks ({len(kg.text_chunks)})\n\n"
                for chunk in list(kg.text_chunks.values())[:3]:
                    content_preview = chunk.content[:100]
                    if len(chunk.content) > 100:
                        content_preview += "..."
                    response_text += f"- **{chunk.source or chunk.id}**: {content_preview}\n"
                if len(kg.text_chunks) > 3:
                    response_text += f"- ... and {len(kg.text_chunks) - 3} more text chunks\n"
                response_text += "\n"
            
            response_text += f"\n**Note**: This is a summary view. Use specific search tools for detailed information about entities, relations, or clusters."
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            logger.error(f"Failed to get knowledge graph: {e}")
            return [TextContent(type="text", text=f"Failed to get knowledge graph: {str(e)}")]

    async def _natural_language_search(self, arguments: Dict) -> List[TextContent]:
        """Perform natural language search"""
        query = arguments.get("query", "")
        if not query:
            return [TextContent(type="text", text="Error: Query is required for natural language search")]
        
        collection_name = arguments.get("collection_name")
        search_type = arguments.get("search_type", "all")
        limit = arguments.get("limit", config.natural_search_default_limit)
        
        await self.initialize_agraph(collection_name)
        
        response_text = f"# Natural Language Search Results for: '{query}'\n\n"
        
        try:
            # Search entities if requested
            if search_type in ["all", "entities"] and self.agraph.knowledge_graph and self.agraph.knowledge_graph.entities:
                try:
                    if self.agraph.vector_store:
                        # Try vector search
                        entity_results = await self.agraph.search_entities(query, top_k=limit)
                        if entity_results:
                            response_text += f"## Matching Entities (Vector Search) ({len(entity_results)})\n\n"
                            for entity, score in entity_results:
                                response_text += f"- **{entity.name}** ({entity.entity_type.value if entity.entity_type else 'unknown'}) [Score: {score:.4f}]: {entity.description or 'N/A'}\n"
                            response_text += "\n"
                    else:
                        # Fallback to simple keyword matching
                        entities = list(self.agraph.knowledge_graph.entities.values())
                        matching_entities = [
                            e for e in entities 
                            if query.lower() in e.name.lower() or 
                               (e.description and query.lower() in e.description.lower())
                        ]
                        
                        if matching_entities:
                            response_text += f"## Matching Entities ({len(matching_entities[:limit])})\n\n"
                            for entity in matching_entities[:limit]:
                                response_text += f"- **{entity.name}** ({entity.entity_type.value if entity.entity_type else 'unknown'}): {entity.description or 'N/A'}\n"
                            response_text += "\n"
                except Exception as e:
                    logger.warning(f"Entity search failed: {e}")
            
            # Search text chunks if requested
            if search_type in ["all", "text_chunks"]:
                try:
                    if self.agraph.vector_store:
                        # Try vector search
                        chunk_results = await self.agraph.search_text_chunks(query, top_k=limit)
                        if chunk_results:
                            response_text += f"## Matching Text Chunks (Vector Search) ({len(chunk_results)})\n\n"
                            for chunk, score in chunk_results:
                                content_preview = chunk.content[:150]
                                if len(chunk.content) > 150:
                                    content_preview += '...'
                                response_text += f"- **{chunk.source or chunk.id}** [Score: {score:.4f}]: {content_preview}\n"
                            response_text += "\n"
                    elif self.agraph.knowledge_graph and self.agraph.knowledge_graph.text_chunks:
                        # Fallback to simple keyword matching
                        text_chunks = list(self.agraph.knowledge_graph.text_chunks.values())
                        matching_chunks = [
                            c for c in text_chunks
                            if query.lower() in c.content.lower() or
                               (c.title and query.lower() in c.title.lower()) or
                               (c.source and query.lower() in c.source.lower())
                        ]
                        
                        if matching_chunks:
                            response_text += f"## Matching Text Chunks ({len(matching_chunks[:limit])})\n\n"
                            for chunk in matching_chunks[:limit]:
                                content_preview = chunk.content[:150]
                                if len(chunk.content) > 150:
                                    content_preview += '...'
                                response_text += f"- **{chunk.source or chunk.id}**: {content_preview}\n"
                            response_text += "\n"
                except Exception as e:
                    logger.warning(f"Text chunk search failed: {e}")
            
            # Search relations if requested
            if search_type in ["all", "relations"] and self.agraph.knowledge_graph and self.agraph.knowledge_graph.relations:
                try:
                    if self.agraph.vector_store:
                        # Try vector search
                        relation_results = await self.agraph.search_relations(query, top_k=limit)
                        if relation_results:
                            response_text += f"## Matching Relations (Vector Search) ({len(relation_results)})\n\n"
                            for relation, score in relation_results:
                                head_name = relation.head_entity.name if relation.head_entity else "Unknown"
                                tail_name = relation.tail_entity.name if relation.tail_entity else "Unknown"
                                response_text += f"- **{head_name}** --[{relation.relation_type.value if relation.relation_type else 'unknown'}]--> **{tail_name}** [Score: {score:.4f}]: {relation.description or 'N/A'}\n"
                            response_text += "\n"
                    else:
                        # Fallback to simple keyword matching
                        relations = list(self.agraph.knowledge_graph.relations.values())
                        matching_relations = [
                            r for r in relations
                            if (r.description and query.lower() in r.description.lower()) or
                               (r.head_entity and query.lower() in r.head_entity.name.lower()) or
                               (r.tail_entity and query.lower() in r.tail_entity.name.lower())
                        ]
                        
                        if matching_relations:
                            response_text += f"## Matching Relations ({len(matching_relations[:limit])})\n\n"
                            for relation in matching_relations[:limit]:
                                head_name = relation.head_entity.name if relation.head_entity else "Unknown"
                                tail_name = relation.tail_entity.name if relation.tail_entity else "Unknown"
                                response_text += f"- **{head_name}** --[{relation.relation_type.value if relation.relation_type else 'unknown'}]--> **{tail_name}**: {relation.description or 'N/A'}\n"
                            response_text += "\n"
                except Exception as e:
                    logger.warning(f"Relation search failed: {e}")
            
            # Search clusters if requested
            if search_type in ["all", "clusters"] and self.agraph.knowledge_graph and self.agraph.knowledge_graph.clusters:
                try:
                    clusters = list(self.agraph.knowledge_graph.clusters.values())
                    # Simple keyword matching for clusters
                    matching_clusters = [
                        c for c in clusters
                        if query.lower() in c.name.lower() or
                           (c.description and query.lower() in c.description.lower())
                    ]
                    
                    if matching_clusters:
                        response_text += f"## Matching Clusters ({len(matching_clusters[:limit])})\n\n"
                        for cluster in matching_clusters[:limit]:
                            response_text += f"- **{cluster.name}**: {cluster.description or 'N/A'} ({len(cluster.entities) if cluster.entities else 0} entities)\n"
                        response_text += "\n"
                except Exception as e:
                    logger.warning(f"Cluster search failed: {e}")
            
            if len(response_text.split('\n')) <= 3:  # Only header, no results found
                response_text += "No matching results found for your query. Try different keywords or search types."
            
            return [TextContent(type="text", text=response_text)]
            
        except Exception as e:
            logger.error(f"Natural language search failed: {e}")
            return [TextContent(type="text", text=f"Natural language search failed: {str(e)}")]

    async def run(self):
        """Run the MCP server"""
        import mcp.server.stdio
        
        logger.info("Starting AGraph MCP Server...")
        logger.info(f"Workdir: {config.workdir}")
        logger.info(f"Default Collection: {config.default_collection}")
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(read_stream, write_stream)
    
    async def cleanup(self):
        """Cleanup resources"""
        if self.agraph:
            await self.agraph.close()


def main():
    """Main entry point"""
    server = AGraphMCPServer()
    
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
    finally:
        asyncio.run(server.cleanup())


if __name__ == "__main__":
    main()