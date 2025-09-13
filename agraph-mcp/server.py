#!/usr/bin/env python3
"""
AGraph MCP Server - Model Context Protocol server for AGraph semantic search

This MCP server provides semantic search capabilities using AGraph knowledge graphs.
It supports searching entities, relations, clusters, and text chunks across projects.
"""

import asyncio
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
from collections.abc import AsyncIterator

from mcp.server.fastmcp import Context, FastMCP
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from agraph import AGraph
    from agraph.config import get_settings, update_settings
    from config import get_config_manager, load_server_config, get_agraph_settings
    from exceptions import (
        AGrapeMCPError, AGraphInitializationError, SearchError, 
        handle_agraph_error
    )
    
    # ChromaDB compatibility fix for AGraph
    try:
        import chromadb.errors as chroma_errors
        if not hasattr(chroma_errors, 'InvalidCollectionException'):
            # Add compatibility alias for older AGraph versions
            chroma_errors.InvalidCollectionException = chroma_errors.NotFoundError
            print("✅ ChromaDB compatibility patch applied")
    except ImportError:
        print("⚠️  ChromaDB not available for compatibility patch")
    
except ImportError as e:
    print(f"Error importing AGraph or MCP modules: {e}")
    print("Please ensure AGraph is installed and accessible")
    sys.exit(1)


@dataclass
class AGraphContext:
    """Application context with AGraph instances per project."""
    instances: Dict[str, AGraph]
    project_base_dir: Path
    config: Dict[str, Any]


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AGraphContext]:
    """Manage AGraph instances lifecycle."""
    print("🚀 Starting AGraph MCP Server...")
    
    # Load configuration
    config = load_server_config()
    print("📋 Configuration loaded successfully")
    
    # Initialize project base directory
    project_dir = config['agraph']['project_dir']
    project_base_dir = Path(project_dir)
    project_base_dir.mkdir(parents=True, exist_ok=True)
    
    app_context = AGraphContext(
        instances={},
        project_base_dir=project_base_dir,
        config=config
    )
    
    print(f"✅ AGraph MCP Server initialized")
    print(f"   📁 Project base directory: {project_base_dir}")
    print(f"   🧠 Default model: {config['agraph']['default_model']}")
    print(f"   ⚙️ Max concurrent projects: {config['mcp_server']['max_concurrent_projects']}")
    
    try:
        yield app_context
    finally:
        # Cleanup AGraph instances
        print("🔄 Cleaning up AGraph instances...")
        cleanup_count = 0
        for project_name, agraph in app_context.instances.items():
            try:
                await agraph.__aexit__(None, None, None)
                cleanup_count += 1
                print(f"   ✅ Cleaned up AGraph instance for project: {project_name}")
            except Exception as e:
                print(f"   ⚠️ Error cleaning up {project_name}: {e}")
        
        print(f"✅ AGraph MCP Server shutdown complete ({cleanup_count} instances cleaned)")


# Create FastMCP server with lifespan management
mcp = FastMCP("AGraph Semantic Search", lifespan=app_lifespan)


async def get_or_create_agraph(
    ctx: Context[ServerSession, AGraphContext], 
    project: str
) -> AGraph:
    """Get existing AGraph instance or create new one for project."""
    app_ctx = ctx.request_context.lifespan_context
    
    # Check if instance already exists
    if project in app_ctx.instances:
        await ctx.debug(f"Using existing AGraph instance for project: {project}")
        return app_ctx.instances[project]
    
    # Check project limit
    max_projects = app_ctx.config['mcp_server']['max_concurrent_projects']
    if len(app_ctx.instances) >= max_projects:
        error_msg = f"Maximum concurrent projects ({max_projects}) exceeded"
        await ctx.error(error_msg)
        raise AGraphInitializationError(project, error_msg)
    
    try:
        # Use the project path directly as workdir (not create subdirectory)
        project_path = app_ctx.project_base_dir / project
        
        # For existing projects, don't create directory, just use it
        if not project_path.exists():
            await ctx.warning(f"Project path does not exist: {project_path}")
            project_path.mkdir(parents=True, exist_ok=True)
        
        # Get base settings and update with project-specific config
        settings = get_settings()
        agraph_config = app_ctx.config['agraph']
        
        settings_override = {
            "workdir": str(project_path),
            "llm_config": {
                "model": agraph_config['default_model']
            },
            "processing_config": {
                "chunk_size": agraph_config['chunk_size'],
                "chunk_overlap": agraph_config['chunk_overlap']
            },
            "extraction_config": {
                "entity_confidence_threshold": agraph_config['entity_confidence_threshold'],
                "relation_confidence_threshold": agraph_config['relation_confidence_threshold']
            },
            "max_current": agraph_config['max_current']
        }
        
        settings = update_settings(settings_override)
        await ctx.debug(f"AGraph settings updated for project: {project}")
        
        await ctx.info(f"Creating AGraph instance for project: {project}")
        await ctx.debug(f"Project workdir: {project_path}, model: {agraph_config['default_model']}")
        
        # Create AGraph with collection name that matches existing data
        # Check if there's existing data to determine collection name
        collection_name = project  # Use project name directly
        if (project_path / "chroma").exists():
            # Try to find existing collection
            try:
                chroma_dirs = list((project_path / "chroma").glob("*"))
                if chroma_dirs and chroma_dirs[0].is_dir():
                    # Use the actual collection name from chroma directory
                    collection_name = chroma_dirs[0].name
                    await ctx.debug(f"Found existing collection: {collection_name}")
            except Exception as e:
                await ctx.debug(f"Could not detect collection name: {e}")
        
        agraph = AGraph(collection_name=collection_name)
        
        # Try to enter async context and initialize with ChromaDB compatibility
        try:
            await agraph.__aenter__()
        except Exception as init_error:
            init_error_msg = str(init_error)
            if "InvalidCollectionException" in init_error_msg:
                await ctx.warning(f"ChromaDB compatibility issue detected, attempting workaround...")
                # Try to patch ChromaDB errors for compatibility
                try:
                    import chromadb.errors as errors
                    if not hasattr(errors, 'InvalidCollectionException'):
                        # Create a compatibility alias
                        errors.InvalidCollectionException = errors.NotFoundError
                    await agraph.__aenter__()
                except Exception as patch_error:
                    await ctx.error(f"ChromaDB compatibility patch failed: {patch_error}")
                    raise init_error
            else:
                raise init_error
        
        # Initialize the AGraph instance
        try:
            await agraph.initialize()
        except Exception as load_error:
            load_error_msg = str(load_error)
            if "collection" in load_error_msg.lower() and "not found" in load_error_msg.lower():
                await ctx.warning(f"Collection not found, this may be normal for new projects")
                # Don't fail here as collection might be created on first use
            else:
                await ctx.warning(f"AGraph initialization warning: {load_error_msg}")
                # Continue anyway as some initialization issues may be recoverable
        
        # Cache the instance
        app_ctx.instances[project] = agraph
        
        await ctx.info(f"✅ AGraph instance loaded for project: {project}")
        await ctx.debug(f"📊 Active projects: {len(app_ctx.instances)}/{max_projects}")
        
        return agraph
        
    except Exception as e:
        error = handle_agraph_error(e, "AGraph initialization", {"project": project})
        await ctx.error(f"Failed to initialize AGraph for project '{project}': {error.message}")
        raise error


# Pydantic models for structured output
class EntityResult(BaseModel):
    """Search result for an entity."""
    name: str = Field(description="Entity name")
    entity_type: str = Field(description="Type of the entity") 
    confidence: float = Field(description="Confidence score")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Entity properties")
    score: float = Field(description="Search similarity score")


class RelationResult(BaseModel):
    """Search result for a relation."""
    source: str = Field(description="Source entity name")
    target: str = Field(description="Target entity name")
    relation_type: str = Field(description="Type of the relation")
    confidence: float = Field(description="Confidence score")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Relation properties")
    score: float = Field(description="Search similarity score")


class TextChunkResult(BaseModel):
    """Search result for a text chunk."""
    content: str = Field(description="Text chunk content")
    chunk_id: str = Field(description="Unique chunk identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    score: float = Field(description="Search similarity score")


class ClusterResult(BaseModel):
    """Search result for a cluster."""
    cluster_id: str = Field(description="Unique cluster identifier")
    cluster_name: str = Field(description="Cluster name or description")
    entities: List[str] = Field(default_factory=list, description="Entity names in cluster")
    cluster_type: str = Field(description="Type or category of cluster")
    score: float = Field(description="Search similarity score")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Cluster properties")


class SemanticSearchResults(BaseModel):
    """Combined semantic search results."""
    query: str = Field(description="Original search query")
    project: str = Field(description="Project name")
    entities: List[EntityResult] = Field(default_factory=list)
    relations: List[RelationResult] = Field(default_factory=list) 
    text_chunks: List[TextChunkResult] = Field(default_factory=list)
    clusters: List[ClusterResult] = Field(default_factory=list)
    total_results: int = Field(description="Total number of results found")


@mcp.tool()
async def search_entities(
    project: str,
    query: str,
    top_k: int = 5,
    ctx: Context[ServerSession, AGraphContext] = None
) -> List[EntityResult]:
    """
    Search for entities in the specified project's knowledge graph.
    
    Args:
        project: Project name to search within
        query: Search query text
        top_k: Maximum number of results to return (default: 5)
    
    Returns:
        List of matching entities with similarity scores
    """
    try:
        await ctx.info(f"Searching entities for query: '{query}' in project: {project}")
        
        agraph = await get_or_create_agraph(ctx, project)
        results = await agraph.search_entities(query, top_k=top_k)
        
        entity_results = []
        for entity, score in results:
            entity_results.append(EntityResult(
                name=entity.name,
                entity_type=entity.entity_type.value if hasattr(entity.entity_type, 'value') else str(entity.entity_type),
                confidence=entity.confidence,
                properties=entity.properties,
                score=score
            ))
        
        await ctx.info(f"Found {len(entity_results)} entities")
        return entity_results
        
    except Exception as e:
        await ctx.error(f"Entity search error: {str(e)}")
        return []


@mcp.tool()
async def search_relations(
    project: str,
    query: str, 
    top_k: int = 5,
    ctx: Context[ServerSession, AGraphContext] = None
) -> List[RelationResult]:
    """
    Search for relations in the specified project's knowledge graph.
    
    Args:
        project: Project name to search within
        query: Search query text
        top_k: Maximum number of results to return (default: 5)
    
    Returns:
        List of matching relations with similarity scores
    """
    try:
        await ctx.info(f"Searching relations for query: '{query}' in project: {project}")
        
        agraph = await get_or_create_agraph(ctx, project)
        results = await agraph.search_relations(query, top_k=top_k)
        
        relation_results = []
        for relation, score in results:
            relation_results.append(RelationResult(
                source=relation.source,
                target=relation.target,
                relation_type=relation.relation_type.value if hasattr(relation.relation_type, 'value') else str(relation.relation_type),
                confidence=relation.confidence,
                properties=relation.properties,
                score=score
            ))
        
        await ctx.info(f"Found {len(relation_results)} relations")
        return relation_results
        
    except Exception as e:
        await ctx.error(f"Relation search error: {str(e)}")
        return []


@mcp.tool()
async def search_text_chunks(
    project: str,
    query: str,
    top_k: int = 5,
    ctx: Context[ServerSession, AGraphContext] = None
) -> List[TextChunkResult]:
    """
    Search for text chunks in the specified project's knowledge graph.
    
    Args:
        project: Project name to search within
        query: Search query text  
        top_k: Maximum number of results to return (default: 5)
    
    Returns:
        List of matching text chunks with similarity scores
    """
    try:
        await ctx.info(f"Searching text chunks for query: '{query}' in project: {project}")
        
        agraph = await get_or_create_agraph(ctx, project)
        results = await agraph.search_text_chunks(query, top_k=top_k)
        
        chunk_results = []
        for chunk, score in results:
            chunk_results.append(TextChunkResult(
                content=chunk.content,
                chunk_id=getattr(chunk, 'chunk_id', str(hash(chunk.content))[:8]),
                metadata=getattr(chunk, 'metadata', {}),
                score=score
            ))
        
        await ctx.info(f"Found {len(chunk_results)} text chunks")  
        return chunk_results
        
    except Exception as e:
        await ctx.error(f"Text chunk search error: {str(e)}")
        return []


@mcp.tool()
async def search_clusters(
    project: str,
    query: str,
    top_k: int = 5,
    ctx: Context[ServerSession, AGraphContext] = None
) -> List[ClusterResult]:
    """
    Search for clusters in the specified project's knowledge graph.
    
    Args:
        project: Project name to search within
        query: Search query text
        top_k: Maximum number of results to return (default: 5)
    
    Returns:
        List of matching clusters with similarity scores
    """
    try:
        await ctx.info(f"Searching clusters for query: '{query}' in project: {project}")
        
        agraph = await get_or_create_agraph(ctx, project)
        
        # Check if AGraph has cluster search capability
        if hasattr(agraph, 'search_clusters'):
            results = await agraph.search_clusters(query, top_k=top_k)
            
            cluster_results = []
            for cluster, score in results:
                cluster_results.append(ClusterResult(
                    cluster_id=getattr(cluster, 'cluster_id', str(hash(str(cluster)))[:8]),
                    cluster_name=getattr(cluster, 'name', f"Cluster_{hash(str(cluster))%1000}"),
                    entities=getattr(cluster, 'entities', []),
                    cluster_type=getattr(cluster, 'cluster_type', 'general'),
                    score=score,
                    properties=getattr(cluster, 'properties', {})
                ))
            
            await ctx.info(f"Found {len(cluster_results)} clusters")
            return cluster_results
        else:
            # Fallback: cluster entities by similarity to query
            await ctx.info("Using entity-based clustering fallback")
            entities = await agraph.search_entities(query, top_k=top_k*2)
            
            # Group entities by type for basic clustering
            entity_groups = {}
            for entity, score in entities:
                entity_type = str(entity.entity_type.value if hasattr(entity.entity_type, 'value') else entity.entity_type)
                if entity_type not in entity_groups:
                    entity_groups[entity_type] = []
                entity_groups[entity_type].append((entity.name, score))
            
            cluster_results = []
            for cluster_type, entities_in_cluster in entity_groups.items():
                if len(entities_in_cluster) >= 2:  # Only create clusters with multiple entities
                    avg_score = sum(score for _, score in entities_in_cluster) / len(entities_in_cluster)
                    cluster_results.append(ClusterResult(
                        cluster_id=f"{cluster_type}_{hash(query)%1000}",
                        cluster_name=f"{cluster_type.title()} Cluster",
                        entities=[name for name, _ in entities_in_cluster],
                        cluster_type=cluster_type,
                        score=avg_score,
                        properties={"entity_count": len(entities_in_cluster)}
                    ))
            
            # Sort by score and limit results
            cluster_results.sort(key=lambda x: x.score, reverse=True)
            cluster_results = cluster_results[:top_k]
            
            await ctx.info(f"Generated {len(cluster_results)} entity-type clusters")
            return cluster_results
        
    except Exception as e:
        await ctx.error(f"Cluster search error: {str(e)}")
        return []


@mcp.tool() 
async def semantic_search_all(
    project: str,
    query: str,
    top_k_per_type: int = 3,
    ctx: Context[ServerSession, AGraphContext] = None
) -> SemanticSearchResults:
    """
    Perform comprehensive semantic search across entities, relations, and text chunks.
    
    Args:
        project: Project name to search within
        query: Search query text
        top_k_per_type: Maximum results per search type (default: 3)
    
    Returns:
        Combined search results from all types
    """
    try:
        await ctx.info(f"Performing comprehensive semantic search for: '{query}' in project: {project}")
        
        # Search all types concurrently for better performance
        entities_task = search_entities(project, query, top_k_per_type, ctx)
        relations_task = search_relations(project, query, top_k_per_type, ctx)  
        chunks_task = search_text_chunks(project, query, top_k_per_type, ctx)
        clusters_task = search_clusters(project, query, top_k_per_type, ctx)
        
        entities, relations, text_chunks, clusters = await asyncio.gather(
            entities_task, relations_task, chunks_task, clusters_task,
            return_exceptions=True
        )
        
        # Handle any exceptions from concurrent searches
        if isinstance(entities, Exception):
            await ctx.warning(f"Entity search failed: {entities}")
            entities = []
        if isinstance(relations, Exception):
            await ctx.warning(f"Relation search failed: {relations}")
            relations = []
        if isinstance(text_chunks, Exception):
            await ctx.warning(f"Text chunk search failed: {text_chunks}")
            text_chunks = []
        if isinstance(clusters, Exception):
            await ctx.warning(f"Cluster search failed: {clusters}")
            clusters = []
        
        total = len(entities) + len(relations) + len(text_chunks) + len(clusters)
        
        await ctx.info(f"Comprehensive search completed: {total} total results")
        
        return SemanticSearchResults(
            query=query,
            project=project,
            entities=entities,
            relations=relations,
            text_chunks=text_chunks,
            clusters=clusters,
            total_results=total
        )
        
    except Exception as e:
        await ctx.error(f"Comprehensive search error: {str(e)}")
        return SemanticSearchResults(
            query=query,
            project=project,
            entities=[],
            relations=[],
            text_chunks=[],
            clusters=[],
            total_results=0
        )


@mcp.tool()
async def get_project_stats(
    project: str,
    ctx: Context[ServerSession, AGraphContext] = None
) -> Dict[str, Any]:
    """
    Get statistics for a project's knowledge graph.
    
    Args:
        project: Project name
    
    Returns:
        Project statistics including entity, relation, and chunk counts
    """
    try:
        await ctx.info(f"Getting stats for project: {project}")
        
        agraph = await get_or_create_agraph(ctx, project)
        stats = await agraph.get_stats()
        
        project_stats = {
            "project": project,
            "initialized": True,
            **stats
        }
        
        await ctx.info(f"Project stats retrieved: {project_stats}")
        return project_stats
        
    except Exception as e:
        await ctx.error(f"Error getting project stats: {str(e)}")
        return {
            "project": project,
            "initialized": False,
            "error": str(e)
        }


@mcp.tool()
async def get_server_config(
    ctx: Context[ServerSession, AGraphContext] = None
) -> Dict[str, Any]:
    """
    Get current server configuration.
    
    Returns:
        Current server configuration including MCP server and AGraph settings
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        
        # Add runtime information
        runtime_info = {
            "active_projects": list(app_ctx.instances.keys()),
            "active_project_count": len(app_ctx.instances),
            "max_projects": app_ctx.config['mcp_server']['max_concurrent_projects']
        }
        
        return {
            **app_ctx.config,
            "runtime": runtime_info
        }
        
    except Exception as e:
        await ctx.error(f"Error getting server config: {str(e)}")
        return {"error": str(e)}


@mcp.tool()
async def list_available_projects(
    ctx: Context[ServerSession, AGraphContext] = None
) -> List[Dict[str, Any]]:
    """
    List all available projects in the project directory.
    
    Returns:
        List of available projects (both active and inactive)
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        projects = []
        
        # Scan project directory for available projects
        project_base = app_ctx.project_base_dir
        if project_base.exists():
            for project_path in project_base.iterdir():
                if project_path.is_dir() and not project_path.name.startswith('.'):
                    project_name = project_path.name
                    
                    # Check if project has knowledge graph data
                    has_data = any([
                        (project_path / "vector_store").exists(),
                        (project_path / "agraph_vectordb").exists(),
                        (project_path / "chroma").exists(),
                        (project_path / "knowledge_graph.json").exists(),
                        (project_path / "knowledge_graphs").exists(),
                        (project_path / "entities.json").exists(),
                        (project_path / "config.json").exists(),
                        (project_path / "document_storage").exists(),
                        (project_path / "cache").exists(),
                        list(project_path.glob("*.db")),
                        list(project_path.glob("*.index")),
                        list(project_path.glob("**/*.sqlite3")),
                        list(project_path.glob("**/chroma.sqlite3")),
                    ])
                    
                    project_info = {
                        "project_name": project_name,
                        "project_path": str(project_path),
                        "has_data": has_data,
                        "is_active": project_name in app_ctx.instances,
                        "status": "unknown"
                    }
                    
                    # Get stats if project is active
                    if project_name in app_ctx.instances:
                        try:
                            stats = await app_ctx.instances[project_name].get_stats()
                            project_info["status"] = "active"
                            project_info["stats"] = stats
                        except Exception as e:
                            project_info["status"] = "error"
                            project_info["error"] = str(e)
                    else:
                        project_info["status"] = "available" if has_data else "empty"
                    
                    projects.append(project_info)
        
        await ctx.info(f"Found {len(projects)} projects in {project_base}")
        return sorted(projects, key=lambda x: x["project_name"])
        
    except Exception as e:
        await ctx.error(f"Error listing available projects: {str(e)}")
        return []


@mcp.tool()
async def list_active_projects(
    ctx: Context[ServerSession, AGraphContext] = None
) -> List[Dict[str, Any]]:
    """
    List currently active projects (loaded in memory).
    
    Returns:
        List of active projects with basic statistics
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        projects = []
        
        for project_name, agraph in app_ctx.instances.items():
            try:
                stats = await agraph.get_stats()
                projects.append({
                    "project_name": project_name,
                    "status": "active",
                    "stats": stats
                })
            except Exception as e:
                projects.append({
                    "project_name": project_name,
                    "status": "error",
                    "error": str(e)
                })
        
        await ctx.info(f"Listed {len(projects)} active projects")
        return projects
        
    except Exception as e:
        await ctx.error(f"Error listing active projects: {str(e)}")
        return []


@mcp.tool()
async def cleanup_project(
    project: str,
    ctx: Context[ServerSession, AGraphContext] = None
) -> Dict[str, Any]:
    """
    Cleanup and remove a specific project instance.
    
    Args:
        project: Project name to cleanup
    
    Returns:
        Cleanup operation result
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        
        if project not in app_ctx.instances:
            return {
                "project": project,
                "status": "not_found",
                "message": f"Project '{project}' is not active"
            }
        
        await ctx.info(f"Cleaning up project: {project}")
        
        # Get the AGraph instance and cleanup
        agraph = app_ctx.instances[project]
        try:
            await agraph.__aexit__(None, None, None)
        except Exception as cleanup_error:
            await ctx.warning(f"Error during AGraph cleanup: {cleanup_error}")
        
        # Remove from instances
        del app_ctx.instances[project]
        
        await ctx.info(f"✅ Project '{project}' cleaned up successfully")
        return {
            "project": project,
            "status": "cleaned",
            "message": f"Project '{project}' has been cleaned up",
            "remaining_projects": len(app_ctx.instances)
        }
        
    except Exception as e:
        error = handle_agraph_error(e, "project cleanup", {"project": project})
        await ctx.error(f"Failed to cleanup project '{project}': {error.message}")
        return {
            "project": project,
            "status": "error",
            "error": error.message
        }


@mcp.tool()
async def validate_project(
    project: str,
    ctx: Context[ServerSession, AGraphContext] = None
) -> Dict[str, Any]:
    """
    Validate if a project has valid knowledge graph data.
    
    Args:
        project: Project name to validate
    
    Returns:
        Validation result with details
    """
    try:
        app_ctx = ctx.request_context.lifespan_context
        project_path = app_ctx.project_base_dir / project
        
        if not project_path.exists():
            return {
                "project": project,
                "valid": False,
                "reason": "Project directory does not exist",
                "path": str(project_path)
            }
        
        # Check for various knowledge graph data indicators
        data_indicators = {
            "vector_store": (project_path / "vector_store").exists(),
            "agraph_vectordb": (project_path / "agraph_vectordb").exists(),
            "chroma": (project_path / "chroma").exists(),
            "knowledge_graph": (project_path / "knowledge_graph.json").exists(),
            "entities": (project_path / "entities.json").exists(),
            "relations": (project_path / "relations.json").exists(),
            "config": (project_path / "config.json").exists(),
            "document_storage": (project_path / "document_storage").exists(),
            "database_files": bool(list(project_path.glob("*.db"))),
            "index_files": bool(list(project_path.glob("*.index"))),
            "pickle_files": bool(list(project_path.glob("*.pkl"))),
            "sqlite_files": bool(list(project_path.glob("**/*.sqlite3"))),
            "chroma_db": bool(list(project_path.glob("**/chroma.sqlite3"))),
        }
        
        has_data = any(data_indicators.values())
        
        # Get file count and size
        all_files = list(project_path.rglob("*"))
        file_count = len([f for f in all_files if f.is_file()])
        total_size = sum(f.stat().st_size for f in all_files if f.is_file())
        
        return {
            "project": project,
            "valid": has_data,
            "path": str(project_path),
            "data_indicators": data_indicators,
            "file_count": file_count,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "is_active": project in app_ctx.instances
        }
        
    except Exception as e:
        error = handle_agraph_error(e, "project validation", {"project": project})
        await ctx.error(f"Failed to validate project '{project}': {error.message}")
        return {
            "project": project,
            "valid": False,
            "error": error.message
        }


@mcp.tool()
async def set_project_directory(
    project_dir: str,
    ctx: Context[ServerSession, AGraphContext] = None
) -> Dict[str, Any]:
    """
    Update the project base directory configuration.
    
    Args:
        project_dir: New project directory path
    
    Returns:
        Operation result
    """
    try:
        from pathlib import Path
        
        # Validate the new directory
        new_path = Path(project_dir)
        if not new_path.is_absolute():
            new_path = Path.cwd() / project_dir
        
        # Create directory if it doesn't exist
        new_path.mkdir(parents=True, exist_ok=True)
        
        # Update configuration
        config_manager = get_config_manager()
        config_manager.agraph_config.project_dir = str(new_path)
        config_manager.save_config()
        
        # Update app context
        app_ctx = ctx.request_context.lifespan_context
        app_ctx.project_base_dir = new_path
        app_ctx.config['agraph']['project_dir'] = str(new_path)
        
        await ctx.info(f"Project directory updated to: {new_path}")
        
        # List available projects in new directory
        available_projects = []
        if new_path.exists():
            for project_path in new_path.iterdir():
                if project_path.is_dir() and not project_path.name.startswith('.'):
                    available_projects.append(project_path.name)
        
        return {
            "success": True,
            "old_directory": str(app_ctx.project_base_dir),
            "new_directory": str(new_path),
            "available_projects": sorted(available_projects),
            "project_count": len(available_projects)
        }
        
    except Exception as e:
        error = handle_agraph_error(e, "set project directory", {"project_dir": project_dir})
        await ctx.error(f"Failed to set project directory: {error.message}")
        return {
            "success": False,
            "error": error.message
        }


def main():
    """Run the AGraph MCP server."""
    try:
        mcp.run()
    except KeyboardInterrupt:
        print("\n⏹️  AGraph MCP Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()