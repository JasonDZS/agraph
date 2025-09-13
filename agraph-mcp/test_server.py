#!/usr/bin/env python3
"""
Simple test script for AGraph MCP Server (Direct AGraph Instance Version)
"""

import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add the parent directory to the path to import agraph
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from server import AGraphMCPServer
from config import config
from exceptions import AGraphMCPError, ConfigurationError

async def test_server_initialization():
    """Test server initialization"""
    print("Testing server initialization...")
    
    try:
        server = AGraphMCPServer()
        print("✅ Server initialized successfully")
        await server.cleanup()
    except Exception as e:
        print(f"❌ Server initialization failed: {e}")

async def test_agraph_initialization_mock():
    """Test AGraph initialization with mock"""
    print("\nTesting AGraph initialization with mock...")
    
    server = AGraphMCPServer()
    
    # Mock AGraph and its methods
    with patch('server.AGraph') as MockAGraph, \
         patch('server.get_settings') as mock_get_settings:
        
        mock_settings = MagicMock()
        mock_settings.workdir = config.workdir
        mock_get_settings.return_value = mock_settings
        
        mock_agraph_instance = AsyncMock()
        mock_agraph_instance.collection_name = "test_collection"
        mock_agraph_instance.initialize = AsyncMock()
        MockAGraph.return_value = mock_agraph_instance
        
        try:
            await server.initialize_agraph("test_collection")
            assert server.agraph == mock_agraph_instance
            print("✅ Mock AGraph initialization successful")
        except Exception as e:
            print(f"❌ Mock AGraph initialization failed: {e}")
    
    await server.cleanup()

async def test_knowledge_graph_status_mock():
    """Test knowledge graph status functionality with mock"""
    print("\nTesting knowledge graph status with mock...")
    
    server = AGraphMCPServer()
    
    # Mock AGraph instance and its methods
    mock_agraph = AsyncMock()
    mock_agraph.collection_name = "test_collection"
    mock_agraph.vector_store_type = "chroma"
    mock_agraph.enable_knowledge_graph = True
    mock_agraph.is_initialized = True
    mock_agraph.has_knowledge_graph = True
    
    # Mock statistics
    mock_stats = {
        "knowledge_graph": {
            "entities": 10,
            "relations": 15,
            "clusters": 3,
            "text_chunks": 20
        },
        "vector_store": {
            "total_documents": 100,
            "total_embeddings": 50
        }
    }
    mock_agraph.get_stats = AsyncMock(return_value=mock_stats)
    
    # Mock knowledge graph with sample data
    mock_entity = MagicMock()
    mock_entity.entity_type.value = "PERSON"
    
    mock_relation = MagicMock()
    mock_relation.relation_type.value = "WORKS_FOR"
    
    mock_kg = MagicMock()
    mock_kg.entities = {"entity1": mock_entity}
    mock_kg.relations = {"relation1": mock_relation}
    mock_agraph.knowledge_graph = mock_kg
    
    server.agraph = mock_agraph
    server._is_initialized = True
    
    try:
        result = await server._get_knowledge_graph_status({"collection_name": "test_collection"})
        assert len(result) == 1
        assert "test_collection" in result[0].text
        assert "Entities**: 10" in result[0].text
        print("✅ Knowledge graph status test successful")
    except Exception as e:
        print(f"❌ Knowledge graph status test failed: {e}")
    
    await server.cleanup()

async def test_search_entities_mock():
    """Test entity search functionality with mock"""
    print("\nTesting entity search with mock...")
    
    server = AGraphMCPServer()
    
    # Mock AGraph and knowledge graph
    mock_agraph = AsyncMock()
    mock_agraph.collection_name = "test_collection"
    
    # Create mock entities
    mock_entity1 = MagicMock()
    mock_entity1.id = "entity_001"
    mock_entity1.name = "人工智能"
    mock_entity1.entity_type.value = "CONCEPT"
    mock_entity1.description = "计算机科学的一个分支"
    mock_entity1.confidence = 0.95
    mock_entity1.aliases = ["AI", "Artificial Intelligence"]
    mock_entity1.properties = {"domain": "technology"}
    
    mock_entity2 = MagicMock()
    mock_entity2.id = "entity_002"
    mock_entity2.name = "张三"
    mock_entity2.entity_type.value = "PERSON"
    mock_entity2.description = "AI研究员"
    mock_entity2.confidence = 0.87
    mock_entity2.aliases = ["Dr. Zhang"]
    mock_entity2.properties = {"position": "研究员"}
    
    mock_kg = MagicMock()
    mock_kg.entities = {
        "entity_001": mock_entity1,
        "entity_002": mock_entity2
    }
    
    mock_agraph.knowledge_graph = mock_kg
    server.agraph = mock_agraph
    server._is_initialized = True
    
    try:
        # Test without filter
        result = await server._search_entities({})
        assert len(result) == 1
        assert "人工智能" in result[0].text
        assert "张三" in result[0].text
        assert "Found**: 2 entities" in result[0].text
        
        # Test with entity type filter
        with patch('server.EntityType') as MockEntityType:
            MockEntityType.return_value = mock_entity2.entity_type
            result = await server._search_entities({"entity_type": "PERSON"})
            # Note: This would filter in real implementation
        
        print("✅ Entity search test successful")
    except Exception as e:
        print(f"❌ Entity search test failed: {e}")
    
    await server.cleanup()

async def test_natural_language_search_mock():
    """Test natural language search functionality with mock"""
    print("\nTesting natural language search with mock...")
    
    server = AGraphMCPServer()
    
    # Mock AGraph instance
    mock_agraph = AsyncMock()
    mock_agraph.collection_name = "test_collection"
    mock_agraph.vector_store = MagicMock()  # Has vector store
    
    # Mock search results
    mock_entity = MagicMock()
    mock_entity.name = "人工智能"
    mock_entity.entity_type.value = "CONCEPT"
    mock_entity.description = "计算机科学的一个分支"
    
    mock_text_chunk = MagicMock()
    mock_text_chunk.id = "chunk_001"
    mock_text_chunk.source = "AI_introduction.pdf"
    mock_text_chunk.content = "人工智能是计算机科学的一个分支..."
    
    mock_agraph.search_entities = AsyncMock(return_value=[(mock_entity, 0.95)])
    mock_agraph.search_text_chunks = AsyncMock(return_value=[(mock_text_chunk, 0.90)])
    mock_agraph.search_relations = AsyncMock(return_value=[])
    
    # Mock knowledge graph for fallback
    mock_kg = MagicMock()
    mock_kg.entities = {"entity1": mock_entity}
    mock_kg.text_chunks = {"chunk1": mock_text_chunk}
    mock_kg.relations = {}
    mock_kg.clusters = {}
    mock_agraph.knowledge_graph = mock_kg
    
    server.agraph = mock_agraph
    server._is_initialized = True
    
    try:
        result = await server._natural_language_search({
            "query": "人工智能",
            "search_type": "all",
            "limit": 5
        })
        assert len(result) == 1
        assert "人工智能" in result[0].text
        assert "Vector Search" in result[0].text  # Should use vector search
        print("✅ Natural language search test successful")
    except Exception as e:
        print(f"❌ Natural language search test failed: {e}")
    
    await server.cleanup()

async def test_configuration():
    """Test configuration management"""
    print("\nTesting configuration...")
    
    try:
        # Test default configuration
        assert config.workdir is not None
        assert config.default_collection is not None
        assert config.default_entity_limit > 0
        
        # Test validation
        config.validate()
        print("✅ Configuration test successful")
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")

async def test_error_handling():
    """Test error handling"""
    print("\nTesting error handling...")
    
    server = AGraphMCPServer()
    
    try:
        # Test with uninitialized AGraph
        result = await server._get_knowledge_graph_status({})
        # Should initialize AGraph automatically
        
        # Test with invalid entity type
        server.agraph = AsyncMock()
        mock_kg = MagicMock()
        mock_kg.entities = {}
        server.agraph.knowledge_graph = mock_kg
        server._is_initialized = True
        
        result = await server._search_entities({"entity_type": "INVALID_TYPE"})
        assert "Invalid entity type" in result[0].text
        
        print("✅ Error handling test successful")
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    await server.cleanup()

async def main():
    """Run all tests"""
    print("🧪 Starting AGraph MCP Server Tests (Direct AGraph Instance Version)\n")
    print(f"Configuration: Workdir = {config.workdir}, Collection = {config.default_collection}")
    print("=" * 60)
    
    await test_configuration()
    await test_server_initialization()
    await test_agraph_initialization_mock()
    await test_knowledge_graph_status_mock()
    await test_search_entities_mock()
    await test_natural_language_search_mock()
    await test_error_handling()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())