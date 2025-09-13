#!/usr/bin/env python3
"""
Example usage of AGraph MCP Server functionality

This script demonstrates how the MCP server methods work internally,
useful for testing and understanding the functionality.
"""

import asyncio
import sys
from pathlib import Path

# Add the parent directory to the path to import agraph
current_dir = Path(__file__).parent.absolute()
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

from server import AGraphMCPServer


async def example_usage():
    """Demonstrate MCP server functionality"""
    print("🚀 AGraph MCP Server Example Usage\n")
    print("=" * 50)

    # Create server instance
    server = AGraphMCPServer()

    try:
        print("\n1. 📊 Getting Knowledge Graph Status...")
        status_result = await server._get_knowledge_graph_status({})
        print("Status:", status_result[0].text[:200] + "..." if len(status_result[0].text) > 200 else status_result[0].text)

        print("\n2. 🔍 Searching Entities...")
        entities_result = await server._search_entities({"limit": 5})
        print("Entities:", entities_result[0].text[:200] + "..." if len(entities_result[0].text) > 200 else entities_result[0].text)

        print("\n3. 🔗 Searching Relations...")
        relations_result = await server._search_relations({"limit": 3})
        print("Relations:", relations_result[0].text[:200] + "..." if len(relations_result[0].text) > 200 else relations_result[0].text)

        print("\n4. 📄 Searching Text Chunks...")
        text_chunks_result = await server._search_text_chunks({"limit": 3})
        print("Text Chunks:", text_chunks_result[0].text[:200] + "..." if len(text_chunks_result[0].text) > 200 else text_chunks_result[0].text)

        print("\n5. 🧩 Searching Clusters...")
        clusters_result = await server._search_clusters({"limit": 3})
        print("Clusters:", clusters_result[0].text[:200] + "..." if len(clusters_result[0].text) > 200 else clusters_result[0].text)

        print("\n6. 🌐 Getting Full Knowledge Graph...")
        full_kg_result = await server._get_full_knowledge_graph({"include_clusters": True})
        print("Full KG:", full_kg_result[0].text[:200] + "..." if len(full_kg_result[0].text) > 200 else full_kg_result[0].text)

        print("\n7. 🗣️ Natural Language Search...")
        nl_search_result = await server._natural_language_search({
            "query": "人工智能",
            "search_type": "all",
            "limit": 5
        })
        print("NL Search:", nl_search_result[0].text[:300] + "..." if len(nl_search_result[0].text) > 300 else nl_search_result[0].text)

    except Exception as e:
        print(f"❌ Error during example usage: {e}")

    finally:
        print("\n🧹 Cleaning up...")
        await server.cleanup()
        print("✅ Example completed!")


if __name__ == "__main__":
    print("Note: This example will create AGraph instances and may take some time to initialize.")
    print("Make sure you have the required dependencies installed.\n")

    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\n⏹️ Example interrupted by user")
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("Make sure AGraph is properly installed and configured.")
