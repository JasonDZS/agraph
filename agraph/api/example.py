"""Example usage of AGraph API."""

import asyncio
import json

import httpx


async def test_agraph_api() -> None:
    """Test AGraph API endpoints."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=60.0) as client:
        # 1. Health check
        print("1. Testing health check...")
        response = await client.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")

        # 2. Get configuration
        print("\n2. Testing get configuration...")
        response = await client.get(f"{base_url}/config")
        print(f"Config: {response.status_code} - {response.json()}")

        # 3. Upload texts for processing
        print("\n3. Testing text upload...")
        texts_data = {
            "texts": [
                "Apple Inc. is a technology company founded by Steve Jobs.",
                "Microsoft Corporation was founded by Bill Gates and Paul Allen.",
                "Google was founded by Larry Page and Sergey Brin at Stanford University.",
            ],
            "graph_name": "Tech Companies",
            "graph_description": "Knowledge graph about technology companies",
            "use_cache": True,
            "save_to_vector_store": True,
        }

        response = await client.post(f"{base_url}/documents/from-text", json=texts_data)
        print(f"Text upload: {response.status_code} - {response.json()}")

        # 4. Search text chunks
        print("\n4. Testing text chunk search...")
        search_data = {"query": "Steve Jobs Apple", "top_k": 5, "search_type": "text_chunks"}

        response = await client.post(f"{base_url}/search", json=search_data)
        print(f"Search: {response.status_code} - {response.json()}")

        # 5. Search entities (if knowledge graph is enabled)
        print("\n5. Testing entity search...")
        entity_search_data = {"query": "technology company", "top_k": 5, "search_type": "entities"}

        response = await client.post(f"{base_url}/search", json=entity_search_data)
        print(f"Entity search: {response.status_code} - {response.json()}")

        # 6. Chat with the knowledge base
        print("\n6. Testing chat...")
        chat_data = {
            "question": "Tell me about technology companies and their founders",
            "entity_top_k": 3,
            "relation_top_k": 3,
            "text_chunk_top_k": 3,
            "response_type": "详细回答",
            "stream": False,
        }

        response = await client.post(f"{base_url}/chat", json=chat_data)
        result = response.json()
        print(f"Chat: {response.status_code}")
        if response.status_code == 200:
            print(f"Question: {result['data']['question']}")
            print(f"Answer: {result['data']['answer'][:200]}...")
        else:
            print(f"Error: {result}")

        # 7. Get system statistics
        print("\n7. Testing system stats...")
        response = await client.get(f"{base_url}/system/stats")
        print(f"Stats: {response.status_code} - {response.json()}")

        # 8. Get build status
        print("\n8. Testing build status...")
        response = await client.get(f"{base_url}/system/build-status")
        print(f"Build status: {response.status_code} - {response.json()}")

        # 9. View cached text chunks
        print("\n9. Testing cached text chunks view...")
        response = await client.get(f"{base_url}/cache/text-chunks?page=1&page_size=5")
        result = response.json()
        print(f"Text chunks: {response.status_code}")
        if response.status_code == 200 and result["data"]["text_chunks"]:
            print(f"Found {result['data']['total_count']} text chunks")
            for chunk in result["data"]["text_chunks"][:2]:  # Show first 2
                print(f"  - {chunk['id']}: {chunk['content'][:100]}...")

        # 10. View cached entities
        print("\n10. Testing cached entities view...")
        response = await client.get(f"{base_url}/cache/entities?page=1&page_size=5")
        result = response.json()
        print(f"Entities: {response.status_code}")
        if response.status_code == 200 and result["data"]["entities"]:
            print(f"Found {result['data']['total_count']} entities")
            for entity in result["data"]["entities"][:2]:  # Show first 2
                print(
                    f"  - {entity['name']} ({entity['entity_type']}): {entity['description'] or 'No description'}"
                )

        # 11. View cached relations
        print("\n11. Testing cached relations view...")
        response = await client.get(f"{base_url}/cache/relations?page=1&page_size=5")
        result = response.json()
        print(f"Relations: {response.status_code}")
        if response.status_code == 200 and result["data"]["relations"]:
            print(f"Found {result['data']['total_count']} relations")
            for relation in result["data"]["relations"][:2]:  # Show first 2
                head_name = (
                    relation["head_entity"]["name"] if relation["head_entity"] else "Unknown"
                )
                tail_name = (
                    relation["tail_entity"]["name"] if relation["tail_entity"] else "Unknown"
                )
                print(f"  - {head_name} --[{relation['relation_type']}]--> {tail_name}")

        # 12. View cached clusters
        print("\n12. Testing cached clusters view...")
        response = await client.get(f"{base_url}/cache/clusters?page=1&page_size=5")
        result = response.json()
        print(f"Clusters: {response.status_code}")
        if response.status_code == 200 and result["data"]["clusters"]:
            print(f"Found {result['data']['total_count']} clusters")
            for cluster in result["data"]["clusters"][:2]:  # Show first 2
                print(
                    f"  - {cluster['name']}: {cluster['entity_count']} entities, {cluster['relation_count']} relations"
                )


async def test_file_upload() -> None:
    """Test file upload functionality."""
    base_url = "http://localhost:8000"

    # Create a test file
    test_content = """
    Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
    Machine Learning is a subset of AI that focuses on the ability of machines to receive data and learn for themselves.
    Deep Learning is a subset of Machine Learning that uses neural networks with many layers.
    """

    with open("/tmp/test_ai_document.txt", "w", encoding="utf-8") as f:
        f.write(test_content)

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Testing file upload...")

        with open("/tmp/test_ai_document.txt", "rb") as f:
            files = {"files": ("test_ai_document.txt", f, "text/plain")}
            data = {
                "graph_name": "AI Knowledge",
                "graph_description": "Knowledge graph about AI concepts",
                "use_cache": "true",
                "save_to_vector_store": "true",
            }

            response = await client.post(f"{base_url}/documents/upload", files=files, data=data)

            print(f"File upload: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Result: {json.dumps(result, indent=2)}")
            else:
                print(f"Error: {response.text}")


async def test_streaming_chat() -> None:
    """Test streaming chat functionality."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Testing streaming chat...")

        chat_data = {
            "question": "Explain artificial intelligence and machine learning",
            "entity_top_k": 3,
            "relation_top_k": 3,
            "text_chunk_top_k": 3,
            "response_type": "详细回答",
            "stream": True,
        }

        async with client.stream("POST", f"{base_url}/chat/stream", json=chat_data) as response:
            print(f"Streaming chat status: {response.status_code}")

            if response.status_code == 200:
                async for chunk in response.aiter_text():
                    if chunk.strip() and chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:])  # Remove "data: " prefix
                            print(f"Chunk: {data.get('chunk', '')}", end="", flush=True)
                            if data.get("finished"):
                                print("\n\nStreaming completed.")
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"Error: {response.text}")


async def test_cache_viewing() -> None:
    """Test cache viewing functionality."""
    base_url = "http://localhost:8000"

    async with httpx.AsyncClient(timeout=60.0) as client:
        print("Testing cache viewing functionality...")

        # Test text chunks with filter
        print("\n1. Testing text chunks with filter...")
        response = await client.get(
            f"{base_url}/cache/text-chunks?filter_by=technology&page_size=3"
        )
        if response.status_code == 200:
            result = response.json()
            print(f"Found {result['data']['total_count']} text chunks matching 'technology'")
            for chunk in result["data"]["text_chunks"]:
                print(f"  - {chunk['id']}: {chunk['content'][:150]}...")

        # Test entities with pagination
        print("\n2. Testing entities pagination...")
        response = await client.get(f"{base_url}/cache/entities?page=1&page_size=5")
        if response.status_code == 200:
            result = response.json()
            print(
                f"Page 1 of entities (Total: {result['data']['total_count']}, Pages: {result['data']['total_pages']})"
            )
            for entity in result["data"]["entities"]:
                aliases_str = ", ".join(entity["aliases"][:3]) if entity["aliases"] else "None"
                print(f"  - {entity['name']} ({entity['entity_type']})")
                print(f"    Confidence: {entity['confidence']}, Aliases: {aliases_str}")
                if entity["description"]:
                    print(f"    Description: {entity['description'][:100]}...")

        # Test relations with detailed view
        print("\n3. Testing relations detailed view...")
        response = await client.get(f"{base_url}/cache/relations?page_size=3")
        if response.status_code == 200:
            result = response.json()
            print(f"Relations (Total: {result['data']['total_count']})")
            for relation in result["data"]["relations"]:
                head_info = relation["head_entity"]
                tail_info = relation["tail_entity"]
                print(f"  - Relation: {relation['id']}")
                print(
                    f"    {head_info['name'] if head_info else 'Unknown'} --[{relation['relation_type']}]--> {tail_info['name'] if tail_info else 'Unknown'}"
                )
                print(f"    Confidence: {relation['confidence']}")
                if relation["description"]:
                    print(f"    Description: {relation['description']}")

        # Test clusters with entity details
        print("\n4. Testing clusters with entity details...")
        response = await client.get(f"{base_url}/cache/clusters?page_size=2")
        if response.status_code == 200:
            result = response.json()
            print(f"Clusters (Total: {result['data']['total_count']})")
            for cluster in result["data"]["clusters"]:
                print(f"  - Cluster: {cluster['name']}")
                print(f"    Description: {cluster['description'] or 'No description'}")
                print(
                    f"    Contains {cluster['entity_count']} entities and {cluster['relation_count']} relations"
                )
                if cluster["entities"]:
                    print("    Entities:")
                    for entity in cluster["entities"][:3]:  # Show first 3
                        print(f"      * {entity['name']} ({entity['entity_type']})")
                if cluster["relations"]:
                    print("    Relations:")
                    for relation in cluster["relations"][:2]:  # Show first 2
                        print(
                            f"      * {relation['head_entity_name']} --[{relation['relation_type']}]--> {relation['tail_entity_name']}"
                        )


if __name__ == "__main__":
    print("AGraph API Test Suite")
    print("=" * 50)

    print("\nMake sure to start the API server first:")
    print("uvicorn agraph.api.app:app --reload --host 0.0.0.0 --port 8000")
    print("\nRunning tests...\n")

    # Run basic API tests
    asyncio.run(test_agraph_api())

    print("\n" + "=" * 50)
    print("Testing file upload...")
    asyncio.run(test_file_upload())

    print("\n" + "=" * 50)
    print("Testing streaming chat...")
    asyncio.run(test_streaming_chat())

    print("\n" + "=" * 50)
    print("Testing cache viewing...")
    asyncio.run(test_cache_viewing())

    print("\n" + "=" * 50)
    print("All tests completed!")
