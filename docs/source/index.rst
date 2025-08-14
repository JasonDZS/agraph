.. Knowledge Graph Toolkit documentation master file, created by
   sphinx-quickstart on Tue Aug 12 18:17:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AGraph - Knowledge Graph Toolkit Documentation
=============================================

Welcome to AGraph documentation! AGraph is a powerful knowledge graph toolkit that enables you to build,
manage, and analyze knowledge graphs from text documents with advanced AI capabilities including semantic search,
intelligent Q&A, and vector-based storage.

Features
--------

* **Intelligent Knowledge Graph Construction**: Automatically build knowledge graphs from text using LLMs
* **Semantic Search**: Advanced vector-based search for entities and text chunks
* **Smart Q&A System**: Ask questions and get contextual answers from your knowledge graph
* **Multiple Vector Store Support**: Support for Chroma, in-memory, and custom vector databases
* **Document Processing**: Handle various document formats (txt, md, json, csv, pdf, docx, html)
* **Caching System**: Intelligent caching for improved performance
* **Streaming Responses**: Real-time streaming for chat interactions
* **Persistent Storage**: Save and load knowledge graphs across sessions

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install -e .

Set up environment:

.. code-block:: bash

   export OPENAI_API_KEY=your_api_key_here

Basic usage:

.. code-block:: python

   import asyncio
   from agraph import AGraph, get_settings

   async def main():
       # Configure settings
       settings = get_settings()
       settings.workdir = "workdir/my_project"

       # Initialize AGraph
       async with AGraph(
           collection_name="my_knowledge_graph",
           persist_directory=settings.workdir,
           vector_store_type="chroma",
           use_openai_embeddings=True
       ) as agraph:
           await agraph.initialize()

           # Build knowledge graph from texts
           texts = [
               "AI company focused on machine learning.",
               "Team of 50 engineers in Beijing office."
           ]

           graph = await agraph.build_from_texts(
               texts=texts,
               graph_name="Company Graph",
               use_cache=True,
               save_to_vector_store=True
           )

           # Search entities
           entities = await agraph.search_entities("company", top_k=3)

           # Ask questions
           response = await agraph.chat("What does the company do?")
           print(response)

   asyncio.run(main())

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   agraph

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   agraph_tutorial.md
   vectordb_tutorial.md
   import_export_tutorial.md
   quick_start_import_export.md
   graphml_integration_guide.md
   custom_vectordb_guide.md
