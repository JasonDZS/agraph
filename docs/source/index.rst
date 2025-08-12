.. Knowledge Graph Toolkit documentation master file, created by
   sphinx-quickstart on Tue Aug 12 18:17:42 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Knowledge Graph Toolkit documentation
=====================================

Welcome to the Knowledge Graph Toolkit documentation! This package provides tools for creating,
managing, and analyzing knowledge graphs with focus on entities, relations, and text processing capabilities.

Features
--------

* **Entity Management**: Create and manage entities with types and properties
* **Relation Management**: Define and handle relationships between entities
* **Text Processing**: Advanced text analysis and processing capabilities
* **Configuration**: Flexible configuration system
* **CLI Interface**: Command-line tools for common operations

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install agraph

Basic usage:

.. code-block:: python

   from agraph import Entity, Relation, EntityType, RelationType

   # Create entities
   person = Entity(name="Alice", entity_type=EntityType.PERSON)
   company = Entity(name="TechCorp", entity_type=EntityType.ORGANIZATION)

   # Create relation
   relation = Relation(
       source=person,
       target=company,
       relation_type=RelationType.WORKS_FOR
   )

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   agraph
