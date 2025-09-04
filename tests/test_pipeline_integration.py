"""
Integration tests for the new pipeline-based KnowledgeGraphBuilder.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, MagicMock, Mock
from typing import List

from agraph.base.models.text import TextChunk
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation
from agraph.base.models.clusters import Cluster
from agraph.base.graphs.optimized import OptimizedKnowledgeGraph
from agraph.base.core.types import EntityType, RelationType
from agraph.builder.builder_v2 import KnowledgeGraphBuilderV2
from agraph.builder.compatibility import (
    BackwardCompatibleKnowledgeGraphBuilder,
    MigrationHelper,
    quick_migration_test
)
from agraph.builder.steps.context import BuildContext
from agraph.builder.pipeline import BuildPipeline
from agraph.config import BuilderConfig, BuildSteps


class TestPipelineIntegration(unittest.TestCase):
    """Test pipeline integration with KnowledgeGraphBuilderV2."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock configuration to avoid external dependencies
        self.config = BuilderConfig()
        self.config.llm_provider = "mock"
        self.config.llm_model = "mock-model"
        self.config.cache_dir = "/tmp/test_cache"
        
        # Sample test data
        self.test_texts = [
            "Apple Inc. is a technology company founded by Steve Jobs.",
            "Microsoft Corporation was founded by Bill Gates and Paul Allen.",
            "Google was founded by Larry Page and Sergey Brin at Stanford University."
        ]
        
        self.test_documents = [
            "/tmp/test_doc1.txt",
            "/tmp/test_doc2.pdf"
        ]
    
    def test_pipeline_builder_initialization(self):
        """Test that pipeline builder initializes correctly."""
        builder = KnowledgeGraphBuilderV2(
            config=self.config,
            enable_knowledge_graph=True
        )
        
        # Check that all components are initialized
        self.assertIsNotNone(builder.cache_manager)
        self.assertIsNotNone(builder.text_chunker_handler)
        self.assertIsNotNone(builder.entity_handler)
        self.assertIsNotNone(builder.relation_handler)
        self.assertIsNotNone(builder.cluster_handler)
        self.assertIsNotNone(builder.graph_assembler)
        self.assertIsNotNone(builder.pipeline_factory)
        self.assertTrue(builder.enable_knowledge_graph)
    
    def test_build_context_creation(self):
        """Test build context creation and configuration."""
        context = BuildContext(
            texts=self.test_texts,
            graph_name="test_graph",
            graph_description="Test graph description",
            use_cache=True,
            enable_knowledge_graph=True
        )
        
        self.assertEqual(context.texts, self.test_texts)
        self.assertEqual(context.graph_name, "test_graph")
        self.assertEqual(context.graph_description, "Test graph description")
        self.assertTrue(context.use_cache)
        self.assertTrue(context.enable_knowledge_graph)
        self.assertIsNone(context.from_step)
    
    def test_build_context_step_control(self):
        """Test build context step control logic."""
        context = BuildContext(
            texts=self.test_texts,
            from_step=BuildSteps.ENTITY_EXTRACTION,
            enable_knowledge_graph=False
        )
        
        # Should execute steps at or after from_step
        self.assertFalse(context.should_execute_step(BuildSteps.TEXT_CHUNKING))
        self.assertTrue(context.should_execute_step(BuildSteps.ENTITY_EXTRACTION))
        self.assertTrue(context.should_execute_step(BuildSteps.RELATION_EXTRACTION))
        
        # Should skip knowledge graph steps when disabled
        self.assertTrue(context.should_skip_step(BuildSteps.ENTITY_EXTRACTION))
        self.assertTrue(context.should_skip_step(BuildSteps.RELATION_EXTRACTION))
        self.assertTrue(context.should_skip_step(BuildSteps.CLUSTER_FORMATION))
        self.assertFalse(context.should_skip_step(BuildSteps.TEXT_CHUNKING))
    
    def test_pipeline_factory_creation(self):
        """Test pipeline factory creates correct pipelines."""
        builder = KnowledgeGraphBuilderV2(config=self.config)
        
        # Test text-only pipeline creation
        text_pipeline = builder.pipeline_factory.create_text_only_pipeline(
            builder.text_chunker_handler,
            builder.entity_handler,
            builder.relation_handler,
            builder.cluster_handler,
            builder.graph_assembler
        )
        
        self.assertIsInstance(text_pipeline, BuildPipeline)
        self.assertEqual(len(text_pipeline), 5)  # 5 steps for text-only
        
        # Test minimal pipeline creation
        minimal_pipeline = builder.pipeline_factory.create_minimal_pipeline(
            builder.text_chunker_handler,
            builder.graph_assembler
        )
        
        self.assertIsInstance(minimal_pipeline, BuildPipeline)
        self.assertEqual(len(minimal_pipeline), 2)  # Only 2 steps for minimal
    
    def test_custom_pipeline_creation(self):
        """Test custom pipeline creation with builder pattern."""
        builder = KnowledgeGraphBuilderV2(config=self.config)
        
        # Create custom pipeline using builder pattern
        custom_pipeline = (builder.pipeline_builder
            .with_text_chunking(builder.text_chunker_handler)
            .with_entity_extraction(builder.entity_handler)
            .with_graph_assembly(builder.graph_assembler)
            .build())
        
        self.assertIsInstance(custom_pipeline, BuildPipeline)
        self.assertEqual(len(custom_pipeline), 3)
        
        step_names = custom_pipeline.get_step_names()
        self.assertIn(BuildSteps.TEXT_CHUNKING, step_names)
        self.assertIn(BuildSteps.ENTITY_EXTRACTION, step_names)
        self.assertIn(BuildSteps.GRAPH_ASSEMBLY, step_names)
        self.assertNotIn(BuildSteps.RELATION_EXTRACTION, step_names)


class TestMockPipelineExecution(unittest.TestCase):
    """Test pipeline execution with mocked components."""
    
    def setUp(self):
        """Set up mocked test environment."""
        self.config = BuilderConfig()
        self.config.llm_provider = "mock"
        
        # Create sample data
        self.test_texts = ["Sample text for testing pipeline execution."]
        
        # Mock components will be created in test methods
    
    def test_mock_pipeline_execution_success(self):
        """Test successful pipeline execution with mocked components."""
        # This test would require extensive mocking of all handlers
        # For now, we'll test the structure
        
        builder = KnowledgeGraphBuilderV2(config=self.config)
        context = BuildContext(
            texts=self.test_texts,
            graph_name="mock_test"
        )
        
        # Verify context is properly configured
        self.assertEqual(context.texts, self.test_texts)
        self.assertEqual(context.graph_name, "mock_test")
        
        # Verify pipeline can be created
        pipeline = builder.pipeline_factory.create_text_only_pipeline(
            builder.text_chunker_handler,
            builder.entity_handler,
            builder.relation_handler,
            builder.cluster_handler,
            builder.graph_assembler
        )
        
        self.assertIsNotNone(pipeline)
        self.assertEqual(len(pipeline.get_step_names()), 5)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility features."""
    
    def setUp(self):
        """Set up compatibility test environment."""
        self.config = BuilderConfig()
        self.config.llm_provider = "mock"
        self.test_texts = ["Test text for compatibility testing."]
    
    def test_backward_compatible_wrapper(self):
        """Test backward compatible wrapper functionality."""
        # Test new implementation through wrapper
        new_builder = BackwardCompatibleKnowledgeGraphBuilder(
            use_legacy=False,
            show_deprecation_warnings=False,
            config=self.config
        )
        
        self.assertIsInstance(new_builder._builder, KnowledgeGraphBuilderV2)
        
        # Test that wrapper delegates attributes correctly
        self.assertIsNotNone(new_builder.cache_manager)
        self.assertIsNotNone(new_builder.pipeline_factory)
    
    def test_migration_helper_structure(self):
        """Test migration helper functionality structure."""
        # Test that migration helper methods exist and are callable
        self.assertTrue(callable(MigrationHelper.compare_implementations))
        self.assertTrue(callable(MigrationHelper.generate_migration_report))
        self.assertTrue(callable(MigrationHelper.create_migration_guide))
        
        # Test migration guide generation
        guide = MigrationHelper.create_migration_guide()
        self.assertIsInstance(guide, str)
        self.assertIn("Migration Guide", guide)
        self.assertIn("Step 1", guide)
    
    def test_quick_migration_test_structure(self):
        """Test quick migration test function structure.""" 
        # Test that quick migration test function exists
        self.assertTrue(callable(quick_migration_test))


class TestPipelineMetrics(unittest.TestCase):
    """Test pipeline metrics and monitoring."""
    
    def setUp(self):
        """Set up metrics test environment."""
        self.config = BuilderConfig()
        self.builder = KnowledgeGraphBuilderV2(config=self.config)
    
    def test_pipeline_metrics_collection(self):
        """Test that pipeline metrics are collected."""
        # Create a pipeline
        pipeline = self.builder.pipeline_factory.create_minimal_pipeline(
            self.builder.text_chunker_handler,
            self.builder.graph_assembler
        )
        
        # Test metrics structure
        metrics = pipeline.get_pipeline_metrics()
        self.assertIn("pipeline_info", metrics)
        self.assertIn("execution_metrics", metrics)
        self.assertIn("step_metrics", metrics)
        
        # Test pipeline info
        pipeline_info = metrics["pipeline_info"]
        self.assertIn("total_steps", pipeline_info)
        self.assertIn("step_names", pipeline_info)
        self.assertEqual(pipeline_info["total_steps"], 2)
    
    def test_step_metrics_initialization(self):
        """Test step metrics are properly initialized."""
        pipeline = self.builder.pipeline_factory.create_text_only_pipeline(
            self.builder.text_chunker_handler,
            self.builder.entity_handler,
            self.builder.relation_handler,
            self.builder.cluster_handler,
            self.builder.graph_assembler
        )
        
        # Initially, step metrics should be empty
        metrics = pipeline.get_pipeline_metrics()
        step_metrics = metrics["step_metrics"]
        self.assertEqual(len(step_metrics), 0)
        
        # Execution metrics should be initialized
        exec_metrics = metrics["execution_metrics"]
        self.assertEqual(exec_metrics["total_executions"], 0)
        self.assertEqual(exec_metrics["successful_executions"], 0)
        self.assertEqual(exec_metrics["failed_executions"], 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in pipeline architecture."""
    
    def setUp(self):
        """Set up error handling test environment."""
        self.config = BuilderConfig()
        self.builder = KnowledgeGraphBuilderV2(config=self.config)
    
    def test_build_context_error_tracking(self):
        """Test error tracking in build context."""
        context = BuildContext(texts=["test"])
        
        # Initially no errors
        self.assertEqual(len(context.errors), 0)
        self.assertEqual(len(context.warnings), 0)
        
        # Add error
        test_error = Exception("Test error")
        context.add_error(test_error, BuildSteps.ENTITY_EXTRACTION)
        
        self.assertEqual(len(context.errors), 1)
        self.assertEqual(context.errors[0], test_error)
        
        # Add warning
        context.add_warning("Test warning")
        self.assertEqual(len(context.warnings), 1)
        self.assertEqual(context.warnings[0], "Test warning")
    
    def test_build_context_validation(self):
        """Test build context validation."""
        # Valid context
        valid_context = BuildContext(
            texts=["test text"],
            enable_knowledge_graph=True
        )
        valid_context.chunks = [TextChunk(content="test", chunk_id="1")]
        valid_context.mark_step_completed(BuildSteps.TEXT_CHUNKING, 100.0)
        
        errors = valid_context.validate_state()
        # Should have no validation errors for basic valid state
        self.assertIsInstance(errors, list)
        
        # Invalid context - no texts
        invalid_context = BuildContext(texts=[])
        errors = invalid_context.validate_state()
        self.assertGreater(len(errors), 0)
        self.assertIn("No input texts provided", errors[0])


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)