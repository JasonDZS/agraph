"""
Unit tests for build steps abstraction.
"""

import unittest
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock

from agraph.base.core.types import EntityType, RelationType
from agraph.base.graphs.optimized import KnowledgeGraph
from agraph.base.models.clusters import Cluster
from agraph.base.models.entities import Entity
from agraph.base.models.relations import Relation
from agraph.base.models.text import TextChunk
from agraph.builder.cache import CacheManager
from agraph.builder.pipeline import BuildPipeline
from agraph.builder.steps.base import BuildStep, StepError, StepResult
from agraph.builder.steps.context import BuildContext
from agraph.builder.steps.entity_extraction_step import EntityExtractionStep
from agraph.builder.steps.text_chunking_step import TextChunkingStep
from agraph.config import BuildSteps


class MockBuildStep(BuildStep):
    """Mock build step for testing."""

    def __init__(self, name: str, cache_manager: CacheManager, should_fail: bool = False):
        super().__init__(name, cache_manager)
        self.should_fail = should_fail
        self.execution_count = 0

    async def _execute_step(self, context: BuildContext) -> StepResult:
        self.execution_count += 1

        if self.should_fail:
            return StepResult.failure_result("Mock step failed")

        return StepResult.success_result(
            f"result_from_{self.name}", metadata={"execution_count": self.execution_count}
        )

    def _get_cache_input_data(self, context: BuildContext):
        return context.texts

    def _get_expected_result_type(self):
        return str


class TestStepResult(unittest.TestCase):
    """Test StepResult functionality."""

    def test_success_result_creation(self):
        """Test creating successful result."""
        data = ["item1", "item2"]
        metadata = {"count": 2}

        result = StepResult.success_result(data, metadata)

        self.assertTrue(result.is_success())
        self.assertFalse(result.is_failure())
        self.assertEqual(result.data, data)
        self.assertEqual(result.metadata, metadata)
        self.assertIsNone(result.error)

    def test_failure_result_creation(self):
        """Test creating failure result."""
        error_msg = "Test error"

        result = StepResult.failure_result(error_msg)

        self.assertFalse(result.is_success())
        self.assertTrue(result.is_failure())
        self.assertIsNone(result.data)
        self.assertIsNotNone(result.error)
        self.assertEqual(result.error.message, error_msg)

    def test_failure_result_from_exception(self):
        """Test creating failure result from exception."""
        exception = ValueError("Test exception")

        result = StepResult.failure_result(exception)

        self.assertFalse(result.is_success())
        self.assertEqual(result.error.cause, exception)
        self.assertEqual(result.error.message, str(exception))


class TestBuildContext(unittest.TestCase):
    """Test BuildContext functionality."""

    def setUp(self):
        """Set up test context."""
        self.texts = ["text1", "text2", "text3"]
        self.context = BuildContext(texts=self.texts, graph_name="test_graph", use_cache=True)

    def test_context_initialization(self):
        """Test context initialization."""
        self.assertEqual(self.context.texts, self.texts)
        self.assertEqual(self.context.graph_name, "test_graph")
        self.assertTrue(self.context.use_cache)
        self.assertIsNone(self.context.from_step)
        self.assertTrue(self.context.enable_knowledge_graph)

    def test_should_execute_step(self):
        """Test step execution logic."""
        # Without from_step, all steps should execute
        self.assertTrue(self.context.should_execute_step(BuildSteps.TEXT_CHUNKING))
        self.assertTrue(self.context.should_execute_step(BuildSteps.ENTITY_EXTRACTION))

        # With from_step, only steps at or after should execute
        self.context.from_step = BuildSteps.ENTITY_EXTRACTION
        self.assertFalse(self.context.should_execute_step(BuildSteps.TEXT_CHUNKING))
        self.assertTrue(self.context.should_execute_step(BuildSteps.ENTITY_EXTRACTION))
        self.assertTrue(self.context.should_execute_step(BuildSteps.RELATION_EXTRACTION))

    def test_should_skip_step(self):
        """Test step skipping logic."""
        # With knowledge graph enabled, no steps should be skipped
        self.assertFalse(self.context.should_skip_step(BuildSteps.ENTITY_EXTRACTION))
        self.assertFalse(self.context.should_skip_step(BuildSteps.RELATION_EXTRACTION))

        # With knowledge graph disabled, related steps should be skipped
        self.context.enable_knowledge_graph = False
        self.assertTrue(self.context.should_skip_step(BuildSteps.ENTITY_EXTRACTION))
        self.assertTrue(self.context.should_skip_step(BuildSteps.RELATION_EXTRACTION))
        self.assertTrue(self.context.should_skip_step(BuildSteps.CLUSTER_FORMATION))
        self.assertFalse(self.context.should_skip_step(BuildSteps.TEXT_CHUNKING))

    def test_step_completion_tracking(self):
        """Test step completion tracking."""
        step_name = BuildSteps.TEXT_CHUNKING
        result_data = ["chunk1", "chunk2"]
        metadata = {"count": 2}

        # Initially not completed
        self.assertFalse(self.context.is_step_completed(step_name))

        # Mark as completed
        self.context.mark_step_completed(step_name, 1000.0, result_data, metadata)  # end_time

        # Should now be completed
        self.assertTrue(self.context.is_step_completed(step_name))
        self.assertEqual(self.context.get_result_for_step(step_name), result_data)
        self.assertEqual(self.context.get_metadata_for_step(step_name), metadata)

    def test_execution_summary(self):
        """Test execution summary generation."""
        # Mark some steps as completed
        self.context.mark_step_completed(BuildSteps.TEXT_CHUNKING, 100.0, ["chunk1"])
        self.context.mark_step_skipped(BuildSteps.ENTITY_EXTRACTION, "disabled")

        summary = self.context.get_execution_summary()

        self.assertIn(BuildSteps.TEXT_CHUNKING, summary["completed_steps"])
        self.assertIn(BuildSteps.ENTITY_EXTRACTION, summary["skipped_steps"])
        self.assertGreaterEqual(summary["total_execution_time"], 0)


class TestBuildPipeline(unittest.TestCase):
    """Test BuildPipeline functionality."""

    def setUp(self):
        """Set up test pipeline."""
        self.cache_manager = Mock(spec=CacheManager)
        self.cache_manager.update_build_status = Mock()
        self.pipeline = BuildPipeline(self.cache_manager)

        # Create test context
        self.context = BuildContext(texts=["text1", "text2"], graph_name="test_graph")

    def test_pipeline_step_management(self):
        """Test adding and managing steps."""
        step1 = MockBuildStep("step1", self.cache_manager)
        step2 = MockBuildStep("step2", self.cache_manager)

        # Add steps
        self.pipeline.add_step(step1).add_step(step2)

        self.assertEqual(len(self.pipeline), 2)
        self.assertEqual(self.pipeline.get_step_names(), ["step1", "step2"])
        self.assertTrue(self.pipeline.has_step("step1"))
        self.assertFalse(self.pipeline.has_step("step3"))

        # Get step by name
        retrieved_step = self.pipeline.get_step_by_name("step1")
        self.assertEqual(retrieved_step, step1)

        # Remove step
        self.assertTrue(self.pipeline.remove_step("step1"))
        self.assertEqual(len(self.pipeline), 1)
        self.assertFalse(self.pipeline.has_step("step1"))

    async def test_pipeline_execution_success(self):
        """Test successful pipeline execution."""
        # Mock the final step to return a knowledge graph
        mock_kg = Mock(spec=KnowledgeGraph)
        mock_kg.name = "test_graph"
        mock_kg.entities = {}
        mock_kg.relations = {}

        # Create steps
        step1 = MockBuildStep("step1", self.cache_manager)
        step2 = MockBuildStep("step2", self.cache_manager)

        # Override step2 to set knowledge graph in context
        original_execute = step2._execute_step

        async def mock_execute(context):
            result = await original_execute(context)
            context.knowledge_graph = mock_kg
            return result

        step2._execute_step = mock_execute

        # Add steps to pipeline
        self.pipeline.add_step(step1).add_step(step2)

        # Execute pipeline
        result = await self.pipeline.execute(self.context)

        # Verify execution
        self.assertEqual(result, mock_kg)
        self.assertEqual(step1.execution_count, 1)
        self.assertEqual(step2.execution_count, 1)

        # Check pipeline metrics
        metrics = self.pipeline.get_pipeline_metrics()
        self.assertEqual(metrics["execution_metrics"]["successful_executions"], 1)
        self.assertEqual(metrics["execution_metrics"]["failed_executions"], 0)

    async def test_pipeline_execution_failure(self):
        """Test pipeline execution with failure."""
        step1 = MockBuildStep("step1", self.cache_manager)
        step2 = MockBuildStep("step2", self.cache_manager, should_fail=True)

        self.pipeline.add_step(step1).add_step(step2)

        # Execute pipeline - should fail
        with self.assertRaises(Exception) as cm:
            await self.pipeline.execute(self.context)

        self.assertIn("Mock step failed", str(cm.exception))

        # Check that step1 executed but step2 failed
        self.assertEqual(step1.execution_count, 1)
        self.assertEqual(step2.execution_count, 1)

        # Check pipeline metrics
        metrics = self.pipeline.get_pipeline_metrics()
        self.assertEqual(metrics["execution_metrics"]["failed_executions"], 1)

    async def test_pipeline_step_skipping(self):
        """Test pipeline step skipping based on context."""
        # Create context that skips knowledge graph steps
        context = BuildContext(texts=["text1"], enable_knowledge_graph=False)

        step1 = MockBuildStep(BuildSteps.TEXT_CHUNKING, self.cache_manager)
        step2 = MockBuildStep(BuildSteps.ENTITY_EXTRACTION, self.cache_manager)

        # Override to set knowledge graph for successful completion
        mock_kg = Mock(spec=KnowledgeGraph)
        mock_kg.name = "test"
        mock_kg.entities = {}
        mock_kg.relations = {}

        async def mock_execute(ctx):
            ctx.knowledge_graph = mock_kg
            return StepResult.success_result("dummy")

        step1._execute_step = mock_execute

        self.pipeline.add_step(step1).add_step(step2)

        # Execute pipeline
        result = await self.pipeline.execute(context)

        # step1 should execute, step2 should be skipped
        self.assertEqual(step1.execution_count, 1)
        self.assertEqual(step2.execution_count, 0)
        self.assertTrue(context.is_step_completed(BuildSteps.TEXT_CHUNKING))
        self.assertTrue(context.is_step_skipped(BuildSteps.ENTITY_EXTRACTION))


if __name__ == "__main__":
    # Run tests
    unittest.main()
