"""
Backward compatibility utilities and migration helpers.

This module provides utilities to help users migrate from the old
KnowledgeGraphBuilder implementation to the new pipeline-based architecture.
"""

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..logger import logger
from .builder_legacy_backup import KnowledgeGraphBuilder as OriginalKnowledgeGraphBuilder
from .builder_v2 import KnowledgeGraphBuilderV2


class DeprecationWarning(UserWarning):
    """Custom warning for deprecated functionality."""
    pass


class BackwardCompatibleKnowledgeGraphBuilder:
    """
    Backward compatible wrapper for KnowledgeGraphBuilder.
    
    This class provides the same interface as the original KnowledgeGraphBuilder
    but with warnings and migration guidance for deprecated patterns.
    """
    
    def __init__(
        self,
        use_legacy: bool = False,
        show_deprecation_warnings: bool = True,
        **kwargs
    ):
        """
        Initialize backward compatible builder.
        
        Args:
            use_legacy: If True, use original implementation
            show_deprecation_warnings: Whether to show deprecation warnings
            **kwargs: Arguments passed to the underlying builder
        """
        self.use_legacy = use_legacy
        self.show_deprecation_warnings = show_deprecation_warnings
        
        if use_legacy:
            if show_deprecation_warnings:
                warnings.warn(
                    "Using legacy KnowledgeGraphBuilder. "
                    "Consider migrating to the new pipeline-based architecture. "
                    "Set use_legacy=False to use the new implementation.",
                    DeprecationWarning,
                    stacklevel=2
                )
            self._builder = OriginalKnowledgeGraphBuilder(**kwargs)
            logger.info("Initialized legacy KnowledgeGraphBuilder")
        else:
            self._builder = KnowledgeGraphBuilderV2(**kwargs)
            logger.info("Initialized pipeline-based KnowledgeGraphBuilderV2")
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying builder."""
        return getattr(self._builder, name)
    
    async def build_from_text(self, *args, **kwargs):
        """Build from text with potential migration warnings."""
        if self.show_deprecation_warnings and hasattr(self, '_warn_about_methods'):
            self._warn_about_build_methods()
        
        return await self._builder.build_from_text(*args, **kwargs)
    
    async def build_from_documents(self, *args, **kwargs):
        """Build from documents with potential migration warnings.""" 
        if self.show_deprecation_warnings and hasattr(self, '_warn_about_methods'):
            self._warn_about_build_methods()
        
        return await self._builder.build_from_documents(*args, **kwargs)
    
    def _warn_about_build_methods(self):
        """Warn about potential improvements in new architecture."""
        if not self.use_legacy:
            return
            
        warnings.warn(
            "The legacy build methods have known performance and maintainability issues. "
            "The new pipeline-based architecture provides better error handling, "
            "caching, and extensibility. Consider migrating by setting use_legacy=False.",
            DeprecationWarning,
            stacklevel=3
        )


class MigrationHelper:
    """Helper class for migrating from old to new architecture."""
    
    @staticmethod
    def compare_implementations(
        texts: List[str],
        graph_name: str = "test",
        **builder_kwargs
    ) -> Dict[str, Any]:
        """
        Compare old vs new implementation performance and results.
        
        Args:
            texts: Test texts to process
            graph_name: Name for the test graph
            **builder_kwargs: Arguments for builder initialization
            
        Returns:
            Comparison results
        """
        import time
        import asyncio
        
        results = {
            "legacy": {"success": False, "time": 0.0, "error": None},
            "pipeline": {"success": False, "time": 0.0, "error": None},
            "comparison": {}
        }
        
        # Test legacy implementation
        try:
            start_time = time.time()
            legacy_builder = OriginalKnowledgeGraphBuilder(**builder_kwargs)
            legacy_kg = asyncio.run(legacy_builder.build_from_text(
                texts, graph_name=graph_name
            ))
            results["legacy"]["time"] = time.time() - start_time
            results["legacy"]["success"] = True
            results["legacy"]["entities"] = len(legacy_kg.entities)
            results["legacy"]["relations"] = len(legacy_kg.relations)
        except Exception as e:
            results["legacy"]["error"] = str(e)
        
        # Test pipeline implementation
        try:
            start_time = time.time()
            pipeline_builder = KnowledgeGraphBuilderV2(**builder_kwargs)
            pipeline_kg = asyncio.run(pipeline_builder.build_from_text(
                texts, graph_name=graph_name
            ))
            results["pipeline"]["time"] = time.time() - start_time
            results["pipeline"]["success"] = True
            results["pipeline"]["entities"] = len(pipeline_kg.entities)
            results["pipeline"]["relations"] = len(pipeline_kg.relations)
        except Exception as e:
            results["pipeline"]["error"] = str(e)
        
        # Generate comparison
        if results["legacy"]["success"] and results["pipeline"]["success"]:
            time_diff = results["pipeline"]["time"] - results["legacy"]["time"]
            results["comparison"] = {
                "time_difference": time_diff,
                "time_improvement_percent": (time_diff / results["legacy"]["time"]) * 100,
                "entities_match": results["legacy"]["entities"] == results["pipeline"]["entities"],
                "relations_match": results["legacy"]["relations"] == results["pipeline"]["relations"]
            }
        
        return results
    
    @staticmethod
    def generate_migration_report(comparison_results: Dict[str, Any]) -> str:
        """Generate a human-readable migration report."""
        report = ["# Migration Comparison Report\n"]
        
        # Legacy results
        legacy = comparison_results["legacy"]
        report.append(f"## Legacy Implementation")
        report.append(f"- Success: {legacy['success']}")
        report.append(f"- Execution Time: {legacy.get('time', 0):.2f}s")
        if legacy.get('error'):
            report.append(f"- Error: {legacy['error']}")
        else:
            report.append(f"- Entities: {legacy.get('entities', 0)}")
            report.append(f"- Relations: {legacy.get('relations', 0)}")
        report.append("")
        
        # Pipeline results
        pipeline = comparison_results["pipeline"]
        report.append(f"## Pipeline Implementation")
        report.append(f"- Success: {pipeline['success']}")
        report.append(f"- Execution Time: {pipeline.get('time', 0):.2f}s")
        if pipeline.get('error'):
            report.append(f"- Error: {pipeline['error']}")
        else:
            report.append(f"- Entities: {pipeline.get('entities', 0)}")
            report.append(f"- Relations: {pipeline.get('relations', 0)}")
        report.append("")
        
        # Comparison
        comparison = comparison_results.get("comparison", {})
        if comparison:
            report.append("## Comparison")
            time_improvement = comparison.get('time_improvement_percent', 0)
            if time_improvement < 0:
                report.append(f"- ✅ Pipeline is {abs(time_improvement):.1f}% faster")
            else:
                report.append(f"- ⚠️ Pipeline is {time_improvement:.1f}% slower")
            
            if comparison.get('entities_match', False):
                report.append("- ✅ Entity counts match")
            else:
                report.append("- ⚠️ Entity counts differ")
                
            if comparison.get('relations_match', False):
                report.append("- ✅ Relation counts match")
            else:
                report.append("- ⚠️ Relation counts differ")
        
        report.append("\n## Recommendation")
        if pipeline['success'] and not legacy['success']:
            report.append("✅ **Migrate to pipeline implementation** - Legacy version failed")
        elif legacy['success'] and not pipeline['success']:
            report.append("⚠️ **Stay with legacy for now** - Pipeline version has issues")
        elif comparison and comparison.get('time_improvement_percent', 0) < -10:
            report.append("✅ **Migrate to pipeline implementation** - Significant performance improvement")
        elif comparison:
            report.append("✅ **Consider migrating to pipeline implementation** - Better architecture and maintainability")
        else:
            report.append("ℹ️ **Further testing recommended** - Results inconclusive")
        
        return "\n".join(report)
    
    @staticmethod
    def create_migration_guide() -> str:
        """Create a step-by-step migration guide."""
        guide = [
            "# Migration Guide: Legacy to Pipeline Architecture\n",
            
            "## Step 1: Test Current Implementation",
            "```python",
            "from agraph.builder.compatibility import MigrationHelper",
            "",
            "# Compare implementations with your data",
            "results = MigrationHelper.compare_implementations(",
            "    texts=your_texts,",
            "    graph_name='migration_test'",
            ")",
            "print(MigrationHelper.generate_migration_report(results))",
            "```\n",
            
            "## Step 2: Update Import Statement",
            "```python",
            "# Old way:",
            "from agraph.builder import KnowledgeGraphBuilder",
            "",
            "# New way:",
            "from agraph.builder.builder_v2 import KnowledgeGraphBuilderV2 as KnowledgeGraphBuilder",
            "",
            "# Or use compatibility wrapper:",
            "from agraph.builder.compatibility import BackwardCompatibleKnowledgeGraphBuilder",
            "builder = BackwardCompatibleKnowledgeGraphBuilder(use_legacy=False)",
            "```\n",
            
            "## Step 3: No Code Changes Required",
            "The new implementation maintains the same public API:",
            "```python",
            "builder = KnowledgeGraphBuilder()",
            "kg = await builder.build_from_text(texts)",
            "# All existing methods work the same way",
            "```\n",
            
            "## Step 4: Optional - Leverage New Features",
            "```python",
            "# Access new pipeline features",
            "custom_pipeline = builder.create_custom_pipeline({",
            "    'text_chunking': builder.text_chunker_handler,",
            "    'entity_extraction': builder.entity_handler",
            "})",
            "",
            "# Get detailed metrics",
            "metrics = builder.get_pipeline_metrics()",
            "```\n",
            
            "## Benefits of New Architecture",
            "- ✅ Better error handling and recovery",
            "- ✅ Improved caching and performance",
            "- ✅ Modular, testable components",
            "- ✅ Extensible pipeline system",
            "- ✅ Better logging and debugging",
            "- ✅ Support for custom steps",
            "",
            "## Troubleshooting",
            "If you encounter issues:",
            "1. Use the compatibility wrapper with `use_legacy=True`",
            "2. Compare results using `MigrationHelper.compare_implementations()`", 
            "3. Check logs for detailed error information",
            "4. Report issues with specific error messages"
        ]
        
        return "\n".join(guide)


# Convenience function for quick migration testing
def quick_migration_test(
    texts: List[str], 
    graph_name: str = "migration_test",
    **kwargs
) -> None:
    """
    Quickly test migration with sample data.
    
    Args:
        texts: Sample texts to test with
        graph_name: Name for test graph
        **kwargs: Builder configuration
    """
    print("🔄 Running migration comparison test...\n")
    
    results = MigrationHelper.compare_implementations(
        texts=texts,
        graph_name=graph_name,
        **kwargs
    )
    
    report = MigrationHelper.generate_migration_report(results)
    print(report)
    
    print("\n" + "="*50)
    print("📖 For detailed migration guide, run:")
    print("print(MigrationHelper.create_migration_guide())")
    print("="*50)


# Convenience alias for legacy compatibility
LegacyKnowledgeGraphBuilder = BackwardCompatibleKnowledgeGraphBuilder