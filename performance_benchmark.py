"""
Pipeline性能基准测试和分析工具。

用于对比新旧架构的性能差异，识别瓶颈和优化机会。
"""

import asyncio
import time
import tracemalloc
import psutil
import gc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# 测试用的示例数据
SAMPLE_TEXTS = [
    "Apple Inc. is an American multinational technology company that specializes in consumer electronics.",
    "Microsoft Corporation is an American multinational technology corporation which produces computer software.",
    "Google LLC is an American multinational technology company that focuses on Internet-related services.",
    "Amazon.com Inc. is an American multinational technology company focusing on e-commerce and cloud computing.",
    "Tesla Inc. is an American electric vehicle and clean energy company based in Palo Alto, California."
] * 10  # 50个文本用于测试

LARGE_SAMPLE_TEXTS = SAMPLE_TEXTS * 20  # 1000个文本用于压力测试

@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    execution_time: float
    memory_peak_mb: float
    memory_current_mb: float
    cpu_percent: float
    success: bool
    error: Optional[str] = None
    step_timings: Dict[str, float] = None
    cache_hit_rate: float = 0.0
    
    def __post_init__(self):
        if self.step_timings is None:
            self.step_timings = {}


class PerformanceBenchmark:
    """Pipeline性能基准测试器"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results = {}
    
    async def run_legacy_benchmark(self, texts: List[str], iterations: int = 3) -> PerformanceMetrics:
        """运行Legacy实现的基准测试"""
        try:
            # 动态导入避免初始化问题 - 使用备份的legacy版本进行对比
            from agraph.builder.compatibility import BackwardCompatibleKnowledgeGraphBuilder
            
            total_time = 0
            memory_peaks = []
            cpu_usage = []
            all_step_timings = {}
            
            for i in range(iterations):
                print(f"Legacy测试 - 迭代 {i+1}/{iterations}")
                
                # 内存追踪
                tracemalloc.start()
                gc.collect()
                
                start_memory = self.process.memory_info().rss / 1024 / 1024
                start_time = time.time()
                
                try:
                    builder = BackwardCompatibleKnowledgeGraphBuilder(
                        use_legacy=True, 
                        show_deprecation_warnings=False,
                        enable_knowledge_graph=True
                    )
                    kg = await builder.build_from_text(
                        texts, 
                        graph_name=f"legacy_test_{i}",
                        use_cache=False  # 避免缓存影响性能测试
                    )
                    
                    end_time = time.time()
                    end_memory = self.process.memory_info().rss / 1024 / 1024
                    
                    # 内存峰值
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    iteration_time = end_time - start_time
                    total_time += iteration_time
                    memory_peaks.append(peak / 1024 / 1024)
                    cpu_usage.append(self.process.cpu_percent())
                    
                    print(f"  执行时间: {iteration_time:.2f}s")
                    print(f"  内存使用: {end_memory - start_memory:.1f}MB")
                    print(f"  实体数: {len(kg.entities)}, 关系数: {len(kg.relations)}")
                    
                except Exception as e:
                    tracemalloc.stop()
                    return PerformanceMetrics(
                        execution_time=0,
                        memory_peak_mb=0,
                        memory_current_mb=0,
                        cpu_percent=0,
                        success=False,
                        error=str(e)
                    )
            
            return PerformanceMetrics(
                execution_time=total_time / iterations,
                memory_peak_mb=sum(memory_peaks) / len(memory_peaks),
                memory_current_mb=self.process.memory_info().rss / 1024 / 1024,
                cpu_percent=sum(cpu_usage) / len(cpu_usage),
                success=True
            )
            
        except Exception as e:
            return PerformanceMetrics(
                execution_time=0,
                memory_peak_mb=0,
                memory_current_mb=0,
                cpu_percent=0,
                success=False,
                error=f"Legacy测试失败: {str(e)}"
            )
    
    async def run_pipeline_benchmark(self, texts: List[str], iterations: int = 3) -> PerformanceMetrics:
        """运行Pipeline实现的基准测试"""
        try:
            # 使用主入口点，现在默认指向Pipeline版本
            from agraph import KnowledgeGraphBuilder
            
            total_time = 0
            memory_peaks = []
            cpu_usage = []
            all_step_timings = {}
            cache_hits = 0
            total_operations = 0
            
            for i in range(iterations):
                print(f"Pipeline测试 - 迭代 {i+1}/{iterations}")
                
                # 内存追踪
                tracemalloc.start()
                gc.collect()
                
                start_memory = self.process.memory_info().rss / 1024 / 1024
                start_time = time.time()
                
                try:
                    builder = KnowledgeGraphBuilder(enable_knowledge_graph=True)
                    kg = await builder.build_from_text(
                        texts, 
                        graph_name=f"pipeline_test_{i}",
                        use_cache=False  # 避免缓存影响性能测试
                    )
                    
                    end_time = time.time()
                    end_memory = self.process.memory_info().rss / 1024 / 1024
                    
                    # 内存峰值
                    current, peak = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    
                    iteration_time = end_time - start_time
                    total_time += iteration_time
                    memory_peaks.append(peak / 1024 / 1024)
                    cpu_usage.append(self.process.cpu_percent())
                    
                    # 获取管道指标
                    try:
                        metrics = builder.get_pipeline_metrics()
                        # 分析缓存命中率等指标
                    except:
                        pass
                    
                    print(f"  执行时间: {iteration_time:.2f}s")
                    print(f"  内存使用: {end_memory - start_memory:.1f}MB")
                    print(f"  实体数: {len(kg.entities)}, 关系数: {len(kg.relations)}")
                    
                except Exception as e:
                    tracemalloc.stop()
                    return PerformanceMetrics(
                        execution_time=0,
                        memory_peak_mb=0,
                        memory_current_mb=0,
                        cpu_percent=0,
                        success=False,
                        error=str(e)
                    )
            
            return PerformanceMetrics(
                execution_time=total_time / iterations,
                memory_peak_mb=sum(memory_peaks) / len(memory_peaks),
                memory_current_mb=self.process.memory_info().rss / 1024 / 1024,
                cpu_percent=sum(cpu_usage) / len(cpu_usage),
                success=True,
                cache_hit_rate=cache_hits / max(total_operations, 1)
            )
            
        except Exception as e:
            return PerformanceMetrics(
                execution_time=0,
                memory_peak_mb=0,
                memory_current_mb=0,
                cpu_percent=0,
                success=False,
                error=f"Pipeline测试失败: {str(e)}"
            )
    
    async def run_comprehensive_benchmark(self) -> Dict[str, Any]:
        """运行综合性能基准测试"""
        print("🚀 开始综合性能基准测试...\n")
        
        results = {
            "test_environment": self._get_system_info(),
            "small_dataset": {},
            "large_dataset": {},
            "comparison": {}
        }
        
        # 小数据集测试 (50个文本)
        print("📊 小数据集测试 (50个文本):")
        print("-" * 40)
        
        legacy_small = await self.run_legacy_benchmark(SAMPLE_TEXTS)
        pipeline_small = await self.run_pipeline_benchmark(SAMPLE_TEXTS)
        
        results["small_dataset"] = {
            "legacy": legacy_small,
            "pipeline": pipeline_small
        }
        
        # 大数据集测试 (1000个文本)
        print("\n📊 大数据集测试 (1000个文本):")
        print("-" * 40)
        
        legacy_large = await self.run_legacy_benchmark(LARGE_SAMPLE_TEXTS, iterations=1)
        pipeline_large = await self.run_pipeline_benchmark(LARGE_SAMPLE_TEXTS, iterations=1)
        
        results["large_dataset"] = {
            "legacy": legacy_large,
            "pipeline": pipeline_large
        }
        
        # 性能对比分析
        results["comparison"] = self._analyze_performance(results)
        
        return results
    
    def _get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        return {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / 1024**3,
            "python_version": __import__("sys").version,
            "platform": __import__("platform").platform()
        }
    
    def _analyze_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """分析性能差异"""
        comparison = {}
        
        for dataset in ["small_dataset", "large_dataset"]:
            if dataset not in results:
                continue
                
            legacy = results[dataset].get("legacy")
            pipeline = results[dataset].get("pipeline")
            
            if not legacy or not pipeline or not legacy.success or not pipeline.success:
                comparison[dataset] = {"status": "测试失败或不完整"}
                continue
            
            # 时间对比
            time_improvement = ((legacy.execution_time - pipeline.execution_time) / legacy.execution_time) * 100
            
            # 内存对比
            memory_improvement = ((legacy.memory_peak_mb - pipeline.memory_peak_mb) / legacy.memory_peak_mb) * 100
            
            comparison[dataset] = {
                "time_improvement_percent": time_improvement,
                "memory_improvement_percent": memory_improvement,
                "pipeline_faster": time_improvement > 0,
                "pipeline_memory_efficient": memory_improvement > 0,
                "performance_verdict": self._get_verdict(time_improvement, memory_improvement)
            }
        
        return comparison
    
    def _get_verdict(self, time_improvement: float, memory_improvement: float) -> str:
        """获取性能评价"""
        if time_improvement > 10 and memory_improvement > 5:
            return "显著性能提升"
        elif time_improvement > 5 or memory_improvement > 5:
            return "性能有所提升"
        elif time_improvement > -5 and memory_improvement > -5:
            return "性能相近"
        else:
            return "性能有所下降"
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成性能测试报告"""
        report = ["# AGraph Pipeline性能基准测试报告\n"]
        
        # 系统环境
        env = results["test_environment"]
        report.append("## 测试环境")
        report.append(f"- CPU核心数: {env['cpu_count']}")
        report.append(f"- 内存总量: {env['memory_total_gb']:.1f}GB")
        report.append(f"- Python版本: {env['python_version'].split()[0]}")
        report.append(f"- 操作系统: {env['platform']}")
        report.append("")
        
        # 测试结果
        for dataset_name, dataset in [("small_dataset", "小数据集(50文本)"), ("large_dataset", "大数据集(1000文本)")]:
            if dataset_name not in results:
                continue
                
            report.append(f"## {dataset} 测试结果")
            
            legacy = results[dataset_name].get("legacy")
            pipeline = results[dataset_name].get("pipeline")
            
            if not legacy or not pipeline:
                report.append("❌ 测试数据不完整")
                continue
            
            # Legacy结果
            report.append("### Legacy实现")
            if legacy.success:
                report.append(f"- ✅ 执行成功")
                report.append(f"- ⏱️ 执行时间: {legacy.execution_time:.2f}秒")
                report.append(f"- 💾 内存峰值: {legacy.memory_peak_mb:.1f}MB")
                report.append(f"- 🖥️ CPU使用: {legacy.cpu_percent:.1f}%")
            else:
                report.append(f"- ❌ 执行失败: {legacy.error}")
            
            # Pipeline结果
            report.append("\n### Pipeline实现")
            if pipeline.success:
                report.append(f"- ✅ 执行成功")
                report.append(f"- ⏱️ 执行时间: {pipeline.execution_time:.2f}秒")
                report.append(f"- 💾 内存峰值: {pipeline.memory_peak_mb:.1f}MB")
                report.append(f"- 🖥️ CPU使用: {pipeline.cpu_percent:.1f}%")
                if pipeline.cache_hit_rate > 0:
                    report.append(f"- 🎯 缓存命中率: {pipeline.cache_hit_rate:.1%}")
            else:
                report.append(f"- ❌ 执行失败: {pipeline.error}")
            
            # 对比分析
            comparison = results["comparison"].get(dataset_name)
            if comparison and "time_improvement_percent" in comparison:
                report.append("\n### 性能对比")
                time_imp = comparison["time_improvement_percent"]
                memory_imp = comparison["memory_improvement_percent"]
                
                if time_imp > 0:
                    report.append(f"- ⚡ 执行时间提升: {time_imp:.1f}%")
                else:
                    report.append(f"- ⚡ 执行时间变化: {time_imp:.1f}% (稍慢)")
                
                if memory_imp > 0:
                    report.append(f"- 💾 内存使用优化: {memory_imp:.1f}%")
                else:
                    report.append(f"- 💾 内存使用变化: {memory_imp:.1f}% (稍高)")
                
                report.append(f"- 📊 **评价**: {comparison['performance_verdict']}")
            
            report.append("")
        
        return "\n".join(report)


async def main():
    """主测试函数"""
    print("🔬 AGraph Pipeline性能基准测试")
    print("=" * 50)
    
    benchmark = PerformanceBenchmark()
    results = await benchmark.run_comprehensive_benchmark()
    
    # 生成报告
    report = benchmark.generate_report(results)
    print("\n" + "=" * 50)
    print("📋 测试报告:")
    print("=" * 50)
    print(report)
    
    # 保存报告到文件
    report_file = Path("pipeline_performance_report.md")
    report_file.write_text(report, encoding="utf-8")
    print(f"\n📄 报告已保存到: {report_file.absolute()}")


if __name__ == "__main__":
    asyncio.run(main())