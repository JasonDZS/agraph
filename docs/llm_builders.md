# LLM Builders 使用教程

本教程将详细介绍如何使用 agraph 项目中的各种 LLM 构建器来构建和管理知识图谱。这些构建器遵循接口隔离原则（ISP），为不同的使用场景提供了专门的接口。

## 概述

agraph 提供了多种 LLM 构建器，每种都专门针对特定的用例设计：

- **MinimalLLMGraphBuilder**: 基础构建器，只提供最基本的图谱构建功能
- **FlexibleLLMGraphBuilder**: 灵活构建器，支持构建和更新功能
- **StreamingLLMGraphBuilder**: 流式构建器，支持实时文档流处理
- **BatchLLMGraphBuilder**: 批量构建器，优化大量文档和多数据源处理
- **LLMGraphBuilder**: 全功能构建器，包含所有功能（谨慎使用）

## 准备工作

### 环境配置

首先确保安装了必要的依赖：

```bash
# 安装开发依赖
make install-dev

# 或者直接使用 pip
pip install -e .
```

设置环境变量：

```bash
export OPENAI_API_KEY="your-openai-api-key"
export OPENAI_API_BASE="https://api.openai.com/v1"  # 可选
```

### 导入依赖

```python
import asyncio
from agraph.builders.llm_builders import (
    MinimalLLMGraphBuilder,
    FlexibleLLMGraphBuilder,
    StreamingLLMGraphBuilder,
    BatchLLMGraphBuilder,
    LLMGraphBuilder,
)
from agraph.embeddings import JsonVectorStorage
from agraph.entities import Entity
from agraph.types import EntityType
from agraph.config import Settings
```

## 1. 基础构建器（MinimalLLMGraphBuilder）

### 适用场景
- 只需要简单的文本到图谱转换
- 不需要更新、合并、验证等功能
- 追求最小依赖和简单性
- 轻量级应用

### 使用示例

```python
async def basic_builder_example():
    # 创建基础构建器
    builder = MinimalLLMGraphBuilder(
        openai_api_key=Settings.OPENAI_API_KEY,
        openai_api_base=Settings.OPENAI_API_BASE,
        llm_model=Settings.LLM_MODEL,
        temperature=0.1,
    )

    # 准备文档
    texts = [
        "苹果公司是一家美国跨国技术公司，总部位于加利福尼亚州库比蒂诺。",
        "史蒂夫·乔布斯是苹果公司的联合创始人，他在2011年去世。",
        "iPhone是苹果公司开发的智能手机产品线。"
    ]

    try:
        # 构建图谱 - 异步操作
        graph = await builder.build_graph(
            texts=texts,
            graph_name="basic_example_graph"
        )

        print(f"✅ 成功构建基础图谱:")
        print(f"   - 实体数量: {len(graph.entities)}")
        print(f"   - 关系数量: {len(graph.relations)}")
        print(f"   - 图谱名称: {graph.name}")

        # 显示实体示例
        print(f"\n📋 实体示例:")
        for i, (entity_id, entity) in enumerate(list(graph.entities.items())[:3]):
            print(f"   {i+1}. {entity.name} ({entity.entity_type.value})")

        return graph

    except Exception as e:
        print(f"❌ 基础构建器示例失败: {e}")
        return None

# 运行示例
asyncio.run(basic_builder_example())
```

## 2. 灵活构建器（FlexibleLLMGraphBuilder）

### 适用场景
- 需要构建图谱后进行增量更新
- 不需要合并、验证等高级功能
- 追求构建+更新的组合功能
- 中等复杂度应用

### 使用示例

```python
async def flexible_builder_example():
    # 创建灵活构建器
    builder = FlexibleLLMGraphBuilder(
        openai_api_key=Settings.OPENAI_API_KEY,
        openai_api_base=Settings.OPENAI_API_BASE,
        llm_model=Settings.LLM_MODEL,
        embedding_model=Settings.EMBEDDING_MODEL,
        vector_storage=JsonVectorStorage(file_path="workdir/vector_store.json"),
    )

    # 初始文本
    initial_texts = [
        "微软公司是一家美国跨国技术公司。",
        "比尔·盖茨是微软公司的联合创始人。"
    ]

    try:
        # 构建初始图谱
        graph = await builder.build_graph(
            texts=initial_texts,
            graph_name="updatable_example_graph"
        )

        print(f"✅ 初始图谱构建完成:")
        print(f"   - 初始实体数: {len(graph.entities)}")
        print(f"   - 初始关系数: {len(graph.relations)}")

        # 准备新实体进行更新
        new_entity = Entity(
            id="entity_new_001",
            name="Windows操作系统",
            entity_type=EntityType.PRODUCT,
            description="微软公司开发的操作系统",
        )

        # 更新图谱
        updated_graph = await builder.update_graph(
            graph=graph,
            new_entities=[new_entity],
        )

        print(f"\n🔄 图谱更新完成:")
        print(f"   - 更新后实体数: {len(updated_graph.entities)}")
        print(f"   - 更新后关系数: {len(updated_graph.relations)}")

        return updated_graph

    except Exception as e:
        print(f"❌ 可更新构建器示例失败: {e}")
        return None

# 运行示例
asyncio.run(flexible_builder_example())
```

## 3. 流式构建器（StreamingLLMGraphBuilder）

### 适用场景
- 实时处理流式到达的文档
- 需要文档级别的增删操作
- 追求增量处理能力
- 实时数据处理应用

### 使用示例

```python
async def streaming_builder_example():
    # 创建流式构建器
    builder = StreamingLLMGraphBuilder(
        openai_api_key=Settings.OPENAI_API_KEY,
        openai_api_base=Settings.OPENAI_API_BASE,
        llm_model=Settings.LLM_MODEL,
    )

    # 初始文档
    initial_docs = [
        "谷歌公司是一家美国跨国技术公司。",
        "拉里·佩奇和谢尔盖·布林创立了谷歌。"
    ]

    try:
        # 构建初始图谱
        graph = await builder.build_graph(
            texts=initial_docs,
            graph_name="streaming_example_graph"
        )

        print(f"✅ 流式图谱初始化:")
        print(f"   - 初始实体数: {len(graph.entities)}")

        # 模拟新文档到达
        new_documents = [
            "YouTube是谷歌旗下的视频分享平台。",
            "Android是谷歌开发的移动操作系统。"
        ]

        # 增量添加文档
        updated_graph = await builder.add_documents_async(
            documents=new_documents,
            document_ids=["doc_youtube", "doc_android"]
        )

        print(f"\n📄 新文档处理完成:")
        print(f"   - 更新后实体数: {len(updated_graph.entities)}")

        # 查看文档注册表
        registry = builder.get_document_registry()
        print(f"\n📚 文档注册表:")
        for doc_id, entity_ids in registry.items():
            print(f"   - {doc_id}: {len(entity_ids)} 个实体")

        return updated_graph

    except Exception as e:
        print(f"❌ 流式构建器示例失败: {e}")
        return None

# 运行示例
asyncio.run(streaming_builder_example())
```

## 4. 批量构建器（BatchLLMGraphBuilder）

### 适用场景
- 需要处理大量文档
- 需要合并多个数据源
- 追求高性能批量处理
- 大规模数据处理

### 使用示例

```python
async def batch_builder_example():
    # 创建批量构建器
    builder = BatchLLMGraphBuilder(
        openai_api_key=Settings.OPENAI_API_KEY,
        openai_api_base=Settings.OPENAI_API_BASE,
        llm_model=Settings.LLM_MODEL,
        embedding_model=Settings.EMBEDDING_MODEL,
        max_concurrent=8,  # 高并发批量处理
    )

    # 大量文档示例
    batch_texts = [
        "特斯拉是一家美国电动汽车制造商。",
        "埃隆·马斯克是特斯拉的CEO。",
        "Model S是特斯拉的豪华电动轿车。",
        "Autopilot是特斯拉的自动驾驶技术。",
        "Gigafactory是特斯拉的电池工厂。",
    ]

    try:
        # 批量构建图谱
        graph = await builder.build_graph(
            texts=batch_texts,
            graph_name="batch_example_graph"
        )

        print(f"✅ 批量图谱构建完成:")
        print(f"   - 处理文档数: {len(batch_texts)}")
        print(f"   - 生成实体数: {len(graph.entities)}")
        print(f"   - 生成关系数: {len(graph.relations)}")

        # 演示多源合并
        sources = [
            {
                "type": "text",
                "data": ["亚马逊是全球最大的电子商务公司。"]
            },
            {
                "type": "text",
                "data": ["杰夫·贝佐斯创立了亚马逊公司。"]
            }
        ]

        merged_graph = await builder.build_from_multiple_sources(
            sources=sources,
            graph_name="multi_source_graph"
        )

        print(f"\n🔗 多源合并完成:")
        print(f"   - 合并后实体数: {len(merged_graph.entities)}")

        return merged_graph

    except Exception as e:
        print(f"❌ 批量构建器示例失败: {e}")
        return None

# 运行示例
asyncio.run(batch_builder_example())
```

## 5. 全功能构建器（LLMGraphBuilder）

### 适用场景
⚠️ **注意：这违反了ISP原则，只有真正需要所有功能时才使用！**

- 需要所有功能：构建、更新、合并、验证、导出
- 企业级应用，需要完整功能
- 可以承受额外的复杂性和依赖

### 使用示例

```python
async def full_featured_builder_example():
    # 创建全功能构建器
    builder = LLMGraphBuilder(
        openai_api_key=Settings.OPENAI_API_KEY,
        openai_api_base=Settings.OPENAI_API_BASE,
        llm_model=Settings.LLM_MODEL,
        embedding_model=Settings.EMBEDDING_MODEL,
        max_concurrent=10,
        vector_storage=JsonVectorStorage(file_path="workdir/full_vector_store.json"),
    )

    # 复杂文档
    complex_texts = [
        "阿里巴巴集团是中国的一家跨国技术公司。",
        "马云是阿里巴巴集团的创始人之一。",
        "淘宝是阿里巴巴旗下的在线购物平台。",
        "支付宝是阿里巴巴的数字支付平台。"
    ]

    try:
        # 构建图谱（自动包含嵌入和验证）
        graph = await builder.build_graph(
            texts=complex_texts,
            graph_name="full_featured_graph"
        )

        print(f"✅ 全功能图谱构建完成:")
        print(f"   - 实体数量: {len(graph.entities)}")
        print(f"   - 关系数量: {len(graph.relations)}")

        # 验证图谱
        validation_result = await builder.validate_graph(graph)
        print(f"\n🔍 图谱验证结果:")
        print(f"   - 验证通过: {validation_result.get('valid', False)}")
        if validation_result.get('issues'):
            print(f"   - 发现问题: {len(validation_result['issues'])} 个")

        # 导出图谱
        exported_data = await builder.export_to_format(graph, "json")
        print(f"\n📤 图谱导出:")
        print(f"   - 导出格式: JSON")
        print(f"   - 数据键: {list(exported_data.keys())}")

        # 保存导出数据
        import json
        with open("workdir/full_featured_graph.json", "w", encoding="utf8") as f:
            json.dump(exported_data, f, ensure_ascii=False, indent=2)

        # 获取详细统计
        detailed_stats = await builder.get_detailed_statistics(graph)
        print(f"\n📊 详细统计:")
        for key, value in detailed_stats.items():
            if isinstance(value, (int, float)):
                print(f"   - {key}: {value}")

        # 打印使用摘要
        builder.print_usage_summary()

        # 清理资源
        builder.cleanup()

        return graph

    except Exception as e:
        print(f"❌ 全功能构建器示例失败: {e}")
        return None

# 运行示例
asyncio.run(full_featured_builder_example())
```

## 完整示例运行

```python
import asyncio
from pathlib import Path

async def main():
    """运行所有示例"""
    print("🎯 LLM ISP构建器示例集合")
    print("演示Interface Segregation Principle在LLM图构建中的应用")
    print("=" * 60)

    # 确保工作目录存在
    Path("./workdir").mkdir(exist_ok=True)

    try:
        # 依次运行各个示例
        await basic_builder_example()
        await flexible_builder_example()
        await streaming_builder_example()
        await batch_builder_example()
        await full_featured_builder_example()

    except Exception as e:
        print(f"示例执行过程中出现错误: {e}")

    print(f"\n🎉 所有示例运行完成!")
    print(f"📝 通过这些示例，你可以看到:")
    print(f"   1. 每个构建器只实现必要的接口")
    print(f"   2. 客户端可以选择最适合的构建器")
    print(f"   3. 避免了不必要的依赖和复杂性")
    print(f"   4. 遵循了Interface Segregation Principle")

if __name__ == "__main__":
    asyncio.run(main())
```

## 配置说明

### Settings 配置

项目使用 `Settings` 类来管理配置：

```python
from agraph.config import Settings

# 常用设置
Settings.OPENAI_API_KEY      # OpenAI API 密钥
Settings.OPENAI_API_BASE     # OpenAI API 基础URL
Settings.LLM_MODEL           # 使用的LLM模型
Settings.EMBEDDING_MODEL     # 使用的嵌入模型
```

### 向量存储配置

```python
from agraph.embeddings import JsonVectorStorage

# JSON文件存储
vector_storage = JsonVectorStorage(file_path="workdir/vector_store.json")

# 传递给构建器
builder = FlexibleLLMGraphBuilder(
    # ... 其他参数
    vector_storage=vector_storage
)
```

## 最佳实践

### 1. 选择合适的构建器

```python
# ✅ 好的做法：根据需求选择最小接口
if only_need_basic_build:
    builder = MinimalLLMGraphBuilder(...)
elif need_updates:
    builder = FlexibleLLMGraphBuilder(...)
elif need_streaming:
    builder = StreamingLLMGraphBuilder(...)

# ❌ 避免：总是使用全功能构建器
# builder = LLMGraphBuilder(...)  # 除非真的需要所有功能
```

### 2. 资源管理

```python
try:
    builder = LLMGraphBuilder(...)
    graph = await builder.build_graph(...)
    # 使用图谱...
finally:
    # 清理资源
    builder.cleanup()
```

### 3. 错误处理

```python
try:
    graph = await builder.build_graph(texts=texts)
except Exception as e:
    logger.error(f"构建图谱失败: {e}")
    # 处理错误...
```

### 4. 异步最佳实践

```python
# 并发处理多个任务
tasks = [
    builder1.build_graph(texts1),
    builder2.build_graph(texts2),
]
results = await asyncio.gather(*tasks, return_exceptions=True)
```

## 性能优化

### 1. 批量处理

```python
# ✅ 使用批量构建器处理大量文档
builder = BatchLLMGraphBuilder(max_concurrent=8)
graph = await builder.build_graph(texts=large_text_list)
```

### 2. 流式处理

```python
# ✅ 使用流式构建器处理实时数据
builder = StreamingLLMGraphBuilder()
for batch in document_stream:
    await builder.add_documents_async(batch)
```

### 3. 向量存储优化

```python
# 使用高效的向量存储
vector_storage = JsonVectorStorage(
    file_path="workdir/optimized_vectors.json"
)
```

## 故障排除

### 常见问题

1. **API 密钥错误**
   ```bash
   export OPENAI_API_KEY="your-actual-api-key"
   ```

2. **依赖缺失**
   ```bash
   make install-dev
   ```

3. **权限问题**
   ```bash
   mkdir -p workdir
   chmod 755 workdir
   ```

4. **内存不足**
   ```python
   # 减少并发数
   builder = BatchLLMGraphBuilder(max_concurrent=2)
   ```

### 调试技巧

```python
import logging
logging.basicConfig(level=logging.INFO)

# 启用详细日志
builder = MinimalLLMGraphBuilder(
    # ... 其他参数
    verbose=True
)
```

## 总结

通过本教程，你学会了：

1. **接口隔离原则**：根据需求选择最小的必要接口
2. **性能优化**：使用合适的构建器处理不同规模的数据
3. **资源管理**：正确处理异步操作和资源清理
4. **错误处理**：实现健壮的错误处理机制

选择合适的 LLM 构建器可以让你的知识图谱应用更加高效、可维护和可扩展。
