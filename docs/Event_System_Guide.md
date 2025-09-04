# AGraph 事件系统使用指南

## 概述

AGraph v0.2.0+ 引入了企业级的事件驱动架构，提供了完整的事件发布、订阅和处理机制。事件系统与现有的索引管理、缓存系统和事务系统深度集成，确保数据一致性和系统可观测性。

## 核心特性

### 🎯 事件驱动架构
- **自动事件发布**: CRUD 操作自动触发相应事件
- **异步事件处理**: 不影响主要业务逻辑性能
- **优先级队列**: 支持事件优先级和批量处理
- **线程安全**: 完全的并发安全保证

### 🎧 智能事件监听
- **索引自动更新**: 实体/关系变更时自动维护索引
- **缓存智能失效**: 数据修改时精确失效相关缓存
- **完整性检查**: 自动检测数据不一致和悬空引用
- **可扩展监听器**: 支持自定义事件处理逻辑

### 📊 监控和分析
- **实时统计**: 事件发布、处理统计和性能指标
- **事件持久化**: 可选的审计日志和事件回放
- **性能监控**: 平均处理时间、吞吐量等指标
- **故障诊断**: 详细的错误信息和调试数据

## 快速开始

### 基本设置

```python
from agraph.base.infrastructure.cache import CacheManager
from agraph.base.infrastructure.dao import MemoryDataAccessLayer
from agraph.base.events.events import EventManager
from agraph.base.events.listeners import setup_default_listeners
from agraph.base.infrastructure.indexes import IndexManager

# 创建核心组件
event_manager = EventManager(enable_async=True)
index_manager = IndexManager()
cache_manager = CacheManager()
dao = MemoryDataAccessLayer(event_manager)

# 设置默认事件监听器
subscriptions = setup_default_listeners(
    event_manager, index_manager, cache_manager, dao
)
print(f"已注册 {len(subscriptions)} 个监听器")
```

### 基本事件操作

```python
from agraph.base.models.entities import Entity
from agraph.base.core.types import EntityType

# 创建实体（自动触发 ENTITY_ADDED 事件）
entity = Entity(
    id="example_entity",
    name="示例实体",
    entity_type=EntityType.PERSON.value,
    description="这是一个示例实体"
)

dao.save_entity(entity)  # 自动发布事件，触发索引更新

# 更新实体（自动触发 ENTITY_UPDATED 事件）
entity.description = "更新后的描述"
dao.save_entity(entity)  # 触发缓存失效

# 删除实体（自动触发 ENTITY_REMOVED 事件）
dao.delete_entity("example_entity")  # 触发完整性检查
```

## 事件类型

### 实体事件
- `ENTITY_ADDED`: 实体被添加
- `ENTITY_UPDATED`: 实体被更新
- `ENTITY_REMOVED`: 实体被删除
- `ENTITY_LOADED`: 实体被加载（缓存预热）

### 关系事件
- `RELATION_ADDED`: 关系被添加
- `RELATION_UPDATED`: 关系被更新
- `RELATION_REMOVED`: 关系被删除
- `RELATION_LOADED`: 关系被加载

### 系统事件
- `GRAPH_CLEARED`: 图谱被清空
- `CACHE_CLEARED`: 缓存被清空
- `INDEX_REBUILT`: 索引被重建
- `OPERATION_FAILED`: 操作失败

### 事务事件
- `TRANSACTION_STARTED`: 事务开始
- `TRANSACTION_COMMITTED`: 事务提交
- `TRANSACTION_ROLLED_BACK`: 事务回滚
- `BATCH_STARTED`: 批处理开始
- `BATCH_COMMITTED`: 批处理提交

## 自定义事件监听器

### 创建自定义监听器

```python
from agraph.base.events.events import EventListener, EventType, GraphEvent
from agraph.base.core.result import Result
from typing import Set

class CustomEventListener(EventListener):
    """自定义事件监听器示例"""

    def get_event_types(self) -> Set[EventType]:
        """定义要监听的事件类型"""
        return {
            EventType.ENTITY_ADDED,
            EventType.RELATION_ADDED
        }

    def handle_event(self, event: GraphEvent) -> Result[bool]:
        """处理事件"""
        print(f"处理事件: {event.event_type} for {event.target_id}")

        if event.event_type == EventType.ENTITY_ADDED:
            # 自定义实体添加处理逻辑
            return self._handle_entity_added(event)
        elif event.event_type == EventType.RELATION_ADDED:
            # 自定义关系添加处理逻辑
            return self._handle_relation_added(event)

        return Result.ok(True)

    def _handle_entity_added(self, event: GraphEvent) -> Result[bool]:
        """处理实体添加事件"""
        if event.data:
            entity_name = event.data.get("name", "Unknown")
            print(f"新实体被添加: {entity_name}")
        return Result.ok(True)

    def _handle_relation_added(self, event: GraphEvent) -> Result[bool]:
        """处理关系添加事件"""
        if event.data:
            relation_type = event.data.get("relation_type", "Unknown")
            print(f"新关系被添加: {relation_type}")
        return Result.ok(True)

# 注册自定义监听器
custom_listener = CustomEventListener()
subscription_result = event_manager.subscribe(custom_listener)
if subscription_result.is_ok():
    print(f"自定义监听器已注册: {subscription_result.data}")
```

### 异步事件监听器

```python
from agraph.base.events.events import AsyncEventListener

class AsyncAnalyticsListener(AsyncEventListener):
    """异步分析监听器"""

    def get_event_types(self) -> Set[EventType]:
        return {EventType.ENTITY_ADDED, EventType.RELATION_ADDED}

    async def handle_event_async(self, event: GraphEvent) -> Result[bool]:
        """异步处理事件"""
        # 模拟异步 I/O 操作（如发送到外部分析系统）
        await asyncio.sleep(0.001)

        # 处理分析逻辑
        if event.event_type == EventType.ENTITY_ADDED:
            await self._analyze_entity(event)

        return Result.ok(True)

    async def _analyze_entity(self, event: GraphEvent):
        """分析实体数据"""
        # 异步分析逻辑
        print(f"正在分析实体: {event.target_id}")
```

## 事件持久化

### 设置事件持久化

```python
from agraph.base.events.persistence import setup_event_persistence
from agraph.base.events.events import EventPriority

# 设置事件持久化到文件
persistence_result = setup_event_persistence(
    event_manager,
    storage_dir="./event_logs",
    min_priority=EventPriority.LOW  # 记录所有事件
)

if persistence_result.is_ok():
    print("事件持久化已启用")
```

### 事件分析

```python
from agraph.base.events.persistence import analyze_persisted_events

# 分析最近的事件
analysis_result = analyze_persisted_events(
    backend,
    start_time=time.time() - 3600  # 最近1小时
)

if analysis_result.is_ok():
    analysis = analysis_result.data
    print(f"分析了 {analysis['total_events']} 个事件")
    print(f"最常见事件: {analysis['most_common_event_type']}")
    print(f"事件类型分布: {analysis['event_type_distribution']}")
```

## 与事务系统集成

### 事务性事件

```python
from agraph.base.transactions.transaction import TransactionManager

transaction_manager = TransactionManager(dao)

# 事务中的操作会延迟发布事件直到提交
with transaction_manager.transaction() as tx:
    # 这些操作的事件会在事务提交时一起发布
    tx.add_entity(entity1)
    tx.add_entity(entity2)
    # 事务提交时，所有事件一次性发布
```

### 批量操作事件

```python
from agraph.base.transactions.batch import atomic_batch_operations

# 批量操作产生批量事件
with atomic_batch_operations(dao, transaction_manager) as batch_ctx:
    for entity_data in batch_entities:
        operation = create_entity_batch_operation(
            BatchOperationType.ADD, entity_data
        )
        batch_ctx.add_operation_with_transaction(operation)
    # 批量提交时触发所有相关事件
```

## 性能和最佳实践

### 性能考虑

1. **异步处理**: 默认使用异步事件处理，不阻塞主线程
2. **批量优化**: 事件处理支持批量操作，减少开销
3. **选择性监听**: 只监听需要的事件类型，减少处理负担
4. **缓存友好**: 事件监听器与缓存系统协同，提升性能

### 最佳实践

1. **监听器设计**:
   ```python
   # ✅ 好的做法：轻量级处理
   def handle_event(self, event):
       if event.event_type == EventType.ENTITY_ADDED:
           self.update_counter()
       return Result.ok(True)

   # ❌ 避免：重量级处理
   def handle_event(self, event):
       # 避免在监听器中进行耗时操作
       time.sleep(1.0)  # 不要这样做
   ```

2. **错误处理**:
   ```python
   def handle_event(self, event):
       try:
           # 处理逻辑
           return Result.ok(True)
       except Exception as e:
           # 记录错误但不阻止其他监听器
           self.logger.error(f"处理事件失败: {e}")
           return Result.internal_error(e)
   ```

3. **资源管理**:
   ```python
   # 使用上下文管理器确保清理
   try:
       # 事件处理逻辑
       pass
   finally:
       # 清理资源
       event_manager.shutdown()
   ```

## 监控和调试

### 事件统计

```python
# 获取详细统计信息
stats = event_manager.get_statistics()
print(f"已发布事件: {stats['events_published']}")
print(f"已处理事件: {stats['events_processed']}")
print(f"处理失败: {stats['events_failed']}")
print(f"平均处理时间: {stats['average_processing_time']}")
```

### 调试技巧

1. **同步处理调试**:
   ```python
   # 强制同步处理便于调试
   event_manager.publish(event, synchronous=True)
   ```

2. **刷新事件队列**:
   ```python
   # 确保所有事件都被处理
   flush_result = event_manager.flush_events(timeout=5.0)
   ```

3. **监听器状态检查**:
   ```python
   # 检查活跃的监听器
   for sub_id, subscription in event_manager._subscription_by_id.items():
       print(f"监听器: {type(subscription.listener).__name__}")
       print(f"事件类型: {[et.value for et in subscription.event_types]}")
   ```

## 集成指南

### 与现有系统集成

1. **渐进式采用**:
   ```python
   # 步骤1: 只启用事件管理器
   event_manager = EventManager()
   dao = MemoryDataAccessLayer(event_manager)

   # 步骤2: 添加基本监听器
   index_listener = IndexUpdateListener(index_manager)
   event_manager.subscribe(index_listener)

   # 步骤3: 逐步添加更多监听器
   cache_listener = CacheInvalidationListener(cache_manager)
   event_manager.subscribe(cache_listener)
   ```

2. **配置管理**:
   ```python
   # 配置事件管理器
   event_manager = EventManager(
       max_queue_size=10000,  # 事件队列大小
       enable_async=True      # 启用异步处理
   )
   ```

### 企业级部署

1. **事件持久化**:
   ```python
   # 生产环境建议启用事件持久化
   setup_event_persistence(
       event_manager,
       storage_dir="/var/log/agraph/events",
       min_priority=EventPriority.NORMAL  # 只记录重要事件
   )
   ```

2. **监控集成**:
   ```python
   # 定期收集统计信息
   async def collect_metrics():
       while True:
           stats = event_manager.get_statistics()
           # 发送到监控系统（如 Prometheus）
           await send_to_monitoring(stats)
           await asyncio.sleep(60)  # 每分钟收集一次
   ```

## 故障排除

### 常见问题

1. **事件处理慢**:
   - 检查监听器是否有阻塞操作
   - 考虑使用异步监听器
   - 检查事件队列大小

2. **监听器错误**:
   - 查看事件管理器统计中的 `events_failed`
   - 在监听器中添加详细的错误处理
   - 使用同步模式调试

3. **内存使用高**:
   - 检查事件队列积压
   - 调整 `max_queue_size` 参数
   - 确保监听器处理足够快

### 调试工具

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 检查事件队列状态
stats = event_manager.get_statistics()
if stats['queue_size'] > 1000:
    print("警告：事件队列积压")

# 强制处理所有事件
flush_result = event_manager.flush_events(timeout=10.0)
if not flush_result.is_ok():
    print(f"事件处理超时: {flush_result.error_message}")
```

## 示例代码

### 完整示例

参考 `examples/` 目录中的示例：

- **`events_quickstart.py`**: 事件系统基础入门
- **`transactions_quickstart.py`**: 事务系统基础入门
- **`event_transaction_demo.py`**: 完整功能演示

### 运行示例

```bash
# 事件系统快速入门
python examples/events_quickstart.py

# 事务系统快速入门
python examples/transactions_quickstart.py

# 完整功能演示
python examples/event_transaction_demo.py
```

## API 参考

### EventManager

| 方法 | 描述 | 参数 |
|------|------|------|
| `subscribe(listener, event_types, priority)` | 订阅事件 | 监听器、事件类型、优先级 |
| `unsubscribe(subscription_id)` | 取消订阅 | 订阅ID |
| `publish(event, synchronous)` | 发布事件 | 事件对象、是否同步 |
| `flush_events(timeout)` | 刷新事件队列 | 超时时间 |
| `get_statistics()` | 获取统计信息 | - |
| `shutdown(timeout)` | 关闭事件管理器 | 超时时间 |

### EventListener

| 方法 | 描述 | 返回值 |
|------|------|--------|
| `get_event_types()` | 获取监听的事件类型 | Set[EventType] |
| `handle_event(event)` | 处理事件 | Result[bool] |
| `should_handle_event(event)` | 判断是否处理事件 | bool |
| `get_priority()` | 获取监听器优先级 | EventPriority |

### GraphEvent

| 属性 | 类型 | 描述 |
|------|------|------|
| `event_type` | EventType | 事件类型 |
| `event_id` | str | 事件唯一ID |
| `timestamp` | float | 事件时间戳 |
| `source` | str | 事件源组件 |
| `target_type` | str | 目标对象类型 |
| `target_id` | str | 目标对象ID |
| `data` | Dict | 事件数据 |
| `metadata` | Dict | 元数据 |
| `priority` | EventPriority | 事件优先级 |
| `transaction_id` | str | 关联事务ID |

## 版本历史

### v0.2.0 (当前版本)
- ✅ 完整的事件系统实现
- ✅ 与索引、缓存、事务系统集成
- ✅ 事件持久化和分析功能
- ✅ 企业级监控和统计

### 未来版本
- 🔮 更多事件类型和监听器
- 🔮 分布式事件支持
- 🔮 事件重放和时间旅行调试
- 🔮 GraphQL 事件订阅

## 支持

如有问题或建议，请：
1. 查看 `examples/` 目录中的示例代码
2. 检查 `test_event_transaction_system.py` 中的测试用例
3. 参考其他相关文档：
   - `Transaction_System_Guide.md` - 事务系统详细指南
   - `Performance_Optimization_Guide.md` - 性能优化指南
   - `Migration_Guide.md` - 迁移指南

---

*最后更新: 2025-08-30*
*AGraph 版本: v0.2.0+*
