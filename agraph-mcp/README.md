# AGraph MCP Server

一个基于 AGraph 知识图谱库的 MCP (Model Context Protocol) 服务器，为 AI 系统提供本地知识图谱检索功能。该服务器直接实例化 AGraph 对象进行本地操作，而非通过远程 API 调用。

## 功能特性

- 🔍 **知识图谱状态查询** - 获取知识图谱基本统计信息
- 🏷️ **实体检索** - 按类型、名称等条件搜索实体
- 🔗 **关系检索** - 查询实体间的关系连接
- 📄 **文本块检索** - 搜索相关的文本内容（支持向量搜索）
- 🧩 **聚类检索** - 获取实体聚类信息
- 🌐 **完整图谱获取** - 获取完整的知识图谱数据
- 🗣️ **自然语言搜索** - 支持自然语言查询所有类型数据（优先使用向量搜索）

## 安装

1. 确保已安装 AGraph 库：
```bash
cd ../  # 回到 agraph 目录
pip install -e .
```

2. 安装 MCP 服务器依赖：
```bash
cd agraph-mcp
uv sync
```

或使用 pip：
```bash
pip install httpx mcp
```

## 配置

### 环境变量

创建 `.env` 文件或设置以下环境变量：

```bash
# AGraph 配置
AGRAPH_WORKDIR=agraph_workspace              # AGraph 工作目录
AGRAPH_DEFAULT_COLLECTION=agraph_knowledge   # 默认集合名称
AGRAPH_PERSIST_DIRECTORY=agraph_workspace    # 数据持久化目录
AGRAPH_VECTOR_STORE_TYPE=chroma              # 向量存储类型

# 默认限制配置
AGRAPH_DEFAULT_ENTITY_LIMIT=50               # 默认实体查询限制
AGRAPH_DEFAULT_RELATION_LIMIT=30             # 默认关系查询限制
AGRAPH_DEFAULT_TEXT_CHUNK_LIMIT=20           # 默认文本块查询限制
AGRAPH_DEFAULT_CLUSTER_LIMIT=15              # 默认聚类查询限制
AGRAPH_MAX_CONTENT_PREVIEW=200               # 文本内容预览最大长度

# 自然语言搜索配置
AGRAPH_NATURAL_SEARCH_LIMIT=10               # 自然语言搜索默认限制
```

## 使用方法

### 启动服务器

```bash
python server.py
```

或使用启动脚本：
```bash
python start_server.py
```

### MCP 工具

服务器提供以下 MCP 工具：

#### 1. get_knowledge_graph_status
获取知识图谱状态和统计信息。

**参数：**
- `collection_name` (可选): 集合名称

#### 2. search_entities
搜索实体。

**参数：**
- `collection_name` (可选): 集合名称
- `entity_type` (可选): 实体类型过滤 (PERSON, ORGANIZATION, LOCATION, CONCEPT, EVENT, OTHER)
- `limit` (可选): 返回数量限制
- `offset` (可选): 分页偏移

#### 3. search_relations
搜索关系。

**参数：**
- `collection_name` (可选): 集合名称
- `relation_type` (可选): 关系类型过滤
- `entity_id` (可选): 按实体ID过滤
- `limit` (可选): 返回数量限制
- `offset` (可选): 分页偏移

#### 4. search_text_chunks
搜索文本块。

**参数：**
- `collection_name` (可选): 集合名称
- `search_query` (可选): 搜索查询（优先使用向量搜索）
- `entity_id` (可选): 按实体ID过滤
- `limit` (可选): 返回数量限制
- `offset` (可选): 分页偏移

#### 5. search_clusters
搜索聚类。

**参数：**
- `collection_name` (可选): 集合名称
- `cluster_type` (可选): 聚类类型过滤
- `limit` (可选): 返回数量限制
- `offset` (可选): 分页偏移

#### 6. get_full_knowledge_graph
获取完整知识图谱。

**参数：**
- `collection_name` (可选): 集合名称
- `include_text_chunks` (可选): 是否包含文本块
- `include_clusters` (可选): 是否包含聚类
- `entity_limit` (可选): 实体数量限制
- `relation_limit` (可选): 关系数量限制

#### 7. natural_language_search
自然语言搜索。

**参数：**
- `query` (必需): 自然语言查询
- `collection_name` (可选): 集合名称
- `search_type` (可选): 搜索类型 (all, entities, relations, text_chunks, clusters)
- `limit` (可选): 每种类型的结果限制

## 使用示例

### 在 Claude 中使用

1. 配置 Claude Desktop 或其他 MCP 客户端连接到此服务器
2. 使用自然语言查询：

```
找到所有与"人工智能"相关的实体和关系
```

```
搜索包含"机器学习"关键词的文本块
```

```
获取集合"my_project"的知识图谱状态
```

### Claude Desktop 集成

编辑 Claude Desktop 配置文件，添加：

```json
{
  "mcpServers": {
    "agraph": {
      "command": "python",
      "args": ["/path/to/agraph-mcp/start_server.py"],
      "env": {
        "AGRAPH_WORKDIR": "agraph_workspace",
        "AGRAPH_DEFAULT_COLLECTION": "agraph_knowledge"
      }
    }
  }
}
```

## 架构说明

该 MCP 服务器直接使用 AGraph 实例进行本地操作，架构包括：

```
server.py          # 主服务器实现 (直接使用 AGraph 实例)
config.py          # 配置管理
exceptions.py      # 自定义异常
start_server.py    # 启动脚本
test_server.py     # 测试代码
```

### 核心组件

- **AGraphMCPServer**: 主服务器类，负责初始化和管理 AGraph 实例
- **AGraphConfig**: 配置管理类
- **AGraph Instance**: 每个集合对应一个 AGraph 实例
- **Vector Search**: 优先使用向量搜索提高检索精度

## 关键特性

### 本地 AGraph 实例

- 直接实例化 AGraph 对象，无需远程 API 调用
- 支持多个集合，每个集合独立的 AGraph 实例
- 自动初始化和资源管理

### 智能搜索策略

- **向量搜索优先**: 对于文本块和实体搜索，优先使用向量搜索
- **回退机制**: 向量搜索失败时自动回退到关键词匹配
- **多类型搜索**: 自然语言搜索支持同时搜索多种类型的数据

### 性能优化

- **懒初始化**: 仅在需要时初始化 AGraph 实例
- **实例复用**: 相同集合名称复用 AGraph 实例
- **异步操作**: 全异步设计，提高并发性能

## 开发

### 运行测试

```bash
python test_server.py
```

### 代码格式化

```bash
black .
isort .
```

## 依赖关系

- **AGraph**: 本地知识图谱库（需要在父目录安装）
- **MCP**: Model Context Protocol 库
- **httpx**: HTTP 客户端库（用于 AGraph 内部）

## 与 API 版本的区别

| 特性 | API 版本 | 直接实例版本 |
|------|----------|--------------|
| 部署方式 | 需要独立 API 服务器 | 嵌入式，无需额外服务 |
| 性能 | 网络延迟 | 本地调用，更快 |
| 资源消耗 | 分离式 | 集成式 |
| 配置复杂度 | 需要配置 API 端点 | 直接配置 AGraph |
| 向量搜索 | 取决于 API 实现 | 直接使用 AGraph 向量能力 |

## 许可证

MIT License

## 贡献

欢迎提交 Issue 和 Pull Request！