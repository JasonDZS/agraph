# AGraph MCP Server - 项目发现指南

## 🎯 项目设置

MCP服务器会自动发现 `workdir/projects/` 目录下的所有项目。

### 📁 目录结构
```
workdir/projects/
├── project_a/              # 项目A
│   ├── vector_store/        # 向量存储
│   ├── knowledge_graph.json # 知识图谱数据
│   └── entities.json        # 实体数据
├── project_b/              # 项目B  
│   ├── *.db                # 数据库文件
│   ├── *.index             # 索引文件
│   └── *.pkl               # Pickle文件
└── project_c/              # 项目C
    └── relations.json       # 关系数据
```

## 🛠️ 可用工具

### 项目发现
- `list_available_projects` - 列出所有可用项目（包括未加载的）
- `list_active_projects` - 列出当前活跃项目（已加载到内存）
- `validate_project` - 验证项目是否包含有效的知识图谱数据

### 项目管理
- `set_project_directory` - 设置/更改项目基础目录
- `cleanup_project` - 清理特定项目实例

### 语义搜索
- `search_entities` - 搜索实体
- `search_relations` - 搜索关系
- `search_text_chunks` - 搜索文本块
- `search_clusters` - 搜索聚类
- `semantic_search_all` - 综合语义搜索

## ⚙️ 环境变量配置

```bash
# 设置项目目录（可选，默认为 workdir/projects）
export PROJECT_DIR="/path/to/your/projects"

# 设置API密钥
export OPENAI_API_KEY="your-key"

# 启用调试模式（可选）
export DEBUG=true
```

## 🚀 使用示例

1. **启动服务器**：
   ```bash
   python server.py
   ```

2. **发现可用项目**：
   ```python
   projects = await session.call_tool("list_available_projects")
   ```

3. **验证项目**：
   ```python
   result = await session.call_tool("validate_project", {"project": "my_project"})
   ```

4. **搜索项目内容**：
   ```python
   results = await session.call_tool("search_entities", {
       "project": "my_project", 
       "query": "人工智能",
       "top_k": 5
   })
   ```

## 📋 项目数据指标

服务器会检查以下文件/目录来判断项目是否有效：
- `agraph_vectordb/` 目录 (AGraph向量数据库)
- `chroma/` 目录 (Chroma向量数据库)
- `document_storage/` 目录 (文档存储)
- `config.json` 配置文件
- `vector_store/` 目录 (通用向量存储)
- `knowledge_graph.json` 知识图谱数据
- `entities.json` / `relations.json` 实体关系数据
- `*.db` / `*.sqlite3` 数据库文件
- `*.index` 索引文件
- `*.pkl` Pickle文件

## 🔧 故障排除

1. **项目未发现**：检查目录路径和权限
2. **项目验证失败**：确保包含上述数据文件之一
3. **搜索无结果**：验证项目是否包含有效的知识图谱数据

---
使用 `get_server_config` 工具查看当前配置信息。