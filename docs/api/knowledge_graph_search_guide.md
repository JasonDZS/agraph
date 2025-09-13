# AGraph 知识图谱检索 API 操作指南

本文档提供了通过 AGraph API 检索知识图谱的详细指导，包括实体查询、关系查询、文本块搜索、可视化数据获取等操作。

## 目录

1. [准备工作](#1-准备工作)
2. [知识图谱状态查询](#2-知识图谱状态查询)
3. [完整知识图谱获取](#3-完整知识图谱获取)
4. [实体检索](#4-实体检索)
5. [关系检索](#5-关系检索)
6. [文本块检索](#6-文本块检索)
7. [聚类检索](#7-聚类检索)
8. [可视化数据获取](#8-可视化数据获取)
9. [搜索功能](#9-搜索功能)
10. [完整检索工作流示例](#10-完整检索工作流示例)

## 1. 准备工作

### 1.1 确认服务状态

在开始检索前，请确保 API 服务正常运行：

```bash
curl -X GET "http://localhost:8000/health"
```

### 1.2 确认知识图谱状态

**API 端点：** `GET /knowledge-graph/status`

```bash
curl -X GET "http://localhost:8000/knowledge-graph/status?project_name=my_knowledge_base"
```

**响应示例：**
```json
{
  "status": "success",
  "message": "Knowledge graph status retrieved successfully",
  "data": {
    "exists": true,
    "graph_name": "完整知识图谱",
    "graph_description": "基于所有上传文档的知识图谱",
    "statistics": {
      "entities": 145,
      "relations": 234,
      "clusters": 23,
      "text_chunks": 456
    },
    "entity_types": {
      "person": 45,
      "organization": 32,
      "concept": 38,
      "location": 20,
      "other": 10
    },
    "relation_types": {
      "related_to": 89,
      "belongs_to": 67,
      "is_a": 45,
      "located_in": 33
    },
    "system_info": {
      "agraph_initialized": true,
      "vector_store_type": "chroma",
      "enable_knowledge_graph": true
    }
  }
}
```

## 2. 知识图谱状态查询

### 2.1 获取基本统计信息

**API 端点：** `GET /knowledge-graph/status`

**查询参数：**
- `project_name` (optional): 项目名称，用于指定特定的知识图谱

**响应字段说明：**
- `exists`: 是否存在知识图谱
- `statistics`: 基本统计信息（实体数、关系数、聚类数、文本块数）
- `entity_types`: 各类型实体的数量分布
- `relation_types`: 各类型关系的数量分布
- `system_info`: 系统配置信息

## 3. 完整知识图谱获取

### 3.1 获取完整知识图谱数据

**API 端点：** `GET /knowledge-graph/get`

```bash
curl -X GET "http://localhost:8000/knowledge-graph/get?project_name=my_knowledge_base&include_clusters=true&include_text_chunks=false&entity_limit=100&relation_limit=200"
```

**查询参数：**
- `include_text_chunks` (boolean, 默认: false): 是否包含文本块
- `include_clusters` (boolean, 默认: false): 是否包含聚类信息
- `entity_limit` (integer, 可选): 限制返回的实体数量
- `relation_limit` (integer, 可选): 限制返回的关系数量
- `project_name` (string, 可选): 项目名称

**响应示例：**
```json
{
  "status": "success",
  "message": "Knowledge graph retrieved successfully",
  "data": {
    "graph_name": "完整知识图谱",
    "graph_description": "基于所有上传文档的知识图谱",
    "entities": [
      {
        "id": "entity_001",
        "name": "人工智能",
        "entity_type": "concept",
        "description": "计算机科学的一个分支",
        "confidence": 0.95,
        "properties": {
          "domain": "technology",
          "relevance": "high"
        },
        "aliases": ["AI", "Artificial Intelligence"]
      },
      {
        "id": "entity_002",
        "name": "机器学习",
        "entity_type": "concept",
        "description": "人工智能的子集",
        "confidence": 0.92,
        "properties": {},
        "aliases": ["ML", "Machine Learning"]
      }
    ],
    "relations": [
      {
        "id": "relation_001",
        "head_entity_id": "entity_002",
        "tail_entity_id": "entity_001",
        "relation_type": "part_of",
        "description": "机器学习是人工智能的一部分",
        "confidence": 0.89,
        "properties": {}
      }
    ],
    "clusters": [
      {
        "id": "cluster_001",
        "name": "AI技术概念",
        "description": "人工智能相关的技术概念聚类",
        "entities": ["entity_001", "entity_002"],
        "relations": [],
        "confidence": 1.0
      }
    ]
  }
}
```

## 4. 实体检索

### 4.1 获取所有实体

**API 端点：** `GET /knowledge-graph/entities`

```bash
curl -X GET "http://localhost:8000/knowledge-graph/entities?project_name=my_knowledge_base&entity_type=person&limit=50&offset=0"
```

**查询参数：**
- `project_name` (string, 可选): 项目名称
- `entity_type` (string, 可选): 按实体类型过滤（person, organization, location, concept, event, other 等）
- `limit` (integer, 可选): 每页返回的实体数量
- `offset` (integer, 默认: 0): 分页偏移量

**响应示例：**
```json
{
  "status": "success",
  "message": "Retrieved 25 entities",
  "data": {
    "entities": [
      {
        "id": "entity_003",
        "name": "张三",
        "entity_type": "person",
        "description": "AI研究员",
        "confidence": 0.87,
        "properties": {
          "position": "研究员",
          "department": "AI实验室"
        },
        "aliases": ["Dr. Zhang"],
        "text_chunks": ["chunk_001", "chunk_015"],
        "created_at": "2024-01-20T10:00:00.000Z",
        "updated_at": "2024-01-20T15:30:00.000Z"
      }
    ],
    "pagination": {
      "total": 45,
      "limit": 50,
      "offset": 0,
      "has_more": false
    },
    "filters": {
      "entity_type": "person"
    }
  }
}
```

### 4.2 实体类型统计

通过状态接口可以获得各类型实体的数量统计：

```bash
curl -X GET "http://localhost:8000/knowledge-graph/status?project_name=my_knowledge_base"
```

## 5. 关系检索

### 5.1 获取所有关系

**API 端点：** `GET /knowledge-graph/relations`

```bash
curl -X GET "http://localhost:8000/knowledge-graph/relations?project_name=my_knowledge_base&relation_type=works_for&entity_id=entity_003&limit=30&offset=0"
```

**查询参数：**
- `project_name` (string, 可选): 项目名称
- `relation_type` (string, 可选): 按关系类型过滤
- `entity_id` (string, 可选): 按实体ID过滤（头实体或尾实体）
- `limit` (integer, 可选): 每页返回的关系数量
- `offset` (integer, 默认: 0): 分页偏移量

**响应示例：**
```json
{
  "status": "success",
  "message": "Retrieved 15 relations",
  "data": {
    "relations": [
      {
        "id": "relation_002",
        "head_entity_id": "entity_003",
        "tail_entity_id": "entity_004",
        "head_entity_name": "张三",
        "tail_entity_name": "AI实验室",
        "relation_type": "works_for",
        "description": "张三在AI实验室工作",
        "confidence": 0.91,
        "properties": {
          "start_date": "2023-01-01",
          "position": "研究员"
        },
        "text_chunks": ["chunk_003", "chunk_012"],
        "created_at": "2024-01-20T10:05:00.000Z",
        "updated_at": "2024-01-20T10:05:00.000Z"
      }
    ],
    "pagination": {
      "total": 15,
      "limit": 30,
      "offset": 0,
      "has_more": false
    },
    "filters": {
      "relation_type": "works_for",
      "entity_id": "entity_003"
    }
  }
}
```

### 5.2 支持的关系类型

常见的关系类型包括：
- `contains`: 包含关系
- `belongs_to`: 属于关系
- `located_in`: 位于关系
- `works_for`: 工作关系
- `causes`: 因果关系
- `part_of`: 部分关系
- `is_a`: 类别关系
- `references`: 引用关系
- `similar_to`: 相似关系
- `related_to`: 相关关系

## 6. 文本块检索

### 6.1 获取所有文本块

**API 端点：** `GET /knowledge-graph/text-chunks`

```bash
curl -X GET "http://localhost:8000/knowledge-graph/text-chunks?project_name=my_knowledge_base&limit=20&offset=0"
```

**查询参数：**
- `project_name` (string, 可选): 项目名称
- `limit` (integer, 可选): 每页返回的文本块数量
- `offset` (integer, 默认: 0): 分页偏移量

**响应示例：**
```json
{
  "status": "success",
  "message": "Retrieved 20 text chunks",
  "data": {
    "text_chunks": [
      {
        "id": "chunk_001",
        "content": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
        "source": "AI_introduction.pdf",
        "start_index": 0,
        "end_index": 45,
        "entities": ["entity_001"],
        "relations": ["relation_001"],
        "created_at": "2024-01-20T10:00:00.000Z",
        "updated_at": "2024-01-20T10:00:00.000Z"
      }
    ],
    "pagination": {
      "total": 456,
      "limit": 20,
      "offset": 0,
      "has_more": true
    }
  }
}
```

### 6.2 搜索文本块

**API 端点：** `POST /knowledge-graph/text-chunks`

```bash
curl -X POST "http://localhost:8000/knowledge-graph/text-chunks?project_name=my_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "search": "人工智能",
    "entity_id": "entity_001",
    "limit": 10,
    "offset": 0
  }'
```

**请求参数：**
- `search` (string, 可选): 搜索关键词，在文本内容和来源中搜索
- `entity_id` (string, 可选): 按实体ID过滤，返回包含该实体的文本块
- `limit` (integer, 默认: 20): 每页返回的文本块数量
- `offset` (integer, 默认: 0): 分页偏移量

**响应示例：**
```json
{
  "status": "success",
  "message": "Found 8 text chunks (returning 8)",
  "data": {
    "text_chunks": [
      {
        "id": "chunk_001",
        "content": "人工智能是计算机科学的一个分支...",
        "source": "AI_introduction.pdf",
        "start_index": 0,
        "end_index": 45,
        "entities": ["entity_001"],
        "relations": [],
        "entity_details": [
          {
            "id": "entity_001",
            "name": "人工智能",
            "entity_type": "concept"
          }
        ]
      }
    ],
    "pagination": {
      "total": 8,
      "limit": 10,
      "offset": 0,
      "has_more": false
    },
    "filters": {
      "search": "人工智能",
      "entity_id": "entity_001"
    }
  }
}
```

## 7. 聚类检索

### 7.1 获取所有聚类

**API 端点：** `GET /knowledge-graph/clusters`

```bash
curl -X GET "http://localhost:8000/knowledge-graph/clusters?project_name=my_knowledge_base&cluster_type=concept_group&limit=15&offset=0"
```

**查询参数：**
- `project_name` (string, 可选): 项目名称
- `cluster_type` (string, 可选): 按聚类类型过滤
- `limit` (integer, 可选): 每页返回的聚类数量
- `offset` (integer, 默认: 0): 分页偏移量

**响应示例：**
```json
{
  "status": "success",
  "message": "Retrieved 5 clusters",
  "data": {
    "clusters": [
      {
        "id": "cluster_001",
        "name": "AI技术概念",
        "description": "人工智能相关的技术概念聚类",
        "cluster_type": "concept_group",
        "entities": ["entity_001", "entity_002", "entity_005"],
        "relations": ["relation_001", "relation_003"],
        "confidence": 1.0,
        "properties": {
          "domain": "technology",
          "size": "medium"
        },
        "created_at": "2024-01-20T10:15:00.000Z",
        "updated_at": "2024-01-20T10:15:00.000Z"
      }
    ],
    "pagination": {
      "total": 23,
      "limit": 15,
      "offset": 0,
      "has_more": true
    },
    "filters": {
      "cluster_type": "concept_group"
    }
  }
}
```
