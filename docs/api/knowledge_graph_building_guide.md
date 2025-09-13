# AGraph 知识库构建 API 操作指南

本文档提供了通过 AGraph API 构建知识库的详细步骤指导。按照以下步骤，您可以从上传文档到构建完整的知识图谱。

## 目录

1. [准备工作](#1-准备工作)
2. [项目管理](#2-项目管理)
3. [文档上传](#3-文档上传)
4. [知识图谱构建](#4-知识图谱构建)
5. [知识图谱管理](#5-知识图谱管理)
6. [查询与搜索](#6-查询与搜索)
7. [聊天对话](#7-聊天对话)
8. [完整工作流示例](#8-完整工作流示例)

## 1. 准备工作

### 1.1 启动 AGraph API 服务

```bash
# 启动 API 服务器
python -m agraph.api.app

# 或者使用 uvicorn 直接启动
uvicorn agraph.api.app:app --host 0.0.0.0 --port 8000 --reload
```

服务器启动后，API 将在 `http://localhost:8000` 上可用。

### 1.2 检查服务状态

```bash
curl -X GET "http://localhost:8000/health"
```

**响应示例：**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-20T10:00:00.000Z",
  "version": "0.2.2"
}
```

## 2. 项目管理

### 2.1 创建新项目

为了更好地组织您的知识库，建议为每个独立的知识域创建单独的项目。**项目之间完全独立，可以并行操作，无需切换**。

**API 端点：** `POST /projects/create`

```bash
curl -X POST "http://localhost:8000/projects/create" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my_knowledge_base",
    "description": "我的知识库项目"
  }'
```

**响应示例：**
```json
{
    "status": "success",
    "message": "Project 'my_knowledge_base' created successfully with complete Settings configuration",
    "timestamp": "2025-09-12T16:09:27.408182",
    "data": {
        "project_name": "my_knowledge_base",
        "description": "我的知识库项目",
        "created_at": "2025-09-12T16:09:27.405318",
        "version": "0.2.2",
        "paths": {
            "project_dir": "workdir/projects/my_knowledge_base",
            "config_file": "workdir/projects/my_knowledge_base/config.json",
            "document_storage": "workdir/projects/my_knowledge_base/document_storage",
            "vector_db": "workdir/projects/my_knowledge_base/agraph_vectordb",
            "cache": "workdir/projects/my_knowledge_base/cache",
            "logs": "workdir/projects/my_knowledge_base/logs"
        },
        "config_path": "workdir/projects/my_knowledge_base/config.json",
        "backup_config_path": "workdir/projects/my_knowledge_base/config.json",
        "project_paths": {
            "project_dir": "workdir/projects/my_knowledge_base",
            "config_file": "workdir/projects/my_knowledge_base/config.json",
            "document_storage": "workdir/projects/my_knowledge_base/document_storage",
            "vector_db": "workdir/projects/my_knowledge_base/agraph_vectordb",
            "cache": "workdir/projects/my_knowledge_base/cache",
            "logs": "workdir/projects/my_knowledge_base/logs"
        },
        "settings_saved": true,
        "complete_settings": {
            "workdir": "workdir",
            "current_project": "my_knowledge_base",
            "max_current": 5,
            "openai": {
                "api_key": "sk-*******************************************************PjvE",
                "api_base": "https://one-api.s.metames.cn:38443/v1"
            },
            "llm": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "temperature": 0.0,
                "max_tokens": 4096,
                "provider": "openai"
            },
            "embedding": {
                "model": "BAAI/bge-m3",
                "provider": "openai",
                "dimension": 1024,
                "max_token_size": 8192,
                "batch_size": 32
            },
            "graph": {
                "entity_types": [
                    "person",
                    "organization",
                    "location",
                    "concept",
                    "event",
                    "other",
                    "table",
                    "column",
                    "database",
                    "document",
                    "keyword",
                    "product",
                    "software",
                    "unknown"
                ],
                "relation_types": [
                    "contains",
                    "belongs_to",
                    "located_in",
                    "works_for",
                    "causes",
                    "part_of",
                    "is_a",
                    "references",
                    "similar_to",
                    "related_to",
                    "depends_on",
                    "foreign_key",
                    "mentions",
                    "describes",
                    "synonyms",
                    "develops",
                    "creates",
                    "founded_by",
                    "other"
                ]
            },
            "text": {
                "max_chunk_size": 512,
                "chunk_overlap": 100
            },
            "rag": {
                "system_prompt": "# System Role\n\nYou are an expert knowledge assistant specializing in information retrieval and synthesis from structured knowledge graphs and document collections.\n\n## Objective\n\nProvide comprehensive, well-structured responses to user queries by synthesizing information from the provided data sources. Your responses must be grounded exclusively in the given data sources while maintaining accuracy, clarity, and proper attribution.\n\n## Data Sources Available\n\n**Knowledge Graph (KG)**: Structured entities, relationships, and semantic connections\n**Document Chunks (DC)**: Relevant text segments from documents with contextual information\n\n### Temporal Information Handling\n- Each data point includes a `created_at` timestamp indicating knowledge acquisition time\n- For conflicting information, evaluate both content relevance and temporal context\n- Prioritize content-based temporal information over creation timestamps\n- Apply contextual judgment rather than defaulting to most recent information\n\n---\n\n## Conversation Context\n{history}\n\n## Available Knowledge Sources\n{kg_context}\n\n---\n\n## Response Requirements\n\n### Format and Structure\n- **Response Type**: {response_type}\n- **Language**: Respond in the same language as the user's question\n- **Formatting**: Use markdown with clear section headers and proper structure\n- **Continuity**: Maintain coherence with conversation history\n\n### Content Organization\n- Structure responses with focused sections addressing distinct aspects\n- Use descriptive section headers that clearly indicate content focus\n- Present information in logical, easily digestible segments\n\n### Citation System\n- **Inline Citations**: Use the format `[ID:reference_number]` immediately after each statement or claim that references data sources\n  - Example: `The system processes over 10,000 queries daily [ID:1].`\n  - Example: `According to the latest research findings [ID:2], performance improved significantly.`\n  - Place citations at the end of sentences or clauses, before punctuation\n\n- **References Section**: Always conclude with a \"# References\" section containing:\n  - Format: `ID:number - [Source_Type] Brief description of the source content`\n  - Source type indicators: `[KG]` for Knowledge Graph, `[DC]` for Document Chunks\n  - Maximum 5 most relevant references\n\n### Reference Format Template\n```\n### References\n- ID:1 [KG] Entity relationship describing system performance metrics\n- ID:2 [DC] Research document excerpt about performance improvements\n- ID:3 [KG] Semantic connection between entities showing growth trends\n```\n\n### Quality Standards\n- **Accuracy**: Base all claims exclusively on provided data sources\n- **Transparency**: Clearly distinguish between different source types\n- **Completeness**: Address all relevant aspects found in the data sources\n- **Honesty**: State limitations clearly when information is insufficient\n- **No Fabrication**: Never generate information not present in the provided sources\n\nIf the available data sources are insufficient to answer the query, explicitly state this limitation and describe what additional information would be needed."
            },
            "builder": {
                "enable_cache": true,
                "cache_dir": "workdir/projects/kb1bfc5d3c/agraph_vectordb/cache",
                "cache_ttl": 86400,
                "auto_cleanup": true,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "entity_confidence_threshold": 0.7,
                "entity_types": [
                    "person",
                    "organization",
                    "location",
                    "concept",
                    "event",
                    "other",
                    "table",
                    "column",
                    "database",
                    "document",
                    "keyword",
                    "product",
                    "software",
                    "unknown"
                ],
                "relation_confidence_threshold": 0.6,
                "relation_types": [
                    "contains",
                    "belongs_to",
                    "located_in",
                    "works_for",
                    "causes",
                    "part_of",
                    "is_a",
                    "references",
                    "similar_to",
                    "related_to",
                    "depends_on",
                    "foreign_key",
                    "mentions",
                    "describes",
                    "synonyms",
                    "develops",
                    "creates",
                    "founded_by",
                    "other"
                ],
                "cluster_algorithm": "community_detection",
                "min_cluster_size": 2,
                "enable_user_interaction": true,
                "auto_save_edits": true,
                "llm_config": {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "provider": "openai"
                },
                "embedding_config": {
                    "model": "BAAI/bge-m3",
                    "provider": "openai",
                    "dimension": 1024,
                    "max_token_size": 8192,
                    "batch_size": 32
                },
                "openai_config": {
                    "api_key": "sk-*******************************************************PjvE",
                    "api_base": "https://one-api.s.metames.cn:38443/v1"
                }
            }
        }
    },
    "project_name": "my_knowledge_base"
}
```

### 2.2 项目列表查询

**API 端点：** `GET /projects/list`

```bash
curl -X GET "http://localhost:8000/projects/list"
```

**响应示例：**
```json
{
    "status": "success",
    "message": "Found 7 projects",
    "timestamp": "2025-09-12T16:12:42.140677",
    "projects": [
        "finance",
        "kb1bfc5d3c",
        "kb4dc6750d",
        "kbb7ca371c",
        "kbef3ffcf4",
        "my_knowledge_base",
        "test_postman"
    ]
}
```

### 2.3 项目信息查询

**API 端点：** `GET /projects/{project_name}`

```bash
curl -X GET "http://localhost:8000/projects/my_knowledge_base"
```

**响应示例：**

```json
{
    "status": "success",
    "message": "Project 'my_knowledge_base' information retrieved with local config",
    "timestamp": "2025-09-12T16:30:53.923783",
    "data": {
        "project_name": "my_knowledge_base",
        "paths": {
            "project_dir": "workdir/projects/my_knowledge_base",
            "config_file": "workdir/projects/my_knowledge_base/config.json",
            "document_storage": "workdir/projects/my_knowledge_base/document_storage",
            "vector_db": "workdir/projects/my_knowledge_base/agraph_vectordb",
            "cache": "workdir/projects/my_knowledge_base/cache",
            "logs": "workdir/projects/my_knowledge_base/logs"
        },
        "settings_data": {
            "workdir": "workdir",
            "current_project": "my_knowledge_base",
            "max_current": 5,
            "openai": {
                "api_key": "sk-*******************************************************PjvE",
                "api_base": "https://one-api.s.metames.cn:38443/v1"
            },
            "llm": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "temperature": 0.0,
                "max_tokens": 4096,
                "provider": "openai"
            },
            "embedding": {
                "model": "BAAI/bge-m3",
                "provider": "openai",
                "dimension": 1024,
                "max_token_size": 8192,
                "batch_size": 32
            },
            "graph": {
                "entity_types": [
                    "person",
                    "organization",
                    "location",
                    "concept",
                    "event",
                    "other",
                    "table",
                    "column",
                    "database",
                    "document",
                    "keyword",
                    "product",
                    "software",
                    "unknown"
                ],
                "relation_types": [
                    "contains",
                    "belongs_to",
                    "located_in",
                    "works_for",
                    "causes",
                    "part_of",
                    "is_a",
                    "references",
                    "similar_to",
                    "related_to",
                    "depends_on",
                    "foreign_key",
                    "mentions",
                    "describes",
                    "synonyms",
                    "develops",
                    "creates",
                    "founded_by",
                    "other"
                ]
            },
            "text": {
                "max_chunk_size": 512,
                "chunk_overlap": 100
            },
            "rag": {
                "system_prompt": "# System Role\n\nYou are an expert knowledge assistant specializing in information retrieval and synthesis from structured knowledge graphs and document collections.\n\n## Objective\n\nProvide comprehensive, well-structured responses to user queries by synthesizing information from the provided data sources. Your responses must be grounded exclusively in the given data sources while maintaining accuracy, clarity, and proper attribution.\n\n## Data Sources Available\n\n**Knowledge Graph (KG)**: Structured entities, relationships, and semantic connections\n**Document Chunks (DC)**: Relevant text segments from documents with contextual information\n\n### Temporal Information Handling\n- Each data point includes a `created_at` timestamp indicating knowledge acquisition time\n- For conflicting information, evaluate both content relevance and temporal context\n- Prioritize content-based temporal information over creation timestamps\n- Apply contextual judgment rather than defaulting to most recent information\n\n---\n\n## Conversation Context\n{history}\n\n## Available Knowledge Sources\n{kg_context}\n\n---\n\n## Response Requirements\n\n### Format and Structure\n- **Response Type**: {response_type}\n- **Language**: Respond in the same language as the user's question\n- **Formatting**: Use markdown with clear section headers and proper structure\n- **Continuity**: Maintain coherence with conversation history\n\n### Content Organization\n- Structure responses with focused sections addressing distinct aspects\n- Use descriptive section headers that clearly indicate content focus\n- Present information in logical, easily digestible segments\n\n### Citation System\n- **Inline Citations**: Use the format `[ID:reference_number]` immediately after each statement or claim that references data sources\n  - Example: `The system processes over 10,000 queries daily [ID:1].`\n  - Example: `According to the latest research findings [ID:2], performance improved significantly.`\n  - Place citations at the end of sentences or clauses, before punctuation\n\n- **References Section**: Always conclude with a \"# References\" section containing:\n  - Format: `ID:number - [Source_Type] Brief description of the source content`\n  - Source type indicators: `[KG]` for Knowledge Graph, `[DC]` for Document Chunks\n  - Maximum 5 most relevant references\n\n### Reference Format Template\n```\n### References\n- ID:1 [KG] Entity relationship describing system performance metrics\n- ID:2 [DC] Research document excerpt about performance improvements\n- ID:3 [KG] Semantic connection between entities showing growth trends\n```\n\n### Quality Standards\n- **Accuracy**: Base all claims exclusively on provided data sources\n- **Transparency**: Clearly distinguish between different source types\n- **Completeness**: Address all relevant aspects found in the data sources\n- **Honesty**: State limitations clearly when information is insufficient\n- **No Fabrication**: Never generate information not present in the provided sources\n\nIf the available data sources are insufficient to answer the query, explicitly state this limitation and describe what additional information would be needed."
            },
            "builder": {
                "enable_cache": true,
                "cache_dir": "./cache",
                "cache_ttl": 86400,
                "auto_cleanup": true,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "entity_confidence_threshold": 0.7,
                "entity_types": [
                    "person",
                    "organization",
                    "location",
                    "concept",
                    "event",
                    "other",
                    "table",
                    "column",
                    "database",
                    "document",
                    "keyword",
                    "product",
                    "software",
                    "unknown"
                ],
                "relation_confidence_threshold": 0.6,
                "relation_types": [
                    "contains",
                    "belongs_to",
                    "located_in",
                    "works_for",
                    "causes",
                    "part_of",
                    "is_a",
                    "references",
                    "similar_to",
                    "related_to",
                    "depends_on",
                    "foreign_key",
                    "mentions",
                    "describes",
                    "synonyms",
                    "develops",
                    "creates",
                    "founded_by",
                    "other"
                ],
                "cluster_algorithm": "community_detection",
                "min_cluster_size": 2,
                "enable_user_interaction": true,
                "auto_save_edits": true,
                "llm_config": {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "provider": "openai"
                },
                "embedding_config": {
                    "model": "BAAI/bge-m3",
                    "provider": "openai",
                    "dimension": 1024,
                    "max_token_size": 8192,
                    "batch_size": 32
                },
                "openai_config": {
                    "api_key": "sk-*******************************************************PjvE",
                    "api_base": "https://one-api.s.metames.cn:38443/v1"
                }
            }
        },
        "backup_available": true,
        "config_format": "direct_settings",
        "current_settings": {
            "workdir": "workdir",
            "current_project": "my_knowledge_base",
            "max_current": 5,
            "openai": {
                "api_key": "sk-*******************************************************PjvE",
                "api_base": "https://one-api.s.metames.cn:38443/v1"
            },
            "llm": {
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "temperature": 0.0,
                "max_tokens": 4096,
                "provider": "openai"
            },
            "embedding": {
                "model": "BAAI/bge-m3",
                "provider": "openai",
                "dimension": 1024,
                "max_token_size": 8192,
                "batch_size": 32
            },
            "graph": {
                "entity_types": [
                    "person",
                    "organization",
                    "location",
                    "concept",
                    "event",
                    "other",
                    "table",
                    "column",
                    "database",
                    "document",
                    "keyword",
                    "product",
                    "software",
                    "unknown"
                ],
                "relation_types": [
                    "contains",
                    "belongs_to",
                    "located_in",
                    "works_for",
                    "causes",
                    "part_of",
                    "is_a",
                    "references",
                    "similar_to",
                    "related_to",
                    "depends_on",
                    "foreign_key",
                    "mentions",
                    "describes",
                    "synonyms",
                    "develops",
                    "creates",
                    "founded_by",
                    "other"
                ]
            },
            "text": {
                "max_chunk_size": 512,
                "chunk_overlap": 100
            },
            "rag": {
                "system_prompt": "# System Role\n\nYou are an expert knowledge assistant specializing in information retrieval and synthesis from structured knowledge graphs and document collections.\n\n## Objective\n\nProvide comprehensive, well-structured responses to user queries by synthesizing information from the provided data sources. Your responses must be grounded exclusively in the given data sources while maintaining accuracy, clarity, and proper attribution.\n\n## Data Sources Available\n\n**Knowledge Graph (KG)**: Structured entities, relationships, and semantic connections\n**Document Chunks (DC)**: Relevant text segments from documents with contextual information\n\n### Temporal Information Handling\n- Each data point includes a `created_at` timestamp indicating knowledge acquisition time\n- For conflicting information, evaluate both content relevance and temporal context\n- Prioritize content-based temporal information over creation timestamps\n- Apply contextual judgment rather than defaulting to most recent information\n\n---\n\n## Conversation Context\n{history}\n\n## Available Knowledge Sources\n{kg_context}\n\n---\n\n## Response Requirements\n\n### Format and Structure\n- **Response Type**: {response_type}\n- **Language**: Respond in the same language as the user's question\n- **Formatting**: Use markdown with clear section headers and proper structure\n- **Continuity**: Maintain coherence with conversation history\n\n### Content Organization\n- Structure responses with focused sections addressing distinct aspects\n- Use descriptive section headers that clearly indicate content focus\n- Present information in logical, easily digestible segments\n\n### Citation System\n- **Inline Citations**: Use the format `[ID:reference_number]` immediately after each statement or claim that references data sources\n  - Example: `The system processes over 10,000 queries daily [ID:1].`\n  - Example: `According to the latest research findings [ID:2], performance improved significantly.`\n  - Place citations at the end of sentences or clauses, before punctuation\n\n- **References Section**: Always conclude with a \"# References\" section containing:\n  - Format: `ID:number - [Source_Type] Brief description of the source content`\n  - Source type indicators: `[KG]` for Knowledge Graph, `[DC]` for Document Chunks\n  - Maximum 5 most relevant references\n\n### Reference Format Template\n```\n### References\n- ID:1 [KG] Entity relationship describing system performance metrics\n- ID:2 [DC] Research document excerpt about performance improvements\n- ID:3 [KG] Semantic connection between entities showing growth trends\n```\n\n### Quality Standards\n- **Accuracy**: Base all claims exclusively on provided data sources\n- **Transparency**: Clearly distinguish between different source types\n- **Completeness**: Address all relevant aspects found in the data sources\n- **Honesty**: State limitations clearly when information is insufficient\n- **No Fabrication**: Never generate information not present in the provided sources\n\nIf the available data sources are insufficient to answer the query, explicitly state this limitation and describe what additional information would be needed."
            },
            "builder": {
                "enable_cache": true,
                "cache_dir": "workdir/projects/kb1bfc5d3c/agraph_vectordb/cache",
                "cache_ttl": 86400,
                "auto_cleanup": true,
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "entity_confidence_threshold": 0.7,
                "entity_types": [
                    "person",
                    "organization",
                    "location",
                    "concept",
                    "event",
                    "other",
                    "table",
                    "column",
                    "database",
                    "document",
                    "keyword",
                    "product",
                    "software",
                    "unknown"
                ],
                "relation_confidence_threshold": 0.6,
                "relation_types": [
                    "contains",
                    "belongs_to",
                    "located_in",
                    "works_for",
                    "causes",
                    "part_of",
                    "is_a",
                    "references",
                    "similar_to",
                    "related_to",
                    "depends_on",
                    "foreign_key",
                    "mentions",
                    "describes",
                    "synonyms",
                    "develops",
                    "creates",
                    "founded_by",
                    "other"
                ],
                "cluster_algorithm": "community_detection",
                "min_cluster_size": 2,
                "enable_user_interaction": true,
                "auto_save_edits": true,
                "llm_config": {
                    "model": "Qwen/Qwen2.5-7B-Instruct",
                    "temperature": 0.0,
                    "max_tokens": 4096,
                    "provider": "openai"
                },
                "embedding_config": {
                    "model": "BAAI/bge-m3",
                    "provider": "openai",
                    "dimension": 1024,
                    "max_token_size": 8192,
                    "batch_size": 32
                },
                "openai_config": {
                    "api_key": "sk-*******************************************************PjvE",
                    "api_base": "https://one-api.s.metames.cn:38443/v1"
                }
            }
        },
        "config_cache_status": "loaded_from_local",
        "config_file_path": "workdir/projects/my_knowledge_base/config.json",
        "statistics": {
            "project_exists": true,
            "document_count": 0,
            "has_vector_db": false,
            "cache_size": 0,
            "total_size_mb": 0,
            "config_file_exists": true,
            "config_file_size": 6067
        }
    },
    "project_name": "my_knowledge_base"
}
```

## 3. 文档上传

### 3.1 上传文件

支持多种文件格式：PDF、DOCX、TXT、MD 等。

**API 端点：** `POST /documents/upload`

```bash
curl -X POST "http://localhost:8000/documents/upload?project_name=my_knowledge_base" \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "files=@document3.txt" \
  -F "metadata={\"source\": \"research_papers\", \"category\": \"AI\"}" \
  -F "tags=[\"machine_learning\", \"research\"]"
```

**响应示例：**
```json
{
  "status": "success",
  "message": "Successfully uploaded 3 documents",
  "data": {
    "uploaded_documents": [
      {
        "id": "doc_001",
        "filename": "document1.pdf",
        "content_type": "application/pdf",
        "size": 1024000,
        "extracted_metadata": {
          "page_count": 10,
          "title": "Machine Learning Fundamentals"
        }
      },
      {
        "id": "doc_002", 
        "filename": "document2.docx",
        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "size": 512000
      }
    ],
    "total_uploaded": 3
  }
}
```

### 3.2 从文本直接上传

**API 端点：** `POST /documents/from-text`

```bash
curl -X POST "http://localhost:8000/documents/from-text?project_name=my_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
      "机器学习是人工智能的一个子集，它使计算机能够自动学习和改进，而无需显式编程。"
    ],
    "metadata": {"source": "manual_input", "topic": "AI_basics"},
    "tags": ["AI", "definition"]
  }'
```

### 3.3 查看文档列表

**API 端点：** `GET /documents/list`

```bash
curl -X GET "http://localhost:8000/documents/list?project_name=my_knowledge_base&page=1&page_size=10"
```

**响应示例：**
```json
{
  "status": "success",
  "message": "Retrieved 3 documents",
  "data": {
    "documents": [
      {
        "id": "doc_001",
        "filename": "document1.pdf",
        "created_at": "2024-01-20T10:00:00.000Z",
        "tags": ["machine_learning", "research"],
        "metadata": {
          "content_type": "application/pdf",
          "file_size": 1024000
        }
      }
    ],
    "pagination": {
      "page": 1,
      "page_size": 10,
      "total_count": 3,
      "total_pages": 1
    }
  }
}
```

## 4. 知识图谱构建

### 4.1 构建知识图谱

**API 端点：** `POST /knowledge-graph/build`

#### 基于所有文档构建

```bash
curl -X POST "http://localhost:8000/knowledge-graph/build?project_name=my_knowledge_base" \
  -H "Content-Type: application/json" \
  -d '{
    "graph_name": "完整知识图谱",
    "graph_description": "基于所有上传文档的知识图谱",
    "use_cache": true,
    "save_to_vector_store": true,
    "enable_graph": false
  }'
```

注意 `"enable_graph"` 可以用来控制知识图谱生成，当为false时不启用知识图谱

### 4.2 查看构建状态

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
    "graph_name": "AI知识图谱",
    "graph_description": "基于AI研究文档构建的知识图谱",
    "statistics": {
      "entities": 45,
      "relations": 78,
      "clusters": 12,
      "text_chunks": 156
    },
    "entity_types": {
      "PERSON": 8,
      "ORGANIZATION": 12,
      "CONCEPT": 15,
      "TECHNOLOGY": 10
    },
    "relation_types": {
      "DEVELOPS": 15,
      "USES": 20,
      "RELATED_TO": 25,
      "PART_OF": 18
    }
  }
}
```

### 4.3 删除知识图谱

**API 端点：** `DELETE /knowledge-graph/delete`

```bash
curl -X DELETE "http://localhost:8000/knowledge-graph/delete?project_name=my_knowledge_base"
```

**响应示例：**
```json
{
  "status": "success",
  "message": "Knowledge graph '完整知识图谱' deleted successfully",
  "timestamp": "2025-09-12T17:00:00.000Z",
  "data": {
    "exists": false,
    "graph_name": null,
    "graph_description": null,
    "statistics": {
      "entities": 0,
      "relations": 0,
      "clusters": 0,
      "text_chunks": 0
    },
    "deleted": true,
    "deleted_stats": {
      "entities": 45,
      "relations": 78,
      "clusters": 12,
      "text_chunks": 156
    }
  }
}
```

**注意事项：**
- 此操作将删除所有知识图谱数据（实体、关系、聚类、文本分块）
- 同时会清空向量数据库中相关数据
- **会删除磁盘上保存的知识图谱文件（.json 文件）**
- **此操作不可逆，请谨慎使用**
- 删除后需要重新构建知识图谱才能使用相关功能