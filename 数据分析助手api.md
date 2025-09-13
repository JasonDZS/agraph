# 数据分析助手使用指南 📊

## 概述

数据分析助手是一个基于AI的智能数据分析平台，能够将自然语言查询转换为SQL语句，并生成美观的可视化图表。本指南将详细介绍如何使用该平台进行数据分析。


## 完整使用流程

### 第一步：启动服务

```bash
# 进入backend目录
cd backend

# 安装依赖
uv sync

# 启动服务
uv run python start.py
```

服务启动后，API将在 `http://localhost:8000` 运行。

### 第二步：上传数据文件

#### API端点
```
POST /api/v1/database/upload-files
```

#### 请求参数
- **files**: 文件列表（支持.csv和.xlsx格式）
- **db_name**: 数据库名称（字符串）

#### 请求示例
```bash
curl -X POST http://localhost:8000/api/v1/database/upload-files \
  -F "files=@sales_data.csv" \
  -F "files=@customer_data.xlsx" \
  -F "db_name=sales_analysis"
```


### 第三步：查看数据库信息

#### 列出所有数据库
```
GET /api/v1/database/list
```

#### 获取数据库Schema
```
GET /api/v1/database/schema/{db_name}
```

#### 请求示例
```bash
# 列出所有数据库
curl http://localhost:8000/api/v1/database/list

# 获取特定数据库的schema
curl http://localhost:8000/api/v1/database/schema/sales_analysis
```

### 第四步：生成可视化图表

#### API端点
```
POST /api/v1/visualization/generate
```

#### 请求参数
- **query**: 自然语言查询（字符串）
- **db_name**: 数据库名称（字符串）
- **chart_type**: 图表类型（可选，如果没提供则从自然语言中推理，字符串）

#### 支持的图表类型
- `bar` - 柱状图
- `line` - 折线图
- `pie` - 饼图
- `scatter` - 散点图
- `area` - 面积图
- `radar` - 雷达图

#### 请求示例
```bash
curl -X POST http://localhost:8000/api/v1/visualization/generate \
  -F "query=Create a bar chart showing sales by region" \
  -F "db_name=sales_analysis" \
  -F "chart_type=bar"
```

#### 响应
返回完整的HTML页面，包含ECharts可视化图表。



## 高级功能

### 1. 查看系统状态

#### API端点
```
GET /api/v1/system/health
GET /api/v1/system/status
```

#### 请求示例
```bash
# 健康检查
curl http://localhost:8000/api/v1/system/health

# 系统状态
curl http://localhost:8000/api/v1/system/status
```

### 2. 查看日志

#### API端点
```
GET /api/v1/logs/requests
GET /api/v1/logs/requests/{request_id}
GET /api/v1/logs/stats
```

#### 请求示例
```bash
# 获取一些日志
curl http://localhost:8000/api/v1/logs/requests

# 获取指定ID的日志
curl http://localhost:8000/api/v1/logs/requests/{request_id}

# 获取日志的统计数据
curl http://localhost:8000/api/v1/logs/stats
```