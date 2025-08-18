# API Services Layer

这个目录包含了 AGraph 前端应用的完整 API 服务层实现，提供了与后端 API 交互的所有功能。

## 🏗️ 架构概览

### 核心组件

- **ApiClient**: 基础 HTTP 客户端，包含请求/响应拦截器、错误处理、重试机制
- **各种服务**: 针对不同功能模块的专门服务 (项目、文档、知识图谱等)
- **状态管理**: 统一的请求状态和缓存管理
- **类型定义**: 完整的 TypeScript 类型支持

## 📂 文件结构

```
src/services/
├── api.ts                    # 基础 API 客户端
├── projectService.ts         # 项目管理服务
├── documentService.ts        # 文档管理服务
├── knowledgeGraphService.ts  # 知识图谱服务
├── chatService.ts           # 对话系统服务
├── searchService.ts         # 搜索功能服务
├── configService.ts         # 配置管理服务
├── requestStateManager.ts   # 请求状态管理器
├── index.ts                 # 统一导出
└── README.md               # 文档 (本文件)
```

## 🚀 功能特性

### ApiClient 基础功能

- ✅ 请求/响应拦截器支持
- ✅ 自动重试机制 (指数退避)
- ✅ 请求超时控制
- ✅ 内存缓存系统
- ✅ 统一错误处理
- ✅ 认证令牌管理

### 各服务模块功能

#### 项目服务 (projectService)

- 创建、列出、切换、删除项目
- 项目详情和统计信息
- 项目导入/导出
- 项目名称验证

#### 文档服务 (documentService)

- 文档上传 (支持多种格式)
- 文件上传进度跟踪
- 文档列表和搜索
- 文档元数据管理
- 批量操作支持

#### 知识图谱服务 (knowledgeGraphService)

- 图谱构建和更新
- 实体、关系、文本块查询
- 图谱可视化数据转换
- 图谱统计和状态监控
- 图谱导入/导出

#### 对话服务 (chatService)

- 发送消息 (同步/流式)
- 对话历史管理
- 流式响应支持 (Server-Sent Events)
- 上下文信息预览
- 消息验证和格式化

#### 搜索服务 (searchService)

- 基本搜索和高级搜索
- 实体、关系、文本块搜索
- 搜索建议和历史
- 语义搜索和相似度搜索
- 搜索结果高亮

#### 配置服务 (configService)

- 系统配置管理
- 配置导入/导出
- 系统状态监控
- 缓存管理
- 配置验证

### 状态管理功能

- ✅ 请求状态跟踪 (loading, error, data)
- ✅ 智能缓存机制
- ✅ 状态订阅/取消订阅
- ✅ 预配置的请求处理器
- ✅ 调试信息支持

## 💻 使用示例

### 基本用法

```typescript
import { projectService, documentService } from '@/services';

// 获取项目列表
const projects = await projectService.listProjects();

// 上传文档
const uploadResult = await documentService.uploadDocuments({
  texts: ['文档内容1', '文档内容2'],
  metadata: { source: 'user_upload' },
});
```

### 使用状态管理

```typescript
import { requestHandlers } from '@/services';

// 创建项目列表请求处理器
const projectHandler = requestHandlers.loadProjects();

// 订阅状态变化
const unsubscribe = projectHandler.subscribe(state => {
  if (state.loading) {
    console.log('正在加载项目...');
  } else if (state.error) {
    console.error('加载失败:', state.error);
  } else if (state.data) {
    console.log('项目列表:', state.data);
  }
});

// 执行请求
await projectHandler.execute();

// 清理订阅
unsubscribe();
```

### 流式对话

```typescript
import { chatService } from '@/services';

await chatService.sendStreamMessage(
  {
    question: '什么是知识图谱？',
    stream: true,
  },
  chunk => {
    // 处理流式响应块
    console.log('收到响应块:', chunk.chunk);
    if (chunk.finished) {
      console.log('完整回答:', chunk.answer);
    }
  },
  error => {
    console.error('流式响应错误:', error);
  }
);
```

### 文件上传进度

```typescript
import { documentService } from '@/services';

const files = [
  /* File objects */
];

await documentService.uploadFilesWithProgress(files, {
  onProgress: progress => {
    console.log(`上传进度: ${progress.percentage}%`);
  },
  timeout: 300000, // 5分钟超时
});
```

## 🔧 高级配置

### 自定义拦截器

```typescript
import { apiClient } from '@/services';

// 添加请求拦截器
apiClient.addRequestInterceptor(config => {
  // 添加自定义头部
  config.headers = {
    ...config.headers,
    'X-Custom-Header': 'value',
  };
  return config;
});

// 添加响应拦截器
apiClient.addResponseInterceptor({
  onFulfilled: response => {
    // 处理成功响应
    return response;
  },
  onRejected: error => {
    // 处理错误响应
    throw error;
  },
});
```

### 缓存管理

```typescript
import { requestStateManager } from '@/services';

// 清除特定模式的缓存
requestStateManager.clearCache('projects.*');

// 清除所有缓存
requestStateManager.clearCache();

// 获取调试信息
const debugInfo = requestStateManager.getDebugInfo();
console.log('缓存状态:', debugInfo);
```

## 🛡️ 错误处理

所有服务都包含统一的错误处理机制：

- **网络错误**: 自动重试 (最多3次)
- **HTTP错误**: 根据状态码处理
- **超时错误**: 可配置超时时间
- **认证错误**: 自动清除无效令牌
- **数据验证错误**: 详细错误信息

## 📊 性能优化

- **请求缓存**: 自动缓存 GET 请求结果
- **请求去重**: 避免重复请求
- **分页加载**: 大数据量分页处理
- **文件分块**: 大文件分块上传
- **连接池**: 复用 HTTP 连接

## 🔒 安全特性

- **输入验证**: 所有用户输入验证
- **XSS防护**: 自动转义危险内容
- **CSRF防护**: 自动添加防护头部
- **认证管理**: 安全的令牌存储和传输

## 🚀 TODO 完成状态

✅ **TODO 3: API服务层实现** - 已完成

- ✅ 实现 ApiClient 基础类，包含请求/响应拦截器
- ✅ 实现统一错误处理和重试机制
- ✅ 创建项目服务接口 (projectService.ts)
- ✅ 创建文档服务接口 (documentService.ts)
- ✅ 创建知识图谱服务接口 (knowledgeGraphService.ts)
- ✅ 创建对话服务接口 (chatService.ts)
- ✅ 实现请求缓存和状态管理
- ✅ 添加流式响应支持 (Server-Sent Events)

所有功能都已完整实现，支持现代化的 API 交互模式，为前端应用提供了强大且可靠的后端通信能力。
