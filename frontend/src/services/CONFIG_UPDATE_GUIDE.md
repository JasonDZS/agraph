# 前端配置服务更新说明

## 概述

前端配置服务已更新以匹配后端的最新接口状态，支持完整的统一Settings管理、Builder配置、本地备份文件操作和Settings恢复功能。

## 主要更新内容

### 1. 类型定义更新 (`@/types/config.ts`)

#### 新增Builder配置接口
```typescript
export interface BuilderConfig {
  enable_cache: boolean;
  cache_dir: string;
  cache_ttl: number;
  auto_cleanup: boolean;
  chunk_size: number;
  chunk_overlap: number;
  entity_confidence_threshold: number;
  entity_types: string[];
  relation_confidence_threshold: number;
  relation_types: string[];
  cluster_algorithm: string;
  min_cluster_size: number;
  enable_user_interaction: boolean;
  auto_save_edits: boolean;
}
```

#### 增强的ProjectSettings
```typescript
export interface ProjectSettings {
  workdir: string;
  current_project?: string;
  max_current: number;
  openai: OpenAIConfig;
  llm: LLMConfig;
  embedding: EmbeddingConfig;
  graph: GraphConfig;
  text: TextConfig;
  rag: RAGConfig;
  builder: BuilderConfig;  // 新增builder配置
}
```

#### 完整的ConfigUpdateRequest
现在支持所有builder配置字段：
- `builder_enable_cache`
- `builder_cache_dir`
- `builder_cache_ttl`
- `builder_auto_cleanup`
- `builder_chunk_size`
- `builder_chunk_overlap`
- `builder_entity_confidence_threshold`
- `builder_relation_confidence_threshold`
- `builder_cluster_algorithm`
- `builder_min_cluster_size`
- `builder_enable_user_interaction`
- `builder_auto_save_edits`

#### 新增备份和恢复相关接口
```typescript
export interface BackupStatus {
  project_name: string;
  backup_file_path: string;
  backup_exists: boolean;
  has_complete_settings?: boolean;
  settings_version?: string;
  created_at?: string;
  updated_at?: string;
  file_size_kb?: number;
  // ... 更多状态信息
}
```

### 2. 配置服务更新 (`configService.ts`)

#### 新增核心方法
```typescript
// 获取项目备份状态
async getProjectBackupStatus(projectName: string)

// 从备份文件加载配置
async loadProjectConfigFromBackup(projectName: string)

// 获取全局配置
async getGlobalConfig()

// 更新全局配置
async updateGlobalConfig(updates: ConfigUpdateRequest)

// 保存配置到文件
async saveProjectConfigToFile(projectName: string, filePath?: string)
async saveGlobalConfigToFile(filePath?: string)

// 从文件加载配置
async loadProjectConfigFromFile(projectName: string, filePath?: string)
async loadGlobalConfigFromFile(filePath?: string)

// 获取配置文件路径
async getConfigFilePath(projectName?: string)
```

#### Builder配置专用方法
```typescript
// 仅更新builder配置
async updateBuilderConfig(projectName: string, builderConfig: Partial<BuilderConfig>)

// 配置验证
validateConfigUpdate(updates: ConfigUpdateRequest): { valid: boolean; errors: string[] }
```

### 3. 项目服务更新 (`projectService.ts`)

#### 新增恢复方法
```typescript
// 从备份恢复项目设置
async recoverProjectSettings(projectName: string)
```

### 4. 高级配置管理器 (`configManager.ts`)

新增的高级配置管理器提供以下功能：

#### 获取完整配置状态
```typescript
async getProjectConfigWithStatus(projectName: string)
// 返回配置 + 备份状态的完整信息
```

#### 安全配置更新
```typescript
async updateProjectConfigSafe(
  projectName: string,
  updates: ConfigUpdateRequest,
  options: { validateFirst?: boolean; createBackup?: boolean } = {}
)
// 支持预验证和自动备份
```

#### 安全配置恢复
```typescript
async recoverProjectConfigSafe(projectName: string)
// 验证备份完整性后再恢复
```

#### 项目配置克隆
```typescript
async cloneProjectConfig(
  sourceProject: string,
  targetProject: string,
  options: {
    overrideProjectName?: boolean;
    excludeSections?: string[];
    createBackup?: boolean;
  } = {}
)
// 支持选择性克隆和备份
```

#### 配置比较
```typescript
async compareProjectConfigs(projectA: string, projectB: string)
// 详细比较两个项目的配置差异
```

#### 配置健康检查
```typescript
async getConfigHealthStatus(projectName: string)
// 全面的配置健康状态评估
```

## 使用示例

### 基础配置操作
```typescript
import { configService } from '@/services';

// 获取项目配置
const config = await configService.getProjectConfig('my-project');

// 更新Builder配置
await configService.updateBuilderConfig('my-project', {
  enable_cache: true,
  cache_ttl: 86400,
  entity_confidence_threshold: 0.8
});

// 检查备份状态
const backupStatus = await configService.getProjectBackupStatus('my-project');
```

### 高级配置管理
```typescript
import { configManager } from '@/services';

// 获取配置状态
const status = await configManager.getProjectConfigWithStatus('my-project');
if (status.hasBackup) {
  console.log('Backup available:', status.backupStatus);
}

// 安全更新配置
await configManager.updateProjectConfigSafe('my-project', updates, {
  validateFirst: true,
  createBackup: true
});

// 克隆配置
await configManager.cloneProjectConfig('source-project', 'target-project', {
  excludeSections: ['openai'], // 排除敏感信息
  createBackup: true
});

// 配置健康检查
const health = await configManager.getConfigHealthStatus('my-project');
console.log('Health score:', health.healthScore, 'Issues:', health.issues);
```

### 错误处理和恢复
```typescript
import { configManager, configService } from '@/services';

try {
  // 尝试获取配置
  const config = await configService.getProjectConfig('my-project');
} catch (error) {
  console.log('Config load failed, attempting recovery...');

  // 尝试从备份恢复
  try {
    await configManager.recoverProjectConfigSafe('my-project');
    console.log('Recovery successful!');
  } catch (recoveryError) {
    console.error('Recovery also failed:', recoveryError);
  }
}
```

## API端点对应关系

| 前端方法 | 后端端点 | 功能 |
|---------|----------|------|
| `getProjectConfig` | `GET /config?project_name={name}` | 获取项目配置 |
| `updateProjectConfig` | `POST /config?project_name={name}` | 更新项目配置 |
| `getProjectBackupStatus` | `GET /config/projects/{name}/backup-status` | 获取备份状态 |
| `loadProjectConfigFromBackup` | `POST /config/projects/{name}/load-from-backup` | 从备份恢复 |
| `recoverProjectSettings` | `POST /projects/{name}/recover-settings` | 项目设置恢复 |
| `saveProjectConfigToFile` | `POST /config/save?project_name={name}` | 保存到文件 |
| `loadProjectConfigFromFile` | `POST /config/load?project_name={name}` | 从文件加载 |

## 向后兼容性

- 保持所有现有方法的接口不变
- 新增字段都是可选的
- 支持legacy字段映射
- 验证和错误处理增强但不影响现有代码

## 配置验证

新增的配置验证包括：
- LLM temperature 范围检查 (0-2)
- Token限制正数验证
- 置信度阈值范围检查 (0-1)
- 缓存TTL正数验证
- Chunk大小合理性检查

这次更新确保前端配置服务完全匹配后端的统一Settings管理系统，支持完整的配置操作和恢复功能。
