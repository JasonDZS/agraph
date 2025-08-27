import type { Project as BaseProject } from '@/types/api';

export interface ProjectStatistics {
  document_count: number;
  entity_count?: number;
  relation_count?: number;
  has_vector_db: boolean;
  cache_size?: number;
  size_mb: number;
  total_size_mb?: number;
  error?: string;
}

export interface EnhancedProject extends BaseProject {
  statistics?: ProjectStatistics;
  is_current?: boolean;
  status?: 'active' | 'inactive' | 'building' | 'error';
  last_accessed?: string;
  created_at?: string;
  updated_at?: string;
}

export interface KnowledgeGraphBuildRequest {
  document_ids?: string[];
  texts?: string[];
  graph_name: string;
  graph_description: string;
  use_cache: boolean;
  save_to_vector_store: boolean;
  from_step?: string;
  enable_graph: boolean;
}

export interface ProjectCardProps {
  project: EnhancedProject;
  onSelect?: (project: EnhancedProject) => void;
  onDelete?: (projectName: string) => void;
  onSwitch?: (projectName: string) => void;
  onConfig?: (projectName: string) => void;
  onBuild?: (
    projectName: string,
    request: KnowledgeGraphBuildRequest
  ) => Promise<void>;
  isSelected?: boolean;
  isLoading?: boolean;
  showActions?: boolean;
}

export interface ProjectCreateModalProps {
  visible: boolean;
  onCancel: () => void;
  onSuccess: (project: EnhancedProject) => void;
  confirmLoading?: boolean;
}

export interface ProjectListProps {
  projects: EnhancedProject[];
  loading?: boolean;
  onRefresh?: () => void;
  onCreateProject?: () => void;
  onProjectSelect?: (project: EnhancedProject) => void;
  onProjectDelete?: (projectName: string) => void;
  onProjectSwitch?: (projectName: string) => void;
}

export interface ProjectFilterOptions {
  search: string;
  sortBy: 'name' | 'created_at' | 'updated_at' | 'document_count' | 'size_mb';
  sortOrder: 'asc' | 'desc';
  status: 'all' | 'active' | 'inactive';
  hasDocuments: boolean | null;
}

export interface ProjectOperationResult {
  success: boolean;
  message: string;
  project?: EnhancedProject;
  error?: string;
}

export interface ProjectFormData {
  name: string;
  description?: string;
}

export interface ProjectDeleteConfirmation {
  projectName: string;
  confirm: boolean;
  understoodConsequences: boolean;
}
