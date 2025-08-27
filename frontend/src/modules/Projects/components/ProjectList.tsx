import React, { useState, useMemo } from 'react';
import {
  Card,
  Button,
  Input,
  Select,
  Row,
  Col,
  Space,
  Typography,
  Empty,
  Spin,
  Radio,
  Statistic,
  Divider,
  Badge,
  Tooltip,
} from 'antd';
import {
  PlusOutlined,
  SearchOutlined,
  ReloadOutlined,
  FolderOutlined,
  FileTextOutlined,
  NodeIndexOutlined,
  SortAscendingOutlined,
  SortDescendingOutlined,
} from '@ant-design/icons';
import ProjectCard from './ProjectCard';
import ProjectCreateModal from './ProjectCreateModal';
import ProjectDeleteModal from './ProjectDeleteModal';
import ProjectConfigModal from './ProjectConfigModal';
import { useProject } from '../hooks/useProject';
import type { EnhancedProject, ProjectFilterOptions } from '../types/project';

const { Title, Text } = Typography;
const { Option } = Select;

const ProjectList: React.FC = () => {
  const {
    projects,
    currentProject,
    loading,
    error,
    loadProjects,
    createProject,
    switchProject,
    deleteProject,
    buildKnowledgeGraph,
  } = useProject();

  // Modal states
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [showConfigModal, setShowConfigModal] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<string>('');
  const [projectToConfig, setProjectToConfig] = useState<string>('');
  const [operationLoading, setOperationLoading] = useState(false);

  // Filter and sort state
  const [filters, setFilters] = useState<ProjectFilterOptions>({
    search: '',
    sortBy: 'updated_at',
    sortOrder: 'desc',
    status: 'all',
    hasDocuments: null,
  });

  // Filter and sort projects
  const filteredProjects = useMemo(() => {
    let filtered = [...projects];

    // Apply search filter
    if (filters.search.trim()) {
      const searchTerm = filters.search.toLowerCase();
      filtered = filtered.filter(
        project =>
          project.name.toLowerCase().includes(searchTerm) ||
          project.description?.toLowerCase().includes(searchTerm)
      );
    }

    // Apply status filter
    if (filters.status !== 'all') {
      filtered = filtered.filter(project => project.status === filters.status);
    }

    // Apply document filter
    if (filters.hasDocuments !== null) {
      filtered = filtered.filter(project =>
        filters.hasDocuments
          ? (project.document_count || 0) > 0
          : (project.document_count || 0) === 0
      );
    }

    // Apply sorting
    filtered.sort((a, b) => {
      let aValue: any, bValue: any;

      switch (filters.sortBy) {
        case 'name':
          aValue = a.name.toLowerCase();
          bValue = b.name.toLowerCase();
          break;
        case 'created_at':
          aValue = new Date(a.created_at || 0).getTime();
          bValue = new Date(b.created_at || 0).getTime();
          break;
        case 'updated_at':
          aValue = new Date(a.updated_at || 0).getTime();
          bValue = new Date(b.updated_at || 0).getTime();
          break;
        case 'document_count':
          aValue = a.document_count || 0;
          bValue = b.document_count || 0;
          break;
        case 'size_mb':
          aValue = a.statistics?.size_mb || 0;
          bValue = b.statistics?.size_mb || 0;
          break;
        default:
          return 0;
      }

      if (aValue < bValue) return filters.sortOrder === 'asc' ? -1 : 1;
      if (aValue > bValue) return filters.sortOrder === 'asc' ? 1 : -1;
      return 0;
    });

    return filtered;
  }, [projects, filters]);

  // Calculate statistics
  const statistics = useMemo(() => {
    return {
      total: projects.length,
      active: projects.filter(p => p.status === 'active').length,
      withDocuments: projects.filter(p => (p.document_count || 0) > 0).length,
      totalDocuments: projects.reduce(
        (sum, p) => sum + (p.document_count || 0),
        0
      ),
      totalEntities: projects.reduce(
        (sum, p) => sum + (p.entity_count || 0),
        0
      ),
      totalRelations: projects.reduce(
        (sum, p) => sum + (p.relation_count || 0),
        0
      ),
    };
  }, [projects]);

  // Handle project creation
  const handleCreateProject = async (project: EnhancedProject) => {
    setOperationLoading(true);
    try {
      const result = await createProject({
        name: project.name,
        description: project.description,
      });

      if (result.success) {
        setShowCreateModal(false);
        // loadProjects is automatically called by the hook
      }
    } finally {
      setOperationLoading(false);
    }
  };

  // Handle project switching
  const handleSwitchProject = async (projectName: string) => {
    setOperationLoading(true);
    try {
      await switchProject(projectName);
      // loadProjects is automatically called by the hook
    } finally {
      setOperationLoading(false);
    }
  };

  // Handle project deletion
  const handleDeleteProject = (projectName: string) => {
    setProjectToDelete(projectName);
    setShowDeleteModal(true);
  };

  const handleConfirmDelete = async (confirmation: any) => {
    setOperationLoading(true);
    try {
      const result = await deleteProject(confirmation);
      if (result.success) {
        setShowDeleteModal(false);
        setProjectToDelete('');
        // loadProjects is automatically called by the hook
      }
    } finally {
      setOperationLoading(false);
    }
  };

  // Handle project configuration
  const handleConfigProject = (projectName: string) => {
    setProjectToConfig(projectName);
    setShowConfigModal(true);
  };

  // Handle project knowledge graph build
  const handleBuildKnowledgeGraph = async (
    projectName: string,
    request: any
  ) => {
    setOperationLoading(true);
    try {
      const result = await buildKnowledgeGraph(projectName, request);
      if (result.success) {
        // loadProjects will be called automatically after build
      }
    } finally {
      setOperationLoading(false);
    }
  };

  return (
    <div style={{ padding: '24px' }}>
      {/* Header */}
      <div style={{ marginBottom: '24px' }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            marginBottom: '16px',
          }}
        >
          <div>
            <Title level={2} style={{ margin: 0 }}>
              <FolderOutlined style={{ marginRight: '8px' }} />
              Projects
            </Title>
            <Text type="secondary">
              Manage your knowledge graph projects and workspaces
            </Text>
          </div>
          <Space>
            <Button
              icon={<ReloadOutlined />}
              onClick={() => loadProjects(true)}
              loading={loading}
            >
              Refresh
            </Button>
            <Button
              type="primary"
              icon={<PlusOutlined />}
              onClick={() => setShowCreateModal(true)}
            >
              New Project
            </Button>
          </Space>
        </div>

        {/* Statistics Cards */}
        <Row gutter={16} style={{ marginBottom: '24px' }}>
          <Col xs={24} sm={12} md={6}>
            <Card size="small">
              <Statistic
                title="Total Projects"
                value={statistics.total}
                prefix={<FolderOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card size="small">
              <Statistic
                title="Active Project"
                value={statistics.active}
                prefix={<Badge status="processing" />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card size="small">
              <Statistic
                title="Total Documents"
                value={statistics.totalDocuments}
                prefix={<FileTextOutlined />}
              />
            </Card>
          </Col>
          <Col xs={24} sm={12} md={6}>
            <Card size="small">
              <Statistic
                title="Total Entities"
                value={statistics.totalEntities}
                prefix={<NodeIndexOutlined />}
              />
            </Card>
          </Col>
        </Row>
      </div>

      {/* Filters and Search */}
      <Card style={{ marginBottom: '24px' }}>
        <Row gutter={[16, 16]} align="middle">
          <Col xs={24} sm={12} md={8}>
            <Input
              placeholder="Search projects..."
              prefix={<SearchOutlined />}
              value={filters.search}
              onChange={e => setFilters({ ...filters, search: e.target.value })}
              allowClear
            />
          </Col>

          <Col xs={12} sm={6} md={4}>
            <Select
              value={filters.sortBy}
              onChange={value => setFilters({ ...filters, sortBy: value })}
              style={{ width: '100%' }}
            >
              <Option value="updated_at">Last Updated</Option>
              <Option value="created_at">Created Date</Option>
              <Option value="name">Name</Option>
              <Option value="document_count">Documents</Option>
              <Option value="size_mb">Size</Option>
            </Select>
          </Col>

          <Col xs={12} sm={6} md={3}>
            <Tooltip
              title={`Sort ${filters.sortOrder === 'asc' ? 'Ascending' : 'Descending'}`}
            >
              <Button
                icon={
                  filters.sortOrder === 'asc' ? (
                    <SortAscendingOutlined />
                  ) : (
                    <SortDescendingOutlined />
                  )
                }
                onClick={() =>
                  setFilters({
                    ...filters,
                    sortOrder: filters.sortOrder === 'asc' ? 'desc' : 'asc',
                  })
                }
              />
            </Tooltip>
          </Col>

          <Col xs={24} sm={12} md={5}>
            <Radio.Group
              value={filters.status}
              onChange={e => setFilters({ ...filters, status: e.target.value })}
              size="small"
            >
              <Radio.Button value="all">All</Radio.Button>
              <Radio.Button value="active">Active</Radio.Button>
              <Radio.Button value="inactive">Inactive</Radio.Button>
            </Radio.Group>
          </Col>

          <Col xs={24} sm={12} md={4}>
            <Select
              placeholder="Documents"
              value={filters.hasDocuments}
              onChange={value =>
                setFilters({ ...filters, hasDocuments: value })
              }
              style={{ width: '100%' }}
              allowClear
            >
              <Option value={true}>With Documents</Option>
              <Option value={false}>Empty</Option>
            </Select>
          </Col>
        </Row>
      </Card>

      {/* Project Grid */}
      {error && (
        <Card style={{ marginBottom: '24px' }}>
          <div style={{ textAlign: 'center', color: '#ff4d4f' }}>
            <Title level={4}>Error Loading Projects</Title>
            <Text>{error}</Text>
            <br />
            <Button
              type="primary"
              onClick={() => loadProjects(true)}
              style={{ marginTop: '12px' }}
            >
              Retry
            </Button>
          </div>
        </Card>
      )}

      <Spin spinning={loading || operationLoading} size="large">
        {filteredProjects.length === 0 ? (
          <Card>
            <Empty
              image={Empty.PRESENTED_IMAGE_SIMPLE}
              description={
                projects.length === 0 ? (
                  <div>
                    <Text>No projects found</Text>
                    <br />
                    <Text type="secondary">
                      Create your first project to get started
                    </Text>
                  </div>
                ) : (
                  <div>
                    <Text>No projects match your filters</Text>
                    <br />
                    <Text type="secondary">
                      Try adjusting your search criteria
                    </Text>
                  </div>
                )
              }
            >
              {projects.length === 0 && (
                <Button
                  type="primary"
                  icon={<PlusOutlined />}
                  onClick={() => setShowCreateModal(true)}
                >
                  Create Project
                </Button>
              )}
            </Empty>
          </Card>
        ) : (
          <Row gutter={[16, 16]}>
            {filteredProjects.map(project => (
              <Col xs={24} sm={12} lg={8} xl={6} key={project.name}>
                <ProjectCard
                  project={project}
                  onSwitch={handleSwitchProject}
                  onDelete={handleDeleteProject}
                  onConfig={handleConfigProject}
                  onBuild={handleBuildKnowledgeGraph}
                  isLoading={operationLoading}
                  showActions={true}
                />
              </Col>
            ))}
          </Row>
        )}
      </Spin>

      {/* Current Project Info */}
      {currentProject && (
        <Card
          style={{
            marginTop: '24px',
            backgroundColor: '#f6ffed',
            borderColor: '#b7eb8f',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Badge status="processing" />
            <Text strong>Current Project: {currentProject.name}</Text>
            {currentProject.description && (
              <>
                <Divider type="vertical" />
                <Text type="secondary">{currentProject.description}</Text>
              </>
            )}
          </div>
        </Card>
      )}

      {/* Modals */}
      <ProjectCreateModal
        visible={showCreateModal}
        onCancel={() => setShowCreateModal(false)}
        onSuccess={handleCreateProject}
        confirmLoading={operationLoading}
      />

      <ProjectDeleteModal
        visible={showDeleteModal}
        projectName={projectToDelete}
        onConfirm={handleConfirmDelete}
        onCancel={() => {
          setShowDeleteModal(false);
          setProjectToDelete('');
        }}
        confirmLoading={operationLoading}
        projectStats={{
          documentCount:
            projects.find(p => p.name === projectToDelete)?.document_count || 0,
          entityCount:
            projects.find(p => p.name === projectToDelete)?.entity_count || 0,
          relationCount:
            projects.find(p => p.name === projectToDelete)?.relation_count || 0,
          sizeInMB:
            projects.find(p => p.name === projectToDelete)?.statistics
              ?.size_mb || 0,
        }}
      />

      <ProjectConfigModal
        visible={showConfigModal}
        projectName={projectToConfig}
        onCancel={() => {
          setShowConfigModal(false);
          setProjectToConfig('');
        }}
        onSuccess={() => {
          // Configuration saved successfully
          // Could reload projects if needed
        }}
      />
    </div>
  );
};

export default ProjectList;
