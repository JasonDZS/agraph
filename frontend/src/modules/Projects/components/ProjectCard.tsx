import React, { useState } from 'react';
import {
  Card,
  Button,
  Dropdown,
  Tag,
  Statistic,
  Progress,
  Space,
  Typography,
  Tooltip,
} from 'antd';
import {
  MoreOutlined,
  FolderOutlined,
  FileTextOutlined,
  NodeIndexOutlined,
  DeleteOutlined,
  EditOutlined,
  PlayCircleOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import type { ProjectCardProps } from '../types/project';

const { Text, Title } = Typography;

export const ProjectCard: React.FC<ProjectCardProps> = ({
  project,
  onSelect,
  onDelete,
  onSwitch,
  onConfig,
  isSelected = false,
  isLoading = false,
  showActions = true,
}) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);

  const handleCardClick = () => {
    if (onSelect && !isLoading) {
      onSelect(project);
    }
  };

  const handleSwitchProject = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (onSwitch && !project.is_current) {
      onSwitch(project.name);
    }
  };

  const handleDeleteProject = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (onDelete) {
      onDelete(project.name);
    }
  };

  const handleConfig = (e?: React.MouseEvent) => {
    e?.stopPropagation();
    if (onConfig) {
      onConfig(project.name);
    }
  };

  // Status badge component
  const StatusBadge = () => {
    const statusConfig = {
      active: { color: '#52c41a', text: 'Active' },
      inactive: { color: '#d9d9d9', text: 'Inactive' },
      building: { color: '#1890ff', text: 'Building' },
      error: { color: '#ff4d4f', text: 'Error' },
    };

    const config = statusConfig[project.status || 'inactive'];

    return (
      <Tag color={config.color} style={{ margin: 0 }}>
        {config.text}
      </Tag>
    );
  };

  // Format file size
  const formatSize = (sizeInMB: number): string => {
    if (sizeInMB < 1) return `${(sizeInMB * 1024).toFixed(0)} KB`;
    if (sizeInMB < 1024) return `${sizeInMB.toFixed(1)} MB`;
    return `${(sizeInMB / 1024).toFixed(1)} GB`;
  };

  // Format date
  const formatDate = (dateString?: string): string => {
    if (!dateString) return 'Unknown';
    return new Date(dateString).toLocaleDateString();
  };

  // Calculate project health score based on available data
  const calculateHealthScore = (): number => {
    const stats = project.statistics;
    if (!stats) return 0;

    let score = 0;
    let maxPossibleScore = 0;

    // Document count (essential for any project)
    maxPossibleScore += 40;
    if (stats.document_count > 0) {
      score += 40;
    }

    // Vector DB (indicates processed data)
    maxPossibleScore += 30;
    if (stats.has_vector_db) {
      score += 30;
    }

    // Knowledge Graph entities (if available)
    if (stats.entity_count !== undefined) {
      maxPossibleScore += 15;
      if (stats.entity_count > 0) {
        score += 15;
      }
    }

    // Knowledge Graph relations (if available)
    if (stats.relation_count !== undefined) {
      maxPossibleScore += 15;
      if (stats.relation_count > 0) {
        score += 15;
      }
    }

    // Calculate percentage based on available features
    return maxPossibleScore > 0
      ? Math.round((score / maxPossibleScore) * 100)
      : 0;
  };

  const healthScore = calculateHealthScore();

  // Dropdown menu items
  const menuItems: MenuProps['items'] = [
    {
      key: 'switch',
      label: 'Switch to Project',
      icon: <PlayCircleOutlined />,
      disabled: project.is_current,
      onClick: () => handleSwitchProject(),
    },
    {
      key: 'config',
      label: 'Configuration',
      icon: <SettingOutlined />,
      onClick: () => handleConfig(),
    },
    {
      key: 'edit',
      label: 'Edit Project',
      icon: <EditOutlined />,
      disabled: true, // Will be implemented later
    },
    {
      type: 'divider',
    },
    {
      key: 'delete',
      label: 'Delete Project',
      icon: <DeleteOutlined />,
      danger: true,
      onClick: () => handleDeleteProject(),
    },
  ];

  return (
    <Card
      className={`project-card ${isSelected ? 'selected' : ''}`}
      hoverable={!isLoading}
      loading={isLoading}
      onClick={handleCardClick}
      styles={{
        body: { padding: '16px' },
      }}
      style={{
        border: isSelected ? '2px solid #1890ff' : undefined,
        cursor: isLoading ? 'default' : 'pointer',
      }}
      actions={
        showActions
          ? [
              <Button
                key="switch"
                type={project.is_current ? 'default' : 'primary'}
                size="small"
                icon={<PlayCircleOutlined />}
                disabled={project.is_current}
                onClick={handleSwitchProject}
              >
                {project.is_current ? 'Current' : 'Switch'}
              </Button>,
              <Dropdown
                key="more"
                menu={{ items: menuItems }}
                trigger={['click']}
                open={dropdownOpen}
                onOpenChange={setDropdownOpen}
              >
                <Button
                  type="text"
                  icon={<MoreOutlined />}
                  onClick={e => {
                    e.stopPropagation();
                    setDropdownOpen(!dropdownOpen);
                  }}
                />
              </Dropdown>,
            ]
          : undefined
      }
    >
      {/* Header */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'flex-start',
          marginBottom: 12,
        }}
      >
        <div style={{ flex: 1 }}>
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 8,
              marginBottom: 4,
            }}
          >
            <FolderOutlined style={{ color: '#1890ff' }} />
            <Title level={4} style={{ margin: 0 }}>
              {project.name}
            </Title>
            <StatusBadge />
          </div>
          {project.description && (
            <Text type="secondary" style={{ fontSize: '12px' }}>
              {project.description}
            </Text>
          )}
        </div>
      </div>

      {/* Statistics Grid */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 12,
          marginBottom: 16,
        }}
      >
        <Statistic
          title="Documents"
          value={project.statistics?.document_count || 0}
          prefix={<FileTextOutlined />}
          valueStyle={{ fontSize: '14px' }}
        />
        <Statistic
          title="Size"
          value={formatSize(project.statistics?.size_mb || 0)}
          valueStyle={{ fontSize: '14px' }}
        />
        <Statistic
          title="Entities"
          value={project.statistics?.entity_count || 0}
          prefix={<NodeIndexOutlined />}
          valueStyle={{ fontSize: '14px' }}
        />
        <Statistic
          title="Relations"
          value={project.statistics?.relation_count || 0}
          prefix={<DatabaseOutlined />}
          valueStyle={{ fontSize: '14px' }}
        />
      </div>

      {/* Vector DB Status */}
      <div style={{ marginBottom: 12 }}>
        <Space>
          <Text style={{ fontSize: '12px' }}>Vector DB:</Text>
          <Tag
            color={project.statistics?.has_vector_db ? 'green' : 'orange'}
            style={{ fontSize: '10px' }}
          >
            {project.statistics?.has_vector_db ? 'Enabled' : 'Not Ready'}
          </Tag>
        </Space>
      </div>

      {/* Error Display */}
      {project.statistics?.error && (
        <div style={{ marginBottom: 12 }}>
          <Tag color="red" style={{ fontSize: '10px' }}>
            Error: {project.statistics.error}
          </Tag>
        </div>
      )}

      {/* Health Score */}
      <div style={{ marginBottom: 12 }}>
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: 4,
          }}
        >
          <Text strong style={{ fontSize: '12px' }}>
            Project Health
          </Text>
          <Text style={{ fontSize: '12px' }}>{healthScore}%</Text>
        </div>
        <Progress
          percent={healthScore}
          size="small"
          strokeColor={{
            '0%': '#ff4d4f',
            '50%': '#faad14',
            '100%': '#52c41a',
          }}
          showInfo={false}
        />
      </div>

      {/* Footer Info */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        <Space size={12}>
          {project.statistics?.has_vector_db && (
            <Tooltip title="Has Vector Database">
              <DatabaseOutlined style={{ color: '#52c41a' }} />
            </Tooltip>
          )}
        </Space>

        <Tooltip title={`Last updated: ${formatDate(project.updated_at)}`}>
          <Text
            type="secondary"
            style={{
              fontSize: '11px',
              display: 'flex',
              alignItems: 'center',
              gap: 4,
            }}
          >
            <ClockCircleOutlined />
            {formatDate(project.updated_at)}
          </Text>
        </Tooltip>
      </div>
    </Card>
  );
};

export default ProjectCard;
