import { useState } from 'react';
import { Dropdown, Button, Space, Typography, Spin, Badge } from 'antd';
import {
  DownOutlined,
  FolderOutlined,
  CheckCircleOutlined,
  PlusOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import { useProject } from '@/modules/Projects/hooks/useProject';
import { useProjectStore } from '@/store';

const { Text } = Typography;

interface ProjectDropdownProps {
  onCreateProject?: () => void;
}

const ProjectDropdown = ({ onCreateProject }: ProjectDropdownProps) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const { projects, currentProject, switchProject, loading } = useProject();
  const { projectsLoading } = useProjectStore();

  const handleProjectSwitch = async (projectName: string) => {
    await switchProject(projectName);
    setDropdownOpen(false);
  };

  const handleCreateProject = () => {
    onCreateProject?.();
    setDropdownOpen(false);
  };

  const menuItems: MenuProps['items'] = [
    // Current projects
    ...projects.map(project => ({
      key: project.name,
      label: (
        <Space style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space>
            <FolderOutlined />
            <span>{project.name}</span>
            {project.is_current && (
              <CheckCircleOutlined style={{ color: '#52c41a' }} />
            )}
          </Space>
          {project.statistics && (
            <Badge
              count={project.statistics.document_count}
              size="small"
              color="#1890ff"
              showZero={false}
            />
          )}
        </Space>
      ),
      onClick: () => handleProjectSwitch(project.name),
      disabled: project.is_current,
    })),
    // Default workspace option
    {
      key: 'default',
      label: (
        <Space>
          <FolderOutlined />
          <span>Default Workspace</span>
          {!currentProject && (
            <CheckCircleOutlined style={{ color: '#52c41a' }} />
          )}
        </Space>
      ),
      onClick: () => handleProjectSwitch(''),
      disabled: !currentProject,
    },
    // Divider
    {
      type: 'divider',
    },
    // Create new project
    {
      key: 'create',
      label: (
        <Space>
          <PlusOutlined />
          <span>Create New Project</span>
        </Space>
      ),
      onClick: handleCreateProject,
    },
  ];

  const currentProjectName = currentProject?.name || 'Default Workspace';
  const isLoading = loading || projectsLoading;

  return (
    <Dropdown
      menu={{ items: menuItems }}
      trigger={['click']}
      open={dropdownOpen}
      onOpenChange={setDropdownOpen}
      placement="bottomLeft"
      overlayStyle={{ minWidth: 280 }}
    >
      <Button
        type="text"
        style={{
          height: 32,
          padding: '0 8px',
          border: '1px solid #607D8B', // 蓝灰色边框
          borderRadius: 6,
          backgroundColor: 'transparent',
        }}
      >
        <Space>
          {isLoading ? (
            <Spin size="small" />
          ) : (
            <FolderOutlined style={{ color: '#FFA000' }} /> // 暖阳橙图标
          )}
          <Text
            style={{
              maxWidth: 120,
              overflow: 'hidden',
              textOverflow: 'ellipsis',
              whiteSpace: 'nowrap',
              color: '#FFFFFF', // 白色文字
            }}
            title={currentProjectName}
          >
            {currentProjectName}
          </Text>
          <DownOutlined style={{ fontSize: '12px', color: '#D9DCE0' }} />
        </Space>
      </Button>
    </Dropdown>
  );
};

export default ProjectDropdown;
