import { useState, useEffect } from 'react';
import { Layout, theme, ConfigProvider } from 'antd';
import { Outlet } from 'react-router-dom';
import Header from '../Header';
import Sidebar from '../Sidebar';
import { useProject } from '@/modules/Projects/hooks/useProject';
import ProjectCreateModal from '@/modules/Projects/components/ProjectCreateModal';
import type { EnhancedProject } from '@/modules/Projects/types/project';

const { Content } = Layout;

const MainLayout = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [createModalVisible, setCreateModalVisible] = useState(false);
  const [createLoading, setCreateLoading] = useState(false);
  const { token } = theme.useToken();
  const { loadProjects, createProject } = useProject();

  // Initialize projects on layout mount
  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  const toggleCollapsed = () => {
    setCollapsed(!collapsed);
  };

  const toggleTheme = () => {
    setIsDarkMode(!isDarkMode);
  };

  const handleCreateProject = () => {
    setCreateModalVisible(true);
  };

  const handleCreateSuccess = async (project: EnhancedProject) => {
    setCreateLoading(true);
    try {
      await createProject({
        name: project.name,
        description: project.description,
      });
      setCreateModalVisible(false);
    } catch (error) {
      console.error('Failed to create project:', error);
    } finally {
      setCreateLoading(false);
    }
  };

  const handleCreateCancel = () => {
    setCreateModalVisible(false);
  };

  return (
    <ConfigProvider
      theme={{
        algorithm: isDarkMode ? theme.darkAlgorithm : theme.defaultAlgorithm,
      }}
    >
      <Layout style={{ minHeight: '100vh' }}>
        <Sidebar collapsed={collapsed} />

        <Layout
          style={{
            marginLeft: collapsed ? 80 : 240, // Add left margin for fixed sidebar
            transition: 'margin-left 0.2s', // Smooth transition when collapsing
            minHeight: '100vh',
          }}
        >
          <Header
            collapsed={collapsed}
            onToggleCollapsed={toggleCollapsed}
            isDarkMode={isDarkMode}
            onToggleTheme={toggleTheme}
            onCreateProject={handleCreateProject}
          />

          <Content
            style={{
              margin: '16px',
              padding: '24px',
              background: token.colorBgContainer,
              borderRadius: token.borderRadius,
              height: 'calc(100vh - 112px)', // 64px header + 32px margins + 16px padding
              overflow: 'hidden', // 让子组件处理滚动
              flex: 1, // 确保内容区域占据可用空间
              display: 'flex', // 添加flex布局
              flexDirection: 'column', // 垂直布局
            }}
          >
            <Outlet />
          </Content>
        </Layout>

        <ProjectCreateModal
          visible={createModalVisible}
          onCancel={handleCreateCancel}
          onSuccess={handleCreateSuccess}
          confirmLoading={createLoading}
        />
      </Layout>
    </ConfigProvider>
  );
};

export default MainLayout;
