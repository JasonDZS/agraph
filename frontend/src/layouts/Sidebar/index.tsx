import { Layout, Menu, theme } from 'antd';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  DashboardOutlined,
  FolderOutlined,
  FileTextOutlined,
  NodeIndexOutlined,
  MessageOutlined,
  SearchOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';

const { Sider } = Layout;

interface SidebarProps {
  collapsed: boolean;
}

const Sidebar = ({ collapsed }: SidebarProps) => {
  const { token } = theme.useToken();
  const navigate = useNavigate();
  const location = useLocation();

  const menuItems: MenuProps['items'] = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: 'Dashboard',
    },
    {
      key: '/projects',
      icon: <FolderOutlined />,
      label: 'Projects',
    },
    {
      key: '/documents',
      icon: <FileTextOutlined />,
      label: 'Documents',
    },
    {
      key: '/knowledge-graph',
      icon: <NodeIndexOutlined />,
      label: 'Knowledge Graph',
    },
    {
      key: '/chat',
      icon: <MessageOutlined />,
      label: 'Chat',
    },
    {
      key: '/search',
      icon: <SearchOutlined />,
      label: 'Search',
    },
  ];

  const handleMenuClick: MenuProps['onClick'] = ({ key }) => {
    navigate(key);
  };

  return (
    <Sider
      trigger={null}
      collapsible
      collapsed={collapsed}
      width={240}
      collapsedWidth={80}
      style={{
        background: '#34495E', // 深蓝色侧边栏背景
        borderRight: `1px solid #607D8B`, // 蓝灰色边框
        height: '100vh', // Full viewport height
        position: 'fixed', // Fixed positioning
        left: 0, // Fixed to left side
        top: 0, // Start from top
        zIndex: 999, // High z-index but below header
        overflow: 'hidden', // Prevent overflow
      }}
    >
      <div
        style={{
          height: '64px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          borderBottom: `1px solid #607D8B`, // 蓝灰色分割线
          padding: '0 16px',
        }}
      >
        {collapsed ? (
          <div
            style={{
              fontSize: '16px',
              fontWeight: 'bold',
              color: '#FFA000',
            }}
          >
            AG
          </div>
        ) : (
          <img
            src="/logo.svg"
            alt="AGraph"
            style={{
              height: '40px',
              width: 'auto',
            }}
          />
        )}
      </div>

      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        items={menuItems}
        onClick={handleMenuClick}
        style={{
          height: 'calc(100vh - 64px)',
          borderRight: 0,
          background: 'transparent',
          color: '#D9DCE0', // 浅灰色文字
        }}
        theme="dark"
      />
    </Sider>
  );
};

export default Sidebar;
