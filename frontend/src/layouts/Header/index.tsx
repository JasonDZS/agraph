import { Layout, Button, Dropdown, Avatar, Space, theme } from 'antd';
import {
  MenuOutlined,
  UserOutlined,
  SettingOutlined,
  LogoutOutlined,
  SunOutlined,
  MoonOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import ProjectDropdown from '@/components/Projects/ProjectDropdown';

const { Header: AntHeader } = Layout;

interface HeaderProps {
  collapsed: boolean;
  onToggleCollapsed: () => void;
  isDarkMode: boolean;
  onToggleTheme: () => void;
  onCreateProject?: () => void;
}

const Header = ({
  onToggleCollapsed,
  isDarkMode,
  onToggleTheme,
  onCreateProject,
}: HeaderProps) => {
  const { token } = theme.useToken();

  const userMenuItems: MenuProps['items'] = [
    {
      key: 'profile',
      label: 'Profile',
      icon: <UserOutlined />,
    },
    {
      key: 'settings',
      label: 'Settings',
      icon: <SettingOutlined />,
    },
    {
      type: 'divider',
    },
    {
      key: 'logout',
      label: 'Logout',
      icon: <LogoutOutlined />,
      danger: true,
    },
  ];

  return (
    <AntHeader
      style={{
        padding: '0 16px',
        background: '#2D3E50', // 深蓝灰色顶部导航栏背景
        borderBottom: `1px solid #607D8B`, // 蓝灰色边框
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        height: 64, // Fixed height
        minHeight: 64, // Minimum height
        maxHeight: 64, // Maximum height
        flexShrink: 0, // Prevent shrinking
        position: 'sticky', // Sticky positioning
        top: 0, // Stick to top
        zIndex: 1000, // High z-index to stay above content
        width: '100%', // Full width
      }}
    >
      <Space>
        <Button
          type="text"
          icon={<MenuOutlined />}
          onClick={onToggleCollapsed}
          style={{
            fontSize: '16px',
            width: 40,
            height: 40,
            color: '#D9DCE0', // 浅灰色图标
          }}
        />
        <img
          src="/logo.svg"
          alt="AGraph"
          style={{
            height: '32px',
            width: 'auto',
          }}
        />
        <ProjectDropdown onCreateProject={onCreateProject} />
      </Space>

      <Space>
        <Button
          type="text"
          icon={isDarkMode ? <SunOutlined /> : <MoonOutlined />}
          onClick={onToggleTheme}
          style={{
            fontSize: '16px',
            color: '#D9DCE0', // 浅灰色图标
          }}
        />

        <Dropdown menu={{ items: userMenuItems }} placement="bottomRight">
          <Button type="text" style={{
            height: 40,
            color: '#D9DCE0', // 浅灰色文字
          }}>
            <Space>
              <Avatar
                size="small"
                icon={<UserOutlined />}
                style={{
                  backgroundColor: '#FFA000', // 暖阳橙头像背景
                  color: '#2D3E50', // 深色图标
                }}
              />
              <span style={{ color: '#D9DCE0' }}>User</span>
            </Space>
          </Button>
        </Dropdown>
      </Space>
    </AntHeader>
  );
};

export default Header;
