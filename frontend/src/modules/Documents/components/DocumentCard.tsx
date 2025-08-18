import React from 'react';
import {
  Card,
  Typography,
  Tag,
  Space,
  Button,
  Dropdown,
  Tooltip,
  Avatar,
} from 'antd';
import {
  FileTextOutlined,
  DownloadOutlined,
  EyeOutlined,
  DeleteOutlined,
  EditOutlined,
  MoreOutlined,
  CalendarOutlined,
  TagsOutlined,
  FileOutlined,
} from '@ant-design/icons';
import type { MenuProps } from 'antd';
import type { Document } from '../types';

const { Text, Paragraph } = Typography;

interface DocumentCardProps {
  document: Document;
  selected?: boolean;
  onSelect?: (documentId: string, selected: boolean) => void;
  onView?: (document: Document) => void;
  onEdit?: (document: Document) => void;
  onDelete?: (document: Document) => void;
  onDownload?: (document: Document) => void;
  showCheckbox?: boolean;
}

const DocumentCard: React.FC<DocumentCardProps> = ({
  document,
  selected = false,
  onSelect,
  onView,
  onEdit,
  onDelete,
  onDownload,
  showCheckbox = false,
}) => {
  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    const date = new Date(dateString);
    const now = new Date();
    const diffTime = Math.abs(now.getTime() - date.getTime());
    const diffDays = Math.ceil(diffTime / (1000 * 60 * 60 * 24));

    if (diffDays === 1) return '今天';
    if (diffDays === 2) return '昨天';
    if (diffDays <= 7) return `${diffDays} 天前`;
    return date.toLocaleDateString('zh-CN');
  };

  const getFileTypeIcon = (filename: string, contentType?: string) => {
    const ext = filename.split('.').pop()?.toLowerCase();
    const type = contentType?.toLowerCase();

    if (type?.includes('pdf') || ext === 'pdf') {
      return (
        <Avatar
          style={{ backgroundColor: '#ff4d4f' }}
          icon={<FileTextOutlined />}
        />
      );
    }
    if (type?.includes('word') || ['doc', 'docx'].includes(ext || '')) {
      return (
        <Avatar
          style={{ backgroundColor: '#1890ff' }}
          icon={<FileTextOutlined />}
        />
      );
    }
    if (type?.includes('text') || ['txt', 'md'].includes(ext || '')) {
      return (
        <Avatar
          style={{ backgroundColor: '#52c41a' }}
          icon={<FileOutlined />}
        />
      );
    }
    return (
      <Avatar
        style={{ backgroundColor: '#722ed1' }}
        icon={<FileTextOutlined />}
      />
    );
  };

  const getContentPreview = (
    content: string,
    maxLength: number = 100
  ): string => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  const menuItems: MenuProps['items'] = [
    {
      key: 'view',
      label: '预览',
      icon: <EyeOutlined />,
      onClick: () => onView?.(document),
    },
    {
      key: 'edit',
      label: '编辑',
      icon: <EditOutlined />,
      onClick: () => onEdit?.(document),
    },
    {
      key: 'download',
      label: '下载',
      icon: <DownloadOutlined />,
      onClick: () => onDownload?.(document),
    },
    {
      type: 'divider',
    },
    {
      key: 'delete',
      label: '删除',
      icon: <DeleteOutlined />,
      danger: true,
      onClick: () => onDelete?.(document),
    },
  ];

  return (
    <Card
      className={`document-card ${selected ? 'selected' : ''}`}
      hoverable
      size="small"
      style={{
        marginBottom: 16,
        border: selected ? '2px solid #1890ff' : undefined,
      }}
      onClick={() => {
        if (showCheckbox) {
          onSelect?.(document.id, !selected);
        }
      }}
      actions={[
        <Tooltip title="预览">
          <Button
            type="text"
            icon={<EyeOutlined />}
            onClick={e => {
              e.stopPropagation();
              onView?.(document);
            }}
          />
        </Tooltip>,
        <Tooltip title="下载">
          <Button
            type="text"
            icon={<DownloadOutlined />}
            onClick={e => {
              e.stopPropagation();
              onDownload?.(document);
            }}
          />
        </Tooltip>,
        <Dropdown menu={{ items: menuItems }} trigger={['click']}>
          <Button
            type="text"
            icon={<MoreOutlined />}
            onClick={e => e.stopPropagation()}
          />
        </Dropdown>,
      ]}
    >
      <Card.Meta
        avatar={getFileTypeIcon(document.filename, document.content_type)}
        title={
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
            }}
          >
            <Tooltip title={document.filename}>
              <Text strong ellipsis style={{ maxWidth: '200px' }}>
                {document.filename}
              </Text>
            </Tooltip>
            {showCheckbox && (
              <input
                type="checkbox"
                checked={selected}
                onChange={e => {
                  e.stopPropagation();
                  onSelect?.(document.id, e.target.checked);
                }}
                style={{ marginLeft: 8 }}
              />
            )}
          </div>
        }
        description={
          <div>
            {/* File Info */}
            <div style={{ marginBottom: 8 }}>
              <Space size="small">
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  {formatFileSize(
                    document.file_size || document.content_length
                  )}
                </Text>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  •
                </Text>
                <Text type="secondary" style={{ fontSize: '12px' }}>
                  <CalendarOutlined style={{ marginRight: 4 }} />
                  {formatDate(document.stored_at)}
                </Text>
              </Space>
            </div>

            {/* Content Preview */}
            {document.content && (
              <Paragraph
                style={{ marginBottom: 8, fontSize: '12px' }}
                type="secondary"
                ellipsis={{ rows: 2 }}
              >
                {getContentPreview(document.content)}
              </Paragraph>
            )}

            {/* Tags */}
            {document.tags && document.tags.length > 0 && (
              <div style={{ marginBottom: 8 }}>
                <TagsOutlined
                  style={{ marginRight: 4, fontSize: '12px', color: '#8c8c8c' }}
                />
                <Space size={[0, 4]} wrap>
                  {document.tags.slice(0, 3).map(tag => (
                    <Tag key={tag} color="blue">
                      {tag}
                    </Tag>
                  ))}
                  {document.tags.length > 3 && (
                    <Tag color="default">+{document.tags.length - 3}</Tag>
                  )}
                </Space>
              </div>
            )}

            {/* Project Info */}
            {document.project_name && (
              <div style={{ marginBottom: 4 }}>
                <Text type="secondary" style={{ fontSize: '11px' }}>
                  项目: {document.project_name}
                </Text>
              </div>
            )}

            {/* Metadata Info */}
            {document.extracted_metadata &&
              Object.keys(document.extracted_metadata).length > 0 && (
                <div>
                  <Text type="secondary" style={{ fontSize: '11px' }}>
                    包含元数据
                  </Text>
                </div>
              )}
          </div>
        }
      />
    </Card>
  );
};

export default DocumentCard;
