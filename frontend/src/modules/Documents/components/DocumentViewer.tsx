import React, { useState, useEffect } from 'react';
import {
  Modal,
  Tabs,
  Typography,
  Space,
  Tag,
  Button,
  Divider,
  Descriptions,
  Card,
  Row,
  Col,
  Input,
  Spin,
  App,
} from 'antd';
import {
  FileTextOutlined,
  DownloadOutlined,
  EditOutlined,
  CopyOutlined,
  InfoCircleOutlined,
  TagsOutlined,
  CalendarOutlined,
  FileOutlined,
} from '@ant-design/icons';
import type { Document } from '../types';
import { documentService } from '@/services/documentService';
import { useAppStore } from '@/store';

const { Title, Text, Paragraph } = Typography;
const { TextArea } = Input;
const { TabPane } = Tabs;

interface DocumentViewerProps {
  document: Document | null;
  open: boolean;
  onClose: () => void;
  onEdit?: (document: Document) => void;
  onDownload?: (document: Document) => void;
}

const DocumentViewer: React.FC<DocumentViewerProps> = ({
  document,
  open,
  onClose,
  onEdit,
  onDownload,
}) => {
  const [activeTab, setActiveTab] = useState('content');
  const [fullDocument, setFullDocument] = useState<Document | null>(null);
  const [loading, setLoading] = useState(false);
  const { currentProject } = useAppStore();
  const { message } = App.useApp();

  // Load full document content when viewer opens
  useEffect(() => {
    if (open && document) {
      if (document.content) {
        // Document already has content, use it directly
        console.log('DocumentViewer: Using existing content');
        setFullDocument(document);
      } else {
        // Need to load content from API
        console.log('DocumentViewer: Loading content from API');
        loadDocumentContent();
      }
    }
  }, [open, document]);

  const loadDocumentContent = async () => {
    if (!document) return;

    setLoading(true);
    try {
      const response = await documentService.getDocument(
        document.id,
        currentProject || undefined
      );
      console.log('Document API response:', response); // Debug log
      if (response.success && response.data) {
        setFullDocument(response.data);
      } else {
        message.error('获取文档内容失败');
      }
    } catch (error) {
      console.error('Failed to load document content:', error);
      message.error('获取文档内容失败');
    } finally {
      setLoading(false);
    }
  };

  const displayDocument = fullDocument || document;

  // Clear state when viewer closes
  const handleClose = () => {
    setFullDocument(null);
    setActiveTab('content');
    onClose();
  };

  if (!document) return null;

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString('zh-CN');
  };

  const handleCopyContent = () => {
    if (displayDocument?.content) {
      navigator.clipboard.writeText(displayDocument.content);
      message.success('内容已复制到剪贴板');
    }
  };

  const handleCopyMetadata = () => {
    const metadata = JSON.stringify(document.metadata, null, 2);
    navigator.clipboard.writeText(metadata);
    message.success('元数据已复制到剪贴板');
  };

  const handleDownload = () => {
    if (onDownload) {
      onDownload(displayDocument || document);
    } else {
      // Default download implementation
      const content = displayDocument?.content || '';
      const blob = new Blob([content], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = window.document.createElement('a');
      a.href = url;
      a.download =
        document.filename || document.title || document.source || '未命名文档';
      window.document.body.appendChild(a);
      a.click();
      window.document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const renderContent = () => {
    if (loading) {
      return (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <Spin size="large" />
          <div style={{ marginTop: 16 }}>
            <Text type="secondary">正在加载文档内容...</Text>
          </div>
        </div>
      );
    }

    const content = displayDocument?.content;
    if (!content) {
      return (
        <div style={{ textAlign: 'center', padding: '40px' }}>
          <Text type="secondary">无法获取文档内容</Text>
        </div>
      );
    }

    // Simple content display with syntax highlighting for different file types
    const filename =
      document.filename || document.title || document.source || '';
    const extension = filename.split('.').pop()?.toLowerCase();

    return (
      <div style={{ position: 'relative' }}>
        <div
          style={{
            marginBottom: 16,
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <Space>
            <Text strong>文档内容</Text>
            <Tag color="blue">
              {formatFileSize(
                displayDocument?.content_length || content.length
              )}
            </Tag>
          </Space>
          <Button
            size="small"
            icon={<CopyOutlined />}
            onClick={handleCopyContent}
            disabled={!content}
          >
            复制内容
          </Button>
        </div>

        <Card
          size="small"
          style={{
            maxHeight: '400px',
            overflow: 'auto',
            backgroundColor: '#fafafa',
          }}
        >
          <pre
            style={{
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
              margin: 0,
              fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
              fontSize: '12px',
              lineHeight: '1.5',
            }}
          >
            {content}
          </pre>
        </Card>
      </div>
    );
  };

  const renderMetadata = () => {
    const basicInfo = [
      {
        label: '文件名',
        value: document.filename || document.title || document.source,
      },
      {
        label: '文件大小',
        value: formatFileSize(
          document.file_size || document.content?.length || 0
        ),
      },
      { label: '内容类型', value: document.content_type || '未知' },
      {
        label: '存储时间',
        value: formatDate(
          document.stored_at || document.created_at || document.updated_at || ''
        ),
      },
      { label: '项目名称', value: document.project_name || '未指定' },
      { label: '内容哈希', value: document.content_hash || '未知' },
    ];

    return (
      <div>
        {/* Basic Information */}
        <div style={{ marginBottom: 24 }}>
          <Title level={5}>
            <InfoCircleOutlined style={{ marginRight: 8 }} />
            基本信息
          </Title>
          <Descriptions column={2} size="small" bordered>
            {basicInfo.map(item => (
              <Descriptions.Item key={item.label} label={item.label}>
                {item.value}
              </Descriptions.Item>
            ))}
          </Descriptions>
        </div>

        {/* Tags */}
        {document.tags && document.tags.length > 0 && (
          <div style={{ marginBottom: 24 }}>
            <Title level={5}>
              <TagsOutlined style={{ marginRight: 8 }} />
              标签
            </Title>
            <Space wrap>
              {document.tags.map(tag => (
                <Tag key={tag} color="blue">
                  {tag}
                </Tag>
              ))}
            </Space>
          </div>
        )}

        {/* Custom Metadata */}
        {document.metadata && Object.keys(document.metadata).length > 0 && (
          <div style={{ marginBottom: 24 }}>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginBottom: 12,
              }}
            >
              <Title level={5} style={{ margin: 0 }}>
                <FileOutlined style={{ marginRight: 8 }} />
                自定义元数据
              </Title>
              <Button
                size="small"
                icon={<CopyOutlined />}
                onClick={handleCopyMetadata}
              >
                复制元数据
              </Button>
            </div>
            <Card size="small">
              <pre
                style={{
                  margin: 0,
                  fontSize: '12px',
                  whiteSpace: 'pre-wrap',
                  wordBreak: 'break-word',
                }}
              >
                {JSON.stringify(document.metadata, null, 2)}
              </pre>
            </Card>
          </div>
        )}

        {/* Extracted Metadata */}
        {document.extracted_metadata &&
          Object.keys(document.extracted_metadata).length > 0 && (
            <div>
              <Title level={5}>
                <FileOutlined style={{ marginRight: 8 }} />
                提取的元数据
              </Title>
              <Card size="small">
                <pre
                  style={{
                    margin: 0,
                    fontSize: '12px',
                    whiteSpace: 'pre-wrap',
                    wordBreak: 'break-word',
                  }}
                >
                  {JSON.stringify(document.extracted_metadata, null, 2)}
                </pre>
              </Card>
            </div>
          )}
      </div>
    );
  };

  return (
    <Modal
      title={
        <div
          style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
          }}
        >
          <Space>
            <FileTextOutlined />
            <Text strong ellipsis style={{ maxWidth: '300px' }}>
              {document.filename}
            </Text>
          </Space>
          <Space>
            {onEdit && (
              <Button
                size="small"
                icon={<EditOutlined />}
                onClick={() => onEdit(document)}
              >
                编辑
              </Button>
            )}
            <Button
              size="small"
              icon={<DownloadOutlined />}
              onClick={handleDownload}
              type="primary"
            >
              下载
            </Button>
          </Space>
        </div>
      }
      open={open}
      onCancel={handleClose}
      footer={null}
      width={800}
      style={{ top: 20 }}
      styles={{ body: { padding: 0 } }}
    >
      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        style={{ margin: 0 }}
        tabBarStyle={{ paddingLeft: 24, paddingRight: 24, marginBottom: 0 }}
      >
        <TabPane tab="内容" key="content">
          <div style={{ padding: 24 }}>{renderContent()}</div>
        </TabPane>

        <TabPane tab="元数据" key="metadata">
          <div style={{ padding: 24 }}>{renderMetadata()}</div>
        </TabPane>
      </Tabs>
    </Modal>
  );
};

export default DocumentViewer;
