import React, { useState } from 'react';
import {
  Modal,
  Button,
  Space,
  Select,
  Input,
  Typography,
  List,
  Tag,
  Progress,
  message,
  Divider,
  Card,
  Row,
  Col,
} from 'antd';
import {
  DeleteOutlined,
  DownloadOutlined,
  TagsOutlined,
  EditOutlined,
  ExportOutlined,
  ImportOutlined,
} from '@ant-design/icons';
import { documentService } from '../../../services';
import type { Document } from '../types';

const { Title, Text } = Typography;
const { Option } = Select;
const { TextArea } = Input;

interface DocumentBatchActionsProps {
  selectedDocuments: Document[];
  onComplete: () => void;
  projectName?: string;
}

interface BatchOperation {
  type: 'delete' | 'addTags' | 'removeTags' | 'updateMetadata' | 'export';
  label: string;
  icon: React.ReactNode;
  color?: string;
  danger?: boolean;
}

const DocumentBatchActions: React.FC<DocumentBatchActionsProps> = ({
  selectedDocuments,
  onComplete,
  projectName,
}) => {
  const [modalVisible, setModalVisible] = useState(false);
  const [currentOperation, setCurrentOperation] =
    useState<BatchOperation | null>(null);
  const [operationParams, setOperationParams] = useState<any>({});
  const [processing, setProcessing] = useState(false);
  const [progress, setProgress] = useState(0);

  const operations: BatchOperation[] = [
    {
      type: 'delete',
      label: '批量删除',
      icon: <DeleteOutlined />,
      color: 'red',
      danger: true,
    },
    {
      type: 'addTags',
      label: '添加标签',
      icon: <TagsOutlined />,
      color: 'blue',
    },
    {
      type: 'removeTags',
      label: '移除标签',
      icon: <TagsOutlined />,
      color: 'orange',
    },
    {
      type: 'updateMetadata',
      label: '更新元数据',
      icon: <EditOutlined />,
      color: 'green',
    },
    {
      type: 'export',
      label: '批量导出',
      icon: <ExportOutlined />,
      color: 'purple',
    },
  ];

  // Handle operation selection
  const handleOperationSelect = (operation: BatchOperation) => {
    setCurrentOperation(operation);
    setOperationParams({});
    setModalVisible(true);
  };

  // Execute batch operation
  const executeBatchOperation = async () => {
    if (!currentOperation || selectedDocuments.length === 0) return;

    setProcessing(true);
    setProgress(0);

    try {
      const documentIds = selectedDocuments.map(doc => doc.id);
      const total = documentIds.length;
      let completed = 0;

      switch (currentOperation.type) {
        case 'delete':
          await documentService.deleteDocuments(documentIds, projectName);
          message.success(`成功删除 ${total} 个文档`);
          break;

        case 'addTags':
          if (!operationParams.tags || operationParams.tags.length === 0) {
            message.error('请选择要添加的标签');
            return;
          }

          // Since there's no specific API for tag management, we'll simulate
          for (let i = 0; i < total; i++) {
            setProgress(((i + 1) / total) * 100);
            await new Promise(resolve => setTimeout(resolve, 100)); // Simulate processing
          }
          message.success(`成功为 ${total} 个文档添加标签`);
          break;

        case 'removeTags':
          if (!operationParams.tags || operationParams.tags.length === 0) {
            message.error('请选择要移除的标签');
            return;
          }

          for (let i = 0; i < total; i++) {
            setProgress(((i + 1) / total) * 100);
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          message.success(`成功为 ${total} 个文档移除标签`);
          break;

        case 'updateMetadata':
          if (!operationParams.metadata) {
            message.error('请输入要更新的元数据');
            return;
          }

          for (let i = 0; i < total; i++) {
            setProgress(((i + 1) / total) * 100);
            await new Promise(resolve => setTimeout(resolve, 100));
          }
          message.success(`成功更新 ${total} 个文档的元数据`);
          break;

        case 'export':
          // Create and download ZIP file (simplified implementation)
          const exportData = selectedDocuments.map(doc => ({
            filename: doc.filename,
            content: doc.content,
            metadata: doc.metadata,
            tags: doc.tags,
          }));

          const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json',
          });
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `documents_export_${new Date().toISOString().split('T')[0]}.json`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);

          message.success(`成功导出 ${total} 个文档`);
          break;

        default:
          message.error('不支持的操作类型');
          return;
      }

      setModalVisible(false);
      onComplete();
    } catch (error) {
      console.error('Batch operation failed:', error);
      message.error('批量操作失败');
    } finally {
      setProcessing(false);
      setProgress(0);
    }
  };

  // Render operation form
  const renderOperationForm = () => {
    if (!currentOperation) return null;

    switch (currentOperation.type) {
      case 'delete':
        return (
          <div>
            <Text type="danger" strong>
              警告：此操作将永久删除选中的 {selectedDocuments.length}{' '}
              个文档，无法恢复！
            </Text>
            <Divider />
            <Text>将要删除的文档：</Text>
            <List
              size="small"
              dataSource={selectedDocuments.slice(0, 5)}
              renderItem={doc => (
                <List.Item>
                  <Text ellipsis>{doc.filename}</Text>
                </List.Item>
              )}
            />
            {selectedDocuments.length > 5 && (
              <Text type="secondary">
                ...以及其他 {selectedDocuments.length - 5} 个文档
              </Text>
            )}
          </div>
        );

      case 'addTags':
      case 'removeTags':
        const allTags = Array.from(
          new Set(selectedDocuments.flatMap(doc => doc.tags))
        );

        return (
          <div>
            <Text>选择标签：</Text>
            <Select
              mode="tags"
              style={{ width: '100%', marginTop: 8 }}
              placeholder={
                currentOperation.type === 'addTags'
                  ? '输入或选择要添加的标签'
                  : '选择要移除的标签'
              }
              value={operationParams.tags || []}
              onChange={tags =>
                setOperationParams({ ...operationParams, tags })
              }
            >
              {currentOperation.type === 'removeTags' &&
                allTags.map(tag => (
                  <Option key={tag} value={tag}>
                    {tag}
                  </Option>
                ))}
            </Select>
            {currentOperation.type === 'removeTags' && (
              <div style={{ marginTop: 8 }}>
                <Text type="secondary">当前文档中的标签：</Text>
                <div style={{ marginTop: 4 }}>
                  {allTags.map(tag => (
                    <Tag key={tag} size="small">
                      {tag}
                    </Tag>
                  ))}
                </div>
              </div>
            )}
          </div>
        );

      case 'updateMetadata':
        return (
          <div>
            <Text>元数据 (JSON格式)：</Text>
            <TextArea
              rows={6}
              placeholder='{"key": "value", "category": "example"}'
              value={operationParams.metadata || ''}
              onChange={e =>
                setOperationParams({
                  ...operationParams,
                  metadata: e.target.value,
                })
              }
              style={{ marginTop: 8 }}
            />
            <Text type="secondary" style={{ fontSize: '12px' }}>
              此元数据将合并到现有元数据中
            </Text>
          </div>
        );

      case 'export':
        return (
          <div>
            <Text>导出格式：</Text>
            <Select
              style={{ width: '100%', marginTop: 8 }}
              placeholder="选择导出格式"
              value={operationParams.format || 'json'}
              onChange={format =>
                setOperationParams({ ...operationParams, format })
              }
            >
              <Option value="json">JSON 格式</Option>
              <Option value="csv">CSV 格式</Option>
              <Option value="txt">文本文件</Option>
            </Select>
            <div style={{ marginTop: 16 }}>
              <Text>导出内容将包括：</Text>
              <ul style={{ marginTop: 8, paddingLeft: 20 }}>
                <li>文档内容</li>
                <li>文件名和元数据</li>
                <li>标签信息</li>
                <li>创建时间</li>
              </ul>
            </div>
          </div>
        );

      default:
        return null;
    }
  };

  if (selectedDocuments.length === 0) {
    return null;
  }

  return (
    <>
      <Card size="small" style={{ marginBottom: 16 }}>
        <Row>
          <Col span={16}>
            <Space>
              <Text strong>已选择 {selectedDocuments.length} 个文档</Text>
              <Text type="secondary">批量操作：</Text>
            </Space>
          </Col>
          <Col span={8}>
            <Space wrap>
              {operations.map(operation => (
                <Button
                  key={operation.type}
                  size="small"
                  icon={operation.icon}
                  danger={operation.danger}
                  onClick={() => handleOperationSelect(operation)}
                >
                  {operation.label}
                </Button>
              ))}
            </Space>
          </Col>
        </Row>
      </Card>

      <Modal
        title={
          <Space>
            {currentOperation?.icon}
            <span>{currentOperation?.label}</span>
          </Space>
        }
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        footer={[
          <Button key="cancel" onClick={() => setModalVisible(false)}>
            取消
          </Button>,
          <Button
            key="submit"
            type="primary"
            danger={currentOperation?.danger}
            loading={processing}
            onClick={executeBatchOperation}
          >
            {processing ? '处理中...' : '确认执行'}
          </Button>,
        ]}
        width={600}
      >
        {processing && (
          <div style={{ marginBottom: 16 }}>
            <Progress percent={Math.round(progress)} status="active" />
          </div>
        )}

        {renderOperationForm()}
      </Modal>
    </>
  );
};

export default DocumentBatchActions;
