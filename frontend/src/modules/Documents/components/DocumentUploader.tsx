import React, { useState, useCallback } from 'react';
import {
  Upload,
  Button,
  Progress,
  Card,
  Typography,
  Tag,
  Input,
  Form,
  Modal,
  Space,
  Divider,
  List,
  Alert,
} from 'antd';
import {
  InboxOutlined,
  FileTextOutlined,
  DeleteOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
} from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { useDocumentUpload } from '../hooks/useDocumentUpload';
import type { DocumentUploadProgress, UploadOptions } from '../types';

const { Dragger } = Upload;
const { Title, Text } = Typography;
const { TextArea } = Input;

interface DocumentUploaderProps {
  projectName?: string;
  onUploadComplete?: (uploadedFiles: any[]) => void;
  maxFileSize?: number; // MB
  acceptedFileTypes?: string[];
}

const DocumentUploader: React.FC<DocumentUploaderProps> = ({
  projectName,
  onUploadComplete,
  maxFileSize = 10,
  acceptedFileTypes = ['.pdf', '.txt', '.doc', '.docx', '.md'],
}) => {
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [uploadOptions, setUploadOptions] = useState<UploadOptions>({
    metadata: {},
    tags: [],
    project_name: projectName,
  });
  const [form] = Form.useForm();

  const {
    uploadProgress,
    uploading,
    uploadFiles,
    removeFile,
    clearCompleted,
    clearAll,
  } = useDocumentUpload();

  const handleUpload = useCallback(
    async (files: File[]) => {
      try {
        const results = await uploadFiles(files, uploadOptions);
        onUploadComplete?.(results);
      } catch (error) {
        console.error('Upload failed:', error);
      }
    },
    [uploadFiles, uploadOptions, onUploadComplete]
  );

  const uploadProps: UploadProps = {
    name: 'files',
    multiple: true,
    accept: acceptedFileTypes.join(','),
    beforeUpload: file => {
      // Validate file size
      if (file.size / 1024 / 1024 > maxFileSize) {
        Modal.error({
          title: '文件过大',
          content: `文件 "${file.name}" 超过了 ${maxFileSize}MB 的大小限制`,
        });
        return false;
      }
      return false; // Prevent auto upload
    },
    onChange: ({ fileList }) => {
      const files = fileList
        .filter(file => file.originFileObj)
        .map(file => file.originFileObj as File);

      if (files.length > 0) {
        handleUpload(files);
      }
    },
    showUploadList: false,
  };

  const handleAdvancedUpload = () => {
    setIsModalVisible(true);
  };

  const handleModalOk = () => {
    form.validateFields().then(values => {
      const options: UploadOptions = {
        metadata: values.metadata ? JSON.parse(values.metadata || '{}') : {},
        tags: values.tags
          ? values.tags
              .split(',')
              .map((tag: string) => tag.trim())
              .filter(Boolean)
          : [],
        project_name: values.project_name || projectName,
      };
      setUploadOptions(options);
      setIsModalVisible(false);
    });
  };

  const getStatusIcon = (status: DocumentUploadProgress['status']) => {
    switch (status) {
      case 'success':
        return <CheckCircleOutlined style={{ color: '#52c41a' }} />;
      case 'error':
        return <CloseCircleOutlined style={{ color: '#ff4d4f' }} />;
      default:
        return <FileTextOutlined />;
    }
  };

  const getStatusColor = (status: DocumentUploadProgress['status']) => {
    switch (status) {
      case 'success':
        return 'success';
      case 'error':
        return 'error';
      case 'uploading':
        return 'processing';
      default:
        return 'default';
    }
  };

  const hasActiveUploads = uploadProgress.length > 0;
  const completedCount = uploadProgress.filter(
    p => p.status === 'success'
  ).length;
  const errorCount = uploadProgress.filter(p => p.status === 'error').length;
  const totalProgress =
    uploadProgress.length > 0
      ? Math.round(
          uploadProgress.reduce((sum, p) => sum + p.progress, 0) /
            uploadProgress.length
        )
      : 0;

  return (
    <div className="document-uploader">
      <Card
        title={
          <div
            style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <Title level={4} style={{ margin: 0 }}>
              文档上传
            </Title>
            <Space>
              <Button type="link" onClick={handleAdvancedUpload}>
                高级设置
              </Button>
              {hasActiveUploads && (
                <>
                  <Button
                    size="small"
                    onClick={clearCompleted}
                    disabled={uploading}
                  >
                    清除已完成
                  </Button>
                  <Button
                    size="small"
                    onClick={clearAll}
                    disabled={uploading}
                    danger
                  >
                    清除全部
                  </Button>
                </>
              )}
            </Space>
          </div>
        }
      >
        <Dragger {...uploadProps} disabled={uploading}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">
            支持单个或批量上传。文件大小限制: {maxFileSize}MB
            <br />
            支持格式: {acceptedFileTypes.join(', ')}
          </p>
        </Dragger>

        {/* Upload Progress */}
        {hasActiveUploads && (
          <>
            <Divider />
            <div style={{ marginBottom: 16 }}>
              <div
                style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  marginBottom: 8,
                }}
              >
                <Text strong>上传进度</Text>
                <Text type="secondary">
                  {completedCount}/{uploadProgress.length} 完成
                  {errorCount > 0 && `, ${errorCount} 失败`}
                </Text>
              </div>
              <Progress
                percent={totalProgress}
                status={
                  uploading
                    ? 'active'
                    : errorCount > 0
                      ? 'exception'
                      : 'success'
                }
                strokeColor={{
                  '0%': '#108ee9',
                  '100%': '#87d068',
                }}
              />
            </div>

            <List
              size="small"
              dataSource={uploadProgress}
              renderItem={item => (
                <List.Item
                  actions={[
                    item.status === 'error' && (
                      <Button
                        type="text"
                        size="small"
                        icon={<DeleteOutlined />}
                        onClick={() => removeFile(item.file.name)}
                        danger
                      />
                    ),
                  ].filter(Boolean)}
                >
                  <List.Item.Meta
                    avatar={getStatusIcon(item.status)}
                    title={
                      <div
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: 8,
                        }}
                      >
                        <Text strong>{item.file.name}</Text>
                        <Tag color={getStatusColor(item.status)}>
                          {item.status === 'pending' && '等待中'}
                          {item.status === 'uploading' && '上传中'}
                          {item.status === 'success' && '成功'}
                          {item.status === 'error' && '失败'}
                        </Tag>
                      </div>
                    }
                    description={
                      <div>
                        <div style={{ marginBottom: 4 }}>
                          <Text type="secondary">
                            {(item.file.size / 1024 / 1024).toFixed(2)} MB
                          </Text>
                        </div>
                        {item.status === 'uploading' && (
                          <Progress
                            percent={item.progress}
                            size="small"
                            status="active"
                          />
                        )}
                        {item.status === 'error' && item.error && (
                          <Alert message={item.error} type="error" showIcon />
                        )}
                      </div>
                    }
                  />
                </List.Item>
              )}
            />
          </>
        )}
      </Card>

      {/* Advanced Upload Modal */}
      <Modal
        title="高级上传设置"
        open={isModalVisible}
        onOk={handleModalOk}
        onCancel={() => setIsModalVisible(false)}
        width={600}
      >
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            project_name: projectName,
            tags: '',
            metadata: '{}',
          }}
        >
          <Form.Item
            name="project_name"
            label="项目名称"
            help="如果不指定，将使用当前项目"
          >
            <Input placeholder="输入项目名称" />
          </Form.Item>

          <Form.Item
            name="tags"
            label="标签"
            help="多个标签用逗号分隔，例如: 技术文档, API, 重要"
          >
            <Input placeholder="输入标签" />
          </Form.Item>

          <Form.Item
            name="metadata"
            label="元数据 (JSON)"
            help="额外的元数据信息，必须是有效的 JSON 格式"
            rules={[
              {
                validator: (_, value) => {
                  if (!value || value.trim() === '') return Promise.resolve();
                  try {
                    JSON.parse(value);
                    return Promise.resolve();
                  } catch {
                    return Promise.reject(new Error('请输入有效的 JSON 格式'));
                  }
                },
              },
            ]}
          >
            <TextArea
              rows={4}
              placeholder='{"category": "documentation", "priority": "high"}'
            />
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default DocumentUploader;
