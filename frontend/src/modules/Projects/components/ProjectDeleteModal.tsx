import React, { useState } from 'react';
import {
  Modal,
  Form,
  Input,
  Checkbox,
  Typography,
  Alert,
  Space,
  Divider,
} from 'antd';
import { ExclamationCircleOutlined, DeleteOutlined } from '@ant-design/icons';
import type { ProjectDeleteConfirmation } from '../types/project';

const { Title, Text } = Typography;

interface ProjectDeleteModalProps {
  visible: boolean;
  projectName: string;
  onConfirm: (confirmation: ProjectDeleteConfirmation) => void;
  onCancel: () => void;
  confirmLoading?: boolean;
  projectStats?: {
    documentCount: number;
    entityCount: number;
    relationCount: number;
    sizeInMB: number;
  };
}

export const ProjectDeleteModal: React.FC<ProjectDeleteModalProps> = ({
  visible,
  projectName,
  onConfirm,
  onCancel,
  confirmLoading = false,
  projectStats = {
    documentCount: 0,
    entityCount: 0,
    relationCount: 0,
    sizeInMB: 0,
  },
}) => {
  const [form] = Form.useForm();
  const [confirmationText, setConfirmationText] = useState('');
  const [understoodConsequences, setUnderstoodConsequences] = useState(false);

  const expectedText = `delete ${projectName}`;
  const isConfirmationValid =
    confirmationText === expectedText && understoodConsequences;

  const handleOk = async () => {
    if (!isConfirmationValid) {
      return;
    }

    const confirmation: ProjectDeleteConfirmation = {
      projectName,
      confirm: true,
      understoodConsequences,
    };

    onConfirm(confirmation);
  };

  const handleCancel = () => {
    setConfirmationText('');
    setUnderstoodConsequences(false);
    form.resetFields();
    onCancel();
  };

  const formatSize = (sizeInMB: number): string => {
    if (sizeInMB < 1) return `${(sizeInMB * 1024).toFixed(0)} KB`;
    if (sizeInMB < 1024) return `${sizeInMB.toFixed(1)} MB`;
    return `${(sizeInMB / 1024).toFixed(1)} GB`;
  };

  return (
    <Modal
      title={
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 8,
            color: '#ff4d4f',
          }}
        >
          <ExclamationCircleOutlined />
          <span>Delete Project</span>
        </div>
      }
      open={visible}
      onOk={handleOk}
      onCancel={handleCancel}
      confirmLoading={confirmLoading}
      width={600}
      destroyOnClose
      okText="Delete Project"
      cancelText="Cancel"
      okButtonProps={{
        danger: true,
        disabled: !isConfirmationValid,
        icon: <DeleteOutlined />,
      }}
    >
      <div style={{ padding: '8px 0' }}>
        {/* Warning Alert */}
        <Alert
          message="Permanent Deletion Warning"
          description="This action cannot be undone. All project data will be permanently deleted from your system."
          type="error"
          showIcon
          style={{ marginBottom: 20 }}
        />

        {/* Project Info */}
        <div style={{ marginBottom: 20 }}>
          <Title level={4} style={{ margin: '0 0 12px 0' }}>
            Project: {projectName}
          </Title>

          <div
            style={{
              background: '#fff2f0',
              border: '1px solid #ffccc7',
              borderRadius: '6px',
              padding: '12px',
            }}
          >
            <Title level={5} style={{ margin: '0 0 8px 0', color: '#cf1322' }}>
              Data that will be permanently deleted:
            </Title>
            <Space direction="vertical" size={4} style={{ width: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>Documents:</Text>
                <Text strong>
                  {projectStats.documentCount.toLocaleString()}
                </Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>Entities:</Text>
                <Text strong>{projectStats.entityCount.toLocaleString()}</Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>Relations:</Text>
                <Text strong>
                  {projectStats.relationCount.toLocaleString()}
                </Text>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <Text>Total Size:</Text>
                <Text strong>{formatSize(projectStats.sizeInMB)}</Text>
              </div>
            </Space>
          </div>
        </div>

        {/* Consequences Checklist */}
        <div style={{ marginBottom: 20 }}>
          <Title level={5} style={{ margin: '0 0 12px 0' }}>
            I understand that:
          </Title>
          <Checkbox
            checked={understoodConsequences}
            onChange={e => setUnderstoodConsequences(e.target.checked)}
          >
            <Space direction="vertical" size={2} style={{ marginLeft: 8 }}>
              <Text>
                All project files and data will be permanently deleted
              </Text>
              <Text type="secondary" style={{ fontSize: '12px' }}>
                This includes documents, knowledge graphs, embeddings, and cache
                files
              </Text>
            </Space>
          </Checkbox>
        </div>

        <Divider />

        {/* Confirmation Input */}
        <div>
          <Title level={5} style={{ margin: '0 0 8px 0' }}>
            Confirmation Required
          </Title>
          <Text style={{ marginBottom: 8, display: 'block' }}>
            To confirm deletion, please type <Text code>{expectedText}</Text> in
            the field below:
          </Text>

          <Form form={form} layout="vertical">
            <Form.Item
              name="confirmationText"
              validateStatus={
                confirmationText && confirmationText !== expectedText
                  ? 'error'
                  : undefined
              }
              help={
                confirmationText && confirmationText !== expectedText
                  ? `Please type exactly: ${expectedText}`
                  : undefined
              }
            >
              <Input
                placeholder={`Type "${expectedText}" to confirm`}
                value={confirmationText}
                onChange={e => setConfirmationText(e.target.value)}
                style={{
                  borderColor:
                    confirmationText === expectedText ? '#52c41a' : undefined,
                }}
              />
            </Form.Item>
          </Form>
        </div>

        {/* Final Warning */}
        <Alert
          message="Last Warning"
          description={
            <div>
              <Text strong style={{ color: '#cf1322' }}>
                This action is irreversible.
              </Text>
              <br />
              <Text style={{ fontSize: '12px' }}>
                Make sure you have exported any important data before
                proceeding.
              </Text>
            </div>
          }
          type="warning"
          showIcon
          style={{ marginTop: 16 }}
        />
      </div>
    </Modal>
  );
};

export default ProjectDeleteModal;
