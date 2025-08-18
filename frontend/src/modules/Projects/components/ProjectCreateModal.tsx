import React, { useState, useEffect } from 'react';
import { Modal, Form, Input, Typography, Space, Alert, Divider } from 'antd';
import { FolderAddOutlined, InfoCircleOutlined } from '@ant-design/icons';
import type {
  ProjectCreateModalProps,
  ProjectFormData,
} from '../types/project';

const { Title, Text } = Typography;
const { TextArea } = Input;

export const ProjectCreateModal: React.FC<ProjectCreateModalProps> = ({
  visible,
  onCancel,
  onSuccess,
  confirmLoading = false,
}) => {
  const [form] = Form.useForm<ProjectFormData>();
  const [validationErrors, setValidationErrors] = useState<string[]>([]);

  // Reset form when modal opens/closes
  useEffect(() => {
    if (visible) {
      form.resetFields();
      setValidationErrors([]);
    }
  }, [visible, form]);

  // Project name validation
  const validateProjectName = (name: string): string[] => {
    const errors: string[] = [];

    if (!name || name.length === 0) {
      errors.push('Project name is required');
    }

    if (name.length > 50) {
      errors.push('Project name must be 50 characters or less');
    }

    if (name.length < 2) {
      errors.push('Project name must be at least 2 characters');
    }

    // Check for invalid characters
    const invalidChars = /[<>:"/\\|?*]/;
    if (invalidChars.test(name)) {
      errors.push(
        'Project name contains invalid characters: < > : " / \\ | ? *'
      );
    }

    // Check for leading/trailing spaces
    if (name !== name.trim()) {
      errors.push('Project name cannot start or end with spaces');
    }

    // Check for reserved names (case insensitive)
    const reservedNames = [
      'con',
      'prn',
      'aux',
      'nul',
      'com1',
      'com2',
      'com3',
      'com4',
      'com5',
      'com6',
      'com7',
      'com8',
      'com9',
      'lpt1',
      'lpt2',
      'lpt3',
      'lpt4',
      'lpt5',
      'lpt6',
      'lpt7',
      'lpt8',
      'lpt9',
    ];
    if (reservedNames.includes(name.toLowerCase())) {
      errors.push('Project name cannot be a system reserved name');
    }

    // Check for consecutive dots or special patterns
    if (name.includes('..') || name.startsWith('.') || name.endsWith('.')) {
      errors.push(
        'Project name cannot contain consecutive dots or start/end with dots'
      );
    }

    return errors;
  };

  const handleProjectNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const name = e.target.value;
    const errors = validateProjectName(name);
    setValidationErrors(errors);
  };

  const handleOk = async () => {
    try {
      const values = await form.validateFields();

      // Final validation
      const nameErrors = validateProjectName(values.name);
      if (nameErrors.length > 0) {
        setValidationErrors(nameErrors);
        return;
      }

      // Create project data
      const projectData: ProjectFormData = {
        name: values.name.trim(),
        description: values.description?.trim() || undefined,
      };

      // Call success callback with project data
      onSuccess({
        name: projectData.name,
        description: projectData.description || '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
        document_count: 0,
        entity_count: 0,
        relation_count: 0,
        is_current: false,
        status: 'inactive',
        statistics: {
          document_count: 0,
          entity_count: 0,
          relation_count: 0,
          has_vector_db: false,
          cache_size: 0,
          size_mb: 0,
        },
      });
    } catch (error) {
      console.error('Form validation failed:', error);
    }
  };

  const handleCancel = () => {
    setValidationErrors([]);
    onCancel();
  };

  return (
    <Modal
      title={
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <FolderAddOutlined />
          <span>Create New Project</span>
        </div>
      }
      open={visible}
      onOk={handleOk}
      onCancel={handleCancel}
      confirmLoading={confirmLoading}
      width={600}
      destroyOnClose
      okText="Create Project"
      cancelText="Cancel"
      okButtonProps={{
        disabled: validationErrors.length > 0,
      }}
    >
      <div style={{ padding: '8px 0' }}>
        <Form
          form={form}
          layout="vertical"
          requiredMark={false}
          autoComplete="off"
        >
          {/* Project Name */}
          <Form.Item
            label={
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span>Project Name</span>
                <Text type="danger">*</Text>
              </div>
            }
            name="name"
            rules={[
              { required: true, message: 'Please enter a project name' },
              { min: 2, message: 'Project name must be at least 2 characters' },
              {
                max: 50,
                message: 'Project name must be 50 characters or less',
              },
            ]}
            validateStatus={validationErrors.length > 0 ? 'error' : undefined}
            help={validationErrors.length > 0 ? validationErrors[0] : undefined}
          >
            <Input
              placeholder="Enter project name"
              onChange={handleProjectNameChange}
              autoFocus
            />
          </Form.Item>

          {/* Validation Errors */}
          {validationErrors.length > 1 && (
            <Alert
              message="Project Name Validation Errors"
              description={
                <ul style={{ margin: '4px 0', paddingLeft: '20px' }}>
                  {validationErrors.map((error, index) => (
                    <li key={index}>{error}</li>
                  ))}
                </ul>
              }
              type="error"
              showIcon
              style={{ marginBottom: 16 }}
            />
          )}

          {/* Description */}
          <Form.Item
            label="Description"
            name="description"
            rules={[
              {
                max: 500,
                message: 'Description must be 500 characters or less',
              },
            ]}
          >
            <TextArea
              placeholder="Enter project description (optional)"
              rows={3}
              maxLength={500}
              showCount
            />
          </Form.Item>
        </Form>

        <Divider />

        {/* Info Section */}
        <div
          style={{
            background: '#f5f5f5',
            padding: '12px',
            borderRadius: '6px',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
            <InfoCircleOutlined
              style={{ color: '#1890ff', marginTop: '2px' }}
            />
            <div>
              <Title
                level={5}
                style={{ margin: '0 0 8px 0', fontSize: '14px' }}
              >
                Project Setup Information
              </Title>
              <Space direction="vertical" size={4}>
                <Text style={{ fontSize: '12px' }}>
                  • A new project workspace will be created with separate
                  storage
                </Text>
                <Text style={{ fontSize: '12px' }}>
                  • Documents, entities, and relationships are isolated per
                  project
                </Text>
                <Text style={{ fontSize: '12px' }}>
                  • You can switch between projects at any time
                </Text>
                <Text style={{ fontSize: '12px' }}>
                  • Project data is stored locally and can be exported/imported
                </Text>
              </Space>
            </div>
          </div>
        </div>

        {/* Project Name Guidelines */}
        <div style={{ marginTop: 16 }}>
          <Title level={5} style={{ fontSize: '13px', margin: '0 0 8px 0' }}>
            Project Name Guidelines:
          </Title>
          <Space direction="vertical" size={2}>
            <Text style={{ fontSize: '11px', color: '#666' }}>
              • Use alphanumeric characters, spaces, hyphens, and underscores
            </Text>
            <Text style={{ fontSize: '11px', color: '#666' }}>
              • Avoid system reserved names (con, prn, aux, nul, etc.)
            </Text>
            <Text style={{ fontSize: '11px', color: '#666' }}>
              • Cannot contain: {'< > : " / \\ | ? *'}
            </Text>
            <Text style={{ fontSize: '11px', color: '#666' }}>
              • Length: 2-50 characters
            </Text>
          </Space>
        </div>
      </div>
    </Modal>
  );
};

export default ProjectCreateModal;
