import React, { useState, useEffect } from 'react';
import {
  Modal,
  Form,
  Input,
  InputNumber,
  Button,
  Tabs,
  Space,
  Typography,
  Card,
  Tag,
  Select,
  message,
  Spin,
  Alert,
} from 'antd';
import {
  SaveOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  SettingOutlined,
} from '@ant-design/icons';
import { configService } from '@/services/configService';
import type { ProjectConfig, ConfigUpdateRequest } from '@/types/config';

const { Title, Text } = Typography;
const { TextArea } = Input;
const { Option } = Select;

interface ProjectConfigModalProps {
  visible: boolean;
  projectName: string;
  onCancel: () => void;
  onSuccess?: () => void;
}

const ProjectConfigModal: React.FC<ProjectConfigModalProps> = ({
  visible,
  projectName,
  onCancel,
  onSuccess,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [saving, setSaving] = useState(false);
  const [config, setConfig] = useState<ProjectConfig | null>(null);
  const [activeTab, setActiveTab] = useState('basic');

  const loadConfig = async () => {
    if (!projectName) return;

    setLoading(true);
    try {
      const response = await configService.getProjectConfig(projectName);

      if (response.success && response.data) {
        const configData = response.data as unknown as ProjectConfig;
        setConfig(configData);

        // Set form values
        if (configData && configData.settings) {
          form.setFieldsValue({
            // Basic settings
            description: configData.project_info?.description || '',
            workdir: configData.settings.workdir || '',

            // OpenAI settings
            openai_api_key: configData.settings.openai?.api_key || '',
            openai_api_base: configData.settings.openai?.api_base || '',

            // LLM settings
            llm_model: configData.settings.llm?.model || '',
            llm_temperature: configData.settings.llm?.temperature || 0,
            llm_max_tokens: configData.settings.llm?.max_tokens || 4096,
            llm_provider: configData.settings.llm?.provider || 'openai',

            // Embedding settings
            embedding_model: configData.settings.embedding?.model || '',
            embedding_provider:
              configData.settings.embedding?.provider || 'openai',
            embedding_dimension:
              configData.settings.embedding?.dimension || 1024,
            embedding_max_token_size:
              configData.settings.embedding?.max_token_size || 8192,
            embedding_batch_size:
              configData.settings.embedding?.batch_size || 32,

            // Text processing settings
            max_chunk_size: configData.settings.text?.max_chunk_size || 512,
            chunk_overlap: configData.settings.text?.chunk_overlap || 100,

            // RAG settings
            system_prompt: configData.settings.rag?.system_prompt || '',
          });
        }
      }
    } catch (error) {
      console.error('Failed to load config:', error);
      message.error('Failed to load project configuration');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const values = await form.validateFields();

      const updates: ConfigUpdateRequest = {
        workdir: values.workdir,
        openai_api_key: values.openai_api_key,
        openai_api_base: values.openai_api_base,
        llm_model: values.llm_model,
        llm_temperature: values.llm_temperature,
        llm_max_tokens: values.llm_max_tokens,
        llm_provider: values.llm_provider,
        embedding_model: values.embedding_model,
        embedding_provider: values.embedding_provider,
        embedding_dimension: values.embedding_dimension,
        embedding_max_token_size: values.embedding_max_token_size,
        embedding_batch_size: values.embedding_batch_size,
        max_chunk_size: values.max_chunk_size,
        chunk_overlap: values.chunk_overlap,
        system_prompt: values.system_prompt,
      };

      const response = await configService.updateProjectConfig(
        projectName,
        updates
      );

      if (response.success) {
        message.success('Project configuration saved successfully');
        await loadConfig(); // Reload to show updated values
        onSuccess?.();
      } else {
        message.error('Failed to save configuration');
      }
    } catch (error) {
      console.error('Failed to save config:', error);
      message.error('Failed to save configuration');
    } finally {
      setSaving(false);
    }
  };

  const handleReset = async () => {
    try {
      setSaving(true);
      const response = await configService.resetProjectConfig(projectName);

      if (response.success) {
        message.success('Project configuration reset to defaults');
        await loadConfig(); // Reload to show reset values
      } else {
        message.error('Failed to reset configuration');
      }
    } catch (error) {
      console.error('Failed to reset config:', error);
      message.error('Failed to reset configuration');
    } finally {
      setSaving(false);
    }
  };

  useEffect(() => {
    if (visible && projectName) {
      loadConfig();
    }
  }, [visible, projectName]);

  const basicInfoTab = (
    <div>
      <Title level={4}>
        <InfoCircleOutlined style={{ marginRight: 8 }} />
        Project Information
      </Title>

      {config?.project_info && (
        <Card size="small" style={{ marginBottom: 16 }}>
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>Project Name: </Text>
              <Text>{config.project_info.project_name}</Text>
            </div>
            <div>
              <Text strong>Created: </Text>
              <Text>
                {new Date(config.project_info.created_at).toLocaleString()}
              </Text>
            </div>
            <div>
              <Text strong>Version: </Text>
              <Tag>{config.project_info.version}</Tag>
            </div>
          </Space>
        </Card>
      )}

      <Form.Item
        label="Description"
        name="description"
        rules={[
          { max: 500, message: 'Description cannot exceed 500 characters' },
        ]}
      >
        <TextArea rows={3} placeholder="Project description..." />
      </Form.Item>

      <Form.Item
        label="Working Directory"
        name="workdir"
        rules={[{ required: true, message: 'Please enter working directory' }]}
      >
        <Input placeholder="workdir" />
      </Form.Item>

      {config?.runtime_info && (
        <Card title="Runtime Information" size="small">
          <Space direction="vertical" style={{ width: '100%' }}>
            <div>
              <Text strong>Collection: </Text>
              <Text code>{config.runtime_info.collection_name}</Text>
            </div>
            <div>
              <Text strong>Vector Store: </Text>
              <Text>{config.runtime_info.vector_store_type}</Text>
            </div>
            <div>
              <Text strong>Knowledge Graph: </Text>
              <Tag
                color={
                  config.runtime_info.enable_knowledge_graph
                    ? 'green'
                    : 'orange'
                }
              >
                {config.runtime_info.enable_knowledge_graph
                  ? 'Enabled'
                  : 'Disabled'}
              </Tag>
            </div>
            <div>
              <Text strong>Initialized: </Text>
              <Tag color={config.runtime_info.is_initialized ? 'green' : 'red'}>
                {config.runtime_info.is_initialized ? 'Yes' : 'No'}
              </Tag>
            </div>
          </Space>
        </Card>
      )}
    </div>
  );

  const openaiTab = (
    <div>
      <Title level={4}>OpenAI Configuration</Title>

      <Form.Item
        label="API Key"
        name="openai_api_key"
        rules={[{ required: true, message: 'Please enter OpenAI API key' }]}
      >
        <Input.Password placeholder="sk-..." />
      </Form.Item>

      <Form.Item
        label="API Base URL"
        name="openai_api_base"
        rules={[
          { required: true, message: 'Please enter API base URL' },
          { type: 'url', message: 'Please enter a valid URL' },
        ]}
      >
        <Input placeholder="https://api.openai.com/v1" />
      </Form.Item>
    </div>
  );

  const llmTab = (
    <div>
      <Title level={4}>Language Model Configuration</Title>

      <Form.Item
        label="Model"
        name="llm_model"
        rules={[{ required: true, message: 'Please enter LLM model' }]}
      >
        <Input placeholder="gpt-4" />
      </Form.Item>

      <Form.Item
        label="Provider"
        name="llm_provider"
        rules={[{ required: true, message: 'Please select provider' }]}
      >
        <Select placeholder="Select provider">
          <Option value="openai">OpenAI</Option>
          <Option value="huggingface">HuggingFace</Option>
          <Option value="ollama">Ollama</Option>
        </Select>
      </Form.Item>

      <Form.Item
        label="Temperature"
        name="llm_temperature"
        rules={[
          { required: true, message: 'Please enter temperature' },
          {
            type: 'number',
            min: 0,
            max: 2,
            message: 'Temperature must be between 0 and 2',
          },
        ]}
      >
        <InputNumber
          min={0}
          max={2}
          step={0.1}
          style={{ width: '100%' }}
          placeholder="0.7"
        />
      </Form.Item>

      <Form.Item
        label="Max Tokens"
        name="llm_max_tokens"
        rules={[
          { required: true, message: 'Please enter max tokens' },
          { type: 'number', min: 1, message: 'Max tokens must be positive' },
        ]}
      >
        <InputNumber
          min={1}
          max={32768}
          style={{ width: '100%' }}
          placeholder="4096"
        />
      </Form.Item>
    </div>
  );

  const embeddingTab = (
    <div>
      <Title level={4}>Embedding Configuration</Title>

      <Form.Item
        label="Model"
        name="embedding_model"
        rules={[{ required: true, message: 'Please enter embedding model' }]}
      >
        <Input placeholder="text-embedding-ada-002" />
      </Form.Item>

      <Form.Item
        label="Provider"
        name="embedding_provider"
        rules={[{ required: true, message: 'Please select provider' }]}
      >
        <Select placeholder="Select provider">
          <Option value="openai">OpenAI</Option>
          <Option value="huggingface">HuggingFace</Option>
        </Select>
      </Form.Item>

      <Form.Item
        label="Dimension"
        name="embedding_dimension"
        rules={[
          { required: true, message: 'Please enter dimension' },
          { type: 'number', min: 1, message: 'Dimension must be positive' },
        ]}
      >
        <InputNumber
          min={1}
          max={4096}
          style={{ width: '100%' }}
          placeholder="1536"
        />
      </Form.Item>

      <Form.Item
        label="Max Token Size"
        name="embedding_max_token_size"
        rules={[
          { required: true, message: 'Please enter max token size' },
          {
            type: 'number',
            min: 1,
            message: 'Max token size must be positive',
          },
        ]}
      >
        <InputNumber
          min={1}
          max={32768}
          style={{ width: '100%' }}
          placeholder="8192"
        />
      </Form.Item>

      <Form.Item
        label="Batch Size"
        name="embedding_batch_size"
        rules={[
          { required: true, message: 'Please enter batch size' },
          { type: 'number', min: 1, message: 'Batch size must be positive' },
        ]}
      >
        <InputNumber
          min={1}
          max={100}
          style={{ width: '100%' }}
          placeholder="32"
        />
      </Form.Item>
    </div>
  );

  const textTab = (
    <div>
      <Title level={4}>Text Processing Configuration</Title>

      <Form.Item
        label="Max Chunk Size"
        name="max_chunk_size"
        rules={[
          { required: true, message: 'Please enter max chunk size' },
          { type: 'number', min: 1, message: 'Chunk size must be positive' },
        ]}
      >
        <InputNumber
          min={1}
          max={4096}
          style={{ width: '100%' }}
          placeholder="512"
        />
      </Form.Item>

      <Form.Item
        label="Chunk Overlap"
        name="chunk_overlap"
        rules={[
          { required: true, message: 'Please enter chunk overlap' },
          {
            type: 'number',
            min: 0,
            message: 'Chunk overlap must be non-negative',
          },
        ]}
      >
        <InputNumber
          min={0}
          max={500}
          style={{ width: '100%' }}
          placeholder="100"
        />
      </Form.Item>
    </div>
  );

  const ragTab = (
    <div>
      <Title level={4}>RAG System Prompt</Title>

      <Form.Item
        label="System Prompt"
        name="system_prompt"
        rules={[{ required: true, message: 'Please enter system prompt' }]}
      >
        <TextArea
          rows={10}
          placeholder="Enter the system prompt for RAG..."
          showCount
          maxLength={5000}
        />
      </Form.Item>
    </div>
  );

  const items = [
    {
      key: 'basic',
      label: 'Basic Info',
      children: basicInfoTab,
    },
    {
      key: 'openai',
      label: 'OpenAI',
      children: openaiTab,
    },
    {
      key: 'llm',
      label: 'Language Model',
      children: llmTab,
    },
    {
      key: 'embedding',
      label: 'Embedding',
      children: embeddingTab,
    },
    {
      key: 'text',
      label: 'Text Processing',
      children: textTab,
    },
    {
      key: 'rag',
      label: 'RAG Prompt',
      children: ragTab,
    },
  ];

  return (
    <Modal
      title={
        <Space>
          <SettingOutlined />
          <span>Project Configuration - {projectName}</span>
        </Space>
      }
      open={visible}
      onCancel={onCancel}
      width={800}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          Cancel
        </Button>,
        <Button
          key="reset"
          onClick={handleReset}
          loading={saving}
          icon={<ReloadOutlined />}
        >
          Reset to Defaults
        </Button>,
        <Button
          key="save"
          type="primary"
          onClick={handleSave}
          loading={saving}
          icon={<SaveOutlined />}
        >
          Save Configuration
        </Button>,
      ]}
    >
      <Spin spinning={loading}>
        {config ? (
          <Form form={form} layout="vertical" autoComplete="off">
            <Tabs activeKey={activeTab} onChange={setActiveTab} items={items} />
          </Form>
        ) : (
          <Alert
            message="Loading Configuration"
            description="Please wait while we load the project configuration..."
            type="info"
            showIcon
          />
        )}
      </Spin>
    </Modal>
  );
};

export default ProjectConfigModal;
